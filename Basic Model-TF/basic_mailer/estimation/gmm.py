from __future__ import annotations

"""GMM estimator (Flavor A) for the BASIC model.

Flavor A:
  for each candidate theta:
    - nested inner solve (policy NN)
    - simulate candidate states by forward policy simulation
    - compute Euler residual f(k,z,eps; theta) and instruments q(k,z)
    - enforce orthogonality: E[ E_eps f * q ] = 0
    - minimize Q = g' W g

We reuse:
- evaluation.euler_f_policy_only()  (your existing file)
"""

from dataclasses import replace
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize

from ..config import ModelParams, NetParams, TrainParams
from ..networks import PolicyNet
from ..evaluation import euler_f_policy_only

from .moments import CRNDesign, PathDataset, simulate_paths_crn, make_identity_weight_matrix
from .smm import _train_policy_obj2_inner, transform_tilde_to_theta


def _instruments(k: np.ndarray, z: np.ndarray, mp: ModelParams) -> np.ndarray:
    """Instruments q = [1, log z, k, I/k].  I/k is filled later."""
    k = np.asarray(k, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)
    logz = np.log(np.maximum(z, 1e-12)).astype(np.float32)
    ones = np.ones_like(k, dtype=np.float32)
    return np.stack([ones, logz, k, np.zeros_like(k, dtype=np.float32)], axis=1)


class GMMEstimator:
    """GMM (Flavor A) with nested NN solve at each theta."""

    def __init__(
        self,
        *,
        mp_template: ModelParams,
        npol: NetParams,
        inner_tp: TrainParams,
        W: Optional[np.ndarray] = None,
        crn_design: CRNDesign,
        n_states: int = 2000,
        n_shocks: int = 64,
        invest_targets: Optional[Dict[str, float]] = None,
        burn_in: int = 0,
        seed: int = 123,
    ):
        self.mp_template = mp_template
        self.npol = npol
        self.inner_tp = inner_tp
        self.design = crn_design
        self.n_states = int(n_states)
        self.n_shocks = int(n_shocks)
        self.seed = int(seed)

        # Option G1 (augmentation): add two additional moment conditions that
        # directly target investment distribution moments computed from the
        # *truth* synthetic dataset:
        #   mean(I/k) and var(I/k).
        # This strengthens identification because Euler orthogonality alone can
        # be satisfied by many parameter sets.
        self.invest_targets = invest_targets or {}
        self._use_invest_targets = (
            ("mean_I_over_k" in self.invest_targets)
            and ("var_I_over_k" in self.invest_targets)
        )

        base_dim = 4
        extra_dim = 2 if self._use_invest_targets else 0
        total_dim = base_dim + extra_dim
        self.W = W if W is not None else make_identity_weight_matrix(total_dim)
        self._warm_start_policy: Optional[PolicyNet] = None
        self._rng = np.random.default_rng(self.seed)

        # IMPORTANT: burn_in should match the burn_in used to build truth targets/moments.
        self.burn_in = int(burn_in)

    def _mp_from_tilde(self, theta_tilde: np.ndarray) -> ModelParams:
        params = transform_tilde_to_theta(theta_tilde)
        return replace(
            self.mp_template,
            theta=params["theta"],
            rho=params["rho"],
            sigma_eps=params["sigma_eps"],
            psi0=params["psi0"],
        )

    def _sample_states(self, ds: PathDataset) -> Tuple[np.ndarray, np.ndarray]:
        """Sample N states from the simulated paths (flatten over time)."""
        k_flat = ds.k_curr.reshape(-1)
        z_flat = ds.z_curr.reshape(-1)
        idx = self._rng.choice(k_flat.shape[0], size=self.n_states, replace=True)
        return k_flat[idx], z_flat[idx]

    def evaluate(self, theta_tilde: np.ndarray) -> float:
        """Evaluate Q_GMM at a candidate theta_tilde."""
        mp = self._mp_from_tilde(theta_tilde)

        # 1) nested inner NN solve
        policy = _train_policy_obj2_inner(
            mp=mp, npol=self.npol, tp=self.inner_tp, warm_start_policy=self._warm_start_policy
        )
        self._warm_start_policy = policy

        # 2) forward simulate with CRN, then sample states
        ds = simulate_paths_crn(policy=policy, mp=mp, design=self.design, burn_in=self.burn_in)
        k_s, z_s = self._sample_states(ds)

        # 3) instruments q
        q = _instruments(k_s, z_s, mp)  # [N,4]

        # fill I/k using policy-implied k'
        x = tf.convert_to_tensor(np.stack([k_s, z_s], axis=1), tf.float32)
        k1 = tf.clip_by_value(policy(x), mp.k_min, mp.k_max).numpy()
        I = k1 - (1.0 - mp.delta) * k_s
        I_over_k = (I / np.maximum(k_s, mp.k_min)).astype(np.float32)
        q[:, 3] = I_over_k

        # 4) approximate E_eps[f] with Monte Carlo
        eps_std = self._rng.standard_normal(size=(self.n_states, self.n_shocks)).astype(np.float32)
        eps = (mp.sigma_eps * eps_std).astype(np.float32)

        k_tf = tf.convert_to_tensor(k_s, tf.float32)
        z_tf = tf.convert_to_tensor(z_s, tf.float32)

        f_list = []
        for j in range(self.n_shocks):
            eps_tf = tf.convert_to_tensor(eps[:, j], tf.float32)
            f = euler_f_policy_only(policy, mp, k_tf, z_tf, eps_tf)  # [N]
            f_list.append(f)

        f_bar = tf.reduce_mean(tf.stack(f_list, axis=1), axis=1).numpy()  # [N]

        # moment g_euler = E[ f_bar * q ]
        # IMPORTANT: scale moments to avoid exploding objectives due to units.
        fq = f_bar[:, None] * q  # [N,4]
        g_euler = np.mean(fq, axis=0)  # [4]
        # Floor the scale to avoid exploding moments when std is extremely small.
        scale_euler = np.maximum(np.std(fq, axis=0), 1e-3)
        g_euler_scaled = g_euler / scale_euler

        if self._use_invest_targets:
            # Augmented moments: match mean/var of I/k to targets from truth.
            mean_I = float(np.mean(I_over_k))
            var_I = float(np.var(I_over_k))

            # Normalize gaps to keep magnitude comparable.
            mean_t = float(self.invest_targets["mean_I_over_k"])
            var_t = float(self.invest_targets["var_I_over_k"])
            mean_gap = (mean_I - mean_t) / (abs(mean_t) + 1e-6)
            var_gap = (var_I - var_t) / (abs(var_t) + 1e-6)

            g_hat = np.concatenate(
                [g_euler_scaled, np.array([mean_gap, var_gap], dtype=np.float64)], axis=0
            )
        else:
            g_hat = g_euler_scaled

        Q = float(g_hat.T @ self.W @ g_hat)
        return Q if np.isfinite(Q) else 1e9

    def fit(self, *, x0: np.ndarray, max_evals: int = 30) -> Tuple[np.ndarray, Dict[str, float]]:
        """Run derivative-free outer optimization (Nelder-Mead)."""
        x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
        res = minimize(
            fun=lambda x: self.evaluate(x),
            x0=x0,
            method="Nelder-Mead",
            options={"maxfev": int(max_evals), "maxiter": int(max_evals), "disp": False},
        )
        diag = {
            "success": bool(res.success),
            "status": int(res.status),
            "message": str(res.message),
            "nfev": int(res.nfev),
            "final_loss": float(res.fun),
        }
        return np.asarray(res.x, dtype=np.float64), diag

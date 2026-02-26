from __future__ import annotations

"""SMM estimator (derivative-free outer optimizer) for the BASIC model.

Each evaluation of Q_SMM(theta):
  1) theta_tilde -> constrained theta
  2) inner solve (train policy via Obj2)
  3) forward simulate with CRN  (synthetic data)
  4) compute moments
  5) quadratic loss

Outer optimizer: SciPy Nelder-Mead (derivative-free).
"""

from dataclasses import replace
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize

from ..config import ModelParams, NetParams, TrainParams
from ..networks import PolicyNet
from ..objectives import obj2_batch_loss
from ..simulation import simulate_ergodic_dataset, set_global_seed

from .moments import (
    CRNDesign,
    MomentSpec,
    PathDataset,
    build_default_moment_spec,
    compute_moments,
    make_identity_weight_matrix,
    simulate_paths_crn,
)


def _clip_and_apply(opt: tf.keras.optimizers.Optimizer, grads, vars_, clip: float) -> None:
    """Global-norm clipping + Adam update."""
    grads, _ = tf.clip_by_global_norm(grads, clip)
    opt.apply_gradients(zip(grads, vars_))


def transform_tilde_to_theta(theta_tilde: np.ndarray) -> Dict[str, float]:
    """Unconstrained -> constrained mapping.

    theta_tilde = [tilde_theta, tilde_rho, tilde_sigma, tilde_psi0]
    """
    theta_tilde = np.asarray(theta_tilde, dtype=np.float64).reshape(-1)
    if theta_tilde.shape != (4,):
        raise ValueError("theta_tilde must be length 4")

    t_theta, t_rho, t_sigma, t_psi = theta_tilde

    theta = 1.0 / (1.0 + np.exp(-t_theta))     # (0,1)
    rho = 1.0 / (1.0 + np.exp(-t_rho))         # (0,1)
    sigma_eps = np.log1p(np.exp(t_sigma))      # (0,+inf)
    psi0 = np.log1p(np.exp(t_psi))             # (0,+inf)

    return {
        "theta": float(theta),
        "rho": float(rho),
        "sigma_eps": float(sigma_eps),
        "psi0": float(psi0),
    }


def _train_policy_obj2_inner(
    *,
    mp: ModelParams,
    npol: NetParams,
    tp: TrainParams,
    warm_start_policy: Optional[PolicyNet] = None,
) -> PolicyNet:
    """Inner solve used by estimation: train policy net using Objective 2.

    We keep it simple: just train policy; no value net needed.
    """
    set_global_seed(tp.seed)

    policy = PolicyNet(npol, mp.k_min, mp.k_max)
    _ = policy(tf.zeros((1, 2), dtype=tf.float32))

    # warm-start speeds up derivative-free outer search
    if warm_start_policy is not None:
        policy.set_weights(warm_start_policy.get_weights())

    opt = tf.keras.optimizers.Adam(tp.lr_policy)

    # initial ergodic buffer
    k_buf, z_buf = simulate_ergodic_dataset(policy, mp, tp, seed=tp.seed + 11)
    rng = np.random.default_rng(tp.seed + 99)

    for epoch in range(1, tp.epochs + 1):
        # refresh ergodic dataset occasionally (policy changes during training)
        if epoch == 1 or (epoch % tp.ergodic_refresh_every == 0):
            k_buf, z_buf = simulate_ergodic_dataset(
                policy, mp, tp, seed=tp.seed + 110 + epoch
            )

        for _ in range(tp.steps_per_epoch):
            idx = rng.choice(len(k_buf), size=tp.batch_size, replace=True)
            k = tf.convert_to_tensor(k_buf[idx], tf.float32)
            z = tf.convert_to_tensor(z_buf[idx], tf.float32)

            with tf.GradientTape() as tape:
                loss = obj2_batch_loss(policy, mp, k, z)

            grads = tape.gradient(loss, policy.trainable_variables)
            _clip_and_apply(opt, grads, policy.trainable_variables, tp.grad_clip)

    return policy


class SMMEstimator:
    """Simulated Method of Moments (SMM) with nested NN solve at each theta."""

    def __init__(
        self,
        *,
        mp_template: ModelParams,
        npol: NetParams,
        inner_tp: TrainParams,
        moment_spec: Optional[MomentSpec] = None,
        W: Optional[np.ndarray] = None,
        crn_design: CRNDesign,
        target_moments: Dict[str, float],
        burn_in: int = 0,
    ):
        self.mp_template = mp_template
        self.npol = npol
        self.inner_tp = inner_tp
        self.spec = moment_spec or build_default_moment_spec()
        self.W = W if W is not None else make_identity_weight_matrix(len(self.spec.names))
        self.design = crn_design
        self.m_target = target_moments

        # IMPORTANT: burn_in must match the burn_in used to construct target_moments.
        # Otherwise the outer objective optimizes moments from a different part of the path.
        self.burn_in = int(burn_in)

        self._warm_start_policy: Optional[PolicyNet] = None
        self._mhat_vec = np.asarray([self.m_target[n] for n in self.spec.names], dtype=np.float64)

    def _mp_from_tilde(self, theta_tilde: np.ndarray) -> ModelParams:
        params = transform_tilde_to_theta(theta_tilde)
        return replace(
            self.mp_template,
            theta=params["theta"],
            rho=params["rho"],
            sigma_eps=params["sigma_eps"],
            psi0=params["psi0"],
        )

    def evaluate(self, theta_tilde: np.ndarray) -> float:
        """Evaluate Q_SMM at a candidate theta_tilde."""
        mp = self._mp_from_tilde(theta_tilde)

        # 1) nested inner NN solve
        policy = _train_policy_obj2_inner(
            mp=mp, npol=self.npol, tp=self.inner_tp, warm_start_policy=self._warm_start_policy
        )
        self._warm_start_policy = policy

        # 2) forward simulate with CRN (synthetic data)
        ds: PathDataset = simulate_paths_crn(
            policy=policy, mp=mp, design=self.design, burn_in=self.burn_in
        )

        # 3) compute moments and loss
        m_sim = compute_moments(ds, mp, self.spec)
        m_sim_vec = np.asarray([m_sim[n] for n in self.spec.names], dtype=np.float64)
        g = self._mhat_vec - m_sim_vec
        Q = float(g.T @ self.W @ g)
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

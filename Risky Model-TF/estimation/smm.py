"""estimation.smm

Simulated Method of Moments (SMM) for the risky-debt model.

"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Callable, Dict, List, Optional, Tuple

import json
import os
import time

import numpy as np
import tensorflow as tf

from risky_debt.config import ModelParams, NetParams, TrainParams
from risky_debt.networks import PolicyNet, PricingNet
from risky_debt.primitives import profit_pi, solvency_weight
from risky_debt.simulation import set_global_seed
from risky_debt.objectives import obj1_loss

from .moments import (
    compute_default_moment_vector,
    moment_distance,
)


# ------------------------------
# Small helper: pure-python Nelder–Mead
# ------------------------------


def _nelder_mead(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    step: np.ndarray,
    bounds: List[Tuple[float, float]],
    max_evals: int = 60,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, float, int]:
    """Very small Nelder Mead.
    Returns: (best_x, best_f, evals_used)
    """

    def project(x: np.ndarray) -> np.ndarray:
        y = x.copy()
        for i, (lo, hi) in enumerate(bounds):
            y[i] = float(np.clip(y[i], lo, hi))
        return y

    x0 = project(np.asarray(x0, dtype=float))
    step = np.asarray(step, dtype=float)
    n = x0.size

    # initial simplex
    simplex = [x0]
    for i in range(n):
        xi = x0.copy()
        xi[i] = xi[i] + step[i]
        simplex.append(project(xi))
    simplex = np.stack(simplex, axis=0)  # (n+1, n)

    fvals = np.array([f(x) for x in simplex], dtype=float)
    evals = n + 1

    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

    while evals < max_evals:
        # order
        idx = np.argsort(fvals)
        simplex = simplex[idx]
        fvals = fvals[idx]

        # stopping
        if np.max(np.abs(fvals - fvals[0])) < tol:
            break

        x_best = simplex[0]
        x_worst = simplex[-1]
        x_cent = np.mean(simplex[:-1], axis=0)

        # reflection
        x_r = project(x_cent + alpha * (x_cent - x_worst))
        f_r = f(x_r)
        evals += 1

        if fvals[0] <= f_r < fvals[-2]:
            simplex[-1] = x_r
            fvals[-1] = f_r
            continue

        if f_r < fvals[0]:
            # expansion
            x_e = project(x_cent + gamma * (x_r - x_cent))
            f_e = f(x_e)
            evals += 1
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
            continue

        # contraction
        x_c = project(x_cent + rho * (x_worst - x_cent))
        f_c = f(x_c)
        evals += 1
        if f_c < fvals[-1]:
            simplex[-1] = x_c
            fvals[-1] = f_c
            continue

        # shrink
        for i in range(1, n + 1):
            simplex[i] = project(x_best + sigma * (simplex[i] - x_best))
            fvals[i] = f(simplex[i])
        evals += n

    # final best
    idx = int(np.argmin(fvals))
    return simplex[idx], float(fvals[idx]), evals


# ------------------------------
# Forward simulation (policy-induced) for synthetic data + simulated data
# ------------------------------


def forward_simulate_dataset(
    policy: PolicyNet,
    qnet: PricingNet,
    mp: ModelParams,
    tp: TrainParams,
    eps: np.ndarray,
    T: int,
    burn_in: int,
) -> Dict[str, np.ndarray]:
    """Forward simulate (k,b,z) given CRN eps[n,t].

    Returns a flat dataset (stack paths/time after burn-in).
    """

    eps = np.asarray(eps, dtype=np.float32)
    n_paths = eps.shape[0]
    assert eps.shape[1] >= T + 1

    # initial states
    k = tf.random.uniform((n_paths,), tp.k0_low, tp.k0_high, dtype=tf.float32)
    b = tf.random.uniform((n_paths,), tp.b0_low, tp.b0_high, dtype=tf.float32)
    z = tf.random.uniform((n_paths,), tp.z0_low, tp.z0_high, dtype=tf.float32)

    # storage lists (Tensor -> numpy at the end)
    out: Dict[str, List[tf.Tensor]] = {
        "k": [],
        "b": [],
        "z": [],
        "k_next": [],
        "b_next": [],
        "z_next": [],
        "I": [],
        "q": [],
        "spread": [],
        "default": [],
    }

    beta = 1.0 / (1.0 + mp.r)

    for t in range(T):
        x = tf.stack([k, b, z], axis=1)
        kb_next = policy(x)
        k_next = tf.maximum(kb_next[:, 0], mp.k_min)
        b_next = kb_next[:, 1]

        q_in = tf.stack([z, k_next, b_next], axis=1)
        q = qnet(q_in)
        q_clip = tf.clip_by_value(q, 1e-6, 1.0 - 1e-6)
        r_tilde = (1.0 / q_clip) - 1.0
        spread = r_tilde - mp.r

        # shock update
        eps_t = tf.convert_to_tensor(eps[:, t], dtype=tf.float32)
        z_next = tf.exp(
            tf.cast(mp.rho, tf.float32) * tf.math.log(tf.maximum(z, mp.z_min)) + eps_t
        )

        # default indicator (your convention: s_{t+1} <= 0.5)
        s_next = solvency_weight(k_next, b_next, z_next, mp, tp.kappa_solv)
        default = tf.cast(s_next <= 0.5, tf.float32)

        # investment
        I = k_next - (1.0 - mp.delta) * k

        if t >= burn_in:
            out["k"].append(k)
            out["b"].append(b)
            out["z"].append(z)
            out["k_next"].append(k_next)
            out["b_next"].append(b_next)
            out["z_next"].append(z_next)
            out["I"].append(I)
            out["q"].append(q)
            out["spread"].append(spread)
            out["default"].append(default)

        k, b, z = k_next, b_next, z_next

    # stack -> flatten
    dataset: Dict[str, np.ndarray] = {}
    for key, seq in out.items():
        if len(seq) == 0:
            dataset[key] = np.zeros((0,), dtype=float)
            continue
        mat = tf.stack(seq, axis=1)  # (n_paths, T-burn)
        dataset[key] = tf.reshape(mat, (-1,)).numpy().astype(float)
    return dataset


# ------------------------------
# Inner solver (Objective 1) with warm-start
# ------------------------------


class _InnerObjective1Solver:

    def __init__(
        self,
        mp: ModelParams,
        npol: NetParams,
        nq: NetParams,
        tp_inner: TrainParams,
        nu_zp: float,
    ):
        self.npol = npol
        self.nq = nq
        self.tp_inner = tp_inner
        self.nu_zp = nu_zp

        self.policy = PolicyNet(npol, mp.k_min, mp.b_min, mp.b_max)
        self.qnet = PricingNet(nq, mp.q_min, mp.q_max)
        _ = self.policy(tf.zeros((1, 3), tf.float32))
        _ = self.qnet(tf.zeros((1, 3), tf.float32))

        self.opt_policy = tf.keras.optimizers.Adam(tp_inner.lr_policy)
        self.opt_q = tf.keras.optimizers.Adam(tp_inner.lr_q)

    def solve(self, mp: ModelParams, op1) -> Tuple[PolicyNet, PricingNet]:
        """Run a *small* training budget and return (policy, qnet)."""
        # Note: we assume random seed has been set.
        for _ in range(self.tp_inner.epochs):
            for _ in range(self.tp_inner.steps_per_epoch):
                with tf.GradientTape(persistent=True) as tape:
                    loss, train_reward, zp_loss = obj1_loss(
                        policy=self.policy,
                        qnet=self.qnet,
                        mp=mp,
                        tp=self.tp_inner,
                        op1=op1,
                    )
                g_pol = tape.gradient(loss, self.policy.trainable_variables)
                g_q = tape.gradient(loss, self.qnet.trainable_variables)
                del tape
                self.opt_policy.apply_gradients(
                    zip(g_pol, self.policy.trainable_variables)
                )
                self.opt_q.apply_gradients(zip(g_q, self.qnet.trainable_variables))
        return self.policy, self.qnet


# ------------------------------
# SMM
# ------------------------------


def estimate_smm(
    out_dir: str,
    mp_true: ModelParams,
    npol: NetParams,
    nq: NetParams,
    tp_base: TrainParams,
    policy_true: PolicyNet,
    qnet_true: PricingNet,
    est_bounds: Dict[str, Tuple[float, float]],
    max_evals: int = 60,
    inner_epochs: int = 3,
    inner_steps_per_epoch: int = 20,
    sim_T: int = 200,
    sim_burn: int = 50,
    sim_n_paths: int = 128,
    seed: int = 1234,
) -> Dict[str, object]:
    """Run SMM.

    Returns a dict with keys:
      - theta_hat (dict)
      - objective
      - moment_names, m_data, m_sim
      - ParamError, MomentFit
    """

    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    # 1) CRN shocks: fixed for all evaluations (critical!)
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, mp_true.sigma_eps, size=(sim_n_paths, sim_T + 1)).astype(
        np.float32
    )

    # 2) "Observed" synthetic data generated from the already-trained NN (policy_true, qnet_true)
    set_global_seed(seed)
    data = forward_simulate_dataset(
        policy=policy_true,
        qnet=qnet_true,
        mp=mp_true,
        tp=tp_base,
        eps=eps,
        T=sim_T,
        burn_in=sim_burn,
    )
    moment_names, m_data = compute_default_moment_vector(
        data, mp_true, include_risky_debt_moments=True
    )
    # Standardize / reweight moments so large-scale moments don't dominate.
    # Uses diagonal W with entries 1/(|m_data|+eps)^2, so objective is sum(((m_sim-m_data)/scale)^2).
    _eps_w = 1e-6
    _scale = np.abs(m_data) + _eps_w
    W = np.diag(1.0 / (_scale * _scale))

    # --- extra dampening for unstable / degenerate moments ---
    # This keeps a few problematic moments (e.g. near-zero var(I/k) or degenerate default)
    # from dominating SMM and pulling parameters (especially psi0) to compensate.
    damp = {
        "var_I_over_k": 0.05,
        "ac1_I_over_k": 0.05,
        "default_rate": 0.10,
        "prob_b_pos": 0.10,
        "mean_spread": 0.10,
    }
    for i, nm in enumerate(moment_names):
        if nm in damp:
            W[i, i] *= damp[nm]

    np.savez(
        os.path.join(out_dir, "smm_synth_data.npz"),
        **{k: v for k, v in data.items()},
    )

    # 3) inner solver config (small training budget)
    tp_inner = replace(
        tp_base,
        epochs=inner_epochs,
        steps_per_epoch=inner_steps_per_epoch,
    )

    # A small Op1Params-like object is expected by obj1_loss.
    # It is defined in risky_debt/objectives.py.
    from risky_debt.config import Obj1Params  # local import to avoid circular imports

    op1 = Obj1Params()  # uses defaults
    inner = _InnerObjective1Solver(mp_true, npol, nq, tp_inner, nu_zp=op1.nu_zp)

    # 4) parameter vector definition
    param_names = list(est_bounds.keys())
    bounds = [est_bounds[n] for n in param_names]
    x0 = np.array([getattr(mp_true, n) for n in param_names], dtype=float)
    step = np.array([(hi - lo) * 0.05 for (lo, hi) in bounds], dtype=float)
    step = np.maximum(step, 1e-3)

    # 5) objective evaluator
    cache: Dict[str, float] = {}
    eval_counter = {"n": 0}

    def f(x: np.ndarray) -> float:

        eval_counter["n"] += 1
        key = ",".join([f"{v:.6g}" for v in x])
        if key in cache:
            return cache[key]

        # build candidate params
        mp_cand = mp_true
        for name, val in zip(param_names, x):
            mp_cand = replace(mp_cand, **{name: float(val)})

        # inner solve (warm-start)
        set_global_seed(seed + 1)
        pol_c, q_c = inner.solve(mp_cand, op1)

        # simulate under candidate using SAME eps (CRN)
        set_global_seed(seed + 2)
        sim = forward_simulate_dataset(
            policy=pol_c,
            qnet=q_c,
            mp=mp_cand,
            tp=tp_base,
            eps=eps,
            T=sim_T,
            burn_in=sim_burn,
        )
        _, m_sim = compute_default_moment_vector(
            sim, mp_cand, include_risky_debt_moments=True
        )
        obj = moment_distance(m_data, m_sim, W=W)

        # --- DEBUG: print top moment errors (standardized relative errors) ---
        eps_w = 1e-8
        rel = (m_sim - m_data) / (np.abs(m_data) + eps_w)
        order = np.argsort(rel**2)[::-1]
        print(f"\n[SMM] Eval {eval_counter['n']}: objective={obj:.6g}")
        print("[SMM] Top moment errors:")
        for j in order[:8]:
            nm = moment_names[j]
            print(
                f"  {nm:>14s}: data={m_data[j]: .6g}  sim={m_sim[j]: .6g}  rel_err={rel[j]: .3f}"
            )
        cache[key] = obj
        return obj

    # 6) optimize
    x_hat, f_hat, evals = _nelder_mead(
        f=f,
        x0=x0,
        step=step,
        bounds=bounds,
        max_evals=max_evals,
    )

    # 7) final simulation/moments at theta_hat
    mp_hat = mp_true
    for name, val in zip(param_names, x_hat):
        mp_hat = replace(mp_hat, **{name: float(val)})
    set_global_seed(seed + 3)
    pol_hat, q_hat = inner.solve(mp_hat, op1)
    set_global_seed(seed + 4)
    sim_hat = forward_simulate_dataset(
        policy=pol_hat,
        qnet=q_hat,
        mp=mp_hat,
        tp=tp_base,
        eps=eps,
        T=sim_T,
        burn_in=sim_burn,
    )
    _, m_hat = compute_default_moment_vector(
        sim_hat, mp_hat, include_risky_debt_moments=True
    )

    # Effectiveness metrics
    param_error = {
        n: float(abs(getattr(mp_hat, n) - getattr(mp_true, n))) for n in param_names
    }
    moment_fit = {
        moment_names[i]: float(abs(m_data[i] - m_hat[i]))
        for i in range(len(moment_names))
    }

    res: Dict[str, object] = {
        "method": "SMM",
        "param_names": param_names,
        "theta_true": {n: float(getattr(mp_true, n)) for n in param_names},
        "theta_hat": {n: float(getattr(mp_hat, n)) for n in param_names},
        "objective": float(f_hat),
        "evals": int(evals),
        "moment_names": moment_names,
        "m_data": m_data.tolist(),
        "m_hat": m_hat.tolist(),
        "ParamError": param_error,
        "MomentFit": moment_fit,
        "runtime_sec": float(time.time() - t0),
    }

    with open(
        os.path.join(out_dir, "smm_results.json"), "w", encoding="utf-8"
    ) as f_out:
        json.dump(res, f_out, indent=2)

    return res

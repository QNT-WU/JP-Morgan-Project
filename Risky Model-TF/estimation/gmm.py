"""estimation.gmm

GMM for the risky-debt model.

"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Tuple

import json
import os
import time

import numpy as np
import tensorflow as tf

from risky_debt.config import ModelParams, NetParams, TrainParams, Obj1Params
from risky_debt.networks import PolicyNet, PricingNet
from risky_debt.primitives import recovery_R, solvency_weight, equity_payout_d
from risky_debt.simulation import set_global_seed
from risky_debt.objectives import obj1_loss

from .smm import _nelder_mead, forward_simulate_dataset, _InnerObjective1Solver


def _build_instruments(dataset: Dict[str, np.ndarray], mp: ModelParams) -> np.ndarray:
    """Instruments Z_t (simple, low-dimensional).

    Z = [1, ln z, k, b, I/k]
    """
    k = dataset["k"].astype(float)
    b = dataset["b"].astype(float)
    z = dataset["z"].astype(float)
    I = dataset["I"].astype(float)

    lnz = np.log(np.maximum(z, mp.z_min))
    inv = I / np.maximum(k, mp.k_min)

    # Match I/k scaling convention (some datasets store I/k in percent units).
    inv_mean = float(np.nanmean(inv)) if inv.size > 0 else 0.0
    if np.isfinite(inv_mean) and inv_mean > 2.0:
        inv = inv / 100.0

    Z = np.column_stack(
        [
            np.ones_like(k),
            lnz,
            k,
            b,
            inv,
        ]
    )

    # standardize columns except constant to reduce conditioning issues
    Zs = Z.copy()
    for j in range(1, Z.shape[1]):
        mu = np.mean(Z[:, j])
        sd = np.std(Z[:, j])
        if sd < 1e-8:
            sd = 1.0
        Zs[:, j] = (Z[:, j] - mu) / sd
    return Zs


def _euler_and_zp_residuals(
    policy: PolicyNet,
    qnet: PricingNet,
    mp: ModelParams,
    tp: TrainParams,
    k: tf.Tensor,
    b: tf.Tensor,
    z: tf.Tensor,
    eps: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Compute (Rk, Rb, m_zp) per observation.

    This uses the same autodiff logic style as risky_debt.evaluation.
    Here we use J = d only (policy-only) to keep GMM inner solve light.
    """

    # policy action
    x = tf.stack([k, b, z], axis=1)
    kb_next = policy(x)
    k_next = tf.maximum(kb_next[:, 0], mp.k_min)
    b_next = kb_next[:, 1]

    # shock
    z_next = tf.exp(
        tf.cast(mp.rho, tf.float32) * tf.math.log(tf.maximum(z, mp.z_min)) + eps
    )

    # price q at issuance node
    q_in = tf.stack([z, k_next, b_next], axis=1)
    q = qnet(q_in)

    # payout d today
    d = equity_payout_d(k, k_next, b, b_next, z, q, mp, tp.kappa_issue)

    # Euler residuals via autodiff of J=d
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(k_next)
        tape.watch(b_next)

        q2 = qnet(tf.stack([z, k_next, b_next], axis=1))
        d2 = equity_payout_d(k, k_next, b, b_next, z, q2, mp, tp.kappa_issue)
        Jsum = tf.reduce_sum(d2)

    dJ_dk = tape.gradient(Jsum, k_next)
    dJ_db = tape.gradient(Jsum, b_next)
    del tape

    # gating by solvency weight at t+1
    s_next = solvency_weight(k_next, b_next, z_next, mp, tp.kappa_solv)
    Rk = s_next * dJ_dk
    Rb = s_next * dJ_db

    # pricing ZP residual (same as your objectives.py)
    Rrec = recovery_R(k_next, z_next, mp)
    pay = (1.0 - s_next) * Rrec + s_next * (b_next / tf.clip_by_value(q, 1e-6, 1.0))
    m_zp = (1.0 + mp.r) * b_next - pay

    return Rk, Rb, m_zp


def estimate_gmm(
    out_dir: str,
    mp_true: ModelParams,
    npol: NetParams,
    nq: NetParams,
    tp_base: TrainParams,
    data: Dict[str, np.ndarray],
    est_bounds: Dict[str, Tuple[float, float]],
    max_evals: int = 60,
    inner_epochs: int = 3,
    inner_steps_per_epoch: int = 20,
    seed: int = 4321,
) -> Dict[str, object]:
    """Run GMM on the *fixed* dataset produced by SMM synthetic-data stage."""

    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    # fixed instruments
    Z = _build_instruments(data, mp_true)
    Z_tf = tf.convert_to_tensor(Z, tf.float32)

    # fixed shocks for residual evaluation (CRN)
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, mp_true.sigma_eps, size=(Z.shape[0],)).astype(np.float32)
    eps_tf = tf.convert_to_tensor(eps, tf.float32)

    # inner solver (warm-start)
    tp_inner = replace(
        tp_base, epochs=inner_epochs, steps_per_epoch=inner_steps_per_epoch
    )
    op1 = Obj1Params()
    inner = _InnerObjective1Solver(mp_true, npol, nq, tp_inner, nu_zp=op1.nu_zp)

    # parameter vector
    param_names = list(est_bounds.keys())
    bounds = [est_bounds[n] for n in param_names]
    x0 = np.array([getattr(mp_true, n) for n in param_names], dtype=float)
    step = np.array([(hi - lo) * 0.05 for (lo, hi) in bounds], dtype=float)
    step = np.maximum(step, 1e-3)

    # fixed data tensors
    k_tf = tf.convert_to_tensor(data["k"].astype(np.float32))
    b_tf = tf.convert_to_tensor(data["b"].astype(np.float32))
    z_tf = tf.convert_to_tensor(data["z"].astype(np.float32))

    cache: Dict[str, float] = {}

    def f(x: np.ndarray) -> float:
        key = ",".join([f"{v:.6g}" for v in x])
        if key in cache:
            return cache[key]

        mp_cand = mp_true
        for name, val in zip(param_names, x):
            mp_cand = replace(mp_cand, **{name: float(val)})

        # inner solve
        set_global_seed(seed + 10)
        pol, qnet = inner.solve(mp_cand, op1)

        # residuals on fixed data
        Rk, Rb, mzp = _euler_and_zp_residuals(
            pol, qnet, mp_cand, tp_base, k_tf, b_tf, z_tf, eps_tf
        )

        # moments g = mean(Z * residual)
        # Reweight / standardize moments so one block cannot dominate numerically.
        # Scale each mean moment by an estimate of its sampling std: std(h_t)+eps,
        # where h_t are the per-observation moment conditions.
        eps_w = tf.constant(1e-6, dtype=tf.float32)

        # h_t for each block (shape: [N, d])
        h_k = Z_tf * tf.expand_dims(Rk, 1)
        h_b = Z_tf * tf.expand_dims(Rb, 1)
        h_zp = Z_tf * tf.expand_dims(mzp, 1)

        # std over observations (per instrument)
        s_k = tf.math.reduce_std(h_k, axis=0) + eps_w
        s_b = tf.math.reduce_std(h_b, axis=0) + eps_w
        s_zp = tf.math.reduce_std(h_zp, axis=0) + eps_w

        # standardized mean moments
        gk = tf.reduce_mean(h_k, axis=0) / s_k
        gb = tf.reduce_mean(h_b, axis=0) / s_b
        gz = tf.reduce_mean(h_zp, axis=0) / s_zp
        g = tf.concat([gk, gb, gz], axis=0)  # (3*d,)

        obj = float(tf.reduce_sum(tf.square(g)).numpy())
        cache[key] = obj
        return obj

    x_hat, f_hat, evals = _nelder_mead(
        f=f,
        x0=x0,
        step=step,
        bounds=bounds,
        max_evals=max_evals,
    )

    mp_hat = mp_true
    for name, val in zip(param_names, x_hat):
        mp_hat = replace(mp_hat, **{name: float(val)})

    # effectiveness metrics (parameter recovery)
    param_error = {
        n: float(abs(getattr(mp_hat, n) - getattr(mp_true, n))) for n in param_names
    }

    res: Dict[str, object] = {
        "method": "GMM",
        "param_names": param_names,
        "theta_true": {n: float(getattr(mp_true, n)) for n in param_names},
        "theta_hat": {n: float(getattr(mp_hat, n)) for n in param_names},
        "objective": float(f_hat),
        "evals": int(evals),
        "ParamError": param_error,
        "runtime_sec": float(time.time() - t0),
    }

    with open(
        os.path.join(out_dir, "gmm_results.json"), "w", encoding="utf-8"
    ) as f_out:
        json.dump(res, f_out, indent=2)

    return res

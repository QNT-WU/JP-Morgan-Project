"""Simulated Method of Moments (SMM) for the risky-debt model.

This implementation is materially closer to the user's final plan:
- baseline structural parameters: (theta, psi0, alpha)
- one fixed synthetic observed dataset shared across estimators
- hard continuation based on a policy-evaluated finite-horizon continuation value
- two-step SMM with two second-stage variants:
    * SMM-A: standard covariance of data moments
    * SMM-B: Newey--West covariance of data moments
- multi-start local optimization with transformed parameters and robustness reports
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import json
import os
import time

import numpy as np
import tensorflow as tf

from risky_debt.config import ModelParams, NetParams, TrainParams, Obj1Params
from risky_debt.networks import PolicyNet, PricingNet
from risky_debt.primitives import equity_cashflow_total_e, equity_payout_d_total, recovery_R
from risky_debt.simulation import set_global_seed
from risky_debt.grid_compare import interp_grid_3d

from .progress import EstimationProgressReporter
from .inner_obj1 import ReusableInnerObjective1Solver, finite_horizon_predefault_continuation

# Backward-compatible alias retained for smoke tests and legacy imports.
_InnerObjective1Solver = ReusableInnerObjective1Solver

from .common import (
    apply_params,
    bounded_from_unconstrained,
    bounds_in_order,
    compute_smm_moments,
    covariance_of_mean,
    dataset_pricing_default_summary,
    default_estimation_bounds,
    generate_parameter_starts,
    numerical_jacobian,
    newey_west_lag_length,
    ordered_param_names,
    overall_param_recovery_score,
    stabilized_inverse,
    summarize_multistart_runs,
    summarize_param_errors,
    unconstrained_from_bounded,
    save_identification_report,
    make_json_serializable,
)


def _nelder_mead(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    step: np.ndarray,
    bounds: List[Tuple[float, float]],
    max_evals: int = 60,
    tol: float = 1e-6,
) -> Dict[str, object]:
    """Small Nelder-Mead optimizer with bound projection and metadata."""

    def project(x: np.ndarray) -> np.ndarray:
        """Project a candidate point back into the admissible parameter bounds."""
        y = x.copy()
        for i, (lo, hi) in enumerate(bounds):
            y[i] = float(np.clip(y[i], lo, hi))
        return y

    x0 = project(np.asarray(x0, dtype=float))
    step = np.asarray(step, dtype=float)
    n = x0.size

    simplex = [x0]
    for i in range(n):
        xi = x0.copy()
        xi[i] += step[i]
        simplex.append(project(xi))
    simplex = np.stack(simplex, axis=0)

    fvals = np.array([f(x) for x in simplex], dtype=float)
    evals = simplex.shape[0]

    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    converged = False

    while evals < max_evals:
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]

        simplex_diam = float(np.max(np.linalg.norm(simplex - simplex[0], axis=1)))
        f_spread = float(np.max(np.abs(fvals - fvals[0])))
        if simplex_diam < tol and f_spread < tol:
            converged = True
            break

        x_best = simplex[0]
        x_worst = simplex[-1]
        centroid = np.mean(simplex[:-1], axis=0)

        x_r = project(centroid + alpha * (centroid - x_worst))
        f_r = float(f(x_r))
        evals += 1

        if fvals[0] <= f_r < fvals[-2]:
            simplex[-1] = x_r
            fvals[-1] = f_r
            continue

        if f_r < fvals[0]:
            x_e = project(centroid + gamma * (x_r - centroid))
            f_e = float(f(x_e))
            evals += 1
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
            continue

        x_c = project(centroid + rho * (x_worst - centroid))
        f_c = float(f(x_c))
        evals += 1
        if f_c < fvals[-1]:
            simplex[-1] = x_c
            fvals[-1] = f_c
            continue

        for i in range(1, n + 1):
            simplex[i] = project(x_best + sigma * (simplex[i] - x_best))
            fvals[i] = float(f(simplex[i]))
        evals += n

    order = np.argsort(fvals)
    simplex = simplex[order]
    fvals = fvals[order]
    simplex_diam = float(np.max(np.linalg.norm(simplex - simplex[0], axis=1)))
    f_spread = float(np.max(np.abs(fvals - fvals[0])))
    return {
        "x": simplex[0],
        "objective": float(fvals[0]),
        "evals": int(evals),
        "converged": bool(converged or (simplex_diam < tol and f_spread < tol)),
        "simplex_diameter": simplex_diam,
        "f_spread": f_spread,
    }


# ------------------------------
# Forward simulation
# ------------------------------


def _future_width_from_setting(continuation_horizon: int, remaining_steps: int) -> int:
    """Return the number of future shocks used in continuation evaluation."""
    if continuation_horizon <= 0:
        return max(0, int(remaining_steps))
    return max(0, min(int(continuation_horizon), int(remaining_steps)))


def _forward_simulate_dataset_benchmark(
    benchmark: Dict[str, np.ndarray],
    mp: ModelParams,
    tp: TrainParams,
    eps: np.ndarray,
    T: int,
    burn_in: int,
) -> Dict[str, np.ndarray]:
    """Forward simulation using the grid benchmark as the true DGP."""
    eps = np.asarray(eps, dtype=np.float32)
    n_paths = int(eps.shape[0])
    if eps.shape[1] < T + 1:
        raise ValueError("eps must have at least T+1 columns.")

    k_grid = np.asarray(benchmark["k_grid"], dtype=float)
    b_grid = np.asarray(benchmark["b_grid"], dtype=float)
    z_grid = np.asarray(benchmark["z_grid"], dtype=float)
    kp_star = np.asarray(benchmark["policy_kp_star"], dtype=float)
    bp_star = np.asarray(benchmark["policy_bp_star"], dtype=float)
    c_star = np.asarray(benchmark["C_star"], dtype=float)
    q_star = np.asarray(benchmark["q_star"], dtype=float).transpose(1, 2, 0)

    rng = np.random.default_rng(int(tp.seed) + 7000)
    k = rng.uniform(tp.k0_low, tp.k0_high, size=n_paths).astype(np.float32)
    b = rng.uniform(tp.b0_low, tp.b0_high, size=n_paths).astype(np.float32)
    z = rng.uniform(tp.z0_low, tp.z0_high, size=n_paths).astype(np.float32)

    out: Dict[str, List[np.ndarray]] = {
        "k": [],
        "b": [],
        "z": [],
        "k_next": [],
        "b_next": [],
        "z_next": [],
        "I": [],
        "q": [],
        "spread": [],
        "r_tilde": [],
        "e": [],
        "d": [],
        "default": [],
        "recovery": [],
        "continuation_next": [],
        "continuation_indicator_next": [],
    }

    for t in range(T):
        k_next = interp_grid_3d(k_grid, b_grid, z_grid, kp_star, k, b, z).astype(np.float32)
        b_next = interp_grid_3d(k_grid, b_grid, z_grid, bp_star, k, b, z).astype(np.float32)
        q = interp_grid_3d(z_grid, k_grid, b_grid, np.asarray(benchmark["q_star"], dtype=float), z, k_next, b_next).astype(np.float32)
        q = np.clip(q, mp.q_min, mp.q_max)
        eps_t = eps[:, t]
        z_next = np.exp(mp.rho * np.log(np.maximum(z, mp.z_min)) + eps_t).astype(np.float32)
        cont_next = interp_grid_3d(k_grid, b_grid, z_grid, c_star, k_next, b_next, z_next).astype(np.float32)
        default = (cont_next <= 0.0).astype(np.float32)
        I = (k_next - (1.0 - mp.delta) * k).astype(np.float32)
        recovery = recovery_R(
            tf.convert_to_tensor(k_next, dtype=tf.float32),
            tf.convert_to_tensor(z_next, dtype=tf.float32),
            mp,
        ).numpy().astype(np.float32)
        r_tilde = ((1.0 / q) - 1.0).astype(np.float32)
        spread = (r_tilde - mp.r).astype(np.float32)
        cont_ind = (cont_next > 0.0).astype(np.float32)
        e = equity_cashflow_total_e(
            k=tf.convert_to_tensor(k, dtype=tf.float32),
            k_next=tf.convert_to_tensor(k_next, dtype=tf.float32),
            b=tf.convert_to_tensor(b, dtype=tf.float32),
            b_next=tf.convert_to_tensor(b_next, dtype=tf.float32),
            z=tf.convert_to_tensor(z, dtype=tf.float32),
            q=tf.convert_to_tensor(q, dtype=tf.float32),
            continuation_weight=tf.convert_to_tensor(cont_ind, dtype=tf.float32),
            mp=mp,
        ).numpy().astype(np.float32)
        d = equity_payout_d_total(
            k=tf.convert_to_tensor(k, dtype=tf.float32),
            k_next=tf.convert_to_tensor(k_next, dtype=tf.float32),
            b=tf.convert_to_tensor(b, dtype=tf.float32),
            b_next=tf.convert_to_tensor(b_next, dtype=tf.float32),
            z=tf.convert_to_tensor(z, dtype=tf.float32),
            q=tf.convert_to_tensor(q, dtype=tf.float32),
            continuation_weight=tf.convert_to_tensor(cont_ind, dtype=tf.float32),
            mp=mp,
            kappa_issue=tp.kappa_issue,
        ).numpy().astype(np.float32)

        if t >= burn_in:
            out["k"].append(k.copy())
            out["b"].append(b.copy())
            out["z"].append(z.copy())
            out["k_next"].append(k_next.copy())
            out["b_next"].append(b_next.copy())
            out["z_next"].append(z_next.copy())
            out["I"].append(I.copy())
            out["q"].append(q.copy())
            out["spread"].append(spread.copy())
            out["r_tilde"].append(r_tilde.copy())
            out["e"].append(e.copy())
            out["d"].append(d.copy())
            out["default"].append(default.copy())
            out["recovery"].append(recovery.copy())
            out["continuation_next"].append(cont_next.copy())
            out["continuation_indicator_next"].append(cont_ind.copy())

        k, b, z = k_next, b_next, z_next

    dataset: Dict[str, np.ndarray] = {}
    for key, seq in out.items():
        if not seq:
            dataset[key] = np.zeros((0,), dtype=float)
            continue
        mat = np.stack(seq, axis=1)
        dataset[key] = mat.reshape(-1).astype(float)

    t_eff = max(0, T - burn_in)
    dataset["n_paths"] = np.asarray(n_paths, dtype=np.int32)
    dataset["T_eff"] = np.asarray(t_eff, dtype=np.int32)
    dataset["burn_in"] = np.asarray(burn_in, dtype=np.int32)
    dataset["sim_T"] = np.asarray(T, dtype=np.int32)
    dataset["continuation_horizon"] = np.asarray(0, dtype=np.int32)
    dataset["eps_full"] = eps.astype(np.float32)
    dataset["dgp_source"] = np.asarray("benchmark", dtype='<U16')
    return dataset


def forward_simulate_dataset(
    policy: Optional[PolicyNet],
    qnet: Optional[PricingNet],
    mp: ModelParams,
    tp: TrainParams,
    eps: np.ndarray,
    T: int,
    burn_in: int,
    continuation_horizon: int = 0,
    benchmark: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """Forward simulate a synthetic dataset under a fixed policy/pricing rule.

    When ``benchmark`` is supplied, the synthetic observed sample is generated
    from the grid benchmark rather than the Obj1 NN solution. This is closer to
    the user's final written design for the true DGP.
    """
    if benchmark is not None:
        return _forward_simulate_dataset_benchmark(
            benchmark=benchmark,
            mp=mp,
            tp=tp,
            eps=eps,
            T=T,
            burn_in=burn_in,
        )

    if policy is None or qnet is None:
        raise ValueError("policy and qnet are required when benchmark is not provided.")

    eps = np.asarray(eps, dtype=np.float32)
    n_paths = int(eps.shape[0])
    if eps.shape[1] < T + 1:
        raise ValueError("eps must have at least T+1 columns.")

    k = tf.random.uniform((n_paths,), tp.k0_low, tp.k0_high, dtype=tf.float32)
    b = tf.random.uniform((n_paths,), tp.b0_low, tp.b0_high, dtype=tf.float32)
    z = tf.random.uniform((n_paths,), tp.z0_low, tp.z0_high, dtype=tf.float32)

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
        "r_tilde": [],
        "e": [],
        "d": [],
        "default": [],
        "recovery": [],
        "continuation_next": [],
        "continuation_indicator_next": [],
    }

    for t in range(T):
        x = tf.stack([k, b, z], axis=1)
        kb_next = policy(x)
        k_next = tf.maximum(kb_next[:, 0], mp.k_min)
        b_next = kb_next[:, 1]

        q_in = tf.stack([z, k_next, b_next], axis=1)
        q = qnet(q_in)
        q_clip = tf.clip_by_value(q, mp.q_min, mp.q_max)
        r_tilde = (1.0 / q_clip) - 1.0
        spread = r_tilde - mp.r

        eps_t = tf.convert_to_tensor(eps[:, t], dtype=tf.float32)
        z_next = tf.exp(
            tf.cast(mp.rho, tf.float32) * tf.math.log(tf.maximum(z, mp.z_min)) + eps_t
        )

        future_width = _future_width_from_setting(continuation_horizon, int(T - (t + 1)))
        eps_future = eps[:, t + 1 : t + 1 + future_width]
        cont_next = finite_horizon_predefault_continuation(
            policy=policy,
            qnet=qnet,
            mp=mp,
            tp=tp,
            k0=k_next,
            b0=b_next,
            z0=z_next,
            eps_future=tf.convert_to_tensor(eps_future, dtype=tf.float32),
        )
        default = tf.cast(cont_next <= 0.0, tf.float32)
        cont_ind = tf.cast(cont_next > 0.0, tf.float32)
        I = k_next - (1.0 - mp.delta) * k
        recovery = recovery_R(k_next, z_next, mp)
        e = equity_cashflow_total_e(
            k=k,
            k_next=k_next,
            b=b,
            b_next=b_next,
            z=z,
            q=q_clip,
            continuation_weight=cont_ind,
            mp=mp,
        )
        d = equity_payout_d_total(
            k=k,
            k_next=k_next,
            b=b,
            b_next=b_next,
            z=z,
            q=q_clip,
            continuation_weight=cont_ind,
            mp=mp,
            kappa_issue=tp.kappa_issue,
        )

        if t >= burn_in:
            out["k"].append(k)
            out["b"].append(b)
            out["z"].append(z)
            out["k_next"].append(k_next)
            out["b_next"].append(b_next)
            out["z_next"].append(z_next)
            out["I"].append(I)
            out["q"].append(q_clip)
            out["spread"].append(spread)
            out["r_tilde"].append(r_tilde)
            out["e"].append(e)
            out["d"].append(d)
            out["default"].append(default)
            out["recovery"].append(recovery)
            out["continuation_next"].append(cont_next)
            out["continuation_indicator_next"].append(cont_ind)

        k, b, z = k_next, b_next, z_next

    dataset: Dict[str, np.ndarray] = {}
    for key, seq in out.items():
        if not seq:
            dataset[key] = np.zeros((0,), dtype=float)
            continue
        mat = tf.stack(seq, axis=1)
        dataset[key] = tf.reshape(mat, (-1,)).numpy().astype(float)

    t_eff = max(0, T - burn_in)
    dataset["n_paths"] = np.asarray(n_paths, dtype=np.int32)
    dataset["T_eff"] = np.asarray(t_eff, dtype=np.int32)
    dataset["burn_in"] = np.asarray(burn_in, dtype=np.int32)
    dataset["sim_T"] = np.asarray(T, dtype=np.int32)
    dataset["continuation_horizon"] = np.asarray(continuation_horizon, dtype=np.int32)
    dataset["eps_full"] = eps.astype(np.float32)
    dataset["dgp_source"] = np.asarray("obj1_nn", dtype='<U16')
    return dataset


class _SMMEvaluator:
    """Cached evaluator for candidate SMM moments."""

    def __init__(
        self,
        mp_true: ModelParams,
        tp_base: TrainParams,
        eps: np.ndarray,
        sim_T: int,
        sim_burn: int,
        continuation_horizon: int,
        param_names: Sequence[str],
        inner: ReusableInnerObjective1Solver,
        seed: int,
    ):
        """Initialize _SMMEvaluator."""
        self.mp_true = mp_true
        self.tp_base = tp_base
        self.eps = np.asarray(eps, dtype=np.float32)
        self.sim_T = int(sim_T)
        self.sim_burn = int(sim_burn)
        self.continuation_horizon = int(continuation_horizon)
        self.param_names = list(param_names)
        self.inner = inner
        self.seed = int(seed)
        self.op1 = Obj1Params()
        self.cache: Dict[str, Dict[str, object]] = {}

    def _key(self, x: Sequence[float]) -> str:
        """Return the cache key for the current parameter vector."""
        return ",".join(f"{float(v):.10g}" for v in x)

    def evaluate(self, x: Sequence[float]) -> Dict[str, object]:
        """Solve the inner model and cache the simulated SMM moments."""
        key = self._key(x)
        if key in self.cache:
            return self.cache[key]

        mp_cand = apply_params(self.mp_true, self.param_names, x)
        set_global_seed(self.seed + 1)
        policy_c, qnet_c = self.inner.solve(mp_cand, self.op1)
        snapshot = self.inner.snapshot_weights()
        set_global_seed(self.seed + 2)
        sim = forward_simulate_dataset(
            policy=policy_c,
            qnet=qnet_c,
            mp=mp_cand,
            tp=self.tp_base,
            eps=self.eps,
            T=self.sim_T,
            burn_in=self.sim_burn,
            continuation_horizon=self.continuation_horizon,
        )
        moment_names, moments, _, _, _ = compute_smm_moments(sim, mp_cand)
        out = {
            "mp": mp_cand,
            "network_snapshot": snapshot,
            "dataset": sim,
            "moment_names": moment_names,
            "moments": moments,
        }
        self.cache[key] = out
        return out


def _moment_table(
    names: Sequence[str],
    observed: np.ndarray,
    simulated: np.ndarray,
    cov: np.ndarray,
) -> List[Dict[str, float]]:
    """Build a tabular comparison of observed and simulated SMM moments."""
    observed = np.asarray(observed, dtype=float)
    simulated = np.asarray(simulated, dtype=float)
    std = np.sqrt(np.maximum(np.diag(np.asarray(cov, dtype=float)), 1e-12))
    rows: List[Dict[str, float]] = []
    for i, name in enumerate(names):
        raw = float(simulated[i] - observed[i])
        rows.append(
            {
                "moment": str(name),
                "observed": float(observed[i]),
                "simulated": float(simulated[i]),
                "raw_error": raw,
                "percent_error": float(raw / max(abs(observed[i]), 1e-8)),
                "standardized_error": float(raw / max(std[i], 1e-8)),
            }
        )
    return rows


def _compute_reportable(base_mean: np.ndarray) -> np.ndarray:
    """Compute eportable."""
    base_mean = np.asarray(base_mean, dtype=float)
    return np.asarray(
        [
            base_mean[0],
            np.sqrt(max(base_mean[1] - base_mean[0] * base_mean[0], 1e-12)),
            (base_mean[2] - base_mean[0] * base_mean[0])
            / max(base_mean[1] - base_mean[0] * base_mean[0], 1e-12),
            base_mean[3],
            np.sqrt(max(base_mean[4] - base_mean[3] * base_mean[3], 1e-12)),
            (base_mean[5] - base_mean[3] * base_mean[0])
            / max(
                np.sqrt(max(base_mean[4] - base_mean[3] * base_mean[3], 1e-12))
                * np.sqrt(max(base_mean[1] - base_mean[0] * base_mean[0], 1e-12)),
                1e-12,
            ),
            base_mean[6],
            np.sqrt(max(base_mean[7] - base_mean[6] * base_mean[6], 1e-12)),
            (base_mean[8] - base_mean[6] * base_mean[6])
            / max(base_mean[7] - base_mean[6] * base_mean[6], 1e-12),
            base_mean[9],
            np.clip(base_mean[10], 0.0, 1.0),
            base_mean[11] / max(base_mean[10], 1e-12),
        ],
        dtype=float,
    )


def _optimize_from_start(
    objective_x: Callable[[np.ndarray], float],
    x_start: np.ndarray,
    bounds: Sequence[Tuple[float, float]],
    max_evals: int,
    progress_reporter: Optional[EstimationProgressReporter] = None,
    phase_name: str = "optimization",
    start_id: int = 0,
) -> Dict[str, object]:
    """Run a transformed-parameter local optimization from one start."""
    u_bounds = [(-8.0, 8.0)] * len(bounds)
    u0 = unconstrained_from_bounded(x_start, bounds)
    u_step = np.full_like(u0, 0.8, dtype=float)

    eval_counter = {"count": 0}

    def objective_u(u: np.ndarray) -> float:
        """Return the moment-gap vector used in the SMM criterion."""
        x = bounded_from_unconstrained(u, bounds)
        eval_counter["count"] += 1
        eval_id = int(eval_counter["count"])
        if progress_reporter is not None:
            progress_reporter.local_eval_start(phase_name, start_id, eval_id, np.round(x, 6).tolist())
        val = float(objective_x(x))
        if not np.isfinite(val):
            val = 1.0e30
        if progress_reporter is not None:
            progress_reporter.local_eval_done(phase_name, start_id, eval_id, float(val))
        return val

    nm = _nelder_mead(
        f=objective_u,
        x0=u0,
        step=u_step,
        bounds=u_bounds,
        max_evals=max_evals,
    )
    x_hat = bounded_from_unconstrained(np.asarray(nm["x"], dtype=float), bounds)
    return {
        "theta_hat_vector": x_hat.astype(float),
        "objective": float(nm["objective"]),
        "evals": int(nm["evals"]),
        "converged": bool(nm["converged"]),
        "simplex_diameter": float(nm["simplex_diameter"]),
        "f_spread": float(nm["f_spread"]),
        "success": bool(np.isfinite(nm["objective"])),
    }


def _run_multistart(
    objective_x: Callable[[np.ndarray], float],
    starts: Sequence[np.ndarray],
    bounds: Sequence[Tuple[float, float]],
    max_evals: int,
    progress_reporter: Optional[EstimationProgressReporter] = None,
    phase_name: str = "optimization",
) -> List[Dict[str, object]]:
    """Run ultistart."""
    runs: List[Dict[str, object]] = []
    if progress_reporter is not None:
        progress_reporter.start_multistart(phase_name, n_starts=len(starts), max_evals=max_evals)
    for i, x0 in enumerate(starts):
        if progress_reporter is not None:
            progress_reporter.start_local_run(phase_name, int(i), np.round(np.asarray(x0, dtype=float), 6).tolist())
        try:
            run = _optimize_from_start(
                objective_x,
                np.asarray(x0, dtype=float),
                bounds,
                max_evals,
                progress_reporter=progress_reporter,
                phase_name=phase_name,
                start_id=int(i),
            )
            run["start_id"] = int(i)
            run["start_theta"] = np.asarray(x0, dtype=float).tolist()
        except Exception as exc:
            import traceback

            print(f"[{phase_name}] start_id={int(i)} FAILED with exception: {exc!r}")
            traceback.print_exc()
            run = {
                "start_id": int(i),
                "start_theta": np.asarray(x0, dtype=float).tolist(),
                "theta_hat_vector": np.asarray(x0, dtype=float).tolist(),
                "objective": float("inf"),
                "evals": 0,
                "converged": False,
                "simplex_diameter": None,
                "f_spread": None,
                "success": False,
                "error": str(exc),
            }
        runs.append(run)
        if progress_reporter is not None:
            progress_reporter.finish_local_run(
                phase_name,
                int(i),
                success=bool(run.get("success", False)),
                objective=float(run.get("objective", float("inf"))),
                evals=int(run.get("evals", 0)),
            )
    return runs


def _finalize_smm_variant(
    label: str,
    evaluator: _SMMEvaluator,
    runs: Sequence[Dict[str, object]],
    m_obs: np.ndarray,
    omega_data: np.ndarray,
    sim_ratio_J: float,
    param_names: Sequence[str],
    mp_true: ModelParams,
    observed_stats: Dict[str, float],
    n_obs_moments: int,
) -> Dict[str, object]:
    """Finalize reporting fields for one SMM variant."""
    successful = [r for r in runs if bool(r.get("success", False)) and np.isfinite(float(r["objective"]))]
    if not successful:
        return {
            "label": label,
            "success": False,
            "starts": list(runs),
            "multistart_summary": summarize_multistart_runs(runs, param_names),
        }

    best = min(successful, key=lambda d: float(d["objective"]))
    x_hat = np.asarray(best["theta_hat_vector"], dtype=float)
    eval_hat = evaluator.evaluate(x_hat)
    mp_hat = eval_hat["mp"]
    m_hat = np.asarray(eval_hat["moments"], dtype=float)
    dataset_hat = eval_hat["dataset"]

    w, cond = stabilized_inverse(omega_data)

    def moment_map(x: np.ndarray) -> np.ndarray:
        """Map one simulated dataset into the reportable SMM moment vector."""
        return np.asarray(evaluator.evaluate(x)["moments"], dtype=float)

    g_num = numerical_jacobian(moment_map, np.asarray(x_hat, dtype=float))
    a = g_num.T @ w @ g_num
    a_inv = np.linalg.pinv(a + 1e-8 * np.eye(a.shape[0]))
    vcov = (1.0 + 1.0 / max(sim_ratio_J, 1e-8)) * (a_inv @ (g_num.T @ w @ omega_data @ w @ g_num) @ a_inv)
    vcov = 0.5 * (vcov + vcov.T)
    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))

    param_table = summarize_param_errors(mp_hat, mp_true, param_names)
    for i, name in enumerate(param_names):
        param_table[name]["std_error"] = float(se[i])

    simulated_stats = dataset_pricing_default_summary(dataset_hat, mp_hat)
    g_gap = np.asarray(m_obs - m_hat, dtype=float)
    spec_stat = float(max(1, n_obs_moments) * (g_gap @ w @ g_gap))
    spec_dof = int(max(0, len(m_obs) - len(param_names)))

    return {
        "label": label,
        "success": True,
        "theta_hat": {name: float(getattr(mp_hat, name)) for name in param_names},
        "theta_hat_vector": x_hat.tolist(),
        "objective": float(best["objective"]),
        "specification_stat": spec_stat,
        "specification_dof": spec_dof,
        "best_start_id": int(best["start_id"]),
        "best_start_theta": list(best["start_theta"]),
        "convergence_flag": bool(best.get("converged", False)),
        "evals": int(best.get("evals", 0)),
        "weight_matrix_condition": float(cond),
        "parameter_table": param_table,
        "recovery_score": overall_param_recovery_score(mp_hat, mp_true, param_names),
        "moment_table": _moment_table(eval_hat["moment_names"], m_obs, m_hat, omega_data),
        "pricing_default_fit": {
            "observed": observed_stats,
            "simulated": simulated_stats,
            "errors": {
                k: float(simulated_stats[k] - observed_stats[k]) for k in observed_stats.keys()
            },
        },
        "starts": list(runs),
        "multistart_summary": summarize_multistart_runs(runs, param_names),
    }



def _build_smm_identification_report(
    evaluator: _SMMEvaluator,
    mp_true: ModelParams,
    param_names: Sequence[str],
    bounds: Sequence[Tuple[float, float]],
    m_obs: np.ndarray,
    n_grid: int = 5,
) -> Dict[str, object]:
    """Build mm identification report."""
    base_x = np.asarray([float(getattr(mp_true, n)) for n in param_names], dtype=float)
    sweeps: Dict[str, object] = {}
    for i, name in enumerate(param_names):
        lo, hi = bounds[i]
        grid = np.linspace(lo, hi, n_grid, dtype=float)
        objectives: List[float] = []
        mean_i: List[float] = []
        mean_pi: List[float] = []
        mean_lev: List[float] = []
        mean_spread: List[float] = []
        default_rate: List[float] = []
        for val in grid:
            x = base_x.copy()
            x[i] = float(val)
            eval_x = evaluator.evaluate(x)
            m_sim = np.asarray(eval_x["moments"], dtype=float)
            gap = m_obs - m_sim
            objectives.append(float(gap @ gap))
            mean_i.append(float(m_sim[0]))
            mean_pi.append(float(m_sim[3]))
            mean_lev.append(float(m_sim[6]))
            mean_spread.append(float(m_sim[9]))
            default_rate.append(float(m_sim[10]))
        sweeps[name] = {
            "grid": grid.tolist(),
            "objective": objectives,
            "curves": {
                "mean_I_over_k": mean_i,
                "mean_pi_over_k": mean_pi,
                "mean_b_over_k": mean_lev,
                "mean_spread": mean_spread,
                "default_rate": default_rate,
            },
        }
    return {
        "parameter_names": list(param_names),
        "true_theta": {name: float(getattr(mp_true, name)) for name in param_names},
        "sweeps": sweeps,
    }


def estimate_smm(
    out_dir: str,
    mp_true: ModelParams,
    npol: NetParams,
    nq: NetParams,
    tp_base: TrainParams,
    policy_true: Optional[PolicyNet],
    qnet_true: Optional[PricingNet],
    benchmark: Optional[Dict[str, np.ndarray]] = None,
    est_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    max_evals: int = 60,
    inner_epochs: int = 3,
    inner_steps_per_epoch: int = 20,
    sim_T: int = 200,
    sim_burn: int = 50,
    sim_n_paths: int = 1,
    seed: int = 1234,
    n_starts: int = 3,
    continuation_horizon: int = 0,
    progress_reporter: Optional[EstimationProgressReporter] = None,
) -> Dict[str, object]:
    """Run plan-aligned two-step SMM with A/B covariance variants."""
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    reporter = progress_reporter
    if reporter is not None:
        reporter.start_phase("total_estimation", out_dir=out_dir)

    if est_bounds is None:
        est_bounds = default_estimation_bounds()
    else:
        est_bounds = {k: v for k, v in est_bounds.items() if k in default_estimation_bounds()}
        if not est_bounds:
            est_bounds = default_estimation_bounds()

    param_names = ordered_param_names(est_bounds)
    bounds = bounds_in_order(est_bounds, param_names)

    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, mp_true.sigma_eps, size=(sim_n_paths, sim_T + 1)).astype(np.float32)

    if reporter is not None:
        reporter.start_phase("synthetic_data_generation", sim_T=sim_T, sim_burn=sim_burn, sim_n_paths=sim_n_paths)
    set_global_seed(seed)
    data = forward_simulate_dataset(
        policy=policy_true,
        qnet=qnet_true,
        mp=mp_true,
        tp=tp_base,
        eps=eps,
        T=sim_T,
        burn_in=sim_burn,
        continuation_horizon=continuation_horizon,
        benchmark=benchmark,
    )
    np.savez_compressed(os.path.join(out_dir, "smm_synth_data.npz"), **data)
    if reporter is not None:
        reporter.finish_phase("synthetic_data_generation", n_obs=int(np.asarray(data["k"]).reshape(-1).size))

    if reporter is not None:
        reporter.start_phase("observed_moment_construction")
    moment_names, m_obs, _, base_mean, base_features = compute_smm_moments(data, mp_true)
    if reporter is not None:
        reporter.finish_phase("observed_moment_construction", n_obs=int(base_features.shape[0]), n_moments=int(len(moment_names)))
    jac_m = numerical_jacobian(lambda u: np.asarray(_compute_reportable(u), dtype=float), np.asarray(base_mean, dtype=float))

    n_obs = base_features.shape[0]
    nw_lags = newey_west_lag_length(n_obs)
    omega_base_std = covariance_of_mean(base_features, hac_lags=0)
    omega_base_nw = covariance_of_mean(base_features, hac_lags=nw_lags)
    omega_std = jac_m @ omega_base_std @ jac_m.T
    omega_nw = jac_m @ omega_base_nw @ jac_m.T

    tp_inner = replace(tp_base, epochs=inner_epochs, steps_per_epoch=inner_steps_per_epoch)
    inner = ReusableInnerObjective1Solver(mp_true, npol, nq, tp_inner, seed=seed + 100)
    evaluator = _SMMEvaluator(
        mp_true=mp_true,
        tp_base=tp_base,
        eps=eps,
        sim_T=sim_T,
        sim_burn=sim_burn,
        continuation_horizon=continuation_horizon,
        param_names=param_names,
        inner=inner,
        seed=seed + 200,
    )

    stage1_cache: Dict[str, float] = {}

    def obj_stage1(x: np.ndarray) -> float:
        """Evaluate the first-stage identity-weight SMM objective."""
        key = ",".join(f"{float(v):.10g}" for v in x)
        if key in stage1_cache:
            return stage1_cache[key]
        m_sim = np.asarray(evaluator.evaluate(x)["moments"], dtype=float)
        g = m_obs - m_sim
        val = float(g @ g)
        stage1_cache[key] = val
        return val

    starts = generate_parameter_starts(bounds=bounds, n_starts=n_starts, seed=seed + 300)
    if reporter is not None:
        reporter.start_phase("stage1_identity_weight")
    stage1_runs = _run_multistart(
        obj_stage1,
        starts,
        bounds,
        max_evals,
        progress_reporter=reporter,
        phase_name="stage1_identity_weight",
    )
    successful_stage1 = [r for r in stage1_runs if bool(r.get("success", False)) and np.isfinite(float(r["objective"]))]
    if reporter is not None:
        reporter.finish_phase(
            "stage1_identity_weight",
            successful_starts=len(successful_stage1),
            best_objective=float(min([r["objective"] for r in successful_stage1], default=float("inf"))),
        )
    if not successful_stage1:
        raise RuntimeError("All SMM stage-1 starts failed.")

    n_data = max(1, base_features.shape[0])
    n_sim = max(1, int(np.asarray(data["k"]).size))
    sim_ratio_J = float(n_sim) / float(n_data)
    observed_stats = dataset_pricing_default_summary(data, mp_true)

    def make_stage2_runs(omega_data: np.ndarray, phase_name: str) -> List[Dict[str, object]]:
        """Create and execute the second-stage multistart SMM runs."""
        w, _ = stabilized_inverse(omega_data)
        stage2_cache: Dict[str, float] = {}

        def obj_stage2(x: np.ndarray) -> float:
            """Evaluate one second-stage weighted SMM objective."""
            key = ",".join(f"{float(v):.10g}" for v in x)
            if key in stage2_cache:
                return stage2_cache[key]
            m_sim = np.asarray(evaluator.evaluate(x)["moments"], dtype=float)
            g = m_obs - m_sim
            val = float(g @ w @ g)
            stage2_cache[key] = val
            return val

        stage1_endpoints = [np.asarray(r["theta_hat_vector"], dtype=float) for r in successful_stage1]
        return _run_multistart(
            obj_stage2,
            stage1_endpoints,
            bounds,
            max_evals,
            progress_reporter=reporter,
            phase_name=phase_name,
        )

    if reporter is not None:
        reporter.start_phase("stage2_smm_a")
    variant_a_runs = make_stage2_runs(omega_std, "stage2_smm_a")
    if reporter is not None:
        reporter.finish_phase("stage2_smm_a", successful_starts=sum(bool(r.get("success", False)) for r in variant_a_runs))
    if reporter is not None:
        reporter.start_phase("stage2_smm_b")
    variant_b_runs = make_stage2_runs(omega_nw, "stage2_smm_b")
    if reporter is not None:
        reporter.finish_phase("stage2_smm_b", successful_starts=sum(bool(r.get("success", False)) for r in variant_b_runs))

    variant_a = _finalize_smm_variant(
        label="SMM-A",
        evaluator=evaluator,
        runs=variant_a_runs,
        m_obs=m_obs,
        omega_data=omega_std,
        sim_ratio_J=sim_ratio_J,
        param_names=param_names,
        mp_true=mp_true,
        observed_stats=observed_stats,
        n_obs_moments=n_data,
    )
    variant_b = _finalize_smm_variant(
        label="SMM-B",
        evaluator=evaluator,
        runs=variant_b_runs,
        m_obs=m_obs,
        omega_data=omega_nw,
        sim_ratio_J=sim_ratio_J,
        param_names=param_names,
        mp_true=mp_true,
        observed_stats=observed_stats,
        n_obs_moments=n_data,
    )

    successful_variants = [v for v in [variant_a, variant_b] if bool(v.get("success", False))]
    best_variant = min(successful_variants, key=lambda d: float(d["objective"])) if successful_variants else variant_a

    identification = _build_smm_identification_report(
        evaluator=evaluator,
        mp_true=mp_true,
        param_names=param_names,
        bounds=bounds,
        m_obs=m_obs,
        n_grid=5,
    )
    save_identification_report(out_dir, "smm", identification)

    res: Dict[str, object] = {
        "method": "SMM",
        "baseline_parameters": param_names,
        "theta_true": {name: float(getattr(mp_true, name)) for name in param_names},
        "best_variant": str(best_variant.get("label", "")),
        "theta_hat": best_variant.get("theta_hat", {}),
        "objective": float(best_variant.get("objective", np.inf)),
        "moment_names": moment_names,
        "m_data": m_obs.tolist(),
        "newey_west_lags": int(nw_lags),
        "sim_ratio_J": float(sim_ratio_J),
        "continuation_horizon": int(continuation_horizon),
        "dgp_source": str(np.asarray(data.get("dgp_source", "obj1_nn")).reshape(())),
        "runtime_sec": float(time.time() - t0),
        "identification_report": "smm_identification.json",
        "stage1": {
            "runs": stage1_runs,
            "summary": summarize_multistart_runs(stage1_runs, param_names),
        },
        "variants": {
            "SMM-A": variant_a,
            "SMM-B": variant_b,
        },
    }

    with open(os.path.join(out_dir, "smm_results.json"), "w", encoding="utf-8") as f_out:
        json.dump(make_json_serializable(res), f_out, indent=2)

    if reporter is not None:
        reporter.finish_phase(
            "total_estimation",
            runtime_sec=float(time.time() - t0),
            best_variant=str(res.get("best_variant", "SMM-A")),
            objective=float(res.get("objective", np.inf)),
        )
    return res

"""Generalized Method of Moments (GMM) for the risky-debt model.

This implementation is materially closer to the user's final plan:
- baseline structural parameters: (theta, psi0, alpha)
- hard continuation based on a policy-evaluated finite-horizon continuation value
- instruments Z_t = [1, log k_t, log z_t, b_t/k_t, I_t/k_t]
- derivative-mode GMM blocks:
    * k'-FOC residual
    * b'-FOC residual
    * lender zero-profit residual
- optional pricing-only GMM block for long Colab runs
- two-step GMM with standard and Newey--West covariance variants
- multi-start local optimization with transformed parameters and robustness reports
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import json
import os
import time

import numpy as np
try:
    from scipy.stats import chi2 as _chi2_dist
except Exception:  # pragma: no cover - scipy may be absent in minimal installs
    _chi2_dist = None
import tensorflow as tf

from risky_debt.config import ModelParams, NetParams, TrainParams
from risky_debt.primitives import beta_tensor_from_r, equity_payout_d_total, recovery_R
from risky_debt.objectives import _price_q

from .common import (
    apply_params,
    bounded_from_unconstrained,
    bounds_in_order,
    covariance_of_mean,
    dataset_pricing_default_summary,
    default_estimation_bounds,
    generate_parameter_starts,
    newey_west_lag_length,
    numerical_jacobian,
    jacobian_singular_value_report,
    ordered_param_names,
    overall_param_recovery_score,
    stabilized_inverse,
    summarize_multistart_runs,
    summarize_param_errors,
    unconstrained_from_bounded,
    save_identification_report,
    make_json_serializable,
)
from .smm import _nelder_mead, _checkpoint_eval_count
from .inner_obj1 import ReusableInnerObjective1Solver, finite_horizon_predefault_continuation
from .progress import EstimationProgressReporter


def _zero_if_none(grad: tf.Tensor | None, like: tf.Tensor) -> tf.Tensor:
    """Return a zero tensor when GradientTape reports a disconnected gradient."""
    return tf.zeros_like(like) if grad is None else grad


class _GMMDataset:
    """Panel-aligned tensors for the GMM residual system."""

    def __init__(self, data: Dict[str, np.ndarray], mp: ModelParams, continuation_horizon: int):
        """Initialize _GMMDataset."""
        self.mp = mp
        self.continuation_horizon = int(max(0, continuation_horizon))
        n_paths = int(np.asarray(data.get("n_paths", 1)).reshape(()))
        t_eff = int(np.asarray(data.get("T_eff", np.asarray(data["k"]).size)).reshape(()))
        if n_paths * t_eff != np.asarray(data["k"]).size:
            n_paths, t_eff = 1, int(np.asarray(data["k"]).size)

        def panel(key: str) -> np.ndarray:
            """Reshape one flattened synthetic-data series into panel form."""
            arr = np.asarray(data[key], dtype=float).reshape(-1)
            return arr.reshape(n_paths, t_eff)

        k = panel("k")
        b = panel("b")
        z = panel("z")
        k_next_obs = panel("k_next")
        b_next_obs = panel("b_next")
        z_next = panel("z_next")
        I = panel("I")

        if t_eff < 2:
            raise ValueError("Synthetic dataset is too short for GMM (needs t+1 information).")

        burn_in = int(np.asarray(data.get("burn_in", 0)).reshape(()))
        eps_full = np.asarray(data.get("eps_full"), dtype=np.float32)
        if eps_full.ndim != 2:
            raise ValueError("Synthetic dataset must store eps_full for hard-continuation GMM.")

        n_obs_t = t_eff - 1
        if self.continuation_horizon <= 0:
            max_horizon = max(0, eps_full.shape[1] - (burn_in + 1))
        else:
            max_horizon = self.continuation_horizon
        eps_windows = np.zeros((n_paths, n_obs_t, max_horizon), dtype=np.float32)
        eps_masks = np.zeros((n_paths, n_obs_t, max_horizon), dtype=np.float32)
        for p in range(n_paths):
            for t in range(n_obs_t):
                start = burn_in + t + 1
                if self.continuation_horizon <= 0:
                    stop = eps_full.shape[1]
                else:
                    stop = min(start + self.continuation_horizon, eps_full.shape[1])
                width = max(0, stop - start)
                if width > 0:
                    eps_windows[p, t, :width] = eps_full[p, start:stop]
                    eps_masks[p, t, :width] = 1.0

        k_t = k[:, :-1].reshape(-1)
        b_t = b[:, :-1].reshape(-1)
        z_t = z[:, :-1].reshape(-1)
        k1 = k_next_obs[:, :-1].reshape(-1)
        b1 = b_next_obs[:, :-1].reshape(-1)
        z1 = z_next[:, :-1].reshape(-1)
        I_t = I[:, :-1].reshape(-1)
        eps_tp1 = eps_windows.reshape(-1, max_horizon)
        eps_tp1_mask = eps_masks.reshape(-1, max_horizon)

        instr = np.column_stack(
            [
                np.ones_like(k_t),
                np.log(np.maximum(k_t, mp.k_min)),
                np.log(np.maximum(z_t, mp.z_min)),
                b_t / np.maximum(k_t, mp.k_min),
                I_t / np.maximum(k_t, mp.k_min),
            ]
        ).astype(np.float32)

        self.k_t = tf.convert_to_tensor(k_t.astype(np.float32))
        self.b_t = tf.convert_to_tensor(b_t.astype(np.float32))
        self.z_t = tf.convert_to_tensor(z_t.astype(np.float32))
        self.k1 = tf.convert_to_tensor(k1.astype(np.float32))
        self.b1 = tf.convert_to_tensor(b1.astype(np.float32))
        self.z1 = tf.convert_to_tensor(z1.astype(np.float32))
        self.Z = tf.convert_to_tensor(instr)
        self.eps_tp1 = tf.convert_to_tensor(eps_tp1.astype(np.float32))
        self.eps_tp1_mask = tf.convert_to_tensor(eps_tp1_mask.astype(np.float32))
        self.n_obs = int(instr.shape[0])
        self.L = int(instr.shape[1])


class _GMMEvaluator:
    """Cached evaluator for candidate GMM residuals and moment vectors."""

    def __init__(
        self,
        mp_true: ModelParams,
        npol: NetParams,
        nq: NetParams,
        tp_base: TrainParams,
        data: Dict[str, np.ndarray],
        inner_epochs: int,
        inner_steps_per_epoch: int,
        seed: int,
        param_names: Sequence[str],
        continuation_horizon: int,
        moment_mode: str = "derivative",
        initial_policy_weights: Optional[Sequence[np.ndarray]] = None,
        initial_q_weights: Optional[Sequence[np.ndarray]] = None,
    ):
        """Initialize _GMMEvaluator.

        ``moment_mode='derivative'`` is the written-summary GMM system with
        capital/debt FOC residuals plus zero-profit residuals.
        ``moment_mode='pricing_only'`` is an explicitly optional computational
        shortcut for long Colab runs; it keeps only the lender zero-profit
        residual moments and avoids the expensive GradientTape-based FOC block.
        """
        self.mp_true = mp_true
        self.tp_base = tp_base
        self.param_names = list(param_names)
        self.dataset = _GMMDataset(data, mp_true, continuation_horizon=continuation_horizon)
        self.moment_mode = str(moment_mode).lower().strip()
        if self.moment_mode not in {"derivative", "pricing_only"}:
            raise ValueError("moment_mode must be 'derivative' or 'pricing_only'.")
        tp_inner = TrainParams(
            **{**tp_base.__dict__, "epochs": inner_epochs, "steps_per_epoch": inner_steps_per_epoch}
        )
        self.inner = ReusableInnerObjective1Solver(
            mp_true,
            npol,
            nq,
            tp_inner,
            seed=seed + 100,
            initial_policy_weights=initial_policy_weights,
            initial_q_weights=initial_q_weights,
        )
        self.seed = int(seed)
        self.cache: Dict[str, Dict[str, object]] = {}

    def _key(self, x: Sequence[float]) -> str:
        """Return the cache key for the current parameter vector."""
        return ",".join(f"{float(v):.10g}" for v in x)

    @staticmethod
    def _obj1_params():
        """Return training parameters used by the inner Objective 1 solve."""
        from risky_debt.config import Obj1Params
        return Obj1Params()

    def evaluate(self, x: Sequence[float]) -> Dict[str, object]:
        """Solve the inner model and cache the resulting GMM moment objects."""
        key = self._key(x)
        if key in self.cache:
            return self.cache[key]

        mp_cand = apply_params(self.mp_true, self.param_names, x)
        policy_c, qnet_c = self.inner.solve(mp_cand, op1=self._obj1_params())
        snapshot = self.inner.snapshot_weights()
        m_series = self._moment_contributions(policy_c, qnet_c, mp_cand)
        g = np.mean(m_series, axis=0)
        out = {
            "mp": mp_cand,
            "cache_key": key,
            "network_snapshot": snapshot,
            "m_series": m_series,
            "g": g,
            "computational_controls": self.inner.computational_controls(),
        }
        self.cache[key] = out
        return out

    def _moment_contributions(self, policy, qnet, mp: ModelParams) -> np.ndarray:
        """Compute stacked GMM moment contributions for one candidate model."""
        ds = self.dataset
        beta = beta_tensor_from_r(mp.r)

        if self.moment_mode == "pricing_only":
            return self._pricing_only_moment_contributions(policy, qnet, mp)

        with tf.GradientTape(persistent=True) as tape_choice:
            tape_choice.watch(ds.k1)
            tape_choice.watch(ds.b1)

            q_t = _price_q(qnet, ds.z_t, ds.k1, ds.b1, mp, self.tp_base)

            with tf.GradientTape(persistent=True) as tape_value:
                tape_value.watch(ds.k1)
                tape_value.watch(ds.b1)
                cont_tp1 = finite_horizon_predefault_continuation(
                    policy=policy,
                    qnet=qnet,
                    mp=mp,
                    tp=self.tp_base,
                    k0=ds.k1,
                    b0=ds.b1,
                    z0=ds.z1,
                    eps_future=ds.eps_tp1,
                    valid_mask=ds.eps_tp1_mask,
                )
                value_tp1 = tf.maximum(cont_tp1, 0.0)
            v_k = _zero_if_none(tape_value.gradient(tf.reduce_sum(value_tp1), ds.k1), ds.k1)
            v_b = _zero_if_none(tape_value.gradient(tf.reduce_sum(value_tp1), ds.b1), ds.b1)
            del tape_value

            hard1 = tf.stop_gradient(tf.cast(cont_tp1 > 0.0, tf.float32))
            d_t = equity_payout_d_total(
                k=ds.k_t,
                k_next=ds.k1,
                b=ds.b_t,
                b_next=ds.b1,
                z=ds.z_t,
                q=q_t,
                continuation_weight=hard1,
                mp=mp,
                kappa_issue=self.tp_base.kappa_issue,
            )
            d_sum = tf.reduce_sum(d_t)

        d_k = _zero_if_none(tape_choice.gradient(d_sum, ds.k1), ds.k1)
        d_b = _zero_if_none(tape_choice.gradient(d_sum, ds.b1), ds.b1)
        del tape_choice

        u_k = hard1 * (d_k + beta * v_k)
        u_b = hard1 * (d_b + beta * v_b)
        repay = hard1 * (ds.b1 / tf.clip_by_value(q_t, mp.q_min, mp.q_max))
        recover = (1.0 - hard1) * recovery_R(ds.k1, ds.z1, mp)
        u_q = (1.0 + mp.r) * ds.b1 - (recover + repay)

        m_k = ds.Z * tf.expand_dims(u_k, 1)
        m_b = ds.Z * tf.expand_dims(u_b, 1)
        m_q = ds.Z * tf.expand_dims(u_q, 1)
        m = tf.concat([m_k, m_b, m_q], axis=1)
        return m.numpy().astype(float)

    def _pricing_only_moment_contributions(self, policy, qnet, mp: ModelParams) -> np.ndarray:
        """Compute GMM moments that use only lender zero-profit residuals.

        This optional mode is designed for long Colab runs where the
        derivative-based FOC moment system is too costly.  It remains a genuine
        GMM criterion because it forms instrumented sample moments, but it is a
        narrower moment set than the written-summary derivative mode.
        """
        ds = self.dataset
        q_t = _price_q(qnet, ds.z_t, ds.k1, ds.b1, mp, self.tp_base)
        cont_tp1 = finite_horizon_predefault_continuation(
            policy=policy,
            qnet=qnet,
            mp=mp,
            tp=self.tp_base,
            k0=ds.k1,
            b0=ds.b1,
            z0=ds.z1,
            eps_future=ds.eps_tp1,
            valid_mask=ds.eps_tp1_mask,
        )
        hard1 = tf.stop_gradient(tf.cast(cont_tp1 > 0.0, tf.float32))
        repay = hard1 * (ds.b1 / tf.clip_by_value(q_t, mp.q_min, mp.q_max))
        recover = (1.0 - hard1) * recovery_R(ds.k1, ds.z1, mp)
        u_q = (1.0 + mp.r) * ds.b1 - (recover + repay)
        m_q = ds.Z * tf.expand_dims(u_q, 1)
        return m_q.numpy().astype(float)


def _optimize_from_start(
    objective_x,
    x_start: np.ndarray,
    bounds: Sequence[Tuple[float, float]],
    max_evals: int,
    progress_reporter: Optional[EstimationProgressReporter] = None,
    phase_name: str = "optimization",
    start_id: int = 0,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, object]:
    """Run one local optimization from a single starting vector."""
    u_bounds = [(-8.0, 8.0)] * len(bounds)
    u0 = unconstrained_from_bounded(x_start, bounds)
    u_step = np.full_like(u0, 0.8, dtype=float)

    eval_counter = {"count": _checkpoint_eval_count(checkpoint_path)}

    def objective_u(u: np.ndarray) -> float:
        """Return the stacked residual blocks used in the GMM moments."""
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

    # Smoke tests and legacy monkeypatches may replace `_nelder_mead` with a
    # small stub that predates evaluation-level checkpointing and therefore does
    # not accept `checkpoint_path`.  Keep the production resumable path while
    # remaining backward-compatible with those stubs.
    import inspect

    nm_kwargs = {
        "f": objective_u,
        "x0": u0,
        "step": u_step,
        "bounds": u_bounds,
        "max_evals": max_evals,
    }
    if "checkpoint_path" in inspect.signature(_nelder_mead).parameters:
        nm_kwargs["checkpoint_path"] = checkpoint_path
    nm = _nelder_mead(**nm_kwargs)
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
    objective_x,
    starts,
    bounds,
    max_evals,
    progress_reporter: Optional[EstimationProgressReporter] = None,
    phase_name: str = "optimization",
    resume_dir: Optional[str] = None,
) -> List[Dict[str, object]]:
    """Run multistart optimization with optional completed-start resume."""
    runs: List[Dict[str, object]] = []
    if resume_dir is not None:
        os.makedirs(resume_dir, exist_ok=True)
    if progress_reporter is not None:
        progress_reporter.start_multistart(phase_name, n_starts=len(starts), max_evals=max_evals)
    for i, x0 in enumerate(starts):
        start_path = os.path.join(resume_dir, f"start_{int(i):03d}.json") if resume_dir is not None else None
        if start_path is not None and os.path.exists(start_path):
            try:
                with open(start_path, "r", encoding="utf-8") as fh:
                    run = json.load(fh)
                run["resumed_from_completed_start"] = True
                runs.append(run)
                if progress_reporter is not None:
                    progress_reporter.finish_local_run(
                        phase_name,
                        int(i),
                        success=bool(run.get("success", False)),
                        objective=float(run.get("objective", float("inf"))),
                        evals=int(run.get("evals", 0)),
                    )
                print(f"[{phase_name}] start_id={int(i)} already completed; loaded from resume cache.", flush=True)
                continue
            except Exception:
                pass
        if progress_reporter is not None:
            progress_reporter.start_local_run(phase_name, int(i), np.round(np.asarray(x0, dtype=float), 6).tolist())
        try:
            inprogress_path = (
                os.path.join(resume_dir, f"start_{int(i):03d}_inprogress.json")
                if resume_dir is not None
                else None
            )
            run = _optimize_from_start(
                objective_x,
                np.asarray(x0, dtype=float),
                bounds,
                max_evals,
                progress_reporter=progress_reporter,
                phase_name=phase_name,
                start_id=int(i),
                checkpoint_path=inprogress_path,
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
        if start_path is not None:
            with open(start_path, "w", encoding="utf-8") as fh:
                json.dump(make_json_serializable(run), fh, indent=2)
            inprogress_path = os.path.join(resume_dir, f"start_{int(i):03d}_inprogress.json") if resume_dir is not None else None
            if inprogress_path is not None and os.path.exists(inprogress_path) and bool(run.get("success", False)):
                try:
                    os.remove(inprogress_path)
                except OSError:
                    pass
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


def _finalize_gmm_variant(
    label: str,
    evaluator: _GMMEvaluator,
    runs: Sequence[Dict[str, object]],
    omega_mean: np.ndarray,
    param_names: Sequence[str],
    mp_true: ModelParams,
    observed_stats: Dict[str, float],
    report_mode: str = "full",
) -> Dict[str, object]:
    """Finalize reporting fields for one GMM variant."""
    light_report = str(report_mode).lower().strip() == "light"
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
    g_hat = np.asarray(eval_hat["g"], dtype=float)
    m_series = np.asarray(eval_hat["m_series"], dtype=float)

    w, cond = stabilized_inverse(omega_mean)

    param_table = summarize_param_errors(mp_hat, mp_true, param_names)
    if light_report:
        # Light reporting is designed for interrupted Colab runs: save core
        # GMM estimates immediately and skip the costly finite-difference
        # Jacobian/standard-error block.  Full mode keeps the written-summary
        # diagnostics.
        jacobian_diagnostics = {
            "skipped": True,
            "reason": "estimation_report_mode=light",
            "rank": None,
            "full_column_rank": None,
            "singular_values": [],
            "condition_number": float("nan"),
            "rank_tolerance": float("nan"),
        }
        for name in param_names:
            param_table[name]["std_error"] = float("nan")
    else:
        def g_map(x: np.ndarray) -> np.ndarray:
            """Map residual blocks into the full stacked sample-moment vector."""
            return np.asarray(evaluator.evaluate(x)["g"], dtype=float)

        d = numerical_jacobian(g_map, np.asarray(x_hat, dtype=float))
        # Report-ready local-identification diagnostics required by the written summary.
        # Rows are GMM moments and columns are estimated structural parameters.
        jacobian_diagnostics = jacobian_singular_value_report(
            d,
            param_names=param_names,
            moment_names=[f"gmm_moment_{i}" for i in range(int(g_hat.size))],
        )
        a = d.T @ w @ d
        a_inv = np.linalg.pinv(a + 1e-8 * np.eye(a.shape[0]))
        vcov = a_inv @ (d.T @ w @ omega_mean @ w @ d) @ a_inv
        vcov = 0.5 * (vcov + vcov.T)
        se = np.sqrt(np.maximum(np.diag(vcov), 0.0))
        for i, name in enumerate(param_names):
            param_table[name]["std_error"] = float(se[i])

    ds = evaluator.dataset
    if getattr(evaluator, "moment_mode", "derivative") == "pricing_only":
        zp_block = m_series
    else:
        zp_block = m_series[:, 2 * ds.L : 3 * ds.L]
    mean_zp_mom = np.mean(zp_block, axis=0) if zp_block.size else np.asarray([np.nan])

    if light_report:
        # Avoid the slow full-panel pricing/default diagnostics in light mode.
        # The core GMM estimates, objective values, J-statistics, and moment
        # norms are still saved.
        simulated_stats = {k: float("nan") for k in observed_stats.keys()}
    else:
        # Simulated fit objects for the final parameter estimate.
        # Production evaluations cache detached weight snapshots so reporting can
        # rebuild stable networks later. Smoke tests may monkeypatch ``evaluate`` to
        # return lightweight live networks instead. Support both shapes here so the
        # finalization path remains robust under reduced test fixtures.
        snapshot = eval_hat.get("network_snapshot")
        if snapshot is not None:
            policy_hat, qnet_hat = evaluator.inner.materialize_networks(mp_hat, snapshot)
        else:
            policy_hat = eval_hat.get("policy")
            qnet_hat = eval_hat.get("qnet")
            if policy_hat is None or qnet_hat is None:
                raise KeyError(
                    "GMM evaluation must provide either 'network_snapshot' or live "
                    "'policy' and 'qnet' objects for reporting."
                )
        q_obs = np.asarray(_price_q(qnet_hat, ds.z_t, ds.k1, ds.b1, mp_hat, evaluator.tp_base).numpy(), dtype=float)
        cont_obs = finite_horizon_predefault_continuation(
            policy=policy_hat,
            qnet=qnet_hat,
            mp=mp_hat,
            tp=evaluator.tp_base,
            k0=ds.k1,
            b0=ds.b1,
            z0=ds.z1,
            eps_future=ds.eps_tp1,
            valid_mask=ds.eps_tp1_mask,
        ).numpy()
        default_obs = (cont_obs <= 0.0).astype(float)
        recovery_obs = recovery_R(ds.k1, ds.z1, mp_hat).numpy().astype(float)
        sim_fit_dataset = {
            "q": q_obs,
            "b_next": ds.b1.numpy().astype(float),
            "default": default_obs,
            "recovery": recovery_obs,
            "spread": ((1.0 / np.clip(q_obs, mp_hat.q_min, mp_hat.q_max)) - 1.0 - mp_hat.r).astype(float),
        }
        simulated_stats = dataset_pricing_default_summary(sim_fit_dataset, mp_hat)

    dof = max(0, g_hat.size - len(param_names))
    J_stat = float(ds.n_obs * (g_hat @ w @ g_hat))
    J_p_value = float(_chi2_dist.sf(J_stat, dof)) if (_chi2_dist is not None and dof > 0 and np.isfinite(J_stat)) else float("nan")
    return {
        "label": label,
        "success": True,
        "moment_mode": str(getattr(evaluator, "moment_mode", "derivative")),
        "light_reporting": bool(light_report),
        "theta_hat": {name: float(getattr(mp_hat, name)) for name in param_names},
        "theta_hat_vector": x_hat.tolist(),
        "objective": float(best["objective"]),
        "best_start_id": int(best["start_id"]),
        "best_start_theta": list(best["start_theta"]),
        "convergence_flag": bool(best.get("converged", False)),
        "evals": int(best.get("evals", 0)),
        "J_stat": J_stat,
        "J_dof": int(dof),
        "J_p_value": J_p_value,
        "J_reject_5pct": bool(J_p_value < 0.05) if np.isfinite(J_p_value) else False,
        "g_norm": float(np.linalg.norm(g_hat)),
        "max_abs_moment": float(np.max(np.abs(g_hat))),
        "weight_matrix_condition": float(cond),
        "jacobian_diagnostics": jacobian_diagnostics,
        "parameter_table": param_table,
        "recovery_score": overall_param_recovery_score(mp_hat, mp_true, param_names),
        "moment_dimension": int(g_hat.size),
        "pricing_default_fit": {
            "observed": observed_stats,
            "simulated": simulated_stats,
            "errors": {k: float(simulated_stats[k] - observed_stats[k]) for k in observed_stats.keys()},
        },
        "mean_abs_zero_profit_moment": float(np.mean(np.abs(mean_zp_mom))),
        "starts": list(runs),
        "multistart_summary": summarize_multistart_runs(runs, param_names),
    }



def _build_gmm_identification_report(
    evaluator: _GMMEvaluator,
    mp_true: ModelParams,
    param_names: Sequence[str],
    bounds: Sequence[Tuple[float, float]],
    n_grid: int = 5,
) -> Dict[str, object]:
    """Build mm identification report."""
    base_x = np.asarray([float(getattr(mp_true, n)) for n in param_names], dtype=float)
    sweeps: Dict[str, object] = {}
    for i, name in enumerate(param_names):
        lo, hi = bounds[i]
        grid = np.linspace(lo, hi, n_grid, dtype=float)
        objectives: List[float] = []
        g_norm: List[float] = []
        mean_abs_zp: List[float] = []
        mean_u_k: List[float] = []
        mean_u_b: List[float] = []
        for val in grid:
            x = base_x.copy()
            x[i] = float(val)
            out = evaluator.evaluate(x)
            g = np.asarray(out["g"], dtype=float)
            m = np.asarray(out["m_series"], dtype=float)
            ds = evaluator.dataset
            if getattr(evaluator, "moment_mode", "derivative") == "pricing_only":
                u_k_block = np.zeros((m.shape[0], 0), dtype=float)
                u_b_block = np.zeros((m.shape[0], 0), dtype=float)
                zp_block = m
            else:
                u_k_block = m[:, : ds.L]
                u_b_block = m[:, ds.L : 2 * ds.L]
                zp_block = m[:, 2 * ds.L : 3 * ds.L]
            objectives.append(float(g @ g))
            g_norm.append(float(np.linalg.norm(g)))
            mean_abs_zp.append(float(np.mean(np.abs(zp_block))) if zp_block.size else float("nan"))
            mean_u_k.append(float(np.mean(np.abs(u_k_block))) if u_k_block.size else float("nan"))
            mean_u_b.append(float(np.mean(np.abs(u_b_block))) if u_b_block.size else float("nan"))
        sweeps[name] = {
            "grid": grid.tolist(),
            "objective": objectives,
            "curves": {
                "g_norm": g_norm,
                "mean_abs_zero_profit_moment": mean_abs_zp,
                "mean_abs_k_block": mean_u_k,
                "mean_abs_b_block": mean_u_b,
            },
        }
    return {
        "parameter_names": list(param_names),
        "true_theta": {name: float(getattr(mp_true, name)) for name in param_names},
        "sweeps": sweeps,
    }



def estimate_gmm(
    out_dir: str,
    mp_true: ModelParams,
    npol: NetParams,
    nq: NetParams,
    tp_base: TrainParams,
    data: Dict[str, np.ndarray],
    est_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    max_evals: int = 60,
    inner_epochs: int = 3,
    inner_steps_per_epoch: int = 20,
    seed: int = 4321,
    n_starts: int = 3,
    continuation_horizon: int = 0,
    moment_mode: str = "derivative",
    report_mode: str = "full",
    progress_reporter: Optional[EstimationProgressReporter] = None,
    initial_policy_weights: Optional[Sequence[np.ndarray]] = None,
    initial_q_weights: Optional[Sequence[np.ndarray]] = None,
) -> Dict[str, object]:
    """Run two-step GMM with selectable moment systems.

    ``moment_mode='derivative'`` is the written-summary setting.
    ``moment_mode='pricing_only'`` is an optional faster robustness/completion
    mode that avoids derivative-based FOC moments.
    """
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    reporter = progress_reporter
    report_mode = str(report_mode).lower().strip()
    if report_mode not in {"full", "light"}:
        raise ValueError("report_mode must be 'full' or 'light'.")
    if reporter is not None:
        reporter.start_phase("total_estimation", out_dir=out_dir, report_mode=report_mode)

    if est_bounds is None:
        est_bounds = default_estimation_bounds()
    else:
        est_bounds = {k: v for k, v in est_bounds.items() if k in default_estimation_bounds()}
        if not est_bounds:
            est_bounds = default_estimation_bounds()

    param_names = ordered_param_names(est_bounds)
    bounds = bounds_in_order(est_bounds, param_names)

    if reporter is not None:
        reporter.start_phase("dataset_alignment_and_evaluator_build")
    evaluator = _GMMEvaluator(
        mp_true=mp_true,
        npol=npol,
        nq=nq,
        tp_base=tp_base,
        data=data,
        inner_epochs=inner_epochs,
        inner_steps_per_epoch=inner_steps_per_epoch,
        seed=seed,
        param_names=param_names,
        continuation_horizon=continuation_horizon,
        moment_mode=moment_mode,
        initial_policy_weights=initial_policy_weights,
        initial_q_weights=initial_q_weights,
    )

    if reporter is not None:
        reporter.finish_phase("dataset_alignment_and_evaluator_build", n_obs=int(evaluator.dataset.n_obs), n_instruments=int(evaluator.dataset.L))

    def obj_stage1(x: np.ndarray) -> float:
        """Evaluate the first-stage identity-weight GMM objective."""
        g = np.asarray(evaluator.evaluate(x)["g"], dtype=float)
        return float(g @ g)

    starts = generate_parameter_starts(bounds=bounds, n_starts=n_starts, seed=seed + 100)
    if reporter is not None:
        reporter.start_phase("stage1_identity_weight")
    stage1_runs = _run_multistart(
        obj_stage1,
        starts,
        bounds,
        max_evals,
        progress_reporter=reporter,
        phase_name="stage1_identity_weight",
        resume_dir=os.path.join(out_dir, "resume", "gmm", "stage1_identity_weight") if str(moment_mode) == "derivative" else os.path.join(out_dir, "resume", "gmm", str(moment_mode), "stage1_identity_weight"),
    )
    successful_stage1 = [r for r in stage1_runs if bool(r.get("success", False)) and np.isfinite(float(r["objective"]))]
    if reporter is not None:
        reporter.finish_phase(
            "stage1_identity_weight",
            successful_starts=len(successful_stage1),
            best_objective=float(min([r["objective"] for r in successful_stage1], default=float("inf"))),
        )
    if not successful_stage1:
        raise RuntimeError("All GMM stage-1 starts failed.")

    best_stage1 = min(successful_stage1, key=lambda d: float(d["objective"]))
    m_stage1 = np.asarray(evaluator.evaluate(best_stage1["theta_hat_vector"])["m_series"], dtype=float)
    nw_lags = newey_west_lag_length(m_stage1.shape[0])
    omega_std = covariance_of_mean(m_stage1, hac_lags=0)
    omega_nw = covariance_of_mean(m_stage1, hac_lags=nw_lags)

    observed_fit_dataset = {
        "q": np.asarray(data["q"], dtype=float),
        "b_next": np.asarray(data["b_next"], dtype=float),
        "default": np.asarray(data["default"], dtype=float),
        "recovery": np.asarray(data["recovery"], dtype=float),
        "spread": np.asarray(data["spread"], dtype=float),
    }
    observed_stats = dataset_pricing_default_summary(observed_fit_dataset, mp_true)

    def make_stage2_runs(omega_mean: np.ndarray, phase_name: str) -> List[Dict[str, object]]:
        """Create and execute the second-stage multistart GMM runs."""
        w, _ = stabilized_inverse(omega_mean)

        def obj_stage2(x: np.ndarray) -> float:
            """Evaluate one second-stage weighted GMM objective."""
            g = np.asarray(evaluator.evaluate(x)["g"], dtype=float)
            return float(g @ w @ g)

        stage1_endpoints = [np.asarray(r["theta_hat_vector"], dtype=float) for r in successful_stage1]
        return _run_multistart(
            obj_stage2,
            stage1_endpoints,
            bounds,
            max_evals,
            progress_reporter=reporter,
            phase_name=phase_name,
            resume_dir=os.path.join(out_dir, "resume", "gmm", phase_name) if str(moment_mode) == "derivative" else os.path.join(out_dir, "resume", "gmm", str(moment_mode), phase_name),
        )

    if reporter is not None:
        reporter.start_phase("stage2_gmm_a")
    variant_a_runs = make_stage2_runs(omega_std, "stage2_gmm_a")
    if reporter is not None:
        reporter.finish_phase("stage2_gmm_a", successful_starts=sum(bool(r.get("success", False)) for r in variant_a_runs))
    if reporter is not None:
        reporter.start_phase("stage2_gmm_b")
    variant_b_runs = make_stage2_runs(omega_nw, "stage2_gmm_b")
    if reporter is not None:
        reporter.finish_phase("stage2_gmm_b", successful_starts=sum(bool(r.get("success", False)) for r in variant_b_runs))

    variant_a = _finalize_gmm_variant(
        label="GMM-A",
        evaluator=evaluator,
        runs=variant_a_runs,
        omega_mean=omega_std,
        param_names=param_names,
        mp_true=mp_true,
        observed_stats=observed_stats,
        report_mode=report_mode,
    )
    variant_b = _finalize_gmm_variant(
        label="GMM-B",
        evaluator=evaluator,
        runs=variant_b_runs,
        omega_mean=omega_nw,
        param_names=param_names,
        mp_true=mp_true,
        observed_stats=observed_stats,
        report_mode=report_mode,
    )

    successful_variants = [v for v in [variant_a, variant_b] if bool(v.get("success", False))]
    best_variant = min(successful_variants, key=lambda d: float(d["objective"])) if successful_variants else variant_a

    if report_mode == "light":
        identification = {"skipped": True, "reason": "estimation_report_mode=light"}
    else:
        if reporter is not None:
            reporter.start_phase("gmm_identification_report")
        identification = _build_gmm_identification_report(
            evaluator=evaluator,
            mp_true=mp_true,
            param_names=param_names,
            bounds=bounds,
            n_grid=5,
        )
        save_identification_report(out_dir, "gmm", identification)
        if reporter is not None:
            reporter.finish_phase("gmm_identification_report")

    res: Dict[str, object] = {
        "method": "GMM",
        "moment_mode": str(moment_mode),
        "report_mode": str(report_mode),
        "baseline_parameters": param_names,
        "theta_true": {name: float(getattr(mp_true, name)) for name in param_names},
        "best_variant": str(best_variant.get("label", "")),
        "theta_hat": best_variant.get("theta_hat", {}),
        "objective": float(best_variant.get("objective", np.inf)),
        "newey_west_lags": int(nw_lags),
        "continuation_horizon": int(continuation_horizon),
        "dgp_source": str(np.asarray(data.get("dgp_source", "obj1_nn")).reshape(())),
        "runtime_sec": float(time.time() - t0),
        "identification_report": "gmm_identification.json",
        "computational_controls": {
            **evaluator.inner.computational_controls(),
            "gmm_moment_mode": str(moment_mode),
            "estimation_report_mode": str(report_mode),
            "derivative_foc_moments": bool(str(moment_mode) == "derivative"),
        },
        "candidate_cache_size": int(len(evaluator.cache)),
        "stage1": {
            "runs": stage1_runs,
            "summary": summarize_multistart_runs(stage1_runs, param_names),
        },
        "variants": {
            "GMM-A": variant_a,
            "GMM-B": variant_b,
        },
    }

    with open(os.path.join(out_dir, "gmm_results.json"), "w", encoding="utf-8") as f_out:
        json.dump(make_json_serializable(res), f_out, indent=2)

    if reporter is not None:
        reporter.finish_phase(
            "total_estimation",
            runtime_sec=float(time.time() - t0),
            best_variant=str(best_variant.get("label", "GMM-A")),
            objective=float(res.get("objective", np.inf)),
        )
    return res

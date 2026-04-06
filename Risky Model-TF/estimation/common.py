"""Shared helpers for risky-debt estimation.

These utilities keep the GMM/SMM code aligned with the user's final
three-parameter baseline:
    Theta = (theta, psi0, alpha)
with (rho, sigma_eps, r, tau, delta, eta0, eta1) held fixed.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import json
import os

import numpy as np

from risky_debt.config import ModelParams


BASELINE_PARAM_NAMES: Tuple[str, ...] = ("theta", "psi0", "alpha")


def make_json_serializable(obj):
    """Recursively convert NumPy-heavy objects into JSON-safe Python types."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return make_json_serializable(obj.item())
        return [make_json_serializable(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_serializable(v) for v in obj]
    if hasattr(obj, 'item'):
        try:
            return make_json_serializable(obj.item())
        except Exception:
            pass
    return str(obj)


def default_estimation_bounds() -> Dict[str, Tuple[float, float]]:
    """Baseline structural-parameter bounds used in the final plan."""
    return {
        "theta": (0.10, 0.90),
        "psi0": (0.10, 10.0),
        "alpha": (0.01, 0.95),
    }


def ordered_param_names(est_bounds: Dict[str, Tuple[float, float]]) -> List[str]:
    """Return parameter names in the baseline order when available."""
    ordered = [name for name in BASELINE_PARAM_NAMES if name in est_bounds]
    ordered += [name for name in est_bounds.keys() if name not in ordered]
    return ordered


def bounds_in_order(
    est_bounds: Dict[str, Tuple[float, float]], param_names: Sequence[str]
) -> List[Tuple[float, float]]:
    """Return ordered lower and upper bounds for the active parameters."""
    return [tuple(map(float, est_bounds[n])) for n in param_names]


def project_to_bounds(x: Sequence[float], bounds: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Project a parameter vector componentwise into the admissible bounds."""
    x = np.asarray(x, dtype=float).copy()
    for i, (lo, hi) in enumerate(bounds):
        x[i] = np.clip(x[i], lo, hi)
    return x


def unconstrained_from_bounded(
    x: Sequence[float], bounds: Sequence[Tuple[float, float]], eps: float = 1e-8
) -> np.ndarray:
    """Logit transform from bounded parameter space to unconstrained space."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    for i, (lo, hi) in enumerate(bounds):
        y = (x[i] - lo) / max(hi - lo, eps)
        y = np.clip(y, eps, 1.0 - eps)
        out[i] = np.log(y / (1.0 - y))
    return out


def bounded_from_unconstrained(
    u: Sequence[float], bounds: Sequence[Tuple[float, float]]
) -> np.ndarray:
    """Inverse-logit transform from unconstrained space to bounded parameters."""
    u = np.asarray(u, dtype=float)
    out = np.zeros_like(u, dtype=float)
    for i, (lo, hi) in enumerate(bounds):
        sig = 1.0 / (1.0 + np.exp(-u[i]))
        out[i] = lo + (hi - lo) * sig
    return out


def generate_parameter_starts(
    bounds: Sequence[Tuple[float, float]],
    n_starts: int,
    seed: int,
    include_midpoint: bool = True,
) -> List[np.ndarray]:
    """Generate multiple bounded starting values for local optimization.

    Starts deliberately avoid relying on the true parameter vector, since the
    user's final comparison design treats multi-start robustness as part of the
    effectiveness evaluation.
    """
    rng = np.random.default_rng(seed)
    starts: List[np.ndarray] = []

    if include_midpoint and n_starts > 0:
        midpoint = np.asarray([(lo + hi) * 0.5 for (lo, hi) in bounds], dtype=float)
        starts.append(midpoint)

    while len(starts) < max(1, n_starts):
        x = np.asarray([rng.uniform(lo, hi) for (lo, hi) in bounds], dtype=float)
        starts.append(x)

    # Deduplicate near-identical starts while preserving order.
    uniq: List[np.ndarray] = []
    for x in starts:
        if not any(np.allclose(x, y, atol=1e-10, rtol=0.0) for y in uniq):
            uniq.append(x)
    return uniq


def panel_shape_from_dataset(dataset: Dict[str, np.ndarray]) -> Tuple[int, int]:
    """Recover (n_paths, T_eff) from a flattened synthetic dataset."""
    if "n_paths" in dataset and "T_eff" in dataset:
        n_paths = int(np.asarray(dataset["n_paths"]).reshape(()))
        T_eff = int(np.asarray(dataset["T_eff"]).reshape(()))
        return max(1, n_paths), max(0, T_eff)

    n_obs = int(np.asarray(dataset["k"]).reshape(-1).size)
    return 1, n_obs


def panel_from_flat(dataset: Dict[str, np.ndarray], key: str) -> np.ndarray:
    """Reshape a flattened series into panel form (n_paths, T_eff)."""
    arr = np.asarray(dataset[key], dtype=float).reshape(-1)
    n_paths, t_eff = panel_shape_from_dataset(dataset)
    if n_paths * t_eff != arr.size:
        n_paths, t_eff = 1, arr.size
    return arr.reshape(n_paths, t_eff)


def dataset_observation_count(dataset: Dict[str, np.ndarray]) -> int:
    """Return the number of usable observations stored in a dataset bundle."""
    n_paths, t_eff = panel_shape_from_dataset(dataset)
    return max(1, n_paths * max(0, t_eff - 1))


def _safe_div(num: np.ndarray, den: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Return a numerically safe version of iv."""
    return np.asarray(num, dtype=float) / np.maximum(np.asarray(den, dtype=float), floor)


def _reportable_moments_from_base_mean(mu: np.ndarray) -> np.ndarray:
    """Map base means into the SMM reportable moment vector."""
    mu = np.asarray(mu, dtype=float)
    eps = 1e-12

    mean_i = mu[0]
    var_i = max(mu[1] - mean_i * mean_i, eps)
    std_i = np.sqrt(var_i)
    ac1_i = (mu[2] - mean_i * mean_i) / var_i

    mean_p = mu[3]
    var_p = max(mu[4] - mean_p * mean_p, eps)
    std_p = np.sqrt(var_p)
    corr_pi_i = (mu[5] - mean_p * mean_i) / max(std_p * std_i, eps)

    mean_l = mu[6]
    var_l = max(mu[7] - mean_l * mean_l, eps)
    std_l = np.sqrt(var_l)
    ac1_l = (mu[8] - mean_l * mean_l) / var_l

    mean_spread = mu[9]
    default_rate = np.clip(mu[10], 0.0, 1.0)
    mean_recovery_default = mu[11] / max(default_rate, eps)

    return np.asarray(
        [
            mean_i,
            std_i,
            ac1_i,
            mean_p,
            std_p,
            corr_pi_i,
            mean_l,
            std_l,
            ac1_l,
            mean_spread,
            default_rate,
            mean_recovery_default,
        ],
        dtype=float,
    )


def reportable_moment_names() -> List[str]:
    """Return the canonical names of the reportable SMM moments."""
    return [
        "mean_I_over_k",
        "std_I_over_k",
        "ac1_I_over_k",
        "mean_pi_over_k",
        "std_pi_over_k",
        "corr_pi_over_k_I_over_k",
        "mean_b_over_k",
        "std_b_over_k",
        "ac1_b_over_k",
        "mean_spread",
        "default_rate",
        "mean_recovery_default",
    ]


def build_smm_base_feature_panel(
    dataset: Dict[str, np.ndarray],
    mp: ModelParams,
) -> Tuple[List[str], np.ndarray]:
    """Construct time-series base features used for SMM moments."""
    k = panel_from_flat(dataset, "k")
    b = panel_from_flat(dataset, "b")
    z = panel_from_flat(dataset, "z")
    I = panel_from_flat(dataset, "I")
    spread = panel_from_flat(dataset, "spread")
    default = panel_from_flat(dataset, "default")
    k_next = panel_from_flat(dataset, "k_next")
    recovery = panel_from_flat(dataset, "recovery")

    if k.shape[1] < 2:
        raise ValueError("Synthetic dataset is too short for lagged SMM moments.")

    inv = _safe_div(I, k, mp.k_min)
    prof = z * np.power(np.maximum(k, mp.k_min), mp.theta - 1.0)
    lev = _safe_div(b, k, mp.k_min)

    inv_t = inv[:, 1:]
    inv_tm1 = inv[:, :-1]
    prof_t = prof[:, 1:]
    lev_t = lev[:, 1:]
    lev_tm1 = lev[:, :-1]
    spread_t = spread[:, 1:]
    default_t = default[:, 1:]
    recovery_t = recovery[:, 1:]

    features = np.column_stack(
        [
            inv_t.reshape(-1),
            np.square(inv_t).reshape(-1),
            (inv_t * inv_tm1).reshape(-1),
            prof_t.reshape(-1),
            np.square(prof_t).reshape(-1),
            (prof_t * inv_t).reshape(-1),
            lev_t.reshape(-1),
            np.square(lev_t).reshape(-1),
            (lev_t * lev_tm1).reshape(-1),
            spread_t.reshape(-1),
            default_t.reshape(-1),
            (default_t * recovery_t).reshape(-1),
        ]
    )
    names = [
        "E_I",
        "E_I2",
        "E_I_lagprod",
        "E_pi",
        "E_pi2",
        "E_piI",
        "E_lev",
        "E_lev2",
        "E_lev_lagprod",
        "E_spread",
        "E_default",
        "E_default_recovery",
    ]
    return names, features.astype(float)


def compute_smm_moments(
    dataset: Dict[str, np.ndarray],
    mp: ModelParams,
) -> Tuple[List[str], np.ndarray, List[str], np.ndarray, np.ndarray]:
    """Return reportable SMM moments plus the underlying feature objects."""
    base_names, features = build_smm_base_feature_panel(dataset, mp)
    base_mean = np.mean(features, axis=0)
    moments = _reportable_moments_from_base_mean(base_mean)
    return reportable_moment_names(), moments, base_names, base_mean, features


def dataset_pricing_default_summary(dataset: Dict[str, np.ndarray], mp: ModelParams) -> Dict[str, float]:
    """Pricing/default fit summary used by both GMM and SMM reports."""
    q = np.asarray(dataset["q"], dtype=float).reshape(-1)
    b_next = np.asarray(dataset["b_next"], dtype=float).reshape(-1)
    default = np.asarray(dataset["default"], dtype=float).reshape(-1)
    recovery = np.asarray(dataset["recovery"], dtype=float).reshape(-1)
    spread = np.asarray(dataset["spread"], dtype=float).reshape(-1)

    q = np.clip(q, mp.q_min, mp.q_max)
    zp = (1.0 + mp.r) * b_next - (default * recovery + (1.0 - default) * (b_next / q))

    return {
        "mean_spread": float(np.mean(spread)) if spread.size else 0.0,
        "default_rate": float(np.mean(default)) if default.size else 0.0,
        "mean_recovery_default": float(np.sum(default * recovery) / max(np.sum(default), 1.0e-12)),
        "mean_zero_profit_residual": float(np.mean(zp)) if zp.size else 0.0,
        "mean_abs_zero_profit_residual": float(np.mean(np.abs(zp))) if zp.size else 0.0,
    }


def numerical_jacobian(
    f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    rel_step: float = 1e-4,
    abs_step: float = 1e-6,
) -> np.ndarray:
    """Simple central-difference Jacobian."""
    x0 = np.asarray(x0, dtype=float)
    y0 = np.asarray(f(x0), dtype=float)
    jac = np.zeros((y0.size, x0.size), dtype=float)
    for j in range(x0.size):
        h = max(abs_step, rel_step * max(1.0, abs(x0[j])))
        x_lo = x0.copy()
        x_hi = x0.copy()
        x_lo[j] -= h
        x_hi[j] += h
        y_lo = np.asarray(f(x_lo), dtype=float)
        y_hi = np.asarray(f(x_hi), dtype=float)
        jac[:, j] = (y_hi - y_lo) / (2.0 * h)
    return jac


def newey_west_lag_length(n_obs: int) -> int:
    """Default HAC lag length."""
    if n_obs <= 2:
        return 0
    return int(max(1, round(4.0 * (n_obs / 100.0) ** (2.0 / 9.0))))


def covariance_of_mean(features: np.ndarray, hac_lags: int = 0) -> np.ndarray:
    """Covariance of the sample mean of a vector time series."""
    x = np.asarray(features, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n_obs, q = x.shape
    if n_obs <= 1:
        return np.eye(q, dtype=float) * 1e-6

    x = x - np.mean(x, axis=0, keepdims=True)
    gamma0 = (x.T @ x) / n_obs
    s = gamma0.copy()
    max_lag = min(int(max(0, hac_lags)), n_obs - 1)
    for lag in range(1, max_lag + 1):
        w = 1.0 - lag / (max_lag + 1.0)
        gamma = (x[lag:].T @ x[:-lag]) / n_obs
        s += w * (gamma + gamma.T)
    omega_mean = s / n_obs
    omega_mean = 0.5 * (omega_mean + omega_mean.T)
    return omega_mean


def stabilized_inverse(mat: np.ndarray, ridge: float = 1e-6) -> Tuple[np.ndarray, float]:
    """Return a ridge-stabilized inverse and condition number."""
    a = np.asarray(mat, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Matrix must be square.")
    scale = max(1.0, float(np.mean(np.diag(a))) if a.size else 1.0)
    reg = ridge * scale
    a_reg = a + reg * np.eye(a.shape[0], dtype=float)
    cond = float(np.linalg.cond(a_reg))
    inv = np.linalg.pinv(a_reg)
    return inv, cond


def apply_params(mp: ModelParams, param_names: Sequence[str], x: Sequence[float]) -> ModelParams:
    """Return a new ModelParams with updated structural parameters."""
    out = mp
    for name, val in zip(param_names, x):
        out = replace(out, **{name: float(val)})
    return out


def summarize_param_errors(
    mp_hat: ModelParams,
    mp_true: ModelParams,
    param_names: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """Absolute and relative errors by parameter."""
    out: Dict[str, Dict[str, float]] = {}
    for name in param_names:
        true_val = float(getattr(mp_true, name))
        hat_val = float(getattr(mp_hat, name))
        abs_err = abs(hat_val - true_val)
        rel_err = abs_err / max(abs(true_val), 1e-12)
        out[name] = {
            "true": true_val,
            "hat": hat_val,
            "abs_error": float(abs_err),
            "rel_error": float(rel_err),
        }
    return out


def overall_param_recovery_score(mp_hat: ModelParams, mp_true: ModelParams, param_names: Sequence[str]) -> float:
    """Return one scalar summary of overall parameter-recovery quality."""
    err = np.asarray([float(getattr(mp_hat, n) - getattr(mp_true, n)) for n in param_names], dtype=float)
    return float(np.linalg.norm(err))



def save_identification_report(
    out_dir: str,
    prefix: str,
    report: Dict[str, object],
) -> None:
    """Persist identification sweeps to JSON and line plots."""
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{prefix}_identification.json")
    with open(json_path, "w", encoding="utf-8") as f_out:
        json.dump(make_json_serializable(report), f_out, indent=2)

    try:
        import matplotlib.pyplot as plt

        sweeps = report.get("sweeps", {})
        for param_name, payload in sweeps.items():
            grid = np.asarray(payload.get("grid", []), dtype=float)
            objective = np.asarray(payload.get("objective", []), dtype=float)
            curves = payload.get("curves", {})
            fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.0), constrained_layout=True)
            axes[0].plot(grid, objective, marker="o")
            axes[0].set_title(f"{prefix} identification: {param_name}")
            axes[0].set_xlabel(param_name)
            axes[0].set_ylabel("stage-1 objective")

            legend_labels = []
            for curve_name, curve_vals in curves.items():
                curve_arr = np.asarray(curve_vals, dtype=float)
                axes[1].plot(grid, curve_arr, marker="o")
                legend_labels.append(curve_name)
            axes[1].set_xlabel(param_name)
            axes[1].set_ylabel("moment / fit statistic")
            if legend_labels:
                axes[1].legend(legend_labels, fontsize=8)
            fig.savefig(os.path.join(out_dir, f"{prefix}_identification_{param_name}.png"), dpi=150)
            plt.close(fig)
    except Exception:
        # Plotting is auxiliary; JSON is the primary artifact.
        pass

def summarize_multistart_runs(
    runs: Sequence[Dict[str, object]],
    param_names: Sequence[str],
) -> Dict[str, object]:
    """Summarize multiple local optimizations for robustness reporting."""
    successful = [r for r in runs if bool(r.get("success", False))]
    if not successful:
        return {
            "n_starts": int(len(runs)),
            "n_successful": 0,
            "n_failed": int(len(runs)),
            "objective_min": None,
            "objective_max": None,
            "objective_std": None,
            "param_std": {name: None for name in param_names},
        }

    objs = np.asarray([float(r["objective"]) for r in successful], dtype=float)
    theta = np.asarray([r["theta_hat_vector"] for r in successful], dtype=float)
    return {
        "n_starts": int(len(runs)),
        "n_successful": int(len(successful)),
        "n_failed": int(len(runs) - len(successful)),
        "objective_min": float(np.min(objs)),
        "objective_max": float(np.max(objs)),
        "objective_std": float(np.std(objs)),
        "param_std": {name: float(np.std(theta[:, i])) for i, name in enumerate(param_names)},
    }

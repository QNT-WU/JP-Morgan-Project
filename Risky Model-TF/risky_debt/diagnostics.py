"""Post-training diagnostics and report tables for risky-debt solvers.

The training histories show whether the neural objectives optimized their own
losses, but the written risky-debt specification also requires an economic
reporting layer.  This module evaluates each learned policy under both the
smooth training rules and the exact nonsmoothed economic rules, then writes
CSV/TeX/JSON tables and diagnostic plots for comparison across Objective 1,
Objective 2, and Objective 3.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from estimation.common import make_json_serializable

from .config import ModelParams, TrainParams
from .evaluation import eval_test_reward
from .networks import PolicyNet, PricingNet, ValueNet, VtildeNet, MultiplierNet
from .pricing import (
    crn_inner_eps,
    debt_region_weight,
    exact_positive_debt_tax_shield,
    exact_price_from_proxy,
    exact_price_from_vtilde,
    exact_pricing_moments_from_proxy,
    exact_pricing_moments_from_vtilde,
    positive_debt_tax_shield,
    smooth_price_from_proxy,
    smooth_price_from_vtilde,
    pricing_moments_from_proxy,
    pricing_moments_from_vtilde,
)
from .primitives import (
    beta_tensor_from_r,
    continuation_weight_from_value,
    equity_cashflow_base_e,
    equity_payout_d,
    equity_payout_d_exact,
    issuance_eta_exact,
    solvency_weight,
)


@dataclass(frozen=True)
class SolverDiagnosticConfig:
    """Configuration for final post-training solver diagnostics."""

    n_states: int = 2048
    reward_seed: int = 200123
    state_seed: int = 200456
    shock_seed: int = 200789
    near_zero_band: float = 0.05
    boundary_tol: float = 1e-3


def _ensure_dir(path: str) -> None:
    """Create ``path`` when it does not already exist."""
    os.makedirs(path, exist_ok=True)


def _to_numpy(x) -> np.ndarray:
    """Convert a TensorFlow tensor or array-like object to a NumPy array."""
    if isinstance(x, tf.Tensor):
        return x.numpy()
    return np.asarray(x)


def _safe_mean(x: np.ndarray) -> float:
    """Return a finite mean or ``nan`` for an empty vector."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def _safe_rmse(x: np.ndarray) -> float:
    """Return a root-mean-square value or ``nan`` for an empty vector."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.nanmean(np.square(arr))))


def _percentiles(x: np.ndarray, prefix: str) -> Dict[str, float]:
    """Return common percentiles for one vector with prefixed names."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return {f"{prefix}_p05": float("nan"), f"{prefix}_p50": float("nan"), f"{prefix}_p95": float("nan")}
    return {
        f"{prefix}_p05": float(np.nanpercentile(arr, 5)),
        f"{prefix}_p50": float(np.nanpercentile(arr, 50)),
        f"{prefix}_p95": float(np.nanpercentile(arr, 95)),
    }


def _write_csv(path: str, rows: Iterable[Mapping[str, object]], fieldnames: Optional[list[str]] = None) -> None:
    """Write a list of dictionaries as a CSV file."""
    rows = list(rows)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _format_tex_value(value: object) -> str:
    """Format one scalar value for a simple LaTeX table."""
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return "--"
        return f"{float(value):.6g}"
    return str(value).replace("_", "\\_")


def _write_tex(path: str, rows: Iterable[Mapping[str, object]], columns: list[str], caption: str) -> None:
    """Write a compact standalone LaTeX tabular file."""
    rows = list(rows)
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\\begin{table}[htbp]\n\\centering\n")
        handle.write(f"\\caption{{{caption}}}\n")
        handle.write("\\begin{tabular}{" + "l" * len(columns) + "}\n")
        handle.write("\\hline\n")
        handle.write(" & ".join(col.replace("_", "\\_") for col in columns) + " \\\\ \n")
        handle.write("\\hline\n")
        for row in rows:
            handle.write(" & ".join(_format_tex_value(row.get(col, "")) for col in columns) + " \\\\ \n")
        handle.write("\\hline\n\\end{tabular}\n\\end{table}\n")


def _sample_test_states(mp: ModelParams, tp: TrainParams, cfg: SolverDiagnosticConfig):
    """Draw common diagnostic states for all objectives."""
    tf.random.set_seed(cfg.state_seed)
    n = int(min(max(cfg.n_states, 32), max(tp.N_test_states * 4, 64)))
    k = tf.random.uniform((n,), tp.k0_low, tp.k0_high, dtype=tf.float32)
    b = tf.random.uniform((n,), tp.b0_low, tp.b0_high, dtype=tf.float32)
    z = tf.random.uniform((n,), tp.z0_low, tp.z0_high, dtype=tf.float32)
    return tf.maximum(k, mp.k_min), b, tf.maximum(z, mp.z_min)


def _policy_next(policy: PolicyNet, mp: ModelParams, k, b, z):
    """Evaluate a policy on diagnostic states."""
    x = tf.stack([k, b, z], axis=1)
    kb = policy(x)
    return tf.maximum(kb[:, 0], mp.k_min), kb[:, 1]


def _value_default_objects(vtilde: Optional[VtildeNet], mp: ModelParams, tp: TrainParams, k_next, b_next, z_next):
    """Return smooth and exact continuation/default objects."""
    if vtilde is None:
        smooth = solvency_weight(k_next, b_next, z_next, mp, tp.kappa_solv)
        exact = tf.cast(smooth > 0.5, tf.float32)
        vt = tf.zeros_like(k_next)
    else:
        vt = vtilde(tf.stack([k_next, b_next, z_next], axis=1))
        smooth = continuation_weight_from_value(vt, tp.kappa_solv)
        exact = tf.cast(vt > 0.0, tf.float32)
    return vt, smooth, exact


def _plot_hist(values: np.ndarray, title: str, xlabel: str, path: str, bins: int = 40) -> None:
    """Save a histogram diagnostic."""
    _ensure_dir(os.path.dirname(path))
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    plt.figure()
    if arr.size:
        plt.hist(arr, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_bar(names: list[str], values: list[float], title: str, ylabel: str, path: str) -> None:
    """Save a bar chart for cross-objective comparison."""
    _ensure_dir(os.path.dirname(path))
    plt.figure()
    plt.bar(names, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_scatter(x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str, path: str) -> None:
    """Save a small scatter diagnostic."""
    _ensure_dir(os.path.dirname(path))
    plt.figure()
    n = min(len(x), 2000)
    if n > 0:
        plt.scatter(np.asarray(x)[:n], np.asarray(y)[:n], s=6, alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _fb_tf(a: tf.Tensor, c: tf.Tensor) -> tf.Tensor:
    """TensorFlow Fischer--Burmeister residual used in diagnostics."""
    return a + c - tf.sqrt(a * a + c * c + 1e-12)


def _scatter_colored(x: np.ndarray, y: np.ndarray, c: np.ndarray, title: str, xlabel: str, ylabel: str, path: str) -> None:
    """Save a colored scatter plot for policy/default/price visual diagnostics."""
    _ensure_dir(os.path.dirname(path))
    plt.figure()
    n = min(len(x), 2500)
    if n > 0:
        sc = plt.scatter(np.asarray(x)[:n], np.asarray(y)[:n], c=np.asarray(c)[:n], s=7, alpha=0.65)
        plt.colorbar(sc)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


class SolverDiagnosticsReporter:
    """Generate exact-vs-smooth solver diagnostics after training."""

    def __init__(self, *, mp: ModelParams, tp_by_obj: Mapping[str, TrainParams], tables_dir: str, figures_dir: str) -> None:
        """Initialize the reporter with model parameters and output folders."""
        self.mp = mp
        self.tp_by_obj = dict(tp_by_obj)
        self.tables_dir = tables_dir
        self.figures_dir = figures_dir
        _ensure_dir(self.tables_dir)
        _ensure_dir(self.figures_dir)

    def _residual_vectors(self, artifacts, tp: TrainParams, k: tf.Tensor, b: tf.Tensor, z: tf.Tensor, z_next: tf.Tensor) -> Dict[str, np.ndarray]:
        """Compute Bellman/default/Euler/KKT residual vectors for final reporting.

        These diagnostics are evaluated after training and do not affect the
        optimizer.  They use the smoothed differentiable rules to report
        residual accuracy and the exact rules to report the exact Bellman gap.
        """
        policy = artifacts.policy
        value = artifacts.value
        vtilde = artifacts.vtilde
        lambda_k = artifacts.lambda_k
        beta = beta_tensor_from_r(self.mp.r)
        k_next, b_next = _policy_next(policy, self.mp, k, b, z)
        x = tf.stack([k, b, z], axis=1)
        x_next = tf.stack([k_next, b_next, z_next], axis=1)
        if vtilde is None:
            vt = tf.zeros_like(k)
            vt_next = tf.zeros_like(k_next)
            s_next = solvency_weight(k_next, b_next, z_next, self.mp, tp.kappa_solv)
        else:
            vt = vtilde(x)
            vt_next = vtilde(x_next)
            s_next = continuation_weight_from_value(vt_next, tp.kappa_solv)
        if value is None:
            V = tf.nn.relu(vt)
            V_next = tf.nn.relu(vt_next)
        else:
            V = value(x)
            V_next = value(x_next)

        eps_q = crn_inner_eps(z, tp)
        if vtilde is None:
            q_s, _, rd_s, _ = smooth_price_from_proxy(z, k_next, b_next, eps_q, self.mp, tp)
            q_e, _, rd_e = exact_price_from_proxy(z, k_next, b_next, eps_q, self.mp, tp)
        else:
            q_s, _, rd_s, _ = smooth_price_from_vtilde(vtilde, z, k_next, b_next, eps_q, self.mp, tp)
            q_e, _, rd_e = exact_price_from_vtilde(vtilde, z, k_next, b_next, eps_q, self.mp, tp)

        d0_s = equity_payout_d(k, k_next, b, b_next, z, q_s, self.mp, tp.kappa_issue)
        ts_s = positive_debt_tax_shield(b_next, rd_s, s_next, self.mp, tp)
        R_bell_s = vt - (d0_s + beta * (V_next + ts_s))

        s_exact = tf.cast(vt_next > 0.0, tf.float32) if vtilde is not None else tf.cast(s_next > 0.5, tf.float32)
        d0_e = equity_payout_d_exact(k, k_next, b, b_next, z, q_e, self.mp)
        ts_e = exact_positive_debt_tax_shield(b_next, rd_e, s_exact, self.mp)
        R_bell_e = vt - (d0_e + beta * (V_next + ts_e))
        R_def = _fb_tf(V, V - vt)

        with tf.GradientTape() as tape:
            tape.watch([k_next, b_next])
            if vtilde is None:
                q_g, _, rd_g, _ = smooth_price_from_proxy(z, k_next, b_next, eps_q, self.mp, tp)
            else:
                q_g, _, rd_g, _ = smooth_price_from_vtilde(vtilde, z, k_next, b_next, eps_q, self.mp, tp)
            x_next_g = tf.stack([k_next, b_next, z_next], axis=1)
            if value is None:
                vt_next_g = tf.zeros_like(k_next) if vtilde is None else vtilde(x_next_g)
                V_next_g = tf.nn.relu(vt_next_g)
            else:
                V_next_g = value(x_next_g)
                vt_next_g = V_next_g if vtilde is None else vtilde(x_next_g)
            s_next_g = solvency_weight(k_next, b_next, z_next, self.mp, tp.kappa_solv) if vtilde is None else continuation_weight_from_value(vt_next_g, tp.kappa_solv)
            d0_g = equity_payout_d(k, k_next, b, b_next, z, q_g, self.mp, tp.kappa_issue)
            ts_g = positive_debt_tax_shield(b_next, rd_g, s_next_g, self.mp, tp)
            J = d0_g + beta * (V_next_g + ts_g)
            Jsum = tf.reduce_sum(J)
        Gk, Gb = tape.gradient(Jsum, [k_next, b_next])
        if lambda_k is None:
            lam = tf.nn.softplus(-Gk)
        else:
            lam = lambda_k(x)
        R_stat_k = Gk + lam
        R_comp_k = _fb_tf(lam, k_next - self.mp.k_min)

        return {
            "bellman_residual_smooth": _to_numpy(R_bell_s),
            "bellman_residual_exact": _to_numpy(R_bell_e),
            "default_residual": _to_numpy(R_def),
            "Gk": _to_numpy(Gk),
            "Gb": _to_numpy(Gb),
            "R_stat_k": _to_numpy(R_stat_k),
            "R_comp_k": _to_numpy(R_comp_k),
            "R_bprime": _to_numpy(Gb),
            "lambda_k": _to_numpy(lam),
        }

    def _diagnose_one(self, name: str, artifacts, cfg: SolverDiagnosticConfig) -> tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """Compute scalar diagnostics and vectors for one trained objective."""
        tp = self.tp_by_obj.get(name, next(iter(self.tp_by_obj.values())))
        policy = artifacts.policy
        qnet = artifacts.qnet
        vtilde = artifacts.vtilde
        if qnet is None:
            # All current artifacts carry the compatibility qnet, but keep a clear
            # error message if future callers omit it.
            raise ValueError(f"{name} diagnostics require the compatibility pricing network object.")

        k, b, z = _sample_test_states(self.mp, tp, cfg)
        k_next, b_next = _policy_next(policy, self.mp, k, b, z)

        tf.random.set_seed(cfg.shock_seed)
        eps_realized = tf.random.normal(tf.shape(z), 0.0, self.mp.sigma_eps, tf.float32)
        z_next = tf.exp(self.mp.rho * tf.math.log(tf.maximum(z, self.mp.z_min)) + eps_realized)
        vt_next, s_smooth_next, s_exact_next = _value_default_objects(vtilde, self.mp, tp, k_next, b_next, z_next)

        tf.random.set_seed(cfg.shock_seed + 11)
        eps_q = crn_inner_eps(z, tp)
        if vtilde is None:
            p_s_smooth, er_d_smooth = pricing_moments_from_proxy(z, k_next, b_next, eps_q, self.mp, tp)
            p_s_exact, er_d_exact = exact_pricing_moments_from_proxy(z, k_next, b_next, eps_q, self.mp, tp)
            q_smooth, qd_smooth, rd_smooth, p_pen = smooth_price_from_proxy(z, k_next, b_next, eps_q, self.mp, tp)
            q_exact, qd_exact, rd_exact = exact_price_from_proxy(z, k_next, b_next, eps_q, self.mp, tp)
        else:
            p_s_smooth, er_d_smooth = pricing_moments_from_vtilde(vtilde, z, k_next, b_next, eps_q, self.mp, tp)
            p_s_exact, er_d_exact = exact_pricing_moments_from_vtilde(vtilde, z, k_next, b_next, eps_q, self.mp, tp)
            q_smooth, qd_smooth, rd_smooth, p_pen = smooth_price_from_vtilde(vtilde, z, k_next, b_next, eps_q, self.mp, tp)
            q_exact, qd_exact, rd_exact = exact_price_from_vtilde(vtilde, z, k_next, b_next, eps_q, self.mp, tp)

        e0_smooth = equity_cashflow_base_e(k, k_next, b, b_next, z, q_smooth, self.mp)
        e0_exact = equity_cashflow_base_e(k, k_next, b, b_next, z, q_exact, self.mp)
        d0_smooth = equity_payout_d(k, k_next, b, b_next, z, q_smooth, self.mp, tp.kappa_issue)
        d0_exact = equity_payout_d_exact(k, k_next, b, b_next, z, q_exact, self.mp)
        ts_smooth = positive_debt_tax_shield(b_next, rd_smooth, s_smooth_next, self.mp, tp)
        ts_exact = exact_positive_debt_tax_shield(b_next, rd_exact, s_exact_next, self.mp)

        b_weight = debt_region_weight(b_next, tp.kappa_b)
        hard_debt = tf.cast(b_next > 0.0, tf.float32)
        debt_mask = _to_numpy(hard_debt) > 0.5
        qd_exact_np = _to_numpy(qd_exact)
        b_next_np = _to_numpy(b_next)
        p_s_exact_np = _to_numpy(p_s_exact)
        er_d_exact_np = _to_numpy(er_d_exact)
        zp_res = (1.0 + self.mp.r) * b_next_np - er_d_exact_np - np.divide(
            p_s_exact_np * b_next_np,
            np.maximum(qd_exact_np, 1e-12),
        )
        zp_pos = zp_res[debt_mask]

        reward_smooth = eval_test_reward(policy, qnet, self.mp, tp, cfg.reward_seed, vtilde=vtilde, mode="smooth")
        reward_exact = eval_test_reward(policy, qnet, self.mp, tp, cfg.reward_seed, vtilde=vtilde, mode="exact")

        q_smooth_np = _to_numpy(q_smooth)
        q_exact_np = _to_numpy(q_exact)
        rd_smooth_np = _to_numpy(rd_smooth)
        rd_exact_np = _to_numpy(rd_exact)
        vt_next_np = _to_numpy(vt_next)
        smooth_default_np = 1.0 - _to_numpy(s_smooth_next)
        exact_default_np = 1.0 - _to_numpy(s_exact_next)
        smooth_debt_np = _to_numpy(b_weight)
        exact_debt_np = _to_numpy(hard_debt)
        k_next_np = _to_numpy(k_next)
        e0_exact_np = _to_numpy(e0_exact)
        issuance_size = np.maximum(-e0_exact_np, 0.0)
        eta_exact_np = _to_numpy(issuance_eta_exact(e0_exact, self.mp.eta0, self.mp.eta1))
        ts_exact_np = _to_numpy(ts_exact)
        spread_exact = rd_exact_np - float(self.mp.r)
        spread_pos = spread_exact[debt_mask]
        residuals = self._residual_vectors(artifacts, tp, k, b, z, z_next)
        Gk_np = residuals["Gk"]
        Gb_np = residuals["Gb"]
        R_stat_k_np = residuals["R_stat_k"]
        R_comp_k_np = residuals["R_comp_k"]
        R_bprime_np = residuals["R_bprime"]
        R_bell_s_np = residuals["bellman_residual_smooth"]
        R_bell_e_np = residuals["bellman_residual_exact"]
        R_def_np = residuals["default_residual"]

        lower_b_hit = np.mean(b_next_np <= self.mp.b_min + cfg.boundary_tol)
        upper_b_hit = np.mean(b_next_np >= self.mp.b_max - cfg.boundary_tol)
        q_low_hit = np.mean(q_exact_np <= self.mp.q_min + cfg.boundary_tol)
        q_high_hit = np.mean(q_exact_np >= self.mp.q_max - cfg.boundary_tol)
        k_floor_hit = np.mean(k_next_np <= self.mp.k_min + cfg.boundary_tol)
        debt_discrepancy = np.mean(np.abs(smooth_debt_np - exact_debt_np))
        default_discrepancy = np.mean(np.abs(smooth_default_np - exact_default_np))

        metrics: Dict[str, float] = {
            "objective": name,
            "test_reward_smooth": float(reward_smooth),
            "test_reward_exact": float(reward_exact),
            "delta_test_reward_smooth_minus_exact": float(reward_smooth - reward_exact),
            "default_frequency_smooth": _safe_mean(smooth_default_np),
            "default_frequency_exact": _safe_mean(exact_default_np),
            "default_classification_discrepancy": float(default_discrepancy),
            "positive_debt_frequency_smooth": _safe_mean(smooth_debt_np),
            "positive_debt_frequency_exact": _safe_mean(exact_debt_np),
            "debt_region_discrepancy": float(debt_discrepancy),
            "mean_q_smooth": _safe_mean(q_smooth_np),
            "mean_q_exact": _safe_mean(q_exact_np),
            "mean_risky_rate_exact": _safe_mean(rd_exact_np[debt_mask]),
            "mean_credit_spread_exact_pos_debt": _safe_mean(spread_pos),
            "rmse_zero_profit_pos_debt_exact": _safe_rmse(zp_pos),
            "pricing_admissibility_penalty_mean": _safe_mean(_to_numpy(p_pen)),
            "capital_bound_frequency": float(k_floor_hit),
            "debt_lower_bound_frequency": float(lower_b_hit),
            "debt_upper_bound_frequency": float(upper_b_hit),
            "debt_bound_frequency": float(lower_b_hit + upper_b_hit),
            "q_lower_bound_frequency": float(q_low_hit),
            "q_upper_bound_frequency": float(q_high_hit),
            "q_bound_frequency": float(q_low_hit + q_high_hit),
            "external_equity_frequency_exact": float(np.mean(e0_exact_np < 0.0)),
            "external_equity_size_mean_exact": _safe_mean(issuance_size[issuance_size > 0.0]),
            "issuance_cost_mean_exact": _safe_mean(-eta_exact_np[eta_exact_np < 0.0]),
            "tax_shield_mean_exact": _safe_mean(ts_exact_np),
            "tax_shield_positive_frequency_exact": float(np.mean(ts_exact_np > 1e-12)),
            "vtilde_near_zero_frequency": float(np.mean(np.abs(vt_next_np) <= cfg.near_zero_band)) if vtilde is not None else float("nan"),
            "bprime_near_zero_frequency": float(np.mean(np.abs(b_next_np) <= cfg.near_zero_band)),
            "mean_kprime": _safe_mean(k_next_np),
            "mean_bprime": _safe_mean(b_next_np),
            "RMSE_E": float(np.sqrt(np.nanmean(np.square(Gk_np) + np.square(Gb_np)))) if Gk_np.size and Gb_np.size else float("nan"),
            "MAE_E": float(np.nanmean(np.abs(Gk_np) + np.abs(Gb_np))) if Gk_np.size and Gb_np.size else float("nan"),
            "Gk_RMSE": _safe_rmse(Gk_np),
            "Gb_RMSE": _safe_rmse(Gb_np),
            "Gk_abs_mean": _safe_mean(np.abs(Gk_np)),
            "Gb_abs_mean": _safe_mean(np.abs(Gb_np)),
            "R_stat_k_RMSE": _safe_rmse(R_stat_k_np),
            "R_bprime_RMSE": _safe_rmse(R_bprime_np),
            "Fischer_Burmeister_RMSE": _safe_rmse(R_comp_k_np),
            "default_residual_RMSE": _safe_rmse(R_def_np),
            "Bellman_RMSE_smooth": _safe_rmse(R_bell_s_np),
            "Bellman_RMSE_exact": _safe_rmse(R_bell_e_np),
            "Bellman_MSE_smooth": float(np.nanmean(np.square(R_bell_s_np))) if R_bell_s_np.size else float("nan"),
            "Bellman_MSE_exact": float(np.nanmean(np.square(R_bell_e_np))) if R_bell_e_np.size else float("nan"),
        }
        metrics.update(_percentiles(k_next_np, "kprime"))
        metrics.update(_percentiles(b_next_np, "bprime"))
        metrics.update(_percentiles(q_exact_np, "q_exact"))
        metrics.update(_percentiles(rd_exact_np[debt_mask], "risky_rate_exact_pos_debt"))
        metrics.update(_percentiles(spread_pos, "credit_spread_exact_pos_debt"))
        metrics.update(_percentiles(ts_exact_np, "tax_shield_exact"))
        metrics.update(_percentiles(issuance_size, "external_equity_size_exact"))
        metrics.update(_percentiles(np.abs(Gk_np), "abs_Gk"))
        metrics.update(_percentiles(np.abs(Gb_np), "abs_Gb"))
        metrics.update(_percentiles(R_bell_s_np, "Bellman_residual_smooth"))
        metrics.update(_percentiles(R_bell_e_np, "Bellman_residual_exact"))

        if artifacts.lambda_k is not None:
            lam = _to_numpy(artifacts.lambda_k(tf.stack([k, b, z], axis=1)))
            comp = lam * (k_next_np - self.mp.k_min)
            metrics["lambda_k_mean"] = _safe_mean(lam)
            metrics["kkt_complementarity_abs_mean"] = _safe_mean(np.abs(comp))
            metrics["kkt_complementarity_rmse"] = _safe_rmse(comp)
        else:
            metrics["lambda_k_mean"] = float("nan")
            metrics["kkt_complementarity_abs_mean"] = float("nan")
            metrics["kkt_complementarity_rmse"] = float("nan")

        vectors = {
            "bprime": b_next_np,
            "kprime": k_next_np,
            "q_smooth": q_smooth_np,
            "q_exact": q_exact_np,
            "risky_rate_exact": rd_exact_np,
            "credit_spread_exact": spread_exact,
            "zp_residual_exact": zp_res,
            "tax_shield_exact": ts_exact_np,
            "issuance_size_exact": issuance_size,
            "vtilde_next": vt_next_np,
            "smooth_default": smooth_default_np,
            "exact_default": exact_default_np,
            "smooth_debt": smooth_debt_np,
            "exact_debt": exact_debt_np,
            "d0_smooth": _to_numpy(d0_smooth),
            "d0_exact": _to_numpy(d0_exact),
            "Gk": Gk_np,
            "Gb": Gb_np,
            "R_stat_k": R_stat_k_np,
            "R_comp_k": R_comp_k_np,
            "R_bprime": R_bprime_np,
            "bellman_residual_smooth": R_bell_s_np,
            "bellman_residual_exact": R_bell_e_np,
            "default_residual": R_def_np,
        }
        return metrics, vectors

    def _write_plots(self, name: str, vectors: Mapping[str, np.ndarray]) -> None:
        """Write objective-specific diagnostic plots."""
        prefix = os.path.join(self.figures_dir, f"solver_{name}")
        _plot_hist(vectors["vtilde_next"], f"{name}: continuation value near default", "Vtilde(k',b',z')", f"{prefix}_vtilde_near_zero_histogram.png")
        _plot_hist(vectors["bprime"], f"{name}: debt choice distribution", "b'", f"{prefix}_bprime_histogram.png")
        _plot_hist(vectors["q_exact"], f"{name}: exact debt price distribution", "q exact", f"{prefix}_q_exact_histogram.png")
        _plot_hist(vectors["credit_spread_exact"], f"{name}: exact credit spread distribution", "rD - r", f"{prefix}_credit_spread_histogram.png")
        _plot_hist(vectors["tax_shield_exact"], f"{name}: exact tax shield distribution", "tax shield", f"{prefix}_tax_shield_histogram.png")
        _plot_hist(vectors["issuance_size_exact"], f"{name}: external equity size distribution", "max(-e0,0)", f"{prefix}_external_equity_histogram.png")
        _plot_hist(vectors["zp_residual_exact"], f"{name}: exact zero-profit residual", "ZP residual", f"{prefix}_zp_residual_histogram.png")
        _plot_scatter(vectors["smooth_default"], vectors["exact_default"], f"{name}: smooth vs exact default", "smooth default weight", "exact default indicator", f"{prefix}_smooth_vs_exact_default.png")
        _plot_scatter(vectors["smooth_debt"], vectors["exact_debt"], f"{name}: smooth vs exact debt region", "smooth debt weight", "exact positive-debt indicator", f"{prefix}_smooth_vs_exact_debt_region.png")
        _plot_scatter(vectors["q_smooth"], vectors["q_exact"], f"{name}: smooth vs exact price", "q smooth", "q exact", f"{prefix}_smooth_vs_exact_q.png")
        _plot_scatter(vectors["kprime"], vectors["bprime"], f"{name}: policy choices", "k'", "b'", f"{prefix}_policy_kprime_bprime.png")
        _plot_hist(vectors["Gk"], f"{name}: capital FOC residual", "G_k", f"{prefix}_Gk_residual_histogram.png")
        _plot_hist(vectors["Gb"], f"{name}: debt FOC residual", "G_b", f"{prefix}_Gb_residual_histogram.png")
        _plot_hist(vectors["R_comp_k"], f"{name}: capital FB residual", "R_comp,k", f"{prefix}_capital_FB_residual_histogram.png")
        _plot_hist(vectors["bellman_residual_smooth"], f"{name}: smooth Bellman residual", "R_B smooth", f"{prefix}_bellman_residual_smooth_histogram.png")
        _plot_hist(vectors["bellman_residual_exact"], f"{name}: exact Bellman residual", "R_B exact", f"{prefix}_bellman_residual_exact_histogram.png")
        _plot_hist(vectors["default_residual"], f"{name}: default complementarity residual", "R_def", f"{prefix}_default_residual_histogram.png")
        _scatter_colored(vectors["kprime"], vectors["bprime"], vectors["exact_default"], f"{name}: default region over policy choices", "k'", "b'", f"{prefix}_default_region_policy_slice.png")
        _scatter_colored(vectors["kprime"], vectors["bprime"], vectors["q_exact"], f"{name}: risky-debt price schedule over choices", "k'", "b'", f"{prefix}_price_schedule_policy_slice.png")
        _scatter_colored(vectors["kprime"], vectors["bprime"], vectors["kprime"], f"{name}: capital policy surface proxy", "k'", "b'", f"{prefix}_capital_policy_surface_proxy.png")
        _scatter_colored(vectors["kprime"], vectors["bprime"], vectors["bprime"], f"{name}: debt policy surface proxy", "k'", "b'", f"{prefix}_debt_policy_surface_proxy.png")

    def write(self, artifacts_by_name: Mapping[str, object]) -> Dict[str, Dict[str, float]]:
        """Compute diagnostics, save tables/plots, and return metrics by objective."""
        cfg = SolverDiagnosticConfig()
        rows = []
        metrics_by_name: Dict[str, Dict[str, float]] = {}
        vectors_by_name: Dict[str, Dict[str, np.ndarray]] = {}
        for name, artifacts in artifacts_by_name.items():
            metrics, vectors = self._diagnose_one(name, artifacts, cfg)
            rows.append(metrics)
            metrics_by_name[name] = metrics
            vectors_by_name[name] = vectors
            self._write_plots(name, vectors)

        summary_columns = [
            "objective",
            "test_reward_smooth",
            "test_reward_exact",
            "delta_test_reward_smooth_minus_exact",
            "default_frequency_exact",
            "positive_debt_frequency_exact",
            "mean_credit_spread_exact_pos_debt",
            "rmse_zero_profit_pos_debt_exact",
            "capital_bound_frequency",
            "debt_bound_frequency",
            "q_bound_frequency",
            "external_equity_frequency_exact",
            "tax_shield_mean_exact",
        ]
        boundary_columns = [
            "objective",
            "capital_bound_frequency",
            "debt_lower_bound_frequency",
            "debt_upper_bound_frequency",
            "debt_bound_frequency",
            "q_lower_bound_frequency",
            "q_upper_bound_frequency",
            "q_bound_frequency",
            "bprime_near_zero_frequency",
            "vtilde_near_zero_frequency",
        ]
        pricing_columns = [
            "objective",
            "mean_q_smooth",
            "mean_q_exact",
            "mean_risky_rate_exact",
            "mean_credit_spread_exact_pos_debt",
            "credit_spread_exact_pos_debt_p05",
            "credit_spread_exact_pos_debt_p50",
            "credit_spread_exact_pos_debt_p95",
            "rmse_zero_profit_pos_debt_exact",
            "pricing_admissibility_penalty_mean",
        ]
        policy_columns = [
            "objective",
            "mean_kprime",
            "kprime_p05",
            "kprime_p50",
            "kprime_p95",
            "mean_bprime",
            "bprime_p05",
            "bprime_p50",
            "bprime_p95",
            "external_equity_frequency_exact",
            "external_equity_size_mean_exact",
            "tax_shield_mean_exact",
            "tax_shield_exact_p50",
            "tax_shield_exact_p95",
        ]
        exact_columns = [
            "objective",
            "test_reward_smooth",
            "test_reward_exact",
            "delta_test_reward_smooth_minus_exact",
            "default_frequency_smooth",
            "default_frequency_exact",
            "default_classification_discrepancy",
            "positive_debt_frequency_smooth",
            "positive_debt_frequency_exact",
            "debt_region_discrepancy",
        ]
        residual_columns = [
            "objective",
            "RMSE_E",
            "MAE_E",
            "Gk_RMSE",
            "Gb_RMSE",
            "abs_Gk_p50",
            "abs_Gk_p95",
            "abs_Gb_p50",
            "abs_Gb_p95",
            "R_stat_k_RMSE",
            "R_bprime_RMSE",
            "Fischer_Burmeister_RMSE",
            "default_residual_RMSE",
            "Bellman_RMSE_smooth",
            "Bellman_RMSE_exact",
            "Bellman_MSE_smooth",
            "Bellman_MSE_exact",
        ]

        table_specs = [
            ("solver_effectiveness_summary", summary_columns, "Solver effectiveness summary"),
            ("solver_exact_vs_smooth", exact_columns, "Smooth versus exact economic evaluation"),
            ("solver_boundary_diagnostics", boundary_columns, "Boundary and near-threshold diagnostics"),
            ("solver_pricing_diagnostics", pricing_columns, "Risky-debt pricing diagnostics"),
            ("solver_policy_distribution", policy_columns, "Policy and payout distribution diagnostics"),
            ("solver_residual_diagnostics", residual_columns, "Euler/KKT/Bellman/default residual diagnostics"),
        ]
        for stem, columns, caption in table_specs:
            _write_csv(os.path.join(self.tables_dir, f"{stem}.csv"), rows, columns)
            _write_tex(os.path.join(self.tables_dir, f"{stem}.tex"), rows, columns, caption)

        with open(os.path.join(self.tables_dir, "solver_diagnostics.json"), "w", encoding="utf-8") as handle:
            json.dump(make_json_serializable(metrics_by_name), handle, indent=2)

        names = [row["objective"] for row in rows]
        _plot_bar(names, [float(row["test_reward_smooth"]) for row in rows], "Smooth test reward across objectives", "TestReward smooth", os.path.join(self.figures_dir, "solver_smooth_test_reward_comparison.png"))
        _plot_bar(names, [float(row["test_reward_exact"]) for row in rows], "Exact test reward across objectives", "TestReward exact", os.path.join(self.figures_dir, "solver_exact_test_reward_comparison.png"))
        _plot_bar(names, [float(row["default_frequency_exact"]) for row in rows], "Exact default frequency across objectives", "Default frequency", os.path.join(self.figures_dir, "solver_exact_default_frequency_comparison.png"))
        _plot_bar(names, [float(row["positive_debt_frequency_exact"]) for row in rows], "Exact positive-debt frequency across objectives", "Positive-debt frequency", os.path.join(self.figures_dir, "solver_exact_positive_debt_frequency_comparison.png"))
        _plot_bar(names, [float(row["mean_credit_spread_exact_pos_debt"]) for row in rows], "Exact mean credit spread across objectives", "Mean credit spread", os.path.join(self.figures_dir, "solver_exact_credit_spread_comparison.png"))
        return metrics_by_name

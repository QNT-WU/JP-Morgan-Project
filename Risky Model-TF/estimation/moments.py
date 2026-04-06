"""estimation.moments

Moment construction for SMM / diagnostics.

"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Tuple

import numpy as np

from risky_debt.config import ModelParams


def _safe_log(x: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Return a numerically safe version of og."""
    return np.log(np.maximum(x, floor))


def _autocorr_lag1(x: np.ndarray) -> float:
    """Lag-1 autocorrelation, robust to near-constant series."""
    x = np.asarray(x).reshape(-1)
    if x.size < 3:
        return float("nan")
    x0 = x[:-1]
    x1 = x[1:]
    v0 = np.var(x0)
    v1 = np.var(x1)
    if v0 < 1e-14 or v1 < 1e-14:
        return 0.0
    return float(np.corrcoef(x0, x1)[0, 1])


def compute_default_moment_vector(
    dataset: Dict[str, np.ndarray],
    mp: ModelParams,
    include_risky_debt_moments: bool = True,
) -> Tuple[List[str], np.ndarray]:
    # Compute the moment vector m(D) used by SMM.

    """Compute the default SMM moment vector from one synthetic dataset."""
    k = np.asarray(dataset["k"], dtype=float)
    b = np.asarray(dataset["b"], dtype=float)
    z = np.asarray(dataset["z"], dtype=float)
    I = np.asarray(dataset["I"], dtype=float)
    q = np.asarray(dataset["q"], dtype=float)
    spread = np.asarray(dataset["spread"], dtype=float)
    default = np.asarray(dataset["default"], dtype=float)

    lnz = _safe_log(z)
    inv = I / np.maximum(k, mp.k_min)

    # --- Moment scaling guard ---
    # Some datasets store I/k in percent units.
    # To keep SMM comparable across 'data' and 'sim', we map large mean(I/k)
    # back to fraction units by dividing by 100.
    inv_mean = float(np.nanmean(inv)) if inv.size > 0 else 0.0
    if np.isfinite(inv_mean) and inv_mean > 2.0:
        inv = inv / 100.0
    prof_over_k = z * (np.maximum(k, mp.k_min) ** (mp.theta - 1.0))
    lev = b / np.maximum(k, mp.k_min)

    names: List[str] = []
    vals: List[float] = []

    # --- Shock moments (identify rho, sigma_eps) ---
    names += ["mean_lnz", "var_lnz", "ac1_lnz"]
    vals += [float(np.mean(lnz)), float(np.var(lnz)), _autocorr_lag1(lnz)]

    # --- Investment moments (identify theta, psi0) ---
    names += ["mean_I_over_k", "var_I_over_k", "ac1_I_over_k"]
    vals += [float(np.mean(inv)), float(np.var(inv)), _autocorr_lag1(inv)]

    # --- Profit/productivity moments (help identify theta) ---
    names += ["mean_pi_over_k", "var_pi_over_k"]
    vals += [float(np.mean(prof_over_k)), float(np.var(prof_over_k))]

    if include_risky_debt_moments:
        # --- Risky-debt add-ons (identify alpha and discipline credit risk) ---
        names += ["mean_b_over_k", "var_b_over_k", "prob_b_pos"]
        vals += [
            float(np.mean(lev)),
            float(np.var(lev)),
            float(np.mean((b > 0.0).astype(float))),
        ]

        names += ["default_rate", "mean_q", "mean_spread"]
        vals += [float(np.mean(default)), float(np.mean(q)), float(np.mean(spread))]

    return names, np.asarray(vals, dtype=float)


def moment_distance(
    m_data: np.ndarray,
    m_sim: np.ndarray,
    W: np.ndarray | None = None,
) -> float:
    """Quadratic form (m_data - m_sim)' W (m_data - m_sim)."""
    dm = np.asarray(m_data, dtype=float) - np.asarray(m_sim, dtype=float)
    if W is None:
        return float(dm @ dm)
    W = np.asarray(W, dtype=float)
    return float(dm @ W @ dm)

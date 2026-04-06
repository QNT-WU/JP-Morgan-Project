"""Covariance and weighting-matrix utilities for GMM and SMM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CovarianceEstimate:
    """Store a covariance estimate and its conditioning diagnostics."""
    method: str
    covariance: np.ndarray
    weight_matrix: np.ndarray
    lags: int
    ridge: float
    condition_number: float
    min_eigenvalue: float


def _validate_series(moment_series: np.ndarray) -> np.ndarray:
    """Validate that the moment series is two-dimensional and finite."""
    x = np.asarray(moment_series, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("moment_series must be 2D [T,q]")
    if x.shape[0] < 2:
        raise ValueError("moment_series must have at least two time observations")
    return x


def choose_newey_west_lags(T: int) -> int:
    """Choose a Newey--West truncation lag from the sample length."""
    if T <= 2:
        return 1
    return max(1, int(round(4.0 * (T / 100.0) ** (2.0 / 9.0))))


def estimate_standard_covariance(moment_series: np.ndarray) -> np.ndarray:
    """Estimate the contemporaneous covariance matrix of moment series."""
    x = _validate_series(moment_series)
    xc = x - x.mean(axis=0, keepdims=True)
    return (xc.T @ xc) / float(x.shape[0])


def estimate_newey_west_covariance(moment_series: np.ndarray, lags: Optional[int] = None) -> np.ndarray:
    """Estimate a Newey--West covariance matrix for moment series."""
    x = _validate_series(moment_series)
    T = x.shape[0]
    xc = x - x.mean(axis=0, keepdims=True)
    L = choose_newey_west_lags(T) if lags is None else int(max(1, lags))
    S = (xc.T @ xc) / float(T)
    for ell in range(1, min(L, T - 1) + 1):
        weight = 1.0 - (ell / float(L + 1))
        gamma = (xc[ell:].T @ xc[:-ell]) / float(T)
        S = S + weight * (gamma + gamma.T)
    return S


def invert_weight_matrix(covariance: np.ndarray, ridge: float = 1e-8):
    """Invert a covariance matrix into a weighting matrix with ridge stabilization."""
    S = np.asarray(covariance, dtype=np.float64)
    S = 0.5 * (S + S.T)
    eigvals = np.linalg.eigvalsh(S)
    min_eig = float(np.min(eigvals))
    ridge_used = float(ridge)
    if min_eig < ridge:
        ridge_used = float(ridge - min_eig + ridge)
    S_reg = S + ridge_used * np.eye(S.shape[0], dtype=np.float64)
    W = np.linalg.pinv(S_reg)
    cond = float(np.linalg.cond(S_reg))
    return W, ridge_used, cond, min_eig


def estimate_weighting_matrix(moment_series: np.ndarray, method: str, ridge: float = 1e-8, lags: Optional[int] = None) -> CovarianceEstimate:
    """Estimate a weighting matrix and associated diagnostics."""
    method = str(method).lower().strip()
    if method == "standard":
        cov = estimate_standard_covariance(moment_series)
        used_lags = 0
    elif method in {"newey_west", "hac"}:
        used_lags = choose_newey_west_lags(moment_series.shape[0]) if lags is None else int(max(1, lags))
        cov = estimate_newey_west_covariance(moment_series, lags=used_lags)
        method = "newey_west"
    else:
        raise ValueError(f"Unknown covariance method: {method}")
    W, ridge_used, cond, min_eig = invert_weight_matrix(cov, ridge=ridge)
    return CovarianceEstimate(method, cov, W, used_lags, ridge_used, cond, min_eig)


def numerical_jacobian(func, x: np.ndarray, step: float = 1e-4) -> np.ndarray:
    """Estimate a Jacobian matrix by finite differences."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    f0 = np.asarray(func(x), dtype=np.float64).reshape(-1)
    q, p = f0.shape[0], x.shape[0]
    J = np.zeros((q, p), dtype=np.float64)
    for j in range(p):
        h = step * max(1.0, abs(float(x[j])))
        e = np.zeros_like(x)
        e[j] = h
        fp = np.asarray(func(x + e), dtype=np.float64).reshape(-1)
        fm = np.asarray(func(x - e), dtype=np.float64).reshape(-1)
        J[:, j] = (fp - fm) / (2.0 * h)
    return J


def sandwich_parameter_covariance(jacobian: np.ndarray, covariance_of_moments: np.ndarray, weight_matrix: np.ndarray, n_obs: int, simulation_adjustment: float = 1.0, ridge: float = 1e-8) -> np.ndarray:
    """Compute a sandwich covariance estimate for structural parameters."""
    D = np.asarray(jacobian, dtype=np.float64)
    S = np.asarray(covariance_of_moments, dtype=np.float64)
    W = np.asarray(weight_matrix, dtype=np.float64)
    A = D.T @ W @ D
    A_reg = A + ridge * np.eye(A.shape[0], dtype=np.float64)
    A_inv = np.linalg.pinv(A_reg)
    B = D.T @ W @ S @ W @ D
    V = (float(simulation_adjustment) / float(max(n_obs, 1))) * (A_inv @ B @ A_inv)
    return 0.5 * (V + V.T)

"""Moment construction and CRN-based path simulation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

from ..config import ModelParams
from ..networks import PolicyNet


@dataclass(frozen=True)
class CRNDesign:
    """Store the common-random-number design for simulated paths."""
    init_k: np.ndarray
    init_z: np.ndarray
    eps_std_norm: np.ndarray

    @property
    def n_paths(self) -> int:
        """Return the number of simulated paths stored in ``paths``.

    This small helper keeps downstream reporting code independent from the
    concrete container used to store simulated path arrays.
    """
        return int(self.init_k.shape[0])

    @property
    def T(self) -> int:
        """Return the time dimension represented by this dataset view."""
        return int(self.eps_std_norm.shape[1])


@dataclass(frozen=True)
class PathDataset:
    """Container for simulated state paths and derived panels."""
    k: np.ndarray
    z: np.ndarray

    def __post_init__(self):
        """Validate the path arrays and derive shared dimensional metadata."""
        if self.k.ndim != 2 or self.z.ndim != 2:
            raise ValueError("k and z must be 2D arrays")
        if self.k.shape != self.z.shape:
            raise ValueError("k and z must have same shape")
        if self.k.shape[1] < 3:
            raise ValueError("Need at least three time points")

    @property
    def T(self) -> int:
        """Return the time dimension represented by this dataset view."""
        return int(self.k.shape[1] - 1)

    @property
    def k_curr(self) -> np.ndarray:
        """Return the current-period capital observations."""
        return self.k[:, :-1]

    @property
    def k_next(self) -> np.ndarray:
        """Return the next-period capital observations."""
        return self.k[:, 1:]

    @property
    def z_curr(self) -> np.ndarray:
        """Return the current-period productivity observations."""
        return self.z[:, :-1]

    @property
    def z_next(self) -> np.ndarray:
        """Return the next-period productivity observations."""
        return self.z[:, 1:]


@dataclass(frozen=True)
class MomentSpec:
    """Describe the moments included in the SMM target vector."""
    names: Tuple[str, ...]


def make_crn_design(*, n_paths: int, T: int, seed: int, k0_low: float = 0.5, k0_high: float = 2.0, z0_low: float = 0.5, z0_high: float = 2.0) -> CRNDesign:
    """Create a reproducible common-random-number design."""
    rng = np.random.default_rng(seed)
    return CRNDesign(
        init_k=rng.uniform(k0_low, k0_high, size=(n_paths,)).astype(np.float32),
        init_z=rng.uniform(z0_low, z0_high, size=(n_paths,)).astype(np.float32),
        eps_std_norm=rng.standard_normal(size=(n_paths, T)).astype(np.float32),
    )


def simulate_paths_crn(*, policy: PolicyNet, mp: ModelParams, design: CRNDesign, burn_in: int = 0) -> PathDataset:
    """Simulate paths under a fixed policy and CRN design."""
    if burn_in < 0 or burn_in >= design.T:
        raise ValueError("invalid burn_in")
    k = tf.convert_to_tensor(design.init_k, dtype=tf.float32)
    z = tf.convert_to_tensor(design.init_z, dtype=tf.float32)
    eps_std = tf.convert_to_tensor(design.eps_std_norm, dtype=tf.float32)
    ks, zs = [k], [z]
    rho = tf.constant(mp.rho, tf.float32)
    sigma = tf.constant(mp.sigma_eps, tf.float32)
    for t in range(design.T):
        x = tf.stack([k, z], axis=1)
        k_next = tf.clip_by_value(policy(x), mp.k_min, mp.k_max)
        z_safe = tf.maximum(z, tf.constant(1e-12, tf.float32))
        z_next = tf.exp(rho * tf.math.log(z_safe) + sigma * eps_std[:, t])
        k, z = k_next, z_next
        ks.append(k)
        zs.append(z)
    k_path = tf.stack(ks, axis=1).numpy().astype(np.float32)
    z_path = tf.stack(zs, axis=1).numpy().astype(np.float32)
    if burn_in > 0:
        k_path = k_path[:, burn_in:]
        z_path = z_path[:, burn_in:]
    return PathDataset(k=k_path, z=z_path)


def build_default_moment_spec() -> MomentSpec:
    """Build the default set of SMM target moments."""
    return MomentSpec(names=(
        "mean_I_over_k",
        "mean_I_over_k_sq",
        "mean_lag_I_over_k_prod",
        "mean_profit",
        "mean_profit_sq",
        "mean_lag_profit_prod",
        "mean_I_over_k_profit",
        "mean_dlogk",
    ))


def make_identity_weight_matrix(J: int) -> np.ndarray:
    """Return an identity weighting matrix with the requested dimension."""
    return np.eye(int(J), dtype=np.float64)


def _profit_series(ds: PathDataset, mp: ModelParams) -> np.ndarray:
    """Compute the flattened profitability series used in SMM moment formulas."""
    return ds.z_curr * np.power(np.maximum(ds.k_curr, mp.k_min), mp.theta)


def _investment_ratio_series(ds: PathDataset, mp: ModelParams) -> np.ndarray:
    """Compute the flattened investment-rate series ``I_t / k_t``."""
    I = ds.k_next - (1.0 - mp.delta) * ds.k_curr
    return I / np.maximum(ds.k_curr, mp.k_min)


def _dlogk_series(ds: PathDataset, mp: ModelParams) -> np.ndarray:
    """Compute the flattened capital-growth series in log differences."""
    return np.log(np.maximum(ds.k_next, mp.k_min)) - np.log(np.maximum(ds.k_curr, mp.k_min))


def compute_smm_moment_series(ds: PathDataset, mp: ModelParams, spec: Optional[MomentSpec] = None) -> np.ndarray:
    """Compute the time-series contributions for SMM data moments."""
    spec = spec or build_default_moment_spec()
    I_over_k = _investment_ratio_series(ds, mp)
    profit = _profit_series(ds, mp)
    dlogk = _dlogk_series(ds, mp)
    I_t, I_lag = I_over_k[:, 1:], I_over_k[:, :-1]
    p_t, p_lag = profit[:, 1:], profit[:, :-1]
    dlogk_t = dlogk[:, 1:]
    contrib = {
        "mean_I_over_k": I_t,
        "mean_I_over_k_sq": I_t ** 2,
        "mean_lag_I_over_k_prod": I_t * I_lag,
        "mean_profit": p_t,
        "mean_profit_sq": p_t ** 2,
        "mean_lag_profit_prod": p_t * p_lag,
        "mean_I_over_k_profit": I_t * p_t,
        "mean_dlogk": dlogk_t,
    }
    cols = [np.mean(contrib[name], axis=0) for name in spec.names]
    return np.stack(cols, axis=1).astype(np.float64)


def compute_moments(ds: PathDataset, mp: ModelParams, spec: Optional[MomentSpec] = None) -> Dict[str, float]:
    """Compute the configured sample moments for a path dataset."""
    spec = spec or build_default_moment_spec()
    series = compute_smm_moment_series(ds, mp, spec)
    vec = np.mean(series, axis=0)
    return {name: float(val) for name, val in zip(spec.names, vec)}


def summarize_smm_moments(moment_dict: Dict[str, float]) -> Dict[str, float]:
    """Summarize observed and simulated SMM moments."""
    mu_i = float(moment_dict["mean_I_over_k"])
    e_i2 = float(moment_dict["mean_I_over_k_sq"])
    e_ii = float(moment_dict["mean_lag_I_over_k_prod"])
    mu_p = float(moment_dict["mean_profit"])
    e_p2 = float(moment_dict["mean_profit_sq"])
    e_pp = float(moment_dict["mean_lag_profit_prod"])
    e_ip = float(moment_dict["mean_I_over_k_profit"])
    mu_dlogk = float(moment_dict["mean_dlogk"])
    var_i = max(e_i2 - mu_i**2, 0.0)
    var_p = max(e_p2 - mu_p**2, 0.0)
    std_i = float(np.sqrt(var_i))
    std_p = float(np.sqrt(var_p))
    acf_i = float((e_ii - mu_i**2) / max(var_i, 1e-12))
    acf_p = float((e_pp - mu_p**2) / max(var_p, 1e-12))
    corr_ip = float((e_ip - mu_i * mu_p) / max(std_i * std_p, 1e-12))
    return {
        "mean_I_over_k": mu_i,
        "std_I_over_k": std_i,
        "acf_I_over_k": acf_i,
        "mean_profit": mu_p,
        "std_profit": std_p,
        "acf_profit": acf_p,
        "corr_I_over_k_profit": corr_ip,
        "mean_dlogk": mu_dlogk,
    }


def compute_gmm_moment_series(ds: PathDataset, beta: float, theta: float, psi0: float, delta: float) -> np.ndarray:
    """Compute the per-period GMM moment series."""
    k_t = ds.k[:, :-2]
    k1 = ds.k[:, 1:-1]
    k2 = ds.k[:, 2:]
    z_t = ds.z[:, :-2]
    z1 = ds.z[:, 1:-1]
    I_t = k1 - (1.0 - delta) * k_t
    I1 = k2 - (1.0 - delta) * k1
    left = 1.0 + psi0 * (I_t / np.maximum(k_t, 1e-12))
    term_prod = theta * z1 * np.power(np.maximum(k1, 1e-12), theta - 1.0)
    term_depr = (1.0 - delta) * (1.0 + psi0 * (I1 / np.maximum(k1, 1e-12)))
    term_adj = 0.5 * psi0 * (I1 ** 2) / np.maximum(k1, 1e-12) ** 2
    residual = beta * (term_prod + term_depr + term_adj) - left
    instruments = np.stack([
        np.ones_like(k_t),
        np.log(np.maximum(k_t, 1e-12)),
        np.log(np.maximum(z_t, 1e-12)),
        I_t / np.maximum(k_t, 1e-12),
    ], axis=2)
    return np.mean(residual[:, :, None] * instruments, axis=0).astype(np.float64)


def compute_gmm_moment_vector(ds: PathDataset, beta: float, theta: float, psi0: float, delta: float) -> np.ndarray:
    """Compute the sample-average GMM moment vector."""
    return np.mean(compute_gmm_moment_series(ds, beta, theta, psi0, delta), axis=0)


def path_sample_size(ds: PathDataset) -> int:
    """Return the effective sample size represented by a path dataset."""
    return int(ds.k.shape[0] * (ds.k.shape[1] - 1))

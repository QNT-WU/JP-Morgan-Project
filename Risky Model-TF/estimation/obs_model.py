"""Observation-model helpers for Bayesian risky-debt estimation.

The Bayesian block uses a measurement-error-augmented working likelihood. For a
candidate structural parameter vector and a latent productivity state on the
fixed Rouwenhorst grid, this module constructs emission log densities for the
observed series

.. math::
    y_t = (k_t, b_t, k_{t+1}, b_{t+1}, q_t, d_t, D_{t+1}).

The Gaussian pieces keep the likelihood smooth while the Bernoulli block tracks
continuation/default behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class ObservationNoiseConfig:
    """Fixed measurement-noise scales for the working likelihood.

    Attributes:
        sigma_k: Standard deviation for ``k_{t+1}``.
        sigma_b: Standard deviation for ``b_{t+1}``.
        sigma_q: Standard deviation for ``q_t``.
        sigma_d: Standard deviation for ``d_t``.
        kappa_value: Smoothing scale used to transform continuation values into
            continuation probabilities.
        kappa_issue: Smoothing scale used in the issuance-cost trigger.
    """

    sigma_k: float = 0.15
    sigma_b: float = 0.20
    sigma_q: float = 0.03
    sigma_d: float = 0.20
    kappa_value: float = 0.05
    kappa_issue: float = 0.02


@dataclass(frozen=True)
class PathObservations:
    """Single ordered path used by the finite-state forward filter."""

    k: np.ndarray
    b: np.ndarray
    k_next: np.ndarray
    b_next: np.ndarray
    q: np.ndarray
    d: np.ndarray
    default: np.ndarray


@dataclass(frozen=True)
class PanelObservations:
    """Panel of simulated paths available to the Bayesian estimator.

    The synthetic-data generators used elsewhere in the package usually return
    flattened arrays plus ``n_paths`` and ``T_eff`` metadata. Bayesian
    estimation should not silently discard information, so the panel wrapper
    keeps all available paths and lets the caller choose explicit truncation via
    ``subset``.
    """

    k: np.ndarray
    b: np.ndarray
    k_next: np.ndarray
    b_next: np.ndarray
    q: np.ndarray
    d: np.ndarray
    default: np.ndarray

    @property
    def n_paths(self) -> int:
        """Number of simulated paths in the panel."""
        return int(self.k.shape[0])

    @property
    def t_eff(self) -> int:
        """Number of time observations per path."""
        return int(self.k.shape[1])

    def subset(self, *, max_paths: int = 0, max_obs: int = 0) -> "PanelObservations":
        """Return an explicit truncation of the observation panel.

        Args:
            max_paths: Maximum number of paths to keep. Non-positive values keep
                all paths.
            max_obs: Maximum number of time observations per path. Non-positive
                values keep the full sample length.
        """
        p = self.n_paths if int(max_paths) <= 0 else min(self.n_paths, int(max_paths))
        t = self.t_eff if int(max_obs) <= 0 else min(self.t_eff, int(max_obs))
        return PanelObservations(
            k=self.k[:p, :t].copy(),
            b=self.b[:p, :t].copy(),
            k_next=self.k_next[:p, :t].copy(),
            b_next=self.b_next[:p, :t].copy(),
            q=self.q[:p, :t].copy(),
            d=self.d[:p, :t].copy(),
            default=self.default[:p, :t].copy(),
        )

    def path(self, index: int) -> PathObservations:
        """Return one path as a ``PathObservations`` instance."""
        return PathObservations(
            k=self.k[index].astype(np.float32),
            b=self.b[index].astype(np.float32),
            k_next=self.k_next[index].astype(np.float32),
            b_next=self.b_next[index].astype(np.float32),
            q=self.q[index].astype(np.float32),
            d=self.d[index].astype(np.float32),
            default=self.default[index].astype(np.float32),
        )


def gaussian_logpdf(y: np.ndarray, mu: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian log density evaluated elementwise.

    Args:
        y: Observed values.
        mu: Conditional means.
        sigma: Positive standard deviation.

    Returns:
        Elementwise log density values.
    """
    sigma = float(max(sigma, 1e-8))
    diff = (np.asarray(y, dtype=float) - np.asarray(mu, dtype=float)) / sigma
    return -0.5 * np.log(2.0 * np.pi) - np.log(sigma) - 0.5 * diff * diff


def bernoulli_logpmf(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Bernoulli log mass evaluated elementwise.

    Args:
        y: Observed 0/1 outcomes.
        p: Success probabilities.

    Returns:
        Elementwise log probability values.
    """
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1.0 - 1e-8)
    return y * np.log(p) + (1.0 - y) * np.log(1.0 - p)


def sigmoid(x: np.ndarray, scale: float) -> np.ndarray:
    """Stable sigmoid with explicit scale parameter."""
    scale = float(max(scale, 1e-8))
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float) / scale))


def extract_panel_observations(dataset: Dict[str, np.ndarray]) -> PanelObservations:
    """Extract all simulated paths from a flattened synthetic dataset.

    Args:
        dataset: Flattened dataset produced by the estimation/simulation layer.

    Returns:
        ``PanelObservations`` containing all recoverable paths.
    """
    n_paths = int(np.asarray(dataset.get("n_paths", 1)).reshape(()))
    t_eff = int(np.asarray(dataset.get("T_eff", 0)).reshape(()))

    def _panel(name: str) -> np.ndarray:
        arr = np.asarray(dataset[name], dtype=float).reshape(-1)
        if t_eff <= 0 or n_paths * t_eff != arr.size:
            return arr.reshape(1, -1)
        return arr.reshape(n_paths, t_eff)

    return PanelObservations(
        k=_panel("k").astype(np.float32),
        b=_panel("b").astype(np.float32),
        k_next=_panel("k_next").astype(np.float32),
        b_next=_panel("b_next").astype(np.float32),
        q=_panel("q").astype(np.float32),
        d=_panel("d").astype(np.float32),
        default=_panel("default").astype(np.float32),
    )


def extract_first_path_observations(dataset: Dict[str, np.ndarray]) -> PathObservations:
    """Compatibility wrapper returning the first recovered path only.

    New Bayesian code should prefer :func:`extract_panel_observations` so it can
    use all available simulated paths.
    """
    return extract_panel_observations(dataset).path(0)

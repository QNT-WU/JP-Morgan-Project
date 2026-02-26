"""estimation.obs_model

Observation model for Bayesian likelihood.



We can extend this later by adding likelihood terms for I/k, b/k, q, default.
"""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


@dataclass(frozen=True)
class ObsModelParams:
    """Measurement noise parameters (kept fixed by default)."""

    sigma_lnz: float = 0.02  # std dev of measurement noise on ln z


def log_prob_lnz(y_lnz: tf.Tensor, x_lnz: tf.Tensor, sigma_lnz: tf.Tensor) -> tf.Tensor:
    """log p(y_lnz | x_lnz) under Gaussian measurement noise.

    y_lnz, x_lnz: shape (...,)
    returns: shape (...,)
    """
    return tfd.Normal(loc=x_lnz, scale=sigma_lnz).log_prob(y_lnz)

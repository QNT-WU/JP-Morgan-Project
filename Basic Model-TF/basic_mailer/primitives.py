"""Economic primitives for the basic Mailer model."""

from __future__ import annotations

import tensorflow as tf

from .config import ModelParams


def beta_from_r(r: float) -> float:
    """Return the discount factor implied by the risk-free rate ``r``."""
    return 1.0 / (1.0 + r)


@tf.function(reduce_retracing=True)
def shock_next_z(z: tf.Tensor, rho: float, sigma_eps: float) -> tf.Tensor:
    """Simulate next-period productivity from the AR(1) law in logs.

    Args:
        z: Current productivity values.
        rho: Persistence parameter.
        sigma_eps: Innovation standard deviation.
    """
    z = tf.maximum(tf.convert_to_tensor(z, dtype=tf.float32), 1e-12)
    eps = tf.random.normal(tf.shape(z), mean=0.0, stddev=sigma_eps, dtype=tf.float32)
    return tf.exp(rho * tf.math.log(z) + eps)


@tf.function(reduce_retracing=True)
def reward_basic(k: tf.Tensor, z: tf.Tensor, k_next: tf.Tensor, mp: ModelParams) -> tf.Tensor:
    """Compute the one-period reward to shareholders.

    The reward is ``z * k**theta - psi0 * I**2 / (2 * k) - I`` where
    ``I = k_next - (1 - delta) * k``.
    """
    k = tf.maximum(tf.convert_to_tensor(k, dtype=tf.float32), mp.k_min)
    z = tf.maximum(tf.convert_to_tensor(z, dtype=tf.float32), 1e-12)
    k_next = tf.clip_by_value(tf.convert_to_tensor(k_next, dtype=tf.float32), mp.k_min, mp.k_max)
    investment = k_next - (1.0 - mp.delta) * k
    profit = z * tf.pow(k, mp.theta)
    adjustment_cost = mp.psi0 * tf.square(investment) / (2.0 * k)
    return profit - adjustment_cost - investment

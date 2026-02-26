from __future__ import annotations

import tensorflow as tf
from .config import ModelParams

# is where I encode the economic primitives of the basic model:
# discounting, shock process, and one-period reward.


# defined discount factor
# It’s not @tf.function because it’s a simple scalar conversion used to create TF constants later, e.g.
def beta_from_r(r: float) -> float:
    return 1.0 / (1.0 + r)


# Shock transition: 𝑧′=exp⁡(𝜌ln⁡𝑧+𝜀)
@tf.function
def shock_next_z(z: tf.Tensor, rho: float, sigma_eps: float) -> tf.Tensor:
    # If z is shape [N], then eps is also shape [N].
    # One shock per state in the batch.
    eps = tf.random.normal(tf.shape(z), mean=0.0, stddev=sigma_eps, dtype=tf.float32)
    # tf.maximum(z, 1e-12) protects log(z) from log(0) or negative.
    # ensures numerical stability.
    # tf.math.log(...) takes the log
    return tf.exp(rho * tf.math.log(tf.maximum(z, 1e-12)) + eps)


@tf.function
# This returns a vector of rewards, one per element in the batch
def reward_basic(
    k: tf.Tensor, z: tf.Tensor, k_next: tf.Tensor, mp: ModelParams
) -> tf.Tensor:
    """
    r(k,z,k') = z k^theta - psi0*(I^2/(2k)) - I
    I = k' - (1-delta)k
    """
    # Safety clamps
    # If k is very small, it can explode numerically, so enforce:
    # k≥kmin​
    # k′≥kmin​
    k = tf.maximum(k, mp.k_min)
    # k_next = tf.maximum(k_next, mp.k_min)
    # k_next = tf.clip_by_value(policy(x), mp.k_min, mp.k_max)
    k_next = tf.clip_by_value(k_next, mp.k_min, mp.k_max)

    # Compute investment I
    I = k_next - (1.0 - mp.delta) * k
    # Compute production profit, Implements: 𝜋=𝑧𝑘^𝜃
    pi = z * tf.pow(k, mp.theta)
    # Compute adjustment cost
    adj = mp.psi0 * tf.square(I) / (2.0 * k)
    # Return reward
    return pi - adj - I

"""TensorFlow losses and residuals for the three Mailer objectives."""

from __future__ import annotations

import tensorflow as tf

from .config import ModelParams, Obj2Params, Obj3Params, TrainParams
from .evaluation import (
    euler_derivative_residual_with_value,
    euler_f_policy_only,
)
from .networks import MultiplierNet, PolicyNet, ValueNet
from .primitives import beta_from_r, reward_basic


# ---------- Shared KKT helpers ----------
@tf.function
def fb_residual(slack: tf.Tensor, multiplier: tf.Tensor) -> tf.Tensor:
    """Compute the Fischer--Burmeister complementarity residual.

    The residual is zero if and only if ``slack >= 0``, ``multiplier >= 0``,
    and ``slack * multiplier = 0``. In this project the slack is always
    ``k_next - k_min``.
    """
    slack = tf.convert_to_tensor(slack, dtype=tf.float32)
    multiplier = tf.convert_to_tensor(multiplier, dtype=tf.float32)
    return slack + multiplier - tf.sqrt(tf.square(slack) + tf.square(multiplier) + 1e-12)


@tf.function
def policy_slack_and_multiplier(
    policy: PolicyNet,
    multiplier: MultiplierNet,
    mp: ModelParams,
    k: tf.Tensor,
    z: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return policy choice, lower-bound slack, and multiplier at states."""
    k = tf.maximum(tf.convert_to_tensor(k, dtype=tf.float32), mp.k_min)
    z = tf.maximum(tf.convert_to_tensor(z, dtype=tf.float32), 1e-12)
    x = tf.stack([k, z], axis=1)
    k_next = tf.clip_by_value(policy(x), mp.k_min, mp.k_max)
    slack = k_next - tf.constant(mp.k_min, dtype=tf.float32)
    lam = multiplier(x)
    return k_next, slack, lam


# ---------- Objective 1 ----------
@tf.function
def obj1_loss(
    policy: PolicyNet, mp: ModelParams, tp: TrainParams
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return the negative truncated lifetime reward and the reward itself."""
    n_paths = tp.N_paths_train
    k0 = tf.random.uniform((n_paths,), 0.5, 2.0, dtype=tf.float32)
    z0 = tf.random.uniform((n_paths,), 0.5, 2.0, dtype=tf.float32)

    beta = tf.constant(beta_from_r(mp.r), tf.float32)
    k = tf.maximum(k0, mp.k_min)
    z = tf.maximum(z0, 1e-12)
    reward_sum = tf.zeros_like(k)
    discount = tf.constant(1.0, tf.float32)

    for _ in tf.range(tp.T_train):
        x = tf.stack([k, z], axis=1)
        k_next = tf.clip_by_value(policy(x), mp.k_min, mp.k_max)
        reward_sum = reward_sum + discount * reward_basic(k, z, k_next, mp)
        eps = tf.random.normal(tf.shape(z), mean=0.0, stddev=mp.sigma_eps, dtype=tf.float32)
        z = tf.exp(mp.rho * tf.math.log(z) + eps)
        k = k_next
        discount = discount * beta

    train_reward = tf.reduce_mean(reward_sum)
    return -train_reward, train_reward


# ---------- Objective 2 ----------
@tf.function
def obj2_stationarity_residual(
    policy: PolicyNet,
    multiplier: MultiplierNet,
    mp: ModelParams,
    k: tf.Tensor,
    z: tf.Tensor,
    eps: tf.Tensor,
) -> tf.Tensor:
    """Return ``R_lambda^(2) = lambda(k,z) - f(k,z,eps)``."""
    k = tf.maximum(tf.convert_to_tensor(k, dtype=tf.float32), mp.k_min)
    z = tf.maximum(tf.convert_to_tensor(z, dtype=tf.float32), 1e-12)
    x = tf.stack([k, z], axis=1)
    lam = multiplier(x)
    wedge = euler_f_policy_only(policy, mp, k, z, eps)
    return lam - wedge


@tf.function
def obj2_batch_loss(
    policy: PolicyNet,
    multiplier: MultiplierNet,
    mp: ModelParams,
    op2: Obj2Params,
    k: tf.Tensor,
    z: tf.Tensor,
) -> tf.Tensor:
    """Compute Objective 2's KKT/Euler AiO loss.

    The loss matches the Basic Model document:
    ``R_FB^(2)^2 + nu_lambda R_lambda^(2)(eps1) R_lambda^(2)(eps2)``.
    """
    eps1 = tf.random.normal(tf.shape(k), mean=0.0, stddev=mp.sigma_eps, dtype=tf.float32)
    eps2 = tf.random.normal(tf.shape(k), mean=0.0, stddev=mp.sigma_eps, dtype=tf.float32)
    _, slack, lam = policy_slack_and_multiplier(policy, multiplier, mp, k, z)
    r_fb = fb_residual(slack, lam)
    r_lam1 = obj2_stationarity_residual(policy, multiplier, mp, k, z, eps1)
    r_lam2 = obj2_stationarity_residual(policy, multiplier, mp, k, z, eps2)
    return tf.reduce_mean(tf.square(r_fb) + op2.nu_lambda * r_lam1 * r_lam2)


# ---------- Objective 3 ----------
@tf.function
def bellman_residual(
    policy: PolicyNet,
    value: ValueNet,
    mp: ModelParams,
    k: tf.Tensor,
    z: tf.Tensor,
    eps: tf.Tensor,
) -> tf.Tensor:
    """Compute the one-shock Bellman residual for Objective 3."""
    beta = tf.constant(beta_from_r(mp.r), tf.float32)
    k = tf.maximum(tf.convert_to_tensor(k, dtype=tf.float32), mp.k_min)
    z = tf.maximum(tf.convert_to_tensor(z, dtype=tf.float32), 1e-12)
    z_next = tf.exp(mp.rho * tf.math.log(z) + eps)
    x = tf.stack([k, z], axis=1)
    k_next = tf.clip_by_value(policy(x), mp.k_min, mp.k_max)
    r = reward_basic(k, z, k_next, mp)
    v = value(x)
    v_next = value(tf.stack([k_next, z_next], axis=1))
    return v - (r + beta * v_next)


@tf.function
def obj3_stationarity_residual(
    policy: PolicyNet,
    value: ValueNet,
    multiplier: MultiplierNet,
    mp: ModelParams,
    k: tf.Tensor,
    z: tf.Tensor,
    eps: tf.Tensor,
) -> tf.Tensor:
    """Return ``R_lambda^(3) = lambda(k,z) + R_E(k,z,eps)``."""
    k = tf.maximum(tf.convert_to_tensor(k, dtype=tf.float32), mp.k_min)
    z = tf.maximum(tf.convert_to_tensor(z, dtype=tf.float32), 1e-12)
    lam = multiplier(tf.stack([k, z], axis=1))
    derivative_residual = euler_derivative_residual_with_value(policy, value, mp, k, z, eps)
    return lam + derivative_residual


@tf.function
def obj3_batch_loss(
    policy: PolicyNet,
    value: ValueNet,
    multiplier: MultiplierNet,
    mp: ModelParams,
    op3: Obj3Params,
    k: tf.Tensor,
    z: tf.Tensor,
) -> tf.Tensor:
    """Compute Objective 3's Bellman/KKT AiO loss."""
    eps1 = tf.random.normal(tf.shape(k), mean=0.0, stddev=mp.sigma_eps, dtype=tf.float32)
    eps2 = tf.random.normal(tf.shape(k), mean=0.0, stddev=mp.sigma_eps, dtype=tf.float32)
    rb1 = bellman_residual(policy, value, mp, k, z, eps1)
    rb2 = bellman_residual(policy, value, mp, k, z, eps2)
    _, slack, lam = policy_slack_and_multiplier(policy, multiplier, mp, k, z)
    r_fb = fb_residual(slack, lam)
    r_lam1 = obj3_stationarity_residual(policy, value, multiplier, mp, k, z, eps1)
    r_lam2 = obj3_stationarity_residual(policy, value, multiplier, mp, k, z, eps2)
    return tf.reduce_mean(rb1 * rb2 + op3.nu_fb * tf.square(r_fb) + op3.nu_lambda * r_lam1 * r_lam2)

"""TensorFlow losses and residuals for the three Mailer objectives."""

from __future__ import annotations

import tensorflow as tf

from .config import ModelParams, TrainParams, Obj3Params
from .networks import PolicyNet, ValueNet
from .primitives import beta_from_r, reward_basic
from .evaluation import euler_f_policy_only, euler_residual_with_value_derivative


# ---------- Objective 1 ----------
@tf.function
def obj1_loss(
    policy: PolicyNet, mp: ModelParams, tp: TrainParams
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Loss = -TrainReward
    TrainReward = mean over rollout paths of discounted sum.
    """
    N = tp.N_paths_train  # Create N independent rollout paths.
    k0 = tf.random.uniform((N,), 0.5, 2.0, dtype=tf.float32)
    z0 = tf.random.uniform((N,), 0.5, 2.0, dtype=tf.float32)
    # k0,z0 each shape [N].
    # Uniform sampling in [0.5,2.0] is your “broad initial distribution”.

    # rollout (inline for speed)
    beta = tf.constant(beta_from_r(mp.r), tf.float32)
    # beta is scalar TF tensor.
    k = tf.maximum(k0, mp.k_min)

    z = tf.maximum(z0, 1e-12)
    # clamp k and z for stability.

    W = tf.zeros_like(k)
    # W accumulates discounted reward for each path → shape [N].
    disc = tf.constant(1.0, tf.float32)
    # disc starts at 1 and becomes 𝛽^𝑡

    # This means you compute the discounted sum from 𝑡=0 to 𝑇
    for _ in tf.range(tp.T_train + 1):
        # Inside each step:
        x = tf.stack([k, z], axis=1)  # x has shape [N,2]
        # k_next = tf.maximum(policy(x), mp.k_min)  # policy(x) returns [N]

        k_next = tf.clip_by_value(policy(x), mp.k_min, mp.k_max)
        r_t = reward_basic(k, z, k_next, mp)  # clamp output again

        W = W + disc * r_t
        # reward per path: shape [N]
        # Update W

        # eps shape [N]
        eps = tf.random.normal(
            tf.shape(z), mean=0.0, stddev=mp.sigma_eps, dtype=tf.float32
        )
        z_next = tf.exp(mp.rho * tf.math.log(z) + eps)
        # implements AR(1) in logs

        # Advance and update discount
        k, z = k_next, z_next
        disc = disc * beta

    # Final: average across paths and return loss
    train_reward = tf.reduce_mean(W)
    return -train_reward, train_reward
    # train_reward is scalar
    # loss is negative of it


# ---------- Objective 2 ----------
@tf.function
# batch of states k,z (usually sampled from ergodic buffer)
# both expected shape [N]
def obj2_batch_loss(
    policy: PolicyNet, mp: ModelParams, k: tf.Tensor, z: tf.Tensor
) -> tf.Tensor:
    """
    L2 = E[ f(eps1) f(eps2) ]
    """
    # both shape [N]
    # independent draws

    eps1 = tf.random.normal(
        tf.shape(k), mean=0.0, stddev=mp.sigma_eps, dtype=tf.float32
    )
    eps2 = tf.random.normal(
        tf.shape(k), mean=0.0, stddev=mp.sigma_eps, dtype=tf.float32
    )

    # f1,f2 shape [N]
    f1 = euler_f_policy_only(policy, mp, k, z, eps1)
    f2 = euler_f_policy_only(policy, mp, k, z, eps2)

    # multiply elementwise: f1*f2 shape [N]
    # mean gives scalar
    return tf.reduce_mean(f1 * f2)


# ---------- Objective 3 ----------
# This objective trains policy + value.
@tf.function
# Bellman residual function
def bellman_residual(
    policy: PolicyNet,
    value: ValueNet,
    mp: ModelParams,
    k: tf.Tensor,
    z: tf.Tensor,
    eps: tf.Tensor,
) -> tf.Tensor:

    # Beta and clamps
    """Compute the one-step Bellman residual used by Objective 3."""
    beta = tf.constant(beta_from_r(mp.r), tf.float32)

    k = tf.maximum(k, mp.k_min)
    z = tf.maximum(z, 1e-12)

    # Transition and policy
    z_next = tf.exp(mp.rho * tf.math.log(z) + eps)
    x = tf.stack([k, z], axis=1)
    # k_next = tf.maximum(policy(x), mp.k_min)
    k_next = tf.clip_by_value(policy(x), mp.k_min, mp.k_max)

    # Reward, current value, next value
    # v shape [N]
    # v_next shape [N]
    r = reward_basic(k, z, k_next, mp)

    v = value(x)
    x_next = tf.stack([k_next, z_next], axis=1)
    v_next = value(x_next)

    return v - (r + beta * v_next)


@tf.function
def obj3_batch_loss(
    policy: PolicyNet,
    value: ValueNet,
    mp: ModelParams,
    op3: Obj3Params,
    k: tf.Tensor,
    z: tf.Tensor,
) -> tf.Tensor:
    # Draw two independent shocks:
    """Compute the Objective 3 batch loss."""
    eps1 = tf.random.normal(
        tf.shape(k), mean=0.0, stddev=mp.sigma_eps, dtype=tf.float32
    )
    eps2 = tf.random.normal(
        tf.shape(k), mean=0.0, stddev=mp.sigma_eps, dtype=tf.float32
    )

    # Compute two Bellman residuals
    RB1 = bellman_residual(policy, value, mp, k, z, eps1)
    RB2 = bellman_residual(policy, value, mp, k, z, eps2)
    # Compute Euler residuals that use 𝑑𝑉/𝑑𝑘′

    RE1 = euler_residual_with_value_derivative(policy, value, mp, k, z, eps1)
    RE2 = euler_residual_with_value_derivative(policy, value, mp, k, z, eps2)

    # Return mean of combined AiO:
    return tf.reduce_mean(RB1 * RB2 + op3.nu * (RE1 * RE2))

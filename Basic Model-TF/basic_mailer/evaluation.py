"""Evaluation helpers for rewards and KKT/Euler-residual diagnostics."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from .config import ModelParams, Obj2Params, Obj3Params, TrainParams
from .networks import MultiplierNet, PolicyNet, ValueNet
from .primitives import beta_from_r, reward_basic, shock_next_z


@tf.function
def rollout_discounted_reward(policy: PolicyNet, mp: ModelParams, k0: tf.Tensor, z0: tf.Tensor, T: int) -> tf.Tensor:
    """Compute discounted rewards along simulated policy paths."""
    beta = tf.constant(beta_from_r(mp.r), tf.float32)
    k = tf.maximum(k0, mp.k_min)
    z = tf.maximum(z0, 1e-12)
    reward_sum = tf.zeros_like(k)
    discount = tf.constant(1.0, tf.float32)
    for _ in tf.range(T):
        x = tf.stack([k, z], axis=1)
        k_next = tf.clip_by_value(policy(x), mp.k_min, mp.k_max)
        reward_sum = reward_sum + discount * reward_basic(k, z, k_next, mp)
        z_next = shock_next_z(z, mp.rho, mp.sigma_eps)
        k, z = k_next, z_next
        discount = discount * beta
    return reward_sum


def eval_test_reward(policy: PolicyNet, mp: ModelParams, tp: TrainParams, seed: int) -> float:
    """Evaluate out-of-sample discounted reward for a policy."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    n_paths = tp.N_paths_test
    k0 = tf.random.uniform((n_paths,), 0.5, 2.0, dtype=tf.float32)
    z0 = tf.random.uniform((n_paths,), 0.5, 2.0, dtype=tf.float32)
    return float(tf.reduce_mean(rollout_discounted_reward(policy, mp, k0, z0, tp.T_test)).numpy())


@tf.function
def euler_f_policy_only(policy: PolicyNet, mp: ModelParams, k: tf.Tensor, z: tf.Tensor, eps: tf.Tensor) -> tf.Tensor:
    """Evaluate Objective 2's policy-only Euler wedge ``f``.

    This equals marginal cost minus marginal benefit. Under the lower-bound KKT
    formulation, the conditional expectation of this wedge equals the multiplier.
    """
    beta = tf.constant(beta_from_r(mp.r), tf.float32)
    k = tf.maximum(tf.convert_to_tensor(k, dtype=tf.float32), mp.k_min)
    z = tf.maximum(tf.convert_to_tensor(z, dtype=tf.float32), 1e-12)
    z_next = tf.exp(mp.rho * tf.math.log(z) + eps)
    k1 = tf.clip_by_value(policy(tf.stack([k, z], axis=1)), mp.k_min, mp.k_max)
    k2 = tf.clip_by_value(policy(tf.stack([k1, z_next], axis=1)), mp.k_min, mp.k_max)
    I = k1 - (1.0 - mp.delta) * k
    I1 = k2 - (1.0 - mp.delta) * k1
    left = 1.0 + mp.psi0 * (I / k)
    continuation_marginal_value = (
        mp.theta * z_next * tf.pow(k1, mp.theta - 1.0)
        + (1.0 - mp.delta)
        + mp.psi0 * ((1.0 - mp.delta) * I1) / k1
        + mp.psi0 * tf.square(I1) / (2.0 * tf.square(k1))
    )
    return left - beta * continuation_marginal_value


@tf.function
def fb_residual_from_policy_multiplier(policy: PolicyNet, multiplier: MultiplierNet, mp: ModelParams, k: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
    """Compute the Fischer--Burmeister residual using slack ``k' - k_min``."""
    k = tf.maximum(tf.convert_to_tensor(k, dtype=tf.float32), mp.k_min)
    z = tf.maximum(tf.convert_to_tensor(z, dtype=tf.float32), 1e-12)
    x = tf.stack([k, z], axis=1)
    k_next = tf.clip_by_value(policy(x), mp.k_min, mp.k_max)
    slack = k_next - tf.constant(mp.k_min, dtype=tf.float32)
    lam = multiplier(x)
    return slack + lam - tf.sqrt(tf.square(slack) + tf.square(lam) + 1e-12)


@tf.function
def euler_derivative_residual_with_value(policy: PolicyNet, value: ValueNet, mp: ModelParams, k: tf.Tensor, z: tf.Tensor, eps: tf.Tensor) -> tf.Tensor:
    """Evaluate Objective 3's derivative residual ``R_E = dJ/dk'``.

    This follows the document's sign convention:
    ``R_E = -1 - psi0 * I/k + beta * V_k(k', z')``.
    """
    beta = tf.constant(beta_from_r(mp.r), tf.float32)
    k = tf.maximum(tf.convert_to_tensor(k, dtype=tf.float32), mp.k_min)
    z = tf.maximum(tf.convert_to_tensor(z, dtype=tf.float32), 1e-12)
    z_next = tf.exp(mp.rho * tf.math.log(z) + eps)
    k_next = tf.clip_by_value(policy(tf.stack([k, z], axis=1)), mp.k_min, mp.k_max)
    I = k_next - (1.0 - mp.delta) * k
    with tf.GradientTape() as tape:
        tape.watch(k_next)
        v_next = value(tf.stack([k_next, z_next], axis=1))
        v_sum = tf.reduce_sum(v_next)
    dV_dk = tape.gradient(v_sum, k_next)
    return -1.0 - mp.psi0 * (I / k) + beta * dV_dk


# Backward-compatible alias for older tests/imports. New code should use the
# sign-explicit ``euler_derivative_residual_with_value`` name.
euler_residual_with_value_derivative = euler_derivative_residual_with_value


def _conditional_mean_over_shocks(residual_fn, states_k: np.ndarray, states_z: np.ndarray, N_eps: int, seed: int) -> tf.Tensor:
    """Evaluate a one-shock residual and average it by state."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    k = tf.convert_to_tensor(states_k, dtype=tf.float32)
    z = tf.convert_to_tensor(states_z, dtype=tf.float32)
    eps = tf.random.normal((tf.shape(k)[0], N_eps), mean=0.0, stddev=1.0, dtype=tf.float32)
    eps = eps * tf.constant(1.0, dtype=tf.float32)  # keeps dtype explicit
    # residual functions expect shocks already scaled by sigma_eps, so caller draws inside closure.
    k_rep = tf.repeat(k[:, None], repeats=N_eps, axis=1)
    z_rep = tf.repeat(z[:, None], repeats=N_eps, axis=1)
    return residual_fn(tf.reshape(k_rep, (-1,)), tf.reshape(z_rep, (-1,)), tf.reshape(eps, (-1,)))


def eval_test_euler_mse_policy_only(policy: PolicyNet, mp: ModelParams, states_k: np.ndarray, states_z: np.ndarray, N_eps: int, seed: int) -> float:
    """Return the interior-style policy-only Euler MSE, ``mean(E[f]^2)``."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    k = tf.convert_to_tensor(states_k, dtype=tf.float32)
    z = tf.convert_to_tensor(states_z, dtype=tf.float32)
    eps = tf.random.normal((tf.shape(k)[0], N_eps), mean=0.0, stddev=mp.sigma_eps, dtype=tf.float32)
    k_flat = tf.reshape(tf.repeat(k[:, None], repeats=N_eps, axis=1), (-1,))
    z_flat = tf.reshape(tf.repeat(z[:, None], repeats=N_eps, axis=1), (-1,))
    f = euler_f_policy_only(policy, mp, k_flat, z_flat, tf.reshape(eps, (-1,)))
    f = tf.reshape(f, (tf.shape(k)[0], N_eps))
    return float(tf.reduce_mean(tf.square(tf.reduce_mean(f, axis=1))).numpy())


def eval_obj1_kkt_diagnostics(policy: PolicyNet, mp: ModelParams, states_k: np.ndarray, states_z: np.ndarray, N_eps: int, seed: int, interior_tol: float = 1e-4) -> dict[str, float]:
    """Evaluate Objective 1 with post-training KKT diagnostics."""
    tf.random.set_seed(seed)
    k = tf.convert_to_tensor(states_k, dtype=tf.float32)
    z = tf.convert_to_tensor(states_z, dtype=tf.float32)
    x = tf.stack([tf.maximum(k, mp.k_min), tf.maximum(z, 1e-12)], axis=1)
    k_next = tf.clip_by_value(policy(x), mp.k_min, mp.k_max)
    slack = k_next - tf.constant(mp.k_min, dtype=tf.float32)
    eps = tf.random.normal((tf.shape(k)[0], N_eps), mean=0.0, stddev=mp.sigma_eps, dtype=tf.float32)
    k_flat = tf.reshape(tf.repeat(k[:, None], repeats=N_eps, axis=1), (-1,))
    z_flat = tf.reshape(tf.repeat(z[:, None], repeats=N_eps, axis=1), (-1,))
    f = euler_f_policy_only(policy, mp, k_flat, z_flat, tf.reshape(eps, (-1,)))
    f = tf.reshape(f, (tf.shape(k)[0], N_eps))
    fbar = tf.reduce_mean(f, axis=1)
    r_fb = slack + fbar - tf.sqrt(tf.square(slack) + tf.square(fbar) + 1e-12)
    interior = slack > tf.constant(interior_tol, dtype=tf.float32)
    interior_count = tf.reduce_sum(tf.cast(interior, tf.float32))
    interior_mse = tf.cond(
        interior_count > 0.0,
        lambda: tf.reduce_mean(tf.square(tf.boolean_mask(fbar, interior))),
        lambda: tf.constant(float("nan"), dtype=tf.float32),
    )
    return {
        "test_kkt_mse": float(tf.reduce_mean(tf.square(r_fb)).numpy()),
        "test_euler_mse": float(tf.reduce_mean(tf.square(fbar)).numpy()),
        "test_euler_mse_interior": float(interior_mse.numpy()),
        "interior_share": float(tf.reduce_mean(tf.cast(interior, tf.float32)).numpy()),
        "mean_slack": float(tf.reduce_mean(slack).numpy()),
    }


def eval_obj2_kkt_diagnostics(policy: PolicyNet, multiplier: MultiplierNet, mp: ModelParams, op2: Obj2Params, states_k: np.ndarray, states_z: np.ndarray, N_eps: int, seed: int) -> dict[str, float]:
    """Evaluate Objective 2 FB, stationarity, and combined KKT diagnostics."""
    tf.random.set_seed(seed)
    k = tf.convert_to_tensor(states_k, dtype=tf.float32)
    z = tf.convert_to_tensor(states_z, dtype=tf.float32)
    r_fb = fb_residual_from_policy_multiplier(policy, multiplier, mp, k, z)
    eps = tf.random.normal((tf.shape(k)[0], N_eps), mean=0.0, stddev=mp.sigma_eps, dtype=tf.float32)
    k_flat = tf.reshape(tf.repeat(k[:, None], repeats=N_eps, axis=1), (-1,))
    z_flat = tf.reshape(tf.repeat(z[:, None], repeats=N_eps, axis=1), (-1,))
    x_flat = tf.stack([tf.maximum(k_flat, mp.k_min), tf.maximum(z_flat, 1e-12)], axis=1)
    lam_flat = multiplier(x_flat)
    f_flat = euler_f_policy_only(policy, mp, k_flat, z_flat, tf.reshape(eps, (-1,)))
    r_lam = tf.reshape(lam_flat - f_flat, (tf.shape(k)[0], N_eps))
    mean_r_lam = tf.reduce_mean(r_lam, axis=1)
    fb_mse = tf.reduce_mean(tf.square(r_fb))
    stat_mse = tf.reduce_mean(tf.square(mean_r_lam))
    return {
        "test_fb_mse": float(fb_mse.numpy()),
        "test_stationarity_mse": float(stat_mse.numpy()),
        "test_kkt_mse": float((fb_mse + op2.nu_lambda * stat_mse).numpy()),
        "test_euler_mse": float(stat_mse.numpy()),
    }


def eval_test_euler_mse_obj3(policy: PolicyNet, value: ValueNet, mp: ModelParams, states_k: np.ndarray, states_z: np.ndarray, N_eps: int, seed: int) -> float:
    """Backward-compatible Objective 3 stationarity-style MSE without multiplier."""
    tf.random.set_seed(seed)
    k = tf.convert_to_tensor(states_k, dtype=tf.float32)
    z = tf.convert_to_tensor(states_z, dtype=tf.float32)
    eps = tf.random.normal((tf.shape(k)[0], N_eps), mean=0.0, stddev=mp.sigma_eps, dtype=tf.float32)
    k_flat = tf.reshape(tf.repeat(k[:, None], repeats=N_eps, axis=1), (-1,))
    z_flat = tf.reshape(tf.repeat(z[:, None], repeats=N_eps, axis=1), (-1,))
    re = euler_derivative_residual_with_value(policy, value, mp, k_flat, z_flat, tf.reshape(eps, (-1,)))
    return float(tf.reduce_mean(tf.square(re)).numpy())


def eval_obj3_kkt_diagnostics(policy: PolicyNet, value: ValueNet, multiplier: MultiplierNet, mp: ModelParams, op3: Obj3Params, states_k: np.ndarray, states_z: np.ndarray, N_eps: int, seed: int) -> dict[str, float]:
    """Evaluate Objective 3 Bellman, FB, stationarity, and total residuals."""
    # Local import avoids an import cycle at module import time.
    from .objectives import bellman_residual

    tf.random.set_seed(seed)
    k = tf.convert_to_tensor(states_k, dtype=tf.float32)
    z = tf.convert_to_tensor(states_z, dtype=tf.float32)
    r_fb = fb_residual_from_policy_multiplier(policy, multiplier, mp, k, z)
    eps = tf.random.normal((tf.shape(k)[0], N_eps), mean=0.0, stddev=mp.sigma_eps, dtype=tf.float32)
    k_flat = tf.reshape(tf.repeat(k[:, None], repeats=N_eps, axis=1), (-1,))
    z_flat = tf.reshape(tf.repeat(z[:, None], repeats=N_eps, axis=1), (-1,))
    eps_flat = tf.reshape(eps, (-1,))
    rb = tf.reshape(bellman_residual(policy, value, mp, k_flat, z_flat, eps_flat), (tf.shape(k)[0], N_eps))
    re = euler_derivative_residual_with_value(policy, value, mp, k_flat, z_flat, eps_flat)
    lam = multiplier(tf.stack([tf.maximum(k_flat, mp.k_min), tf.maximum(z_flat, 1e-12)], axis=1))
    r_lam = tf.reshape(lam + re, (tf.shape(k)[0], N_eps))
    bellman_mse = tf.reduce_mean(tf.square(tf.reduce_mean(rb, axis=1)))
    fb_mse = tf.reduce_mean(tf.square(r_fb))
    stat_mse = tf.reduce_mean(tf.square(tf.reduce_mean(r_lam, axis=1)))
    total = bellman_mse + op3.nu_fb * fb_mse + op3.nu_lambda * stat_mse
    return {
        "test_bellman_mse": float(bellman_mse.numpy()),
        "test_fb_mse": float(fb_mse.numpy()),
        "test_stationarity_mse": float(stat_mse.numpy()),
        "test_total_residual": float(total.numpy()),
        "test_euler_mse": float(stat_mse.numpy()),
    }

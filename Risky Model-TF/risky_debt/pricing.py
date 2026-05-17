"""Constructed zero-profit pricing operators for the risky-debt model.

The model treats the risky debt price as an equilibrium object, not as a free
neural-network control.  During training we use smooth debt/default switches and
safe denominators; during final evaluation callers can use the hard exact
helpers.
"""

from __future__ import annotations

from typing import Tuple
import tensorflow as tf

from .config import ModelParams, TrainParams
from .primitives import recovery_R, solvency_weight, continuation_weight_from_value


@tf.function
def softplus_positive(x: tf.Tensor, kappa: float) -> tf.Tensor:
    """Smooth approximation to ``max(x, 0)`` with temperature ``kappa``."""
    x = tf.convert_to_tensor(x, tf.float32)
    k = tf.maximum(tf.cast(kappa, tf.float32), tf.constant(1e-8, tf.float32))
    return k * tf.nn.softplus(x / k)


@tf.function
def safe_positive_debt(b_next: tf.Tensor, kappa_b: float, eps_b: float) -> tf.Tensor:
    """Strictly positive proxy used only inside the safe risky-debt price."""
    return tf.cast(eps_b, tf.float32) + softplus_positive(b_next, kappa_b)


@tf.function
def debt_region_weight(b_next: tf.Tensor, kappa_b: float) -> tf.Tensor:
    """Smooth approximation to ``1{b' > 0}``."""
    return tf.sigmoid(tf.convert_to_tensor(b_next, tf.float32) / tf.cast(kappa_b, tf.float32))


@tf.function
def next_z_from_eps(z: tf.Tensor, eps: tf.Tensor, mp: ModelParams) -> tf.Tensor:
    """Apply the log-AR(1) transition to scalar or batched shock innovations."""
    z = tf.maximum(tf.convert_to_tensor(z, tf.float32), mp.z_min)
    return tf.exp(mp.rho * tf.math.log(z) + tf.cast(eps, tf.float32))


@tf.function
def crn_inner_eps(z: tf.Tensor, tp: TrainParams) -> tf.Tensor:
    """Draw the inner Monte Carlo innovations used by the pricing operator.

    The same returned matrix should be reused for all residual factors that share
    the same state-policy-price object.  This implements common random numbers
    inside a TensorFlow training step.
    """
    n_q = tf.maximum(tf.cast(tp.N_q, tf.int32), 1)
    return tf.random.normal((tf.shape(z)[0], n_q), 0.0, 1.0, dtype=tf.float32)


@tf.function
def pricing_moments_from_vtilde(
    vtilde,
    z: tf.Tensor,
    k_next: tf.Tensor,
    b_next: tf.Tensor,
    eps_q: tf.Tensor,
    mp: ModelParams,
    tp: TrainParams,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Return smooth ``(P_S, ER_D)`` from a continuation-value network."""
    z = tf.maximum(tf.convert_to_tensor(z, tf.float32), mp.z_min)
    k_next = tf.maximum(tf.convert_to_tensor(k_next, tf.float32), mp.k_min)
    b_next = tf.convert_to_tensor(b_next, tf.float32)
    z_next = next_z_from_eps(tf.expand_dims(z, 1), tf.cast(eps_q, tf.float32) * mp.sigma_eps, mp)

    batch = tf.shape(z_next)[0]
    n_q = tf.shape(z_next)[1]
    k_rep = tf.broadcast_to(tf.expand_dims(k_next, 1), (batch, n_q))
    b_rep = tf.broadcast_to(tf.expand_dims(b_next, 1), (batch, n_q))
    x_next = tf.stack([tf.reshape(k_rep, (-1,)), tf.reshape(b_rep, (-1,)), tf.reshape(z_next, (-1,))], axis=1)
    vt_next = tf.reshape(vtilde(x_next), (batch, n_q))
    s_next = continuation_weight_from_value(vt_next, tp.kappa_solv)
    rec = recovery_R(k_rep, z_next, mp)
    p_s = tf.reduce_mean(s_next, axis=1)
    er_d = tf.reduce_mean((1.0 - s_next) * rec, axis=1)
    return p_s, er_d


@tf.function
def pricing_moments_from_proxy(
    z: tf.Tensor,
    k_next: tf.Tensor,
    b_next: tf.Tensor,
    eps_q: tf.Tensor,
    mp: ModelParams,
    tp: TrainParams,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Return smooth pricing moments from the asset-value solvency proxy.

    This is used only by legacy objective-1 paths that do not yet carry a
    continuation-value critic in their public API.
    """
    z = tf.maximum(tf.convert_to_tensor(z, tf.float32), mp.z_min)
    z_next = next_z_from_eps(tf.expand_dims(z, 1), tf.cast(eps_q, tf.float32) * mp.sigma_eps, mp)
    batch = tf.shape(z_next)[0]
    n_q = tf.shape(z_next)[1]
    k_rep = tf.broadcast_to(tf.expand_dims(tf.maximum(k_next, mp.k_min), 1), (batch, n_q))
    b_rep = tf.broadcast_to(tf.expand_dims(tf.convert_to_tensor(b_next, tf.float32), 1), (batch, n_q))
    s_next = solvency_weight(k_rep, b_rep, z_next, mp, tp.kappa_solv)
    rec = recovery_R(k_rep, z_next, mp)
    return tf.reduce_mean(s_next, axis=1), tf.reduce_mean((1.0 - s_next) * rec, axis=1)


@tf.function
def safe_risky_debt_price(
    p_s: tf.Tensor,
    er_d: tf.Tensor,
    b_next: tf.Tensor,
    mp: ModelParams,
    tp: TrainParams,
) -> tf.Tensor:
    """Safe smooth positive-debt price component ``q_safe^D``."""
    b_pos_safe = safe_positive_debt(b_next, tp.kappa_b, tp.eps_b)
    denom = (1.0 + mp.r) * b_pos_safe - er_d + tf.cast(tp.eps_den, tf.float32)
    qd = p_s * b_pos_safe / denom
    return tf.clip_by_value(qd, mp.q_min, mp.q_max)


@tf.function
def smooth_price_from_moments(
    p_s: tf.Tensor,
    er_d: tf.Tensor,
    b_next: tf.Tensor,
    mp: ModelParams,
    tp: TrainParams,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return ``(q, qD_safe, rD_safe, penalty)`` under smooth training rules."""
    qd = safe_risky_debt_price(p_s, er_d, b_next, mp, tp)
    omega = debt_region_weight(b_next, tp.kappa_b)
    q_cash = tf.ones_like(qd) / (1.0 + tf.cast(mp.r_c, tf.float32))
    q = omega * qd + (1.0 - omega) * q_cash
    q = tf.clip_by_value(q, mp.q_min, mp.q_max)
    r_d = 1.0 / qd - 1.0
    b_pos_safe = safe_positive_debt(b_next, tp.kappa_b, tp.eps_b)
    penalty = tf.cast(tp.omega_q, tf.float32) * (
        tf.square(tf.nn.relu(tf.cast(tp.eps_q, tf.float32) - p_s))
        + tf.square(tf.nn.relu(tf.cast(tp.eps_q, tf.float32) + er_d - (1.0 + mp.r) * b_pos_safe))
    )
    return q, qd, r_d, penalty


@tf.function
def smooth_price_from_vtilde(vtilde, z, k_next, b_next, eps_q, mp: ModelParams, tp: TrainParams):
    """Construct the smooth price from a continuation-value network."""
    p_s, er_d = pricing_moments_from_vtilde(vtilde, z, k_next, b_next, eps_q, mp, tp)
    return smooth_price_from_moments(p_s, er_d, b_next, mp, tp)


@tf.function
def smooth_price_from_proxy(z, k_next, b_next, eps_q, mp: ModelParams, tp: TrainParams):
    """Construct the smooth price from the legacy asset-value proxy."""
    p_s, er_d = pricing_moments_from_proxy(z, k_next, b_next, eps_q, mp, tp)
    return smooth_price_from_moments(p_s, er_d, b_next, mp, tp)


@tf.function
def exact_price_from_moments(p_s: tf.Tensor, er_d: tf.Tensor, b_next: tf.Tensor, mp: ModelParams):
    """Exact nonsmoothed price from hard moment estimates.

    The caller is responsible for providing hard-indicator moment estimates.
    """
    b_next = tf.convert_to_tensor(b_next, tf.float32)
    q_cash = tf.ones_like(b_next) / (1.0 + tf.cast(mp.r_c, tf.float32))
    denom = (1.0 + mp.r) * b_next - er_d
    qd = tf.math.divide_no_nan(p_s * b_next, denom)
    qd = tf.clip_by_value(qd, mp.q_min, mp.q_max)
    return tf.where(b_next > 0.0, qd, q_cash), qd, 1.0 / qd - 1.0


@tf.function
def exact_pricing_moments_from_vtilde(
    vtilde,
    z: tf.Tensor,
    k_next: tf.Tensor,
    b_next: tf.Tensor,
    eps_q: tf.Tensor,
    mp: ModelParams,
    tp: TrainParams,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Return hard-indicator ``(P_S, ER_D)`` from a continuation-value network.

    This is for final nonsmoothed economic evaluation, not differentiable
    training.  It uses the exact default rule Vtilde(k',b',z') > 0.
    """
    z = tf.maximum(tf.convert_to_tensor(z, tf.float32), mp.z_min)
    k_next = tf.maximum(tf.convert_to_tensor(k_next, tf.float32), mp.k_min)
    b_next = tf.convert_to_tensor(b_next, tf.float32)
    z_next = next_z_from_eps(tf.expand_dims(z, 1), tf.cast(eps_q, tf.float32) * mp.sigma_eps, mp)
    batch = tf.shape(z_next)[0]
    n_q = tf.shape(z_next)[1]
    k_rep = tf.broadcast_to(tf.expand_dims(k_next, 1), (batch, n_q))
    b_rep = tf.broadcast_to(tf.expand_dims(b_next, 1), (batch, n_q))
    x_next = tf.stack([tf.reshape(k_rep, (-1,)), tf.reshape(b_rep, (-1,)), tf.reshape(z_next, (-1,))], axis=1)
    vt_next = tf.reshape(vtilde(x_next), (batch, n_q))
    s_hard = tf.cast(vt_next > 0.0, tf.float32)
    rec = recovery_R(k_rep, z_next, mp)
    p_s = tf.reduce_mean(s_hard, axis=1)
    er_d = tf.reduce_mean((1.0 - s_hard) * rec, axis=1)
    return p_s, er_d


@tf.function
def exact_pricing_moments_from_proxy(
    z: tf.Tensor,
    k_next: tf.Tensor,
    b_next: tf.Tensor,
    eps_q: tf.Tensor,
    mp: ModelParams,
    tp: TrainParams,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Return hard-indicator pricing moments from the legacy solvency proxy."""
    z = tf.maximum(tf.convert_to_tensor(z, tf.float32), mp.z_min)
    z_next = next_z_from_eps(tf.expand_dims(z, 1), tf.cast(eps_q, tf.float32) * mp.sigma_eps, mp)
    batch = tf.shape(z_next)[0]
    n_q = tf.shape(z_next)[1]
    k_rep = tf.broadcast_to(tf.expand_dims(tf.maximum(k_next, mp.k_min), 1), (batch, n_q))
    b_rep = tf.broadcast_to(tf.expand_dims(tf.convert_to_tensor(b_next, tf.float32), 1), (batch, n_q))
    s_hard = tf.cast(solvency_weight(k_rep, b_rep, z_next, mp, tp.kappa_solv) > 0.5, tf.float32)
    rec = recovery_R(k_rep, z_next, mp)
    return tf.reduce_mean(s_hard, axis=1), tf.reduce_mean((1.0 - s_hard) * rec, axis=1)


@tf.function
def exact_price_from_vtilde(vtilde, z, k_next, b_next, eps_q, mp: ModelParams, tp: TrainParams):
    """Construct exact nonsmoothed price from a continuation-value network."""
    p_s, er_d = exact_pricing_moments_from_vtilde(vtilde, z, k_next, b_next, eps_q, mp, tp)
    return exact_price_from_moments(p_s, er_d, b_next, mp)


@tf.function
def exact_price_from_proxy(z, k_next, b_next, eps_q, mp: ModelParams, tp: TrainParams):
    """Construct exact nonsmoothed price from the legacy solvency proxy."""
    p_s, er_d = exact_pricing_moments_from_proxy(z, k_next, b_next, eps_q, mp, tp)
    return exact_price_from_moments(p_s, er_d, b_next, mp)


@tf.function
def positive_debt_tax_shield(b_next: tf.Tensor, r_d: tf.Tensor, solvency_weight: tf.Tensor, mp: ModelParams, tp: TrainParams) -> tf.Tensor:
    """Training tax shield: positive-debt smooth amount times solvency weight."""
    b_pos = softplus_positive(b_next, tp.kappa_b)
    return mp.tau * r_d * b_pos * tf.convert_to_tensor(solvency_weight, tf.float32)


@tf.function
def exact_positive_debt_tax_shield(b_next: tf.Tensor, r_d: tf.Tensor, solvency_indicator: tf.Tensor, mp: ModelParams) -> tf.Tensor:
    """Exact tax shield: hard positive debt and hard solvency indicator."""
    b_pos = tf.maximum(tf.convert_to_tensor(b_next, tf.float32), 0.0)
    return mp.tau * r_d * b_pos * tf.cast(solvency_indicator, tf.float32)

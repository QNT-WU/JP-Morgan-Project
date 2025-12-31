from __future__ import annotations

import tensorflow as tf
from .config import ModelParams


def beta_from_r(r: float) -> float:
    return 1.0 / (1.0 + r)


@tf.function
def shock_next_z(z: tf.Tensor, rho: float, sigma_eps: float) -> tf.Tensor:
    z = tf.maximum(z, 1e-12)
    eps = tf.random.normal(tf.shape(z), mean=0.0, stddev=sigma_eps, dtype=tf.float32)
    return tf.exp(rho * tf.math.log(z) + eps)


# Production / profit function
@tf.function
def profit_pi(k: tf.Tensor, z: tf.Tensor, theta: float) -> tf.Tensor:
    return z * tf.pow(k, theta)


# Investment law
@tf.function
def investment_I(k: tf.Tensor, k_next: tf.Tensor, delta: float) -> tf.Tensor:
    return k_next - (1.0 - delta) * k


# Adjustment cost
@tf.function
def adj_cost_psi(I: tf.Tensor, k: tf.Tensor, psi0: float, k_min: float) -> tf.Tensor:
    k = tf.maximum(k, k_min)
    return psi0 * tf.square(I) / (2.0 * k)


# Smoothed indicator 1{𝑥<0}
# If 𝑥 is very negative → −𝑥/𝑘𝑎𝑝𝑝𝑎 large positive → sigmoid ≈ 1
# If 𝑥 is positive → sigmoid ≈ 0
@tf.function
def smooth_indicator_neg(x: tf.Tensor, kappa: float) -> tf.Tensor:
    # approx 1_{x<0}
    return tf.sigmoid(-x / tf.constant(kappa, tf.float32))


# Equity issuance cost function 𝜂(𝑒)
@tf.function
def issuance_eta(
    e: tf.Tensor, eta0: float, eta1: float, kappa_issue: float
) -> tf.Tensor:
    # eta(e) = (eta0 + eta1*e) * 1_{e<0} (smoothed)
    ind = smooth_indicator_neg(e, kappa_issue)
    return (eta0 + eta1 * e) * ind


# Equity cashflow identity
@tf.function
def equity_cashflow_e(
    k: tf.Tensor,
    k_next: tf.Tensor,
    b: tf.Tensor,
    b_next: tf.Tensor,
    z: tf.Tensor,
    q: tf.Tensor,
    mp: ModelParams,
) -> tf.Tensor:
    """
    e = (1-tau) pi(k,z) - psi(I,k) - I + b' q + (tau b'/(1+r)) (1-q) - b
    """
    k = tf.maximum(k, mp.k_min)
    z = tf.maximum(z, mp.z_min)

    pi = profit_pi(k, z, mp.theta)
    I = investment_I(k, k_next, mp.delta)
    psi = adj_cost_psi(I, k, mp.psi0, mp.k_min)

    term_debt_issue = b_next * q
    term_tax_shield = (mp.tau * b_next / (1.0 + mp.r)) * (1.0 - q)

    e = (1.0 - mp.tau) * pi - psi - I + term_debt_issue + term_tax_shield - b
    return e


# Equity payout 𝑑(⋅)
# d=e+η(e)
@tf.function
def equity_payout_d(
    k: tf.Tensor,
    k_next: tf.Tensor,
    b: tf.Tensor,
    b_next: tf.Tensor,
    z: tf.Tensor,
    q: tf.Tensor,
    mp: ModelParams,
    kappa_issue: float,
) -> tf.Tensor:
    e = equity_cashflow_e(k, k_next, b, b_next, z, q, mp)
    eta = issuance_eta(e, mp.eta0, mp.eta1, kappa_issue)
    return e + eta


# Asset value Proxy: A(k,z)
# Meaning: A(k,z)=(1−τ)π(k,z)+(1−δ)k
@tf.function
def asset_value_A(k: tf.Tensor, z: tf.Tensor, mp: ModelParams) -> tf.Tensor:
    """
    A(k,z) = (1-tau) pi(k,z) + (1-delta)k
    Used for solvency proxy.
    """
    k = tf.maximum(k, mp.k_min)
    z = tf.maximum(z, mp.z_min)
    pi = profit_pi(k, z, mp.theta)
    return (1.0 - mp.tau) * pi + (1.0 - mp.delta) * k


@tf.function
def recovery_R(k_next: tf.Tensor, z_next: tf.Tensor, mp: ModelParams) -> tf.Tensor:
    """
    R(k',z') = (1-alpha) [ (1-tau) pi(k',z') + (1-delta) k' ]
    """
    A = asset_value_A(k_next, z_next, mp)
    return (1.0 - mp.alpha) * A


@tf.function
def solvency_weight(
    k_next: tf.Tensor,
    b_next: tf.Tensor,
    z_next: tf.Tensor,
    mp: ModelParams,
    kappa_solv: float = 0.05,
    eps: float = 1e-6,
) -> tf.Tensor:
    k_next = tf.maximum(tf.convert_to_tensor(k_next, tf.float32), mp.k_min)
    b_next = tf.convert_to_tensor(b_next, tf.float32)
    z_next = tf.maximum(tf.convert_to_tensor(z_next, tf.float32), mp.z_min)

    proxy = (
        (1.0 - mp.tau) * (z_next * tf.pow(k_next, mp.theta))
        + (1.0 - mp.delta) * k_next
        - (1.0 + mp.r) * b_next
    )
    s = tf.sigmoid(proxy / tf.cast(kappa_solv, tf.float32))
    return tf.clip_by_value(s, eps, 1.0 - eps)

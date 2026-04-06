"""Economic primitive functions for the risky-debt structural model."""

from __future__ import annotations

import tensorflow as tf
from .config import ModelParams


def beta_from_r(r: float) -> float:
    """Return the one-period discount factor implied by a risk-free rate."""
    return 1.0 / (1.0 + r)


def beta_tensor_from_r(r) -> tf.Tensor:
    """Return the one-period discount factor as a float32 tensor.

    This helper is safe inside ``@tf.function`` traces because it accepts either
    a Python float or a Tensor/SymbolicTensor and avoids wrapping symbolic values
    with ``tf.constant``.
    """
    r_t = tf.convert_to_tensor(r, dtype=tf.float32)
    return tf.math.reciprocal(1.0 + r_t)


@tf.function
def shock_next_z(z: tf.Tensor, rho: float, sigma_eps: float) -> tf.Tensor:
    """Draw the next productivity level under the log-AR(1) shock process."""
    z = tf.maximum(z, 1e-12)
    eps = tf.random.normal(tf.shape(z), mean=0.0, stddev=sigma_eps, dtype=tf.float32)
    return tf.exp(rho * tf.math.log(z) + eps)


@tf.function
def profit_pi(k: tf.Tensor, z: tf.Tensor, theta: float) -> tf.Tensor:
    """Return operating profit ``pi(k,z)=z k^theta``."""
    return z * tf.pow(k, theta)


@tf.function
def investment_I(k: tf.Tensor, k_next: tf.Tensor, delta: float) -> tf.Tensor:
    """Return net investment implied by current and next-period capital."""
    return k_next - (1.0 - delta) * k


@tf.function
def adj_cost_psi(I: tf.Tensor, k: tf.Tensor, psi0: float, k_min: float) -> tf.Tensor:
    """Return the convex adjustment cost ``psi0 * I^2 / (2k)``."""
    k = tf.maximum(k, k_min)
    return psi0 * tf.square(I) / (2.0 * k)


@tf.function
def smooth_indicator_neg(x: tf.Tensor, kappa: float) -> tf.Tensor:
    """Smooth approximation to 1{x < 0}."""
    return tf.sigmoid(-x / tf.constant(kappa, tf.float32))


@tf.function
def issuance_eta(
    e: tf.Tensor, eta0: float, eta1: float, kappa_issue: float
) -> tf.Tensor:
    """
    External-equity issuance cost contribution added to equity payout.

    Economic cost when e < 0:
        cost = eta0 + eta1 * (-e) >= 0

    Since payout is d = e + eta(e), we return eta(e) = -cost.
    """
    ind = smooth_indicator_neg(e, kappa_issue)
    cost = (eta0 - eta1 * e) * ind
    return -cost


@tf.function
def interest_tax_shield_pv(
    b_next: tf.Tensor,
    q: tf.Tensor,
    continuation_weight: tf.Tensor,
    mp: ModelParams,
) -> tf.Tensor:
    """
    Present value of the interest tax shield.

    The shield is contingent on next-period continuation/solvency and is zero in
    default states. It also applies only to positive debt issuance (b' > 0).

    PV at time t:
        beta * tau * r_tilde * b' * continuation_weight * 1{b' > 0}
    where r_tilde = 1/q - 1.
    """
    beta = beta_tensor_from_r(mp.r)
    q = tf.clip_by_value(tf.convert_to_tensor(q, tf.float32), 1e-6, 1.0 - 1e-6)
    b_next = tf.convert_to_tensor(b_next, tf.float32)
    continuation_weight = tf.convert_to_tensor(continuation_weight, tf.float32)

    r_tilde = (1.0 / q) - 1.0
    debt_mask = tf.cast(b_next > 0.0, tf.float32)
    return beta * mp.tau * r_tilde * b_next * continuation_weight * debt_mask


@tf.function
def equity_cashflow_base_e(
    k: tf.Tensor,
    k_next: tf.Tensor,
    b: tf.Tensor,
    b_next: tf.Tensor,
    z: tf.Tensor,
    q: tf.Tensor,
    mp: ModelParams,
) -> tf.Tensor:
    """
    Base risky-debt equity cash-flow identity without the contingent tax shield.

    This keeps backward compatibility with older code paths while letting the
    aligned codebase build the full payout object from a single total-e object.
    """
    k = tf.maximum(k, mp.k_min)
    z = tf.maximum(z, mp.z_min)

    pi = profit_pi(k, z, mp.theta)
    I = investment_I(k, k_next, mp.delta)
    psi = adj_cost_psi(I, k, mp.psi0, mp.k_min)
    term_debt_issue = b_next * q

    return (1.0 - mp.tau) * pi - psi - I + term_debt_issue - b


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
    """Backward-compatible alias for the base cash-flow object."""
    return equity_cashflow_base_e(k, k_next, b, b_next, z, q, mp)


@tf.function
def equity_cashflow_total_e(
    k: tf.Tensor,
    k_next: tf.Tensor,
    b: tf.Tensor,
    b_next: tf.Tensor,
    z: tf.Tensor,
    q: tf.Tensor,
    continuation_weight: tf.Tensor,
    mp: ModelParams,
) -> tf.Tensor:
    """
    Total risky-debt cash-flow identity including the contingent tax shield.

    This is the aligned object corresponding to the model-consistent e(·):
        e_total = e_base + tax_shield_pv
    where the tax shield is zero in default states.
    """
    e_base = equity_cashflow_base_e(k, k_next, b, b_next, z, q, mp)
    tax_pv = interest_tax_shield_pv(
        b_next=b_next,
        q=q,
        continuation_weight=continuation_weight,
        mp=mp,
    )
    return e_base + tax_pv


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
    """Backward-compatible base payout d = e + eta(e), excluding the tax shield."""
    e = equity_cashflow_base_e(k, k_next, b, b_next, z, q, mp)
    eta = issuance_eta(e, mp.eta0, mp.eta1, kappa_issue)
    return e + eta


@tf.function
def equity_payout_d_total(
    k: tf.Tensor,
    k_next: tf.Tensor,
    b: tf.Tensor,
    b_next: tf.Tensor,
    z: tf.Tensor,
    q: tf.Tensor,
    continuation_weight: tf.Tensor,
    mp: ModelParams,
    kappa_issue: float,
) -> tf.Tensor:
    """
    Model-consistent payout object.

    The issuance-cost trigger is applied to the shield-adjusted total cash flow:
        d = e_total + eta(e_total)
    This keeps the NN, benchmark, and diagnostics on the same definition.
    """
    e_total = equity_cashflow_total_e(
        k=k,
        k_next=k_next,
        b=b,
        b_next=b_next,
        z=z,
        q=q,
        continuation_weight=continuation_weight,
        mp=mp,
    )
    eta = issuance_eta(e_total, mp.eta0, mp.eta1, kappa_issue)
    return e_total + eta


@tf.function
def asset_value_A(k: tf.Tensor, z: tf.Tensor, mp: ModelParams) -> tf.Tensor:
    """A(k,z) = (1-tau) pi(k,z) + (1-delta)k used for the solvency proxy."""
    k = tf.maximum(k, mp.k_min)
    z = tf.maximum(z, mp.z_min)
    pi = profit_pi(k, z, mp.theta)
    return (1.0 - mp.tau) * pi + (1.0 - mp.delta) * k


@tf.function
def recovery_R(k_next: tf.Tensor, z_next: tf.Tensor, mp: ModelParams) -> tf.Tensor:
    """R(k',z') = (1-alpha) [ (1-tau) pi(k',z') + (1-delta) k' ]."""
    A = asset_value_A(k_next, z_next, mp)
    return (1.0 - mp.alpha) * A


@tf.function
def continuation_weight_from_value(
    continuation_value: tf.Tensor,
    kappa_solv: float = 0.05,
    eps: float = 1e-6,
) -> tf.Tensor:
    """Smooth continuation/default gate based on a value or continuation object."""
    continuation_value = tf.convert_to_tensor(continuation_value, tf.float32)
    s = tf.sigmoid(continuation_value / tf.cast(kappa_solv, tf.float32))
    return tf.clip_by_value(s, eps, 1.0 - eps)


@tf.function
def solvency_weight(
    k_next: tf.Tensor,
    b_next: tf.Tensor,
    z_next: tf.Tensor,
    mp: ModelParams,
    kappa_solv: float = 0.05,
    eps: float = 1e-6,
) -> tf.Tensor:
    """Return a smooth continuation weight based on one-step solvency."""
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

"""Unit tests for risky-debt primitive functions."""

# Test/test_primitives.py
# Unit tests for risky-debt primitives: shape sanity + basic sign checks + finite checks
import numpy as np
import tensorflow as tf

from risky_debt.config import ModelParams
from risky_debt.primitives import (
    profit_pi,
    investment_I,
    adj_cost_psi,
    equity_cashflow_e,
    equity_cashflow_total_e,
    equity_payout_d,
    equity_payout_d_total,
    issuance_eta,
    recovery_R,
    solvency_weight,
)


def test_profit_positive():
    """Verify profit positive."""
    mp = ModelParams()
    k = tf.constant([0.5, 1.0, 2.0], tf.float32)
    z = tf.constant([0.8, 1.0, 1.2], tf.float32)
    pi = profit_pi(k, z, mp.theta)
    assert pi.shape == (3,)
    assert tf.reduce_all(pi > 0).numpy()


def test_investment_identity():
    """Verify investment identity."""
    mp = ModelParams()
    k = tf.constant([1.0, 2.0], tf.float32)
    k_next = tf.constant([1.1, 1.9], tf.float32)
    I = investment_I(k, k_next, mp.delta)
    expected = k_next - (1.0 - mp.delta) * k
    np.testing.assert_allclose(I.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6)


def test_adjustment_cost_nonnegative():
    """Verify adjustment cost nonnegative."""
    mp = ModelParams()
    k = tf.constant([1.0, 2.0], tf.float32)
    I = tf.constant([0.1, -0.2], tf.float32)
    psi = adj_cost_psi(I, k, mp.psi0, mp.k_min)
    assert tf.reduce_all(psi >= 0).numpy()


def test_cashflow_and_payout_shapes_and_finite():
    """Verify cashflow and payout shapes and finite."""
    mp = ModelParams()
    n = 10
    k = tf.ones((n,), tf.float32)
    k_next = tf.ones((n,), tf.float32) * 1.1
    b = tf.zeros((n,), tf.float32)
    b_next = tf.ones((n,), tf.float32) * 0.2
    z = tf.ones((n,), tf.float32)
    q = tf.ones((n,), tf.float32) * 0.95

    e = equity_cashflow_e(k, k_next, b, b_next, z, q, mp)
    d = equity_payout_d(k, k_next, b, b_next, z, q, mp, kappa_issue=0.02)

    assert e.shape == (n,)
    assert d.shape == (n,)
    assert np.isfinite(e.numpy()).all()
    assert np.isfinite(d.numpy()).all()


def test_recovery_positive_and_finite():
    """Verify recovery positive and finite."""
    mp = ModelParams()
    k_next = tf.constant([1.0, 2.0], tf.float32)
    z_next = tf.constant([0.8, 1.2], tf.float32)
    R = recovery_R(k_next, z_next, mp)
    assert R.shape == (2,)
    tf.debugging.assert_all_finite(R, "Recovery returned non-finite values")
    assert tf.reduce_all(R > 0).numpy()


def test_solvency_weight_in_0_1_and_monotone_in_b():
    """Verify solvency weight in 0 1 and monotone in b."""
    mp = ModelParams()
    k_next = tf.constant([1.0, 1.0], tf.float32)
    z_next = tf.constant([1.0, 1.0], tf.float32)
    # smaller debt should be "more solvent" => larger s
    b_small = tf.constant([0.1, 10.0], tf.float32)
    s = solvency_weight(k_next, b_small, z_next, mp, kappa_solv=0.05)

    assert s.shape == (2,)
    assert tf.reduce_all((s > 0) & (s < 1)).numpy()
    assert float(s[0].numpy()) > float(s[1].numpy())



def test_total_payout_uses_shield_adjusted_cashflow_for_eta_trigger():
    """Verify total payout uses shield adjusted cashflow for eta trigger."""
    mp = ModelParams()
    n = 4
    k = tf.ones((n,), tf.float32)
    k_next = tf.ones((n,), tf.float32) * 1.05
    b = tf.zeros((n,), tf.float32)
    b_next = tf.constant([0.5, 0.5, 0.5, 0.5], tf.float32)
    z = tf.ones((n,), tf.float32)
    q = tf.ones((n,), tf.float32) * 0.95
    cont = tf.constant([0.0, 0.25, 0.75, 1.0], tf.float32)

    e_total = equity_cashflow_total_e(k, k_next, b, b_next, z, q, cont, mp)
    d_total = equity_payout_d_total(k, k_next, b, b_next, z, q, cont, mp, kappa_issue=0.02)
    eta_total = issuance_eta(e_total, mp.eta0, mp.eta1, kappa_issue=0.02)

    np.testing.assert_allclose(
        d_total.numpy(),
        (e_total + eta_total).numpy(),
        rtol=1e-6,
        atol=1e-6,
    )

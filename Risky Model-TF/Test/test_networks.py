"""Unit tests for neural-network modules."""

# Test/test_networks.py
import tensorflow as tf

from risky_debt.config import ModelParams, NetParams
from risky_debt.networks import PolicyNet, ValueNet, PricingNet


def test_policy_output_shape_and_constraints():
    """Verify policy output shape and constraints."""
    mp = ModelParams()
    npol = NetParams(hidden_units=16, hidden_layers=2, activation="tanh")

    pol = PolicyNet(npol, mp.k_min, mp.b_min, mp.b_max)

    # Input is (k,b,z) => shape (N,3)
    x = tf.constant(
        [[1.0, 0.1, 1.0], [2.0, -0.2, 0.7], [0.5, 0.0, 1.3]],
        dtype=tf.float32,
    )
    y = pol(x)

    # Output is (k', b') => shape (N,2)
    assert y.shape == (3, 2)
    tf.debugging.assert_all_finite(y, "PolicyNet produced non-finite outputs")

    k_next = y[:, 0]
    b_next = y[:, 1]

    # k' must be positive (>= k_min)
    assert tf.reduce_all(k_next >= mp.k_min).numpy()
    # b' must be bounded
    assert tf.reduce_all(b_next >= mp.b_min).numpy()
    assert tf.reduce_all(b_next <= mp.b_max).numpy()


def test_value_output_shape_and_finite():
    """Verify value output shape and finite."""
    nval = NetParams(hidden_units=16, hidden_layers=2, activation="tanh")
    val = ValueNet(nval)

    x = tf.constant(
        [[1.0, 0.1, 1.0], [2.0, -0.2, 0.7], [0.5, 0.0, 1.3]],
        dtype=tf.float32,
    )
    v = val(x)

    assert v.shape == (3,)
    tf.debugging.assert_all_finite(v, "ValueNet produced non-finite outputs")
    # In our risky-debt implementation, ValueNet is enforced nonnegative (softplus)
    assert tf.reduce_all(v >= -1e-8).numpy()


def test_pricing_output_shape_and_bounds():
    """Verify pricing output shape and bounds."""
    mp = ModelParams()
    nq = NetParams(hidden_units=16, hidden_layers=2, activation="tanh")
    qnet = PricingNet(nq, mp.q_min, mp.q_max)

    x_q = tf.constant(
        [
            [1.0, 1.0, 0.2],  # (z,k',b') packed the same shape (N,3)
            [0.8, 2.0, -0.1],
            [1.2, 0.9, 0.3],
        ],
        dtype=tf.float32,
    )
    q = qnet(x_q)

    assert q.shape == (3,)
    tf.debugging.assert_all_finite(q, "PricingNet produced non-finite outputs")
    assert tf.reduce_all(q >= mp.q_min).numpy()
    assert tf.reduce_all(q <= mp.q_max).numpy()

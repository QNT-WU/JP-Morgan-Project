"""Tests for policy and value networks."""

# Test/test_networks.py
import tensorflow as tf

from basic_mailer.config import ModelParams, NetParams
from basic_mailer.networks import MultiplierNet, PolicyNet, ValueNet


# This test checks two invariants:
# Shape correctness
# Positivity constraint
def test_policy_output_positive_and_shape():
    # Explicitly sets k_min
    # This is the lower bound enforced by PolicyNet
    # Making it explicit avoids relying on defaults
    """Verify policy outputs are positive and correctly shaped."""
    mp = ModelParams(k_min=1e-6)
    # Defines a small but nontrivial network:
    # 2 hidden layers
    # 16 units per layer
    # Smooth activation (tanh)
    npol = NetParams(hidden_units=16, hidden_layers=2, activation="tanh")
    # Constructs the policy network
    # Internally: dense layers and a sigmoid head bounded in [k_min, k_max]
    pol = PolicyNet(npol, mp.k_min, mp.k_max)

    # Shape: (3, 2)
    # Interpretation:3 observations
    # 2 state variables: (k, z)
    # This tests batch behavior, not just scalar input.
    x = tf.constant([[1.0, 1.0], [2.0, 0.7], [0.5, 1.3]], dtype=tf.float32)
    y = pol(x)

    # Expected behavior:
    # Input: [N, 2]
    # Output: [N]
    assert y.shape == (3,)
    # This checks:
    # Output is a 1D tensor
    # One scalar decision 𝑘′per input state
    # If this fails:Network wiring is wrong
    # tf.squeeze missing or misused
    # Numerical sanity check
    # This will fail if: NaN, +∞, −∞
    tf.debugging.assert_all_finite(y, "PolicyNet produced non-finite outputs")
    # Economic constraint check
    # This will fail if:
    # Your policy violates feasibility
    # Euler equations become undefined
    # Entire model breaks
    assert tf.reduce_all(y >= mp.k_min).numpy()


# This test checks shape + numerical stability for ValueNet.
def test_value_output_shape_and_finite():
    """Verify value outputs are finite and correctly shaped."""
    nval = NetParams(hidden_units=16, hidden_layers=2, activation="tanh")
    val = ValueNet(nval)

    # Same shape: (3, 2)
    x = tf.constant([[1.0, 1.0], [2.0, 0.7], [0.5, 1.3]], dtype=tf.float32)
    # Output shape (3,)
    # One scalar value per state
    v = val(x)

    # Shape check
    assert v.shape == (3,)
    # Finite check
    # exploding weights, bad initialization, illegal operations
    tf.debugging.assert_all_finite(v, "ValueNet produced non-finite outputs")


def test_multiplier_output_nonnegative_and_shape():
    """Verify multiplier outputs are nonnegative and correctly shaped."""
    nlam = NetParams(hidden_units=16, hidden_layers=2, activation="tanh")
    mul = MultiplierNet(nlam)
    x = tf.constant([[1.0, 1.0], [2.0, 0.7], [0.5, 1.3]], dtype=tf.float32)
    lam = mul(x)
    assert lam.shape == (3,)
    tf.debugging.assert_all_finite(lam, "MultiplierNet produced non-finite outputs")
    assert tf.reduce_all(lam >= 0.0).numpy()

# Test/test_networks.py
import tensorflow as tf

from basic_mailer.config import ModelParams, NetParams
from basic_mailer.networks import PolicyNet, ValueNet


# This test checks two invariants:
# Shape correctness
# Positivity constraint
def test_policy_output_positive_and_shape():
    # Explicitly sets k_min
    mp = ModelParams(k_min=1e-6)
    # Defines a small but nontrivial network:
    npol = NetParams(hidden_units=16, hidden_layers=2, activation="tanh")
    # Constructs the policy network
    # Internally:dense layers, softplus output, + k_min
    pol = PolicyNet(npol, mp.k_min)

    # Shape: (3, 2)
    x = tf.constant([[1.0, 1.0], [2.0, 0.7], [0.5, 1.3]], dtype=tf.float32)
    y = pol(x)

    # Expected behavior:
    # Input: [N, 2]
    # Output: [N]
    assert y.shape == (3,)
    # This checks:
    # Output is a 1D tensor
    tf.debugging.assert_all_finite(y, "PolicyNet produced non-finite outputs")
    # Economic constraint check
    assert tf.reduce_all(y >= mp.k_min).numpy()


# This test checks shape + numerical stability for ValueNet.
def test_value_output_shape_and_finite():
    nval = NetParams(hidden_units=16, hidden_layers=2, activation="tanh")
    val = ValueNet(nval)

    # Same shape: (3, 2)
    x = tf.constant([[1.0, 1.0], [2.0, 0.7], [0.5, 1.3]], dtype=tf.float32)
    # Output shape (3,)
    v = val(x)

    # Shape check
    assert v.shape == (3,)
    # Finite check
    tf.debugging.assert_all_finite(v, "ValueNet produced non-finite outputs")

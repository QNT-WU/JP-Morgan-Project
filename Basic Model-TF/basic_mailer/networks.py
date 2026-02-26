from __future__ import annotations

# delays evaluating type hints (safe, modern default).

import tensorflow as tf
from .config import NetParams


# Activation chooser: _get_activation
# This function maps a string like "tanh" into an actual TensorFlow function like tf.nn.tanh.
def _get_activation(name: str):
    name = name.lower().strip()
    if name == "tanh":
        return tf.nn.tanh
    if name == "elu":
        return tf.nn.elu
    if name == "softplus":
        return tf.nn.softplus
    if name == "relu":
        return tf.nn.relu
    raise ValueError(f"Unknown activation: {name}")


class PolicyNet(tf.keras.Model):
    """
    Policy: k' = phi(k,z)
    Enforces k' > 0 via softplus + k_min.
    Input: [N,2]
    Output: [N]
    """

    # def __init__(self, net_params: NetParams, k_min: float):
    def __init__(self, net_params: NetParams, k_min: float, k_max: float):
        # super().__init__() initializes the Keras model machinery.
        super().__init__()
        # act = ... converts the string to a TF activation function.
        act = _get_activation(net_params.activation)
        # Creates a Python list of Dense layers.
        self.hidden = [
            tf.keras.layers.Dense(net_params.hidden_units, activation=act)
            for _ in range(net_params.hidden_layers)
        ]
        # If hidden_layers=2 and hidden_units=64, then you get:
        # Dense(64, activation=act)
        # Dense(64, activation=act)
        # Each Dense expects input [N, d] and outputs [N, hidden_units]
        # So the pipeline is:
        # input x: [N,2], 2” is the number of input variables (features)
        # after first Dense: [N,64]
        # after second Dense: [N,64]
        self.out = tf.keras.layers.Dense(1, activation=None)
        # A final Dense layer producing one number per sample.
        # Output shape before squeezing: [N, 1]
        # No activation: you want raw output first.
        self.k_min = tf.constant(k_min, tf.float32)
        self.k_max = tf.constant(k_max, tf.float32)
        # Stores k_min as a TensorFlow constant tensor.

    @tf.function(reduce_retracing=True)
    # Compiles the forward pass for speed.
    # In graph mode, this runs faster in training loops.
    def call(self, x: tf.Tensor) -> tf.Tensor:
        h = x
        # h = x starts with [N,2]
        for layer in self.hidden:
            h = layer(h)
            # Loop applies each Dense:
            # h = layer(h) transforms h each time.
        raw = self.out(h)  # [N,1]
        # raw = self.out(h) gives shape [N,1].
        # Values can be any real number.
        # kprime = tf.nn.softplus(raw) + self.k_min
        s = tf.nn.sigmoid(raw)
        kprime = self.k_min + (self.k_max - self.k_min) * s

        # Enforce positivity
        # softplus(u) = log(1 + exp(u)) which is always > 0
        # So softplus(raw) guarantees strictly positive
        return tf.squeeze(kprime, axis=1)  # Converts [N,1] → [N]
        # Because everywhere else you treat k and z as 1D vectors


class ValueNet(tf.keras.Model):
    # Similar structure, but no positivity constraint.
    """
    Value: V(k,z)
    Input: [N,2]
    Output: [N]
    """

    def __init__(self, net_params: NetParams):
        super().__init__()
        act = _get_activation(net_params.activation)
        self.hidden = [
            tf.keras.layers.Dense(net_params.hidden_units, activation=act)
            for _ in range(net_params.hidden_layers)
        ]
        self.out = tf.keras.layers.Dense(1, activation=None)

    @tf.function(reduce_retracing=True)
    def call(self, x: tf.Tensor) -> tf.Tensor:
        h = x
        for layer in self.hidden:
            h = layer(h)
        v = self.out(h)  # [N,1]
        return tf.squeeze(v, axis=1)

    # outputs v with shape [N], can be negative or positive
    # smooth activations matter
    # because you later compute: ∂𝑉(𝑘′,𝑧′)∂𝑘′​using autodiff.
    # Smooth activations like tanh produce smoother derivatives.

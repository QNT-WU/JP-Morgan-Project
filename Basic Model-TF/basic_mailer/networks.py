from __future__ import annotations


import tensorflow as tf
from .config import NetParams


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

    def __init__(self, net_params: NetParams, k_min: float):
        # super().__init__() initializes the Keras model machinery.
        super().__init__()
        # act = ... converts the string to a TF activation function.
        act = _get_activation(net_params.activation)
        # Creates a Python list of Dense layers.
        self.hidden = [
            tf.keras.layers.Dense(net_params.hidden_units, activation=act)
            for _ in range(net_params.hidden_layers)
        ]

        # Each Dense expects input [N, d] and outputs [N, hidden_units]
        self.out = tf.keras.layers.Dense(1, activation=None)
        # Output shape before squeezing: [N, 1]
        self.k_min = tf.constant(k_min, tf.float32)

    @tf.function
    # Compiles the forward pass for speed.
    def call(self, x: tf.Tensor) -> tf.Tensor:
        h = x
        # h = x starts with [N,2]
        for layer in self.hidden:
            h = layer(h)
            # h = layer(h) transforms h each time.
        raw = self.out(h)  # [N,1]
        # raw = self.out(h) gives shape [N,1].
        kprime = tf.nn.softplus(raw) + self.k_min
        # Enforce positivity
        # Softplus(raw) guarantees strictly positive
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

    @tf.function
    def call(self, x: tf.Tensor) -> tf.Tensor:
        h = x
        for layer in self.hidden:
            h = layer(h)
        v = self.out(h)  # [N,1]
        return tf.squeeze(v, axis=1)

    # outputs v with shape [N], can be negative or positive
    # smooth activations matter
    # because I later compute: ∂𝑉(𝑘′,𝑧′)∂𝑘′​using autodiff.

# risky_debt/netowrks.py
from __future__ import annotations

import tensorflow as tf
from .config import NetParams


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
    Policy: (k', b') = phi(k,b,z)
    Input: [N,3]
    Output: [N,2] = [k', b']
      - k' enforced > 0 via softplus + k_min
      - b' bounded to [b_min, b_max] via tanh squash
    """

    def __init__(self, net_params: NetParams, k_min: float, b_min: float, b_max: float):
        super().__init__()
        act = _get_activation(net_params.activation)
        self.hidden = [
            tf.keras.layers.Dense(net_params.hidden_units, activation=act)
            for _ in range(net_params.hidden_layers)
        ]
        self.out = tf.keras.layers.Dense(2, activation=None)  # raw (k', b')
        self.k_min = tf.constant(k_min, tf.float32)
        self.b_min = tf.constant(b_min, tf.float32)
        self.b_max = tf.constant(b_max, tf.float32)

    @tf.function
    def call(self, x: tf.Tensor) -> tf.Tensor:
        # Input x shape is [N,3] representing [k,b,z]
        # Output raw is [N,2]
        h = x
        for layer in self.hidden:
            h = layer(h)
        raw = self.out(h)  # [N,2]
        raw_k = raw[:, 0:1]
        raw_b = raw[:, 1:2]

        # Enforcing 𝑘′>0
        # softplus(u) = log(1+exp(u)) is always positive
        # Then add k_min so 𝑘′≥𝑘min⁡
        kprime = tf.nn.softplus(raw_k) + self.k_min  # [N,1]

        # tanh outputs between -1 and +1.
        # map tanh to [b_min,b_max]
        mid = 0.5 * (self.b_min + self.b_max)
        half = 0.5 * (self.b_max - self.b_min)
        bprime = mid + half * tf.tanh(raw_b)  # [N,1]

        # So PolicyNet outputs [k', b']
        return tf.concat([kprime, bprime], axis=1)  # [N,2]


class ValueNet(tf.keras.Model):
    """
    Equity value V(k,b,z) >= 0.
    Input: [N,3]
    Output: [N]
    """

    def __init__(self, net_params: NetParams):
        super().__init__()
        act = _get_activation(net_params.activation)
        self.hidden = [
            tf.keras.layers.Dense(net_params.hidden_units, activation=act)
            for _ in range(net_params.hidden_layers)
        ]
        # Same hidden layers style, output layer size 1
        self.out = tf.keras.layers.Dense(1, activation=None)

    @tf.function
    def call(self, x: tf.Tensor) -> tf.Tensor:
        h = x
        for layer in self.hidden:
            h = layer(h)
        # softplus ensures output is nonnegative
        # squeeze turns shape [N,1] to [N]
        raw = self.out(h)  # [N,1]
        v = tf.nn.softplus(raw)  # enforce >=0
        return tf.squeeze(v, axis=1)


class VtildeNet(tf.keras.Model):
    """
    Continuation (pre-default) value \\tilde V(k,b,z), unconstrained real.
    Input: [N,3]
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
        # No softplus. So it can be negative or positive
        raw = self.out(h)  # [N,1]
        return tf.squeeze(raw, axis=1)


class PricingNet(tf.keras.Model):
    """
    Pricing: q(z, k', b') in (0,1), bounded to [q_min,q_max].
    Input: [N,3] = [z, k', b']
    Output: [N]
    """

    def __init__(self, net_params: NetParams, q_min: float, q_max: float):
        super().__init__()
        act = _get_activation(net_params.activation)
        self.hidden = [
            tf.keras.layers.Dense(net_params.hidden_units, activation=act)
            for _ in range(net_params.hidden_layers)
        ]
        self.out = tf.keras.layers.Dense(1, activation=None)
        self.q_min = tf.constant(q_min, tf.float32)
        self.q_max = tf.constant(q_max, tf.float32)

    @tf.function
    def call(self, x: tf.Tensor) -> tf.Tensor:
        h = x
        for layer in self.hidden:
            h = layer(h)
        # sigmoid gives something in (0,1)
        raw = self.out(h)  # [N,1]
        s = tf.sigmoid(raw)
        q = self.q_min + (self.q_max - self.q_min) * s
        return tf.squeeze(q, axis=1)

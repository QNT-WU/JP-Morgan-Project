"""TensorFlow model classes used by the risky-debt objectives.

The package uses subclassed ``tf.keras.Model`` modules instead of loose helper
functions so the training stack remains object-oriented, checkpointable, and
compatible with TensorFlow best practices.
"""

from __future__ import annotations

from typing import Sequence

import tensorflow as tf

from .config import NetParams
from .layers import (
    BoundedScalarHead,
    BoundedTanhHead,
    FeatureStandardization,
    PositiveScalarHead,
)


def _get_activation(name: str):
    """Return the TensorFlow activation function requested by name."""
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


def _vector3_or_default(value, default_values: Sequence[float]) -> tf.Tensor:
    """Return a length-three ``float32`` tensor with defaults when absent."""
    if value is None:
        return tf.constant(default_values, dtype=tf.float32)
    tensor = tf.convert_to_tensor(value, dtype=tf.float32)
    return tf.reshape(tensor, (3,))


class DenseBackbone(tf.keras.layers.Layer):
    """Shared fully-connected backbone for policy, value, and pricing models."""

    def __init__(self, net_params: NetParams, input_center=None, input_scale=None, **kwargs) -> None:
        """Initialize DenseBackbone."""
        super().__init__(**kwargs)
        activation = _get_activation(net_params.activation)
        center = _vector3_or_default(input_center, [0.0, 0.0, 0.0])
        scale = _vector3_or_default(input_scale, [1.0, 1.0, 1.0])
        self.standardize = FeatureStandardization(center=center, scale=scale)
        self.hidden_layers = [
            tf.keras.layers.Dense(net_params.hidden_units, activation=activation)
            for _ in range(net_params.hidden_layers)
        ]

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Encode a batch of state vectors into hidden features."""
        hidden = self.standardize(inputs)
        for layer in self.hidden_layers:
            hidden = layer(hidden)
        return hidden


class PolicyNet(tf.keras.Model):
    """Policy network returning bounded controls ``(k', b')``."""

    def __init__(
        self,
        net_params: NetParams,
        k_min: float,
        b_min: float,
        b_max: float,
        input_center=None,
        input_scale=None,
    ) -> None:
        """Initialize PolicyNet."""
        super().__init__()
        self.backbone = DenseBackbone(
            net_params,
            input_center=input_center,
            input_scale=input_scale,
            name="policy_backbone",
        )
        self.raw_head = tf.keras.layers.Dense(2, activation=None, name="policy_raw")
        self.k_head = PositiveScalarHead(floor=k_min, name="kprime_head")
        self.b_head = BoundedTanhHead(lower=b_min, upper=b_max, name="bprime_head")

    @tf.function
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Return policy controls in level units for a batch of states."""
        features = self.backbone(x)
        raw = self.raw_head(features)
        kprime = self.k_head(raw[:, 0:1])
        bprime = self.b_head(raw[:, 1:2])
        return tf.concat([kprime, bprime], axis=1)


class ValueNet(tf.keras.Model):
    """Non-negative equity value network ``V(k,b,z)``."""

    def __init__(self, net_params: NetParams, input_center=None, input_scale=None) -> None:
        """Initialize ValueNet."""
        super().__init__()
        self.backbone = DenseBackbone(
            net_params,
            input_center=input_center,
            input_scale=input_scale,
            name="value_backbone",
        )
        self.raw_head = tf.keras.layers.Dense(1, activation=None, name="value_raw")
        self.value_head = PositiveScalarHead(floor=0.0, name="value_head")

    @tf.function
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Return non-negative equity values for a batch of states."""
        features = self.backbone(x)
        value = self.value_head(self.raw_head(features))
        return tf.squeeze(value, axis=1)


class VtildeNet(tf.keras.Model):
    """Unconstrained continuation-value network ``\tilde V(k,b,z)``."""

    def __init__(self, net_params: NetParams, input_center=None, input_scale=None) -> None:
        """Initialize VtildeNet."""
        super().__init__()
        self.backbone = DenseBackbone(
            net_params,
            input_center=input_center,
            input_scale=input_scale,
            name="vtilde_backbone",
        )
        self.raw_head = tf.keras.layers.Dense(1, activation=None, name="vtilde_raw")

    @tf.function
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Return unconstrained continuation values for a batch of states."""
        features = self.backbone(x)
        return tf.squeeze(self.raw_head(features), axis=1)


class MultiplierNet(tf.keras.Model):
    """Non-negative multiplier network for the capital lower-bound KKT block."""

    def __init__(self, net_params: NetParams, input_center=None, input_scale=None) -> None:
        """Initialize MultiplierNet."""
        super().__init__()
        self.backbone = DenseBackbone(
            net_params,
            input_center=input_center,
            input_scale=input_scale,
            name="multiplier_backbone",
        )
        self.raw_head = tf.keras.layers.Dense(1, activation=None, name="lambda_k_raw")
        self.lambda_head = PositiveScalarHead(floor=0.0, name="lambda_k_head")

    @tf.function
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Return non-negative KKT multipliers for a batch of states."""
        features = self.backbone(x)
        lam = self.lambda_head(self.raw_head(features))
        return tf.squeeze(lam, axis=1)


class ConstructedPricingCompatibilityNet(tf.keras.Model):
    """Compatibility TensorFlow module for legacy pricing-network APIs.

    Economic pricing is constructed in :mod:`risky_debt.pricing` from the
    lender zero-profit condition.  This lightweight bounded network is kept so
    older trainers, tests, checkpoints, and public function signatures that
    expect a ``qnet`` object continue to work without treating the network as
    the economic debt-pricing rule.
    """

    def __init__(
        self,
        net_params: NetParams,
        q_min: float,
        q_max: float,
        input_center=None,
        input_scale=None,
    ) -> None:
        """Initialize the compatibility pricing module."""
        super().__init__()
        self.backbone = DenseBackbone(
            net_params,
            input_center=input_center,
            input_scale=input_scale,
            name="pricing_backbone",
        )
        self.raw_head = tf.keras.layers.Dense(1, activation=None, name="pricing_raw")
        self.pricing_head = BoundedScalarHead(lower=q_min, upper=q_max, name="pricing_head")

    @tf.function
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Return bounded compatibility prices for legacy callers.

        The returned tensor is not used as the structural debt price in the
        aligned risky-debt objectives; constructed zero-profit pricing is used
        instead.
        """
        features = self.backbone(x)
        q = self.pricing_head(self.raw_head(features))
        return tf.squeeze(q, axis=1)


# Backward-compatible alias.  Public code may still import ``PricingNet``, but
# the class name now documents that this object is a compatibility module rather
# than the economic pricing rule.
PricingNet = ConstructedPricingCompatibilityNet

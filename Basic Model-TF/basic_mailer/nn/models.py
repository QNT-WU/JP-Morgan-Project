"""TensorFlow model classes for the basic Mailer package."""

from __future__ import annotations

from dataclasses import asdict

import tensorflow as tf

from basic_mailer.config import NetParams
from .layers import BoundedPolicyHead, MLPBlock, StateNormalization


@tf.keras.utils.register_keras_serializable(package="basic_mailer")
class PolicyModel(tf.keras.Model):
    """Policy network for next-period capital.

    The model maps the current state ``(k_t, z_t)`` to the feasible choice
    ``k_{t+1}``. It is intentionally small and explicit so it can be reused in
    both training and estimation code.
    """

    def __init__(self, net_params: NetParams, k_min: float, k_max: float, **kwargs):
        """Construct the policy model.

        Args:
            net_params: Architectural choices for the shared MLP backbone.
            k_min: Lower feasible bound for next-period capital.
            k_max: Upper feasible bound for next-period capital.
            **kwargs: Extra keyword arguments forwarded to ``Model``.
        """
        kwargs.setdefault("name", "policy_model")
        super().__init__(**kwargs)
        self.net_params = net_params
        self.k_min = float(k_min)
        self.k_max = float(k_max)
        self.state_normalizer = StateNormalization()
        self.backbone = MLPBlock(
            hidden_units=net_params.hidden_units,
            hidden_layers=net_params.hidden_layers,
            activation=net_params.activation,
        )
        self.head = BoundedPolicyHead(k_min=k_min, k_max=k_max)

    @tf.function(reduce_retracing=True)
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Return feasible next-period capital values.

        Args:
            inputs: Tensor of shape ``[batch_size, 2]``.
            training: Whether Keras is in training mode.

        Returns:
            A rank-1 tensor containing the policy-implied capital choices.
        """
        x = self.state_normalizer(inputs)
        h = self.backbone(x, training=training)
        return self.head(h, training=training)

    def get_config(self) -> dict:
        """Return a JSON-serializable configuration for Keras saving."""
        config = super().get_config()
        config.update(
            {
                "net_params": asdict(self.net_params),
                "k_min": self.k_min,
                "k_max": self.k_max,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict):
        """Reconstruct the model from a serialized Keras configuration."""
        net_params = NetParams(**config.pop("net_params"))
        return cls(net_params=net_params, **config)


@tf.keras.utils.register_keras_serializable(package="basic_mailer")
class ValueModel(tf.keras.Model):
    """Value network for the basic ``(k, z)`` state space."""

    def __init__(self, net_params: NetParams, **kwargs):
        """Construct the value model.

        Args:
            net_params: Architectural choices for the shared MLP backbone.
            **kwargs: Extra keyword arguments forwarded to ``Model``.
        """
        kwargs.setdefault("name", "value_model")
        super().__init__(**kwargs)
        self.net_params = net_params
        self.state_normalizer = StateNormalization()
        self.backbone = MLPBlock(
            hidden_units=net_params.hidden_units,
            hidden_layers=net_params.hidden_layers,
            activation=net_params.activation,
        )
        self.readout = tf.keras.layers.Dense(1, activation=None, name="value_readout")

    @tf.function(reduce_retracing=True)
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Return scalar values for a batch of states.

        Args:
            inputs: Tensor of shape ``[batch_size, 2]``.
            training: Whether Keras is in training mode.

        Returns:
            A rank-1 tensor containing one value estimate per state.
        """
        x = self.state_normalizer(inputs)
        h = self.backbone(x, training=training)
        v = self.readout(h, training=training)
        return tf.squeeze(v, axis=-1)

    def get_config(self) -> dict:
        """Return a JSON-serializable configuration for Keras saving."""
        config = super().get_config()
        config.update({"net_params": asdict(self.net_params)})
        return config

    @classmethod
    def from_config(cls, config: dict):
        """Reconstruct the model from a serialized Keras configuration."""
        net_params = NetParams(**config.pop("net_params"))
        return cls(net_params=net_params, **config)


PolicyNet = PolicyModel
ValueNet = ValueModel

"""Reusable TensorFlow layers for the basic Mailer model package.

These layers provide a small, testable abstraction layer around common neural
network components used across policy and value networks.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import tensorflow as tf


class ActivationFactory:
    """Map user-facing activation names to TensorFlow activation callables."""

    _ALIASES = {
        "tanh": tf.nn.tanh,
        "elu": tf.nn.elu,
        "softplus": tf.nn.softplus,
        "relu": tf.nn.relu,
    }

    @classmethod
    def get(cls, name: str):
        """Return the activation function corresponding to ``name``.

        Args:
            name: Case-insensitive activation label.

        Raises:
            ValueError: If the activation name is unknown.
        """
        key = name.lower().strip()
        try:
            return cls._ALIASES[key]
        except KeyError as exc:
            known = ", ".join(sorted(cls._ALIASES))
            raise ValueError(f"Unknown activation '{name}'. Known activations: {known}") from exc


@tf.keras.utils.register_keras_serializable(package="basic_mailer")
class StateNormalization(tf.keras.layers.Layer):
    """Normalize the two-dimensional state vector ``(k, z)``.

    The running statistics are stored as non-trainable TensorFlow variables so
    they participate in checkpointing and Keras serialization while remaining
    fixed during gradient-based optimization.
    """

    def __init__(
        self,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        eps: float = 1e-6,
        **kwargs,
    ):
        """Initialize the normalization layer.

        Args:
            mean: Optional initial state means for ``k`` and ``z``.
            std: Optional initial state standard deviations for ``k`` and ``z``.
            eps: Numerical stabilizer added to the denominator.
            **kwargs: Extra keyword arguments forwarded to ``Layer``.
        """
        kwargs.setdefault("name", "state_normalization")
        super().__init__(**kwargs)
        default_mean = tf.constant(mean if mean is not None else [1.0, 1.0], dtype=tf.float32)
        default_std = tf.constant(std if std is not None else [1.0, 1.0], dtype=tf.float32)
        self._eps_value = float(eps)
        self._eps = tf.constant(eps, dtype=tf.float32)
        self.mean = self.add_weight(
            name="state_mean",
            shape=(2,),
            initializer=tf.keras.initializers.Constant(default_mean.numpy()),
            trainable=False,
        )
        self.std = self.add_weight(
            name="state_std",
            shape=(2,),
            initializer=tf.keras.initializers.Constant(default_std.numpy()),
            trainable=False,
        )

    def update_statistics(self, mean: Iterable[float], std: Iterable[float]) -> None:
        """Update normalization statistics in-place.

        Args:
            mean: Mean values for the state coordinates.
            std: Standard deviations for the state coordinates.
        """
        self.mean.assign(tf.convert_to_tensor(list(mean), dtype=tf.float32))
        self.std.assign(tf.convert_to_tensor(list(std), dtype=tf.float32))

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Normalize an input state batch.

        Args:
            inputs: Tensor with shape ``[batch_size, 2]``.

        Returns:
            A normalized state tensor with the same shape as ``inputs``.
        """
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        return (inputs - self.mean) / (self.std + self._eps)

    def get_config(self) -> dict:
        """Return a JSON-serializable configuration for Keras saving."""
        config = super().get_config()
        config.update(
            {
                "mean": self.mean.numpy().tolist(),
                "std": self.std.numpy().tolist(),
                "eps": self._eps_value,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="basic_mailer")
class MLPBlock(tf.keras.layers.Layer):
    """A simple fully connected stack used by policy and value networks."""

    def __init__(self, hidden_units: int, hidden_layers: int, activation: str, **kwargs):
        """Construct the shared multilayer perceptron backbone.

        Args:
            hidden_units: Width of each dense layer.
            hidden_layers: Number of dense hidden layers.
            activation: Activation name resolved by :class:`ActivationFactory`.
            **kwargs: Extra keyword arguments forwarded to ``Layer``.
        """
        kwargs.setdefault("name", "mlp_block")
        super().__init__(**kwargs)
        self.hidden_units = int(hidden_units)
        self.hidden_layers_count = int(hidden_layers)
        self.activation_name = activation
        act = ActivationFactory.get(activation)
        self.hidden_layers = [
            tf.keras.layers.Dense(self.hidden_units, activation=act, name=f"dense_{idx}")
            for idx in range(self.hidden_layers_count)
        ]

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Apply the block to ``inputs``.

        Args:
            inputs: Input tensor.
            training: Included for Keras compatibility.

        Returns:
            The hidden representation produced by the MLP stack.
        """
        h = inputs
        for layer in self.hidden_layers:
            h = layer(h, training=training)
        return h

    def get_config(self) -> dict:
        """Return a JSON-serializable configuration for Keras saving."""
        config = super().get_config()
        config.update(
            {
                "hidden_units": self.hidden_units,
                "hidden_layers": self.hidden_layers_count,
                "activation": self.activation_name,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="basic_mailer")
class BoundedPolicyHead(tf.keras.layers.Layer):
    """Map an unconstrained scalar to the feasible capital interval.

    The layer uses a sigmoid transform to map the raw scalar output to the
    closed interval ``[k_min, k_max]``.
    """

    def __init__(self, k_min: float, k_max: float, **kwargs):
        """Initialize the bounded output head.

        Args:
            k_min: Lower feasible bound for next-period capital.
            k_max: Upper feasible bound for next-period capital.
            **kwargs: Extra keyword arguments forwarded to ``Layer``.
        """
        kwargs.setdefault("name", "bounded_policy_head")
        super().__init__(**kwargs)
        self.k_min_value = float(k_min)
        self.k_max_value = float(k_max)
        self.k_min = tf.constant(k_min, dtype=tf.float32)
        self.k_max = tf.constant(k_max, dtype=tf.float32)
        self.readout = tf.keras.layers.Dense(1, activation=None, name="policy_readout")

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Return bounded next-period capital levels.

        Args:
            inputs: Hidden representation tensor.
            training: Included for Keras compatibility.

        Returns:
            One feasible capital choice per input row.
        """
        raw = self.readout(inputs, training=training)
        scaled = tf.nn.sigmoid(raw)
        k_next = self.k_min + (self.k_max - self.k_min) * scaled
        return tf.squeeze(k_next, axis=-1)

    def get_config(self) -> dict:
        """Return a JSON-serializable configuration for Keras saving."""
        config = super().get_config()
        config.update({"k_min": self.k_min_value, "k_max": self.k_max_value})
        return config

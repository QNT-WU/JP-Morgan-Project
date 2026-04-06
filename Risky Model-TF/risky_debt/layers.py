"""Custom TensorFlow layers used across the risky-debt package.

These layers keep the neural-network code modular and make the TensorFlow
components explicit for client delivery. They also showcase TensorFlow-native
state management through ``tf.Variable`` and bounded/positive output heads.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import tensorflow as tf


class FeatureStandardization(tf.keras.layers.Layer):
    """Apply fixed affine standardization to a vector input.

    Parameters
    ----------
    center:
        Feature-wise centering vector.
    scale:
        Feature-wise scaling vector.
    eps:
        Small floor used to avoid division by zero.

    Notes
    -----
    The centering and scaling statistics are stored as non-trainable
    ``tf.Variable`` objects so they travel with checkpoints and are visible as
    explicit TensorFlow state in the codebase.
    """

    def __init__(self, center: Sequence[float], scale: Sequence[float], eps: float = 1e-6, **kwargs) -> None:
        """Initialize FeatureStandardization."""
        super().__init__(**kwargs)
        center_tensor = tf.reshape(tf.convert_to_tensor(center, dtype=tf.float32), (-1,))
        scale_tensor = tf.reshape(tf.convert_to_tensor(scale, dtype=tf.float32), (-1,))
        self.center = tf.Variable(center_tensor, trainable=False, name="center")
        self.scale = tf.Variable(scale_tensor, trainable=False, name="scale")
        self.eps = tf.constant(eps, dtype=tf.float32)

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Return the normalized input tensor."""
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        denom = tf.maximum(self.scale, self.eps)
        return (inputs - self.center) / denom


class PositiveScalarHead(tf.keras.layers.Layer):
    """Map raw activations to strictly positive outputs with a lower floor."""

    def __init__(self, floor: float = 0.0, **kwargs) -> None:
        """Initialize PositiveScalarHead."""
        super().__init__(**kwargs)
        self.floor = tf.constant(floor, dtype=tf.float32)

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply a softplus transformation and add the configured floor."""
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        return tf.nn.softplus(inputs) + self.floor


class BoundedScalarHead(tf.keras.layers.Layer):
    """Map raw activations to a closed interval using a sigmoid squashing map."""

    def __init__(self, lower: float, upper: float, **kwargs) -> None:
        """Initialize BoundedScalarHead."""
        super().__init__(**kwargs)
        self.lower = tf.constant(lower, dtype=tf.float32)
        self.upper = tf.constant(upper, dtype=tf.float32)

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Return the bounded output tensor."""
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        return self.lower + (self.upper - self.lower) * tf.sigmoid(inputs)


class BoundedTanhHead(tf.keras.layers.Layer):
    """Map raw activations to a symmetric interval via ``tanh`` re-scaling."""

    def __init__(self, lower: float, upper: float, **kwargs) -> None:
        """Initialize BoundedTanhHead."""
        super().__init__(**kwargs)
        self.lower = tf.constant(lower, dtype=tf.float32)
        self.upper = tf.constant(upper, dtype=tf.float32)
        self.mid = tf.constant(0.5 * (lower + upper), dtype=tf.float32)
        self.half = tf.constant(0.5 * (upper - lower), dtype=tf.float32)

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Return the bounded output tensor."""
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        return self.mid + self.half * tf.tanh(inputs)

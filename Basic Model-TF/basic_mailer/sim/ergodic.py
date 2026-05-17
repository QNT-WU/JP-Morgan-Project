"""Simulation utilities for policy-induced ergodic datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import tensorflow as tf

from basic_mailer.config import ModelParams, TrainParams
from basic_mailer.nn.models import PolicyNet
from basic_mailer.primitives import shock_next_z


@dataclass
class ErgodicDataset:
    """Container for a flattened ergodic sample."""

    k: np.ndarray
    z: np.ndarray


class ErgodicSimulator:
    """Generate ergodic-state datasets under the current policy.

    The simulator encapsulates the policy-induced Markov chain and provides a
    TensorFlow-friendly implementation that uses ``tf.TensorArray`` inside the
    compiled simulation loop.
    """

    def __init__(self, policy: PolicyNet, mp: ModelParams, tp: TrainParams):
        """Initialize the compiled ergodic simulator and cache fixed constants.

        The simulator stores model and training parameters as TensorFlow
        constants so repeated trajectory generation can run inside compiled
        ``tf.function`` kernels without Python-side reconfiguration.
        """
        self.policy = policy
        self.mp = mp
        self.tp = tp
        self.k_min = tf.constant(mp.k_min, dtype=tf.float32)
        self.k_max = tf.constant(mp.k_max, dtype=tf.float32)

    @tf.function(reduce_retracing=True)
    def _simulate_paths(self, k0: tf.Tensor, z0: tf.Tensor, total_steps: int, burn_in: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """Simulate the Markov chain and collect post-burn-in states.

        Args:
            k0: Initial capital values with shape ``[n_paths]``.
            z0: Initial productivity values with shape ``[n_paths]``.
            total_steps: Total number of simulation steps, including burn-in.
            burn_in: Number of initial steps to discard.
        """
        n_keep = total_steps - burn_in + 1
        k_hist = tf.TensorArray(dtype=tf.float32, size=n_keep, clear_after_read=False)
        z_hist = tf.TensorArray(dtype=tf.float32, size=n_keep, clear_after_read=False)

        k = tf.convert_to_tensor(k0, dtype=tf.float32)
        z = tf.convert_to_tensor(z0, dtype=tf.float32)
        keep_index = tf.constant(0, dtype=tf.int32)

        for step in tf.range(total_steps + 1):
            if step >= burn_in:
                k_hist = k_hist.write(keep_index, k)
                z_hist = z_hist.write(keep_index, z)
                keep_index += 1

            state = tf.stack([k, z], axis=1)
            k_next = tf.clip_by_value(self.policy(state, training=False), self.k_min, self.k_max)
            z_next = shock_next_z(z, self.mp.rho, self.mp.sigma_eps)
            k = k_next
            z = z_next

        return k_hist.stack(), z_hist.stack()

    def simulate(self, seed: int) -> ErgodicDataset:
        """Return a flattened ergodic sample as NumPy arrays.

        Args:
            seed: Random seed for both TensorFlow and NumPy.
        """
        tf.random.set_seed(seed)
        np.random.seed(seed)

        n_paths = self.tp.ergodic_n_paths
        total_steps = self.tp.ergodic_burn_in + self.tp.ergodic_T

        k0 = tf.random.uniform((n_paths,), minval=0.5, maxval=2.0, dtype=tf.float32)
        z0 = tf.random.uniform((n_paths,), minval=0.5, maxval=2.0, dtype=tf.float32)
        k_hist, z_hist = self._simulate_paths(k0, z0, total_steps, self.tp.ergodic_burn_in)

        k_flat = tf.reshape(k_hist, (-1,)).numpy()
        z_flat = tf.reshape(z_hist, (-1,)).numpy()

        if k_flat.shape[0] > self.tp.ergodic_buffer_size:
            idx = np.random.choice(k_flat.shape[0], size=self.tp.ergodic_buffer_size, replace=False)
            k_flat = k_flat[idx]
            z_flat = z_flat[idx]

        return ErgodicDataset(k=k_flat, z=z_flat)

"""Backward-compatible simulation helpers.

The public API remains the same, but the underlying implementation now lives in
:mod:`basic_mailer.sim` and is organized around the :class:`ErgodicSimulator`
class.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf

from .config import ModelParams, TrainParams
from .networks import PolicyNet
from .sim import ErgodicSimulator


def set_global_seed(seed: int) -> None:
    """Set TensorFlow and NumPy random seeds."""
    tf.random.set_seed(seed)
    np.random.seed(seed)


def simulate_ergodic_dataset(
    policy: PolicyNet,
    mp: ModelParams,
    tp: TrainParams,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate a policy-induced ergodic dataset.

    Args:
        policy: Policy network used to transition capital.
        mp: Structural model parameters.
        tp: Training and simulation controls.
        seed: Random seed.

    Returns:
        A pair ``(k_flat, z_flat)`` of one-dimensional NumPy arrays.
    """
    simulator = ErgodicSimulator(policy=policy, mp=mp, tp=tp)
    dataset = simulator.simulate(seed=seed)
    return dataset.k, dataset.z

"""Simulation utilities for ergodic datasets and synthetic panels."""

from __future__ import annotations

from typing import Tuple
import numpy as np
import tensorflow as tf

from .config import ModelParams, TrainParams
from .networks import PolicyNet
from .primitives import shock_next_z, solvency_weight

import random


# set_global_seed(seed)
# It sets the random seed for:
# TensorFlow RNG: affects tf.random.uniform, tf.random.normal, etc.
# NumPy RNG: affects np.random.choice, etc.
# It aims to make simulated paths (and ergodic sample) reproducible.
# But: this alone does not guarantee full reproducibility in TF (more later).
"""
def set_global_seed(seed: int) -> None:
    tf.random.set_seed(seed)
    np.random.seed(seed)
"""


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow random number generators."""
    # sets python, numpy, tf together (Keras included)
    tf.keras.utils.set_random_seed(seed)
    # keep explicit for clarity
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


# What it’s supposed to do (your words)
# Given a fixed policy rule (k′,b′)=φ(k,b,z), and shock law for z,
# generate the ergodic distribution of (k,b,z): i.e., “where the system spends time” after it settles down.
def simulate_ergodic_dataset(
    policy: PolicyNet,
    mp: ModelParams,
    tp: TrainParams,
    seed: int,
    record_mode: str = "all",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate policy-induced Markov chain on (k,b,z), store post burn-in states.

    record_mode:
        "all"           -> record every post burn-in state (legacy behavior)
        "continuation"  -> record only states whose next-step solvency proxy
                            implies continuation (s_{t+1} > 0.5)
    Returns flattened (k,b,z) arrays from the empirical ergodic distribution.
    """
    set_global_seed(seed)

    # initialize many parallel paths
    n = tp.ergodic_n_paths
    k = tf.random.uniform((n,), tp.k0_low, tp.k0_high, dtype=tf.float32)
    b = tf.random.uniform((n,), tp.b0_low, tp.b0_high, dtype=tf.float32)
    z = tf.random.uniform((n,), tp.z0_low, tp.z0_high, dtype=tf.float32)

    # allocate storage
    ks, bs, zs = [], [], []
    total_steps = tp.ergodic_burn_in + tp.ergodic_T

    # run the Markov chain forward
    for t in range(total_steps + 1):
        if record_mode not in {"all", "continuation"}:
            raise ValueError("record_mode must be 'all' or 'continuation'")

        # Apply policy:(k′,b′)=φ(k,b,z)
        # So x is [n,3] = [𝑘,𝑏,𝑧] per row.
        # policy(x) returns [n,2] = [𝑘′,𝑏′]per row.
        # clip 𝑘′≥𝑘𝑚𝑖𝑛	​
        # eep 𝑏′as output
        x = tf.stack([k, b, z], axis=1)  # [n,3]
        kb_next = policy(x)  # [n,2]
        k_next = tf.maximum(kb_next[:, 0], mp.k_min)
        b_next = kb_next[:, 1]

        # Shock transition: 𝑧𝑡+1
        z_next = shock_next_z(z, mp.rho, mp.sigma_eps)

        if t >= tp.ergodic_burn_in:
            if record_mode == "all":
                ks.append(k)
                bs.append(b)
                zs.append(z)
            else:
                cont_mask = solvency_weight(k_next, b_next, z_next, mp, tp.kappa_solv) > 0.5
                ks.append(tf.boolean_mask(k, cont_mask))
                bs.append(tf.boolean_mask(b, cont_mask))
                zs.append(tf.boolean_mask(z, cont_mask))

        # Update State:
        # So you step the Markov chain:(k,b,z)←(k′,b′,z′)
        k, b, z = k_next, b_next, z_next

    # Step E — stack saved lists into matrices
    # Each of these has shape:
    if len(ks) == 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    if record_mode == "all":
        k_mat = tf.stack(ks, axis=1)
        b_mat = tf.stack(bs, axis=1)
        z_mat = tf.stack(zs, axis=1)

        k_flat = tf.reshape(k_mat, (-1,)).numpy()
        b_flat = tf.reshape(b_mat, (-1,)).numpy()
        z_flat = tf.reshape(z_mat, (-1,)).numpy()
    else:
        k_flat = tf.concat(ks, axis=0).numpy() if ks else np.zeros((0,), dtype=np.float32)
        b_flat = tf.concat(bs, axis=0).numpy() if bs else np.zeros((0,), dtype=np.float32)
        z_flat = tf.concat(zs, axis=0).numpy() if zs else np.zeros((0,), dtype=np.float32)

    if k_flat.shape[0] > tp.ergodic_buffer_size:
        idx = np.random.choice(
            k_flat.shape[0], size=tp.ergodic_buffer_size, replace=False
        )
        k_flat = k_flat[idx]
        b_flat = b_flat[idx]
        z_flat = z_flat[idx]

    return k_flat, b_flat, z_flat

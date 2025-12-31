from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf

from .config import ModelParams, TrainParams
from .networks import PolicyNet
from .primitives import shock_next_z


# This sets both: TensorFlow RNG seed and NumPy RNG seed
def set_global_seed(seed: int) -> None:
    tf.random.set_seed(seed)
    np.random.seed(seed)


# Inputs:
# policy: your current policy network 𝑘′=𝜑(𝑘,𝑧)
# mp: economic primitives (k_min, rho, sigma)
# tp: simulation controls (burn-in, T, n_paths, buffer_size)
# seed: for reproducibility
# Output:(k_flat, z_flat) as NumPy arrays, both 1D.
def simulate_ergodic_dataset(
    policy: PolicyNet,
    mp: ModelParams,
    tp: TrainParams,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate under policy:
      k_{t+1} = phi(k_t,z_t)
      z_{t+1} = exp(rho ln z_t + eps)
    After burn-in, keep (k_t,z_t) as ergodic sample.
    Returns flattened arrays.
    """

    # set seed
    set_global_seed(seed)

    # Step 1: choose number of parallel paths
    n = tp.ergodic_n_paths

    # Step 2: initialize state for each path
    # k shape: [n], z shape: [n]
    # Each index i corresponds to one chain:(k[i], z[i]) is chain i’s current state
    k = tf.random.uniform((n,), 0.5, 2.0, dtype=tf.float32)
    z = tf.random.uniform((n,), 0.5, 2.0, dtype=tf.float32)

    # Step 3: buffers to collect post burn-in states
    ks, zs = [], []
    # Step 4: number of steps
    # run for burn-in steps + keep steps
    # If burn-in = 2000 and T=10000:
    # simulate 12000 steps total, store states from steps 2000..12000
    total_steps = tp.ergodic_burn_in + tp.ergodic_T

    # Step 5: the simulation loop
    for t in range(total_steps + 1):
        # After burn-in: append k and z
        # Each appended k is shape [n]
        if t >= tp.ergodic_burn_in:
            ks.append(k)
            zs.append(z)

        # compute next k via policy
        # x shape is [n,2].
        # policy(x) returns [n] (next capital for each chain)
        # clamp by k_min
        x = tf.stack([k, z], axis=1)
        k_next = tf.maximum(policy(x), mp.k_min)
        # compute next z via shock transition
        # z_next shape [n]
        # inside shock_next_z it draws eps shape [n]
        # so each chain gets its own shock
        # That is: zt+1(i)​=exp(ρlnzt(i)​+εt+1(i)​)
        z_next = shock_next_z(z, mp.rho, mp.sigma_eps)

        # update state
        k, z = k_next, z_next

    # convert the stored lists into matrices
    # each k in ks is [n]
    # stacking along axis=1 gives:
    # k_mat shape: [n, T_keep]
    # z_mat shape: [n, T_keep]
    # where T_keep = len(ks) = tp.ergodic_T + 1
    k_mat = tf.stack(ks, axis=1)
    z_mat = tf.stack(zs, axis=1)

    # flatten into one long sample
    # tf.reshape(..., (-1,)) turns [n, T_keep] into [n*T_keep]
    # then .numpy() converts to NumPy arrays
    # return:k_flat: shape [n*T_keep];z_flat: shape [n*T_keep]
    k_flat = tf.reshape(k_mat, (-1,)).numpy()
    z_flat = tf.reshape(z_mat, (-1,)).numpy()

    # randomly subsamples down to ergodic_buffer_size (e.g. 200,000)
    # This keeps memory bounded and makes training batches fast.
    if k_flat.shape[0] > tp.ergodic_buffer_size:
        idx = np.random.choice(
            k_flat.shape[0], size=tp.ergodic_buffer_size, replace=False
        )
        k_flat = k_flat[idx]
        z_flat = z_flat[idx]

    return k_flat, z_flat

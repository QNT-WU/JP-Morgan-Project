from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf

from .config import ModelParams, TrainParams
from .networks import PolicyNet, ValueNet
from .primitives import beta_from_r, reward_basic, shock_next_z


@tf.function
def rollout_discounted_reward(
    policy: PolicyNet,  # policy network φ(k,z).
    mp: ModelParams,  # mp: model parameters (rho, sigma, delta, etc.)
    k0: tf.Tensor,  # initial states (shape [N])
    z0: tf.Tensor,  # initial states (shape [N])
    T: int,  # horizon length
) -> tf.Tensor:
    # converts float β into a TF scalar tensor.
    beta = tf.constant(beta_from_r(mp.r), tf.float32)
    k = tf.maximum(k0, mp.k_min)  # k must not go to 0(division by k exists in rewards)
    z = tf.maximum(z0, 1e-12)
    # z must be > 0 to take log(shock process uses log), so they clamp to 1e-12

    # Initialize accumulators
    # W is the discounted lifetime reward per path(shape [N])
    W = tf.zeros_like(k)
    disc = tf.constant(1.0, tf.float32)

    # TF loop from 0 to T inclusive (so T+1 periods)
    for _ in tf.range(T + 1):
        # Inside each period
        # k and z are [N].
        # stack produces x of shape [N, 2](each row is [k_i, z_i])
        x = tf.stack([k, z], axis=1)

        # policy(x) outputs shape [N] or [N,1] depending on implementation
        # clamp again to ensure k_next ≥ k_min.
        k_next = tf.maximum(policy(x), mp.k_min)

        # Compute reward, multiply by current discount factor and add to W
        r_t = reward_basic(k, z, k_next, mp)
        W = W + disc * r_t

        # shock_next_z applies the AR(1) law for z
        # update (k,z) for next period.
        # update discount: disc ← disc * beta so it becomes \beta^(t+1)
        z_next = shock_next_z(z, mp.rho, mp.sigma_eps)
        k, z = k_next, z_next
        disc = disc * beta

    return W
    # return discounted reward for each of the N paths: shape [N].


# This computes TestReward (out-of-sample welfare)
def eval_test_reward(
    policy: PolicyNet, mp: ModelParams, tp: TrainParams, seed: int
) -> float:

    # Seed control
    # ensures reproducibility for: TF random draws, NumPy random draws
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Draw initial states
    # sample N test paths
    # initial states uniformly on [0.5,2.0]
    N = tp.N_paths_test
    k0 = tf.random.uniform((N,), 0.5, 2.0, dtype=tf.float32)
    z0 = tf.random.uniform((N,), 0.5, 2.0, dtype=tf.float32)

    # Rollout and average
    # W is [N] discounted reward.
    W = rollout_discounted_reward(policy, mp, k0, z0, tp.T_test)
    # reduce_mean yields scalar.
    # .numpy() to get python value.
    # float(...) makes it JSON-serializable.
    return float(tf.reduce_mean(W).numpy())


# -------- Obj2-style Euler residual f (policy-only) --------
@tf.function
# This computes the one-shock Euler residual f(k,z,ε) used in Obj2 and its tests.
def euler_f_policy_only(
    policy: PolicyNet,
    mp: ModelParams,
    k: tf.Tensor,
    z: tf.Tensor,
    eps: tf.Tensor,
) -> tf.Tensor:
    beta = tf.constant(beta_from_r(mp.r), tf.float32)

    k = tf.maximum(k, mp.k_min)
    z = tf.maximum(z, 1e-12)

    # Shock transition
    # eps is already drawn from 𝑁(0,𝜎2) elsewhere.
    z_next = tf.exp(mp.rho * tf.math.log(z) + eps)

    # k1 = k' = φ(k,z)
    # k2 = k'' = φ(k', z')
    # This is required because the RHS includes terms involving next period investment I
    x = tf.stack([k, z], axis=1)
    k1 = tf.maximum(policy(x), mp.k_min)

    x1 = tf.stack([k1, z_next], axis=1)
    k2 = tf.maximum(policy(x1), mp.k_min)

    # I = current investment.
    # I1 = next period investment.
    I = k1 - (1.0 - mp.delta) * k
    I1 = k2 - (1.0 - mp.delta) * k1

    # Left side of Euler
    left = 1.0 + mp.psi0 * (I / k)

    # Right side terms
    # derivative of production w.r.t capital: 𝜃𝑧′(𝑘′)^𝜃−1
    term_prod = mp.theta * z_next * tf.pow(k1, mp.theta - 1.0)
    # this corresponds to (1−𝛿) in the marginal value of installed capital.
    term_depr = 1.0 - mp.delta
    # adjustment-cost-related terms as in your Obj2 residual expression.
    term_adj1 = mp.psi0 * ((1.0 - mp.delta) * I1) / k1
    term_adj2 = mp.psi0 * (tf.square(I1)) / (2.0 * tf.square(k1))

    # Combine
    right = beta * (term_prod + term_depr + term_adj1 + term_adj2)

    return left - right


# This implements the Obj2 effectiveness metric:
def eval_test_euler_mse_policy_only(
    policy: PolicyNet,
    mp: ModelParams,
    states_k: np.ndarray,  # states_k, states_z: NumPy arrays of test states(length N_states)
    states_z: np.ndarray,
    N_eps: int,  # number of shocks per state
    seed: int,  # reproducible shocks
) -> float:
    """
    TestEulerMSE = mean_j ( (E_eps[f])^2 )
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)

    k = tf.convert_to_tensor(states_k, dtype=tf.float32)  # Convert states to tensors
    z = tf.convert_to_tensor(states_z, dtype=tf.float32)

    # Draw a matrix of shocks
    # shape [N_states, N_eps]
    # each state gets N_eps independent shocks
    eps = tf.random.normal(
        (tf.shape(k)[0], N_eps), mean=0.0, stddev=mp.sigma_eps, dtype=tf.float32
    )

    # Repeat k and z to align with shock draws
    k_rep = tf.repeat(k[:, None], repeats=N_eps, axis=1)
    z_rep = tf.repeat(z[:, None], repeats=N_eps, axis=1)
    # shapes become [N_states, N_eps]

    # Flatten to call euler_f_policy_only once
    # f becomes [N_states, N_eps]
    k_flat = tf.reshape(k_rep, (-1,))
    z_flat = tf.reshape(z_rep, (-1,))
    eps_flat = tf.reshape(eps, (-1,))

    f_flat = euler_f_policy_only(policy, mp, k_flat, z_flat, eps_flat)
    f = tf.reshape(f_flat, (tf.shape(k)[0], N_eps))

    # Approximate conditional expectation and MSE
    # Ef[j] ≈ E_eps[f|state j]
    # then compute average of Ef[j]^2.
    Ef = tf.reduce_mean(f, axis=1)
    mse = tf.reduce_mean(tf.square(Ef))
    return float(mse.numpy())


# -------- Obj3 Euler residual using dV/dk' --------
@tf.function
def euler_residual_with_value_derivative(
    policy: PolicyNet,
    value: ValueNet,
    mp: ModelParams,
    k: tf.Tensor,
    z: tf.Tensor,
    eps: tf.Tensor,
) -> tf.Tensor:
    beta = tf.constant(beta_from_r(mp.r), tf.float32)
    k = tf.maximum(k, mp.k_min)
    z = tf.maximum(z, 1e-12)

    # Policy step
    z_next = tf.exp(mp.rho * tf.math.log(z) + eps)
    x = tf.stack([k, z], axis=1)
    k_next = tf.maximum(policy(x), mp.k_min)
    I = k_next - (1.0 - mp.delta) * k

    # Key part: compute dV/dk' via autodiff
    # k_next is treated as an input variable we differentiate with respect to
    # tape.watch(k_next) tells TF to track it
    # Summing makes it scalar so gradient is well-defined for the batch;result dV_dk is a vector,one derivative per sample.
    with tf.GradientTape() as tape:
        tape.watch(k_next)
        xw = tf.stack([k_next, z_next], axis=1)
        v_next = value(xw)
        v_sum = tf.reduce_sum(v_next)
    dV_dk = tape.gradient(v_sum, k_next)

    return -1.0 - mp.psi0 * (I / k) + beta * dV_dk


def eval_test_euler_mse_obj3(
    policy: PolicyNet,
    value: ValueNet,
    mp: ModelParams,
    states_k: np.ndarray,
    states_z: np.ndarray,
    N_eps: int,
    seed: int,
) -> float:
    """
    TestEulerMSE_3 = mean_{j,ell} (R_E^{j,ell})^2
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Convert inputs
    k = tf.convert_to_tensor(states_k, tf.float32)
    z = tf.convert_to_tensor(states_z, tf.float32)
    N = int(k.shape[0])

    # Draw shocks
    eps = tf.random.normal((N, N_eps), mean=0.0, stddev=mp.sigma_eps, dtype=tf.float32)

    k_rep = tf.repeat(k[:, None], repeats=N_eps, axis=1)
    z_rep = tf.repeat(z[:, None], repeats=N_eps, axis=1)

    # Repeat and flatten
    k_flat = tf.reshape(k_rep, (-1,))
    z_flat = tf.reshape(z_rep, (-1,))
    eps_flat = tf.reshape(eps, (-1,))

    RE_flat = euler_residual_with_value_derivative(
        policy, value, mp, k_flat, z_flat, eps_flat
    )
    mse = tf.reduce_mean(tf.square(RE_flat))
    return float(mse.numpy())

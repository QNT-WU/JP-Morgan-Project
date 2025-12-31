# risky_debt/evaluation.py
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import tensorflow as tf

from .config import ModelParams, TrainParams
from .networks import PolicyNet, ValueNet, PricingNet
from .primitives import beta_from_r, equity_payout_d, solvency_weight


@tf.function
def rollout_discounted_reward(
    policy: PolicyNet,
    qnet: PricingNet,
    mp: ModelParams,
    tp: TrainParams,
    k0: tf.Tensor,
    b0: tf.Tensor,
    z0: tf.Tensor,
    T: int,
) -> tf.Tensor:

    # discounting: β=1/(1+r)​
    beta = tf.constant(beta_from_r(mp.r), tf.float32)
    # Ensures 𝑘>0 and 𝑧>0 numerically
    # b can be any real number
    k = tf.maximum(k0, mp.k_min)
    b = b0
    z = tf.maximum(z0, mp.z_min)

    # W: stores the accumulated discounted reward for each simulated path
    # disc: current discount factor 𝛽^𝑡

    W = tf.zeros_like(k)
    disc = tf.constant(1.0, tf.float32)

    ## alive: operating indicator (1 = still operating / not defaulted)
    alive = tf.ones_like(k, dtype=tf.float32)  # 1 if still operating

    # Simulate 𝑇+1 periods.
    for _ in tf.range(T + 1):

        # Build state vector 𝑥=(𝑘,𝑏,𝑧)
        # Apply policy net:(𝑘′,𝑏′)=𝜑(𝑘,𝑏,𝑧;𝜃)
        # Enforce 𝑘′>0
        x = tf.stack([k, b, z], axis=1)
        kb_next = policy(x)
        k_next = tf.maximum(kb_next[:, 0], mp.k_min)
        b_next = kb_next[:, 1]

        # This is P1 pricing: learn 𝑞𝜉(𝑧,𝑘′,𝑏′)
        q_in = tf.stack([z, k_next, b_next], axis=1)
        q = qnet(q_in)

        # compute one-period net payout 𝑑(⋅)
        # equity_payout_d implements your eq.(3.26) cash flow identity (in 𝑞 form)and issuance cost (3.14).
        # Multiply by alive: once default happens, you stop counting payouts (equity value becomes 0).
        d = equity_payout_d(k, k_next, b, b_next, z, q, mp, tp.kappa_issue)
        W = W + disc * (alive * d)

        # simulate shock process
        z_next = tf.exp(
            mp.rho * tf.math.log(tf.maximum(z, mp.z_min))
            + tf.random.normal(tf.shape(z), 0.0, mp.sigma_eps, tf.float32)
        )

        # decide “default / continue” for rollout (proxy)
        # The model says default occurs when continuation value is negative:
        # V(k,b,z)=max{0,V(k,b,z)}
        # But in Objective 1 I do not have a trained 𝑉~ and 𝑉 (by design),
        # so the code uses a solvency proxy solvency_weight(...) instead.
        # So “default rule” here is:if solvency proxy 𝑠𝑡+1≤0.5, set alive to 0 forever.
        s_next = solvency_weight(k_next, b_next, z_next, mp, tp.kappa_solv)
        alive = alive * tf.cast(s_next > 0.5, tf.float32)

        # update state and discount
        k, b, z = k_next, b_next, z_next
        disc = disc * beta

    return W
    # Returns discounted sum of payouts per path.


# sets RNG seeds for reproducibility
# draws many initial states from broad uniform ranges
# calls rollout_discounted_reward
# averages over paths
# TestReward = N_1​^i=1∑N​Wi​
def eval_test_reward(
    policy: PolicyNet, qnet: PricingNet, mp: ModelParams, tp: TrainParams, seed: int
) -> float:
    tf.random.set_seed(seed)
    np.random.seed(seed)

    N = tp.N_paths_test
    k0 = tf.random.uniform((N,), tp.k0_low, tp.k0_high, dtype=tf.float32)
    b0 = tf.random.uniform((N,), tp.b0_low, tp.b0_high, dtype=tf.float32)
    z0 = tf.random.uniform((N,), tp.z0_low, tp.z0_high, dtype=tf.float32)

    W = rollout_discounted_reward(policy, qnet, mp, tp, k0, b0, z0, tp.T_test)
    return float(tf.reduce_mean(W).numpy())


# ---------------- Euler residual diagnostics (AiO-style) ----------------
@tf.function
def _euler_residuals_autodiff(
    policy: PolicyNet,
    value: Optional[ValueNet],  # <-- allow None (policy-only diagnostic)
    qnet: PricingNet,
    mp: ModelParams,
    tp: TrainParams,
    k: tf.Tensor,
    b: tf.Tensor,
    z: tf.Tensor,
    eps: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Returns (Rk, Rb) per sample.

    If value is provided:
        J = d + beta * V(next)
    If value is None:
        J = d     (policy-only stability diagnostic; engineering metric)

    Rk = s' * dJ/dk'
    Rb = s' * dJ/db'
    """
    # Basic numerical floors; discount factor.
    beta = tf.constant(beta_from_r(mp.r), tf.float32)
    k = tf.maximum(k, mp.k_min)
    z = tf.maximum(z, mp.z_min)

    # Compute policy-implied controls (𝑘′,𝑏′)
    x = tf.stack([k, b, z], axis=1)
    kb_next = policy(x)
    k_next = tf.maximum(kb_next[:, 0], mp.k_min)
    b_next = kb_next[:, 1]

    # Generate 𝑧′using given eps (so we can do AiO with two independent eps later)
    z_next = tf.exp(mp.rho * tf.math.log(z) + eps)

    # Tell TensorFlow: “I want derivatives of the objective 𝐽 w.r.t. the chosen controls 𝑘′,𝑏′”
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(k_next)
        tape.watch(b_next)

        # Compute 𝑞 and then 𝑑=𝑒+𝜂(𝑒)
        q_in = tf.stack([z, k_next, b_next], axis=1)
        q = qnet(q_in)

        d = equity_payout_d(k, k_next, b, b_next, z, q, mp, tp.kappa_issue)

        # The key “value vs policy-only” switch
        # If value exists: use the true Euler objective 𝑑+𝛽𝑉′
        # If no value: use just 𝑑.
        # This is not the model’s Euler equation — it’s a stability proxy for Obj1 where I didn’t train a value network.
        if value is None:
            v_next = tf.zeros_like(d)
            J = d
        else:
            x_next = tf.stack([k_next, b_next, z_next], axis=1)
            v_next = value(x_next)
            J = d + beta * v_next

        J_sum = tf.reduce_sum(J)
    # This corresponds to the Euler/FOC conditions:
    dJ_dk = tape.gradient(J_sum, k_next)
    dJ_db = tape.gradient(J_sum, b_next)
    del tape

    # If likely default, 𝑠′≈0, residual is suppressed
    # If likely solvent, 𝑠′≈1, residual is active.
    s_next = solvency_weight(k_next, b_next, z_next, mp, tp.kappa_solv)
    Rk = s_next * dJ_dk
    Rb = s_next * dJ_db
    return Rk, Rb


# This implements AiO / two-independent-shocks
def eval_test_euler_aio(
    policy: PolicyNet,
    value: ValueNet,
    qnet: PricingNet,
    mp: ModelParams,
    tp: TrainParams,
    states_k: np.ndarray,
    states_b: np.ndarray,
    states_z: np.ndarray,
    seed: int,
) -> float:
    """
    AiO test Euler metric:
      TestEuler = E[ Rk(eps1)Rk(eps2) + Rb(eps1)Rb(eps2) ] on test states.
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)

    k = tf.convert_to_tensor(states_k, tf.float32)
    b = tf.convert_to_tensor(states_b, tf.float32)
    z = tf.convert_to_tensor(states_z, tf.float32)

    eps1 = tf.random.normal(tf.shape(k), 0.0, mp.sigma_eps, dtype=tf.float32)
    eps2 = tf.random.normal(tf.shape(k), 0.0, mp.sigma_eps, dtype=tf.float32)

    Rk1, Rb1 = _euler_residuals_autodiff(policy, value, qnet, mp, tp, k, b, z, eps1)
    Rk2, Rb2 = _euler_residuals_autodiff(policy, value, qnet, mp, tp, k, b, z, eps2)

    metric = tf.reduce_mean(Rk1 * Rk2 + Rb1 * Rb2)
    return float(metric.numpy())


# Same as above, but uses value=None, so 𝐽=𝑑
def eval_test_euler_mse_policy_only(
    policy: PolicyNet,
    qnet: PricingNet,
    mp: ModelParams,
    tp: TrainParams,
    states_k: np.ndarray,
    states_b: np.ndarray,
    states_z: np.ndarray,
    seed: int,
) -> float:
    """
    Policy-only stability diagnostic that DOES NOT require ValueNet:
      same AiO-style cross-product metric but uses J=d (no beta*V term).
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)

    k = tf.convert_to_tensor(states_k, tf.float32)
    b = tf.convert_to_tensor(states_b, tf.float32)
    z = tf.convert_to_tensor(states_z, tf.float32)

    eps1 = tf.random.normal(tf.shape(k), 0.0, mp.sigma_eps, dtype=tf.float32)
    eps2 = tf.random.normal(tf.shape(k), 0.0, mp.sigma_eps, dtype=tf.float32)

    Rk1, Rb1 = _euler_residuals_autodiff(policy, None, qnet, mp, tp, k, b, z, eps1)
    Rk2, Rb2 = _euler_residuals_autodiff(policy, None, qnet, mp, tp, k, b, z, eps2)

    metric = tf.reduce_mean(Rk1 * Rk2 + Rb1 * Rb2)
    return float(metric.numpy())


def eval_test_euler_mse_obj3(
    policy: PolicyNet,
    value: ValueNet,
    qnet: PricingNet,
    mp: ModelParams,
    tp: TrainParams,
    states_k: np.ndarray,
    states_b: np.ndarray,
    states_z: np.ndarray,
    seed: int,
) -> float:
    """
    Obj3 Euler diagnostic (needs ValueNet):
      this is exactly eval_test_euler_aio with proper name used by trainer.py.
    """
    return eval_test_euler_aio(
        policy=policy,
        value=value,
        qnet=qnet,
        mp=mp,
        tp=tp,
        states_k=states_k,
        states_b=states_b,
        states_z=states_z,
        seed=seed,
    )

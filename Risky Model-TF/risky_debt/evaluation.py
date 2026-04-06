"""Evaluation metrics and diagnostics for trained risky-debt networks."""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import tensorflow as tf

from .config import ModelParams, TrainParams
from .networks import PolicyNet, ValueNet, VtildeNet, PricingNet
from .primitives import (
    beta_from_r,
    beta_tensor_from_r,
    equity_payout_d_total,
    solvency_weight,
    continuation_weight_from_value,
)


class EvaluationSuite:
    """Object-oriented access to the package's main diagnostic routines.

    The training code still imports the functional helpers below, but the
    higher-level application layer can now hold one evaluator instance with the
    fixed model objects needed for repeated reward and Euler diagnostics.
    """

    def __init__(
        self,
        *,
        policy: PolicyNet,
        qnet: PricingNet,
        mp: ModelParams,
        tp: TrainParams,
        value: Optional[ValueNet] = None,
        vtilde: Optional[VtildeNet] = None,
    ) -> None:
        """Initialize EvaluationSuite."""
        self.policy = policy
        self.qnet = qnet
        self.mp = mp
        self.tp = tp
        self.value = value
        self.vtilde = vtilde

    def test_reward(self, seed: int) -> float:
        """Estimate the out-of-sample discounted reward on random test states."""
        return eval_test_reward(
            policy=self.policy,
            qnet=self.qnet,
            mp=self.mp,
            tp=self.tp,
            seed=seed,
        )

    def test_euler_obj1(
        self,
        *,
        states_k: np.ndarray,
        states_b: np.ndarray,
        states_z: np.ndarray,
        seed: int,
    ) -> float:
        """Evaluate the Objective 1 Euler diagnostic on a supplied state sample."""
        return eval_test_euler_mse_obj1(
            policy=self.policy,
            qnet=self.qnet,
            mp=self.mp,
            tp=self.tp,
            states_k=states_k,
            states_b=states_b,
            states_z=states_z,
            seed=seed,
        )

    def test_euler_aio(
        self,
        *,
        states_k: np.ndarray,
        states_b: np.ndarray,
        states_z: np.ndarray,
        seed: int,
    ) -> float:
        """Evaluate the AiO Euler diagnostic using the stored value objects."""
        if self.value is None:
            raise ValueError("EvaluationSuite.test_euler_aio requires a value network.")
        return eval_test_euler_aio(
            policy=self.policy,
            value=self.value,
            vtilde=self.vtilde,
            qnet=self.qnet,
            mp=self.mp,
            tp=self.tp,
            states_k=states_k,
            states_b=states_b,
            states_z=states_z,
            seed=seed,
        )

    def test_euler_obj3(
        self,
        *,
        states_k: np.ndarray,
        states_b: np.ndarray,
        states_z: np.ndarray,
        seed: int,
    ) -> float:
        """Evaluate the Objective 3 Euler diagnostic on a supplied state sample."""
        if self.value is None:
            raise ValueError("EvaluationSuite.test_euler_obj3 requires a value network.")
        return eval_test_euler_mse_obj3(
            policy=self.policy,
            value=self.value,
            vtilde=self.vtilde,
            qnet=self.qnet,
            mp=self.mp,
            tp=self.tp,
            states_k=states_k,
            states_b=states_b,
            states_z=states_z,
            seed=seed,
        )


@tf.function
def _obj3_policy_evaluated_vtilde(
    policy: PolicyNet,
    value: ValueNet,
    qnet: PricingNet,
    mp: ModelParams,
    tp: TrainParams,
    k: tf.Tensor,
    b: tf.Tensor,
    z: tf.Tensor,
    eps: tf.Tensor,
) -> tf.Tensor:
    """One-step continuation proxy used for Obj3 value-based gating in diagnostics."""
    beta = beta_tensor_from_r(mp.r)
    z_next = tf.exp(mp.rho * tf.math.log(tf.maximum(z, mp.z_min)) + eps)

    x = tf.stack([k, b, z], axis=1)
    kb_next = policy(x)
    k_next = tf.maximum(kb_next[:, 0], mp.k_min)
    b_next = kb_next[:, 1]

    q_in = tf.stack([z, k_next, b_next], axis=1)
    q = qnet(q_in)

    x_next = tf.stack([k_next, b_next, z_next], axis=1)
    v_next = value(x_next)
    s_next = continuation_weight_from_value(v_next, tp.kappa_solv)
    d = equity_payout_d_total(
        k=k,
        k_next=k_next,
        b=b,
        b_next=b_next,
        z=z,
        q=q,
        continuation_weight=s_next,
        mp=mp,
        kappa_issue=tp.kappa_issue,
    )
    return d + beta * v_next


@tf.function
def _obj3_continuation_gate_proxy(
    policy: PolicyNet,
    value: ValueNet,
    qnet: PricingNet,
    mp: ModelParams,
    tp: TrainParams,
    k: tf.Tensor,
    b: tf.Tensor,
    z: tf.Tensor,
) -> tf.Tensor:
    """Deterministic, detached Obj3 continuation gate used in diagnostics."""
    eps0 = tf.zeros_like(k, dtype=tf.float32)
    vt = _obj3_policy_evaluated_vtilde(
        policy=policy,
        value=value,
        qnet=qnet,
        mp=mp,
        tp=tp,
        k=k,
        b=b,
        z=z,
        eps=eps0,
    )
    gate = continuation_weight_from_value(vt, tp.kappa_solv)
    return tf.stop_gradient(gate)


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
    """Roll out one stochastic test horizon and return discounted rewards."""
    beta = beta_tensor_from_r(mp.r)
    k = tf.maximum(k0, mp.k_min)
    b = b0
    z = tf.maximum(z0, mp.z_min)

    W = tf.zeros_like(k)
    disc = tf.constant(1.0, tf.float32)
    alive = tf.ones_like(k, dtype=tf.float32)

    for _ in tf.range(T + 1):
        x = tf.stack([k, b, z], axis=1)
        kb_next = policy(x)
        k_next = tf.maximum(kb_next[:, 0], mp.k_min)
        b_next = kb_next[:, 1]

        q_in = tf.stack([z, k_next, b_next], axis=1)
        q = qnet(q_in)

        z_next = tf.exp(
            mp.rho * tf.math.log(tf.maximum(z, mp.z_min))
            + tf.random.normal(tf.shape(z), 0.0, mp.sigma_eps, tf.float32)
        )

        s_next = solvency_weight(k_next, b_next, z_next, mp, tp.kappa_solv)
        cont_ind = tf.cast(s_next > 0.5, tf.float32)
        d = equity_payout_d_total(
            k=k,
            k_next=k_next,
            b=b,
            b_next=b_next,
            z=z,
            q=q,
            continuation_weight=cont_ind,
            mp=mp,
            kappa_issue=tp.kappa_issue,
        )

        W = W + disc * (alive * d)
        alive = alive * cont_ind

        k, b, z = k_next, b_next, z_next
        disc = disc * beta

    return W


def eval_test_reward(
    policy: PolicyNet, qnet: PricingNet, mp: ModelParams, tp: TrainParams, seed: int
) -> float:
    """Estimate the average discounted reward on randomly drawn test states."""
    tf.random.set_seed(seed)
    np.random.seed(seed)

    N = tp.N_paths_test
    k0 = tf.random.uniform((N,), tp.k0_low, tp.k0_high, dtype=tf.float32)
    b0 = tf.random.uniform((N,), tp.b0_low, tp.b0_high, dtype=tf.float32)
    z0 = tf.random.uniform((N,), tp.z0_low, tp.z0_high, dtype=tf.float32)

    W = rollout_discounted_reward(policy, qnet, mp, tp, k0, b0, z0, tp.T_test)
    return float(tf.reduce_mean(W).numpy())


@tf.function
def _policy_induced_rollout_value(
    policy: PolicyNet,
    qnet: PricingNet,
    mp: ModelParams,
    tp: TrainParams,
    k0: tf.Tensor,
    b0: tf.Tensor,
    z0: tf.Tensor,
    horizon: int,
) -> tf.Tensor:
    """Finite-horizon policy-induced continuation value for Obj1 Euler diagnostics."""
    beta = beta_tensor_from_r(mp.r)
    k = tf.maximum(k0, mp.k_min)
    b = b0
    z = tf.maximum(z0, mp.z_min)

    W = tf.zeros_like(k)
    disc = tf.constant(1.0, tf.float32)
    alive = tf.ones_like(k, dtype=tf.float32)

    for _ in tf.range(horizon):
        x = tf.stack([k, b, z], axis=1)
        kb_next = policy(x)
        k_next = tf.maximum(kb_next[:, 0], mp.k_min)
        b_next = kb_next[:, 1]

        q_in = tf.stack([z, k_next, b_next], axis=1)
        q = qnet(q_in)

        z_next = tf.exp(
            mp.rho * tf.math.log(tf.maximum(z, mp.z_min))
            + tf.random.normal(tf.shape(z), 0.0, mp.sigma_eps, tf.float32)
        )
        s_next = solvency_weight(k_next, b_next, z_next, mp, tp.kappa_solv)
        cont_ind = tf.cast(s_next > 0.5, tf.float32)
        d = equity_payout_d_total(
            k=k,
            k_next=k_next,
            b=b,
            b_next=b_next,
            z=z,
            q=q,
            continuation_weight=cont_ind,
            mp=mp,
            kappa_issue=tp.kappa_issue,
        )

        W = W + disc * (alive * d)
        alive = alive * cont_ind
        k, b, z = k_next, b_next, z_next
        disc = disc * beta

    return W


@tf.function
def _euler_residuals_autodiff(
    policy: PolicyNet,
    value: Optional[ValueNet],
    vtilde: Optional[VtildeNet],
    qnet: PricingNet,
    mp: ModelParams,
    tp: TrainParams,
    k: tf.Tensor,
    b: tf.Tensor,
    z: tf.Tensor,
    eps: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Compute esiduals autodiff."""
    beta = beta_tensor_from_r(mp.r)
    k = tf.maximum(k, mp.k_min)
    z = tf.maximum(z, mp.z_min)

    x = tf.stack([k, b, z], axis=1)
    kb_next = policy(x)
    k_next = tf.maximum(kb_next[:, 0], mp.k_min)
    b_next = kb_next[:, 1]

    z_next = tf.exp(mp.rho * tf.math.log(z) + eps)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(k_next)
        tape.watch(b_next)

        q_in = tf.stack([z, k_next, b_next], axis=1)
        q = qnet(q_in)

        if value is None:
            cont_ind = tf.cast(solvency_weight(k_next, b_next, z_next, mp, tp.kappa_solv) > 0.5, tf.float32)
            gate = cont_ind
            horizon = tf.constant(min(25, tp.T_test), tf.int32)
            v_next = _policy_induced_rollout_value(policy, qnet, mp, tp, k_next, b_next, z_next, horizon)
        elif vtilde is not None:
            x_next = tf.stack([k_next, b_next, z_next], axis=1)
            vtilde_next = vtilde(x_next)
            gate = continuation_weight_from_value(vtilde_next, tp.kappa_solv)
            v_next = value(x_next)
        else:
            gate = _obj3_continuation_gate_proxy(
                policy=policy,
                value=value,
                qnet=qnet,
                mp=mp,
                tp=tp,
                k=k_next,
                b=b_next,
                z=z_next,
            )
            x_next = tf.stack([k_next, b_next, z_next], axis=1)
            v_next = value(x_next)

        d = equity_payout_d_total(
            k=k,
            k_next=k_next,
            b=b,
            b_next=b_next,
            z=z,
            q=q,
            continuation_weight=gate,
            mp=mp,
            kappa_issue=tp.kappa_issue,
        )
        J = d + beta * v_next
        J_sum = tf.reduce_sum(J)

    dJ_dk = tape.gradient(J_sum, k_next)
    dJ_db = tape.gradient(J_sum, b_next)
    del tape

    Rk = gate * dJ_dk
    Rb = gate * dJ_db
    return Rk, Rb


def eval_test_euler_aio(
    policy: PolicyNet,
    value: ValueNet,
    vtilde: Optional[VtildeNet],
    qnet: PricingNet,
    mp: ModelParams,
    tp: TrainParams,
    states_k: np.ndarray,
    states_b: np.ndarray,
    states_z: np.ndarray,
    seed: int,
) -> float:
    """AiO test Euler metric using the provided value object."""
    tf.random.set_seed(seed)
    np.random.seed(seed)

    k = tf.convert_to_tensor(states_k, tf.float32)
    b = tf.convert_to_tensor(states_b, tf.float32)
    z = tf.convert_to_tensor(states_z, tf.float32)

    eps1 = tf.random.normal(tf.shape(k), 0.0, mp.sigma_eps, dtype=tf.float32)
    eps2 = tf.random.normal(tf.shape(k), 0.0, mp.sigma_eps, dtype=tf.float32)

    Rk1, Rb1 = _euler_residuals_autodiff(policy, value, vtilde, qnet, mp, tp, k, b, z, eps1)
    Rk2, Rb2 = _euler_residuals_autodiff(policy, value, vtilde, qnet, mp, tp, k, b, z, eps2)

    metric = tf.reduce_mean(Rk1 * Rk2 + Rb1 * Rb2)
    return float(metric.numpy())


def eval_test_euler_mse_obj1(
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
    Objective 1 Euler diagnostic using a finite-horizon policy-induced continuation
    value rather than the old policy-only J=d proxy.
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)

    k = tf.convert_to_tensor(states_k, tf.float32)
    b = tf.convert_to_tensor(states_b, tf.float32)
    z = tf.convert_to_tensor(states_z, tf.float32)

    eps1 = tf.random.normal(tf.shape(k), 0.0, mp.sigma_eps, dtype=tf.float32)
    eps2 = tf.random.normal(tf.shape(k), 0.0, mp.sigma_eps, dtype=tf.float32)

    Rk1, Rb1 = _euler_residuals_autodiff(policy, None, None, qnet, mp, tp, k, b, z, eps1)
    Rk2, Rb2 = _euler_residuals_autodiff(policy, None, None, qnet, mp, tp, k, b, z, eps2)

    metric = tf.reduce_mean(Rk1 * Rk2 + Rb1 * Rb2)
    return float(metric.numpy())


def eval_test_euler_mse_obj3(
    policy: PolicyNet,
    value: ValueNet,
    vtilde: Optional[VtildeNet],
    qnet: PricingNet,
    mp: ModelParams,
    tp: TrainParams,
    states_k: np.ndarray,
    states_b: np.ndarray,
    states_z: np.ndarray,
    seed: int,
) -> float:
    """Return the Objective 3 Euler diagnostic using the supplied value objects."""
    return eval_test_euler_aio(
        policy=policy,
        value=value,
        vtilde=vtilde,
        qnet=qnet,
        mp=mp,
        tp=tp,
        states_k=states_k,
        states_b=states_b,
        states_z=states_z,
        seed=seed,
    )

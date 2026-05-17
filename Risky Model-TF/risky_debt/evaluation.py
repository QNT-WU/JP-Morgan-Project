"""Evaluation metrics and diagnostics for trained risky-debt networks.

Evaluation uses constructed zero-profit pricing.  The legacy ``qnet`` argument is
accepted for API compatibility but is not the economic pricing object.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import tensorflow as tf

from .config import ModelParams, TrainParams
from .networks import PolicyNet, ValueNet, VtildeNet, PricingNet
from .primitives import beta_tensor_from_r, equity_payout_d, equity_payout_d_exact, solvency_weight, continuation_weight_from_value
from .pricing import (
    crn_inner_eps,
    smooth_price_from_proxy,
    smooth_price_from_vtilde,
    positive_debt_tax_shield,
    exact_price_from_vtilde,
    exact_price_from_proxy,
    exact_positive_debt_tax_shield,
)


class EvaluationSuite:
    """Object-oriented access to reward and Euler diagnostics."""

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
        self.policy = policy
        self.qnet = qnet
        self.mp = mp
        self.tp = tp
        self.value = value
        self.vtilde = vtilde

    def test_reward(self, seed: int) -> float:
        """Return the average discounted test reward for the stored policy."""
        return eval_test_reward(self.policy, self.qnet, self.mp, self.tp, seed, vtilde=self.vtilde)

    def test_euler_obj1(self, *, states_k, states_b, states_z, seed: int) -> float:
        """Return the Objective 1 Euler/FOC diagnostic on supplied states."""
        return eval_test_euler_mse_obj1(self.policy, self.qnet, self.mp, self.tp, states_k, states_b, states_z, seed)

    def test_euler_aio(self, *, states_k, states_b, states_z, seed: int) -> float:
        """Return the two-shock AiO Euler diagnostic for value-based objectives."""
        if self.value is None:
            raise ValueError("EvaluationSuite.test_euler_aio requires a value network.")
        return eval_test_euler_aio(self.policy, self.value, self.vtilde, self.qnet, self.mp, self.tp, states_k, states_b, states_z, seed)

    def test_euler_obj3(self, *, states_k, states_b, states_z, seed: int) -> float:
        """Return the Objective 3 Euler/FOC diagnostic on supplied states."""
        if self.value is None:
            raise ValueError("EvaluationSuite.test_euler_obj3 requires a value network.")
        return eval_test_euler_mse_obj3(self.policy, self.value, self.vtilde, self.qnet, self.mp, self.tp, states_k, states_b, states_z, seed)


@tf.function
def _qnet_dummy(qnet: PricingNet, z, k_next, b_next):
    """Zero dependency on legacy PricingNet for checkpoint/API compatibility."""
    return 0.0 * tf.reduce_sum(qnet(tf.stack([z, k_next, b_next], axis=1)))


@tf.function
def _policy_step(policy: PolicyNet, mp: ModelParams, k, b, z):
    """Evaluate the policy and enforce the strict positive capital floor."""
    x = tf.stack([k, b, z], axis=1)
    kb_next = policy(x)
    return tf.maximum(kb_next[:, 0], mp.k_min), kb_next[:, 1]


@tf.function
def _z_next(z, mp: ModelParams, eps):
    """Advance the lognormal productivity shock by one period."""
    return tf.exp(mp.rho * tf.math.log(tf.maximum(z, mp.z_min)) + eps)


@tf.function
def _constructed_price(
    qnet: PricingNet,
    z,
    k_next,
    b_next,
    mp: ModelParams,
    tp: TrainParams,
    vtilde: Optional[VtildeNet] = None,
    mode: str = "smooth",
):
    """Construct smooth or exact price using vtilde when available."""
    eps_q = crn_inner_eps(z, tp)
    if mode == "exact":
        if vtilde is None:
            q, qd, rd = exact_price_from_proxy(z, k_next, b_next, eps_q, mp, tp)
        else:
            q, qd, rd = exact_price_from_vtilde(vtilde, z, k_next, b_next, eps_q, mp, tp)
        pen = tf.zeros_like(q)
    else:
        if vtilde is None:
            q, qd, rd, pen = smooth_price_from_proxy(z, k_next, b_next, eps_q, mp, tp)
        else:
            q, qd, rd, pen = smooth_price_from_vtilde(vtilde, z, k_next, b_next, eps_q, mp, tp)
    return q + _qnet_dummy(qnet, z, k_next, b_next), qd, rd, pen


@tf.function
def _constructed_proxy_price(qnet: PricingNet, z, k_next, b_next, mp: ModelParams, tp: TrainParams):
    """Legacy proxy-price helper retained for old callers."""
    return _constructed_price(qnet, z, k_next, b_next, mp, tp, vtilde=None, mode="smooth")


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
    vtilde: Optional[VtildeNet] = None,
    mode: str = "smooth",
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
        k_next, b_next = _policy_step(policy, mp, k, b, z)
        q, _, r_d, _ = _constructed_price(qnet, z, k_next, b_next, mp, tp, vtilde=vtilde, mode=mode)
        z_next = _z_next(z, mp, tf.random.normal(tf.shape(z), 0.0, mp.sigma_eps, tf.float32))
        if vtilde is None:
            s_next = solvency_weight(k_next, b_next, z_next, mp, tp.kappa_solv)
            cont_ind = tf.cast(s_next > 0.5, tf.float32)
        else:
            vt_next = vtilde(tf.stack([k_next, b_next, z_next], axis=1))
            s_next = continuation_weight_from_value(vt_next, tp.kappa_solv)
            cont_ind = tf.cast(vt_next > 0.0, tf.float32)
        if mode == "exact":
            d0 = equity_payout_d_exact(k, k_next, b, b_next, z, q, mp)
            ts = exact_positive_debt_tax_shield(b_next, r_d, cont_ind, mp)
        else:
            d0 = equity_payout_d(k, k_next, b, b_next, z, q, mp, tp.kappa_issue)
            ts = positive_debt_tax_shield(b_next, r_d, cont_ind, mp, tp)
        W = W + disc * alive * (d0 + beta * ts)
        alive = alive * cont_ind
        k, b, z = k_next, b_next, z_next
        disc = disc * beta
    return W


def eval_test_reward(policy: PolicyNet, qnet: PricingNet, mp: ModelParams, tp: TrainParams, seed: int, vtilde: Optional[VtildeNet] = None, mode: str = "smooth") -> float:
    """Estimate the average discounted reward on random test states."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    n = tp.N_paths_test
    k0 = tf.random.uniform((n,), tp.k0_low, tp.k0_high, dtype=tf.float32)
    b0 = tf.random.uniform((n,), tp.b0_low, tp.b0_high, dtype=tf.float32)
    z0 = tf.random.uniform((n,), tp.z0_low, tp.z0_high, dtype=tf.float32)
    return float(tf.reduce_mean(rollout_discounted_reward(policy, qnet, mp, tp, k0, b0, z0, tp.T_test, vtilde=vtilde, mode=mode)).numpy())


@tf.function
def _policy_induced_rollout_value(policy, qnet, mp, tp, k0, b0, z0, horizon: int):
    """Finite-horizon policy-implied value used by Obj1 Euler diagnostics."""
    beta = beta_tensor_from_r(mp.r)
    k = tf.maximum(k0, mp.k_min)
    b = b0
    z = tf.maximum(z0, mp.z_min)
    W = tf.zeros_like(k)
    disc = tf.constant(1.0, tf.float32)
    alive = tf.ones_like(k, dtype=tf.float32)
    for _ in tf.range(horizon):
        k_next, b_next = _policy_step(policy, mp, k, b, z)
        q, _, r_d, _ = _constructed_proxy_price(qnet, z, k_next, b_next, mp, tp)
        z_next = _z_next(z, mp, tf.random.normal(tf.shape(z), 0.0, mp.sigma_eps, tf.float32))
        s_next = solvency_weight(k_next, b_next, z_next, mp, tp.kappa_solv)
        cont_ind = tf.cast(s_next > 0.5, tf.float32)
        d0 = equity_payout_d(k, k_next, b, b_next, z, q, mp, tp.kappa_issue)
        ts = positive_debt_tax_shield(b_next, r_d, cont_ind, mp, tp)
        W = W + disc * alive * (d0 + beta * ts)
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
    """Compute full-gradient residuals using constructed pricing."""
    beta = beta_tensor_from_r(mp.r)
    k = tf.maximum(k, mp.k_min)
    z = tf.maximum(z, mp.z_min)
    k_next, b_next = _policy_step(policy, mp, k, b, z)
    z_next = _z_next(z, mp, eps)
    eps_q = crn_inner_eps(z, tp)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(k_next)
        tape.watch(b_next)
        if vtilde is not None:
            q, _, r_d, _ = smooth_price_from_vtilde(vtilde, z, k_next, b_next, eps_q, mp, tp)
        elif value is not None:
            q, _, r_d, _ = smooth_price_from_vtilde(value, z, k_next, b_next, eps_q, mp, tp)
        else:
            q, _, r_d, _ = smooth_price_from_proxy(z, k_next, b_next, eps_q, mp, tp)
        q = q + _qnet_dummy(qnet, z, k_next, b_next)

        x_next = tf.stack([k_next, b_next, z_next], axis=1)
        if value is None:
            if vtilde is not None:
                vt_next = vtilde(x_next)
                gate = continuation_weight_from_value(vt_next, tp.kappa_solv)
            else:
                cont_ind = tf.cast(solvency_weight(k_next, b_next, z_next, mp, tp.kappa_solv) > 0.5, tf.float32)
                gate = cont_ind
            v_next = _policy_induced_rollout_value(policy, qnet, mp, tp, k_next, b_next, z_next, tf.constant(min(25, tp.T_test), tf.int32))
        else:
            if vtilde is not None:
                vt_next = vtilde(x_next)
                gate = continuation_weight_from_value(vt_next, tp.kappa_solv)
            else:
                vt_next = value(x_next)
                gate = continuation_weight_from_value(vt_next, tp.kappa_solv)
            v_next = value(x_next)
        d0 = equity_payout_d(k, k_next, b, b_next, z, q, mp, tp.kappa_issue)
        ts = positive_debt_tax_shield(b_next, r_d, gate, mp, tp)
        J = d0 + beta * (v_next + ts)
        Jsum = tf.reduce_sum(J)
    dJ_dk = tape.gradient(Jsum, k_next)
    dJ_db = tape.gradient(Jsum, b_next)
    del tape
    return gate * dJ_dk, gate * dJ_db


def eval_test_euler_aio(policy, value, vtilde, qnet, mp, tp, states_k, states_b, states_z, seed: int) -> float:
    """AiO test Euler metric using constructed pricing."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    k = tf.convert_to_tensor(states_k, tf.float32)
    b = tf.convert_to_tensor(states_b, tf.float32)
    z = tf.convert_to_tensor(states_z, tf.float32)
    eps1 = tf.random.normal(tf.shape(k), 0.0, mp.sigma_eps, dtype=tf.float32)
    eps2 = tf.random.normal(tf.shape(k), 0.0, mp.sigma_eps, dtype=tf.float32)
    Rk1, Rb1 = _euler_residuals_autodiff(policy, value, vtilde, qnet, mp, tp, k, b, z, eps1)
    Rk2, Rb2 = _euler_residuals_autodiff(policy, value, vtilde, qnet, mp, tp, k, b, z, eps2)
    return float(tf.reduce_mean(Rk1 * Rk2 + Rb1 * Rb2).numpy())


def eval_test_euler_mse_obj1(policy, qnet, mp, tp, states_k, states_b, states_z, seed: int, vtilde: Optional[VtildeNet] = None) -> float:
    """Objective 1 Euler diagnostic."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    k = tf.convert_to_tensor(states_k, tf.float32)
    b = tf.convert_to_tensor(states_b, tf.float32)
    z = tf.convert_to_tensor(states_z, tf.float32)
    eps1 = tf.random.normal(tf.shape(k), 0.0, mp.sigma_eps, dtype=tf.float32)
    eps2 = tf.random.normal(tf.shape(k), 0.0, mp.sigma_eps, dtype=tf.float32)
    Rk1, Rb1 = _euler_residuals_autodiff(policy, None, vtilde, qnet, mp, tp, k, b, z, eps1)
    Rk2, Rb2 = _euler_residuals_autodiff(policy, None, vtilde, qnet, mp, tp, k, b, z, eps2)
    return float(tf.reduce_mean(Rk1 * Rk2 + Rb1 * Rb2).numpy())


def eval_test_euler_mse_obj3(policy, value, vtilde, qnet, mp, tp, states_k, states_b, states_z, seed: int) -> float:
    """Objective 3 Euler diagnostic."""
    return eval_test_euler_aio(policy, value, vtilde, qnet, mp, tp, states_k, states_b, states_z, seed)

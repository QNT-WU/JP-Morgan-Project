"""Objective and residual functions for the risky-debt learning problems."""

from __future__ import annotations

from typing import Tuple
import tensorflow as tf

from .config import ModelParams, TrainParams, Obj1Params, Obj2Params, Obj3Params
from .networks import PolicyNet, ValueNet, VtildeNet, PricingNet
from .primitives import (
    beta_from_r,
    beta_tensor_from_r,
    equity_payout_d_total,
    solvency_weight,
    continuation_weight_from_value,
    recovery_R,
)


@tf.function
def fischer_burmeister(a: tf.Tensor, c: tf.Tensor) -> tf.Tensor:
    """Return the Fischer--Burmeister complementarity transform."""
    return tf.sqrt(a * a + c * c) - a - c


@tf.function
def _policy_step(policy: PolicyNet, mp: ModelParams, k, b, z):
    """Evaluate the policy network and enforce the capital lower bound."""
    x = tf.stack([k, b, z], axis=1)
    kb_next = policy(x)
    k_next = tf.maximum(kb_next[:, 0], mp.k_min)
    b_next = kb_next[:, 1]
    return k_next, b_next


@tf.function
def _price_q(qnet: PricingNet, z, k_next, b_next):
    """Evaluate the pricing network on ``[z, k', b']`` inputs."""
    q_in = tf.stack([z, k_next, b_next], axis=1)
    return qnet(q_in)


@tf.function
def _one_shock_z_next(z: tf.Tensor, mp: ModelParams, eps: tf.Tensor) -> tf.Tensor:
    """Propagate the productivity state forward for one shock draw."""
    z = tf.maximum(z, mp.z_min)
    return tf.exp(mp.rho * tf.math.log(z) + eps)


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
    """
    One-step policy-evaluated continuation proxy used to build a value-based
    continuation/default gate for Objective 3.
    """
    beta = beta_tensor_from_r(mp.r)

    z_next = _one_shock_z_next(z, mp, eps)
    k_next, b_next = _policy_step(policy, mp, k, b, z)
    q = _price_q(qnet, z, k_next, b_next)

    x_next = tf.stack([k_next, b_next, z_next], axis=1)
    V_next = value(x_next)
    s_next = continuation_weight_from_value(V_next, tp.kappa_solv)
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
    return d + beta * V_next


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
    """
    Low-variance continuation gate for Objective 3.

    Obj3 became much stiffer after moving to the aligned payout object
    d = e_total + eta(e_total). The original nested stochastic gate used an
    extra random shock and allowed gradients to flow through that inner proxy,
    which can create very noisy updates.

    We therefore build the gate from a one-step continuation proxy using a
    deterministic inner shock (eps = 0) and stop its gradient. The gate still
    reflects the current policy/value/pricing triple, but no longer injects an
    additional source of gradient noise into the Obj3 loss.
    """
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
def obj1_loss(
    policy: PolicyNet,
    qnet: PricingNet,
    mp: ModelParams,
    tp: TrainParams,
    op1: Obj1Params,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Objective 1: maximize discounted lifetime reward with a ZP pricing penalty.

    The contingent tax shield is included in the reward only for continuation
    states, matching the model convention that the shield is zero in default.
    """
    beta = beta_tensor_from_r(mp.r)

    N = tp.N_paths_train
    k = tf.random.uniform((N,), tp.k0_low, tp.k0_high, dtype=tf.float32)
    b = tf.random.uniform((N,), tp.b0_low, tp.b0_high, dtype=tf.float32)
    z = tf.random.uniform((N,), tp.z0_low, tp.z0_high, dtype=tf.float32)

    W = tf.zeros_like(k)
    disc = tf.constant(1.0, tf.float32)
    alive = tf.ones_like(k, dtype=tf.float32)

    for _ in tf.range(tp.T_train + 1):
        k_next, b_next = _policy_step(policy, mp, k, b, z)
        q = _price_q(qnet, z, k_next, b_next)

        eps = tf.random.normal(tf.shape(z), 0.0, mp.sigma_eps, tf.float32)
        z_next = _one_shock_z_next(z, mp, eps)

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

    train_reward = tf.reduce_mean(W)

    M = tf.constant(tp.batch_size, tf.int32)
    k0 = tf.random.uniform((M,), tp.k0_low, tp.k0_high, dtype=tf.float32)
    b0 = tf.random.uniform((M,), tp.b0_low, tp.b0_high, dtype=tf.float32)
    z0 = tf.random.uniform((M,), tp.z0_low, tp.z0_high, dtype=tf.float32)

    k1, b1 = _policy_step(policy, mp, k0, b0, z0)
    q0 = _price_q(qnet, z0, k1, b1)

    eps1 = tf.random.normal(tf.shape(k0), 0.0, mp.sigma_eps, tf.float32)
    eps2 = tf.random.normal(tf.shape(k0), 0.0, mp.sigma_eps, tf.float32)
    z1a = _one_shock_z_next(z0, mp, eps1)
    z1b = _one_shock_z_next(z0, mp, eps2)

    s1a = solvency_weight(k1, b1, z1a, mp, tp.kappa_solv)
    s1b = solvency_weight(k1, b1, z1b, mp, tp.kappa_solv)

    R_a = recovery_R(k1, z1a, mp)
    R_b = recovery_R(k1, z1b, mp)

    pay_a = (1.0 - s1a) * R_a + s1a * (b1 / q0)
    pay_b = (1.0 - s1b) * R_b + s1b * (b1 / q0)

    m_a = (1.0 + mp.r) * b1 - pay_a
    m_b = (1.0 + mp.r) * b1 - pay_b

    zp_loss = tf.reduce_mean(m_a * m_b)

    loss = -train_reward + tf.constant(op1.nu_zp, tf.float32) * zp_loss
    return loss, train_reward, zp_loss


@tf.function
def obj2_batch_metrics(
    policy: PolicyNet,
    value: ValueNet,
    vtilde: VtildeNet,
    qnet: PricingNet,
    mp: ModelParams,
    tp: TrainParams,
    op2: Obj2Params,
    k: tf.Tensor,
    b: tf.Tensor,
    z: tf.Tensor,
) -> tf.Tensor:
    """Compute Objective 2 residual diagnostics for one training batch."""
    beta = beta_tensor_from_r(mp.r)
    k = tf.maximum(k, mp.k_min)
    z = tf.maximum(z, mp.z_min)

    x = tf.stack([k, b, z], axis=1)
    V = value(x)
    Vt = vtilde(x)

    R_def = fischer_burmeister(V, V - Vt)

    eps1 = tf.random.normal(tf.shape(k), 0.0, mp.sigma_eps, tf.float32)
    eps2 = tf.random.normal(tf.shape(k), 0.0, mp.sigma_eps, tf.float32)

    def one_shock_residuals(eps):
        """Evaluate the Objective 2 residual blocks for one shock draw."""
        z_next = _one_shock_z_next(z, mp, eps)
        k_next, b_next = _policy_step(policy, mp, k, b, z)
        q = _price_q(qnet, z, k_next, b_next)

        x_next = tf.stack([k_next, b_next, z_next], axis=1)
        V_next = value(x_next)
        Vt_next = vtilde(x_next)

        s_next = continuation_weight_from_value(Vt_next, tp.kappa_solv)
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

        Vt_tar = d + beta * V_next
        R_bell = Vt - Vt_tar

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(k_next)
            tape.watch(b_next)

            q2 = _price_q(qnet, z, k_next, b_next)
            x_next2 = tf.stack([k_next, b_next, z_next], axis=1)
            Vt_next2 = vtilde(x_next2)
            s_next2 = continuation_weight_from_value(Vt_next2, tp.kappa_solv)
            d2 = equity_payout_d_total(
                k=k,
                k_next=k_next,
                b=b,
                b_next=b_next,
                z=z,
                q=q2,
                continuation_weight=s_next2,
                mp=mp,
                kappa_issue=tp.kappa_issue,
            )
            V_next2 = value(x_next2)

            J = d2 + beta * V_next2
            Jsum = tf.reduce_sum(J)

        dJ_dk = tape.gradient(Jsum, k_next)
        dJ_db = tape.gradient(Jsum, b_next)
        del tape

        Rk = s_next * dJ_dk
        Rb = s_next * dJ_db

        Rrec = recovery_R(k_next, z_next, mp)
        pay = (1.0 - s_next) * Rrec + s_next * (b_next / q)
        m_zp = (1.0 + mp.r) * b_next - pay

        return R_bell, Rk, Rb, m_zp

    Rb1, Rk1, Rb_1, mzp1 = one_shock_residuals(eps1)
    Rb2, Rk2, Rb_2, mzp2 = one_shock_residuals(eps2)

    bell_block = tf.reduce_mean(Rb1 * Rb2)
    foc_block = tf.reduce_mean(Rk1 * Rk2 + Rb_1 * Rb_2)
    zp_block = tf.reduce_mean(mzp1 * mzp2)
    def_block = tf.reduce_mean(tf.square(R_def))

    loss = (
        tf.constant(op2.nu_def, tf.float32) * def_block
        + tf.constant(op2.nu_bell, tf.float32) * bell_block
        + tf.constant(op2.nu_foc, tf.float32) * foc_block
        + tf.constant(op2.nu_zp, tf.float32) * zp_block
    )
    return loss, def_block, bell_block, foc_block, zp_block


@tf.function
def obj2_batch_loss(
    policy: PolicyNet,
    value: ValueNet,
    vtilde: VtildeNet,
    qnet: PricingNet,
    mp: ModelParams,
    tp: TrainParams,
    op2: Obj2Params,
    k: tf.Tensor,
    b: tf.Tensor,
    z: tf.Tensor,
) -> tf.Tensor:
    """Return the weighted Objective 2 training loss for one batch."""
    loss, _, _, _, _ = obj2_batch_metrics(
        policy=policy,
        value=value,
        vtilde=vtilde,
        qnet=qnet,
        mp=mp,
        tp=tp,
        op2=op2,
        k=k,
        b=b,
        z=z,
    )
    return loss


@tf.function
def obj3_batch_metrics(
    policy: PolicyNet,
    value: ValueNet,
    qnet: PricingNet,
    mp: ModelParams,
    tp: TrainParams,
    op3: Obj3Params,
    k: tf.Tensor,
    b: tf.Tensor,
    z: tf.Tensor,
) -> tf.Tensor:
    """Compute Objective 3 residual diagnostics for one training batch."""
    beta = beta_tensor_from_r(mp.r)
    k = tf.maximum(k, mp.k_min)
    z = tf.maximum(z, mp.z_min)

    x = tf.stack([k, b, z], axis=1)
    V = value(x)

    eps1 = tf.random.normal(tf.shape(k), 0.0, mp.sigma_eps, tf.float32)
    eps2 = tf.random.normal(tf.shape(k), 0.0, mp.sigma_eps, tf.float32)

    def one_shock(eps):
        """Evaluate the Objective 3 residual blocks for one shock draw."""
        z_next = _one_shock_z_next(z, mp, eps)
        k_next, b_next = _policy_step(policy, mp, k, b, z)
        q = _price_q(qnet, z, k_next, b_next)

        x_next = tf.stack([k_next, b_next, z_next], axis=1)
        V_next = value(x_next)

        s_next = _obj3_continuation_gate_proxy(
            policy=policy,
            value=value,
            qnet=qnet,
            mp=mp,
            tp=tp,
            k=k_next,
            b=b_next,
            z=z_next,
        )
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

        Vtilde_eval = d + beta * V_next
        R_def = fischer_burmeister(V, V - Vtilde_eval)

        Rrec = recovery_R(k_next, z_next, mp)
        pay = (1.0 - s_next) * Rrec + s_next * (b_next / q)
        m_zp = (1.0 + mp.r) * b_next - pay

        return R_def, m_zp

    Rdef1, mzp1 = one_shock(eps1)
    Rdef2, mzp2 = one_shock(eps2)

    def_block = tf.reduce_mean(Rdef1 * Rdef2)
    zp_block = tf.reduce_mean(mzp1 * mzp2)

    loss = (
        tf.constant(op3.nu_def, tf.float32) * def_block
        + tf.constant(op3.nu_zp, tf.float32) * zp_block
    )
    return loss, def_block, zp_block


@tf.function
def obj3_batch_loss(
    policy: PolicyNet,
    value: ValueNet,
    qnet: PricingNet,
    mp: ModelParams,
    tp: TrainParams,
    op3: Obj3Params,
    k: tf.Tensor,
    b: tf.Tensor,
    z: tf.Tensor,
) -> tf.Tensor:
    """Return the weighted Objective 3 training loss for one batch."""
    loss, _, _ = obj3_batch_metrics(
        policy=policy,
        value=value,
        qnet=qnet,
        mp=mp,
        tp=tp,
        op3=op3,
        k=k,
        b=b,
        z=z,
    )
    return loss

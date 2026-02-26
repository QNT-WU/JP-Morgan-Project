from __future__ import annotations

from typing import Tuple
import tensorflow as tf

from .config import ModelParams, TrainParams, Obj1Params, Obj2Params, Obj3Params
from .networks import PolicyNet, ValueNet, VtildeNet, PricingNet
from .primitives import (
    beta_from_r,
    equity_payout_d,
    solvency_weight,
    recovery_R,
    interest_tax_shield_pv,  # <-- ADD THIS
)


# Fischer–Burmeister
# his is the standard smooth “complementarity” function.
@tf.function
def fischer_burmeister(a: tf.Tensor, c: tf.Tensor) -> tf.Tensor:
    return tf.sqrt(a * a + c * c) - a - c


# -------- shared helpers --------


# This is (k′,b′)=φ(k,b,z), with safety k′≥kmin
@tf.function
def _policy_step(policy: PolicyNet, mp: ModelParams, k, b, z):
    x = tf.stack([k, b, z], axis=1)
    kb_next = policy(x)
    k_next = tf.maximum(kb_next[:, 0], mp.k_min)
    b_next = kb_next[:, 1]
    return k_next, b_next


# This is your pricing rule q(z,k′,b′)
# (P1 style: learned pricing net).
@tf.function
def _price_q(qnet: PricingNet, z, k_next, b_next):
    q_in = tf.stack([z, k_next, b_next], axis=1)
    return qnet(q_in)


# AR(1) in logs.
@tf.function
def _one_shock_z_next(z: tf.Tensor, mp: ModelParams, eps: tf.Tensor) -> tf.Tensor:
    z = tf.maximum(z, mp.z_min)
    return tf.exp(mp.rho * tf.math.log(z) + eps)


# -------- Objective 1 (lifetime reward) --------


@tf.function
def obj1_loss(
    policy: PolicyNet,
    qnet: PricingNet,
    mp: ModelParams,
    tp: TrainParams,
    op1: Obj1Params,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Loss = -TrainReward + nu_zp * ZP_loss_batch
    TrainReward from rollouts.
    ZP_loss computed on a fresh minibatch of random states (wide support),
    to keep q disciplined early.
    """
    beta = tf.constant(beta_from_r(mp.r), tf.float32)

    # rollout initial states
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

        # draw next shock first (needed for solvency)
        eps = tf.random.normal(tf.shape(z), 0.0, mp.sigma_eps, tf.float32)
        z_next = _one_shock_z_next(z, mp, eps)

        k_next, b_next = _policy_step(policy, mp, k, b, z)
        q = _price_q(qnet, z, k_next, b_next)

        # base payout today
        d = equity_payout_d(k, k_next, b, b_next, z, q, mp, tp.kappa_issue)

        # solvency weight at t+1 (your convention)
        s_next = solvency_weight(k_next, b_next, z_next, mp, tp.kappa_solv)
        alive_next = alive * tf.cast(s_next > 0.5, tf.float32)

        # NEW: interest-only tax shield paid at t+1 if solvent (PV booked today)
        tax_pv = interest_tax_shield_pv(
            b_next=b_next, q=q, s_next=tf.cast(s_next > 0.5, tf.float32), mp=mp
        )

        # add both to return (disc is beta^t; tax_pv already includes one extra beta)
        W = W + disc * (alive * d) + disc * (alive * tax_pv)

        # update alive and states
        alive = alive_next
        k, b, z = k_next, b_next, z_next
        disc = disc * beta

    train_reward = tf.reduce_mean(W)

    # ---- ZP batch loss (AiO, asset-based solvency) ----
    # sample random states (wide)
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

    # ZP penalty part (important!)
    # lenders payoff: default => R, solvent => b'/q
    pay_a = (1.0 - s1a) * R_a + s1a * (b1 / q0)
    pay_b = (1.0 - s1b) * R_b + s1b * (b1 / q0)

    m_a = (1.0 + mp.r) * b1 - pay_a
    m_b = (1.0 + mp.r) * b1 - pay_b

    zp_loss = tf.reduce_mean(m_a * m_b)

    loss = -train_reward + tf.constant(op1.nu_zp, tf.float32) * zp_loss
    return loss, train_reward, zp_loss


# -------- Objective 2 (Euler residual minimization) --------


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
    beta = tf.constant(beta_from_r(mp.r), tf.float32)
    k = tf.maximum(k, mp.k_min)
    z = tf.maximum(z, mp.z_min)

    # networks at current state
    x = tf.stack([k, b, z], axis=1)
    V = value(x)
    Vt = vtilde(x)

    R_def = fischer_burmeister(V, V - Vt)  # [N]

    # two independent shocks
    eps1 = tf.random.normal(tf.shape(k), 0.0, mp.sigma_eps, tf.float32)
    eps2 = tf.random.normal(tf.shape(k), 0.0, mp.sigma_eps, tf.float32)

    def one_shock_residuals(eps):
        # ----- next shock -----
        z_next = _one_shock_z_next(z, mp, eps)

        # ----- policy step -----
        k_next, b_next = _policy_step(policy, mp, k, b, z)

        # ----- pricing today (q at issuance node (z, k_next, b_next)) -----
        q = _price_q(qnet, z, k_next, b_next)

        # ----- base payout today (your existing d) -----
        d = equity_payout_d(k, k_next, b, b_next, z, q, mp, tp.kappa_issue)

        # ----- continuation value (next-period value net) -----
        x_next = tf.stack([k_next, b_next, z_next], axis=1)
        V_next = value(x_next)

        # ============================================================
        # NEW TAX SHIELD: interest-only, paid at t+1 only if solvent.
        # PV today: beta * tau * r_tilde * b_next * s_next
        # where r_tilde = 1/q - 1
        # ============================================================

        # solvency weight at t+1 (you already use this for gating / ZP split)
        s_next = solvency_weight(k_next, b_next, z_next, mp, tp.kappa_solv)
        s_next = 0.05 + 0.95 * s_next

        # interest rate implied by q
        q_clip = tf.clip_by_value(q, 1e-6, 1.0 - 1e-6)
        r_tilde = (1.0 / q_clip) - 1.0

        # tax shield applies only to "debt" (b_next > 0). If b_next<=0, set shield=0.
        bpos = tf.cast(b_next > 0.0, tf.float32)

        tax_pv = beta * mp.tau * r_tilde * b_next * s_next * bpos

        # ----- Bellman consistency for Vtilde net -----
        Vt_tar = (d + tax_pv) + beta * V_next
        R_bell = Vt - Vt_tar

        # ----- Euler residuals via autodiff of J = (d + tax_pv) + beta V_next -----
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(k_next)
            tape.watch(b_next)

            q2 = _price_q(qnet, z, k_next, b_next)
            q2_clip = tf.clip_by_value(q2, 1e-6, 1.0 - 1e-6)
            r_tilde2 = (1.0 / q2_clip) - 1.0

            d2 = equity_payout_d(k, k_next, b, b_next, z, q2, mp, tp.kappa_issue)

            # IMPORTANT: use the SAME s_next definition for the tax shield in the tape block
            # (s_next depends on (k_next,b_next,z_next) which is already computed)
            bpos2 = tf.cast(b_next > 0.0, tf.float32)
            tax_pv2 = beta * mp.tau * r_tilde2 * b_next * s_next * bpos2

            x_next2 = tf.stack([k_next, b_next, z_next], axis=1)
            V_next2 = value(x_next2)

            J = (d2 + tax_pv2) + beta * V_next2
            Jsum = tf.reduce_sum(J)

        dJ_dk = tape.gradient(Jsum, k_next)
        dJ_db = tape.gradient(Jsum, b_next)
        del tape

        # gate Euler conditions to continuation region (your existing logic)
        Rk = s_next * dJ_dk
        Rb = s_next * dJ_db

        # ----- ZP residual (one-step), unchanged -----
        Rrec = recovery_R(k_next, z_next, mp)
        pay = (1.0 - s_next) * Rrec + s_next * (b_next / q)
        m_zp = (1.0 + mp.r) * b_next - pay

        return R_bell, Rk, Rb, m_zp

    Rb1, Rk1, Rb_1, mzp1 = one_shock_residuals(eps1)
    Rb2, Rk2, Rb_2, mzp2 = one_shock_residuals(eps2)

    # AiO cross-products
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
    return loss


# -------- Objective 3 (Bellman residual with default + ZP), P1 (learn q) --------


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
    beta = tf.constant(beta_from_r(mp.r), tf.float32)
    k = tf.maximum(k, mp.k_min)
    z = tf.maximum(z, mp.z_min)

    x = tf.stack([k, b, z], axis=1)
    V = value(x)

    # Two independent shocks (AiO trick)
    eps1 = tf.random.normal(tf.shape(k), 0.0, mp.sigma_eps, tf.float32)
    eps2 = tf.random.normal(tf.shape(k), 0.0, mp.sigma_eps, tf.float32)

    def one_shock(eps):
        z_next = _one_shock_z_next(z, mp, eps)
        k_next, b_next = _policy_step(policy, mp, k, b, z)
        q = _price_q(qnet, z, k_next, b_next)

        d = equity_payout_d(k, k_next, b, b_next, z, q, mp, tp.kappa_issue)
        x_next = tf.stack([k_next, b_next, z_next], axis=1)
        V_next = value(x_next)

        # ============================================================
        # NEW TAX SHIELD: interest-only, paid at t+1 only if solvent.
        # PV today: beta * tau * r_tilde * b_next * s_next
        # ============================================================
        s_next = solvency_weight(k_next, b_next, z_next, mp, tp.kappa_solv)
        s_next = 0.05 + 0.95 * s_next

        q_clip = tf.clip_by_value(q, 1e-6, 1.0 - 1e-6)
        r_tilde = (1.0 / q_clip) - 1.0
        bpos = tf.cast(b_next > 0.0, tf.float32)

        tax_pv = beta * mp.tau * r_tilde * b_next * s_next * bpos

        # Bellman continuation value evaluated at policy
        Vtilde_eval = (d + tax_pv) + beta * V_next

        # default/complementarity residual for outer max
        R_def = fischer_burmeister(V, V - Vtilde_eval)

        # pricing ZP residual (unchanged)
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
    return loss

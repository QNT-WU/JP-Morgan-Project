"""Objective and residual functions for the risky-debt learning problems.

The risky-debt price is constructed from the lender zero-profit condition.  The
legacy ``qnet`` argument is kept only for backward-compatible public APIs and
checkpoints; it does not determine the economic price.
"""

from __future__ import annotations

from typing import Tuple
import tensorflow as tf

from .config import ModelParams, TrainParams, Obj1Params, Obj2Params, Obj3Params
from .networks import PolicyNet, ValueNet, VtildeNet, PricingNet, MultiplierNet
from .primitives import (
    beta_tensor_from_r,
    equity_payout_d,
    continuation_weight_from_value,
    recovery_R,
)
from .pricing import (
    crn_inner_eps,
    smooth_price_from_proxy,
    smooth_price_from_vtilde,
    softplus_positive,
    positive_debt_tax_shield,
)


@tf.function
def fischer_burmeister(a: tf.Tensor, c: tf.Tensor) -> tf.Tensor:
    """Return the Fischer--Burmeister residual ``a+c-sqrt(a^2+c^2)``."""
    a = tf.convert_to_tensor(a, tf.float32)
    c = tf.convert_to_tensor(c, tf.float32)
    return a + c - tf.sqrt(a * a + c * c + 1e-12)


@tf.function
def _policy_step(policy: PolicyNet, mp: ModelParams, k, b, z):
    """Evaluate the policy network and enforce the capital lower bound."""
    x = tf.stack([k, b, z], axis=1)
    kb_next = policy(x)
    k_next = tf.maximum(kb_next[:, 0], mp.k_min)
    b_next = kb_next[:, 1]
    return k_next, b_next


@tf.function
def _legacy_qnet_dummy(qnet: PricingNet, z, k_next, b_next) -> tf.Tensor:
    """Return a zero-valued dependency on the legacy pricing net.

    This keeps old trainer/test code that expects qnet gradients from failing,
    while the economic price remains fully constructed from zero-profit pricing.
    """
    q_in = tf.stack([z, k_next, b_next], axis=1)
    return 0.0 * tf.reduce_sum(qnet(q_in))


@tf.function
def _price_tuple(qnet: PricingNet, z, k_next, b_next, mp: ModelParams, tp: TrainParams, vtilde: VtildeNet | None = None):
    """Construct smooth zero-profit price tuple.

    If a continuation-value network is available, pricing moments are computed
    from that network's default rule.  The proxy fallback is retained only for
    legacy callers that have no continuation critic.
    """
    eps_q = crn_inner_eps(z, tp)
    if vtilde is None:
        q, qd, rd, penalty = smooth_price_from_proxy(z, k_next, b_next, eps_q, mp, tp)
    else:
        q, qd, rd, penalty = smooth_price_from_vtilde(vtilde, z, k_next, b_next, eps_q, mp, tp)
    return q + _legacy_qnet_dummy(qnet, z, k_next, b_next), qd, rd, penalty


@tf.function
def _price_q(qnet: PricingNet, z, k_next, b_next, mp: ModelParams, tp: TrainParams, vtilde: VtildeNet | None = None):
    """Backward-compatible helper returning only the constructed price q."""
    q, _, _, _ = _price_tuple(qnet, z, k_next, b_next, mp, tp, vtilde=vtilde)
    return q


@tf.function
def _one_shock_z_next(z: tf.Tensor, mp: ModelParams, eps: tf.Tensor) -> tf.Tensor:
    """Propagate the productivity state forward for one shock draw."""
    z = tf.maximum(z, mp.z_min)
    return tf.exp(mp.rho * tf.math.log(z) + eps)


@tf.function
def _continuation_value_from_network(value: ValueNet, vtilde: VtildeNet | None, x: tf.Tensor) -> tf.Tensor:
    """Use vtilde when available, otherwise use the non-negative value proxy."""
    if vtilde is None:
        return value(x)
    return vtilde(x)


@tf.function
def obj1_loss(
    policy: PolicyNet,
    qnet: PricingNet,
    mp: ModelParams,
    tp: TrainParams,
    op1: Obj1Params,
    critic: VtildeNet | None = None,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Objective 1: lifetime reward with constructed zero-profit pricing.

    When ``critic`` is provided, it is the auxiliary policy-implied continuation
    value used for default classification.  If omitted, the legacy asset-value
    proxy is used only for backward compatibility.  The price itself is no
    longer learned by ``qnet``.
    """
    beta = beta_tensor_from_r(mp.r)
    n_paths = tp.N_paths_train
    k = tf.random.uniform((n_paths,), tp.k0_low, tp.k0_high, dtype=tf.float32)
    b = tf.random.uniform((n_paths,), tp.b0_low, tp.b0_high, dtype=tf.float32)
    z = tf.random.uniform((n_paths,), tp.z0_low, tp.z0_high, dtype=tf.float32)

    reward = tf.zeros_like(k)
    disc = tf.constant(1.0, tf.float32)
    alive = tf.ones_like(k, dtype=tf.float32)

    for _ in tf.range(tp.T_train + 1):
        k_next, b_next = _policy_step(policy, mp, k, b, z)
        q, _, r_d, p_pen = _price_tuple(qnet, z, k_next, b_next, mp, tp, vtilde=critic)
        eps = tf.random.normal(tf.shape(z), 0.0, mp.sigma_eps, tf.float32)
        z_next = _one_shock_z_next(z, mp, eps)

        if critic is None:
            from .primitives import solvency_weight
            s_next = solvency_weight(k_next, b_next, z_next, mp, tp.kappa_solv)
        else:
            x_next_c = tf.stack([k_next, b_next, z_next], axis=1)
            s_next = continuation_weight_from_value(critic(x_next_c), tp.kappa_solv)
        cont_ind = tf.cast(s_next > 0.5, tf.float32)

        d0 = equity_payout_d(k, k_next, b, b_next, z, q, mp, tp.kappa_issue)
        ts = positive_debt_tax_shield(b_next, r_d, cont_ind, mp, tp)
        reward = reward + disc * alive * (d0 + beta * ts - tf.stop_gradient(tf.reduce_mean(p_pen)))

        alive = alive * cont_ind
        k, b, z = k_next, b_next, z_next
        disc = disc * beta

    train_reward = tf.reduce_mean(reward)

    # Pricing-admissibility penalty from a fresh batch.  No separate learned ZP
    # residual is needed because zero profit is imposed algebraically by q.
    m = tf.constant(tp.batch_size, tf.int32)
    k0 = tf.random.uniform((m,), tp.k0_low, tp.k0_high, dtype=tf.float32)
    b0 = tf.random.uniform((m,), tp.b0_low, tp.b0_high, dtype=tf.float32)
    z0 = tf.random.uniform((m,), tp.z0_low, tp.z0_high, dtype=tf.float32)
    k1, b1 = _policy_step(policy, mp, k0, b0, z0)
    _, _, _, p_pen = _price_tuple(qnet, z0, k1, b1, mp, tp, vtilde=critic)
    zp_loss = tf.reduce_mean(p_pen)

    critic_loss = tf.constant(0.0, tf.float32)
    if critic is not None:
        eps_c = tf.random.normal(tf.shape(z0), 0.0, mp.sigma_eps, tf.float32)
        z1 = _one_shock_z_next(z0, mp, eps_c)
        x0 = tf.stack([k0, b0, z0], axis=1)
        x1 = tf.stack([k1, b1, z1], axis=1)
        vt0 = critic(x0)
        vt1 = critic(x1)
        s1 = continuation_weight_from_value(vt1, tp.kappa_solv)
        d0_c = equity_payout_d(k0, k1, b0, b1, z0, _price_q(qnet, z0, k1, b1, mp, tp, vtilde=critic), mp, tp.kappa_issue)
        _, _, rd_c, _ = _price_tuple(qnet, z0, k1, b1, mp, tp, vtilde=critic)
        ts_c = positive_debt_tax_shield(b1, rd_c, s1, mp, tp)
        target = d0_c + beta * (tf.nn.relu(vt1) + ts_c)
        critic_loss = tf.reduce_mean(tf.square(vt0 - tf.stop_gradient(target)))

    loss = (
        -train_reward
        + tf.constant(op1.nu_zp, tf.float32) * zp_loss
        + tf.constant(op1.nu_critic, tf.float32) * critic_loss
        + _legacy_qnet_dummy(qnet, z0, k1, b1)
    )
    if critic is None:
        return loss, train_reward, zp_loss
    return loss, train_reward, zp_loss, critic_loss


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
    lambda_net: MultiplierNet | None = None,
) -> tf.Tensor:
    """Objective 2: hybrid Euler/KKT--Bellman--pricing residuals."""
    beta = beta_tensor_from_r(mp.r)
    k = tf.maximum(k, mp.k_min)
    z = tf.maximum(z, mp.z_min)
    x = tf.stack([k, b, z], axis=1)
    V = value(x)
    Vt = vtilde(x)
    R_def = fischer_burmeister(V, V - Vt)

    k_next, b_next = _policy_step(policy, mp, k, b, z)
    eps_q = crn_inner_eps(z, tp)
    q, qd, r_d, p_pen = smooth_price_from_vtilde(vtilde, z, k_next, b_next, eps_q, mp, tp)
    q = q + _legacy_qnet_dummy(qnet, z, k_next, b_next)

    eps1 = tf.random.normal(tf.shape(k), 0.0, mp.sigma_eps, tf.float32)
    eps2 = tf.random.normal(tf.shape(k), 0.0, mp.sigma_eps, tf.float32)

    def one_shock_residuals(eps):
        z_next = _one_shock_z_next(z, mp, eps)
        x_next = tf.stack([k_next, b_next, z_next], axis=1)
        V_next = value(x_next)
        Vt_next = vtilde(x_next)
        s_next = continuation_weight_from_value(Vt_next, tp.kappa_solv)
        d0 = equity_payout_d(k, k_next, b, b_next, z, q, mp, tp.kappa_issue)
        ts = positive_debt_tax_shield(b_next, r_d, s_next, mp, tp)
        R_bell = Vt - (d0 + beta * (V_next + ts))

        with tf.GradientTape() as tape:
            tape.watch([k_next, b_next])
            q2, _, r_d2, _ = smooth_price_from_vtilde(vtilde, z, k_next, b_next, eps_q, mp, tp)
            q2 = q2 + _legacy_qnet_dummy(qnet, z, k_next, b_next)
            x_next2 = tf.stack([k_next, b_next, z_next], axis=1)
            V_next2 = value(x_next2)
            Vt_next2 = vtilde(x_next2)
            s_next2 = continuation_weight_from_value(Vt_next2, tp.kappa_solv)
            d02 = equity_payout_d(k, k_next, b, b_next, z, q2, mp, tp.kappa_issue)
            ts2 = positive_debt_tax_shield(b_next, r_d2, s_next2, mp, tp)
            J = d02 + beta * (V_next2 + ts2)
            Jsum = tf.reduce_sum(J)
        Gk, Gb = tape.gradient(Jsum, [k_next, b_next])

        # Colab-friendly residual training mode:
        # computing gradients of the loss through Gk/Gb would require
        # second-order autodiff through the constructed pricing operator. That
        # graph is too heavy for typical T4/Colab runs. We still compute the
        # FOC/KKT residuals as diagnostics and train the multiplier against the
        # detached stationarity target, while Bellman/default/pricing blocks
        # continue to train policy and value networks with ordinary first-order
        # gradients.
        Gk_target = tf.stop_gradient(Gk)
        Gb_target = tf.stop_gradient(Gb)

        # Max problem with lower bound k' >= k_min: G_k + lambda = 0.
        if lambda_net is None:
            lam_k = tf.nn.softplus(-Gk_target)
        else:
            lam_k = lambda_net(x)
        R_stat_k = Gk_target + lam_k
        R_comp_k = fischer_burmeister(lam_k, k_next - mp.k_min)
        R_bprime = Gb_target

        return R_bell, R_stat_k, R_comp_k, R_bprime

    Rbell1, Rstat1, Rcomp1, Rb1 = one_shock_residuals(eps1)
    Rbell2, Rstat2, Rcomp2, Rb2 = one_shock_residuals(eps2)

    bell_block = tf.reduce_mean(Rbell1 * Rbell2)
    kkt_block = tf.reduce_mean(Rstat1 * Rstat2 + tf.square(Rcomp1))
    foc_block = tf.reduce_mean(Rb1 * Rb2)
    zp_block = tf.reduce_mean(p_pen)
    def_block = tf.reduce_mean(tf.square(R_def))

    loss = (
        tf.constant(op2.nu_def, tf.float32) * def_block
        + tf.constant(op2.nu_bell, tf.float32) * bell_block
        + tf.constant(op2.nu_foc, tf.float32) * (kkt_block + foc_block)
        + tf.constant(op2.nu_zp, tf.float32) * zp_block
        + _legacy_qnet_dummy(qnet, z, k_next, b_next)
    )
    return loss, def_block, bell_block, kkt_block + foc_block, zp_block


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
    lambda_net: MultiplierNet | None = None,
) -> tf.Tensor:
    """Return the weighted Objective 2 training loss for one batch."""
    loss, _, _, _, _ = obj2_batch_metrics(policy, value, vtilde, qnet, mp, tp, op2, k, b, z, lambda_net=lambda_net)
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
    vtilde: VtildeNet | None = None,
    lambda_net: MultiplierNet | None = None,
) -> tf.Tensor:
    """Objective 3: Bellman-centered residual with constructed pricing.

    Uses a separate continuation-value network and a multiplier network when
    provided.  These are required for the fully aligned Objective 3 trainer; the
    value-as-continuation fallback is retained only for old callers.
    """
    beta = beta_tensor_from_r(mp.r)
    k = tf.maximum(k, mp.k_min)
    z = tf.maximum(z, mp.z_min)
    x = tf.stack([k, b, z], axis=1)
    V = value(x)
    Vt = value(x) if vtilde is None else vtilde(x)

    k_next, b_next = _policy_step(policy, mp, k, b, z)
    eps_q = crn_inner_eps(z, tp)

    # Backward-compatible callers of obj3_batch_loss do not pass a separate
    # continuation-value network.  In that legacy path, use the lightweight
    # asset-value proxy for pricing moments; the fully aligned trainer passes
    # vtilde and therefore uses continuation-value pricing.
    if vtilde is None:
        q, _, r_d, p_pen = smooth_price_from_proxy(z, k_next, b_next, eps_q, mp, tp)
    else:
        q, _, r_d, p_pen = smooth_price_from_vtilde(vtilde, z, k_next, b_next, eps_q, mp, tp)
    q = q + _legacy_qnet_dummy(qnet, z, k_next, b_next)

    eps1 = tf.random.normal(tf.shape(k), 0.0, mp.sigma_eps, tf.float32)
    eps2 = tf.random.normal(tf.shape(k), 0.0, mp.sigma_eps, tf.float32)

    def one_shock(eps):
        z_next = _one_shock_z_next(z, mp, eps)
        x_next = tf.stack([k_next, b_next, z_next], axis=1)
        V_next = value(x_next)
        Vt_next = V_next if vtilde is None else vtilde(x_next)
        s_next = continuation_weight_from_value(Vt_next, tp.kappa_solv)
        d0 = equity_payout_d(k, k_next, b, b_next, z, q, mp, tp.kappa_issue)
        ts = positive_debt_tax_shield(b_next, r_d, s_next, mp, tp)
        H = d0 + beta * (V_next + ts)
        R_bell = Vt - H
        R_def = fischer_burmeister(V, V - Vt)

        with tf.GradientTape() as tape:
            tape.watch([k_next, b_next])
            if vtilde is None:
                q2, _, r_d2, _ = smooth_price_from_proxy(z, k_next, b_next, eps_q, mp, tp)
            else:
                q2, _, r_d2, _ = smooth_price_from_vtilde(vtilde, z, k_next, b_next, eps_q, mp, tp)
            q2 = q2 + _legacy_qnet_dummy(qnet, z, k_next, b_next)
            x_next2 = tf.stack([k_next, b_next, z_next], axis=1)
            V_next2 = value(x_next2)
            Vt_next2 = V_next2 if vtilde is None else vtilde(x_next2)
            s_next2 = continuation_weight_from_value(Vt_next2, tp.kappa_solv)
            d02 = equity_payout_d(k, k_next, b, b_next, z, q2, mp, tp.kappa_issue)
            ts2 = positive_debt_tax_shield(b_next, r_d2, s_next2, mp, tp)
            J = d02 + beta * (V_next2 + ts2)
            Jsum = tf.reduce_sum(J)
        Gk, Gb = tape.gradient(Jsum, [k_next, b_next])

        # As in Objective 2, detach the FOC gradients before they enter the
        # outer training loss. This avoids second-order autodiff through the
        # zero-profit pricing block while preserving the Bellman/default
        # training signal and reporting the same FOC/KKT residual diagnostics.
        Gk_target = tf.stop_gradient(Gk)
        Gb_target = tf.stop_gradient(Gb)
        if lambda_net is None:
            lam_k = tf.nn.softplus(-Gk_target)
        else:
            lam_k = lambda_net(x)
        R_stat_k = Gk_target + lam_k
        R_comp_k = fischer_burmeister(lam_k, k_next - mp.k_min)
        return R_def, R_bell, R_stat_k, R_comp_k, Gb_target

    Rdef1, Rbell1, Rstat1, Rcomp1, Rb1 = one_shock(eps1)
    Rdef2, Rbell2, Rstat2, Rcomp2, Rb2 = one_shock(eps2)

    def_block = tf.reduce_mean(tf.square(Rdef1))
    bell_block = tf.reduce_mean(Rbell1 * Rbell2)
    stat_block = tf.reduce_mean(Rstat1 * Rstat2)
    kkt_block = tf.reduce_mean(tf.square(Rcomp1))
    bprime_block = tf.reduce_mean(Rb1 * Rb2)
    opt_block = stat_block + kkt_block + bprime_block
    zp_block = tf.reduce_mean(p_pen)
    loss = (
        tf.constant(op3.nu_def, tf.float32) * def_block
        + tf.constant(op3.nu_bell, tf.float32) * bell_block
        + tf.constant(op3.nu_foc, tf.float32) * opt_block
        + tf.constant(op3.nu_zp, tf.float32) * zp_block
        + _legacy_qnet_dummy(qnet, z, k_next, b_next)
    )
    return loss, def_block, bell_block, stat_block, kkt_block, bprime_block, zp_block


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
    vtilde: VtildeNet | None = None,
    lambda_net: MultiplierNet | None = None,
) -> tf.Tensor:
    """Return the weighted Objective 3 training loss for one batch."""
    loss, *_ = obj3_batch_metrics(policy, value, qnet, mp, tp, op3, k, b, z, vtilde=vtilde, lambda_net=lambda_net)
    return loss

"""Reusable inner Objective 1 solver for estimation.

This module isolates the expensive estimation-time inner solve used by both
SMM and GMM. The implementation is designed to keep the code aligned with the
user's structural model while improving runtime in three ways:

1. Reuse one ``PolicyNet`` and one ``PricingNet`` instead of rebuilding them
   for every candidate parameter evaluation.
2. Precompile the inner training step with ``tf.function``.
3. Use fixed inner-training random draws so the estimation objective is more
   deterministic and does not pay repeated random-draw overhead.

The solver remains faithful to the user's model plan:
- Objective 1 still maximizes discounted lifetime reward plus the ZP penalty.
- The contingent tax shield remains active only in continuation states.
- Candidate parameters still change only the baseline estimation vector
  ``(theta, psi0, alpha)`` while the calibrated primitives stay fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from risky_debt.config import ModelParams, NetParams, Obj1Params, TrainParams
from risky_debt.networks import PolicyNet, PricingNet
from risky_debt.objectives import _one_shock_z_next, _policy_step, _price_q
from risky_debt.primitives import beta_from_r, beta_tensor_from_r, equity_payout_d_total
from risky_debt.simulation import set_global_seed


def _optimizer_variable_list(optimizer: tf.keras.optimizers.Optimizer) -> List[tf.Variable]:
    """Return optimizer slot variables across TensorFlow / Keras versions."""
    vars_attr = optimizer.variables
    return list(vars_attr() if callable(vars_attr) else vars_attr)


class _TensorModelParams:
    """Tensor-backed model-parameter container for compiled estimation steps.

    The object keeps a stable Python identity while the candidate values of
    ``theta``, ``psi0``, and ``alpha`` are updated through ``tf.Variable``.
    This lets compiled TensorFlow functions reuse the same traced graph across
    candidate evaluations.
    """

    def __init__(self, mp: ModelParams) -> None:
        """Initialize _TensorModelParams."""
        self.rho = tf.constant(mp.rho, dtype=tf.float32)
        self.sigma_eps = tf.constant(mp.sigma_eps, dtype=tf.float32)
        self.theta = tf.Variable(mp.theta, trainable=False, dtype=tf.float32, name="theta")
        self.tau = tf.constant(mp.tau, dtype=tf.float32)
        self.psi0 = tf.Variable(mp.psi0, trainable=False, dtype=tf.float32, name="psi0")
        self.delta = tf.constant(mp.delta, dtype=tf.float32)
        self.r = tf.constant(mp.r, dtype=tf.float32)
        self.phi_borrow = tf.constant(mp.phi_borrow, dtype=tf.float32)
        self.alpha = tf.Variable(mp.alpha, trainable=False, dtype=tf.float32, name="alpha")
        self.eta0 = tf.constant(mp.eta0, dtype=tf.float32)
        self.eta1 = tf.constant(mp.eta1, dtype=tf.float32)
        self.k_min = tf.constant(mp.k_min, dtype=tf.float32)
        self.z_min = tf.constant(mp.z_min, dtype=tf.float32)
        self.b_min = tf.constant(mp.b_min, dtype=tf.float32)
        self.b_max = tf.constant(mp.b_max, dtype=tf.float32)
        self.q_min = tf.constant(mp.q_min, dtype=tf.float32)
        self.q_max = tf.constant(mp.q_max, dtype=tf.float32)

    def assign_from(self, mp: ModelParams) -> None:
        """Update the candidate structural parameters in place."""
        self.theta.assign(float(mp.theta))
        self.psi0.assign(float(mp.psi0))
        self.alpha.assign(float(mp.alpha))


@dataclass(frozen=True)
class FixedObjective1Draws:
    """Fixed Monte Carlo draws used by the inner Objective 1 estimation solve."""

    train_k0: tf.Tensor
    train_b0: tf.Tensor
    train_z0: tf.Tensor
    train_eps: tf.Tensor
    zp_k0: tf.Tensor
    zp_b0: tf.Tensor
    zp_z0: tf.Tensor
    zp_eps1: tf.Tensor
    zp_eps2: tf.Tensor

    @classmethod
    def create(cls, mp: ModelParams, tp: TrainParams, seed: int) -> "FixedObjective1Draws":
        """Create one deterministic bundle of draws for all candidate solves."""
        rng = np.random.default_rng(seed)

        n_train = int(tp.N_paths_train)
        train_horizon = int(tp.T_train) + 1
        n_zp = int(tp.batch_size)

        def uniform(low: float, high: float, size: int) -> tf.Tensor:
            """Draw deterministic uniform samples stored as TensorFlow constants."""
            values = rng.uniform(low, high, size=size).astype(np.float32)
            return tf.constant(values, dtype=tf.float32)

        def normal(std: float, shape: Tuple[int, ...]) -> tf.Tensor:
            """Draw deterministic Gaussian samples stored as TensorFlow constants."""
            values = rng.normal(0.0, std, size=shape).astype(np.float32)
            return tf.constant(values, dtype=tf.float32)

        return cls(
            train_k0=uniform(tp.k0_low, tp.k0_high, n_train),
            train_b0=uniform(tp.b0_low, tp.b0_high, n_train),
            train_z0=uniform(tp.z0_low, tp.z0_high, n_train),
            train_eps=normal(mp.sigma_eps, (train_horizon, n_train)),
            zp_k0=uniform(tp.k0_low, tp.k0_high, n_zp),
            zp_b0=uniform(tp.b0_low, tp.b0_high, n_zp),
            zp_z0=uniform(tp.z0_low, tp.z0_high, n_zp),
            zp_eps1=normal(mp.sigma_eps, (n_zp,)),
            zp_eps2=normal(mp.sigma_eps, (n_zp,)),
        )


class ReusableInnerObjective1Solver:
    """Reusable, compiled Objective 1 solver for estimation.

    The solver owns one policy network, one pricing network, and one pair of
    optimizers. For each candidate parameter vector, it restores the initial
    network and optimizer states, updates the tensor-backed model parameters,
    and runs the same compiled training step over a fixed bundle of draws.
    """

    def __init__(
        self,
        mp: ModelParams,
        npol: NetParams,
        nq: NetParams,
        tp_inner: TrainParams,
        seed: int,
    ) -> None:
        """Initialize ReusableInnerObjective1Solver."""
        self.template_mp = mp
        self.npol = npol
        self.nq = nq
        self.tp_inner = tp_inner
        self.seed = int(seed)
        self.model_state = _TensorModelParams(mp)
        self.obj1_nu_zp = tf.Variable(1.0, trainable=False, dtype=tf.float32, name="obj1_nu_zp")
        self.draws = FixedObjective1Draws.create(mp=mp, tp=tp_inner, seed=self.seed + 1)

        set_global_seed(self.seed)
        self.policy = PolicyNet(npol, mp.k_min, mp.b_min, mp.b_max)
        self.qnet = PricingNet(nq, mp.q_min, mp.q_max)
        _ = self.policy(tf.zeros((1, 3), tf.float32))
        _ = self.qnet(tf.zeros((1, 3), tf.float32))

        self.opt_policy = tf.keras.optimizers.Adam(self.tp_inner.lr_policy)
        self.opt_q = tf.keras.optimizers.Adam(self.tp_inner.lr_q)
        self._initialize_optimizer_slots()

        self._init_policy_weights = [var.numpy().copy() for var in self.policy.weights]
        self._init_q_weights = [var.numpy().copy() for var in self.qnet.weights]
        self._init_policy_opt_state = [var.numpy().copy() for var in _optimizer_variable_list(self.opt_policy)]
        self._init_q_opt_state = [var.numpy().copy() for var in _optimizer_variable_list(self.opt_q)]

    def _initialize_optimizer_slots(self) -> None:
        """Create optimizer slot variables once so they can be restored later."""
        zero_policy = [tf.zeros_like(var) for var in self.policy.trainable_variables]
        zero_q = [tf.zeros_like(var) for var in self.qnet.trainable_variables]
        self.opt_policy.apply_gradients(zip(zero_policy, self.policy.trainable_variables))
        self.opt_q.apply_gradients(zip(zero_q, self.qnet.trainable_variables))
        if hasattr(self.opt_policy, "iterations"):
            self.opt_policy.iterations.assign(0)
        if hasattr(self.opt_q, "iterations"):
            self.opt_q.iterations.assign(0)

    def _restore_network_state(self) -> None:
        """Restore the initial policy and pricing-network weights."""
        for var, value in zip(self.policy.weights, self._init_policy_weights):
            var.assign(value)
        for var, value in zip(self.qnet.weights, self._init_q_weights):
            var.assign(value)

    def _restore_optimizer_state(self) -> None:
        """Restore the initial Adam optimizer state."""
        for var, value in zip(_optimizer_variable_list(self.opt_policy), self._init_policy_opt_state):
            var.assign(value)
        for var, value in zip(_optimizer_variable_list(self.opt_q), self._init_q_opt_state):
            var.assign(value)

    def snapshot_weights(self) -> Dict[str, List[np.ndarray]]:
        """Return detached copies of the current network weights."""
        return {
            "policy_weights": [var.numpy().copy() for var in self.policy.weights],
            "qnet_weights": [var.numpy().copy() for var in self.qnet.weights],
        }

    def materialize_networks(
        self,
        mp: ModelParams,
        snapshot: Dict[str, Sequence[np.ndarray]],
    ) -> Tuple[PolicyNet, PricingNet]:
        """Build standalone networks from a stored weight snapshot.

        Materialization is used only when a cached candidate result needs a
        stable network object later on, for example in final GMM reporting.
        The expensive inner training still happens on the reusable networks.
        """
        policy = PolicyNet(self.npol, mp.k_min, mp.b_min, mp.b_max)
        qnet = PricingNet(self.nq, mp.q_min, mp.q_max)
        _ = policy(tf.zeros((1, 3), tf.float32))
        _ = qnet(tf.zeros((1, 3), tf.float32))
        policy.set_weights([np.asarray(w).copy() for w in snapshot["policy_weights"]])
        qnet.set_weights([np.asarray(w).copy() for w in snapshot["qnet_weights"]])
        return policy, qnet

    @tf.function(reduce_retracing=True)
    def _fixed_draw_obj1_loss(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Compute the Objective 1 loss on a fixed bundle of training draws."""
        beta = tf.constant(beta_from_r(float(self.template_mp.r)), dtype=tf.float32)

        k = self.draws.train_k0
        b = self.draws.train_b0
        z = self.draws.train_z0
        alive = tf.ones_like(k, dtype=tf.float32)
        reward_ta = tf.TensorArray(dtype=tf.float32, size=int(self.tp_inner.T_train) + 1)

        def body(
            t: tf.Tensor,
            k_cur: tf.Tensor,
            b_cur: tf.Tensor,
            z_cur: tf.Tensor,
            alive_cur: tf.Tensor,
            reward_acc: tf.TensorArray,
        ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.TensorArray]:
            """Advance one rollout step inside the compiled Objective 1 loop."""
            k_next, b_next = _policy_step(self.policy, self.model_state, k_cur, b_cur, z_cur)
            q = _price_q(self.qnet, z_cur, k_next, b_next)
            eps = self.draws.train_eps[t]
            z_next = _one_shock_z_next(z_cur, self.model_state, eps)
            s_next = tf.clip_by_value(
                tf.sigmoid(
                    (
                        (1.0 - self.model_state.tau) * (z_next * tf.pow(tf.maximum(k_next, self.model_state.k_min), self.model_state.theta))
                        + (1.0 - self.model_state.delta) * k_next
                        - (1.0 + self.model_state.r) * b_next
                    )
                    / tf.cast(self.tp_inner.kappa_solv, tf.float32)
                ),
                1e-6,
                1.0 - 1e-6,
            )
            cont_ind = tf.cast(s_next > 0.5, tf.float32)
            d = equity_payout_d_total(
                k=k_cur,
                k_next=k_next,
                b=b_cur,
                b_next=b_next,
                z=z_cur,
                q=q,
                continuation_weight=cont_ind,
                mp=self.model_state,
                kappa_issue=self.tp_inner.kappa_issue,
            )
            reward_acc = reward_acc.write(t, alive_cur * d)
            return t + 1, k_next, b_next, z_next, alive_cur * cont_ind, reward_acc

        t0 = tf.constant(0, dtype=tf.int32)
        _, _, _, _, _, reward_ta = tf.while_loop(
            cond=lambda t, *_: t < tf.constant(int(self.tp_inner.T_train) + 1, dtype=tf.int32),
            body=body,
            loop_vars=(t0, k, b, z, alive, reward_ta),
        )
        reward_stack = reward_ta.stack()
        discount_factors = tf.pow(beta, tf.cast(tf.range(int(self.tp_inner.T_train) + 1), tf.float32))[:, None]
        train_reward = tf.reduce_mean(tf.reduce_sum(discount_factors * reward_stack, axis=0))

        k0 = self.draws.zp_k0 * 0.0 + self.draws.zp_k0
        b0 = self.draws.zp_b0 * 0.0 + self.draws.zp_b0
        z0 = self.draws.zp_z0 * 0.0 + self.draws.zp_z0
        k1, b1 = _policy_step(self.policy, self.model_state, k0, b0, z0)
        q0 = _price_q(self.qnet, z0, k1, b1)
        z1a = _one_shock_z_next(z0, self.model_state, self.draws.zp_eps1)
        z1b = _one_shock_z_next(z0, self.model_state, self.draws.zp_eps2)

        proxy_a = (
            (1.0 - self.model_state.tau) * (z1a * tf.pow(tf.maximum(k1, self.model_state.k_min), self.model_state.theta))
            + (1.0 - self.model_state.delta) * k1
            - (1.0 + self.model_state.r) * b1
        )
        proxy_b = (
            (1.0 - self.model_state.tau) * (z1b * tf.pow(tf.maximum(k1, self.model_state.k_min), self.model_state.theta))
            + (1.0 - self.model_state.delta) * k1
            - (1.0 + self.model_state.r) * b1
        )
        s1a = tf.clip_by_value(tf.sigmoid(proxy_a / tf.cast(self.tp_inner.kappa_solv, tf.float32)), 1e-6, 1.0 - 1e-6)
        s1b = tf.clip_by_value(tf.sigmoid(proxy_b / tf.cast(self.tp_inner.kappa_solv, tf.float32)), 1e-6, 1.0 - 1e-6)

        pay_a = (1.0 - s1a) * (1.0 - self.model_state.alpha) * (
            (1.0 - self.model_state.tau) * (z1a * tf.pow(tf.maximum(k1, self.model_state.k_min), self.model_state.theta))
            + (1.0 - self.model_state.delta) * k1
        ) + s1a * (b1 / q0)
        pay_b = (1.0 - s1b) * (1.0 - self.model_state.alpha) * (
            (1.0 - self.model_state.tau) * (z1b * tf.pow(tf.maximum(k1, self.model_state.k_min), self.model_state.theta))
            + (1.0 - self.model_state.delta) * k1
        ) + s1b * (b1 / q0)

        m_a = (1.0 + self.model_state.r) * b1 - pay_a
        m_b = (1.0 + self.model_state.r) * b1 - pay_b
        zp_loss = tf.reduce_mean(m_a * m_b)
        loss = -train_reward + self.obj1_nu_zp * zp_loss
        return loss, train_reward, zp_loss

    @tf.function(reduce_retracing=True)
    def _train_step(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Run one compiled inner Objective 1 optimization step."""
        with tf.GradientTape(persistent=True) as tape:
            loss, train_reward, zp_loss = self._fixed_draw_obj1_loss()
        g_pol = tape.gradient(loss, self.policy.trainable_variables)
        g_q = tape.gradient(loss, self.qnet.trainable_variables)
        del tape
        self.opt_policy.apply_gradients(zip(g_pol, self.policy.trainable_variables))
        self.opt_q.apply_gradients(zip(g_q, self.qnet.trainable_variables))
        return loss, train_reward, zp_loss

    def solve(self, mp: ModelParams, op1: Obj1Params) -> Tuple[PolicyNet, PricingNet]:
        """Solve the inner Objective 1 problem for one candidate parameter vector."""
        self.model_state.assign_from(mp)
        self.obj1_nu_zp.assign(float(op1.nu_zp))
        self._restore_network_state()
        self._restore_optimizer_state()

        for _ in range(int(self.tp_inner.epochs)):
            for _ in range(int(self.tp_inner.steps_per_epoch)):
                self._train_step()
        return self.policy, self.qnet


@tf.function(reduce_retracing=True)
def finite_horizon_predefault_continuation(
    policy: PolicyNet,
    qnet: PricingNet,
    mp: ModelParams,
    tp: TrainParams,
    k0: tf.Tensor,
    b0: tf.Tensor,
    z0: tf.Tensor,
    eps_future: tf.Tensor,
    valid_mask: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """Policy-evaluated hard-continuation recursion on a realized shock path."""
    k0 = tf.maximum(tf.convert_to_tensor(k0, tf.float32), mp.k_min)
    b0 = tf.convert_to_tensor(b0, tf.float32)
    z0 = tf.maximum(tf.convert_to_tensor(z0, tf.float32), mp.z_min)
    eps_future = tf.convert_to_tensor(eps_future, tf.float32)
    if eps_future.shape.rank == 1:
        eps_future = eps_future[None, :]
    horizon = int(eps_future.shape[1] or 0)
    if horizon == 0:
        return tf.zeros_like(k0)

    if valid_mask is None:
        valid_mask = tf.ones_like(eps_future, dtype=tf.float32)
    else:
        valid_mask = tf.convert_to_tensor(valid_mask, tf.float32)
        if valid_mask.shape.rank == 1:
            valid_mask = valid_mask[None, :]

    beta = beta_tensor_from_r(mp.r)

    k_cur = k0
    b_cur = b0
    z_cur = z0
    steps: List[Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]] = []

    for h in range(horizon):
        x = tf.stack([k_cur, b_cur, z_cur], axis=1)
        kb_next = policy(x)
        k_next_raw = tf.maximum(kb_next[:, 0], mp.k_min)
        b_next_raw = kb_next[:, 1]
        q_raw = qnet(tf.stack([z_cur, k_next_raw, b_next_raw], axis=1))
        eps_h = eps_future[:, h]
        valid_h = tf.cast(valid_mask[:, h], tf.float32)
        z_next_raw = tf.exp(tf.cast(mp.rho, tf.float32) * tf.math.log(tf.maximum(z_cur, mp.z_min)) + eps_h)

        v = valid_h
        k_next = v * k_next_raw + (1.0 - v) * k_cur
        b_next = v * b_next_raw + (1.0 - v) * b_cur
        z_next = v * z_next_raw + (1.0 - v) * z_cur
        q = v * q_raw + (1.0 - v) * tf.ones_like(q_raw) * (1.0 / (1.0 + mp.r))

        steps.append((k_cur, b_cur, z_cur, k_next, b_next, q, valid_h))
        k_cur, b_cur, z_cur = k_next, b_next, z_next

    cont_next = tf.zeros_like(k0)
    for h in range(horizon - 1, -1, -1):
        k_t, b_t, z_t, k_next, b_next, q_t, valid_h = steps[h]
        cont_ind = tf.stop_gradient(tf.cast(cont_next > 0.0, tf.float32))
        d_t = equity_payout_d_total(
            k=k_t,
            k_next=k_next,
            b=b_t,
            b_next=b_next,
            z=z_t,
            q=q_t,
            continuation_weight=cont_ind,
            mp=mp,
            kappa_issue=tp.kappa_issue,
        )
        cont_step = d_t + beta * tf.maximum(cont_next, 0.0)
        cont_next = valid_h * cont_step + (1.0 - valid_h) * cont_next
    return cont_next

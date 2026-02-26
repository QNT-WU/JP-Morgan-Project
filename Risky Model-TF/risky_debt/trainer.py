# risky_debt/trainer.py
# This file is the training engine. It:
# creates networks (policy/value/pricing),
# creates optimizers (Adam),
# creates training data (ergodic buffer),
# runs epochs,
# at each step computes a loss (Objective 1/2/3),
# takes gradients and updates the networks,
# evaluates “effectiveness measures” (TestReward, TestEuler…),
# logs and optionally saves checkpoints.
from __future__ import annotations

import os

os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

from typing import Dict, List, Optional, Tuple
import inspect

import numpy as np
import tensorflow as tf

from .config import (
    ModelParams,
    NetParams,
    TrainParams,
    Obj1Params,
    Obj2Params,
    Obj3Params,
)
from .networks import PolicyNet, ValueNet, VtildeNet, PricingNet

# from .networks import PolicyNet, ValueNet, PricingNet
from .simulation import simulate_ergodic_dataset, set_global_seed
from .objectives import obj1_loss, obj2_batch_loss, obj3_batch_loss
from .evaluation import (
    eval_test_reward,
    eval_test_euler_mse_policy_only,
    eval_test_euler_mse_obj3,
)
from .io_utils import JSONLLogger, TFCheckpointIO


# Gradients can be huge and unstable.
# This “clips” the gradient vector so it doesn’t explode:
# Think: “cap how big the step can be”.
# Then updates model parameters.
# ✅ Good engineering.
def _clip_and_apply(
    opt: tf.keras.optimizers.Optimizer, grads, vars_, clip: float
) -> None:
    grads, _ = tf.clip_by_global_norm(grads, clip)
    opt.apply_gradients(zip(grads, vars_))


# It calls obj1_loss / obj2_batch_loss / obj3_batch_loss.
# But it only passes arguments that the function actually accepts.
# This prevents crashes if you later change an objective signature.
def _call_objective(fn, **kwargs):
    """
    Calls objective function using only kwargs that match its signature.
    Prevents crashes if objective signatures evolve.
    """
    params = inspect.signature(fn).parameters
    call_kwargs = {k: v for k, v in kwargs.items() if k in params}
    return fn(**call_kwargs)


# ---------------- Objective 1 ----------------
def train_objective_1(
    mp: ModelParams,
    npol: NetParams,
    nq: NetParams,
    tp: TrainParams,
    op1: Obj1Params,
    jsonl_logger: Optional[JSONLLogger] = None,
    ckptio: Optional[TFCheckpointIO] = None,
) -> Tuple[PolicyNet, PricingNet, Dict[str, List[float]]]:

    # This calls tf.random.set_seed and np.random.seed.
    # But (important): because your objectives use randomness inside @tf.function, this is not enough for strict repeatability (explained below).
    set_global_seed(tp.seed)

    # Create networks
    # policy outputs (𝑘′,𝑏′)with constraints.
    # qnet outputs 𝑞∈[𝑞𝑚𝑖𝑛,𝑞𝑚𝑎𝑥]
    # PolicyNet needs (k_min, b_min, b_max)
    policy = PolicyNet(npol, mp.k_min, mp.b_min, mp.b_max)

    # PricingNet needs (q_min, q_max)
    qnet = PricingNet(nq, mp.q_min, mp.q_max)

    # build once
    # This forces Keras to create weights the first time the network is called.
    _ = policy(tf.zeros((1, 3), dtype=tf.float32))
    _ = qnet(tf.zeros((1, 3), dtype=tf.float32))

    # Adam is the gradient-based solver
    opt_policy = tf.keras.optimizers.Adam(tp.lr_policy)
    opt_q = tf.keras.optimizers.Adam(tp.lr_q)

    # Create initial ergodic buffer
    # This generates a big list of states from the Markov chain implied by the current policy.
    k_buf, b_buf, z_buf = simulate_ergodic_dataset(policy, mp, tp, seed=tp.seed + 10)

    hist = {"epoch": [], "train_reward": [], "test_reward": [], "test_euler_mse": []}

    # Training loop
    # For each epoch:
    # run many gradient steps
    # refresh ergodic buffer sometimes
    # evaluate TestReward and TestEuler metric
    # log + print
    for epoch in range(1, tp.epochs + 1):
        train_rewards = []

        for _ in range(tp.steps_per_epoch):
            # What happens inside obj1_loss:
            # It generates random rollouts (random initial states + random shocks)
            # It computes discounted reward 𝑊
            # It also computes a ZP pricing residual using two independent shocks
            # Final loss: loss=−TrainReward+νzp​⋅ZP_loss
            with tf.GradientTape(persistent=True) as tape:
                out = _call_objective(
                    obj1_loss,
                    policy=policy,
                    qnet=qnet,
                    mp=mp,
                    tp=tp,
                    op1=op1,
                )
                if isinstance(out, (tuple, list)):
                    loss = out[0]
                    train_reward = out[1]
                else:
                    loss = out
                    train_reward = -loss

            grads_p = tape.gradient(loss, policy.trainable_variables)
            grads_q = tape.gradient(loss, qnet.trainable_variables)
            del tape

            _clip_and_apply(
                opt_policy, grads_p, policy.trainable_variables, tp.grad_clip
            )
            _clip_and_apply(opt_q, grads_q, qnet.trainable_variables, tp.grad_clip)

            train_rewards.append(float(tf.convert_to_tensor(train_reward).numpy()))

        # Refresh ergodic buffer occasionally
        # But note: because you use different seed each epoch (seed + 100 + epoch),
        # even if training were deterministic, the buffer changes each epoch by design.
        # That’s not “wrong”, but it means you’re not holding the training distribution fixed.
        if epoch == 1 or (epoch % tp.ergodic_refresh_every == 0):
            k_buf, b_buf, z_buf = simulate_ergodic_dataset(
                policy, mp, tp, seed=tp.seed + 100 + epoch
            )

        # Choose test states from buffer
        idx = np.random.choice(
            len(k_buf),
            size=tp.N_test_states,
            replace=False if len(k_buf) >= tp.N_test_states else True,
        )
        k_test, b_test, z_test = k_buf[idx], b_buf[idx], z_buf[idx]

        test_reward = eval_test_reward(policy, qnet, mp, tp, seed=tp.seed + 200 + epoch)
        test_euler_mse = eval_test_euler_mse_policy_only(
            policy=policy,
            qnet=qnet,
            mp=mp,
            tp=tp,
            states_k=k_test,
            states_b=b_test,
            states_z=z_test,
            seed=tp.seed + 300 + epoch,
        )

        ep_train_reward = float(np.mean(train_rewards))

        hist["epoch"].append(epoch)
        hist["train_reward"].append(ep_train_reward)
        hist["test_reward"].append(test_reward)
        hist["test_euler_mse"].append(test_euler_mse)

        if jsonl_logger is not None:
            jsonl_logger.log(
                {
                    "objective": "obj1",
                    "epoch": epoch,
                    "train_reward": ep_train_reward,
                    "test_reward": test_reward,
                    "test_euler_mse": test_euler_mse,
                }
            )

        if ckptio is not None:
            ckptio.save(step=epoch)

        if epoch == 1 or epoch % max(1, tp.epochs // 10) == 0:
            print(
                f"[Obj1][{epoch:03d}] TrainReward={ep_train_reward:.4f} "
                f"TestReward={test_reward:.4f} TestEulerMSE={test_euler_mse:.6f}"
            )

    return policy, qnet, hist


# ---------------- Objective 2 ----------------
def train_objective_2(
    mp: ModelParams,
    npol: NetParams,
    nval: NetParams,
    nvt: NetParams,
    nq: NetParams,
    tp: TrainParams,
    op2: Obj2Params,
    jsonl_logger: Optional[JSONLLogger] = None,
    ckptio: Optional[TFCheckpointIO] = None,
) -> Tuple[PolicyNet, ValueNet, ValueNet, PricingNet, Dict[str, List[float]]]:
    set_global_seed(tp.seed + 1)

    policy = PolicyNet(npol, mp.k_min, mp.b_min, mp.b_max)
    value = ValueNet(nval)
    # vtilde = ValueNet(nvt)
    vtilde = VtildeNet(nvt)
    qnet = PricingNet(nq, mp.q_min, mp.q_max)

    _ = policy(tf.zeros((1, 3), dtype=tf.float32))
    _ = value(tf.zeros((1, 3), dtype=tf.float32))
    _ = vtilde(tf.zeros((1, 3), dtype=tf.float32))
    _ = qnet(tf.zeros((1, 3), dtype=tf.float32))

    opt_policy = tf.keras.optimizers.Adam(tp.lr_policy)
    opt_value = tf.keras.optimizers.Adam(tp.lr_value)
    opt_vtilde = tf.keras.optimizers.Adam(tp.lr_vtilde)
    opt_q = tf.keras.optimizers.Adam(tp.lr_q)

    k_buf, b_buf, z_buf = simulate_ergodic_dataset(policy, mp, tp, seed=tp.seed + 11)

    hist = {"epoch": [], "train_loss": [], "test_euler_mse": [], "test_reward": []}

    for epoch in range(1, tp.epochs + 1):
        if epoch == 1 or (epoch % tp.ergodic_refresh_every == 0):
            k_buf, b_buf, z_buf = simulate_ergodic_dataset(
                policy, mp, tp, seed=tp.seed + 110 + epoch
            )

        losses = []
        for _ in range(tp.steps_per_epoch):
            idx = np.random.choice(len(k_buf), size=tp.batch_size, replace=True)
            k = tf.convert_to_tensor(k_buf[idx], tf.float32)
            b = tf.convert_to_tensor(b_buf[idx], tf.float32)
            z = tf.convert_to_tensor(z_buf[idx], tf.float32)

            with tf.GradientTape(persistent=True) as tape:
                loss = _call_objective(
                    obj2_batch_loss,
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

            grads_p = tape.gradient(loss, policy.trainable_variables)
            grads_v = tape.gradient(loss, value.trainable_variables)
            grads_t = tape.gradient(loss, vtilde.trainable_variables)
            grads_q = tape.gradient(loss, qnet.trainable_variables)
            del tape

            _clip_and_apply(
                opt_policy, grads_p, policy.trainable_variables, tp.grad_clip
            )
            _clip_and_apply(opt_value, grads_v, value.trainable_variables, tp.grad_clip)
            _clip_and_apply(
                opt_vtilde, grads_t, vtilde.trainable_variables, tp.grad_clip
            )
            _clip_and_apply(opt_q, grads_q, qnet.trainable_variables, tp.grad_clip)

            losses.append(float(tf.convert_to_tensor(loss).numpy()))

        idx_t = np.random.choice(len(k_buf), size=tp.N_test_states, replace=True)
        k_test, b_test, z_test = k_buf[idx_t], b_buf[idx_t], z_buf[idx_t]

        test_euler_mse = eval_test_euler_mse_obj3(
            policy=policy,
            value=value,
            qnet=qnet,
            mp=mp,
            tp=tp,
            states_k=k_test,
            states_b=b_test,
            states_z=z_test,
            seed=tp.seed + 301 + epoch,
        )
        test_reward = eval_test_reward(policy, qnet, mp, tp, seed=tp.seed + 201 + epoch)

        ep_train_loss = float(np.mean(losses))

        hist["epoch"].append(epoch)
        hist["train_loss"].append(ep_train_loss)
        hist["test_euler_mse"].append(test_euler_mse)
        hist["test_reward"].append(test_reward)

        if jsonl_logger is not None:
            jsonl_logger.log(
                {
                    "objective": "obj2",
                    "epoch": epoch,
                    "train_loss": ep_train_loss,
                    "test_euler_mse": test_euler_mse,
                    "test_reward": test_reward,
                }
            )

        if ckptio is not None:
            ckptio.save(step=epoch)

        if epoch == 1 or epoch % max(1, tp.epochs // 10) == 0:
            print(
                f"[Obj2][{epoch:03d}] TrainLoss={ep_train_loss:.6f} "
                f"TestEulerMSE={test_euler_mse:.6f} TestReward={test_reward:.4f}"
            )

    return policy, value, vtilde, qnet, hist


# ---------------- Objective 3 ----------------
def train_objective_3(
    mp: ModelParams,
    npol: NetParams,
    nval: NetParams,
    nq: NetParams,
    tp: TrainParams,
    op3: Obj3Params,
    jsonl_logger: Optional[JSONLLogger] = None,
    ckptio: Optional[TFCheckpointIO] = None,
) -> Tuple[PolicyNet, ValueNet, PricingNet, Dict[str, List[float]]]:
    set_global_seed(tp.seed + 2)

    policy = PolicyNet(npol, mp.k_min, mp.b_min, mp.b_max)
    value = ValueNet(nval)
    qnet = PricingNet(nq, mp.q_min, mp.q_max)

    _ = policy(tf.zeros((1, 3), dtype=tf.float32))
    _ = value(tf.zeros((1, 3), dtype=tf.float32))
    _ = qnet(tf.zeros((1, 3), dtype=tf.float32))

    opt_policy = tf.keras.optimizers.Adam(tp.lr_policy)
    opt_value = tf.keras.optimizers.Adam(tp.lr_value)
    opt_q = tf.keras.optimizers.Adam(tp.lr_q)

    k_buf, b_buf, z_buf = simulate_ergodic_dataset(policy, mp, tp, seed=tp.seed + 12)

    hist = {"epoch": [], "train_loss": [], "test_euler_mse": [], "test_reward": []}

    for epoch in range(1, tp.epochs + 1):
        if epoch == 1 or (epoch % tp.ergodic_refresh_every == 0):
            k_buf, b_buf, z_buf = simulate_ergodic_dataset(
                policy, mp, tp, seed=tp.seed + 120 + epoch
            )

        losses = []
        for _ in range(tp.steps_per_epoch):
            idx = np.random.choice(len(k_buf), size=tp.batch_size, replace=True)
            k = tf.convert_to_tensor(k_buf[idx], tf.float32)
            b = tf.convert_to_tensor(b_buf[idx], tf.float32)
            z = tf.convert_to_tensor(z_buf[idx], tf.float32)

            with tf.GradientTape(persistent=True) as tape:
                loss = _call_objective(
                    obj3_batch_loss,
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

            grads_p = tape.gradient(loss, policy.trainable_variables)
            grads_v = tape.gradient(loss, value.trainable_variables)
            grads_q = tape.gradient(loss, qnet.trainable_variables)
            del tape

            _clip_and_apply(
                opt_policy, grads_p, policy.trainable_variables, tp.grad_clip
            )
            _clip_and_apply(opt_value, grads_v, value.trainable_variables, tp.grad_clip)
            _clip_and_apply(opt_q, grads_q, qnet.trainable_variables, tp.grad_clip)

            losses.append(float(tf.convert_to_tensor(loss).numpy()))

        idx_t = np.random.choice(len(k_buf), size=tp.N_test_states, replace=True)
        k_test, b_test, z_test = k_buf[idx_t], b_buf[idx_t], z_buf[idx_t]

        test_euler_mse = eval_test_euler_mse_obj3(
            policy=policy,
            value=value,
            qnet=qnet,
            mp=mp,
            tp=tp,
            states_k=k_test,
            states_b=b_test,
            states_z=z_test,
            seed=tp.seed + 302 + epoch,
        )
        test_reward = eval_test_reward(policy, qnet, mp, tp, seed=tp.seed + 202 + epoch)

        ep_train_loss = float(np.mean(losses))

        hist["epoch"].append(epoch)
        hist["train_loss"].append(ep_train_loss)
        hist["test_euler_mse"].append(test_euler_mse)
        hist["test_reward"].append(test_reward)

        if jsonl_logger is not None:
            jsonl_logger.log(
                {
                    "objective": "obj3",
                    "epoch": epoch,
                    "train_loss": ep_train_loss,
                    "test_euler_mse": test_euler_mse,
                    "test_reward": test_reward,
                }
            )

        if ckptio is not None:
            ckptio.save(step=epoch)

        if epoch == 1 or epoch % max(1, tp.epochs // 10) == 0:
            print(
                f"[Obj3][{epoch:03d}] TrainLoss={ep_train_loss:.6f} "
                f"TestEulerMSE={test_euler_mse:.6f} TestReward={test_reward:.4f}"
            )

    return policy, value, qnet, hist

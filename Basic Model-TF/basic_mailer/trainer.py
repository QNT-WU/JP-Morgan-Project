# src/basic_mailer/trainer.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from .config import ModelParams, NetParams, TrainParams, Obj3Params
from .networks import PolicyNet, ValueNet
from .simulation import simulate_ergodic_dataset, set_global_seed
from .objectives import obj1_loss, obj2_batch_loss, obj3_batch_loss
from .evaluation import (
    eval_test_reward,
    eval_test_euler_mse_policy_only,
    eval_test_euler_mse_obj3,
)
from .io_utils import JSONLLogger, TFCheckpointIO


# Utility: gradient clipping + optimizer step
# tf.clip_by_global_norm rescales the whole gradient vector if its norm > clip
# opt.apply_gradients(zip(grads, vars_)) applies updates
def _clip_and_apply(
    opt: tf.keras.optimizers.Optimizer, grads, vars_, clip: float
) -> None:
    grads, _ = tf.clip_by_global_norm(grads, clip)
    opt.apply_gradients(zip(grads, vars_))


# Returns: trained policy, history dict hist
def train_objective_1(
    mp: ModelParams,
    npol: NetParams,
    tp: TrainParams,
    jsonl_logger: Optional[JSONLLogger] = None,
    ckptio: Optional[TFCheckpointIO] = None,
) -> Tuple[PolicyNet, Dict[str, List[float]]]:
    set_global_seed(tp.seed)
    policy = PolicyNet(npol, mp.k_min, mp.k_max)

    # “build” the networks once right after you create them
    _ = policy(tf.zeros((1, 2), dtype=tf.float32))

    opt = tf.keras.optimizers.Adam(tp.lr_policy)

    # bind ckptio after objects exist (if provided)
    if ckptio is None:
        # no checkpointing
        pass

    # Initial ergodic dataset build
    # Simulates the policy-induced Markov chain to get an empirical sample of (𝑘,𝑧)
    # But also: this calls policy(x) inside simulation, which builds the policy network and creates its variables.
    k_buf, z_buf = simulate_ergodic_dataset(policy, mp, tp, seed=tp.seed + 10)

    # These keys must match plotting code.
    hist = {"epoch": [], "train_reward": [], "test_reward": [], "test_euler_mse": []}

    # If ckptio was passed-in pre-constructed, it must reference THIS policy/opt.
    # Recommended: create ckptio in run_all.py (shown below) after policy/opt exist.
    # But to support both patterns, we only restore if ckptio matches.
    # (Most users will follow run_all.py approach.)
    # NOTE: do not restore here by default.

    for epoch in range(1, tp.epochs + 1):
        train_rewards = []

        # Steps per epoch (training updates)
        # obj1_loss runs rollouts and produces
        # loss = -mean(discounted reward)
        # train_reward = mean(discounted reward)
        for _ in range(tp.steps_per_epoch):
            with tf.GradientTape() as tape:
                loss, train_reward = obj1_loss(policy, mp, tp)
            grads = tape.gradient(loss, policy.trainable_variables)
            _clip_and_apply(opt, grads, policy.trainable_variables, tp.grad_clip)
            train_rewards.append(float(train_reward.numpy()))

        # Refresh ergodic dataset sometimes
        # policy changes ⇒ ergodic distribution changes
        # refresh every K epochs, not every step (costly)
        if epoch == 1 or (epoch % tp.ergodic_refresh_every == 0):
            k_buf, z_buf = simulate_ergodic_dataset(
                policy, mp, tp, seed=tp.seed + 100 + epoch
            )

        # Sample test states from ergodic buffer
        # So test states come from the ergodic distribution induced by current policy
        idx = np.random.choice(
            len(k_buf),
            size=tp.N_test_states,
            replace=False if len(k_buf) >= tp.N_test_states else True,
        )
        k_test = k_buf[idx]
        z_test = z_buf[idx]

        # Compute Obj1 effectiveness measures
        test_reward = eval_test_reward(policy, mp, tp, seed=tp.seed + 200 + epoch)

        # This computes Obj1 Euler diagnostic on test states
        # Uses policy-only Euler residual f
        test_euler_mse = eval_test_euler_mse_policy_only(
            policy, mp, k_test, z_test, N_eps=tp.N_eps_test, seed=tp.seed + 300 + epoch
        )
        # Average training reward across steps
        # It is an average over gradient steps within the epoch
        # ep_train_reward is the epoch-level average of Monte Carlo estimates of expected lifetime reward,
        # where time and cross-sectional averaging already happened inside each step.
        ep_train_reward = float(np.mean(train_rewards))

        # Save to hist, log to JSONL, save checkpoints, print
        hist["epoch"].append(epoch)
        hist["train_reward"].append(ep_train_reward)
        hist["test_reward"].append(test_reward)
        hist["test_euler_mse"].append(test_euler_mse)

        # JSONL logging (one line per epoch)
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

        # checkpoint save per epoch
        if ckptio is not None:
            ckptio.save(step=epoch)

        if epoch == 1 or epoch % max(1, tp.epochs // 10) == 0:
            print(
                f"[Obj1][{epoch:03d}] TrainReward={ep_train_reward:.4f} "
                f"TestReward={test_reward:.4f} TestEulerMSE={test_euler_mse:.6f}"
            )

    return policy, hist


def train_objective_2(
    mp: ModelParams,
    npol: NetParams,
    tp: TrainParams,
    jsonl_logger: Optional[JSONLLogger] = None,
    ckptio: Optional[TFCheckpointIO] = None,
) -> Tuple[PolicyNet, Dict[str, List[float]]]:
    set_global_seed(tp.seed + 1)
    policy = PolicyNet(npol, mp.k_min, mp.k_max)

    _ = policy(tf.zeros((1, 2), dtype=tf.float32))

    opt = tf.keras.optimizers.Adam(tp.lr_policy)

    # Build initial ergodic dataset
    # This both:
    # creates data
    # builds policy variables by calling policy inside simulation
    k_buf, z_buf = simulate_ergodic_dataset(policy, mp, tp, seed=tp.seed + 11)

    hist = {"epoch": [], "train_loss": [], "test_euler_mse": [], "test_reward": []}

    for epoch in range(1, tp.epochs + 1):
        # Refresh ergodic dataset
        if epoch == 1 or (epoch % tp.ergodic_refresh_every == 0):
            k_buf, z_buf = simulate_ergodic_dataset(
                policy, mp, tp, seed=tp.seed + 110 + epoch
            )

        losses = []
        for _ in range(tp.steps_per_epoch):
            # Training steps: sample batch from ergodic buffer
            # random minibatch sampling from ergodic distribution
            # convert to TF tensors shape [batch_size]
            idx = np.random.choice(len(k_buf), size=tp.batch_size, replace=True)
            k = tf.convert_to_tensor(k_buf[idx], tf.float32)
            z = tf.convert_to_tensor(z_buf[idx], tf.float32)

            # compute loss
            with tf.GradientTape() as tape:
                loss = obj2_batch_loss(policy, mp, k, z)
            # Compute gradients and apply:
            grads = tape.gradient(loss, policy.trainable_variables)
            _clip_and_apply(opt, grads, policy.trainable_variables, tp.grad_clip)
            losses.append(float(loss.numpy()))

        # Sample test states from buffer
        # Compute: test Euler MSE, test reward
        idx_t = np.random.choice(len(k_buf), size=tp.N_test_states, replace=True)
        k_test = k_buf[idx_t]
        z_test = z_buf[idx_t]

        test_euler_mse = eval_test_euler_mse_policy_only(
            policy, mp, k_test, z_test, N_eps=tp.N_eps_test, seed=tp.seed + 301 + epoch
        )
        test_reward = eval_test_reward(policy, mp, tp, seed=tp.seed + 201 + epoch)

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

    return policy, hist


def train_objective_3(
    mp: ModelParams,
    npol: NetParams,
    nval: NetParams,
    tp: TrainParams,
    op3: Obj3Params,
    jsonl_logger: Optional[JSONLLogger] = None,
    ckptio: Optional[TFCheckpointIO] = None,
) -> Tuple[PolicyNet, ValueNet, Dict[str, List[float]]]:
    # Initialize: Two networks and two optimizers.
    set_global_seed(tp.seed + 2)
    policy = PolicyNet(npol, mp.k_min, mp.k_max)
    value = ValueNet(nval)

    _ = policy(tf.zeros((1, 2), dtype=tf.float32))
    _ = value(tf.zeros((1, 2), dtype=tf.float32))

    opt_policy = tf.keras.optimizers.Adam(tp.lr_policy)
    opt_value = tf.keras.optimizers.Adam(tp.lr_value)

    # This builds policy weights.
    # But note: value net is not built yet unless it is called somewhere
    # Obj3 loss calls value(x) inside obj3_batch_loss, so it will build during first training step
    k_buf, z_buf = simulate_ergodic_dataset(policy, mp, tp, seed=tp.seed + 12)

    hist = {"epoch": [], "train_loss": [], "test_euler_mse": [], "test_reward": []}

    for epoch in range(1, tp.epochs + 1):
        if epoch == 1 or (epoch % tp.ergodic_refresh_every == 0):
            k_buf, z_buf = simulate_ergodic_dataset(
                policy, mp, tp, seed=tp.seed + 120 + epoch
            )

        losses = []
        for _ in range(tp.steps_per_epoch):
            idx = np.random.choice(len(k_buf), size=tp.batch_size, replace=True)
            k = tf.convert_to_tensor(k_buf[idx], tf.float32)
            z = tf.convert_to_tensor(z_buf[idx], tf.float32)

            with tf.GradientTape(persistent=True) as tape:
                loss = obj3_batch_loss(policy, value, mp, op3, k, z)

            grads_p = tape.gradient(loss, policy.trainable_variables)
            grads_v = tape.gradient(loss, value.trainable_variables)
            del tape

            _clip_and_apply(
                opt_policy, grads_p, policy.trainable_variables, tp.grad_clip
            )
            _clip_and_apply(opt_value, grads_v, value.trainable_variables, tp.grad_clip)

            losses.append(float(loss.numpy()))

        idx_t = np.random.choice(len(k_buf), size=tp.N_test_states, replace=True)
        k_test = k_buf[idx_t]
        z_test = z_buf[idx_t]

        test_euler_mse = eval_test_euler_mse_obj3(
            policy,
            value,
            mp,
            k_test,
            z_test,
            N_eps=tp.N_eps_test,
            seed=tp.seed + 302 + epoch,
        )
        test_reward = eval_test_reward(policy, mp, tp, seed=tp.seed + 202 + epoch)

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

    return policy, value, hist

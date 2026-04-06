"""Objective-specific trainer classes."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from basic_mailer.config import ModelParams, NetParams, Obj3Params, TrainParams
from basic_mailer.evaluation import (
    eval_test_euler_mse_obj3,
    eval_test_euler_mse_policy_only,
    eval_test_reward,
)
from basic_mailer.io_utils import JSONLLogger, TFCheckpointIO
from basic_mailer.networks import PolicyNet, ValueNet
from basic_mailer.objectives import obj1_loss, obj2_batch_loss, obj3_batch_loss

from .base import BaseTrainer


class Objective1Trainer(BaseTrainer):
    """Trainer for Mailer Objective 1 (lifetime reward maximization)."""

    objective_name = "obj1"

    def __init__(
        self,
        *,
        mp: ModelParams,
        tp: TrainParams,
        npol: Optional[NetParams] = None,
        policy: Optional[PolicyNet] = None,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        jsonl_logger: Optional[JSONLLogger] = None,
        ckptio: Optional[TFCheckpointIO] = None,
        verbose: bool = False,
        progress_stride: Optional[int] = None,
    ) -> None:
        """Build the Objective 1 trainer and its policy-side optimization state.

        The trainer either accepts a prebuilt policy network or constructs one
        from ``npol``, then attaches the optimizer, logging, and progress
        reporting utilities used during lifetime-reward maximization.
        """
        if policy is None:
            if npol is None:
                raise ValueError("Either 'policy' or 'npol' must be provided.")
            policy = PolicyNet(npol, mp.k_min, mp.k_max)
        _ = policy(tf.zeros((1, 2), dtype=tf.float32))
        super().__init__(
            mp=mp,
            tp=tp,
            policy=policy,
            jsonl_logger=jsonl_logger,
            ckptio=ckptio,
        )
        self.optimizer = optimizer or tf.keras.optimizers.Adam(tp.lr_policy)
        self.verbose = verbose
        self.progress_stride = progress_stride or max(1, tp.epochs // 10)
        self.history = {
            "epoch": [],
            "train_reward": [],
            "train_loss": [],
            "test_reward": [],
            "test_euler_mse": [],
        }

    def train(self) -> Tuple[PolicyNet, Dict[str, List[float]]]:
        """Run the training loop and return the trained objects."""
        self.initialize(seed_offset=0)
        for epoch in range(1, self.tp.epochs + 1):
            train_rewards = []
            for _ in range(self.tp.steps_per_epoch):
                with tf.GradientTape() as tape:
                    loss, train_reward = obj1_loss(self.policy, self.mp, self.tp)
                gradients = tape.gradient(loss, self.policy.trainable_variables)
                self.clip_and_apply(
                    self.optimizer,
                    gradients,
                    self.policy.trainable_variables,
                    self.tp.grad_clip,
                )
                train_rewards.append(float(train_reward.numpy()))

            self.maybe_refresh_epoch_buffer(epoch, seed_base=self.tp.seed + 100)
            k_test, z_test = self.sample_test_states()
            test_reward = eval_test_reward(
                self.policy,
                self.mp,
                self.tp,
                seed=self.tp.seed + 200 + epoch,
            )
            test_euler_mse = eval_test_euler_mse_policy_only(
                self.policy,
                self.mp,
                k_test,
                z_test,
                N_eps=self.tp.N_eps_test,
                seed=self.tp.seed + 300 + epoch,
            )

            epoch_train_reward = float(np.mean(train_rewards))
            epoch_train_loss = -epoch_train_reward
            record = {
                "epoch": epoch,
                "train_reward": epoch_train_reward,
                "train_loss": epoch_train_loss,
                "test_reward": test_reward,
                "test_euler_mse": test_euler_mse,
            }
            for key, value in record.items():
                self.history.setdefault(key, []).append(value)
            self.log_epoch(record)
            self.save_checkpoint(epoch)
            if self.verbose and (epoch == 1 or epoch % self.progress_stride == 0):
                print(
                    f"[Obj1][{epoch:03d}] TrainReward={epoch_train_reward:.4f} "
                    f"TestReward={test_reward:.4f} TestEulerMSE={test_euler_mse:.6f}"
                )
        return self.policy, self.history


class Objective2Trainer(BaseTrainer):
    """Trainer for Mailer Objective 2 (Euler residual minimization)."""

    objective_name = "obj2"

    def __init__(
        self,
        *,
        mp: ModelParams,
        tp: TrainParams,
        npol: Optional[NetParams] = None,
        policy: Optional[PolicyNet] = None,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        jsonl_logger: Optional[JSONLLogger] = None,
        ckptio: Optional[TFCheckpointIO] = None,
        verbose: bool = False,
        progress_stride: Optional[int] = None,
    ) -> None:
        """Build the Objective 2 trainer and its Euler-residual optimizer state.

        The constructor wires together the policy network, optimizer, and
        bookkeeping utilities required for AiO Euler-residual minimization.
        """
        if policy is None:
            if npol is None:
                raise ValueError("Either 'policy' or 'npol' must be provided.")
            policy = PolicyNet(npol, mp.k_min, mp.k_max)
        _ = policy(tf.zeros((1, 2), dtype=tf.float32))
        super().__init__(
            mp=mp,
            tp=tp,
            policy=policy,
            jsonl_logger=jsonl_logger,
            ckptio=ckptio,
        )
        self.optimizer = optimizer or tf.keras.optimizers.Adam(tp.lr_policy)
        self.verbose = verbose
        self.progress_stride = progress_stride or max(1, tp.epochs // 10)
        self.history = {
            "epoch": [],
            "train_loss": [],
            "test_euler_mse": [],
            "test_reward": [],
        }

    def train(self) -> Tuple[PolicyNet, Dict[str, List[float]]]:
        """Run the training loop and return the trained objects."""
        self.initialize(seed_offset=1)
        for epoch in range(1, self.tp.epochs + 1):
            self.maybe_refresh_epoch_buffer(epoch, seed_base=self.tp.seed + 110)
            losses = []
            for _ in range(self.tp.steps_per_epoch):
                k, z = self.sample_train_batch()
                with tf.GradientTape() as tape:
                    loss = obj2_batch_loss(self.policy, self.mp, k, z)
                gradients = tape.gradient(loss, self.policy.trainable_variables)
                self.clip_and_apply(
                    self.optimizer,
                    gradients,
                    self.policy.trainable_variables,
                    self.tp.grad_clip,
                )
                losses.append(float(loss.numpy()))

            k_test, z_test = self.sample_test_states()
            test_euler_mse = eval_test_euler_mse_policy_only(
                self.policy,
                self.mp,
                k_test,
                z_test,
                N_eps=self.tp.N_eps_test,
                seed=self.tp.seed + 301 + epoch,
            )
            test_reward = eval_test_reward(
                self.policy,
                self.mp,
                self.tp,
                seed=self.tp.seed + 201 + epoch,
            )
            epoch_loss = float(np.mean(losses))
            record = {
                "epoch": epoch,
                "train_loss": epoch_loss,
                "test_euler_mse": test_euler_mse,
                "test_reward": test_reward,
            }
            for key, value in record.items():
                self.history.setdefault(key, []).append(value)
            self.log_epoch(record)
            self.save_checkpoint(epoch)
            if self.verbose and (epoch == 1 or epoch % self.progress_stride == 0):
                print(
                    f"[Obj2][{epoch:03d}] TrainLoss={epoch_loss:.6f} "
                    f"TestEulerMSE={test_euler_mse:.6f} TestReward={test_reward:.4f}"
                )
        return self.policy, self.history


class Objective3Trainer(BaseTrainer):
    """Trainer for Mailer Objective 3 (Bellman + Euler residual minimization)."""

    objective_name = "obj3"

    def __init__(
        self,
        *,
        mp: ModelParams,
        tp: TrainParams,
        op3: Obj3Params,
        npol: Optional[NetParams] = None,
        nval: Optional[NetParams] = None,
        policy: Optional[PolicyNet] = None,
        value: Optional[ValueNet] = None,
        policy_optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        value_optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        jsonl_logger: Optional[JSONLLogger] = None,
        ckptio: Optional[TFCheckpointIO] = None,
        verbose: bool = False,
        progress_stride: Optional[int] = None,
    ) -> None:
        """Build the Objective 3 trainer with policy and value networks.

        Objective 3 jointly trains a policy model and a value model, so the
        initializer validates or constructs both objects and prepares the paired
        optimizers and reporting hooks used during Bellman-residual training.
        """
        if policy is None:
            if npol is None:
                raise ValueError("Either 'policy' or 'npol' must be provided.")
            policy = PolicyNet(npol, mp.k_min, mp.k_max)
        if value is None:
            if nval is None:
                raise ValueError("Either 'value' or 'nval' must be provided.")
            value = ValueNet(nval)
        _ = policy(tf.zeros((1, 2), dtype=tf.float32))
        _ = value(tf.zeros((1, 2), dtype=tf.float32))
        super().__init__(
            mp=mp,
            tp=tp,
            policy=policy,
            value=value,
            op3=op3,
            jsonl_logger=jsonl_logger,
            ckptio=ckptio,
        )
        self.policy_optimizer = policy_optimizer or tf.keras.optimizers.Adam(tp.lr_policy)
        self.value_optimizer = value_optimizer or tf.keras.optimizers.Adam(tp.lr_value)
        self.verbose = verbose
        self.progress_stride = progress_stride or max(1, tp.epochs // 10)
        self.history = {
            "epoch": [],
            "train_loss": [],
            "test_euler_mse": [],
            "test_reward": [],
        }

    def train(self) -> Tuple[PolicyNet, ValueNet, Dict[str, List[float]]]:
        """Run the training loop and return the trained objects."""
        self.initialize(seed_offset=2)
        for epoch in range(1, self.tp.epochs + 1):
            self.maybe_refresh_epoch_buffer(epoch, seed_base=self.tp.seed + 120)
            losses = []
            for _ in range(self.tp.steps_per_epoch):
                k, z = self.sample_train_batch()
                with tf.GradientTape(persistent=True) as tape:
                    loss = obj3_batch_loss(self.policy, self.value, self.mp, self.op3, k, z)
                policy_grads = tape.gradient(loss, self.policy.trainable_variables)
                value_grads = tape.gradient(loss, self.value.trainable_variables)
                del tape
                self.clip_and_apply(
                    self.policy_optimizer,
                    policy_grads,
                    self.policy.trainable_variables,
                    self.tp.grad_clip,
                )
                self.clip_and_apply(
                    self.value_optimizer,
                    value_grads,
                    self.value.trainable_variables,
                    self.tp.grad_clip,
                )
                losses.append(float(loss.numpy()))

            k_test, z_test = self.sample_test_states()
            test_euler_mse = eval_test_euler_mse_obj3(
                self.policy,
                self.value,
                self.mp,
                k_test,
                z_test,
                N_eps=self.tp.N_eps_test,
                seed=self.tp.seed + 302 + epoch,
            )
            test_reward = eval_test_reward(
                self.policy,
                self.mp,
                self.tp,
                seed=self.tp.seed + 202 + epoch,
            )
            epoch_loss = float(np.mean(losses))
            record = {
                "epoch": epoch,
                "train_loss": epoch_loss,
                "test_euler_mse": test_euler_mse,
                "test_reward": test_reward,
            }
            for key, value in record.items():
                self.history.setdefault(key, []).append(value)
            self.log_epoch(record)
            self.save_checkpoint(epoch)
            if self.verbose and (epoch == 1 or epoch % self.progress_stride == 0):
                print(
                    f"[Obj3][{epoch:03d}] TrainLoss={epoch_loss:.6f} "
                    f"TestEulerMSE={test_euler_mse:.6f} TestReward={test_reward:.4f}"
                )
        return self.policy, self.value, self.history

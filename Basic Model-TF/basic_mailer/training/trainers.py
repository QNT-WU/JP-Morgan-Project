"""Objective-specific trainer classes."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from basic_mailer.config import ModelParams, NetParams, Obj2Params, Obj3Params, TrainParams
from basic_mailer.evaluation import (
    eval_obj1_kkt_diagnostics,
    eval_obj2_kkt_diagnostics,
    eval_obj3_kkt_diagnostics,
    eval_test_reward,
)
from basic_mailer.io_utils import JSONLLogger, TFCheckpointIO
from basic_mailer.networks import MultiplierNet, PolicyNet, ValueNet
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
        """Build Objective 1's policy-side optimization state."""
        if policy is None:
            if npol is None:
                raise ValueError("Either 'policy' or 'npol' must be provided.")
            policy = PolicyNet(npol, mp.k_min, mp.k_max)
        _ = policy(tf.zeros((1, 2), dtype=tf.float32))
        super().__init__(mp=mp, tp=tp, policy=policy, jsonl_logger=jsonl_logger, ckptio=ckptio)
        self.optimizer = optimizer or tf.keras.optimizers.Adam(tp.lr_policy)
        self.verbose = verbose
        self.progress_stride = progress_stride or max(1, tp.epochs // 10)
        self.history = {
            "epoch": [],
            "train_reward": [],
            "train_loss": [],
            "test_reward": [],
            "test_euler_mse": [],
            "test_kkt_mse": [],
            "test_euler_mse_interior": [],
            "interior_share": [],
            "mean_slack": [],
        }

    def train(self) -> Tuple[PolicyNet, Dict[str, List[float]]]:
        """Run the training loop and return the trained policy and history."""
        self.initialize(seed_offset=0)
        for epoch in range(1, self.tp.epochs + 1):
            train_rewards = []
            for _ in range(self.tp.steps_per_epoch):
                with tf.GradientTape() as tape:
                    loss, train_reward = obj1_loss(self.policy, self.mp, self.tp)
                gradients = tape.gradient(loss, self.policy.trainable_variables)
                self.clip_and_apply(self.optimizer, gradients, self.policy.trainable_variables, self.tp.grad_clip)
                train_rewards.append(float(train_reward.numpy()))

            self.maybe_refresh_epoch_buffer(epoch, seed_base=self.tp.seed + 100)
            k_test, z_test = self.sample_test_states()
            test_reward = eval_test_reward(self.policy, self.mp, self.tp, seed=self.tp.seed + 200 + epoch)
            diagnostics = eval_obj1_kkt_diagnostics(
                self.policy,
                self.mp,
                k_test,
                z_test,
                N_eps=self.tp.N_eps_test,
                seed=self.tp.seed + 300 + epoch,
            )
            epoch_train_reward = float(np.mean(train_rewards))
            record = {
                "epoch": epoch,
                "train_reward": epoch_train_reward,
                "train_loss": -epoch_train_reward,
                "test_reward": test_reward,
                **diagnostics,
            }
            for key, value in record.items():
                self.history.setdefault(key, []).append(value)
            self.log_epoch(record)
            self.save_checkpoint(epoch)
            if self.verbose and (epoch == 1 or epoch % self.progress_stride == 0):
                print(
                    f"[Obj1][{epoch:03d}] TrainReward={epoch_train_reward:.4f} "
                    f"TestReward={test_reward:.4f} TestKKT={record['test_kkt_mse']:.6f}"
                )
        return self.policy, self.history


class Objective2Trainer(BaseTrainer):
    """Trainer for Mailer Objective 2 (KKT/Euler residual minimization)."""

    objective_name = "obj2"

    def __init__(
        self,
        *,
        mp: ModelParams,
        tp: TrainParams,
        op2: Obj2Params | None = None,
        npol: Optional[NetParams] = None,
        nlam: Optional[NetParams] = None,
        policy: Optional[PolicyNet] = None,
        multiplier: Optional[MultiplierNet] = None,
        policy_optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        multiplier_optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        jsonl_logger: Optional[JSONLLogger] = None,
        ckptio: Optional[TFCheckpointIO] = None,
        verbose: bool = False,
        progress_stride: Optional[int] = None,
    ) -> None:
        """Build Objective 2 with policy and nonnegative multiplier networks."""
        if policy is None:
            if npol is None:
                raise ValueError("Either 'policy' or 'npol' must be provided.")
            policy = PolicyNet(npol, mp.k_min, mp.k_max)
        if multiplier is None:
            nlam = nlam or npol
            if nlam is None:
                raise ValueError("Either 'multiplier', 'nlam', or 'npol' must be provided.")
            multiplier = MultiplierNet(nlam)
        _ = policy(tf.zeros((1, 2), dtype=tf.float32))
        _ = multiplier(tf.zeros((1, 2), dtype=tf.float32))
        op2 = op2 or Obj2Params()
        super().__init__(
            mp=mp,
            tp=tp,
            policy=policy,
            multiplier=multiplier,
            op2=op2,
            jsonl_logger=jsonl_logger,
            ckptio=ckptio,
        )
        # Backward-compatible single optimizer support. If supplied, it updates both networks.
        self.shared_optimizer = optimizer
        self.policy_optimizer = policy_optimizer or optimizer or tf.keras.optimizers.Adam(tp.lr_policy)
        self.multiplier_optimizer = multiplier_optimizer or optimizer or tf.keras.optimizers.Adam(tp.lr_policy)
        self.verbose = verbose
        self.progress_stride = progress_stride or max(1, tp.epochs // 10)
        self.history = {
            "epoch": [],
            "train_loss": [],
            "test_fb_mse": [],
            "test_stationarity_mse": [],
            "test_kkt_mse": [],
            "test_euler_mse": [],
            "test_reward": [],
        }

    def train(self) -> Tuple[PolicyNet, MultiplierNet, Dict[str, List[float]]]:
        """Run the training loop and return trained policy, multiplier, and history."""
        assert self.multiplier is not None and self.op2 is not None
        self.initialize(seed_offset=1)
        for epoch in range(1, self.tp.epochs + 1):
            self.maybe_refresh_epoch_buffer(epoch, seed_base=self.tp.seed + 110)
            losses = []
            for _ in range(self.tp.steps_per_epoch):
                k, z = self.sample_train_batch()
                variables = self.policy.trainable_variables + self.multiplier.trainable_variables
                with tf.GradientTape() as tape:
                    loss = obj2_batch_loss(self.policy, self.multiplier, self.mp, self.op2, k, z)
                gradients = tape.gradient(loss, variables)
                if self.shared_optimizer is not None:
                    self.clip_and_apply(self.shared_optimizer, gradients, variables, self.tp.grad_clip)
                else:
                    n_policy = len(self.policy.trainable_variables)
                    self.clip_and_apply(
                        self.policy_optimizer,
                        gradients[:n_policy],
                        self.policy.trainable_variables,
                        self.tp.grad_clip,
                    )
                    self.clip_and_apply(
                        self.multiplier_optimizer,
                        gradients[n_policy:],
                        self.multiplier.trainable_variables,
                        self.tp.grad_clip,
                    )
                losses.append(float(loss.numpy()))

            k_test, z_test = self.sample_test_states()
            diagnostics = eval_obj2_kkt_diagnostics(
                self.policy,
                self.multiplier,
                self.mp,
                self.op2,
                k_test,
                z_test,
                N_eps=self.tp.N_eps_test,
                seed=self.tp.seed + 301 + epoch,
            )
            test_reward = eval_test_reward(self.policy, self.mp, self.tp, seed=self.tp.seed + 201 + epoch)
            record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "test_reward": test_reward, **diagnostics}
            for key, value in record.items():
                self.history.setdefault(key, []).append(value)
            self.log_epoch(record)
            self.save_checkpoint(epoch)
            if self.verbose and (epoch == 1 or epoch % self.progress_stride == 0):
                print(
                    f"[Obj2][{epoch:03d}] TrainLoss={record['train_loss']:.6f} "
                    f"TestKKT={record['test_kkt_mse']:.6f} TestReward={test_reward:.4f}"
                )
        return self.policy, self.multiplier, self.history


class Objective3Trainer(BaseTrainer):
    """Trainer for Mailer Objective 3 (Bellman residual with KKT/FB)."""

    objective_name = "obj3"

    def __init__(
        self,
        *,
        mp: ModelParams,
        tp: TrainParams,
        op3: Obj3Params,
        npol: Optional[NetParams] = None,
        nval: Optional[NetParams] = None,
        nlam: Optional[NetParams] = None,
        policy: Optional[PolicyNet] = None,
        value: Optional[ValueNet] = None,
        multiplier: Optional[MultiplierNet] = None,
        policy_optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        value_optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        multiplier_optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        jsonl_logger: Optional[JSONLLogger] = None,
        ckptio: Optional[TFCheckpointIO] = None,
        verbose: bool = False,
        progress_stride: Optional[int] = None,
    ) -> None:
        """Build Objective 3 with policy, value, and multiplier networks."""
        if policy is None:
            if npol is None:
                raise ValueError("Either 'policy' or 'npol' must be provided.")
            policy = PolicyNet(npol, mp.k_min, mp.k_max)
        if value is None:
            if nval is None:
                raise ValueError("Either 'value' or 'nval' must be provided.")
            value = ValueNet(nval)
        if multiplier is None:
            nlam = nlam or npol
            if nlam is None:
                raise ValueError("Either 'multiplier', 'nlam', or 'npol' must be provided.")
            multiplier = MultiplierNet(nlam)
        _ = policy(tf.zeros((1, 2), dtype=tf.float32))
        _ = value(tf.zeros((1, 2), dtype=tf.float32))
        _ = multiplier(tf.zeros((1, 2), dtype=tf.float32))
        super().__init__(
            mp=mp,
            tp=tp,
            policy=policy,
            value=value,
            multiplier=multiplier,
            op3=op3,
            jsonl_logger=jsonl_logger,
            ckptio=ckptio,
        )
        self.policy_optimizer = policy_optimizer or tf.keras.optimizers.Adam(tp.lr_policy)
        self.value_optimizer = value_optimizer or tf.keras.optimizers.Adam(tp.lr_value)
        self.multiplier_optimizer = multiplier_optimizer or tf.keras.optimizers.Adam(tp.lr_policy)
        self.verbose = verbose
        self.progress_stride = progress_stride or max(1, tp.epochs // 10)
        self.history = {
            "epoch": [],
            "train_loss": [],
            "test_bellman_mse": [],
            "test_fb_mse": [],
            "test_stationarity_mse": [],
            "test_total_residual": [],
            "test_euler_mse": [],
            "test_reward": [],
        }

    def train(self) -> Tuple[PolicyNet, ValueNet, MultiplierNet, Dict[str, List[float]]]:
        """Run the training loop and return trained objects and history."""
        assert self.value is not None and self.multiplier is not None and self.op3 is not None
        self.initialize(seed_offset=2)
        for epoch in range(1, self.tp.epochs + 1):
            self.maybe_refresh_epoch_buffer(epoch, seed_base=self.tp.seed + 120)
            losses = []
            for _ in range(self.tp.steps_per_epoch):
                k, z = self.sample_train_batch()
                with tf.GradientTape(persistent=True) as tape:
                    loss = obj3_batch_loss(self.policy, self.value, self.multiplier, self.mp, self.op3, k, z)
                policy_grads = tape.gradient(loss, self.policy.trainable_variables)
                value_grads = tape.gradient(loss, self.value.trainable_variables)
                multiplier_grads = tape.gradient(loss, self.multiplier.trainable_variables)
                del tape
                self.clip_and_apply(self.policy_optimizer, policy_grads, self.policy.trainable_variables, self.tp.grad_clip)
                self.clip_and_apply(self.value_optimizer, value_grads, self.value.trainable_variables, self.tp.grad_clip)
                self.clip_and_apply(
                    self.multiplier_optimizer,
                    multiplier_grads,
                    self.multiplier.trainable_variables,
                    self.tp.grad_clip,
                )
                losses.append(float(loss.numpy()))

            k_test, z_test = self.sample_test_states()
            diagnostics = eval_obj3_kkt_diagnostics(
                self.policy,
                self.value,
                self.multiplier,
                self.mp,
                self.op3,
                k_test,
                z_test,
                N_eps=self.tp.N_eps_test,
                seed=self.tp.seed + 302 + epoch,
            )
            test_reward = eval_test_reward(self.policy, self.mp, self.tp, seed=self.tp.seed + 202 + epoch)
            record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "test_reward": test_reward, **diagnostics}
            for key, value in record.items():
                self.history.setdefault(key, []).append(value)
            self.log_epoch(record)
            self.save_checkpoint(epoch)
            if self.verbose and (epoch == 1 or epoch % self.progress_stride == 0):
                print(
                    f"[Obj3][{epoch:03d}] TrainLoss={record['train_loss']:.6f} "
                    f"TotalResidual={record['test_total_residual']:.6f} TestReward={test_reward:.4f}"
                )
        return self.policy, self.value, self.multiplier, self.history

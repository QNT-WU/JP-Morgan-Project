"""Base training abstractions for Mailer objectives."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from basic_mailer.config import ModelParams, Obj2Params, Obj3Params, TrainParams
from basic_mailer.io_utils import JSONLLogger, TFCheckpointIO
from basic_mailer.networks import MultiplierNet, PolicyNet, ValueNet
from basic_mailer.simulation import set_global_seed, simulate_ergodic_dataset


class BaseTrainer(ABC):
    """Shared infrastructure for objective-specific trainers."""

    objective_name = "base"

    def __init__(
        self,
        *,
        mp: ModelParams,
        tp: TrainParams,
        policy: PolicyNet,
        jsonl_logger: Optional[JSONLLogger] = None,
        ckptio: Optional[TFCheckpointIO] = None,
        value: Optional[ValueNet] = None,
        multiplier: Optional[MultiplierNet] = None,
        op2: Optional[Obj2Params] = None,
        op3: Optional[Obj3Params] = None,
    ) -> None:
        """Store shared trainer dependencies and initialize training buffers.

        All objective-specific trainers reuse the same model primitives,
        checkpoint/logger hooks, and cached ergodic datasets maintained here.
        """
        self.mp = mp
        self.tp = tp
        self.policy = policy
        self.value = value
        self.multiplier = multiplier
        self.op2 = op2
        self.op3 = op3
        self.jsonl_logger = jsonl_logger
        self.ckptio = ckptio
        self.k_buf: np.ndarray | None = None
        self.z_buf: np.ndarray | None = None
        self.history: Dict[str, List[float]] = {"epoch": []}

    def initialize(self, seed_offset: int = 0) -> None:
        """Initialize RNG state and the first ergodic buffer."""
        set_global_seed(self.tp.seed + seed_offset)
        self.refresh_ergodic_data(seed=self.tp.seed + 10 + seed_offset)

    def refresh_ergodic_data(self, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        """Regenerate the policy-induced ergodic state buffer."""
        self.k_buf, self.z_buf = simulate_ergodic_dataset(self.policy, self.mp, self.tp, seed=seed)
        return self.k_buf, self.z_buf

    def sample_train_batch(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """Sample a training minibatch from the ergodic buffer."""
        if self.k_buf is None or self.z_buf is None:
            raise RuntimeError("Ergodic buffer has not been initialized.")
        idx = np.random.choice(len(self.k_buf), size=self.tp.batch_size, replace=True)
        k = tf.convert_to_tensor(self.k_buf[idx], dtype=tf.float32)
        z = tf.convert_to_tensor(self.z_buf[idx], dtype=tf.float32)
        return k, z

    def sample_test_states(self) -> Tuple[np.ndarray, np.ndarray]:
        """Sample test states from the current ergodic buffer."""
        if self.k_buf is None or self.z_buf is None:
            raise RuntimeError("Ergodic buffer has not been initialized.")
        idx = np.random.choice(len(self.k_buf), size=self.tp.N_test_states, replace=True)
        return self.k_buf[idx], self.z_buf[idx]

    @staticmethod
    def clip_and_apply(
        optimizer: tf.keras.optimizers.Optimizer,
        gradients,
        variables,
        clip: float,
    ) -> None:
        """Clip gradients by global norm and apply an optimizer step."""
        gradients, _ = tf.clip_by_global_norm(gradients, clip)
        optimizer.apply_gradients(zip(gradients, variables))

    def maybe_refresh_epoch_buffer(self, epoch: int, seed_base: int) -> None:
        """Refresh ergodic states according to the configured schedule."""
        if epoch == 1 or (epoch % self.tp.ergodic_refresh_every == 0):
            self.refresh_ergodic_data(seed=seed_base + epoch)

    def log_epoch(self, record: Dict[str, float]) -> None:
        """Persist metrics to JSONL if a logger is attached."""
        if self.jsonl_logger is not None:
            payload = {"objective": self.objective_name, **record}
            self.jsonl_logger.log(payload)

    def save_checkpoint(self, epoch: int) -> None:
        """Save a TensorFlow checkpoint if checkpointing is enabled."""
        if self.ckptio is not None:
            self.ckptio.save(step=epoch)

    @abstractmethod
    def train(self):
        """Run the training loop and return trained objects and history."""

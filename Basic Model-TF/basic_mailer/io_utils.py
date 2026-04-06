"""Logging and checkpoint utilities."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import tensorflow as tf


def ensure_dir(path: str) -> None:
    """Create ``path`` if it does not already exist."""
    if path:
        os.makedirs(path, exist_ok=True)


@dataclass
class JSONLLogger:
    """Append-only JSON Lines logger.

    Each call to :meth:`log` writes a single JSON object to disk.
    """

    path: str

    def __post_init__(self) -> None:
        """Create the parent directory for the JSONL log file if necessary."""
        ensure_dir(os.path.dirname(self.path))

    def log(self, record: Dict[str, Any]) -> None:
        """Append ``record`` as one JSON line."""

        def _to_py(value: Any) -> Any:
            if hasattr(value, "numpy"):
                value = value.numpy()
            try:
                import numpy as np

                if isinstance(value, np.generic):
                    return value.item()
            except Exception:
                pass
            return value

        clean = {key: _to_py(value) for key, value in record.items()}
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(clean, ensure_ascii=False) + "\n")
            handle.flush()


@dataclass
class TFCheckpointIO:
    """Save and restore TensorFlow models and optimizers.

    Args:
        directory: Directory used to store checkpoint files.
        policy: Policy model to track.
        opt_policy: Optimizer associated with the policy model.
        value: Optional value model for Objective 3.
        opt_value: Optional optimizer associated with the value model.
        max_to_keep: Number of recent checkpoints to retain.
    """

    directory: str
    policy: tf.Module
    opt_policy: tf.keras.optimizers.Optimizer
    value: Optional[tf.Module] = None
    opt_value: Optional[tf.keras.optimizers.Optimizer] = None
    max_to_keep: int = 3

    def __post_init__(self) -> None:
        """Instantiate the TensorFlow checkpoint and checkpoint manager objects."""
        ensure_dir(self.directory)
        tracked_objects = {"policy": self.policy, "opt_policy": self.opt_policy}
        if self.value is not None:
            tracked_objects["value"] = self.value
        if self.opt_value is not None:
            tracked_objects["opt_value"] = self.opt_value
        self._ckpt = tf.train.Checkpoint(**tracked_objects)
        self._manager = tf.train.CheckpointManager(
            self._ckpt,
            directory=self.directory,
            max_to_keep=self.max_to_keep,
        )

    @property
    def latest_checkpoint(self) -> Optional[str]:
        """Return the latest checkpoint path, if one exists."""
        return self._manager.latest_checkpoint

    def restore_latest(self) -> bool:
        """Restore the most recent checkpoint if available."""
        path = self._manager.latest_checkpoint
        if path is None:
            return False
        self._ckpt.restore(path).expect_partial()
        return True

    def save(self, step: int) -> str:
        """Write a new checkpoint and return its path."""
        return self._manager.save(checkpoint_number=step)

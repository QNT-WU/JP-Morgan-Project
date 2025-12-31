# src/basic_mailer/io_utils.py
from __future__ import annotations


import json  # used to write logs as JSON text.
import os  # used to create directories.
from dataclasses import dataclass  # used to define small “utility objects” cleanly.
from typing import Any, Dict, Optional

import tensorflow as tf


# This guarantees a directory exists.
# if it already exists → do nothing (no error)
# if it doesn’t exist → create it (including parents)
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@dataclass
class JSONLLogger:
    """
    Append-only JSON Lines logger.
    Each call to log(...) writes one JSON object per line.
    """

    path: str
    # This defines a class with one field: path.

    def __post_init__(self) -> None:
        ensure_dir(os.path.dirname(self.path))
        # gets the directory part of the file path:
        # calls ensure_dir(...) to create it if missing

    def log(self, record: Dict[str, Any]) -> None:
        # Make sure everything is JSON serializable (convert numpy/tf scalars to Python)
        def _to_py(x: Any) -> Any:
            if hasattr(x, "numpy"):
                x = x.numpy()
            # numpy scalars
            try:
                import numpy as np

                if isinstance(x, (np.generic,)):
                    return x.item()
            except Exception:
                pass
            return x

        clean = {k: _to_py(v) for k, v in record.items()}
        # Applies the conversion to every value in the dictionary
        # Ensures clean can be JSON serialized
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")
            f.flush()


@dataclass
class TFCheckpointIO:
    """
    TF checkpoint save/restore for:
      - policy
      - (optional) value
      - optimizers

    Usage:
      ckptio = TFCheckpointIO(dir="outputs/run1/ckpt/obj1", policy=policy, opt_policy=opt)
      ckptio.restore_latest()  # if exists
      ckptio.save(step=epoch)  # saves checkpoint
    """

    directory: str  # where checkpoints are stored, e.g."outputs/run1/checkpoints/obj1"
    policy: tf.Module  # the policy model
    opt_policy: tf.keras.optimizers.Optimizer  # the optimizer used to train policy
    value: Optional[tf.Module] = None  # optional value model (Objective 3 only)
    # optional optimizer for value model
    opt_value: Optional[tf.keras.optimizers.Optimizer] = None
    # keep only the most recent 3 checkpoints to save disk
    max_to_keep: int = 3

    def __post_init__(self) -> None:
        ensure_dir(self.directory)
        # Creates the checkpoint folder.
        objs = {"policy": self.policy, "opt_policy": self.opt_policy}

        if self.value is not None:
            objs["value"] = self.value
        if self.opt_value is not None:
            objs["opt_value"] = self.opt_value
        # Build a TensorFlow Checkpoint object

        self._ckpt = tf.train.Checkpoint(**objs)
        # Build a checkpoint manager
        # naming checkpoints (ckpt-1, ckpt-2, …)
        # tracking latest checkpoint
        # deleting older ones automatically
        self._manager = tf.train.CheckpointManager(
            self._ckpt, directory=self.directory, max_to_keep=self.max_to_keep
        )

    @property
    # Returns the filepath of the most recent checkpoint,or None if none exist.
    def latest_checkpoint(self) -> Optional[str]:
        return self._manager.latest_checkpoint

    # If no checkpoint exists → return False
    # policy weights, value weights (if included), optimizer internal state (e.g., Adam’s momentum)
    def restore_latest(self) -> bool:
        path = self._manager.latest_checkpoint
        if path is None:
            return False
        self._ckpt.restore(path).expect_partial()
        return True

    # Returns the saved path.
    def save(self, step: int) -> str:
        # step is typically epoch number
        return self._manager.save(checkpoint_number=step)

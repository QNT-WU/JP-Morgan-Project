from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import tensorflow as tf


# Makes a directory if it doesn’t exist.
# exist_ok=True means “don’t crash if it already exists”
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@dataclass
class JSONLLogger:
    """
    Append-only JSON Lines logger.
    Each call writes one JSON object per line.
    """

    path: str

    def __post_init__(self) -> None:
        d = os.path.dirname(self.path)
        if d:
            ensure_dir(d)

    def log(self, record: Dict[str, Any]) -> None:
        def _to_py(x: Any) -> Any:
            if hasattr(x, "numpy"):
                x = x.numpy()
            try:
                import numpy as np

                if isinstance(x, (np.generic,)):
                    return x.item()
            except Exception:
                pass
            return x

        # Clean dict and write to file
        # It converts everything into JSON-safe values.
        clean = {k: _to_py(v) for k, v in record.items()}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")
            f.flush()


# This is for saving and restoring model weights + optimizer state.
@dataclass
class TFCheckpointIO:
    """
    TF checkpoint save/restore for modules + optimizers.
    """

    directory: str
    policy: tf.Module
    opt_policy: tf.keras.optimizers.Optimizer

    value: Optional[tf.Module] = None
    opt_value: Optional[tf.keras.optimizers.Optimizer] = None

    vtilde: Optional[tf.Module] = None
    opt_vtilde: Optional[tf.keras.optimizers.Optimizer] = None

    qnet: Optional[tf.Module] = None
    opt_qnet: Optional[tf.keras.optimizers.Optimizer] = None

    max_to_keep: int = 3

    def __post_init__(self) -> None:
        ensure_dir(self.directory)
        objs = {
            "policy": self.policy,
            "opt_policy": self.opt_policy,
        }
        if self.value is not None:
            objs["value"] = self.value
        if self.opt_value is not None:
            objs["opt_value"] = self.opt_value
        if self.vtilde is not None:
            objs["vtilde"] = self.vtilde
        if self.opt_vtilde is not None:
            objs["opt_vtilde"] = self.opt_vtilde
        if self.qnet is not None:
            objs["qnet"] = self.qnet
        if self.opt_qnet is not None:
            objs["opt_qnet"] = self.opt_qnet

        self._ckpt = tf.train.Checkpoint(**objs)
        self._manager = tf.train.CheckpointManager(
            self._ckpt, directory=self.directory, max_to_keep=self.max_to_keep
        )

    @property
    def latest_checkpoint(self) -> Optional[str]:
        return self._manager.latest_checkpoint

    def restore_latest(self) -> bool:
        path = self._manager.latest_checkpoint
        if path is None:
            return False
        self._ckpt.restore(path).expect_partial()
        return True

    def save(self, step: int) -> str:
        return self._manager.save(checkpoint_number=step)

"""I/O helpers for logging and TensorFlow checkpoint management."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import tensorflow as tf


# Makes a directory if it doesn’t exist.
# exist_ok=True means “don’t crash if it already exists”
def ensure_dir(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


@dataclass
class JSONLLogger:
    """
    Append-only JSON Lines logger.
    Each call writes one JSON object per line.
    """

    path: str
    _fallback_path: Optional[str] = field(default=None, init=False, repr=False)
    _warned_fallback: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """Finalize initialization for JSONLLogger."""
        d = os.path.dirname(self.path)
        if d:
            ensure_dir(d)

    def _fallback_log_path(self) -> str:
        """Return a stable local fallback path for Colab/Drive write failures."""
        if self._fallback_path is None:
            digest = hashlib.md5(self.path.encode("utf-8")).hexdigest()[:12]
            directory = os.path.join(tempfile.gettempdir(), "risky_model_tf_logs")
            ensure_dir(directory)
            base = os.path.basename(self.path) or "events.jsonl"
            self._fallback_path = os.path.join(directory, f"{digest}_{base}")
        return self._fallback_path

    def _append_jsonl(self, path: str, clean: Dict[str, Any]) -> None:
        """Append sonl."""
        directory = os.path.dirname(path)
        if directory:
            ensure_dir(directory)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")
            f.flush()

    def log(self, record: Dict[str, Any]) -> None:
        """Append one JSON-serializable record to the log file."""
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

        clean = {k: _to_py(v) for k, v in record.items()}
        try:
            self._append_jsonl(self.path, clean)
            return
        except OSError as exc:
            if exc.errno not in {5, 107}:
                raise
            fallback = self._fallback_log_path()
            self._append_jsonl(fallback, clean)
            if not self._warned_fallback:
                print(
                    f"[WARN] JSONLLogger could not write to {self.path!r}; "
                    f"continuing with local fallback log {fallback!r}."
                )
                self._warned_fallback = True


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
        """Finalize initialization for TFCheckpointIO."""
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
        """Return the latest checkpoint path managed by this wrapper."""
        return self._manager.latest_checkpoint

    def restore_latest(self) -> bool:
        """Restore the latest available checkpoint if one exists."""
        path = self._manager.latest_checkpoint
        if path is None:
            return False
        self._ckpt.restore(path).expect_partial()
        return True

    def save(self, step: int) -> str:
        """Save one checkpoint and return the resulting path."""
        return self._manager.save(checkpoint_number=step)

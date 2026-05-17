"""Object-oriented training engines for the risky-debt neural objectives.

This module provides class-based trainers with TensorFlow-native ``train_step``
methods. The design keeps orchestration code outside the mathematical
objective definitions while still supporting the legacy entrypoints used by the
existing experiment scripts.
"""

from __future__ import annotations

import inspect
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from .config import ModelParams, NetParams, Obj1Params, Obj2Params, Obj3Params, TrainParams
from .evaluation import eval_test_euler_mse_obj1, eval_test_euler_mse_obj3, eval_test_reward
from .io_utils import JSONLLogger, TFCheckpointIO
from .networks import PolicyNet, PricingNet, ValueNet, VtildeNet, MultiplierNet
from .objectives import obj1_loss, obj2_batch_metrics, obj3_batch_metrics
from .simulation import set_global_seed, simulate_ergodic_dataset

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")


@dataclass
class ObjectiveTrainingArtifacts:
    """Container bundling trained networks and diagnostics for one objective."""

    name: str
    policy: PolicyNet
    history: Dict[str, List[float]]
    qnet: Optional[PricingNet] = None
    value: Optional[ValueNet] = None
    vtilde: Optional[VtildeNet] = None
    lambda_k: Optional[MultiplierNet] = None
    ergodic_seed_offset: int = 0


class ScalarMetric:
    """Tiny TensorFlow metric accumulator backed by explicit ``tf.Variable`` state."""

    def __init__(self, name: str) -> None:
        """Initialize ScalarMetric."""
        self.name = name
        self.total = tf.Variable(0.0, trainable=False, dtype=tf.float32, name=f"{name}_total")
        self.count = tf.Variable(0.0, trainable=False, dtype=tf.float32, name=f"{name}_count")

    def reset(self) -> None:
        """Reset the accumulated sum and count."""
        self.total.assign(0.0)
        self.count.assign(0.0)

    def update(self, value: tf.Tensor) -> None:
        """Accumulate one scalar observation."""
        self.total.assign_add(tf.cast(value, tf.float32))
        self.count.assign_add(1.0)

    def result(self) -> tf.Tensor:
        """Return the running mean."""
        return tf.math.divide_no_nan(self.total, self.count)


class BestWeightsTracker:
    """Track and restore the best network weights according to a scalar metric."""

    def __init__(self) -> None:
        """Initialize BestWeightsTracker."""
        self.best_score = -float("inf")
        self.best_epoch = 0
        self.weights: Optional[Dict[str, List[np.ndarray]]] = None

    def maybe_update(self, epoch: int, score: float, *, policy, value=None, qnet=None, vtilde=None, lambda_k=None) -> None:
        """Save network weights when the supplied score improves."""
        if not np.isfinite(score):
            return
        if self.weights is None or score > self.best_score:
            self.best_score = float(score)
            self.best_epoch = int(epoch)
            payload = {"policy": policy.get_weights()}
            if value is not None:
                payload["value"] = value.get_weights()
            if qnet is not None:
                payload["qnet"] = qnet.get_weights()
            if vtilde is not None:
                payload["vtilde"] = vtilde.get_weights()
            if lambda_k is not None:
                payload["lambda_k"] = lambda_k.get_weights()
            self.weights = payload

    def restore(self, *, policy, value=None, qnet=None, vtilde=None, lambda_k=None) -> None:
        """Restore the best weights recorded so far."""
        if self.weights is None:
            return
        policy.set_weights(self.weights["policy"])
        if value is not None and "value" in self.weights:
            value.set_weights(self.weights["value"])
        if qnet is not None and "qnet" in self.weights:
            qnet.set_weights(self.weights["qnet"])
        if vtilde is not None and "vtilde" in self.weights:
            vtilde.set_weights(self.weights["vtilde"])
        if lambda_k is not None and "lambda_k" in self.weights:
            lambda_k.set_weights(self.weights["lambda_k"])


def _clip_and_apply(optimizer: tf.keras.optimizers.Optimizer, gradients, variables, clip: float) -> None:
    """Clip gradients by global norm and apply them when at least one is present."""
    if not variables:
        return
    grad_var_pairs = [(g, v) for g, v in zip(gradients, variables) if g is not None]
    if not grad_var_pairs:
        return
    grads, vars_ = zip(*grad_var_pairs)
    clipped, _ = tf.clip_by_global_norm(list(grads), clip)
    optimizer.apply_gradients(zip(clipped, vars_))


def _call_objective(fn, **kwargs):
    """Call an objective using only keyword arguments present in its signature."""
    params = inspect.signature(fn).parameters
    valid_kwargs = {key: value for key, value in kwargs.items() if key in params}
    return fn(**valid_kwargs)


def _default_state_normalization(tp: TrainParams) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed state normalization based on the configured initial support."""
    center = np.array(
        [
            0.5 * (tp.k0_low + tp.k0_high),
            0.5 * (tp.b0_low + tp.b0_high),
            0.5 * (tp.z0_low + tp.z0_high),
        ],
        dtype=np.float32,
    )
    scale = np.array(
        [
            max((tp.k0_high - tp.k0_low) / np.sqrt(12.0), 1e-3),
            max((tp.b0_high - tp.b0_low) / np.sqrt(12.0), 1e-3),
            max((tp.z0_high - tp.z0_low) / np.sqrt(12.0), 1e-3),
        ],
        dtype=np.float32,
    )
    return center, scale


def _pricing_input_normalization(tp: TrainParams) -> tuple[np.ndarray, np.ndarray]:
    """Return normalization constants for pricing inputs ``[z, k', b']``."""
    center, scale = _default_state_normalization(tp)
    return center[[2, 0, 1]], scale[[2, 0, 1]]


class BaseObjectiveTrainer:
    """Base class shared by the three objective-specific trainers.

    The class manages reproducibility, ergodic buffers, TensorFlow datasets,
    logging, checkpoint hooks, and epoch-level bookkeeping. Subclasses only
    provide objective-specific networks, optimizers, and compiled train steps.
    """

    objective_name: str = "base"
    seed_offset: int = 0
    uses_dataset_batches: bool = False
    ergodic_record_mode: str = "all"

    def __init__(
        self,
        *,
        mp: ModelParams,
        tp: TrainParams,
        objective_params,
        jsonl_logger: Optional[JSONLLogger] = None,
        ckptio: Optional[TFCheckpointIO] = None,
        epoch_checkpoint_dir: Optional[str] = None,
        resume_epoch_checkpoint: bool = True,
    ) -> None:
        """Initialize BaseObjectiveTrainer."""
        self.mp = mp
        self.tp = tp
        self.objective_params = objective_params
        self.jsonl_logger = jsonl_logger
        self.ckptio = ckptio
        self.epoch_checkpoint_dir = epoch_checkpoint_dir
        self.resume_epoch_checkpoint = bool(resume_epoch_checkpoint)
        self.numpy_rng = np.random.default_rng(tp.seed + 10_000 + self.seed_offset)
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64, name=f"{self.objective_name}_global_step")
        self.current_epoch = tf.Variable(0, trainable=False, dtype=tf.int64, name=f"{self.objective_name}_epoch")
        self.history = self._initialize_history()
        self.k_buf = np.empty((0,), dtype=np.float32)
        self.b_buf = np.empty((0,), dtype=np.float32)
        self.z_buf = np.empty((0,), dtype=np.float32)
        self._train_iterator: Optional[Iterator[tuple[tf.Tensor, tf.Tensor, tf.Tensor]]] = None
        self._build_components()

    @property
    def ergodic_seed_offset(self) -> int:
        """Seed offset used when plotting the post-training ergodic set."""
        return self.seed_offset

    def _initialize_history(self) -> Dict[str, List[float]]:
        """Return an empty history dictionary for the objective."""
        raise NotImplementedError

    def _build_components(self) -> None:
        """Instantiate networks, optimizers, and metric state."""
        raise NotImplementedError

    def _refresh_ergodic_buffer(self, *, seed: int) -> None:
        """Simulate and cache an updated ergodic dataset under the current policy."""
        t0 = time.perf_counter()
        requested = (int(self.tp.ergodic_burn_in) + int(self.tp.ergodic_T)) * int(self.tp.ergodic_n_paths)
        print(
            f"[{self.objective_name}] refreshing ergodic buffer "
            f"(burn={self.tp.ergodic_burn_in}, T={self.tp.ergodic_T}, "
            f"paths={self.tp.ergodic_n_paths}, approx_steps={requested})...",
            flush=True,
        )
        mode = getattr(self, "ergodic_record_mode", "all")
        self.k_buf, self.b_buf, self.z_buf = simulate_ergodic_dataset(
            self.policy, self.mp, self.tp, seed=seed, record_mode=mode
        )
        min_required = max(16, min(int(self.tp.batch_size), int(self.tp.ergodic_buffer_size)))
        if mode == "continuation" and len(self.k_buf) < min_required:
            print(
                f"[{self.objective_name}] continuation buffer too small "
                f"(n={len(self.k_buf)}); falling back to unconditional states for this refresh.",
                flush=True,
            )
            self.k_buf, self.b_buf, self.z_buf = simulate_ergodic_dataset(
                self.policy, self.mp, self.tp, seed=seed, record_mode="all"
            )
        if self.uses_dataset_batches:
            self._rebuild_dataset_iterator()
        print(
            f"[{self.objective_name}] ergodic buffer ready "
            f"(record_mode={mode}, n={len(self.k_buf)}, elapsed={time.perf_counter() - t0:.1f}s).",
            flush=True,
        )

    def _rebuild_dataset_iterator(self) -> None:
        """Create a TensorFlow dataset backed by the current ergodic buffer."""
        dataset = tf.data.Dataset.from_tensor_slices((self.k_buf, self.b_buf, self.z_buf))
        dataset = dataset.shuffle(
            buffer_size=max(len(self.k_buf), self.tp.batch_size),
            seed=self.tp.seed + self.seed_offset,
            reshuffle_each_iteration=True,
        )
        dataset = dataset.repeat().batch(self.tp.batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
        self._train_iterator = iter(dataset)

    def _next_batch(self) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Return the next batch from the TensorFlow dataset iterator."""
        if self._train_iterator is None:
            raise RuntimeError("Training dataset iterator is not available.")
        return next(self._train_iterator)

    def _sample_test_states(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw test states from the current ergodic buffer."""
        replace = len(self.k_buf) < self.tp.N_test_states
        indices = self.numpy_rng.choice(len(self.k_buf), size=self.tp.N_test_states, replace=replace)
        return self.k_buf[indices], self.b_buf[indices], self.z_buf[indices]

    def _log_epoch(self, payload: Dict[str, float]) -> None:
        """Write one epoch payload to the JSONL logger when enabled."""
        if self.jsonl_logger is not None:
            self.jsonl_logger.log(payload)

    def _save_checkpoint(self, epoch: int) -> None:
        """Save lightweight epoch state so interrupted Colab runs can resume.

        The public artifact writer still saves final report weights after each
        objective.  This method saves *latest* per-epoch weights and the partial
        history during training, which makes full-scale runs robust to Colab
        disconnects or notebook interrupts.
        """
        if self.ckptio is not None:
            self.ckptio.save(step=epoch)
        self._save_epoch_state(epoch)

    def _weight_file_map(self) -> Dict[str, str]:
        """Return component weight file paths for resumable epoch checkpoints."""
        if self.epoch_checkpoint_dir is None:
            return {}
        return {
            "policy": os.path.join(self.epoch_checkpoint_dir, "policy.weights.h5"),
            "value": os.path.join(self.epoch_checkpoint_dir, "value.weights.h5"),
            "vtilde": os.path.join(self.epoch_checkpoint_dir, "vtilde.weights.h5"),
            "qnet": os.path.join(self.epoch_checkpoint_dir, "qnet.weights.h5"),
            "lambda_k": os.path.join(self.epoch_checkpoint_dir, "lambda_k.weights.h5"),
        }

    def _save_epoch_state(self, epoch: int) -> None:
        """Persist latest network weights and partial history for resume."""
        if self.epoch_checkpoint_dir is None:
            return
        os.makedirs(self.epoch_checkpoint_dir, exist_ok=True)
        files = self._weight_file_map()
        components = {
            "policy": getattr(self, "policy", None),
            "value": getattr(self, "value", None),
            "vtilde": getattr(self, "vtilde", None),
            "qnet": getattr(self, "qnet", None),
            "lambda_k": getattr(self, "lambda_k", None),
        }
        saved = {}
        for name, model in components.items():
            if model is None or name not in files:
                continue
            model.save_weights(files[name])
            saved[name] = os.path.basename(files[name])
        hist_path = os.path.join(self.epoch_checkpoint_dir, "history_latest.npz")
        np.savez_compressed(hist_path, **{k: np.asarray(v) for k, v in self.history.items()})
        manifest = {
            "objective": self.objective_name,
            "epoch": int(epoch),
            "weights": saved,
            "history": os.path.basename(hist_path),
        }
        with open(os.path.join(self.epoch_checkpoint_dir, "LATEST.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def _load_epoch_state(self) -> int:
        """Load latest per-epoch checkpoint and return the next epoch index."""
        if not self.resume_epoch_checkpoint or self.epoch_checkpoint_dir is None:
            return 1
        manifest_path = os.path.join(self.epoch_checkpoint_dir, "LATEST.json")
        if not os.path.exists(manifest_path):
            return 1
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            epoch = int(manifest.get("epoch", 0))
            if epoch <= 0:
                return 1
            files = self._weight_file_map()
            components = {
                "policy": getattr(self, "policy", None),
                "value": getattr(self, "value", None),
                "vtilde": getattr(self, "vtilde", None),
                "qnet": getattr(self, "qnet", None),
                "lambda_k": getattr(self, "lambda_k", None),
            }
            for name, model in components.items():
                if model is not None and name in files and os.path.exists(files[name]):
                    model.load_weights(files[name])
            hist_path = os.path.join(self.epoch_checkpoint_dir, "history_latest.npz")
            if os.path.exists(hist_path):
                loaded = np.load(hist_path, allow_pickle=True)
                for key in self.history:
                    if key in loaded:
                        self.history[key] = loaded[key].tolist()
            next_epoch = epoch + 1
            print(
                f"[{self.objective_name}] resumed from epoch {epoch}; "
                f"next epoch is {next_epoch}.",
                flush=True,
            )
            return next_epoch
        except Exception as exc:
            print(
                f"[{self.objective_name}] warning: failed to load resume checkpoint "
                f"from {self.epoch_checkpoint_dir!r}: {exc}. Starting from scratch.",
                flush=True,
            )
            return 1

    def _should_print(self, epoch: int) -> bool:
        """Return whether the trainer should print the current epoch summary."""
        return epoch == 1 or epoch % max(1, self.tp.epochs // 10) == 0

    def _evaluate_epoch(self, epoch: int) -> Dict[str, float]:
        """Compute test metrics and return the epoch summary payload."""
        raise NotImplementedError

    def _append_history(self, metrics: Dict[str, float]) -> None:
        """Append one epoch payload to the in-memory history."""
        raise NotImplementedError

    def _print_epoch_summary(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Print a concise epoch summary."""
        raise NotImplementedError

    def _post_training_hook(self) -> None:
        """Optional hook executed after the main training loop."""

    def _assemble_artifacts(self) -> ObjectiveTrainingArtifacts:
        """Package the trained components into ``ObjectiveTrainingArtifacts``."""
        raise NotImplementedError

    def train(self) -> ObjectiveTrainingArtifacts:
        """Run the full training loop and return train/evaluation artifacts.

        If a latest epoch checkpoint exists, training resumes from the next
        unfinished epoch.  This makes full-scale runs restartable with the same
        command after a Colab interrupt.
        """
        set_global_seed(self.tp.seed + self.seed_offset)
        start_epoch = self._load_epoch_state()
        if start_epoch > self.tp.epochs:
            print(f"[{self.objective_name}] already completed {self.tp.epochs} epochs; skipping training.", flush=True)
            self._post_training_hook()
            return self._assemble_artifacts()
        self._refresh_ergodic_buffer(seed=self.tp.seed + 10 + self.seed_offset + start_epoch)
        for epoch in range(start_epoch, self.tp.epochs + 1):
            self.current_epoch.assign(epoch)
            self._run_epoch(epoch)
            metrics = self._evaluate_epoch(epoch)
            self._append_history(metrics)
            self._log_epoch(metrics)
            self._save_checkpoint(epoch)
            if self._should_print(epoch):
                self._print_epoch_summary(epoch, metrics)
        self._post_training_hook()
        return self._assemble_artifacts()

    def _run_epoch(self, epoch: int) -> None:
        """Run one full training epoch and return the mean training loss."""
        raise NotImplementedError


class Objective1Trainer(BaseObjectiveTrainer):
    """Trainer for Objective 1: lifetime reward maximization with pricing discipline."""

    objective_name = "obj1"
    seed_offset = 0
    uses_dataset_batches = False

    def __init__(
        self,
        *,
        mp: ModelParams,
        npol: NetParams,
        nq: NetParams,
        tp: TrainParams,
        objective_params: Obj1Params,
        jsonl_logger: Optional[JSONLLogger] = None,
        ckptio: Optional[TFCheckpointIO] = None,
        epoch_checkpoint_dir: Optional[str] = None,
        resume_epoch_checkpoint: bool = True,
    ) -> None:
        """Initialize Objective1Trainer."""
        self.npol = npol
        self.nq = nq
        super().__init__(
            mp=mp,
            tp=tp,
            objective_params=objective_params,
            jsonl_logger=jsonl_logger,
            ckptio=ckptio,
            epoch_checkpoint_dir=epoch_checkpoint_dir,
            resume_epoch_checkpoint=resume_epoch_checkpoint,
        )

    def _initialize_history(self) -> Dict[str, List[float]]:
        """Initialize istory."""
        return {"epoch": [], "train_reward": [], "train_critic_loss": [], "test_reward": [], "test_euler_mse": []}

    def _build_components(self) -> None:
        """Build omponents."""
        state_center, state_scale = _default_state_normalization(self.tp)
        price_center, price_scale = _pricing_input_normalization(self.tp)
        self.policy = PolicyNet(self.npol, self.mp.k_min, self.mp.b_min, self.mp.b_max, input_center=state_center, input_scale=state_scale)
        self.vtilde = VtildeNet(self.npol, input_center=state_center, input_scale=state_scale)
        self.qnet = PricingNet(self.nq, self.mp.q_min, self.mp.q_max, input_center=price_center, input_scale=price_scale)
        _ = self.policy(tf.zeros((1, 3), dtype=tf.float32))
        _ = self.vtilde(tf.zeros((1, 3), dtype=tf.float32))
        _ = self.qnet(tf.zeros((1, 3), dtype=tf.float32))
        self.opt_policy = tf.keras.optimizers.Adam(self.tp.lr_policy)
        self.opt_vtilde = tf.keras.optimizers.Adam(self.tp.lr_vtilde)
        self.opt_q = tf.keras.optimizers.Adam(self.tp.lr_q)
        self.train_reward_metric = ScalarMetric("obj1_train_reward")
        self.critic_metric = ScalarMetric("obj1_critic_loss")

    @tf.function(reduce_retracing=True)
    def _train_step(self) -> tf.Tensor:
        """Run one compiled optimization step for Objective 1."""
        with tf.GradientTape(persistent=True) as tape:
            out = _call_objective(obj1_loss, policy=self.policy, qnet=self.qnet, mp=self.mp, tp=self.tp, op1=self.objective_params, critic=self.vtilde)
            if isinstance(out, (tuple, list)):
                loss = tf.convert_to_tensor(out[0], dtype=tf.float32)
                train_reward = tf.convert_to_tensor(out[1], dtype=tf.float32)
                critic_loss = tf.convert_to_tensor(out[3], dtype=tf.float32)
            else:
                loss = tf.convert_to_tensor(out, dtype=tf.float32)
                train_reward = -loss
                critic_loss = tf.constant(0.0, dtype=tf.float32)
        grads_p = tape.gradient(loss, self.policy.trainable_variables)
        grads_vt = tape.gradient(loss, self.vtilde.trainable_variables)
        grads_q = tape.gradient(loss, self.qnet.trainable_variables)
        del tape
        _clip_and_apply(self.opt_policy, grads_p, self.policy.trainable_variables, self.tp.grad_clip)
        _clip_and_apply(self.opt_vtilde, grads_vt, self.vtilde.trainable_variables, self.tp.grad_clip)
        _clip_and_apply(self.opt_q, grads_q, self.qnet.trainable_variables, self.tp.grad_clip)
        self.global_step.assign_add(1)
        return train_reward, critic_loss

    def _run_epoch(self, epoch: int) -> None:
        """Run one training epoch and return the mean in-sample objective."""
        self.train_reward_metric.reset()
        self.critic_metric.reset()
        for _ in range(self.tp.steps_per_epoch):
            train_reward, critic_loss = self._train_step()
            self.train_reward_metric.update(train_reward)
            self.critic_metric.update(critic_loss)
        if epoch != 1 and epoch % self.tp.ergodic_refresh_every == 0:
            self._refresh_ergodic_buffer(seed=self.tp.seed + 100 + epoch)

    def _evaluate_epoch(self, epoch: int) -> Dict[str, float]:
        """Evaluate poch."""
        k_test, b_test, z_test = self._sample_test_states()
        return {
            "objective": self.objective_name,
            "epoch": epoch,
            "train_reward": float(self.train_reward_metric.result().numpy()),
            "train_critic_loss": float(self.critic_metric.result().numpy()),
            "test_reward": float(eval_test_reward(self.policy, self.qnet, self.mp, self.tp, seed=self.tp.seed + 200 + epoch, vtilde=self.vtilde)),
            "test_euler_mse": float(
                eval_test_euler_mse_obj1(
                    policy=self.policy,
                    qnet=self.qnet,
                    mp=self.mp,
                    tp=self.tp,
                    states_k=k_test,
                    states_b=b_test,
                    states_z=z_test,
                    seed=self.tp.seed + 300 + epoch,
                    vtilde=self.vtilde,
                )
            ),
        }

    def _append_history(self, metrics: Dict[str, float]) -> None:
        """Append istory."""
        self.history["epoch"].append(int(metrics["epoch"]))
        self.history["train_reward"].append(float(metrics["train_reward"]))
        self.history["train_critic_loss"].append(float(metrics["train_critic_loss"]))
        self.history["test_reward"].append(float(metrics["test_reward"]))
        self.history["test_euler_mse"].append(float(metrics["test_euler_mse"]))

    def _print_epoch_summary(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Print poch summary."""
        print(
            f"[Obj1][{epoch:03d}] TrainReward={metrics['train_reward']:.4f} "
            f"CriticLoss={metrics['train_critic_loss']:.6f} "
            f"TestReward={metrics['test_reward']:.4f} "
            f"TestEulerMSE={metrics['test_euler_mse']:.6f}"
        )

    def _assemble_artifacts(self) -> ObjectiveTrainingArtifacts:
        """Assemble rtifacts."""
        return ObjectiveTrainingArtifacts(
            name=self.objective_name,
            policy=self.policy,
            qnet=self.qnet,
            vtilde=self.vtilde,
            history=self.history,
            ergodic_seed_offset=self.ergodic_seed_offset,
        )


class Objective2Trainer(BaseObjectiveTrainer):
    """Trainer for Objective 2: Euler-residual minimization."""

    objective_name = "obj2"
    seed_offset = 1
    uses_dataset_batches = True
    ergodic_record_mode = "continuation"

    def __init__(
        self,
        *,
        mp: ModelParams,
        npol: NetParams,
        nval: NetParams,
        nvt: NetParams,
        nq: NetParams,
        tp: TrainParams,
        objective_params: Obj2Params,
        jsonl_logger: Optional[JSONLLogger] = None,
        ckptio: Optional[TFCheckpointIO] = None,
        epoch_checkpoint_dir: Optional[str] = None,
        resume_epoch_checkpoint: bool = True,
    ) -> None:
        """Initialize Objective2Trainer."""
        self.npol = npol
        self.nval = nval
        self.nvt = nvt
        self.nq = nq
        self.best_tracker = BestWeightsTracker()
        super().__init__(
            mp=mp,
            tp=tp,
            objective_params=objective_params,
            jsonl_logger=jsonl_logger,
            ckptio=ckptio,
            epoch_checkpoint_dir=epoch_checkpoint_dir,
            resume_epoch_checkpoint=resume_epoch_checkpoint,
        )

    def _initialize_history(self) -> Dict[str, List[float]]:
        """Initialize istory."""
        return {
            "epoch": [],
            "train_loss": [],
            "train_objective": [],
            "train_euler": [],
            "train_default_block": [],
            "train_bellman_block": [],
            "train_zp_block": [],
            "test_euler_mse": [],
            "test_reward": [],
        }

    def _build_components(self) -> None:
        """Build omponents."""
        state_center, state_scale = _default_state_normalization(self.tp)
        price_center, price_scale = _pricing_input_normalization(self.tp)
        self.policy = PolicyNet(self.npol, self.mp.k_min, self.mp.b_min, self.mp.b_max, input_center=state_center, input_scale=state_scale)
        self.value = ValueNet(self.nval, input_center=state_center, input_scale=state_scale)
        self.vtilde = VtildeNet(self.nvt, input_center=state_center, input_scale=state_scale)
        self.lambda_k = MultiplierNet(self.nvt, input_center=state_center, input_scale=state_scale)
        self.qnet = PricingNet(self.nq, self.mp.q_min, self.mp.q_max, input_center=price_center, input_scale=price_scale)
        _ = self.policy(tf.zeros((1, 3), dtype=tf.float32))
        _ = self.value(tf.zeros((1, 3), dtype=tf.float32))
        _ = self.vtilde(tf.zeros((1, 3), dtype=tf.float32))
        _ = self.lambda_k(tf.zeros((1, 3), dtype=tf.float32))
        _ = self.qnet(tf.zeros((1, 3), dtype=tf.float32))
        self.opt_policy = tf.keras.optimizers.Adam(self.tp.lr_policy)
        self.opt_value = tf.keras.optimizers.Adam(self.tp.lr_value)
        self.opt_vtilde = tf.keras.optimizers.Adam(self.tp.lr_vtilde)
        self.opt_lambda = tf.keras.optimizers.Adam(self.tp.lr_vtilde)
        self.opt_q = tf.keras.optimizers.Adam(self.tp.lr_q)
        self.loss_metric = ScalarMetric("obj2_loss")
        self.foc_metric = ScalarMetric("obj2_foc")
        self.def_metric = ScalarMetric("obj2_def")
        self.bell_metric = ScalarMetric("obj2_bell")
        self.zp_metric = ScalarMetric("obj2_zp")

    @tf.function(reduce_retracing=True)
    def _train_step(self, k: tf.Tensor, b: tf.Tensor, z: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Run one compiled optimization step for Objective 2."""
        with tf.GradientTape(persistent=True) as tape:
            loss, def_block, bell_block, foc_block, zp_block = _call_objective(
                obj2_batch_metrics,
                policy=self.policy,
                value=self.value,
                vtilde=self.vtilde,
                qnet=self.qnet,
                mp=self.mp,
                tp=self.tp,
                op2=self.objective_params,
                k=k,
                b=b,
                z=z,
                lambda_net=self.lambda_k,
            )
            loss = tf.convert_to_tensor(loss, dtype=tf.float32)
            def_block = tf.convert_to_tensor(def_block, dtype=tf.float32)
            bell_block = tf.convert_to_tensor(bell_block, dtype=tf.float32)
            foc_block = tf.convert_to_tensor(foc_block, dtype=tf.float32)
            zp_block = tf.convert_to_tensor(zp_block, dtype=tf.float32)
        grads_p = tape.gradient(loss, self.policy.trainable_variables)
        grads_v = tape.gradient(loss, self.value.trainable_variables)
        grads_t = tape.gradient(loss, self.vtilde.trainable_variables)
        grads_lam = tape.gradient(loss, self.lambda_k.trainable_variables)
        grads_q = tape.gradient(loss, self.qnet.trainable_variables)
        del tape
        _clip_and_apply(self.opt_policy, grads_p, self.policy.trainable_variables, self.tp.grad_clip)
        _clip_and_apply(self.opt_value, grads_v, self.value.trainable_variables, self.tp.grad_clip)
        _clip_and_apply(self.opt_vtilde, grads_t, self.vtilde.trainable_variables, self.tp.grad_clip)
        _clip_and_apply(self.opt_lambda, grads_lam, self.lambda_k.trainable_variables, self.tp.grad_clip)
        _clip_and_apply(self.opt_q, grads_q, self.qnet.trainable_variables, self.tp.grad_clip)
        self.global_step.assign_add(1)
        return loss, def_block, bell_block, foc_block, zp_block

    def _run_epoch(self, epoch: int) -> None:
        """Run one training epoch and return the mean in-sample objective."""
        for metric in (self.loss_metric, self.foc_metric, self.def_metric, self.bell_metric, self.zp_metric):
            metric.reset()
        if epoch != 1 and epoch % self.tp.ergodic_refresh_every == 0:
            self._refresh_ergodic_buffer(seed=self.tp.seed + 110 + epoch)
        for step in range(self.tp.steps_per_epoch):
            if epoch == 1 and step == 0:
                print(f"[Obj2][epoch 001] compiling/running first train step...", flush=True)
            k, b, z = self._next_batch()
            loss, def_block, bell_block, foc_block, zp_block = self._train_step(k, b, z)
            self.loss_metric.update(loss)
            self.def_metric.update(def_block)
            self.bell_metric.update(bell_block)
            self.foc_metric.update(foc_block)
            self.zp_metric.update(zp_block)
            if epoch == 1 and (step + 1) % max(1, self.tp.steps_per_epoch // 5) == 0:
                print(f"[Obj2][epoch 001] step {step + 1}/{self.tp.steps_per_epoch}", flush=True)

    def _evaluate_epoch(self, epoch: int) -> Dict[str, float]:
        """Evaluate Objective 2 and track the best test-reward checkpoint."""
        k_test, b_test, z_test = self._sample_test_states()
        metrics = {
            "objective": self.objective_name,
            "epoch": epoch,
            "train_loss": float(self.loss_metric.result().numpy()),
            "train_objective": float(self.loss_metric.result().numpy()),
            "train_euler": float(self.foc_metric.result().numpy()),
            "train_default_block": float(self.def_metric.result().numpy()),
            "train_bellman_block": float(self.bell_metric.result().numpy()),
            "train_zp_block": float(self.zp_metric.result().numpy()),
            "test_euler_mse": float(
                eval_test_euler_mse_obj3(
                    policy=self.policy,
                    value=self.value,
                    vtilde=self.vtilde,
                    qnet=self.qnet,
                    mp=self.mp,
                    tp=self.tp,
                    states_k=k_test,
                    states_b=b_test,
                    states_z=z_test,
                    seed=self.tp.seed + 300 + epoch,
                )
            ),
            "test_reward": float(eval_test_reward(self.policy, self.qnet, self.mp, self.tp, seed=self.tp.seed + 200 + epoch, vtilde=self.vtilde)),
        }
        self.best_tracker.maybe_update(
            epoch,
            metrics["test_reward"],
            policy=self.policy,
            value=self.value,
            qnet=self.qnet,
            vtilde=self.vtilde,
            lambda_k=self.lambda_k,
        )
        return metrics

    def _append_history(self, metrics: Dict[str, float]) -> None:
        """Append istory."""
        for key in self.history:
            self.history[key].append(float(metrics[key]) if key != "epoch" else int(metrics[key]))

    def _print_epoch_summary(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Print poch summary."""
        print(
            f"[Obj2][{epoch:03d}] TrainObjective={metrics['train_objective']:.6f} "
            f"TrainEuler={metrics['train_euler']:.6f} "
            f"TestEulerMSE={metrics['test_euler_mse']:.6f} "
            f"TestReward={metrics['test_reward']:.4f}"
        )

    def _post_training_hook(self) -> None:
        """Restore the Objective 2 epoch with the highest simulated test reward."""
        self.best_tracker.restore(policy=self.policy, value=self.value, qnet=self.qnet, vtilde=self.vtilde, lambda_k=self.lambda_k)
        if self.best_tracker.weights is not None:
            print(
                f"[Obj2] restored best epoch {self.best_tracker.best_epoch:03d} "
                f"with TestReward={self.best_tracker.best_score:.4f}",
                flush=True,
            )

    def _assemble_artifacts(self) -> ObjectiveTrainingArtifacts:
        """Assemble rtifacts."""
        return ObjectiveTrainingArtifacts(
            name=self.objective_name,
            policy=self.policy,
            value=self.value,
            vtilde=self.vtilde,
            lambda_k=self.lambda_k,
            qnet=self.qnet,
            history=self.history,
            ergodic_seed_offset=self.ergodic_seed_offset,
        )


class Objective3Trainer(BaseObjectiveTrainer):
    """Trainer for Objective 3: Bellman-residual minimization."""

    objective_name = "obj3"
    seed_offset = 2
    uses_dataset_batches = True
    ergodic_record_mode = "continuation"

    def __init__(
        self,
        *,
        mp: ModelParams,
        npol: NetParams,
        nval: NetParams,
        nq: NetParams,
        tp: TrainParams,
        objective_params: Obj3Params,
        jsonl_logger: Optional[JSONLLogger] = None,
        ckptio: Optional[TFCheckpointIO] = None,
        epoch_checkpoint_dir: Optional[str] = None,
        resume_epoch_checkpoint: bool = True,
    ) -> None:
        """Initialize Objective3Trainer."""
        self.npol = npol
        self.nval = nval
        self.nq = nq
        self.best_tracker = BestWeightsTracker()
        super().__init__(
            mp=mp,
            tp=tp,
            objective_params=objective_params,
            jsonl_logger=jsonl_logger,
            ckptio=ckptio,
            epoch_checkpoint_dir=epoch_checkpoint_dir,
            resume_epoch_checkpoint=resume_epoch_checkpoint,
        )

    def _initialize_history(self) -> Dict[str, List[float]]:
        """Initialize istory."""
        return {
            "epoch": [],
            "train_loss": [],
            "train_objective": [],
            "train_default_block": [],
            "train_bellman_block": [],
            "train_stationarity_block": [],
            "train_kkt_block": [],
            "train_bprime_block": [],
            "train_zp_block": [],
            "test_euler_mse": [],
            "test_reward": [],
        }

    def _build_components(self) -> None:
        """Build omponents."""
        state_center, state_scale = _default_state_normalization(self.tp)
        price_center, price_scale = _pricing_input_normalization(self.tp)
        self.policy = PolicyNet(self.npol, self.mp.k_min, self.mp.b_min, self.mp.b_max, input_center=state_center, input_scale=state_scale)
        self.value = ValueNet(self.nval, input_center=state_center, input_scale=state_scale)
        self.vtilde = VtildeNet(self.nval, input_center=state_center, input_scale=state_scale)
        self.lambda_k = MultiplierNet(self.nval, input_center=state_center, input_scale=state_scale)
        self.qnet = PricingNet(self.nq, self.mp.q_min, self.mp.q_max, input_center=price_center, input_scale=price_scale)
        _ = self.policy(tf.zeros((1, 3), dtype=tf.float32))
        _ = self.value(tf.zeros((1, 3), dtype=tf.float32))
        _ = self.vtilde(tf.zeros((1, 3), dtype=tf.float32))
        _ = self.lambda_k(tf.zeros((1, 3), dtype=tf.float32))
        _ = self.qnet(tf.zeros((1, 3), dtype=tf.float32))
        self.opt_policy = tf.keras.optimizers.Adam(self.tp.lr_policy * 0.5)
        self.opt_value = tf.keras.optimizers.Adam(self.tp.lr_value * 0.5)
        self.opt_vtilde = tf.keras.optimizers.Adam(self.tp.lr_vtilde * 0.5)
        self.opt_lambda = tf.keras.optimizers.Adam(self.tp.lr_vtilde * 0.5)
        self.opt_q = tf.keras.optimizers.Adam(self.tp.lr_q * 0.25)
        self.loss_metric = ScalarMetric("obj3_loss")
        self.def_metric = ScalarMetric("obj3_def")
        self.bell_metric = ScalarMetric("obj3_bell")
        self.stat_metric = ScalarMetric("obj3_stat")
        self.kkt_metric = ScalarMetric("obj3_kkt")
        self.bprime_metric = ScalarMetric("obj3_bprime")
        self.zp_metric = ScalarMetric("obj3_zp")

    @tf.function(reduce_retracing=True)
    def _train_step(self, k: tf.Tensor, b: tf.Tensor, z: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Run one compiled optimization step for Objective 3."""
        with tf.GradientTape(persistent=True) as tape:
            loss, def_block, bell_block, stat_block, kkt_block, bprime_block, zp_block = _call_objective(
                obj3_batch_metrics,
                policy=self.policy,
                value=self.value,
                qnet=self.qnet,
                mp=self.mp,
                tp=self.tp,
                op3=self.objective_params,
                k=k,
                b=b,
                z=z,
                vtilde=self.vtilde,
                lambda_net=self.lambda_k,
            )
            loss = tf.convert_to_tensor(loss, dtype=tf.float32)
            def_block = tf.convert_to_tensor(def_block, dtype=tf.float32)
            bell_block = tf.convert_to_tensor(bell_block, dtype=tf.float32)
            stat_block = tf.convert_to_tensor(stat_block, dtype=tf.float32)
            kkt_block = tf.convert_to_tensor(kkt_block, dtype=tf.float32)
            bprime_block = tf.convert_to_tensor(bprime_block, dtype=tf.float32)
            zp_block = tf.convert_to_tensor(zp_block, dtype=tf.float32)
        grads_p = tape.gradient(loss, self.policy.trainable_variables)
        grads_v = tape.gradient(loss, self.value.trainable_variables)
        grads_vt = tape.gradient(loss, self.vtilde.trainable_variables)
        grads_lam = tape.gradient(loss, self.lambda_k.trainable_variables)
        grads_q = tape.gradient(loss, self.qnet.trainable_variables)
        del tape
        _clip_and_apply(self.opt_policy, grads_p, self.policy.trainable_variables, self.tp.grad_clip)
        _clip_and_apply(self.opt_value, grads_v, self.value.trainable_variables, self.tp.grad_clip)
        _clip_and_apply(self.opt_vtilde, grads_vt, self.vtilde.trainable_variables, self.tp.grad_clip)
        _clip_and_apply(self.opt_lambda, grads_lam, self.lambda_k.trainable_variables, self.tp.grad_clip)
        _clip_and_apply(self.opt_q, grads_q, self.qnet.trainable_variables, self.tp.grad_clip)
        self.global_step.assign_add(1)
        return loss, def_block, bell_block, stat_block, kkt_block, bprime_block, zp_block

    def _run_epoch(self, epoch: int) -> None:
        """Run one training epoch and return the mean in-sample objective."""
        for metric in (self.loss_metric, self.def_metric, self.bell_metric, self.stat_metric, self.kkt_metric, self.bprime_metric, self.zp_metric):
            metric.reset()
        if epoch != 1 and epoch % self.tp.ergodic_refresh_every == 0:
            self._refresh_ergodic_buffer(seed=self.tp.seed + 120 + epoch)
        for step in range(self.tp.steps_per_epoch):
            if epoch == 1 and step == 0:
                print(f"[Obj3][epoch 001] compiling/running first train step...", flush=True)
            k, b, z = self._next_batch()
            loss, def_block, bell_block, stat_block, kkt_block, bprime_block, zp_block = self._train_step(k, b, z)
            self.loss_metric.update(loss)
            self.def_metric.update(def_block)
            self.bell_metric.update(bell_block)
            self.stat_metric.update(stat_block)
            self.kkt_metric.update(kkt_block)
            self.bprime_metric.update(bprime_block)
            self.zp_metric.update(zp_block)
            if epoch == 1 and (step + 1) % max(1, self.tp.steps_per_epoch // 5) == 0:
                print(f"[Obj3][epoch 001] step {step + 1}/{self.tp.steps_per_epoch}", flush=True)

    def _evaluate_epoch(self, epoch: int) -> Dict[str, float]:
        """Evaluate poch."""
        k_test, b_test, z_test = self._sample_test_states()
        metrics = {
            "objective": self.objective_name,
            "epoch": epoch,
            "train_loss": float(self.loss_metric.result().numpy()),
            "train_objective": float(self.loss_metric.result().numpy()),
            "train_default_block": float(self.def_metric.result().numpy()),
            "train_bellman_block": float(self.bell_metric.result().numpy()),
            "train_stationarity_block": float(self.stat_metric.result().numpy()),
            "train_kkt_block": float(self.kkt_metric.result().numpy()),
            "train_bprime_block": float(self.bprime_metric.result().numpy()),
            "train_zp_block": float(self.zp_metric.result().numpy()),
            "test_euler_mse": float(
                eval_test_euler_mse_obj3(
                    policy=self.policy,
                    value=self.value,
                    vtilde=self.vtilde,
                    qnet=self.qnet,
                    mp=self.mp,
                    tp=self.tp,
                    states_k=k_test,
                    states_b=b_test,
                    states_z=z_test,
                    seed=self.tp.seed + 300 + epoch,
                )
            ),
            "test_reward": float(eval_test_reward(self.policy, self.qnet, self.mp, self.tp, seed=self.tp.seed + 200 + epoch, vtilde=self.vtilde)),
        }
        self.best_tracker.maybe_update(
            epoch,
            metrics["test_reward"],
            policy=self.policy,
            value=self.value,
            qnet=self.qnet,
            vtilde=self.vtilde,
            lambda_k=self.lambda_k,
        )
        return metrics

    def _append_history(self, metrics: Dict[str, float]) -> None:
        """Append istory."""
        for key in self.history:
            self.history[key].append(float(metrics[key]) if key != "epoch" else int(metrics[key]))

    def _print_epoch_summary(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Print poch summary."""
        print(
            f"[Obj3][{epoch:03d}] TrainObjective={metrics['train_objective']:.6f} "
            f"DefBlock={metrics['train_default_block']:.6f} "
            f"BellBlock={metrics['train_bellman_block']:.6f} "
            f"KKTBlock={metrics['train_kkt_block']:.6f} "
            f"BPrimeBlock={metrics['train_bprime_block']:.6f} "
            f"ZPBlock={metrics['train_zp_block']:.6f} "
            f"TestEulerMSE={metrics['test_euler_mse']:.6f} "
            f"TestReward={metrics['test_reward']:.4f}"
        )

    def _post_training_hook(self) -> None:
        """Run post-training bookkeeping that is specific to the objective."""
        self.best_tracker.restore(policy=self.policy, value=self.value, qnet=self.qnet, vtilde=self.vtilde, lambda_k=self.lambda_k)
        if self.best_tracker.weights is not None:
            print(
                f"[Obj3] restored best epoch {self.best_tracker.best_epoch:03d} "
                f"with TestReward={self.best_tracker.best_score:.4f}"
            )

    def _assemble_artifacts(self) -> ObjectiveTrainingArtifacts:
        """Assemble rtifacts."""
        return ObjectiveTrainingArtifacts(
            name=self.objective_name,
            policy=self.policy,
            value=self.value,
            vtilde=self.vtilde,
            lambda_k=self.lambda_k,
            qnet=self.qnet,
            history=self.history,
            ergodic_seed_offset=self.ergodic_seed_offset,
        )


def train_objective_1(
    mp: ModelParams,
    npol: NetParams,
    nq: NetParams,
    tp: TrainParams,
    op1: Obj1Params,
    jsonl_logger: Optional[JSONLLogger] = None,
    ckptio: Optional[TFCheckpointIO] = None,
) -> Tuple[PolicyNet, PricingNet, Dict[str, List[float]]]:
    """Compatibility wrapper returning the Objective 1 tuple interface."""
    artifacts = Objective1Trainer(
        mp=mp,
        npol=npol,
        nq=nq,
        tp=tp,
        objective_params=op1,
        jsonl_logger=jsonl_logger,
        ckptio=ckptio,
    ).train()
    return artifacts.policy, artifacts.qnet, artifacts.history


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
) -> Tuple[PolicyNet, ValueNet, VtildeNet, PricingNet, Dict[str, List[float]]]:
    """Compatibility wrapper returning the Objective 2 tuple interface."""
    artifacts = Objective2Trainer(
        mp=mp,
        npol=npol,
        nval=nval,
        nvt=nvt,
        nq=nq,
        tp=tp,
        objective_params=op2,
        jsonl_logger=jsonl_logger,
        ckptio=ckptio,
    ).train()
    return artifacts.policy, artifacts.value, artifacts.vtilde, artifacts.qnet, artifacts.history


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
    """Compatibility wrapper returning the Objective 3 tuple interface."""
    artifacts = Objective3Trainer(
        mp=mp,
        npol=npol,
        nval=nval,
        nq=nq,
        tp=tp,
        objective_params=op3,
        jsonl_logger=jsonl_logger,
        ckptio=ckptio,
    ).train()
    return artifacts.policy, artifacts.value, artifacts.qnet, artifacts.history

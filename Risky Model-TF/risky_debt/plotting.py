# risky_debt/plotting.py
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from .config import ModelParams, TrainParams
from .networks import PolicyNet


def save_hist_npz(path: str, hist: Dict[str, List[float]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **{k: np.asarray(v) for k, v in hist.items()})


def _safe_get(hist: Dict[str, List[float]], key: str) -> Optional[np.ndarray]:
    v = hist.get(key, None)
    if v is None:
        return None
    return np.asarray(v)


def _ensure_dir_for_prefix(out_prefix: str) -> None:
    # out_prefix is like ".../figures/obj1" -> directory is ".../figures"
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)


def _plot_series(
    e: np.ndarray, y: np.ndarray, title: str, ylabel: str, out_path: str
) -> None:
    plt.figure()
    plt.plot(e, y)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_effectiveness_obj1(hist: Dict[str, List[float]], out_prefix: str) -> None:
    """
    Obj1: produce THREE separate plots (when available):
      1) TrainReward (or TrainLoss if reward not present)
      2) TestReward  (or TestLoss  if reward not present)
      3) TestEulerMSE
    Also keeps legacy combined plot: out_prefix + "_effectiveness.png"
    """
    _ensure_dir_for_prefix(out_prefix)

    e = np.asarray(hist["epoch"])

    train_reward = _safe_get(hist, "train_reward")
    test_reward = _safe_get(hist, "test_reward")
    train_loss = _safe_get(hist, "train_loss")
    test_loss = _safe_get(hist, "test_loss")
    test_euler = _safe_get(hist, "test_euler_mse")

    # ---- NEW: 3 separate figures ----
    # 1) Train metric
    if train_reward is not None:
        _plot_series(
            e,
            train_reward,
            title="Objective 1: TrainReward",
            ylabel="TrainReward",
            out_path=out_prefix + "_train_reward.png",
        )
    elif train_loss is not None:
        _plot_series(
            e,
            train_loss,
            title="Objective 1: TrainLoss",
            ylabel="TrainLoss",
            out_path=out_prefix + "_train_loss.png",
        )

    # 2) Test metric
    if test_reward is not None:
        _plot_series(
            e,
            test_reward,
            title="Objective 1: TestReward",
            ylabel="TestReward",
            out_path=out_prefix + "_test_reward.png",
        )
    elif test_loss is not None:
        _plot_series(
            e,
            test_loss,
            title="Objective 1: TestLoss",
            ylabel="TestLoss",
            out_path=out_prefix + "_test_loss.png",
        )

    # 3) Euler diagnostic
    if test_euler is not None:
        _plot_series(
            e,
            test_euler,
            title="Objective 1: TestEulerMSE",
            ylabel="TestEulerMSE",
            out_path=out_prefix + "_test_euler.png",
        )

    # ---- Legacy: keep original combined plot (for compatibility) ----
    plt.figure()
    plotted_any = False
    if train_reward is not None:
        plt.plot(e, train_reward, label="TrainReward")
        plotted_any = True
    if test_reward is not None:
        plt.plot(e, test_reward, label="TestReward")
        plotted_any = True
    if (not plotted_any) and (train_loss is not None):
        plt.plot(e, train_loss, label="TrainLoss")
        plotted_any = True
    if (not plotted_any) and (test_loss is not None):
        plt.plot(e, test_loss, label="TestLoss")
        plotted_any = True
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix + "_effectiveness.png", dpi=150)
    plt.close()

    # Legacy Euler (also keep)
    if test_euler is not None:
        plt.figure()
        plt.plot(e, test_euler, label="TestEulerMSE")
        plt.xlabel("Epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_prefix + "_euler.png", dpi=150)
        plt.close()


def plot_effectiveness_obj23(
    hist: Dict[str, List[float]], out_prefix: str, obj_name: str = "Obj2/3"
) -> None:
    """
    Obj2/3: produce THREE separate plots (when available):
      1) TrainLoss
      2) TestEulerMSE
      3) TestReward
    Also keeps legacy combined plot: out_prefix + "_effectiveness.png"
    """
    _ensure_dir_for_prefix(out_prefix)

    e = np.asarray(hist["epoch"])
    train_loss = _safe_get(hist, "train_loss")
    test_euler = _safe_get(hist, "test_euler_mse")
    test_reward = _safe_get(hist, "test_reward")

    # ---- NEW: 3 separate figures ----
    if train_loss is not None:
        _plot_series(
            e,
            train_loss,
            title=f"{obj_name}: TrainLoss",
            ylabel="TrainLoss",
            out_path=out_prefix + "_train_loss.png",
        )

    if test_euler is not None:
        _plot_series(
            e,
            test_euler,
            title=f"{obj_name}: TestEulerMSE",
            ylabel="TestEulerMSE",
            out_path=out_prefix + "_test_euler.png",
        )

    if test_reward is not None:
        _plot_series(
            e,
            test_reward,
            title=f"{obj_name}: TestReward",
            ylabel="TestReward",
            out_path=out_prefix + "_test_reward.png",
        )

    # ---- Legacy combined plot (keep) ----
    plt.figure()
    if train_loss is not None:
        plt.plot(e, train_loss, label=f"{obj_name} TrainLoss")
    if test_euler is not None:
        plt.plot(e, test_euler, label=f"{obj_name} TestEulerMSE")
    if test_reward is not None:
        plt.plot(e, test_reward, label=f"{obj_name} TestReward")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix + "_effectiveness.png", dpi=150)
    plt.close()


def plot_ergodic_set_kb(
    policy: PolicyNet,
    mp: ModelParams,
    tp: TrainParams,
    seed: int,
    out_path: str,
    burn_in: int = 200,
    T: int = 1200,
) -> None:
    """
    Minimal ergodic-set plot in (k,b). We assume policy maps (k,b,z)->(k',b').
    We simulate with a simple AR(1) for z and keep k>=k_min.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # init
    k = tf.fill((1,), tf.constant(tp.k0_low, tf.float32))
    b = tf.fill((1,), tf.constant(tp.b0_low, tf.float32))
    z = tf.fill((1,), tf.constant(tp.z0_low, tf.float32))

    ks = []
    bs = []

    for t in range(T):
        x = tf.stack([k, b, z], axis=1)  # [1,3]
        kb_next = policy(x)
        k = tf.maximum(kb_next[:, 0], mp.k_min)
        b = kb_next[:, 1]

        # AR(1) in logs for z
        z = tf.exp(
            mp.rho * tf.math.log(tf.maximum(z, mp.z_min))
            + tf.random.normal(tf.shape(z), 0.0, mp.sigma_eps, tf.float32)
        )

        if t >= burn_in:
            ks.append(float(k.numpy()[0]))
            bs.append(float(b.numpy()[0]))

    plt.figure()
    plt.scatter(ks, bs, s=6)
    plt.xlabel("k")
    plt.ylabel("b")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

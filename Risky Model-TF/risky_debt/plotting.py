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

    # ---- Legacy: keep your old combined plot (for compatibility) ----
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
    burn_in: int = 500,
    T: int = 2500,
    n_paths: int = 128,
    thin: int = 5,
    max_points: int = 20000,
) -> None:
    """
    Proper "ergodic cloud" plot in (k,b):

    - simulate MANY independent paths in parallel (vectorized)
    - randomize initial states (k0,b0,z0) from broad support
    - discard burn-in
    - pool/thin samples to form an empirical approximation to the invariant distribution

    This matches the theoretical ergodic distribution idea, instead of a single trajectory line.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # --- 1) Random initial states (broad support) ---
    k = tf.random.uniform((n_paths,), tp.k0_low, tp.k0_high, dtype=tf.float32)
    b = tf.random.uniform((n_paths,), tp.b0_low, tp.b0_high, dtype=tf.float32)
    z = tf.random.uniform((n_paths,), tp.z0_low, tp.z0_high, dtype=tf.float32)

    ks_list = []
    bs_list = []
    total_kept = 0

    for t in range(T):
        # --- 2) Policy step ---
        x = tf.stack([k, b, z], axis=1)  # [n_paths, 3]
        kb_next = policy(x)  # [n_paths, 2]
        k = tf.maximum(kb_next[:, 0], mp.k_min)
        b = kb_next[:, 1]

        # --- 3) Shock transition: AR(1) in logs ---
        eps = tf.random.normal((n_paths,), 0.0, mp.sigma_eps, dtype=tf.float32)
        z = tf.exp(mp.rho * tf.math.log(tf.maximum(z, mp.z_min)) + eps)

        # --- 4) After burn-in, record pooled samples (with thinning + cap) ---
        if t >= burn_in and ((t - burn_in) % thin == 0):
            k_np = k.numpy()
            b_np = b.numpy()

            # optional: cap points so plot stays fast and readable
            remaining = max_points - total_kept
            if remaining <= 0:
                break

            if k_np.shape[0] > remaining:
                idx = np.random.choice(k_np.shape[0], size=remaining, replace=False)
                k_np = k_np[idx]
                b_np = b_np[idx]

            ks_list.append(k_np)
            bs_list.append(b_np)
            total_kept += k_np.shape[0]

    if len(ks_list) == 0:
        ks = np.array([])
        bs = np.array([])
    else:
        ks = np.concatenate(ks_list, axis=0)
        bs = np.concatenate(bs_list, axis=0)

    # --- 5) Plot pooled cloud ---
    plt.figure()
    plt.scatter(ks, bs, s=4, alpha=0.35)
    plt.xlabel("k")
    plt.ylabel("b")
    plt.title(
        f"Ergodic cloud (n_paths={n_paths}, burn_in={burn_in}, T={T}, thin={thin})"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

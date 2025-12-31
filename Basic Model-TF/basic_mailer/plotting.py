from __future__ import annotations

import os  # os: create folders + join paths.
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .config import ModelParams, TrainParams
from .networks import PolicyNet
from .simulation import simulate_ergodic_dataset


# Creates the folder path if it doesn’t exist.
# exist_ok=True prevents error if the folder already exists.
# Used before saving any file to ensure directory exists.
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# Saves training history dictionary to a .npz file
def save_hist_npz(path: str, hist: Dict[str, List[float]]) -> None:
    ensure_dir(os.path.dirname(path))
    np.savez(path, **{k: np.array(v) for k, v in hist.items()})


def plot_effectiveness_obj1(hist: Dict[str, List[float]], out_dir: str) -> None:
    """
    Objective 1: produce 3 separate plots (NOT combined):
      1) TrainReward vs Epoch
      2) TestEulerMSE vs Epoch
      3) TestReward vs Epoch
    """
    # Creates output folder (e.g., outputs/run1/figures/obj1)
    ensure_dir(out_dir)
    # Extract epoch index
    # e becomes something like [1,2,3,...].
    # All plots use this as the x-axis.
    e = np.array(hist["epoch"])

    # 1) TrainReward
    plt.figure(figsize=(9, 4))
    plt.plot(e, hist["train_reward"], label="TrainReward")
    plt.xlabel("Epoch")
    plt.title("Objective 1 — TrainReward")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(out_dir, "effectiveness_obj1_train_reward.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # -------- Train Loss (negative reward) --------
    plt.figure(figsize=(9, 4))
    plt.plot(e, hist["train_loss"], label="TrainLoss = -TrainReward")
    plt.xlabel("Epoch")
    plt.title("Objective 1 — Training Loss (Mailer convention)")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(out_dir, "effectiveness_obj1_train_loss.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # 2) TestEulerMSE
    plt.figure(figsize=(9, 4))
    plt.plot(e, hist["test_euler_mse"], label="TestEulerMSE")
    plt.xlabel("Epoch")
    plt.title("Objective 1 — TestEulerMSE (policy-only f diagnostic)")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(out_dir, "effectiveness_obj1_test_euler_mse.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # 3) TestReward
    plt.figure(figsize=(9, 4))
    plt.plot(e, hist["test_reward"], label="TestReward")
    plt.xlabel("Epoch")
    plt.title("Objective 1 — TestReward")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(out_dir, "effectiveness_obj1_test_reward.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def plot_effectiveness_obj23(
    hist: Dict[str, List[float]], out_dir: str, obj_name: str
) -> None:
    ensure_dir(out_dir)
    e = np.array(hist["epoch"])

    plt.figure(figsize=(9, 4))
    plt.plot(e, hist["train_loss"], label="TrainLoss")
    plt.xlabel("Epoch")
    plt.title(f"{obj_name} — TrainLoss")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(out_dir, f"effectiveness_{obj_name.lower()}_train_loss.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(e, hist["test_euler_mse"], label="TestEulerMSE")
    plt.xlabel("Epoch")
    plt.title(f"{obj_name} — TestEulerMSE")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(out_dir, f"effectiveness_{obj_name.lower()}_test_euler_mse.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(e, hist["test_reward"], label="TestReward")
    plt.xlabel("Epoch")
    plt.title(f"{obj_name} — TestReward")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(out_dir, f"effectiveness_{obj_name.lower()}_test_reward.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def plot_testreward_comparison(
    hist1: Dict[str, List[float]],
    hist2: Dict[str, List[float]],
    hist3: Dict[str, List[float]],
    out_dir: str,
) -> None:
    ensure_dir(out_dir)
    e = np.array(hist1["epoch"])
    plt.figure(figsize=(10, 5))
    plt.plot(e, hist1["test_reward"], label="Obj1 TestReward")
    plt.plot(e, hist2["test_reward"], label="Obj2 TestReward")
    plt.plot(e, hist3["test_reward"], label="Obj3 TestReward")
    plt.xlabel("Epoch")
    plt.title("TestReward comparison across objectives")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(out_dir, "effectiveness_testreward_comparison.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def plot_ergodic_set(
    policy: PolicyNet,
    mp: ModelParams,
    tp: TrainParams,
    seed: int,
    out_path: str,
    max_points: int = 200_000,
) -> None:
    k_buf, z_buf = simulate_ergodic_dataset(policy, mp, tp, seed=seed)

    n = len(k_buf)
    if n > max_points:
        idx = np.random.choice(n, size=max_points, replace=False)
        k_plot = k_buf[idx]
        z_plot = z_buf[idx]
    else:
        k_plot, z_plot = k_buf, z_buf

    ensure_dir(os.path.dirname(out_path))
    plt.figure(figsize=(7, 6))
    plt.scatter(k_plot, z_plot, s=1, alpha=0.25)
    plt.xlabel("k_t")
    plt.ylabel("z_t")
    plt.title("Ergodic set projection: (k_t, z_t)")
    plt.grid(True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

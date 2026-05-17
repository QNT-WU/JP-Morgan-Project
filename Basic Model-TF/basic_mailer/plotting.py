"""Plotting utilities for training and benchmark diagnostics."""

from __future__ import annotations

import csv
import os  # os: create folders + join paths.
from typing import Dict, List, Mapping, Sequence, Tuple

# Dict[str, List[float]]: your training history structure (epoch → metrics lists).

import numpy as np
import matplotlib.pyplot as plt

from .config import ModelParams, TrainParams
from .networks import PolicyNet
from .simulation import simulate_ergodic_dataset


# Creates the folder path if it doesn’t exist.
# exist_ok=True prevents error if the folder already exists.
# Used before saving any file to ensure directory exists.
def ensure_dir(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


# Saves your training history dictionary to a .npz file
# Inputs:
# path: file path like "outputs/run1/hist_obj1.npz"
# hist: dict like:


def save_hist_npz(path: str, hist: Dict[str, List[float]]) -> None:
    """Save a history dictionary as a NumPy archive."""
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


    if "test_kkt_mse" in hist:
        plt.figure(figsize=(9, 4))
        plt.plot(e, hist["test_kkt_mse"], label="TestKKT-MSE")
        plt.xlabel("Epoch")
        plt.title("Objective 1 — Test KKT/Fischer-Burmeister MSE")
        plt.grid(True)
        plt.legend()
        plt.savefig(
            os.path.join(out_dir, "effectiveness_obj1_test_kkt_mse.png"),
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


def _plot_history_metric(
    hist: Dict[str, List[float]],
    out_dir: str,
    obj_name: str,
    metric_key: str,
    label: str,
    title: str,
    filename_suffix: str,
) -> None:
    """Plot one history metric if it is available."""
    if metric_key not in hist:
        return
    e = np.array(hist["epoch"])
    plt.figure(figsize=(9, 4))
    plt.plot(e, hist[metric_key], label=label)
    plt.xlabel("Epoch")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(out_dir, f"effectiveness_{obj_name.lower()}_{filename_suffix}.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def plot_effectiveness_obj23(
    hist: Dict[str, List[float]], out_dir: str, obj_name: str
) -> None:
    """Plot training diagnostics for Objectives 2 or 3.

    The Basic Model summary requires component diagnostics, not only a combined
    residual. Therefore this function saves separate plots for FB,
    stationarity, Bellman, combined KKT/total residual, and reward whenever the
    corresponding metric is present in the history dictionary.
    """
    ensure_dir(out_dir)

    _plot_history_metric(
        hist, out_dir, obj_name, "train_loss", "TrainLoss", f"{obj_name} — TrainLoss", "train_loss"
    )
    _plot_history_metric(
        hist, out_dir, obj_name, "test_fb_mse", "FB-MSE",
        f"{obj_name} — Test Fischer-Burmeister MSE", "test_fb_mse"
    )
    _plot_history_metric(
        hist, out_dir, obj_name, "test_stationarity_mse", "StationarityMSE",
        f"{obj_name} — Test KKT Stationarity MSE", "test_stationarity_mse"
    )
    _plot_history_metric(
        hist, out_dir, obj_name, "test_bellman_mse", "BellmanMSE",
        f"{obj_name} — Test Bellman MSE", "test_bellman_mse"
    )
    _plot_history_metric(
        hist, out_dir, obj_name, "test_kkt_mse", "KKT-MSE",
        f"{obj_name} — Combined Test KKT/Euler MSE", "test_kkt_mse"
    )
    _plot_history_metric(
        hist, out_dir, obj_name, "test_total_residual", "TotalResidual",
        f"{obj_name} — Total Bellman/KKT Residual", "test_total_residual"
    )
    # Backward-compatible alias. For Objective 2/3 this is the stationarity diagnostic.
    _plot_history_metric(
        hist, out_dir, obj_name, "test_euler_mse", "Stationarity/Euler MSE",
        f"{obj_name} — Test Stationarity/Euler MSE", "test_euler_mse"
    )
    _plot_history_metric(
        hist, out_dir, obj_name, "test_reward", "TestReward",
        f"{obj_name} — TestReward", "test_reward"
    )


def _latest(history: Mapping[str, Sequence[float]], key: str):
    """Return the latest value of a history metric, or blank if unavailable."""
    values = history.get(key)
    if not values:
        return ""
    return values[-1]


def _write_rows_csv(rows: Sequence[Mapping[str, object]], path: str) -> None:
    """Write rows as a CSV table."""
    if not rows:
        return
    ensure_dir(os.path.dirname(path))
    columns: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(str(key))
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def _write_rows_tex(rows: Sequence[Mapping[str, object]], path: str, caption: str, label: str) -> None:
    """Write rows as a simple LaTeX table."""
    if not rows:
        return
    ensure_dir(os.path.dirname(path))
    columns: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(str(key))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\\begin{table}[htbp]\n\\centering\n")
        handle.write(f"\\caption{{{caption}}}\n")
        handle.write(f"\\label{{{label}}}\n")
        handle.write("\\begin{tabular}{" + "l" * len(columns) + "}\n")
        handle.write("\\hline\n")
        handle.write(" & ".join(columns) + " \\\\ \n")
        handle.write("\\hline\n")
        for row in rows:
            vals = []
            for col in columns:
                val = row.get(col, "")
                vals.append(f"{val:.6g}" if isinstance(val, float) else str(val))
            handle.write(" & ".join(vals) + " \\\\ \n")
        handle.write("\\hline\n\\end{tabular}\n\\end{table}\n")

def save_effectiveness_report(
    histories: Mapping[str, Mapping[str, Sequence[float]]],
    out_dir: str,
) -> None:
    """Save effectiveness-measure tables required by the Basic Model summary."""
    table_dir = os.path.join(out_dir, "tables")
    rows = []
    for obj_name, hist in histories.items():
        rows.append({
            "objective": obj_name,
            "final_epoch": _latest(hist, "epoch"),
            "train_reward": _latest(hist, "train_reward"),
            "train_loss": _latest(hist, "train_loss"),
            "test_reward": _latest(hist, "test_reward"),
            "test_kkt_mse": _latest(hist, "test_kkt_mse"),
            "test_euler_mse_or_stationarity": _latest(hist, "test_euler_mse"),
            "test_euler_mse_interior": _latest(hist, "test_euler_mse_interior"),
            "interior_share": _latest(hist, "interior_share"),
            "mean_slack": _latest(hist, "mean_slack"),
            "test_fb_mse": _latest(hist, "test_fb_mse"),
            "test_stationarity_mse": _latest(hist, "test_stationarity_mse"),
            "test_bellman_mse": _latest(hist, "test_bellman_mse"),
            "test_total_residual": _latest(hist, "test_total_residual"),
        })
    _write_rows_csv(rows, os.path.join(table_dir, "effectiveness_summary.csv"))
    _write_rows_tex(
        rows,
        os.path.join(table_dir, "effectiveness_summary.tex"),
        caption="Effectiveness measures for Basic Model neural-network objectives",
        label="tab:effectiveness_summary",
    )

def plot_testreward_comparison(
    hist1: Dict[str, List[float]],
    hist2: Dict[str, List[float]],
    hist3: Dict[str, List[float]],
    out_dir: str,
) -> None:
    """Plot test-reward trajectories across all objectives."""
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
    """Plot the simulated ergodic state cloud."""
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

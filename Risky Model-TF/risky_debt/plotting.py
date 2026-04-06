"""Plotting helpers for training, benchmark, and estimation outputs."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from .config import ModelParams, TrainParams
from .networks import PolicyNet


def save_hist_npz(path: str, hist: Dict[str, List[float]]) -> None:
    """Persist one training history dictionary as a compressed NumPy archive."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **{k: np.asarray(v) for k, v in hist.items()})


def _safe_get(hist: Dict[str, List[float]], key: str) -> Optional[np.ndarray]:
    """Return one history series as a NumPy array when it exists."""
    v = hist.get(key, None)
    if v is None:
        return None
    return np.asarray(v)


def _ensure_dir_for_prefix(out_prefix: str) -> None:
    """Create the directory that will hold figures for ``out_prefix``."""
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)


def _root_from_prefix(out_prefix: str) -> tuple[str, str]:
    """Return `(figure_dir, stem)` for prefixes like `.../figures/obj1`."""
    return os.path.dirname(out_prefix), os.path.basename(out_prefix)


def _savefig_both(out_path_new: str, out_path_legacy: Optional[str] = None, dpi: int = 150) -> None:
    """Save a figure once using the canonical filename.

    The codebase previously saved both a new filename and a legacy filename for
    the exact same figure, which created duplicate PNG artifacts in the results
    directory. The canonical filename is now the only one written.
    """
    plt.tight_layout()
    plt.savefig(out_path_new, dpi=dpi)
    plt.close()


def _plot_series(
    e: np.ndarray,
    y: np.ndarray,
    title: str,
    ylabel: str,
    out_path: str,
    out_path_legacy: Optional[str] = None,
) -> None:
    """Plot and save one scalar training or evaluation series."""
    plt.figure()
    plt.plot(e, y)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    _savefig_both(out_path, out_path_legacy)


def plot_effectiveness_obj1(hist: Dict[str, List[float]], out_prefix: str) -> None:
    """Save Obj1 effectiveness plots using the basic-model folder/file style.

    New primary names follow the basic-model convention, for example:
      - figures/effectiveness_obj1_train_reward.png
      - figures/effectiveness_obj1_test_reward.png
      - figures/effectiveness_obj1_test_euler_mse.png

No combined summary figure is saved. The canonical per-metric files are the
    only effectiveness outputs kept for Obj1.
    """
    _ensure_dir_for_prefix(out_prefix)
    fig_dir, _stem = _root_from_prefix(out_prefix)

    e = np.asarray(hist["epoch"])
    train_reward = _safe_get(hist, "train_reward")
    test_reward = _safe_get(hist, "test_reward")
    train_loss = _safe_get(hist, "train_loss")
    test_loss = _safe_get(hist, "test_loss")
    test_euler = _safe_get(hist, "test_euler_mse")

    if train_reward is not None:
        _plot_series(
            e,
            train_reward,
            title="Objective 1: TrainReward",
            ylabel="TrainReward",
            out_path=os.path.join(fig_dir, "effectiveness_obj1_train_reward.png"),
            out_path_legacy=out_prefix + "_train_reward.png",
        )
    elif train_loss is not None:
        _plot_series(
            e,
            train_loss,
            title="Objective 1: TrainLoss",
            ylabel="TrainLoss",
            out_path=os.path.join(fig_dir, "effectiveness_obj1_train_loss.png"),
            out_path_legacy=out_prefix + "_train_loss.png",
        )

    if test_reward is not None:
        _plot_series(
            e,
            test_reward,
            title="Objective 1: TestReward",
            ylabel="TestReward",
            out_path=os.path.join(fig_dir, "effectiveness_obj1_test_reward.png"),
            out_path_legacy=out_prefix + "_test_reward.png",
        )
    elif test_loss is not None:
        _plot_series(
            e,
            test_loss,
            title="Objective 1: TestLoss",
            ylabel="TestLoss",
            out_path=os.path.join(fig_dir, "effectiveness_obj1_test_loss.png"),
            out_path_legacy=out_prefix + "_test_loss.png",
        )

    if test_euler is not None:
        _plot_series(
            e,
            test_euler,
            title="Objective 1: TestEulerMSE",
            ylabel="TestEulerMSE",
            out_path=os.path.join(fig_dir, "effectiveness_obj1_test_euler_mse.png"),
            out_path_legacy=out_prefix + "_test_euler.png",
        )

    # Intentionally do not save the old combined Obj1 summary figure or
    # the legacy standalone Euler figure. The canonical per-metric files above
    # are the only effectiveness outputs kept for Obj1.


def plot_effectiveness_obj23(
    hist: Dict[str, List[float]], out_prefix: str, obj_name: str = "Obj2/3"
) -> None:
    """Save canonical Obj2/Obj3 effectiveness plots with cleaner metric names.

    Backward compatibility is preserved by still honoring ``train_loss`` when older
    histories are loaded, but newer runs also save semantically clearer series such
    as ``train_objective`` and the residual-block diagnostics.
    """
    _ensure_dir_for_prefix(out_prefix)
    fig_dir, stem = _root_from_prefix(out_prefix)
    stem_lower = stem.lower()

    e = np.asarray(hist["epoch"])
    train_objective = _safe_get(hist, "train_objective")
    train_loss = _safe_get(hist, "train_loss")
    train_euler = _safe_get(hist, "train_euler")
    train_default = _safe_get(hist, "train_default_block")
    train_bell = _safe_get(hist, "train_bellman_block")
    train_zp = _safe_get(hist, "train_zp_block")
    test_euler = _safe_get(hist, "test_euler_mse")
    test_reward = _safe_get(hist, "test_reward")

    primary_train = train_objective if train_objective is not None else train_loss
    primary_train_label = "TrainObjective" if train_objective is not None else "TrainLoss"
    primary_train_file = "train_objective" if train_objective is not None else "train_loss"

    if primary_train is not None:
        _plot_series(
            e,
            primary_train,
            title=f"{obj_name}: {primary_train_label}",
            ylabel=primary_train_label,
            out_path=os.path.join(fig_dir, f"effectiveness_{stem_lower}_{primary_train_file}.png"),
            out_path_legacy=out_prefix + f"_{primary_train_file}.png",
        )

    if train_euler is not None:
        _plot_series(
            e,
            train_euler,
            title=f"{obj_name}: TrainEuler",
            ylabel="TrainEuler",
            out_path=os.path.join(fig_dir, f"effectiveness_{stem_lower}_train_euler.png"),
            out_path_legacy=out_prefix + "_train_euler.png",
        )

    if train_default is not None:
        _plot_series(
            e,
            train_default,
            title=f"{obj_name}: TrainDefaultBlock",
            ylabel="TrainDefaultBlock",
            out_path=os.path.join(fig_dir, f"effectiveness_{stem_lower}_train_default_block.png"),
            out_path_legacy=out_prefix + "_train_default_block.png",
        )

    if train_bell is not None:
        _plot_series(
            e,
            train_bell,
            title=f"{obj_name}: TrainBellmanBlock",
            ylabel="TrainBellmanBlock",
            out_path=os.path.join(fig_dir, f"effectiveness_{stem_lower}_train_bellman_block.png"),
            out_path_legacy=out_prefix + "_train_bellman_block.png",
        )

    if train_zp is not None:
        _plot_series(
            e,
            train_zp,
            title=f"{obj_name}: TrainZPBlock",
            ylabel="TrainZPBlock",
            out_path=os.path.join(fig_dir, f"effectiveness_{stem_lower}_train_zp_block.png"),
            out_path_legacy=out_prefix + "_train_zp_block.png",
        )

    if test_euler is not None:
        _plot_series(
            e,
            test_euler,
            title=f"{obj_name}: TestEulerMSE",
            ylabel="TestEulerMSE",
            out_path=os.path.join(fig_dir, f"effectiveness_{stem_lower}_test_euler_mse.png"),
            out_path_legacy=out_prefix + "_test_euler.png",
        )

    if test_reward is not None:
        _plot_series(
            e,
            test_reward,
            title=f"{obj_name}: TestReward",
            ylabel="TestReward",
            out_path=os.path.join(fig_dir, f"effectiveness_{stem_lower}_test_reward.png"),
            out_path_legacy=out_prefix + "_test_reward.png",
        )

    # Intentionally do not save the old combined Obj2/Obj3 effectiveness
    # summary figure. The canonical per-metric files above are the only
    # effectiveness outputs kept.


def plot_testreward_comparison(hists: Dict[str, Dict[str, List[float]]], out_path: str) -> None:
    """Plot TestReward curves for Obj1/Obj2/Obj3 on one figure."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    for obj_name, hist in hists.items():
        e = np.asarray(hist.get("epoch", []))
        y = _safe_get(hist, "test_reward")
        if y is not None and e.size == y.size and e.size > 0:
            plt.plot(e, y, label=obj_name)
    plt.xlabel("Epoch")
    plt.ylabel("TestReward")
    plt.title("TestReward comparison across objectives")
    plt.legend()
    _savefig_both(out_path)


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
    """Save risky-debt ergodic-set diagnostics.

    The legacy file at ``out_path`` is preserved, but the risky-debt model now
    also saves additional 2D views that expose the third state variable ``z``:

    - ``*_k_vs_z.png``
    - ``*_b_vs_z.png``
    - ``*_k_vs_b_colored_by_z.png``
    - ``*_3d_k_b_z.png``

    This keeps backward compatibility while making the ergodic cloud much more
    informative than a plain ``(k,b)`` scatter alone.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    k = tf.random.uniform((n_paths,), tp.k0_low, tp.k0_high, dtype=tf.float32)
    b = tf.random.uniform((n_paths,), tp.b0_low, tp.b0_high, dtype=tf.float32)
    z = tf.random.uniform((n_paths,), tp.z0_low, tp.z0_high, dtype=tf.float32)

    ks_list = []
    bs_list = []
    zs_list = []
    total_kept = 0

    for t in range(T):
        x = tf.stack([k, b, z], axis=1)
        kb_next = policy(x)
        k = tf.maximum(kb_next[:, 0], mp.k_min)
        b = kb_next[:, 1]

        eps = tf.random.normal((n_paths,), 0.0, mp.sigma_eps, dtype=tf.float32)
        z = tf.exp(mp.rho * tf.math.log(tf.maximum(z, mp.z_min)) + eps)

        if t >= burn_in and ((t - burn_in) % thin == 0):
            k_np = k.numpy()
            b_np = b.numpy()
            z_np = z.numpy()
            remaining = max_points - total_kept
            if remaining <= 0:
                break
            if k_np.shape[0] > remaining:
                idx = np.random.choice(k_np.shape[0], size=remaining, replace=False)
                k_np = k_np[idx]
                b_np = b_np[idx]
                z_np = z_np[idx]
            ks_list.append(k_np)
            bs_list.append(b_np)
            zs_list.append(z_np)
            total_kept += k_np.shape[0]

    if ks_list:
        k_all = np.concatenate(ks_list)
        b_all = np.concatenate(bs_list)
        z_all = np.concatenate(zs_list)
    else:
        k_all = np.array([], dtype=np.float32)
        b_all = np.array([], dtype=np.float32)
        z_all = np.array([], dtype=np.float32)

    stem, ext = os.path.splitext(out_path)

    # Legacy path: keep a plain k-b scatter for backward compatibility.
    plt.figure(figsize=(6.4, 5.2))
    if k_all.size > 0:
        plt.scatter(k_all, b_all, s=4, alpha=0.35)
        plt.xlim(float(np.min(k_all)), float(np.max(k_all)))
        plt.ylim(float(np.min(b_all)), float(np.max(b_all)))
    plt.xlabel("Capital k")
    plt.ylabel("Debt b")
    plt.title("Ergodic set in (k,b)")
    _savefig_both(out_path)

    plt.figure(figsize=(6.4, 5.2))
    if k_all.size > 0:
        plt.scatter(k_all, z_all, s=4, alpha=0.4)
    plt.xlabel("Capital k")
    plt.ylabel("Productivity z")
    plt.title("Ergodic set: z against k")
    _savefig_both(stem + "_k_vs_z" + ext)

    plt.figure(figsize=(6.4, 5.2))
    if b_all.size > 0:
        plt.scatter(b_all, z_all, s=4, alpha=0.4)
    plt.xlabel("Debt b")
    plt.ylabel("Productivity z")
    plt.title("Ergodic set: z against b")
    _savefig_both(stem + "_b_vs_z" + ext)

    plt.figure(figsize=(6.4, 5.2))
    if k_all.size > 0:
        sc = plt.scatter(k_all, b_all, c=z_all, s=4, alpha=0.45)
        plt.colorbar(sc, label="Productivity z")
    plt.xlabel("Capital k")
    plt.ylabel("Debt b")
    plt.title("Ergodic set: k against b, colored by z")
    _savefig_both(stem + "_k_vs_b_colored_by_z" + ext)

    fig = plt.figure(figsize=(7.0, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    if k_all.size > 0:
        ax.scatter(k_all, b_all, z_all, s=3, alpha=0.25)
    ax.set_xlabel("Capital k")
    ax.set_ylabel("Debt b")
    ax.set_zlabel("Productivity z")
    ax.set_title("Ergodic set: 3D (k,b,z)")
    plt.tight_layout()
    plt.savefig(stem + "_3d_k_b_z" + ext, dpi=150)
    plt.close(fig)


def plot_benchmark_method_summaries(bench: Dict[str, np.ndarray], out_dir: str, method: str) -> None:
    """Save benchmark-method summary plots into ``figures/benchmark_methods``.

    The plots distinguish between inner-solver convergence and outer pricing-loop
    convergence, and also save benchmark value/policy/default/pricing objects.
    """
    os.makedirs(out_dir, exist_ok=True)

    outer_hist = np.asarray(bench.get("outer_q_error_history", []), dtype=float)
    if outer_hist.size > 0:
        plt.figure()
        plt.plot(np.arange(1, outer_hist.size + 1), outer_hist)
        plt.xlabel("Outer iteration")
        plt.ylabel(r"$||\Delta q||_\infty$")
        plt.title(f"Benchmark pricing-loop convergence ({method})")
        _savefig_both(os.path.join(out_dir, f"pricing_loop_convergence_{method}.png"))

    inner_hist = np.asarray(bench.get("inner_error_history", []), dtype=float)
    if inner_hist.size > 0:
        plt.figure()
        plt.plot(np.arange(1, inner_hist.size + 1), inner_hist)
        plt.xlabel("Inner iteration")
        plt.ylabel(r"$||\Delta V||_\infty$")
        plt.title(f"Benchmark inner-solver convergence ({method})")
        _savefig_both(os.path.join(out_dir, f"inner_solver_convergence_{method}.png"))

    polchg_hist = np.asarray(bench.get("policy_change_history", []), dtype=float)
    if polchg_hist.size > 0:
        plt.figure()
        plt.plot(np.arange(1, polchg_hist.size + 1), polchg_hist)
        plt.xlabel("MPI improvement iteration")
        plt.ylabel("Changed policy nodes")
        plt.title(f"Benchmark policy-change convergence ({method})")
        _savefig_both(os.path.join(out_dir, f"policy_change_convergence_{method}.png"))

    k_grid = np.asarray(bench["k_grid"])
    b_grid = np.asarray(bench["b_grid"])
    z_grid = np.asarray(bench["z_grid"])
    V = np.asarray(bench["V_star"])
    kp = np.asarray(bench.get("policy_kp_star", k_grid[np.asarray(bench["pol_k_idx"])]), dtype=float)
    bp = np.asarray(bench.get("policy_bp_star", b_grid[np.asarray(bench["pol_b_idx"])]), dtype=float)
    q = np.asarray(bench["q_star"], dtype=float)
    default_ind = np.asarray(bench.get("default_star", V <= 1e-10), dtype=float)
    mid_z = len(z_grid) // 2

    plt.figure(figsize=(7, 5))
    plt.imshow(
        V[:, :, mid_z].T,
        origin="lower",
        aspect="auto",
        extent=[k_grid[0], k_grid[-1], b_grid[0], b_grid[-1]],
    )
    plt.colorbar()
    plt.xlabel("k")
    plt.ylabel("b")
    plt.title(f"Benchmark value at z[{mid_z}] ({method})")
    _savefig_both(os.path.join(out_dir, f"benchmark_value_function_{method}.png"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    im0 = axes[0].imshow(
        kp[:, :, mid_z].T,
        origin="lower",
        aspect="auto",
        extent=[k_grid[0], k_grid[-1], b_grid[0], b_grid[-1]],
    )
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("b")
    axes[0].set_title(f"k' benchmark ({method})")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        bp[:, :, mid_z].T,
        origin="lower",
        aspect="auto",
        extent=[k_grid[0], k_grid[-1], b_grid[0], b_grid[-1]],
    )
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("b")
    axes[1].set_title(f"b' benchmark ({method})")
    fig.colorbar(im1, ax=axes[1])
    fig.suptitle(f"Benchmark policy functions at z[{mid_z}] ({method})")
    _savefig_both(os.path.join(out_dir, f"benchmark_policy_function_{method}.png"))

    plt.figure(figsize=(7, 5))
    plt.imshow(
        default_ind[:, :, mid_z].T,
        origin="lower",
        aspect="auto",
        extent=[k_grid[0], k_grid[-1], b_grid[0], b_grid[-1]],
    )
    plt.colorbar()
    plt.xlabel("k")
    plt.ylabel("b")
    plt.title(f"Benchmark default region at z[{mid_z}] ({method})")
    _savefig_both(os.path.join(out_dir, f"benchmark_default_region_{method}.png"))

    plt.figure(figsize=(7, 5))
    plt.imshow(
        q[mid_z, :, :].T,
        origin="lower",
        aspect="auto",
        extent=[k_grid[0], k_grid[-1], b_grid[0], b_grid[-1]],
    )
    plt.colorbar()
    plt.xlabel("k'")
    plt.ylabel("b'")
    plt.title(f"Benchmark pricing schedule q at z[{mid_z}] ({method})")
    _savefig_both(os.path.join(out_dir, f"benchmark_pricing_schedule_{method}.png"))



def plot_benchmark_method_comparison(
    bench_vi: Dict[str, np.ndarray],
    bench_mpi: Dict[str, np.ndarray],
    out_dir: str,
) -> Dict[str, float]:
    """Save direct VI-vs-MPI comparison plots and return summary metrics."""
    os.makedirs(out_dir, exist_ok=True)

    k_grid = np.asarray(bench_vi["k_grid"])
    b_grid = np.asarray(bench_vi["b_grid"])
    z_grid = np.asarray(bench_vi["z_grid"])
    mid_z = len(z_grid) // 2

    V_vi = np.asarray(bench_vi["V_star"], dtype=float)
    V_mpi = np.asarray(bench_mpi["V_star"], dtype=float)
    kp_vi = np.asarray(bench_vi.get("policy_kp_star", k_grid[np.asarray(bench_vi["pol_k_idx"])]), dtype=float)
    kp_mpi = np.asarray(bench_mpi.get("policy_kp_star", k_grid[np.asarray(bench_mpi["pol_k_idx"])]), dtype=float)
    bp_vi = np.asarray(bench_vi.get("policy_bp_star", b_grid[np.asarray(bench_vi["pol_b_idx"])]), dtype=float)
    bp_mpi = np.asarray(bench_mpi.get("policy_bp_star", b_grid[np.asarray(bench_mpi["pol_b_idx"])]), dtype=float)
    q_vi = np.asarray(bench_vi["q_star"], dtype=float)
    q_mpi = np.asarray(bench_mpi["q_star"], dtype=float)
    d_vi = np.asarray(bench_vi.get("default_star", bench_vi["V_star"] <= 1e-10), dtype=float)
    d_mpi = np.asarray(bench_mpi.get("default_star", bench_mpi["V_star"] <= 1e-10), dtype=float)

    metrics = {
        "value_rmse": float(np.sqrt(np.mean((V_vi - V_mpi) ** 2))),
        "value_sup": float(np.max(np.abs(V_vi - V_mpi))),
        "kp_rmse": float(np.sqrt(np.mean((kp_vi - kp_mpi) ** 2))),
        "kp_sup": float(np.max(np.abs(kp_vi - kp_mpi))),
        "bp_rmse": float(np.sqrt(np.mean((bp_vi - bp_mpi) ** 2))),
        "bp_sup": float(np.max(np.abs(bp_vi - bp_mpi))),
        "q_rmse": float(np.sqrt(np.mean((q_vi - q_mpi) ** 2))),
        "q_sup": float(np.max(np.abs(q_vi - q_mpi))),
        "default_mismatch_rate": float(np.mean(d_vi != d_mpi)),
        "vi_runtime_seconds": float(bench_vi.get("runtime_seconds", np.nan)),
        "mpi_runtime_seconds": float(bench_mpi.get("runtime_seconds", np.nan)),
    }

    # Convergence overlays.
    plt.figure(figsize=(7, 5))
    outer_vi = np.asarray(bench_vi.get("outer_q_error_history", []), dtype=float)
    outer_mpi = np.asarray(bench_mpi.get("outer_q_error_history", []), dtype=float)
    if outer_vi.size > 0:
        plt.plot(np.arange(1, outer_vi.size + 1), outer_vi, label="VI")
    if outer_mpi.size > 0:
        plt.plot(np.arange(1, outer_mpi.size + 1), outer_mpi, label="MPI")
    plt.xlabel("Outer iteration")
    plt.ylabel(r"$||\Delta q||_\infty$")
    plt.title("Pricing-loop convergence: VI vs MPI")
    plt.legend()
    _savefig_both(os.path.join(out_dir, "vi_vs_mpi_pricing_loop_convergence.png"))

    plt.figure(figsize=(7, 5))
    inner_vi = np.asarray(bench_vi.get("inner_error_history", []), dtype=float)
    inner_mpi = np.asarray(bench_mpi.get("inner_error_history", []), dtype=float)
    if inner_vi.size > 0:
        plt.plot(np.arange(1, inner_vi.size + 1), inner_vi, label="VI inner")
    if inner_mpi.size > 0:
        plt.plot(np.arange(1, inner_mpi.size + 1), inner_mpi, label="MPI inner")
    plt.xlabel("Inner iteration")
    plt.ylabel(r"$||\Delta V||_\infty$")
    plt.title("Inner-solver convergence: VI vs MPI")
    plt.legend()
    _savefig_both(os.path.join(out_dir, "vi_vs_mpi_inner_solver_convergence.png"))

    for arr, fname, title in [
        (V_mpi[:, :, mid_z] - V_vi[:, :, mid_z], "vi_vs_mpi_value_diff.png", "MPI - VI value"),
        (kp_mpi[:, :, mid_z] - kp_vi[:, :, mid_z], "vi_vs_mpi_k_policy_diff.png", "MPI - VI k' policy"),
        (bp_mpi[:, :, mid_z] - bp_vi[:, :, mid_z], "vi_vs_mpi_b_policy_diff.png", "MPI - VI b' policy"),
        (q_mpi[mid_z, :, :] - q_vi[mid_z, :, :], "vi_vs_mpi_pricing_diff.png", "MPI - VI pricing q"),
        ((d_mpi[:, :, mid_z] != d_vi[:, :, mid_z]).astype(float), "vi_vs_mpi_default_mismatch.png", "VI vs MPI default mismatch"),
    ]:
        plt.figure(figsize=(7, 5))
        plt.imshow(
            arr.T,
            origin="lower",
            aspect="auto",
            extent=[k_grid[0], k_grid[-1], b_grid[0], b_grid[-1]],
        )
        plt.colorbar()
        plt.xlabel("k")
        plt.ylabel("b")
        plt.title(f"{title} at z[{mid_z}]")
        _savefig_both(os.path.join(out_dir, fname))

    return metrics

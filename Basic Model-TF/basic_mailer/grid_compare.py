# Src/grid_compare.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Callable

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class CompareResult:
    rmse_policy: float
    sup_policy: float
    rmse_value: Optional[float] = None
    sup_value: Optional[float] = None
    welfare_loss_mean: Optional[float] = None
    welfare_loss_sup: Optional[float] = None


def _find_bracketing(
    x_grid: np.ndarray, x: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each x, find i0,i1 such that x_grid[i0] <= x <= x_grid[i1].
    Returns i0,i1,weight where weight in [0,1] for linear interpolation.
    """
    x = np.clip(x, x_grid[0], x_grid[-1])
    i1 = np.searchsorted(x_grid, x, side="right")
    i1 = np.clip(i1, 1, len(x_grid) - 1)
    i0 = i1 - 1

    x0 = x_grid[i0]
    x1 = x_grid[i1]
    w = (x - x0) / (x1 - x0 + 1e-15)
    return i0, i1, w


def interp_grid_2d(
    k_grid: np.ndarray,
    z_grid: np.ndarray,
    F: np.ndarray,
    k: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """
    Bilinear interpolation on (k,z) where z_grid is in levels.
    F shape: (Nk,Nz)
    """
    i0, i1, wk = _find_bracketing(k_grid, k)
    m0, m1, wz = _find_bracketing(z_grid, z)

    F00 = F[i0, m0]
    F01 = F[i0, m1]
    F10 = F[i1, m0]
    F11 = F[i1, m1]

    Fk0 = (1 - wk) * F00 + wk * F10
    Fk1 = (1 - wk) * F01 + wk * F11
    return (1 - wz) * Fk0 + wz * Fk1


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def supnorm(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def nn_policy_on_numpy(policy_nn, k: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    policy_nn expects tf input [N,2].
    pass numpy and call .numpy() on output.
    """
    import tensorflow as tf

    x = tf.convert_to_tensor(np.stack([k, z], axis=1), dtype=tf.float32)
    k_next = policy_nn(x).numpy()
    return k_next


def nn_value_on_numpy(value_nn, k: np.ndarray, z: np.ndarray) -> np.ndarray:
    import tensorflow as tf

    x = tf.convert_to_tensor(np.stack([k, z], axis=1), dtype=tf.float32)
    v = value_nn(x).numpy()
    return v


def nearest_action_index(k_grid: np.ndarray, k_next: np.ndarray) -> np.ndarray:
    """
    Map a continuous k_next to nearest grid index j.
    """
    j = np.searchsorted(k_grid, k_next)
    j = np.clip(j, 1, len(k_grid) - 1)
    left = k_grid[j - 1]
    right = k_grid[j]
    choose_left = (k_next - left) < (right - k_next)
    j = np.where(choose_left, j - 1, j)
    return j.astype(int)


def policy_evaluation_given_policy_idx(
    u: np.ndarray,
    Pz: np.ndarray,
    beta: float,
    policy_idx: np.ndarray,
    n_sweeps: int = 2000,
) -> np.ndarray:
    """
    Compute V^{pi} on the grid by iterative policy evaluation sweeps:
        V <- T^{pi} V
    u: (Nk,Nz,Nk)
    policy_idx: (Nk,Nz)
    Returns V_pi: (Nk,Nz)
    """
    Nk, Nz = policy_idx.shape
    V = np.zeros((Nk, Nz))

    for _ in range(n_sweeps):
        EV = V @ Pz.T
        V_new = np.empty_like(V)
        for m in range(Nz):
            j = policy_idx[:, m]
            u_pi = u[np.arange(Nk), m, j]
            V_new[:, m] = u_pi + beta * EV[j, m]
        V = V_new

    return V


def compare_on_ergodic_states(
    policy_nn,
    value_nn_or_none,
    k_erg: np.ndarray,
    z_erg: np.ndarray,
    k_grid: np.ndarray,
    z_grid: np.ndarray,
    V_star: np.ndarray,
    policy_star: np.ndarray,
    u: np.ndarray,
    Pz: np.ndarray,
    beta: float,
    out_dir: str,
    tag: str,
) -> Tuple[CompareResult, Dict[str, float]]:
    """
    Step 3:
    - Evaluate NN (policy + optional value) on ergodic states
    - Interpolate benchmark (V*, pi*) to same states
    - Metrics: RMSE/sup for policy, value
    - Welfare loss: V*(s) - V^{pi_NN}(s), where V^{pi_NN} is computed by policy evaluation on grid
      using NN policy mapped to nearest action index on grid.
    - Plots: policy curves, value curves (if value_nn exists), and difference curves.
    """
    import os

    os.makedirs(out_dir, exist_ok=True)

    # NN on ergodic
    k_next_nn = nn_policy_on_numpy(policy_nn, k_erg, z_erg)

    # benchmark interpolation on ergodic
    k_next_star = interp_grid_2d(k_grid, z_grid, policy_star, k_erg, z_erg)

    # policy errors
    rmse_pi = rmse(k_next_nn, k_next_star)
    sup_pi = supnorm(k_next_nn, k_next_star)

    # value comparison if value NN exists (Obj3)
    rmse_V = None
    sup_V = None
    if value_nn_or_none is not None:
        V_nn = nn_value_on_numpy(value_nn_or_none, k_erg, z_erg)
        V_star_interp = interp_grid_2d(k_grid, z_grid, V_star, k_erg, z_erg)
        rmse_V = rmse(V_nn, V_star_interp)
        sup_V = supnorm(V_nn, V_star_interp)

    # Welfare loss:
    # 1) Map NN policy to grid action index at each grid state (i,m)
    Nk, Nz = len(k_grid), len(z_grid)
    k_mesh = np.repeat(k_grid[:, None], Nz, axis=1)  # (Nk,Nz)
    z_mesh = np.repeat(z_grid[None, :], Nk, axis=0)  # (Nk,Nz)
    k_flat = k_mesh.reshape(-1)
    z_flat = z_mesh.reshape(-1)
    k_next_flat = nn_policy_on_numpy(policy_nn, k_flat, z_flat)
    j_flat = nearest_action_index(k_grid, k_next_flat)
    policy_idx_nn = j_flat.reshape(Nk, Nz)

    # 2) Evaluate V^{pi_NN} on the grid
    V_pi_nn = policy_evaluation_given_policy_idx(
        u, Pz, beta, policy_idx_nn, n_sweeps=2000
    )

    # 3) welfare loss on ergodic states
    V_star_s = interp_grid_2d(k_grid, z_grid, V_star, k_erg, z_erg)
    V_pi_s = interp_grid_2d(k_grid, z_grid, V_pi_nn, k_erg, z_erg)
    welfare_loss = V_star_s - V_pi_s
    wl_mean = float(np.mean(welfare_loss))
    wl_sup = float(np.max(np.abs(welfare_loss)))

    # ---- Plots (policy + diff)
    import matplotlib.pyplot as plt

    # policy: compare slices at a few z nodes
    zs_to_plot = [0, Nz // 2, Nz - 1]
    k_dense = np.linspace(k_grid[0], k_grid[-1], 400)

    plt.figure(figsize=(7, 5))
    for m in zs_to_plot:
        z0 = z_grid[m]
        z_vec = np.full_like(k_dense, z0)
        nn_line = nn_policy_on_numpy(policy_nn, k_dense, z_vec)
        star_line = np.interp(k_dense, k_grid, policy_star[:, m])
        plt.plot(k_dense, star_line, label=f"Grid z[{m}]")
        plt.plot(k_dense, nn_line, "--", label=f"NN z[{m}]")
    plt.xlabel("k")
    plt.ylabel("k'")
    plt.title(f"Policy: NN vs Grid ({tag})")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(out_dir, f"policy_nn_vs_grid_{tag}.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # policy diff curves
    plt.figure(figsize=(7, 5))
    for m in zs_to_plot:
        z0 = z_grid[m]
        z_vec = np.full_like(k_dense, z0)
        nn_line = nn_policy_on_numpy(policy_nn, k_dense, z_vec)
        star_line = np.interp(k_dense, k_grid, policy_star[:, m])
        plt.plot(k_dense, nn_line - star_line, label=f"diff z[{m}]")
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("k")
    plt.ylabel("k'_NN - k'_Grid")
    plt.title(f"Policy difference ({tag})")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(out_dir, f"policy_diff_{tag}.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()

    # value plots if available
    if value_nn_or_none is not None:
        plt.figure(figsize=(7, 5))
        for m in zs_to_plot:
            star_line = V_star[:, m]
            nn_line = nn_value_on_numpy(
                value_nn_or_none, k_grid, np.full_like(k_grid, z_grid[m])
            )
            plt.plot(k_grid, star_line, label=f"Grid z[{m}]")
            plt.plot(k_grid, nn_line, "--", label=f"NN z[{m}]")
        plt.xlabel("k")
        plt.ylabel("V(k,z)")
        plt.title(f"Value: NN vs Grid ({tag})")
        plt.grid(True)
        plt.legend()
        plt.savefig(
            os.path.join(out_dir, f"value_nn_vs_grid_{tag}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        plt.figure(figsize=(7, 5))
        for m in zs_to_plot:
            star_line = V_star[:, m]
            nn_line = nn_value_on_numpy(
                value_nn_or_none, k_grid, np.full_like(k_grid, z_grid[m])
            )
            plt.plot(k_grid, nn_line - star_line, label=f"diff z[{m}]")
        plt.axhline(0.0, linewidth=1.0)
        plt.xlabel("k")
        plt.ylabel("V_NN - V_Grid")
        plt.title(f"Value difference ({tag})")
        plt.grid(True)
        plt.legend()
        plt.savefig(
            os.path.join(out_dir, f"value_diff_{tag}.png"), dpi=150, bbox_inches="tight"
        )
        plt.close()

    # welfare loss histogram
    plt.figure(figsize=(7, 4))
    plt.hist(welfare_loss, bins=60)
    plt.xlabel("WelfareLoss = V* - V^{pi_NN}")
    plt.ylabel("count")
    plt.title(f"Welfare loss distribution ({tag})")
    plt.grid(True)
    plt.savefig(
        os.path.join(out_dir, f"welfare_loss_hist_{tag}.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    result = CompareResult(
        rmse_policy=rmse_pi,
        sup_policy=sup_pi,
        rmse_value=rmse_V,
        sup_value=sup_V,
        welfare_loss_mean=wl_mean,
        welfare_loss_sup=wl_sup,
    )

    summary = {
        "rmse_policy": rmse_pi,
        "sup_policy": sup_pi,
        "rmse_value": (rmse_V if rmse_V is not None else float("nan")),
        "sup_value": (sup_V if sup_V is not None else float("nan")),
        "welfare_loss_mean": wl_mean,
        "welfare_loss_sup": wl_sup,
    }

    return result, summary

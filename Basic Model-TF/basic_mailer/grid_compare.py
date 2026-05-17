"""Benchmark-versus-neural-network comparison utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class CompareResult:
    """Store comparison metrics and supporting arrays."""
    rmse_policy: float
    sup_policy: float
    rmse_value_induced: Optional[float] = None
    sup_value_induced: Optional[float] = None
    rmse_value_direct: Optional[float] = None
    sup_value_direct: Optional[float] = None
    welfare_loss_mean: Optional[float] = None
    welfare_loss_sup: Optional[float] = None


def _as_1d(a) -> np.ndarray:
    """Convert an input array-like object to a flattened one-dimensional array."""
    return np.asarray(a, dtype=float).reshape(-1)


def _find_bracketing(
    x_grid: np.ndarray, x: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each x, find i0,i1 such that x_grid[i0] <= x <= x_grid[i1].
    Returns i0,i1,weight where weight in [0,1] for linear interpolation.
    """
    x = np.clip(np.asarray(x, dtype=float), float(x_grid[0]), float(x_grid[-1]))
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
    k = _as_1d(k)
    z = _as_1d(z)
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
    """Compute the root-mean-square error between two arrays."""
    a = _as_1d(a)
    b = _as_1d(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def supnorm(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the sup-norm distance between two arrays."""
    a = _as_1d(a)
    b = _as_1d(b)
    return float(np.max(np.abs(a - b)))


def nn_policy_on_numpy(policy_nn, k: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    policy_nn expects tf input [N,2].
    We pass numpy and call .numpy() on output.
    """
    import tensorflow as tf

    k = _as_1d(k)
    z = _as_1d(z)
    x = tf.convert_to_tensor(np.stack([k, z], axis=1), dtype=tf.float32)
    k_next = policy_nn(x).numpy()
    return _as_1d(k_next)


def nn_value_on_numpy(value_nn, k: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Evaluate the neural-network comparison value object on NumPy states."""
    import tensorflow as tf

    k = _as_1d(k)
    z = _as_1d(z)
    x = tf.convert_to_tensor(np.stack([k, z], axis=1), dtype=tf.float32)
    v = value_nn(x).numpy()
    return _as_1d(v)


def nearest_action_index(k_grid: np.ndarray, k_next: np.ndarray) -> np.ndarray:
    """
    Map a continuous k_next to nearest grid index j.
    """
    k_next = np.clip(_as_1d(k_next), float(k_grid[0]), float(k_grid[-1]))
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


def benchmark_policy_on_numpy(
    k_grid: np.ndarray,
    z_grid: np.ndarray,
    policy_star: np.ndarray,
    k: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Evaluate the interpolated benchmark policy on NumPy states."""
    return interp_grid_2d(k_grid, z_grid, policy_star, k, z)


def simulate_benchmark_ergodic_dataset(
    policy_star: np.ndarray,
    k_grid: np.ndarray,
    z_grid: np.ndarray,
    mp,
    tp,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a continuous-state ergodic sample under the benchmark policy,
    using bilinear interpolation of the benchmark grid policy.
    """
    import tensorflow as tf
    from .simulation import set_global_seed
    from .primitives import shock_next_z

    set_global_seed(seed)

    n = tp.ergodic_n_paths
    k = tf.random.uniform((n,), 0.5, 2.0, dtype=tf.float32)
    z = tf.random.uniform((n,), 0.5, 2.0, dtype=tf.float32)

    ks, zs = [], []
    total_steps = tp.ergodic_burn_in + tp.ergodic_T

    for t in range(total_steps + 1):
        if t >= tp.ergodic_burn_in:
            ks.append(k.numpy())
            zs.append(z.numpy())

        k_np = k.numpy()
        z_np = z.numpy()
        k_next_np = benchmark_policy_on_numpy(k_grid, z_grid, policy_star, k_np, z_np)
        k_next = tf.convert_to_tensor(
            np.clip(k_next_np, mp.k_min, mp.k_max), dtype=tf.float32
        )
        z_next = shock_next_z(z, mp.rho, mp.sigma_eps)
        z_next = tf.clip_by_value(z_next, float(z_grid[0]), float(z_grid[-1]))

        k, z = k_next, z_next

    k_mat = np.stack(ks, axis=1)
    z_mat = np.stack(zs, axis=1)
    k_flat = k_mat.reshape(-1)
    z_flat = z_mat.reshape(-1)

    if k_flat.shape[0] > tp.ergodic_buffer_size:
        idx = np.random.choice(
            k_flat.shape[0], size=tp.ergodic_buffer_size, replace=False
        )
        k_flat = k_flat[idx]
        z_flat = z_flat[idx]

    return k_flat, z_flat


def full_grid_state_sample(
    k_grid: np.ndarray,
    z_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten the full benchmark grid into a state sample."""
    k_mesh = np.repeat(k_grid[:, None], len(z_grid), axis=1)
    z_mesh = np.repeat(z_grid[None, :], len(k_grid), axis=0)
    return k_mesh.reshape(-1), z_mesh.reshape(-1)


def _policy_idx_from_nn_on_grid(policy_nn, k_grid: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
    """Evaluate the NN policy on the benchmark grid and map outputs to action indices."""
    Nk, Nz = len(k_grid), len(z_grid)
    k_mesh = np.repeat(k_grid[:, None], Nz, axis=1)
    z_mesh = np.repeat(z_grid[None, :], Nk, axis=0)
    k_next_flat = nn_policy_on_numpy(policy_nn, k_mesh.reshape(-1), z_mesh.reshape(-1))
    j_flat = nearest_action_index(k_grid, k_next_flat)
    return j_flat.reshape(Nk, Nz)


def compare_on_state_sample(
    policy_nn,
    value_nn_or_none,
    k_states: np.ndarray,
    z_states: np.ndarray,
    k_grid: np.ndarray,
    z_grid: np.ndarray,
    V_star: np.ndarray,
    policy_star: np.ndarray,
    u: np.ndarray,
    Pz: np.ndarray,
    beta: float,
    out_dir: str,
    tag: str,
    V_star_alt: Optional[np.ndarray] = None,
    policy_star_alt: Optional[np.ndarray] = None,
    alt_label: str = "Howard PI",
) -> Tuple[CompareResult, Dict[str, float]]:
    """
    Compare NN against the benchmark on an arbitrary state sample.

    For Obj1 / Obj2:
      - policy distance uses phi_NN(s) vs pi*(s)
      - value distance uses induced value V^{pi_NN}(s) vs V*(s)
      - welfare loss uses V*(s) - V^{pi_NN}(s)

    For Obj3:
      - all of the above, plus direct value distance V_NN(s) vs V*(s)
    """
    import os

    os.makedirs(out_dir, exist_ok=True)

    k_states = _as_1d(k_states)
    z_states = _as_1d(z_states)

    # 1) State-sample policy comparison in levels.
    k_next_nn = nn_policy_on_numpy(policy_nn, k_states, z_states)
    k_next_star = interp_grid_2d(k_grid, z_grid, policy_star, k_states, z_states)
    rmse_pi = rmse(k_next_nn, k_next_star)
    sup_pi = supnorm(k_next_nn, k_next_star)

    rmse_pi_alt = None
    sup_pi_alt = None
    k_next_star_alt = None
    if policy_star_alt is not None:
        k_next_star_alt = interp_grid_2d(k_grid, z_grid, policy_star_alt, k_states, z_states)
        rmse_pi_alt = rmse(k_next_nn, k_next_star_alt)
        sup_pi_alt = supnorm(k_next_nn, k_next_star_alt)

    # 2) Evaluate induced value of the NN policy on the benchmark grid.
    policy_idx_nn = _policy_idx_from_nn_on_grid(policy_nn, k_grid, z_grid)
    V_pi_nn = policy_evaluation_given_policy_idx(
        u, Pz, beta, policy_idx_nn, n_sweeps=2000
    )

    V_star_states = interp_grid_2d(k_grid, z_grid, V_star, k_states, z_states)
    V_pi_states = interp_grid_2d(k_grid, z_grid, V_pi_nn, k_states, z_states)

    rmse_V_induced = rmse(V_pi_states, V_star_states)
    sup_V_induced = supnorm(V_pi_states, V_star_states)

    welfare_loss = V_star_states - V_pi_states
    wl_mean = float(np.mean(welfare_loss))
    wl_sup = float(np.max(welfare_loss))

    rmse_V_induced_alt = None
    sup_V_induced_alt = None
    wl_mean_alt = None
    wl_sup_alt = None
    V_star_states_alt = None
    welfare_loss_alt = None
    if V_star_alt is not None:
        V_star_states_alt = interp_grid_2d(k_grid, z_grid, V_star_alt, k_states, z_states)
        rmse_V_induced_alt = rmse(V_pi_states, V_star_states_alt)
        sup_V_induced_alt = supnorm(V_pi_states, V_star_states_alt)
        welfare_loss_alt = V_star_states_alt - V_pi_states
        wl_mean_alt = float(np.mean(welfare_loss_alt))
        wl_sup_alt = float(np.max(welfare_loss_alt))

    # 3) Direct value-net comparison, only when a value network exists.
    rmse_V_direct = None
    sup_V_direct = None
    rmse_V_direct_alt = None
    sup_V_direct_alt = None
    if value_nn_or_none is not None:
        V_nn_states = nn_value_on_numpy(value_nn_or_none, k_states, z_states)
        rmse_V_direct = rmse(V_nn_states, V_star_states)
        sup_V_direct = supnorm(V_nn_states, V_star_states)
        if V_star_alt is not None and V_star_states_alt is not None:
            rmse_V_direct_alt = rmse(V_nn_states, V_star_states_alt)
            sup_V_direct_alt = supnorm(V_nn_states, V_star_states_alt)

    # ---- Plots ----
    Nz = len(z_grid)
    zs_to_plot = sorted(set([0, Nz // 2, Nz - 1]))
    k_dense = np.linspace(k_grid[0], k_grid[-1], 400)

    # policy curves
    plt.figure(figsize=(7, 5))
    for m in zs_to_plot:
        z0 = z_grid[m]
        z_vec = np.full_like(k_dense, z0)
        nn_line = nn_policy_on_numpy(policy_nn, k_dense, z_vec)
        star_line = np.interp(k_dense, k_grid, policy_star[:, m])
        plt.plot(k_dense, star_line, label=f"VFI z[{m}]")
        if policy_star_alt is not None:
            star_line_alt = np.interp(k_dense, k_grid, policy_star_alt[:, m])
            plt.plot(k_dense, star_line_alt, ":", label=f"{alt_label} z[{m}]")
        plt.plot(k_dense, nn_line, "--", label=f"NN z[{m}]")
    plt.xlabel("k")
    plt.ylabel("k'")
    plt.title(f"Policy: NN vs VFI / {alt_label} ({tag})")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"policy_nn_vs_grid_{tag}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 5))
    for m in zs_to_plot:
        z0 = z_grid[m]
        z_vec = np.full_like(k_dense, z0)
        nn_line = nn_policy_on_numpy(policy_nn, k_dense, z_vec)
        star_line = np.interp(k_dense, k_grid, policy_star[:, m])
        plt.plot(k_dense, nn_line - star_line, label=f"NN-VFI z[{m}]")
        if policy_star_alt is not None:
            star_line_alt = np.interp(k_dense, k_grid, policy_star_alt[:, m])
            plt.plot(k_dense, nn_line - star_line_alt, ":", label=f"NN-{alt_label} z[{m}]")
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("k")
    plt.ylabel("Policy difference")
    plt.title(f"Policy difference ({tag})")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"policy_diff_{tag}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # induced value curves, always available
    plt.figure(figsize=(7, 5))
    for m in zs_to_plot:
        plt.plot(k_grid, V_star[:, m], label=f"V*_VFI z[{m}]")
        if V_star_alt is not None:
            plt.plot(k_grid, V_star_alt[:, m], ":", label=f"V*_{alt_label} z[{m}]")
        plt.plot(k_grid, V_pi_nn[:, m], "--", label=f"V^pi_NN z[{m}]")
    plt.xlabel("k")
    plt.ylabel("Value")
    plt.title(f"Induced value: V^pi_NN vs VFI / {alt_label} ({tag})")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"induced_value_vs_grid_{tag}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 5))
    for m in zs_to_plot:
        plt.plot(k_grid, V_pi_nn[:, m] - V_star[:, m], label=f"NN-VFI z[{m}]")
        if V_star_alt is not None:
            plt.plot(k_grid, V_pi_nn[:, m] - V_star_alt[:, m], ":", label=f"NN-{alt_label} z[{m}]")
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("k")
    plt.ylabel("V^pi_NN - V*")
    plt.title(f"Induced value difference ({tag})")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"induced_value_diff_{tag}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 5))
    for m in zs_to_plot:
        plt.plot(k_grid, V_star[:, m] - V_pi_nn[:, m], label=f"loss vs VFI z[{m}]")
        if V_star_alt is not None:
            plt.plot(k_grid, V_star_alt[:, m] - V_pi_nn[:, m], ":", label=f"loss vs {alt_label} z[{m}]")
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("k")
    plt.ylabel("V* - V^pi_NN")
    plt.title(f"Welfare loss curves ({tag})")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"welfare_loss_curves_{tag}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # direct value-net plots for Obj3 only
    if value_nn_or_none is not None:
        plt.figure(figsize=(7, 5))
        for m in zs_to_plot:
            nn_line = nn_value_on_numpy(value_nn_or_none, k_grid, np.full_like(k_grid, z_grid[m]))
            plt.plot(k_grid, V_star[:, m], label=f"VFI z[{m}]")
            if V_star_alt is not None:
                plt.plot(k_grid, V_star_alt[:, m], ":", label=f"{alt_label} z[{m}]")
            plt.plot(k_grid, nn_line, "--", label=f"NN z[{m}]")
        plt.xlabel("k")
        plt.ylabel("V(k,z)")
        plt.title(f"Direct value: V_NN vs VFI / {alt_label} ({tag})")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"value_nn_vs_grid_{tag}.png"), dpi=150, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(7, 5))
        for m in zs_to_plot:
            nn_line = nn_value_on_numpy(value_nn_or_none, k_grid, np.full_like(k_grid, z_grid[m]))
            plt.plot(k_grid, nn_line - V_star[:, m], label=f"NN-VFI z[{m}]")
            if V_star_alt is not None:
                plt.plot(k_grid, nn_line - V_star_alt[:, m], ":", label=f"NN-{alt_label} z[{m}]")
        plt.axhline(0.0, linewidth=1.0)
        plt.xlabel("k")
        plt.ylabel("V_NN - V*")
        plt.title(f"Direct value difference ({tag})")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"value_diff_{tag}.png"), dpi=150, bbox_inches="tight")
        plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(welfare_loss, bins=60, alpha=0.6, label="vs VFI")
    if welfare_loss_alt is not None:
        plt.hist(welfare_loss_alt, bins=60, alpha=0.6, label=f"vs {alt_label}")
    plt.xlabel("Welfare loss = V* - V^pi_NN")
    plt.ylabel("count")
    plt.title(f"Welfare loss distribution ({tag})")
    plt.grid(True)
    if welfare_loss_alt is not None:
        plt.legend()
    plt.savefig(os.path.join(out_dir, f"welfare_loss_hist_{tag}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    result = CompareResult(
        rmse_policy=rmse_pi,
        sup_policy=sup_pi,
        rmse_value_induced=rmse_V_induced,
        sup_value_induced=sup_V_induced,
        rmse_value_direct=rmse_V_direct,
        sup_value_direct=sup_V_direct,
        welfare_loss_mean=wl_mean,
        welfare_loss_sup=wl_sup,
    )

    summary = {
        "rmse_policy": rmse_pi,
        "sup_policy": sup_pi,
        "rmse_value_induced": rmse_V_induced,
        "sup_value_induced": sup_V_induced,
        "rmse_value_direct": (rmse_V_direct if rmse_V_direct is not None else float("nan")),
        "sup_value_direct": (sup_V_direct if sup_V_direct is not None else float("nan")),
        "welfare_loss_mean": wl_mean,
        "welfare_loss_sup": wl_sup,
        # legacy aliases so old downstream readers do not break immediately.
        "rmse_value": rmse_V_induced,
        "sup_value": sup_V_induced,
    }
    if rmse_pi_alt is not None:
        summary.update({
            "rmse_policy_howard_pi": rmse_pi_alt,
            "sup_policy_howard_pi": sup_pi_alt,
            "rmse_value_induced_howard_pi": rmse_V_induced_alt,
            "sup_value_induced_howard_pi": sup_V_induced_alt,
            "welfare_loss_mean_howard_pi": wl_mean_alt,
            "welfare_loss_sup_howard_pi": wl_sup_alt,
        })
        if rmse_V_direct_alt is not None:
            summary.update({
                "rmse_value_direct_howard_pi": rmse_V_direct_alt,
                "sup_value_direct_howard_pi": sup_V_direct_alt,
            })

    return result, summary


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
    """Backward-compatible wrapper around compare_on_state_sample."""
    return compare_on_state_sample(
        policy_nn=policy_nn,
        value_nn_or_none=value_nn_or_none,
        k_states=k_erg,
        z_states=z_erg,
        k_grid=k_grid,
        z_grid=z_grid,
        V_star=V_star,
        policy_star=policy_star,
        u=u,
        Pz=Pz,
        beta=beta,
        out_dir=out_dir,
        tag=tag,
    )

# risky_debt/grid_compare.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np


@dataclass(frozen=True)
class CompareResultRD:
    # policy errors (ergodic states)
    rmse_kp: float
    sup_kp: float
    rmse_bp: float
    sup_bp: float

    # value errors (ergodic states), optional if value_nn is provided
    rmse_V: Optional[float] = None
    sup_V: Optional[float] = None

    # welfare loss: V* - V^{pi_NN} on ergodic states
    welfare_loss_mean: Optional[float] = None
    welfare_loss_sup: Optional[float] = None


# -----------------------------
# Interpolation helpers (3D)
# -----------------------------
def _find_bracketing_1d(
    grid: np.ndarray, x: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each x, find i0,i1 such that grid[i0] <= x <= grid[i1].
    Returns i0,i1,w where w in [0,1] for linear interpolation.
    """
    x = np.asarray(x)
    x = np.clip(x, grid[0], grid[-1])
    i1 = np.searchsorted(grid, x, side="right")
    i1 = np.clip(i1, 1, len(grid) - 1)
    i0 = i1 - 1

    x0 = grid[i0]
    x1 = grid[i1]
    w = (x - x0) / (x1 - x0 + 1e-15)
    return i0, i1, w


def interp_grid_3d(
    k_grid: np.ndarray,
    b_grid: np.ndarray,
    z_grid: np.ndarray,
    F: np.ndarray,
    k: np.ndarray,
    b: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """
    Trilinear interpolation on (k,b,z) with all grids in levels.
    F shape must be (Nk, Nb, Nz).
    Returns array shape (N,).
    """
    k = np.asarray(k)
    b = np.asarray(b)
    z = np.asarray(z)
    assert F.ndim == 3

    i0, i1, wk = _find_bracketing_1d(k_grid, k)
    j0, j1, wb = _find_bracketing_1d(b_grid, b)
    l0, l1, wz = _find_bracketing_1d(z_grid, z)

    # corner values: 8 corners
    F000 = F[i0, j0, l0]
    F001 = F[i0, j0, l1]
    F010 = F[i0, j1, l0]
    F011 = F[i0, j1, l1]
    F100 = F[i1, j0, l0]
    F101 = F[i1, j0, l1]
    F110 = F[i1, j1, l0]
    F111 = F[i1, j1, l1]

    # interpolate along k
    F00 = (1 - wk) * F000 + wk * F100
    F01 = (1 - wk) * F001 + wk * F101
    F10 = (1 - wk) * F010 + wk * F110
    F11 = (1 - wk) * F011 + wk * F111

    # interpolate along b
    F0 = (1 - wb) * F00 + wb * F10
    F1 = (1 - wb) * F01 + wb * F11

    # interpolate along z
    out = (1 - wz) * F0 + wz * F1
    return out


# -----------------------------
# Metrics
# -----------------------------
def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def supnorm(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.max(np.abs(a - b)))


# -----------------------------
# NN wrappers (numpy -> tf -> numpy)
# -----------------------------
def nn_policy_on_numpy(
    policy_nn, k: np.ndarray, b: np.ndarray, z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    policy_nn expects tf input [N,3] with columns [k,b,z].
    Outputs [N,2] => (k',b').
    """
    import tensorflow as tf

    x = tf.convert_to_tensor(np.stack([k, b, z], axis=1), dtype=tf.float32)
    kb_next = policy_nn(x).numpy()
    return kb_next[:, 0], kb_next[:, 1]


def nn_value_on_numpy(
    value_nn, k: np.ndarray, b: np.ndarray, z: np.ndarray
) -> np.ndarray:
    """
    value_nn expects tf input [N,3] => returns [N] (or [N,] after squeeze).
    """
    import tensorflow as tf

    x = tf.convert_to_tensor(np.stack([k, b, z], axis=1), dtype=tf.float32)
    v = value_nn(x).numpy()
    return v.reshape(-1)


# -----------------------------
# Nearest-neighbor map to grid indices (for policy eval)
# -----------------------------
def nearest_index_1d(grid: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Map x (continuous) to nearest index in grid.
    """
    x = np.asarray(x)
    j = np.searchsorted(grid, x)
    j = np.clip(j, 1, len(grid) - 1)
    left = grid[j - 1]
    right = grid[j]
    choose_left = (x - left) < (right - x)
    j = np.where(choose_left, j - 1, j)
    return j.astype(int)


def nn_policy_to_grid_indices(
    policy_nn,
    k_grid: np.ndarray,
    b_grid: np.ndarray,
    z_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate NN policy on every grid state (k_i,b_j,z_l),
    then map (k',b') to nearest action indices (m,n) on (k_grid,b_grid).
    Returns:
      m_idx shape (Nk,Nb,Nz), n_idx shape (Nk,Nb,Nz)
    """
    Nk, Nb, Nz = len(k_grid), len(b_grid), len(z_grid)

    # Mesh all grid states
    k_mesh = np.repeat(k_grid[:, None, None], Nb, axis=1)
    k_mesh = np.repeat(k_mesh, Nz, axis=2)

    b_mesh = np.repeat(b_grid[None, :, None], Nk, axis=0)
    b_mesh = np.repeat(b_mesh, Nz, axis=2)

    z_mesh = np.repeat(z_grid[None, None, :], Nk, axis=0)
    z_mesh = np.repeat(z_mesh, Nb, axis=1)

    k_flat = k_mesh.reshape(-1)
    b_flat = b_mesh.reshape(-1)
    z_flat = z_mesh.reshape(-1)

    kp_flat, bp_flat = nn_policy_on_numpy(policy_nn, k_flat, b_flat, z_flat)

    m_flat = nearest_index_1d(k_grid, kp_flat)
    n_flat = nearest_index_1d(b_grid, bp_flat)

    m_idx = m_flat.reshape(Nk, Nb, Nz)
    n_idx = n_flat.reshape(Nk, Nb, Nz)
    return m_idx, n_idx


# -----------------------------
# One-period payout d on the grid (uses your primitives)
# -----------------------------
def _equity_payout_d_numpy(
    k: np.ndarray,
    k_next: np.ndarray,
    b: np.ndarray,
    b_next: np.ndarray,
    z: np.ndarray,
    q: np.ndarray,
    mp,
    kappa_issue: float,
) -> np.ndarray:
    """
    Compute d = equity payout using your primitives (TensorFlow), but return numpy.
    We keep this as a thin adapter to guarantee consistency with the model code.
    """
    import tensorflow as tf
    from risky_debt.primitives import equity_payout_d

    k_tf = tf.convert_to_tensor(k, tf.float32)
    k_next_tf = tf.convert_to_tensor(k_next, tf.float32)
    b_tf = tf.convert_to_tensor(b, tf.float32)
    b_next_tf = tf.convert_to_tensor(b_next, tf.float32)
    z_tf = tf.convert_to_tensor(z, tf.float32)
    q_tf = tf.convert_to_tensor(q, tf.float32)

    d_tf = equity_payout_d(
        k_tf, k_next_tf, b_tf, b_next_tf, z_tf, q_tf, mp, kappa_issue=kappa_issue
    )
    return d_tf.numpy().reshape(-1)


# -----------------------------
# Policy evaluation on the grid (fixed policy, limited liability)
# -----------------------------
def policy_evaluation_fixed_policy(
    k_grid: np.ndarray,
    b_grid: np.ndarray,
    z_grid: np.ndarray,
    Pz: np.ndarray,
    beta: float,
    policy_m_idx: np.ndarray,
    policy_n_idx: np.ndarray,
    q_star: np.ndarray,
    mp,
    kappa_issue: float,
    n_sweeps: int = 2000,
    tol: float = 1e-10,
) -> np.ndarray:
    """
    Compute V^{pi} on the full grid under a fixed policy (m_idx,n_idx),
    using the *limited liability* Bellman update:

      V(s) = max(0, d(s, pi(s)) + beta * E[V(s')])

    with pricing q taken from benchmark schedule q_star[z, m, n] at issuance nodes (z, k'_m, b'_n).

    Shapes:
      policy_m_idx, policy_n_idx: (Nk,Nb,Nz)
      q_star: (Nz, Nk, Nb)  (i.e. q_star[l, m, n])

    Returns V_pi: (Nk,Nb,Nz)
    """
    Nk, Nb, Nz = len(k_grid), len(b_grid), len(z_grid)
    assert policy_m_idx.shape == (Nk, Nb, Nz)
    assert policy_n_idx.shape == (Nk, Nb, Nz)
    assert q_star.shape == (Nz, Nk, Nb)
    assert Pz.shape == (Nz, Nz)

    # initialize
    V = np.zeros((Nk, Nb, Nz), dtype=np.float64)

    # Precompute state grids (for vectorized payoff eval)
    k_state = np.repeat(k_grid[:, None, None], Nb, axis=1)
    k_state = np.repeat(k_state, Nz, axis=2)

    b_state = np.repeat(b_grid[None, :, None], Nk, axis=0)
    b_state = np.repeat(b_state, Nz, axis=2)

    z_state = np.repeat(z_grid[None, None, :], Nk, axis=0)
    z_state = np.repeat(z_state, Nb, axis=1)

    for it in range(n_sweeps):
        # Expected continuation value:
        # EV(i,j,l) = sum_{l'} P[l,l'] * V(i',j',l')
        # But next indices depend on policy, so we’ll gather V at (m,n) then take expectation over z'.
        V_new = np.empty_like(V)

        # We loop over current shock l to keep it readable and safe.
        # (You can vectorize later if you want.)
        for l in range(Nz):
            m_idx = policy_m_idx[:, :, l]  # (Nk,Nb)
            n_idx = policy_n_idx[:, :, l]  # (Nk,Nb)

            # Map to next-state levels
            k_next = k_grid[m_idx]
            b_next = b_grid[n_idx]

            # Pricing q at issuance node uses current z index l and action indices (m,n)
            q = q_star[l, m_idx, n_idx]

            # One-period payout d at (k,b,z) choosing (k',b')
            d = _equity_payout_d_numpy(
                k=k_state[:, :, l].reshape(-1),
                k_next=k_next.reshape(-1),
                b=b_state[:, :, l].reshape(-1),
                b_next=b_next.reshape(-1),
                z=z_state[:, :, l].reshape(-1),
                q=q.reshape(-1),
                mp=mp,
                kappa_issue=kappa_issue,
            ).reshape(Nk, Nb)

            # continuation: beta * sum_{l'} P[l,l'] * V(m_idx,n_idx,l')
            cont = np.zeros((Nk, Nb), dtype=np.float64)
            for lp in range(Nz):
                cont += Pz[l, lp] * V[m_idx, n_idx, lp]

            C = d + beta * cont
            V_new[:, :, l] = np.maximum(0.0, C)

        diff = np.max(np.abs(V_new - V))
        V = V_new
        if diff < tol:
            break

    return V


# -----------------------------
# Main Step-3 compare on ergodic states
# -----------------------------
def compare_on_ergodic_states_risky_debt(
    *,
    policy_nn,
    value_nn_or_none,
    k_erg: np.ndarray,
    b_erg: np.ndarray,
    z_erg: np.ndarray,
    # benchmark objects
    k_grid: np.ndarray,
    b_grid: np.ndarray,
    z_grid: np.ndarray,
    V_star: np.ndarray,  # (Nk,Nb,Nz)
    policy_kp_star: np.ndarray,  # (Nk,Nb,Nz)
    policy_bp_star: np.ndarray,  # (Nk,Nb,Nz)
    q_star: np.ndarray,  # (Nz,Nk,Nb)
    Pz: np.ndarray,  # (Nz,Nz)
    beta: float,
    mp,
    kappa_issue: float,
    out_dir: str,
    tag: str,
    n_policy_eval_sweeps: int = 2000,
) -> Tuple[CompareResultRD, Dict[str, float]]:
    """
    Step 3 (full version):
      1) Evaluate NN policy (+ optional NN value) on ergodic states
      2) Trilinear-interpolate benchmark policy/value onto the same ergodic states
      3) Compute policy RMSE/sup (k',b'), value RMSE/sup (if applicable)
      4) Welfare loss:
           - map NN policy to grid indices (m,n) at each grid state
           - compute V^{pi_NN} by fixed-policy evaluation with limited liability
           - compare on ergodic states: welfare_loss = V_star - V_pi_NN
      5) Produce plots:
           - policy slices vs benchmark (k' and b')
           - policy difference slices
           - value slices vs benchmark (if value_nn_or_none)
           - value difference slices
           - scatter plots on ergodic points
           - welfare loss histogram
    """
    import os
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    k_erg = np.asarray(k_erg).reshape(-1)
    b_erg = np.asarray(b_erg).reshape(-1)
    z_erg = np.asarray(z_erg).reshape(-1)

    # 1) NN outputs on ergodic
    kp_nn, bp_nn = nn_policy_on_numpy(policy_nn, k_erg, b_erg, z_erg)

    # 2) benchmark interpolation on ergodic
    kp_star = interp_grid_3d(
        k_grid, b_grid, z_grid, policy_kp_star, k_erg, b_erg, z_erg
    )
    bp_star = interp_grid_3d(
        k_grid, b_grid, z_grid, policy_bp_star, k_erg, b_erg, z_erg
    )

    # policy errors
    rmse_kp = rmse(kp_nn, kp_star)
    sup_kp = supnorm(kp_nn, kp_star)
    rmse_bp = rmse(bp_nn, bp_star)
    sup_bp = supnorm(bp_nn, bp_star)

    # value errors if value net exists
    rmse_V = None
    sup_V = None
    if value_nn_or_none is not None:
        V_nn = nn_value_on_numpy(value_nn_or_none, k_erg, b_erg, z_erg)
        V_star_interp = interp_grid_3d(
            k_grid, b_grid, z_grid, V_star, k_erg, b_erg, z_erg
        )
        rmse_V = rmse(V_nn, V_star_interp)
        sup_V = supnorm(V_nn, V_star_interp)

    # 3) Welfare loss via policy evaluation on grid
    m_idx, n_idx = nn_policy_to_grid_indices(policy_nn, k_grid, b_grid, z_grid)
    V_pi_nn = policy_evaluation_fixed_policy(
        k_grid=k_grid,
        b_grid=b_grid,
        z_grid=z_grid,
        Pz=Pz,
        beta=beta,
        policy_m_idx=m_idx,
        policy_n_idx=n_idx,
        q_star=q_star,
        mp=mp,
        kappa_issue=kappa_issue,
        n_sweeps=n_policy_eval_sweeps,
    )

    V_star_s = interp_grid_3d(k_grid, b_grid, z_grid, V_star, k_erg, b_erg, z_erg)
    V_pi_s = interp_grid_3d(k_grid, b_grid, z_grid, V_pi_nn, k_erg, b_erg, z_erg)
    welfare_loss = V_star_s - V_pi_s
    wl_mean = float(np.mean(welfare_loss))
    wl_sup = float(np.max(np.abs(welfare_loss)))

    # -----------------------------
    # Plots
    # -----------------------------
    Nk, Nb, Nz = len(k_grid), len(b_grid), len(z_grid)

    # Choose a few slices
    z_idx_list = [0, Nz // 2, Nz - 1] if Nz >= 3 else list(range(Nz))
    b_idx_list = [0, Nb // 2, Nb - 1] if Nb >= 3 else list(range(Nb))

    # dense k line for smooth curves
    k_dense = np.linspace(k_grid[0], k_grid[-1], 400)

    # (A) Policy k' slices
    for j in b_idx_list:
        b0 = b_grid[j]
        plt.figure(figsize=(8, 5))
        for l in z_idx_list:
            z0 = z_grid[l]
            b_vec = np.full_like(k_dense, b0)
            z_vec = np.full_like(k_dense, z0)
            kp_line_nn, _ = nn_policy_on_numpy(policy_nn, k_dense, b_vec, z_vec)

            # benchmark slice: evaluate by interpolation along k with fixed (b0,z0)
            kp_line_star = interp_grid_3d(
                k_grid, b_grid, z_grid, policy_kp_star, k_dense, b_vec, z_vec
            )

            plt.plot(k_dense, kp_line_star, label=f"Grid z[{l}]")
            plt.plot(k_dense, kp_line_nn, "--", label=f"NN z[{l}]")

        plt.xlabel("k")
        plt.ylabel("k'")
        plt.title(f"Policy k': NN vs Grid ({tag}), b={b0:.3g}")
        plt.grid(True)
        plt.legend()
        plt.savefig(
            os.path.join(out_dir, f"policy_kp_nn_vs_grid_{tag}_b{j}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        # diff
        plt.figure(figsize=(8, 5))
        for l in z_idx_list:
            z0 = z_grid[l]
            b_vec = np.full_like(k_dense, b0)
            z_vec = np.full_like(k_dense, z0)
            kp_line_nn, _ = nn_policy_on_numpy(policy_nn, k_dense, b_vec, z_vec)
            kp_line_star = interp_grid_3d(
                k_grid, b_grid, z_grid, policy_kp_star, k_dense, b_vec, z_vec
            )
            plt.plot(k_dense, kp_line_nn - kp_line_star, label=f"diff z[{l}]")
        plt.axhline(0.0, linewidth=1.0)
        plt.xlabel("k")
        plt.ylabel("k'_NN - k'_Grid")
        plt.title(f"Policy k' difference ({tag}), b={b0:.3g}")
        plt.grid(True)
        plt.legend()
        plt.savefig(
            os.path.join(out_dir, f"policy_kp_diff_{tag}_b{j}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    # (B) Policy b' slices
    for j in b_idx_list:
        b0 = b_grid[j]
        plt.figure(figsize=(8, 5))
        for l in z_idx_list:
            z0 = z_grid[l]
            b_vec = np.full_like(k_dense, b0)
            z_vec = np.full_like(k_dense, z0)
            _, bp_line_nn = nn_policy_on_numpy(policy_nn, k_dense, b_vec, z_vec)
            bp_line_star = interp_grid_3d(
                k_grid, b_grid, z_grid, policy_bp_star, k_dense, b_vec, z_vec
            )
            plt.plot(k_dense, bp_line_star, label=f"Grid z[{l}]")
            plt.plot(k_dense, bp_line_nn, "--", label=f"NN z[{l}]")
        plt.xlabel("k")
        plt.ylabel("b'")
        plt.title(f"Policy b': NN vs Grid ({tag}), b={b0:.3g}")
        plt.grid(True)
        plt.legend()
        plt.savefig(
            os.path.join(out_dir, f"policy_bp_nn_vs_grid_{tag}_b{j}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        # diff
        plt.figure(figsize=(8, 5))
        for l in z_idx_list:
            z0 = z_grid[l]
            b_vec = np.full_like(k_dense, b0)
            z_vec = np.full_like(k_dense, z0)
            _, bp_line_nn = nn_policy_on_numpy(policy_nn, k_dense, b_vec, z_vec)
            bp_line_star = interp_grid_3d(
                k_grid, b_grid, z_grid, policy_bp_star, k_dense, b_vec, z_vec
            )
            plt.plot(k_dense, bp_line_nn - bp_line_star, label=f"diff z[{l}]")
        plt.axhline(0.0, linewidth=1.0)
        plt.xlabel("k")
        plt.ylabel("b'_NN - b'_Grid")
        plt.title(f"Policy b' difference ({tag}), b={b0:.3g}")
        plt.grid(True)
        plt.legend()
        plt.savefig(
            os.path.join(out_dir, f"policy_bp_diff_{tag}_b{j}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    # (C) Value slices (if value net exists)
    if value_nn_or_none is not None:
        for j in b_idx_list:
            b0 = b_grid[j]
            plt.figure(figsize=(8, 5))
            for l in z_idx_list:
                z0 = z_grid[l]
                b_vec = np.full_like(k_grid, b0)
                z_vec = np.full_like(k_grid, z0)

                V_line_star = interp_grid_3d(
                    k_grid, b_grid, z_grid, V_star, k_grid, b_vec, z_vec
                )
                V_line_nn = nn_value_on_numpy(value_nn_or_none, k_grid, b_vec, z_vec)

                plt.plot(k_grid, V_line_star, label=f"Grid z[{l}]")
                plt.plot(k_grid, V_line_nn, "--", label=f"NN z[{l}]")

            plt.xlabel("k")
            plt.ylabel("V(k,b,z)")
            plt.title(f"Value: NN vs Grid ({tag}), b={b0:.3g}")
            plt.grid(True)
            plt.legend()
            plt.savefig(
                os.path.join(out_dir, f"value_nn_vs_grid_{tag}_b{j}.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

            # diff
            plt.figure(figsize=(8, 5))
            for l in z_idx_list:
                z0 = z_grid[l]
                b_vec = np.full_like(k_grid, b0)
                z_vec = np.full_like(k_grid, z0)

                V_line_star = interp_grid_3d(
                    k_grid, b_grid, z_grid, V_star, k_grid, b_vec, z_vec
                )
                V_line_nn = nn_value_on_numpy(value_nn_or_none, k_grid, b_vec, z_vec)

                plt.plot(k_grid, V_line_nn - V_line_star, label=f"diff z[{l}]")

            plt.axhline(0.0, linewidth=1.0)
            plt.xlabel("k")
            plt.ylabel("V_NN - V_Grid")
            plt.title(f"Value difference ({tag}), b={b0:.3g}")
            plt.grid(True)
            plt.legend()
            plt.savefig(
                os.path.join(out_dir, f"value_diff_{tag}_b{j}.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

    # (D) Scatter plots on ergodic states
    # policy scatter
    plt.figure(figsize=(6, 6))
    plt.scatter(kp_star, kp_nn, s=8)
    plt.xlabel("k'_Grid (interp)")
    plt.ylabel("k'_NN")
    plt.title(f"Policy scatter k' ({tag})")
    plt.grid(True)
    plt.savefig(
        os.path.join(out_dir, f"scatter_kp_{tag}.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(bp_star, bp_nn, s=8)
    plt.xlabel("b'_Grid (interp)")
    plt.ylabel("b'_NN")
    plt.title(f"Policy scatter b' ({tag})")
    plt.grid(True)
    plt.savefig(
        os.path.join(out_dir, f"scatter_bp_{tag}.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()

    # value scatter if exists
    if value_nn_or_none is not None:
        V_nn = nn_value_on_numpy(value_nn_or_none, k_erg, b_erg, z_erg)
        plt.figure(figsize=(6, 6))
        plt.scatter(V_star_s, V_nn, s=8)
        plt.xlabel("V_Grid (interp)")
        plt.ylabel("V_NN")
        plt.title(f"Value scatter ({tag})")
        plt.grid(True)
        plt.savefig(
            os.path.join(out_dir, f"scatter_value_{tag}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    # (E) Welfare loss histogram
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

    # -----------------------------
    # Package results
    # -----------------------------
    result = CompareResultRD(
        rmse_kp=rmse_kp,
        sup_kp=sup_kp,
        rmse_bp=rmse_bp,
        sup_bp=sup_bp,
        rmse_V=rmse_V,
        sup_V=sup_V,
        welfare_loss_mean=wl_mean,
        welfare_loss_sup=wl_sup,
    )

    summary = {
        "rmse_kp": rmse_kp,
        "sup_kp": sup_kp,
        "rmse_bp": rmse_bp,
        "sup_bp": sup_bp,
        "rmse_V": (rmse_V if rmse_V is not None else float("nan")),
        "sup_V": (sup_V if sup_V is not None else float("nan")),
        "welfare_loss_mean": wl_mean,
        "welfare_loss_sup": wl_sup,
    }

    return result, summary


# -----------------------------
# Compatibility wrapper (so run_all.py can import it)
# -----------------------------
from typing import Any


def compare_nn_to_benchmark_on_ergodic(
    *,
    policy,
    value,
    k_e: np.ndarray,
    b_e: np.ndarray,
    z_e: np.ndarray,
    bench: Dict[str, np.ndarray],
    mp: Any,
    kappa_issue: float,
    out_dir: Optional[str] = None,
    tag: str = "obj",
    n_policy_eval_sweeps: int = 2000,
) -> Dict[str, float]:
    """
    Wrapper called by Experiment/run_all.py.

    - Uses the full Step-3 routine compare_on_ergodic_states_risky_debt(...)
    - Returns a json-friendly summary dict.
    - Plots are produced by compare_on_ergodic_states_risky_debt when out_dir is not None.
    """

    # =====================================================
    # DOMAIN CHECK (FAIL LOUDLY IF STATES OUTSIDE GRID)
    # =====================================================
    k_grid = bench["k_grid"]
    b_grid = bench["b_grid"]
    z_grid = bench["z_grid"]

    print("\n================ DOMAIN CHECK =================")
    print("ergodic k:", k_e.min(), k_e.max(), "grid:", k_grid.min(), k_grid.max())
    print("ergodic b:", b_e.min(), b_e.max(), "grid:", b_grid.min(), b_grid.max())
    print("ergodic z:", z_e.min(), z_e.max(), "grid:", z_grid.min(), z_grid.max())

    bad_k = (k_e < k_grid.min()) | (k_e > k_grid.max())
    bad_b = (b_e < b_grid.min()) | (b_e > b_grid.max())
    bad_z = (z_e < z_grid.min()) | (z_e > z_grid.max())

    print(
        "out of domain counts:",
        "k =",
        int(bad_k.sum()),
        "b =",
        int(bad_b.sum()),
        "z =",
        int(bad_z.sum()),
    )

    if bad_k.any() or bad_b.any() or bad_z.any():
        raise RuntimeError("ERGODIC STATES OUTSIDE GRID — INCREASE k_max OR BOUNDS")

    print("================================================\n")

    # ---- unpack benchmark dict ----
    k_grid = bench["k_grid"]
    b_grid = bench["b_grid"]
    z_grid = bench["z_grid"]
    V_star = bench["V_star"]

    q_star = bench.get("q_star")
    if q_star is None:
        raise KeyError("bench is missing 'q_star' (pricing schedule).")

    Pz = bench.get("P")
    if Pz is None:
        raise KeyError("bench is missing 'P' (shock transition matrix).")

    # policy from benchmark: either already in levels, or stored as indices
    if "policy_kp_star" in bench and "policy_bp_star" in bench:
        policy_kp_star = bench["policy_kp_star"]
        policy_bp_star = bench["policy_bp_star"]
    else:
        pol_k_idx = bench.get("pol_k_idx")
        pol_b_idx = bench.get("pol_b_idx")
        if pol_k_idx is None or pol_b_idx is None:
            raise KeyError(
                "bench must contain either (policy_kp_star, policy_bp_star) "
                "or (pol_k_idx, pol_b_idx)."
            )
        policy_kp_star = k_grid[pol_k_idx]
        policy_bp_star = b_grid[pol_b_idx]

        # =========================================================
        # GRID POLICY SATURATION CHECK (debug)
        # =========================================================
        print("\n================ GRID POLICY SATURATION CHECK ================")
        print("[grid] b_grid range:", float(b_grid.min()), float(b_grid.max()))
        print(
            "[grid] policy_bp_star range:",
            float(np.min(policy_bp_star)),
            float(np.max(policy_bp_star)),
        )

        share_bmin = float(np.mean(np.isclose(policy_bp_star, b_grid.min())))
        share_bmax = float(np.mean(np.isclose(policy_bp_star, b_grid.max())))
        print(
            f"[grid] share at b_min: {share_bmin:.3f}   share at b_max: {share_bmax:.3f}"
        )

        uniq = np.unique(np.round(policy_bp_star.astype(np.float64), 6))
        print("[grid] unique b' values (rounded, first 20):", uniq[:20])
        print("[grid] number of unique b' values:", len(uniq))
        print("==============================================================\n")

        # ==============================
        # DEBUG: is the GRID policy b' saturating?
        # ==============================
        print("\n================ GRID POLICY SATURATION CHECK ================")
        print("[grid] b_grid range:", float(b_grid.min()), float(b_grid.max()))
        print(
            "[grid] policy_bp_star range:",
            float(np.min(policy_bp_star)),
            float(np.max(policy_bp_star)),
        )

        # how many points are essentially at bounds?
        eps = 1e-10
        at_bmin = np.mean(np.abs(policy_bp_star - b_grid.min()) < eps)
        at_bmax = np.mean(np.abs(policy_bp_star - b_grid.max()) < eps)
        print(f"[grid] share at b_min: {at_bmin:.3f}   share at b_max: {at_bmax:.3f}")

        # also print a few unique values (rounded) to see if it's almost constant
        u = np.unique(np.round(policy_bp_star.reshape(-1), 6))
        print("[grid] unique b' values (rounded, first 20):", u[:20])
        print("[grid] number of unique b' values:", len(u))
        print("==============================================================\n")

    beta = 1.0 / (1.0 + mp.r)

    # If user didn't pass out_dir, don't plot
    if out_dir is None:
        out_dir = ""

    _, summary = compare_on_ergodic_states_risky_debt(
        policy_nn=policy,
        value_nn_or_none=value,
        k_erg=k_e,
        b_erg=b_e,
        z_erg=z_e,
        k_grid=k_grid,
        b_grid=b_grid,
        z_grid=z_grid,
        V_star=V_star,
        policy_kp_star=policy_kp_star,
        policy_bp_star=policy_bp_star,
        q_star=q_star,
        Pz=Pz,
        beta=beta,
        mp=mp,
        kappa_issue=kappa_issue,
        out_dir=out_dir,
        tag=tag,
        n_policy_eval_sweeps=n_policy_eval_sweeps,
    )
    return summary

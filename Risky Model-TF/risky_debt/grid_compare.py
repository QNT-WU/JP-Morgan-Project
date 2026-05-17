"""Comparison helpers for neural solutions versus the grid benchmark.

The module keeps the numerical comparison logic in plain NumPy for robustness,
while exposing a small service class that packages one comparison request into
a reusable object-oriented interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .grid_benchmark import (
    _update_q,
    _profit_pi,
    _investment_I,
    _adj_cost_psi,
    _eta,
    _recovery_R,
)


@dataclass(frozen=True)
class CompareResultRD:
    """Scalar summary of one neural-network versus benchmark comparison run."""
    rmse_kp: float
    sup_kp: float
    rmse_bp: float
    sup_bp: float
    rmse_V: Optional[float] = None
    sup_V: Optional[float] = None
    welfare_loss_mean: Optional[float] = None
    welfare_loss_sup: Optional[float] = None
    default_mismatch_rate: Optional[float] = None
    price_rmse: Optional[float] = None
    price_sup: Optional[float] = None


def _find_bracketing_1d(
    grid: np.ndarray, x: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find racketing 1d."""
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
    """Trilinearly interpolate a benchmark object defined on the `(k,b,z)` grid."""
    k = np.asarray(k)
    b = np.asarray(b)
    z = np.asarray(z)
    i0, i1, wk = _find_bracketing_1d(k_grid, k)
    j0, j1, wb = _find_bracketing_1d(b_grid, b)
    l0, l1, wz = _find_bracketing_1d(z_grid, z)

    F000 = F[i0, j0, l0]
    F001 = F[i0, j0, l1]
    F010 = F[i0, j1, l0]
    F011 = F[i0, j1, l1]
    F100 = F[i1, j0, l0]
    F101 = F[i1, j0, l1]
    F110 = F[i1, j1, l0]
    F111 = F[i1, j1, l1]

    F00 = (1 - wk) * F000 + wk * F100
    F01 = (1 - wk) * F001 + wk * F101
    F10 = (1 - wk) * F010 + wk * F110
    F11 = (1 - wk) * F011 + wk * F111
    F0 = (1 - wb) * F00 + wb * F10
    F1 = (1 - wb) * F01 + wb * F11
    return (1 - wz) * F0 + wz * F1


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Return the root-mean-square error between two arrays."""
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def supnorm(a: np.ndarray, b: np.ndarray) -> float:
    """Return the sup-norm distance between two arrays."""
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _safe_subset_metrics(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    """Return a numerically safe version of ubset metrics."""
    if not np.any(mask):
        return float("nan"), float("nan")
    return rmse(a[mask], b[mask]), supnorm(a[mask], b[mask])


def _broadcast_flatten_inputs(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Broadcast input arrays to a common shape and flatten them.

    This keeps the NumPy-side evaluation helpers robust when callers pass
    mixtures of scalars, vectors, or mesh-grid style arrays.
    """
    bcast = np.broadcast_arrays(*[np.asarray(a) for a in arrays])
    return tuple(np.asarray(a, dtype=np.float32).reshape(-1) for a in bcast)


def nn_policy_on_numpy(policy_nn, k: np.ndarray, b: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate the policy network on NumPy inputs and return `(k', b')`."""
    import tensorflow as tf

    kf, bf, zf = _broadcast_flatten_inputs(k, b, z)
    x = tf.convert_to_tensor(np.stack([kf, bf, zf], axis=1), dtype=tf.float32)
    kb_next = policy_nn(x).numpy()
    return kb_next[:, 0], kb_next[:, 1]


def nn_value_on_numpy(value_nn, k: np.ndarray, b: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Evaluate the value network on NumPy inputs and return flattened values."""
    import tensorflow as tf

    kf, bf, zf = _broadcast_flatten_inputs(k, b, z)
    x = tf.convert_to_tensor(np.stack([kf, bf, zf], axis=1), dtype=tf.float32)
    return value_nn(x).numpy().reshape(-1)


def nn_pricing_on_numpy(qnet_nn, z: np.ndarray, k_next: np.ndarray, b_next: np.ndarray) -> np.ndarray:
    """Evaluate the legacy compatibility qnet on NumPy inputs.

    Benchmark comparisons primarily use constructed prices where model
    parameters are available. This helper is retained for fallback paths that
    still receive only a qnet-like object.
    """
    import tensorflow as tf

    zf, kf, bf = _broadcast_flatten_inputs(z, k_next, b_next)
    x = tf.convert_to_tensor(np.stack([zf, kf, bf], axis=1), dtype=tf.float32)
    return qnet_nn(x).numpy().reshape(-1)


def nearest_index_1d(grid: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Map arbitrary values to the nearest points on a one-dimensional grid."""
    x = np.asarray(x)
    j = np.searchsorted(grid, x)
    j = np.clip(j, 1, len(grid) - 1)
    left = grid[j - 1]
    right = grid[j]
    choose_left = (x - left) < (right - x)
    return np.where(choose_left, j - 1, j).astype(int)


def nn_policy_to_grid_indices(
    policy_nn,
    k_grid: np.ndarray,
    b_grid: np.ndarray,
    z_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project the neural policy onto nearest-neighbor benchmark grid indices."""
    Nk, Nb, Nz = len(k_grid), len(b_grid), len(z_grid)
    k_mesh = np.repeat(k_grid[:, None, None], Nb, axis=1)
    k_mesh = np.repeat(k_mesh, Nz, axis=2)
    b_mesh = np.repeat(b_grid[None, :, None], Nk, axis=0)
    b_mesh = np.repeat(b_mesh, Nz, axis=2)
    z_mesh = np.repeat(z_grid[None, None, :], Nk, axis=0)
    z_mesh = np.repeat(z_mesh, Nb, axis=1)
    kp_flat, bp_flat = nn_policy_on_numpy(policy_nn, k_mesh.reshape(-1), b_mesh.reshape(-1), z_mesh.reshape(-1))
    m_idx = nearest_index_1d(k_grid, kp_flat).reshape(Nk, Nb, Nz)
    n_idx = nearest_index_1d(b_grid, bp_flat).reshape(Nk, Nb, Nz)
    return m_idx, n_idx


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
    """Compute one-period equity payout using NumPy arrays."""
    import tensorflow as tf
    from .primitives import equity_payout_d_total

    # NumPy-side wrapper around the model-consistent payout object. This helper is
    # kept for diagnostics and any future policy-evaluation extensions that need the
    # TensorFlow primitive on NumPy arrays.
    d_tf = equity_payout_d_total(
        k=tf.convert_to_tensor(k, tf.float32),
        k_next=tf.convert_to_tensor(k_next, tf.float32),
        b=tf.convert_to_tensor(b, tf.float32),
        b_next=tf.convert_to_tensor(b_next, tf.float32),
        z=tf.convert_to_tensor(z, tf.float32),
        q=tf.convert_to_tensor(q, tf.float32),
        continuation_weight=tf.ones_like(tf.convert_to_tensor(q, tf.float32)),
        mp=mp,
        kappa_issue=kappa_issue,
    )
    return d_tf.numpy().reshape(-1)


def _policy_evaluation_one_step(
    *,
    V: np.ndarray,
    q_grid: np.ndarray,
    policy_m_idx: np.ndarray,
    policy_n_idx: np.ndarray,
    k_grid: np.ndarray,
    b_grid: np.ndarray,
    z_grid: np.ndarray,
    Pz: np.ndarray,
    beta: float,
    mp,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate one Bellman sweep for a fixed projected policy and fixed q-grid."""
    Nk, Nb, Nz = len(k_grid), len(b_grid), len(z_grid)
    pi = _profit_pi(k_grid, z_grid, mp.theta)
    I = _investment_I(k_grid, k_grid, mp.delta)
    psi = _adj_cost_psi(I, k_grid, mp.psi0)

    EV = (V.reshape(-1, Nz) @ Pz.T).reshape(Nk, Nb, Nz)
    surv = (V > 0.0).astype(np.float64)

    V_new = np.empty_like(V)
    C_new = np.empty_like(V)

    for iz in range(Nz):
        q_z = q_grid[iz, :, :]
        q_clip = np.clip(q_z, 1e-12, 1.0)
        r_tilde = (1.0 / q_clip) - 1.0
        p_solv = np.tensordot(surv, Pz[iz, :], axes=([2], [0]))
        debt_inflow = b_grid[None, :] * q_z
        debt_mask = (b_grid[None, :] > 0.0).astype(np.float64)
        tax_shield_mn = beta * mp.tau * r_tilde * b_grid[None, :] * p_solv * debt_mask
        term_profit = (1.0 - mp.tau) * pi[:, iz]

        for i in range(Nk):
            for j in range(Nb):
                m = policy_m_idx[i, j, iz]
                n = policy_n_idx[i, j, iz]
                e = (
                    term_profit[i]
                    - psi[i, m]
                    - I[i, m]
                    + debt_inflow[m, n]
                    + tax_shield_mn[m, n]
                    - b_grid[j]
                )
                d = e + _eta(np.array([e]), mp)[0]
                Cij = d + beta * EV[m, n, iz]
                C_new[i, j, iz] = Cij
                V_new[i, j, iz] = max(0.0, Cij)

    return V_new, C_new


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
    outer_max_iter: int = 50,
    tol_q: float = 1e-8,
    damping: float = 0.95,
) -> np.ndarray:
    """
    Policy evaluation for the projected NN policy, with repricing under the
    projected policy's own default set. This keeps welfare comparisons aligned
    with the benchmark equilibrium logic.
    """
    Nk, Nb, Nz = len(k_grid), len(b_grid), len(z_grid)
    V = np.zeros((Nk, Nb, Nz), dtype=np.float64)
    q_grid = np.asarray(q_star, dtype=np.float64).copy()
    R = _recovery_R(k_grid, z_grid, mp)

    C = np.zeros_like(V)
    for _ in range(outer_max_iter):
        for _ in range(n_sweeps):
            V_new, C_new = _policy_evaluation_one_step(
                V=V,
                q_grid=q_grid,
                policy_m_idx=policy_m_idx,
                policy_n_idx=policy_n_idx,
                k_grid=k_grid,
                b_grid=b_grid,
                z_grid=z_grid,
                Pz=Pz,
                beta=beta,
                mp=mp,
            )
            diff = np.max(np.abs(V_new - V))
            V, C = V_new, C_new
            if diff < tol:
                break

        q_comp = _update_q(C, R, Pz, b_grid, mp)
        q_next = damping * q_comp + (1.0 - damping) * q_grid
        q_diff = np.max(np.abs(q_next - q_grid))
        q_grid = q_next
        if q_diff < tol_q:
            break

    return V


def _flatten_full_grid_states(
    k_grid: np.ndarray, b_grid: np.ndarray, z_grid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten ull grid states."""
    k_mesh = np.repeat(k_grid[:, None, None], len(b_grid), axis=1)
    k_mesh = np.repeat(k_mesh, len(z_grid), axis=2)
    b_mesh = np.repeat(b_grid[None, :, None], len(k_grid), axis=0)
    b_mesh = np.repeat(b_mesh, len(z_grid), axis=2)
    z_mesh = np.repeat(z_grid[None, None, :], len(k_grid), axis=0)
    z_mesh = np.repeat(z_mesh, len(b_grid), axis=1)
    return k_mesh.reshape(-1), b_mesh.reshape(-1), z_mesh.reshape(-1)


def simulate_benchmark_ergodic_states(
    bench: Dict[str, np.ndarray],
    n_points: int,
    seed: int,
    burn_in: int = 1000,
    n_paths: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate benchmark ergodic states from the solved grid policy indices."""
    rng = np.random.default_rng(seed)
    k_grid = np.asarray(bench["k_grid"])
    b_grid = np.asarray(bench["b_grid"])
    z_grid = np.asarray(bench["z_grid"])
    pol_k = np.asarray(bench["pol_k_idx"], dtype=int)
    pol_b = np.asarray(bench["pol_b_idx"], dtype=int)
    P = np.asarray(bench["P"], dtype=float)

    n_paths = int(max(8, min(n_paths, n_points)))
    keep_steps = int(np.ceil(n_points / n_paths))
    total_steps = burn_in + keep_steps

    ik = rng.integers(0, len(k_grid), size=n_paths)
    ib = rng.integers(0, len(b_grid), size=n_paths)
    iz = rng.integers(0, len(z_grid), size=n_paths)

    ks, bs, zs = [], [], []
    for t in range(total_steps):
        if t >= burn_in:
            ks.append(k_grid[ik])
            bs.append(b_grid[ib])
            zs.append(z_grid[iz])

        ik_next = pol_k[ik, ib, iz]
        ib_next = pol_b[ik, ib, iz]
        iz_next = np.empty_like(iz)
        for z_now in range(len(z_grid)):
            mask = iz == z_now
            if np.any(mask):
                iz_next[mask] = rng.choice(len(z_grid), size=int(mask.sum()), p=P[z_now])
        ik, ib, iz = ik_next, ib_next, iz_next

    k_out = np.concatenate(ks)[:n_points]
    b_out = np.concatenate(bs)[:n_points]
    z_out = np.concatenate(zs)[:n_points]
    return k_out, b_out, z_out


def _pricing_grid_from_network(qnet, k_grid: np.ndarray, b_grid: np.ndarray, z_grid: np.ndarray, mp=None) -> np.ndarray:
    """Evaluate constructed zero-profit pricing on the comparison grid.

    The legacy qnet argument is ignored except for API compatibility.  A proxy
    default rule is used because this benchmark-comparison helper historically
    receives only policy/value/qnet objects, not the separate vtilde critic.
    """
    import tensorflow as tf
    from .config import TrainParams
    from .pricing import crn_inner_eps, smooth_price_from_proxy

    zz, kk, bb = np.meshgrid(z_grid, k_grid, b_grid, indexing="ij")
    zf, kf, bf = _broadcast_flatten_inputs(zz, kk, bb)
    if mp is None:
        # Fallback only for old external callers; prefer passing mp.
        q_flat = nn_pricing_on_numpy(qnet, zz, kk, bb)
        return q_flat.reshape(len(z_grid), len(k_grid), len(b_grid))
    tp = TrainParams(N_q=16)
    z_tf = tf.convert_to_tensor(zf, tf.float32)
    k_tf = tf.convert_to_tensor(kf, tf.float32)
    b_tf = tf.convert_to_tensor(bf, tf.float32)
    eps_q = crn_inner_eps(z_tf, tp)
    q_tf, _, _, _ = smooth_price_from_proxy(z_tf, k_tf, b_tf, eps_q, mp, tp)
    return q_tf.numpy().reshape(len(z_grid), len(k_grid), len(b_grid))


def _compute_state_set_summary(
    *,
    objective_name: str,
    policy,
    value,
    k_states: np.ndarray,
    b_states: np.ndarray,
    z_states: np.ndarray,
    k_grid: np.ndarray,
    b_grid: np.ndarray,
    z_grid: np.ndarray,
    V_star: np.ndarray,
    policy_kp_star: np.ndarray,
    policy_bp_star: np.ndarray,
    V_pi_nn: np.ndarray,
) -> Dict[str, np.ndarray | float | str]:
    """Compute tate set summary."""
    kp_nn, bp_nn = nn_policy_on_numpy(policy, k_states, b_states, z_states)
    kp_star = interp_grid_3d(k_grid, b_grid, z_grid, policy_kp_star, k_states, b_states, z_states)
    bp_star = interp_grid_3d(k_grid, b_grid, z_grid, policy_bp_star, k_states, b_states, z_states)
    V_star_s = interp_grid_3d(k_grid, b_grid, z_grid, V_star, k_states, b_states, z_states)
    V_pi_s = interp_grid_3d(k_grid, b_grid, z_grid, V_pi_nn, k_states, b_states, z_states)

    if objective_name == "obj3" and value is not None:
        V_cmp_nn = nn_value_on_numpy(value, k_states, b_states, z_states)
        value_compare_label = "direct_value_net"
        D_nn_direct = (V_cmp_nn <= 1e-10).astype(np.int32)
    else:
        V_cmp_nn = V_pi_s
        value_compare_label = "policy_induced_value"
        D_nn_direct = None

    welfare_loss = V_star_s - V_pi_s
    D_star = (V_star_s <= 1e-10).astype(np.int32)
    D_nn_policy = (V_pi_s <= 1e-10).astype(np.int32)
    continuation_mask = D_star == 0

    cont_rmse_kp, cont_sup_kp = _safe_subset_metrics(kp_nn, kp_star, continuation_mask)
    cont_rmse_bp, cont_sup_bp = _safe_subset_metrics(bp_nn, bp_star, continuation_mask)
    cont_rmse_V, cont_sup_V = _safe_subset_metrics(V_cmp_nn, V_star_s, continuation_mask)
    cont_welfare = welfare_loss[continuation_mask] if np.any(continuation_mask) else np.asarray([])

    metrics = {
        "rmse_kp": rmse(kp_nn, kp_star),
        "sup_kp": supnorm(kp_nn, kp_star),
        "rmse_bp": rmse(bp_nn, bp_star),
        "sup_bp": supnorm(bp_nn, bp_star),
        "rmse_V": rmse(V_cmp_nn, V_star_s),
        "sup_V": supnorm(V_cmp_nn, V_star_s),
        "welfare_loss_mean": float(np.mean(welfare_loss)),
        "welfare_loss_sup": float(np.max(welfare_loss)),
        "default_mismatch_rate": float(np.mean(D_star != D_nn_policy)),
        "continuation_welfare_loss_mean": float(np.mean(cont_welfare)) if cont_welfare.size else float("nan"),
        "cont_rmse_kp": cont_rmse_kp,
        "cont_sup_kp": cont_sup_kp,
        "cont_rmse_bp": cont_rmse_bp,
        "cont_sup_bp": cont_sup_bp,
        "cont_rmse_V": cont_rmse_V,
        "cont_sup_V": cont_sup_V,
        "cont_welfare_loss_mean": float(np.mean(cont_welfare)) if cont_welfare.size else float("nan"),
        "cont_welfare_loss_sup": float(np.max(cont_welfare)) if cont_welfare.size else float("nan"),
        "cont_default_mismatch_rate": float(np.mean(D_star[continuation_mask] != D_nn_policy[continuation_mask])) if np.any(continuation_mask) else float("nan"),
    }
    if D_nn_direct is not None:
        metrics["default_mismatch_rate_direct"] = float(np.mean(D_star != D_nn_direct))
        metrics["cont_default_mismatch_rate_direct"] = float(np.mean(D_star[continuation_mask] != D_nn_direct[continuation_mask])) if np.any(continuation_mask) else float("nan")

    return {
        "k_states": k_states,
        "b_states": b_states,
        "z_states": z_states,
        "kp_nn": kp_nn,
        "bp_nn": bp_nn,
        "kp_star": kp_star,
        "bp_star": bp_star,
        "V_star_s": V_star_s,
        "V_pi_s": V_pi_s,
        "V_cmp_nn": V_cmp_nn,
        "value_compare_label": value_compare_label,
        "welfare_loss": welfare_loss,
        "continuation_mask": continuation_mask,
        "D_star": D_star,
        "D_nn": D_nn_policy,
        "D_nn_direct": D_nn_direct,
        "metrics": metrics,
    }

def _plot_policy_slices(out_dir: str, tag: str, policy, k_grid: np.ndarray, b_grid: np.ndarray, z_grid: np.ndarray, policy_star: np.ndarray, which: str) -> None:
    """Plot olicy slices."""
    import os
    import matplotlib.pyplot as plt

    z_idx_list = [0, len(z_grid) // 2, len(z_grid) - 1] if len(z_grid) >= 3 else list(range(len(z_grid)))
    b_idx_list = [0, len(b_grid) // 2, len(b_grid) - 1] if len(b_grid) >= 3 else list(range(len(b_grid)))
    k_dense = np.linspace(k_grid[0], k_grid[-1], 400)

    for j in b_idx_list:
        b0 = b_grid[j]
        plt.figure(figsize=(8, 5))
        for l in z_idx_list:
            z0 = z_grid[l]
            b_vec = np.full_like(k_dense, b0)
            z_vec = np.full_like(k_dense, z0)
            kp_line_nn, bp_line_nn = nn_policy_on_numpy(policy, k_dense, b_vec, z_vec)
            y_nn = kp_line_nn if which == "kp" else bp_line_nn
            y_star = interp_grid_3d(k_grid, b_grid, z_grid, policy_star, k_dense, b_vec, z_vec)
            plt.plot(k_dense, y_star, label=f"Grid z[{l}]")
            plt.plot(k_dense, y_nn, "--", label=f"NN z[{l}]")
        plt.xlabel("k")
        plt.ylabel("k'" if which == "kp" else "b'")
        plt.title(f"Policy {which}: NN vs Grid ({tag}), b={b0:.3g}")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"policy_{which}_nn_vs_grid_{tag}_b{j}.png"), dpi=150, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(8, 5))
        for l in z_idx_list:
            z0 = z_grid[l]
            b_vec = np.full_like(k_dense, b0)
            z_vec = np.full_like(k_dense, z0)
            kp_line_nn, bp_line_nn = nn_policy_on_numpy(policy, k_dense, b_vec, z_vec)
            y_nn = kp_line_nn if which == "kp" else bp_line_nn
            y_star = interp_grid_3d(k_grid, b_grid, z_grid, policy_star, k_dense, b_vec, z_vec)
            plt.plot(k_dense, y_nn - y_star, label=f"diff z[{l}]")
        plt.axhline(0.0, linewidth=1.0)
        plt.xlabel("k")
        plt.ylabel(("k'" if which == "kp" else "b'") + "_NN - Grid")
        plt.title(f"Policy {which} difference ({tag}), b={b0:.3g}")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"policy_{which}_diff_{tag}_b{j}.png"), dpi=150, bbox_inches="tight")
        plt.close()


def _plot_value_slices(
    out_dir: str,
    tag: str,
    objective_name: str,
    value,
    k_grid: np.ndarray,
    b_grid: np.ndarray,
    z_grid: np.ndarray,
    V_star: np.ndarray,
    V_pi_nn: np.ndarray,
) -> None:
    """Plot alue slices."""
    import os
    import matplotlib.pyplot as plt

    z_idx_list = [0, len(z_grid) // 2, len(z_grid) - 1] if len(z_grid) >= 3 else list(range(len(z_grid)))
    b_idx_list = [0, len(b_grid) // 2, len(b_grid) - 1] if len(b_grid) >= 3 else list(range(len(b_grid)))

    for j in b_idx_list:
        b0 = b_grid[j]
        plt.figure(figsize=(8, 5))
        for l in z_idx_list:
            z0 = z_grid[l]
            b_vec = np.full_like(k_grid, b0)
            z_vec = np.full_like(k_grid, z0)
            V_line_star = interp_grid_3d(k_grid, b_grid, z_grid, V_star, k_grid, b_vec, z_vec)
            if objective_name == "obj3" and value is not None:
                V_line_nn = nn_value_on_numpy(value, k_grid, b_vec, z_vec)
            else:
                V_line_nn = interp_grid_3d(k_grid, b_grid, z_grid, V_pi_nn, k_grid, b_vec, z_vec)
            plt.plot(k_grid, V_line_star, label=f"Grid z[{l}]")
            plt.plot(k_grid, V_line_nn, "--", label=f"NN z[{l}]")
        plt.xlabel("k")
        plt.ylabel("V(k,b,z)")
        plt.title(f"Value: NN vs Grid ({tag}), b={b0:.3g}")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"value_nn_vs_grid_{tag}_b{j}.png"), dpi=150, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(8, 5))
        for l in z_idx_list:
            z0 = z_grid[l]
            b_vec = np.full_like(k_grid, b0)
            z_vec = np.full_like(k_grid, z0)
            V_line_star = interp_grid_3d(k_grid, b_grid, z_grid, V_star, k_grid, b_vec, z_vec)
            if objective_name == "obj3" and value is not None:
                V_line_nn = nn_value_on_numpy(value, k_grid, b_vec, z_vec)
            else:
                V_line_nn = interp_grid_3d(k_grid, b_grid, z_grid, V_pi_nn, k_grid, b_vec, z_vec)
            plt.plot(k_grid, V_line_nn - V_line_star, label=f"diff z[{l}]")
        plt.axhline(0.0, linewidth=1.0)
        plt.xlabel("k")
        plt.ylabel("V_NN - V_Grid")
        plt.title(f"Value difference ({tag}), b={b0:.3g}")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"value_diff_{tag}_b{j}.png"), dpi=150, bbox_inches="tight")
        plt.close()


def _plot_scatter_and_histograms(out_dir: str, tag: str, summary: Dict[str, Any]) -> None:
    """Plot catter and histograms."""
    import os
    import matplotlib.pyplot as plt

    def _scatter(x, y, xlabel, ylabel, title, fname):
        plt.figure(figsize=(6, 6))
        plt.scatter(x, y, s=8)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
        plt.close()

    _scatter(summary["kp_star"], summary["kp_nn"], "k'_Grid", "k'_NN", f"Policy scatter k' ({tag})", f"scatter_kp_{tag}.png")
    _scatter(summary["bp_star"], summary["bp_nn"], "b'_Grid", "b'_NN", f"Policy scatter b' ({tag})", f"scatter_bp_{tag}.png")
    _scatter(summary["V_star_s"], summary["V_cmp_nn"], "V_Grid", "V_NN comparison object", f"Value scatter ({tag})", f"scatter_value_{tag}.png")

    plt.figure(figsize=(7, 4))
    plt.hist(summary["welfare_loss"], bins=60)
    plt.xlabel("WelfareLoss = V* - V^{pi_NN}")
    plt.ylabel("count")
    plt.title(f"Welfare loss distribution ({tag})")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, f"welfare_loss_hist_{tag}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    cont_mask = np.asarray(summary.get("continuation_mask", []), dtype=bool)
    if cont_mask.size and np.any(cont_mask):
        _scatter(summary["kp_star"][cont_mask], summary["kp_nn"][cont_mask], "k'_Grid", "k'_NN", f"Policy scatter k' continuation ({tag})", f"scatter_kp_continuation_{tag}.png")
        _scatter(summary["bp_star"][cont_mask], summary["bp_nn"][cont_mask], "b'_Grid", "b'_NN", f"Policy scatter b' continuation ({tag})", f"scatter_bp_continuation_{tag}.png")
        _scatter(summary["V_star_s"][cont_mask], summary["V_cmp_nn"][cont_mask], "V_Grid", "V_NN comparison object", f"Value scatter continuation ({tag})", f"scatter_value_continuation_{tag}.png")

        plt.figure(figsize=(7, 4))
        plt.hist(summary["welfare_loss"][cont_mask], bins=60)
        plt.xlabel("WelfareLoss = V* - V^{pi_NN}")
        plt.ylabel("count")
        plt.title(f"Welfare loss continuation ({tag})")
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, f"welfare_loss_hist_continuation_{tag}.png"), dpi=150, bbox_inches="tight")
        plt.close()

def _plot_default_and_welfare_maps(
    out_dir: str,
    tag: str,
    k_grid: np.ndarray,
    b_grid: np.ndarray,
    z_grid: np.ndarray,
    V_star: np.ndarray,
    V_pi_nn: np.ndarray,
) -> None:
    """Plot efault and welfare maps."""
    import os
    import matplotlib.pyplot as plt

    mid_z = len(z_grid) // 2
    D_star = (V_star[:, :, mid_z] <= 1e-10).astype(float)
    D_nn = (V_pi_nn[:, :, mid_z] <= 1e-10).astype(float)
    mismatch = (D_star != D_nn).astype(float)
    welfare = V_star[:, :, mid_z] - V_pi_nn[:, :, mid_z]

    for arr, name, title in [
        (D_star, "default_region_grid", "Benchmark default region"),
        (D_nn, "default_region_nn_policy", "NN-policy default region"),
        (mismatch, "default_region_mismatch", "Default mismatch"),
        (welfare, "welfare_loss_heatmap", "Welfare loss"),
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
        plt.title(f"{title} at z[{mid_z}] ({tag})")
        plt.savefig(os.path.join(out_dir, f"{name}_{tag}.png"), dpi=150, bbox_inches="tight")
        plt.close()


def _plot_pricing_maps(
    out_dir: str,
    tag: str,
    k_grid: np.ndarray,
    b_grid: np.ndarray,
    z_grid: np.ndarray,
    q_star: np.ndarray,
    q_nn: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Plot ricing maps."""
    import os
    import matplotlib.pyplot as plt

    diff = q_nn - q_star
    pos_b_mask = b_grid > 0.0
    q_nn_pos = q_nn[..., pos_b_mask] if np.any(pos_b_mask) else None
    q_star_pos = q_star[..., pos_b_mask] if np.any(pos_b_mask) else None
    pricing_summary = {
        "full_pricing_grid": {
            "price_rmse": rmse(q_nn, q_star),
            "price_sup": supnorm(q_nn, q_star),
        },
        "positive_debt_only": {
            "price_rmse": rmse(q_nn_pos, q_star_pos) if q_nn_pos is not None else float("nan"),
            "price_sup": supnorm(q_nn_pos, q_star_pos) if q_nn_pos is not None else float("nan"),
        },
    }
    mid_z = len(z_grid) // 2

    for arr, name, title in [
        (q_star[mid_z], "pricing_grid", "Benchmark pricing q"),
        (q_nn[mid_z], "pricing_nn", "NN pricing q"),
        (diff[mid_z], "pricing_diff", "Pricing difference q_NN - q_Grid"),
    ]:
        plt.figure(figsize=(7, 5))
        plt.imshow(
            arr.T,
            origin="lower",
            aspect="auto",
            extent=[k_grid[0], k_grid[-1], b_grid[0], b_grid[-1]],
        )
        plt.colorbar()
        plt.xlabel("k'")
        plt.ylabel("b'")
        plt.title(f"{title} at z[{mid_z}] ({tag})")
        plt.savefig(os.path.join(out_dir, f"{name}_{tag}.png"), dpi=150, bbox_inches="tight")
        plt.close()

    return pricing_summary

@dataclass(frozen=True)
class BenchmarkComparatorConfig:
    """Configuration for one neural-network versus benchmark comparison job."""

    out_dir: Optional[str] = None
    tag: str = "obj"
    objective_name: str = "obj1"
    n_policy_eval_sweeps: int = 2000
    benchmark_ergodic_seed: int = 0


class BenchmarkComparator:
    """Service object that compares one trained neural policy with one benchmark."""

    def __init__(self, *, mp: Any, kappa_issue: float, config: Optional[BenchmarkComparatorConfig] = None) -> None:
        """Initialize BenchmarkComparator."""
        self.mp = mp
        self.kappa_issue = float(kappa_issue)
        self.config = config or BenchmarkComparatorConfig()

    @staticmethod
    def _validate_state_domain(
        k_e: np.ndarray,
        b_e: np.ndarray,
        z_e: np.ndarray,
        k_grid: np.ndarray,
        b_grid: np.ndarray,
        z_grid: np.ndarray,
    ) -> None:
        """Fail fast when comparison states fall outside the benchmark domain."""
        print("\n================ DOMAIN CHECK =================")
        print("ergodic k:", float(np.min(k_e)), float(np.max(k_e)), "grid:", float(np.min(k_grid)), float(np.max(k_grid)))
        print("ergodic b:", float(np.min(b_e)), float(np.max(b_e)), "grid:", float(np.min(b_grid)), float(np.max(b_grid)))
        print("ergodic z:", float(np.min(z_e)), float(np.max(z_e)), "grid:", float(np.min(z_grid)), float(np.max(z_grid)))
        bad_k = (k_e < k_grid.min()) | (k_e > k_grid.max())
        bad_b = (b_e < b_grid.min()) | (b_e > b_grid.max())
        bad_z = (z_e < z_grid.min()) | (z_e > z_grid.max())
        print("out of domain counts:", "k =", int(bad_k.sum()), "b =", int(bad_b.sum()), "z =", int(bad_z.sum()))
        if bad_k.any() or bad_b.any() or bad_z.any():
            raise RuntimeError("ERGODIC STATES OUTSIDE GRID — INCREASE k_max OR BOUNDS")
        print("================================================\n")

    @staticmethod
    def _benchmark_policy_levels(bench: Dict[str, np.ndarray], k_grid: np.ndarray, b_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return benchmark controls in economically meaningful levels."""
        if "policy_kp_star" in bench and "policy_bp_star" in bench:
            return np.asarray(bench["policy_kp_star"]), np.asarray(bench["policy_bp_star"])
        pol_k_idx = np.asarray(bench["pol_k_idx"], dtype=int)
        pol_b_idx = np.asarray(bench["pol_b_idx"], dtype=int)
        return k_grid[pol_k_idx], b_grid[pol_b_idx]

    def compare(
        self,
        *,
        policy,
        value,
        qnet,
        k_e: np.ndarray,
        b_e: np.ndarray,
        z_e: np.ndarray,
        bench: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Run the full benchmark comparison and return the NN-ergodic metrics."""
        import os

        k_grid = np.asarray(bench["k_grid"])
        b_grid = np.asarray(bench["b_grid"])
        z_grid = np.asarray(bench["z_grid"])
        self._validate_state_domain(k_e, b_e, z_e, k_grid, b_grid, z_grid)

        V_star = np.asarray(bench["V_star"])
        q_star = np.asarray(bench["q_star"])
        Pz = np.asarray(bench["P"])
        policy_kp_star, policy_bp_star = self._benchmark_policy_levels(bench, k_grid, b_grid)

        beta = 1.0 / (1.0 + self.mp.r)
        m_idx, n_idx = nn_policy_to_grid_indices(policy, k_grid, b_grid, z_grid)
        V_pi_nn = policy_evaluation_fixed_policy(
            k_grid=k_grid,
            b_grid=b_grid,
            z_grid=z_grid,
            Pz=Pz,
            beta=beta,
            policy_m_idx=m_idx,
            policy_n_idx=n_idx,
            q_star=q_star,
            mp=self.mp,
            kappa_issue=self.kappa_issue,
            n_sweeps=self.config.n_policy_eval_sweeps,
        )
        q_nn_grid = _pricing_grid_from_network(qnet, k_grid, b_grid, z_grid, mp=self.mp)

        full_grid_states = _flatten_full_grid_states(k_grid, b_grid, z_grid)
        benchmark_ergodic_states = simulate_benchmark_ergodic_states(
            bench,
            n_points=len(np.asarray(k_e).reshape(-1)),
            seed=self.config.benchmark_ergodic_seed,
        )
        nn_ergodic_states = (
            np.asarray(k_e).reshape(-1),
            np.asarray(b_e).reshape(-1),
            np.asarray(z_e).reshape(-1),
        )

        out_dir = self.config.out_dir or ""
        full_dir = os.path.join(out_dir, "full_grid")
        bench_erg_dir = os.path.join(out_dir, "benchmark_ergodic")
        nn_erg_dir = os.path.join(out_dir, "nn_ergodic")
        pricing_dir = os.path.join(out_dir, "pricing")
        for directory in [full_dir, bench_erg_dir, nn_erg_dir, pricing_dir]:
            os.makedirs(directory, exist_ok=True)

        summaries: Dict[str, Any] = {}
        for set_name, state_tuple, target_dir in [
            ("full_grid", full_grid_states, full_dir),
            ("benchmark_ergodic", benchmark_ergodic_states, bench_erg_dir),
            ("nn_ergodic", nn_ergodic_states, nn_erg_dir),
        ]:
            state_summary = _compute_state_set_summary(
                objective_name=self.config.objective_name,
                policy=policy,
                value=value,
                k_states=state_tuple[0],
                b_states=state_tuple[1],
                z_states=state_tuple[2],
                k_grid=k_grid,
                b_grid=b_grid,
                z_grid=z_grid,
                V_star=V_star,
                policy_kp_star=policy_kp_star,
                policy_bp_star=policy_bp_star,
                V_pi_nn=V_pi_nn,
            )
            local_tag = f"{self.config.tag}_{set_name}"
            _plot_policy_slices(target_dir, local_tag, policy, k_grid, b_grid, z_grid, policy_kp_star, "kp")
            _plot_policy_slices(target_dir, local_tag, policy, k_grid, b_grid, z_grid, policy_bp_star, "bp")
            _plot_value_slices(target_dir, local_tag, self.config.objective_name, value, k_grid, b_grid, z_grid, V_star, V_pi_nn)
            _plot_scatter_and_histograms(target_dir, local_tag, state_summary)
            _plot_default_and_welfare_maps(target_dir, local_tag, k_grid, b_grid, z_grid, V_star, V_pi_nn)
            state_summary["metrics"]["value_compare_label"] = state_summary["value_compare_label"]
            summaries[set_name] = state_summary["metrics"]

        pricing_summary = _plot_pricing_maps(pricing_dir, self.config.tag, k_grid, b_grid, z_grid, q_star, q_nn_grid)
        nn_metrics = dict(summaries["nn_ergodic"])
        nn_metrics["state_sets"] = summaries
        nn_metrics["pricing"] = pricing_summary
        return nn_metrics


def compare_nn_to_benchmark_on_ergodic(
    *,
    policy,
    value,
    qnet,
    k_e: np.ndarray,
    b_e: np.ndarray,
    z_e: np.ndarray,
    bench: Dict[str, np.ndarray],
    mp: Any,
    kappa_issue: float,
    out_dir: Optional[str] = None,
    tag: str = "obj",
    objective_name: str = "obj1",
    n_policy_eval_sweeps: int = 2000,
    benchmark_ergodic_seed: int = 0,
) -> Dict[str, Any]:
    """Backward-compatible functional wrapper around :class:`BenchmarkComparator`."""
    comparator = BenchmarkComparator(
        mp=mp,
        kappa_issue=kappa_issue,
        config=BenchmarkComparatorConfig(
            out_dir=out_dir,
            tag=tag,
            objective_name=objective_name,
            n_policy_eval_sweeps=n_policy_eval_sweeps,
            benchmark_ergodic_seed=benchmark_ergodic_seed,
        ),
    )
    return comparator.compare(
        policy=policy,
        value=value,
        qnet=qnet,
        k_e=k_e,
        b_e=b_e,
        z_e=z_e,
        bench=bench,
    )

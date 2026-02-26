# Src/grid_benchmark.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import math
import numpy as np


# ----------------------------
# Step 1: grids + Tauchen
# ----------------------------
def tauchen_logz(
    Nz: int,
    rho: float,
    sigma_eps: float,
    m: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tauchen discretization for:
        log z' = rho log z + eps, eps ~ N(0, sigma_eps^2)

    Returns:
        z_grid: shape (Nz,) in levels
        P:      shape (Nz, Nz), row-stochastic
    """
    # stationary std of log z
    sigma_y = sigma_eps / np.sqrt(1.0 - rho**2)

    y_max = m * sigma_y
    y_min = -m * sigma_y
    y_grid = np.linspace(y_min, y_max, Nz)

    step = y_grid[1] - y_grid[0]

    def norm_cdf(x):
        # math.erf works for scalars, so ensure x is a float
        x = float(x)
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    P = np.zeros((Nz, Nz))
    for i in range(Nz):
        for j in range(Nz):
            if j == 0:
                ub = (y_grid[0] - rho * y_grid[i] + step / 2.0) / sigma_eps
                P[i, j] = norm_cdf(ub)
            elif j == Nz - 1:
                lb = (y_grid[-1] - rho * y_grid[i] - step / 2.0) / sigma_eps
                P[i, j] = 1.0 - norm_cdf(lb)
            else:
                lb = (y_grid[j] - rho * y_grid[i] - step / 2.0) / sigma_eps
                ub = (y_grid[j] - rho * y_grid[i] + step / 2.0) / sigma_eps
                P[i, j] = norm_cdf(ub) - norm_cdf(lb)

    z_grid = np.exp(y_grid)
    return z_grid, P


def make_k_grid(Nk: int, k_min: float, k_max: float) -> np.ndarray:
    return np.linspace(k_min, k_max, Nk)


# ----------------------------
# Step 2 and 3: shared plumbing
# ----------------------------
@dataclass(frozen=True)
class GridSpec:
    Nk: int = 200
    Nz: int = 7
    k_min: float = 0.05
    k_max: float = 8.0
    tauchen_m: float = 3.0


@dataclass
class Precomputed:
    k_grid: np.ndarray  # (Nk,)
    z_grid: np.ndarray  # (Nz,)
    Pz: np.ndarray  # (Nz,Nz)
    u: np.ndarray  # (Nk,Nz,Nk)  u(i,m,j)
    feasible: np.ndarray  # (Nk,Nk)     feasibility mask over (i,j)
    beta: float


def precompute_arrays(mp, gs: GridSpec) -> Precomputed:
    """
    Precompute the shared objects used by both VFI and Howard PI:

    - k_grid, z_grid, Pz
    - I_ij
    - psi_ij
    - pi_im
    - u(i,m,j) = pi_im - psi_ij - I_ij
    - feasibility mask (nonempty action set)
    """
    beta = 1.0 / (1.0 + mp.r)

    k_grid = make_k_grid(gs.Nk, gs.k_min, gs.k_max)
    z_grid, Pz = tauchen_logz(gs.Nz, mp.rho, mp.sigma_eps, m=gs.tauchen_m)

    # I_ij = k_j - (1-delta) k_i
    k_i = k_grid[:, None]  # (Nk,1)
    k_j = k_grid[None, :]  # (1,Nk)
    I_ij = k_j - (1.0 - mp.delta) * k_i  # (Nk,Nk)

    # psi_ij = psi0 * I^2/(2k_i)
    psi_ij = mp.psi0 * (I_ij**2) / (2.0 * k_i)  # (Nk,Nk)

    # feasibility A(i,m): here we only enforce k_j >= k_min and k_i>0 already true.
    # You can add investment bounds here if you want.
    feasible = np.isfinite(I_ij)  # all True, but keeps the structure explicit.

    # pi_im = z_m * k_i^theta
    pi_im = z_grid[None, :] * (k_grid[:, None] ** mp.theta)  # (Nk,Nz)

    # u(i,m,j) = pi_im - psi_ij - I_ij
    # shapes:
    # pi_im -> (Nk,Nz,1)
    # psi_ij -> (Nk,1,Nk)
    # I_ij -> (Nk,1,Nk)
    u = pi_im[:, :, None] - psi_ij[:, None, :] - I_ij[:, None, :]

    # apply feasibility mask: if infeasible -> -inf
    u = np.where(feasible[:, None, :], u, -np.inf)

    return Precomputed(
        k_grid=k_grid,
        z_grid=z_grid,
        Pz=Pz,
        u=u,
        feasible=feasible,
        beta=beta,
    )


# ----------------------------
# Step 1A: VFI benchmark
# ----------------------------
def solve_vfi(
    pre: Precomputed,
    tol: float = 1e-7,
    max_iter: int = 5000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Value Function Iteration on the discretized problem.

    Returns:
        V_star:      (Nk,Nz)
        policy_idx:  (Nk,Nz)  indices j
    """
    Nk, Nz = pre.k_grid.shape[0], pre.z_grid.shape[0]
    V = np.zeros((Nk, Nz))
    policy_idx = np.zeros((Nk, Nz), dtype=int)

    for _it in range(max_iter):
        # EV(j,m) = sum_{m'} Pz[m,m'] V(j,m')
        EV = V @ pre.Pz.T  # (Nk,Nz)

        # Q(i,m,j) = u(i,m,j) + beta * EV(j,m)
        Q = pre.u + pre.beta * EV[None, :, :].transpose(0, 2, 1)
        # better explicit shape:
        # EV_jm = EV[j,m] => we want add over j dimension:
        # EV has (Nk,Nz). We want (1,Nz,Nk) aligned to u(i,Nz,Nk) along last axis j
        EV_align = EV.T[:, None, :]  # (Nz,1,Nk) not right
        # We'll do a safer construction:
        EV_align = EV[None, :, :]  # (1,Nk,Nz)
        # u is (Nk,Nz,Nk), so we want EV for (j,m): index last axis.
        # loop over m but vectorized over i,j (it is fast for Nk<=300,Nz<=9)
        V_new = np.empty_like(V)
        for m in range(Nz):
            Qm = pre.u[:, m, :] + pre.beta * EV[:, m][None, :]  # (Nk,Nk)
            policy_idx[:, m] = np.argmax(Qm, axis=1)
            V_new[:, m] = np.max(Qm, axis=1)

        diff = np.max(np.abs(V_new - V))
        V = V_new
        if diff < tol:
            break

    return V, policy_idx


# ----------------------------
# Step 1B: Howard / Modified PI benchmark
# ----------------------------
def policy_evaluation_sweeps(
    pre: Precomputed,
    policy_idx: np.ndarray,
    V_init: Optional[np.ndarray] = None,
    n_sweeps: int = 50,
) -> np.ndarray:
    """
    Evaluate V^pi approximately via n_sweeps of:
        V <- T^pi V
    """
    Nk, Nz = pre.k_grid.shape[0], pre.z_grid.shape[0]
    V = np.zeros((Nk, Nz)) if V_init is None else V_init.copy()

    for _ in range(n_sweeps):
        V_new = np.empty_like(V)
        EV = V @ pre.Pz.T  # (Nk,Nz)

        for m in range(Nz):
            j = policy_idx[:, m]  # (Nk,)
            u_pi = pre.u[np.arange(Nk), m, j]  # (Nk,)
            V_new[:, m] = u_pi + pre.beta * EV[j, m]

        V = V_new

    return V


def solve_howard_pi(
    pre: Precomputed,
    tol_policy: int = 0,
    max_outer: int = 200,
    eval_sweeps: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Howard / Modified Policy Iteration:
    - Start with arbitrary policy
    - Policy evaluation by finite sweeps
    - Policy improvement by greedy step

    Returns:
        V_star, policy_idx
    """
    Nk, Nz = pre.k_grid.shape[0], pre.z_grid.shape[0]
    policy_idx = np.zeros((Nk, Nz), dtype=int)
    V = np.zeros((Nk, Nz))

    for _outer in range(max_outer):
        # evaluate
        V = policy_evaluation_sweeps(pre, policy_idx, V_init=V, n_sweeps=eval_sweeps)

        # improve
        EV = V @ pre.Pz.T
        new_policy = policy_idx.copy()

        for m in range(Nz):
            Qm = pre.u[:, m, :] + pre.beta * EV[:, m][None, :]
            new_policy[:, m] = np.argmax(Qm, axis=1)

        changes = np.sum(new_policy != policy_idx)
        policy_idx = new_policy

        if changes <= tol_policy:
            break

    # final evaluation
    V = policy_evaluation_sweeps(pre, policy_idx, V_init=V, n_sweeps=eval_sweeps * 2)
    return V, policy_idx


def policy_from_idx(k_grid: np.ndarray, policy_idx: np.ndarray) -> np.ndarray:
    return k_grid[policy_idx]

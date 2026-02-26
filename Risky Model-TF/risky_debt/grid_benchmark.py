# risky_debt/grid_benchmark.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import numpy as np

from .config import ModelParams


@dataclass
class GridBenchParams:
    Nk: int = 60
    Nb: int = 61
    Nz: int = 7
    # Nk: int = 15
    # Nb: int = 21
    # Nz: int = 5
    k_max: float = 8.0
    z_m: float = 6.0

    outer_max_iter: int = 40
    # just to confirm degeneracy is gone
    # outer_max_iter: int = 5
    tol_q: float = 1e-3
    damping: float = 0.95

    inner_max_iter: int = 2000
    # inner_max_iter: int = 100
    tol_V: float = 1e-6

    mpi_eval_sweeps: int = 10
    # mpi_eval_sweeps: int = 3
    mpi_max_iter: int = 500


# ----------- shock discretization (Rouwenhorst) -----------
def _rouwenhorst_markov(N: int, rho: float) -> np.ndarray:
    p = (1 + rho) / 2
    q = p
    P = np.array([[1.0]])
    for n in range(2, N + 1):
        Pn = np.zeros((n, n))
        Pn[:-1, :-1] += p * P
        Pn[:-1, 1:] += (1 - p) * P
        Pn[1:, :-1] += (1 - q) * P
        Pn[1:, 1:] += q * P
        P = Pn
        P[1:-1, :] /= 2
    P = P / P.sum(axis=1, keepdims=True)
    return P


def build_z_grid_and_P(mp: ModelParams, Nz: int, z_m: float):
    rho = mp.rho
    sigma = mp.sigma_eps
    sig_z = sigma / np.sqrt(max(1e-12, 1 - rho**2))
    logz = np.linspace(-z_m * sig_z, z_m * sig_z, Nz)
    z = np.exp(logz).astype(np.float64)
    P = _rouwenhorst_markov(Nz, rho)
    print(
        "[DEBUG build_z_grid_and_P] z_m=",
        z_m,
        "z_range=",
        float(z.min()),
        float(z.max()),
        "Nz=",
        Nz,
    )
    return z, P


# ----------- primitives -----------
def _profit_pi(k: np.ndarray, z: np.ndarray, theta: float) -> np.ndarray:
    return (z[None, :] * (k[:, None] ** theta)).astype(np.float64)  # (Nk,Nz)


def _investment_I(k: np.ndarray, k_next: np.ndarray, delta: float) -> np.ndarray:
    return (k_next[None, :] - (1 - delta) * k[:, None]).astype(np.float64)  # (Nk,Nk)


def _adj_cost_psi(I: np.ndarray, k: np.ndarray, psi0: float) -> np.ndarray:
    return (psi0 * (I**2) / (2.0 * (k[:, None] + 1e-12))).astype(np.float64)  # (Nk,Nk)


def _recovery_R(k_next: np.ndarray, z_next: np.ndarray, mp: ModelParams) -> np.ndarray:
    pi_next = z_next[None, :] * (k_next[:, None] ** mp.theta)
    R = (1 - mp.alpha) * ((1 - mp.tau) * pi_next + (1 - mp.delta) * k_next[:, None])
    return R.astype(np.float64)


"""
def _eta(e: np.ndarray, mp: ModelParams) -> np.ndarray:
    return ((mp.eta0 + mp.eta1 * e) * (e < 0)).astype(np.float64)
"""


def _eta(e: np.ndarray, mp: ModelParams) -> np.ndarray:
    # Consistent with Strebulaev risky-debt form: d = e + eta(e)
    # where eta(e) is NEGATIVE when e<0 (issuance cost reduces payout).
    # cost = eta0 + eta1 * (-e) = eta0 - eta1 * e  (since e<0 => -e>0)
    # eta(e) = -cost
    cost = (mp.eta0 - mp.eta1 * e) * (e < 0)
    return (-cost).astype(np.float64)


# ----------- Bellman update given q_grid -----------
def _bellman_update_all(
    V: np.ndarray,  # (Nk,Nb,Nz)
    q_grid: np.ndarray,  # (Nz,Nk,Nb) pricing at issuance nodes (z,k',b')
    pi: np.ndarray,  # (Nk,Nz)
    I: np.ndarray,  # (Nk,Nk)
    psi: np.ndarray,  # (Nk,Nk)
    k: np.ndarray,
    b: np.ndarray,
    P: np.ndarray,  # (Nz,Nz)
    mp: ModelParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Nk, Nb, Nz = V.shape
    beta = 1.0 / (1.0 + mp.r)

    # EV = np.einsum("zzp,kbp->kbz", P, V)  # (Nk,Nb,Nz)
    Nk, Nb, Nz = V.shape
    EV = (V.reshape(-1, Nz) @ P.T).reshape(Nk, Nb, Nz)  # (Nk,Nb,Nz)

    V_new = np.empty_like(V)
    C_new = np.empty_like(V)
    pol_k = np.zeros((Nk, Nb, Nz), dtype=np.int32)
    pol_b = np.zeros((Nk, Nb, Nz), dtype=np.int32)

    for iz in range(Nz):
        q_z = q_grid[iz, :, :]  # (Nk,Nb)

        term_profit = (1 - mp.tau) * pi[:, iz]  # (Nk,)
        # debt_inflow = (b[None, :] * q_z) + (mp.tau * b[None, :] / (1.0 + mp.r)) * (1.0 - q_z)  # (Nk,Nb)
        # debt_inflow = b[None, :] * q_z
        # cont_mn = beta * EV[:, :, iz]  # (Nk,Nb)

        # cash received today from issuing b' at price q
        debt_inflow = b[None, :] * q_z

        # continuation value part
        cont_mn = beta * EV[:, :, iz]  # (Nk,Nb)

        # ============================================================
        # NEW TAX SHIELD in benchmark:
        # PV today = beta * tau * r_tilde(m,n) * b_n * P(survive next period)
        # where r_tilde = 1/q - 1, survive means V(k',b',z')>0
        # ============================================================

        # r_tilde(m,n) from issuance price q(z,k',b')
        q_clip = np.clip(q_z, 1e-12, 1.0)
        r_tilde = (1.0 / q_clip) - 1.0  # (Nk,Nb)

        # survival indicator at next-period states under current V
        surv = (V > 0.0).astype(np.float64)  # (Nk,Nb,Nz)

        # survival probability for each (k',b') conditional on current z=iz:
        # p_solv[m,n] = sum_{z'} P[iz,z'] * 1{ V[m,n,z'] > 0 }
        p_solv = np.tensordot(surv, P[iz, :], axes=([2], [0]))  # (Nk,Nb)

        # tax shield only applies if b' > 0 (debt). If b'<=0 (cash), shield=0.
        bpos = (b[None, :] > 0.0).astype(np.float64)

        tax_shield_mn = (
            beta * mp.tau * r_tilde * (b[None, :]) * p_solv * bpos
        )  # (Nk,Nb)

        for i in range(Nk):
            # base (m,n) part for this i excluding current -b_j
            # base_mn = ((term_profit[i] - psi[i, :] - I[i, :])[:, None] + debt_inflow + cont_mn)  # (Nk,Nb)

            # -----------------------
            # NEW: borrowing cost on b' > 0
            # BorrowCost = 0.5 * phi_borrow * (max(b',0))^2
            # b is the b' grid (shape (Nb,))
            # -----------------------
            bpos = np.maximum(b, 0.0)  # (Nb,)
            borrow_cost_mn = 0.5 * mp.phi_borrow * (bpos[None, :] ** 2)  # (Nk,Nb)
            base_mn = (
                (term_profit[i] - psi[i, :] - I[i, :])[:, None]
                + debt_inflow
                + tax_shield_mn
                - borrow_cost_mn
                + cont_mn
            )  # (Nk,Nb)
            # base_mn = ((term_profit[i] - psi[i, :] - I[i, :])[:, None] + debt_inflow + tax_shield_mn+ cont_mn)  # (Nk,Nb)

            for j in range(Nb):
                e_mn = base_mn - b[j]
                d_mn = e_mn + _eta(e_mn, mp)
                C = np.max(d_mn)
                V_new[i, j, iz] = max(0.0, C)
                C_new[i, j, iz] = C
                a = np.argmax(d_mn)
                pol_k[i, j, iz] = a // Nb
                pol_b[i, j, iz] = a % Nb

    return V_new, C_new, pol_k, pol_b


def _solve_inner(
    method: Literal["vi", "mpi"],
    V0: np.ndarray,
    q_grid: np.ndarray,
    pi: np.ndarray,
    I: np.ndarray,
    psi: np.ndarray,
    k: np.ndarray,
    b: np.ndarray,
    P: np.ndarray,
    mp: ModelParams,
    gp: GridBenchParams,
) -> Dict[str, np.ndarray]:
    V = V0.copy()

    if method == "vi":
        C = None
        pol_k = pol_b = None
        for _ in range(gp.inner_max_iter):
            V_new, C_new, pk, pb = _bellman_update_all(
                V, q_grid, pi, I, psi, k, b, P, mp
            )
            err = np.max(np.abs(V_new - V))
            V = V_new
            C, pol_k, pol_b = C_new, pk, pb
            if err < gp.tol_V:
                break
        return {"V": V, "C": C, "pol_k": pol_k, "pol_b": pol_b}

    if method == "mpi":
        # initialize greedy
        V, C, pol_k, pol_b = _bellman_update_all(V, q_grid, pi, I, psi, k, b, P, mp)

        beta = 1.0 / (1.0 + mp.r)
        Nk, Nb, Nz = V.shape

        for _ in range(gp.mpi_max_iter):
            # policy evaluation sweeps
            for _ in range(gp.mpi_eval_sweeps):
                # EV = np.einsum("zzp,kbp->kbz", P, V)
                EV = np.tensordot(V, P.T, axes=(2, 0))
                V_eval = np.empty_like(V)
                C_eval = np.empty_like(V)

                for iz in range(Nz):
                    q_z = q_grid[iz, :, :]
                    term_profit = (1 - mp.tau) * pi[:, iz]
                    # debt_inflow = (b[None, :] * q_z) + (mp.tau * b[None, :] / (1.0 + mp.r)) * (1.0 - q_z)
                    debt_inflow = b[None, :] * q_z

                    for i in range(Nk):
                        for j in range(Nb):
                            m = pol_k[i, j, iz]
                            n = pol_b[i, j, iz]
                            e = (
                                term_profit[i]
                                - psi[i, m]
                                - I[i, m]
                                + debt_inflow[m, n]
                                - b[j]
                                + beta * EV[m, n, iz]
                            )
                            # Cij = e + (mp.eta0 + mp.eta1 * e if e < 0 else 0.0)

                            # issuance cost: eta(e) = -(eta0 - eta1*e) for e<0, else 0
                            eta = -(mp.eta0 - mp.eta1 * e) if e < 0 else 0.0
                            Cij = e + eta

                            C_eval[i, j, iz] = Cij
                            V_eval[i, j, iz] = max(0.0, Cij)

                V, C = V_eval, C_eval

            # improvement
            V_new, C_new, pk_new, pb_new = _bellman_update_all(
                V, q_grid, pi, I, psi, k, b, P, mp
            )
            stable = (pk_new == pol_k).all() and (pb_new == pol_b).all()
            V, C, pol_k, pol_b = V_new, C_new, pk_new, pb_new
            if stable:
                break

        return {"V": V, "C": C, "pol_k": pol_k, "pol_b": pol_b}

    raise ValueError(f"Unknown inner method: {method}")


def _update_q(
    C: np.ndarray,  # (Nk,Nb,Nz) pre-default continuation; default iff C<0
    R: np.ndarray,  # (Nk,Nz) recovery at (k',z')
    P: np.ndarray,  # (Nz,Nz)
    b: np.ndarray,  # (Nb,) = grid of face values b'
    mp: ModelParams,
) -> np.ndarray:

    print("[DEBUG _update_q] USING LINEAR PRICING VERSION (face-value price form)")
    """
    Correct risky-debt pricing when b' is FACE VALUE due next period
    and the firm receives q*b' today.

    Break-even (discrete Strebulaev lender pricing):
        q(z,k',b') * b' = (1/(1+r)) * E[ 1{sol}*b' + 1{def}*R(k',z') ]

    => for b'>0:
        q = (1/(1+r)) * ( p_sol + E[1{def}*R]/b' )

    For b'<=0 (cash/savings), use risk-free price q_rf.
    """
    Nk, Nb, Nz = C.shape
    disc = 1.0 / (1.0 + mp.r)
    q_rf = disc

    q_new = np.ones((Nz, Nk, Nb), dtype=np.float64) * q_rf
    eps = 1e-12

    for iz in range(Nz):
        Piz = P[iz, :]  # (Nz,)
        for m in range(Nk):
            Cm = C[m, :, :]  # (Nb, Nz) at (k'=k[m], b'=b[n], z')
            Rm = R[m, :]  # (Nz,)

            for n in range(Nb):
                bprime = float(b[n])

                # cash / savings (non-issued debt) priced at risk-free
                if bprime <= 0.0 or abs(bprime) < eps:
                    q_new[iz, m, n] = q_rf
                    continue

                # default next period if continuation is negative at that next state
                D_next = (Cm[n, :] < 0.0).astype(np.float64)  # (Nz,)

                p_def = float(np.dot(Piz, D_next))
                p_sol = 1.0 - p_def

                # expected recovery in default states
                rec = float(np.dot(Piz * D_next, Rm))  # scalar E[1{def} R]

                # q = disc*(p_sol + rec/b')
                q = disc * (p_sol + rec / bprime)

                # numerical guards
                q = float(np.clip(q, mp.q_min, mp.q_max))
                q_new[iz, m, n] = q

    return q_new


def solve_grid_benchmark(
    mp: ModelParams,
    gp: GridBenchParams,
    inner_method: Literal["vi", "mpi"] = "vi",
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    # grids
    k = np.linspace(mp.k_min, gp.k_max, gp.Nk).astype(np.float64)
    b = np.linspace(mp.b_min, mp.b_max, gp.Nb).astype(np.float64)
    z, P = build_z_grid_and_P(mp, gp.Nz, gp.z_m)

    # precompute
    pi = _profit_pi(k, z, mp.theta)  # (Nk,Nz)
    I = _investment_I(k, k, mp.delta)  # (Nk,Nk)
    psi = _adj_cost_psi(I, k, mp.psi0)  # (Nk,Nk)
    R = _recovery_R(k, z, mp)  # (Nk,Nz)

    # initialize
    V = np.zeros((gp.Nk, gp.Nb, gp.Nz), dtype=np.float64)
    q_grid = np.ones((gp.Nz, gp.Nk, gp.Nb), dtype=np.float64) * (1.0 / (1.0 + mp.r))

    C = None
    pol_k = pol_b = None

    for it in range(gp.outer_max_iter):
        inner = _solve_inner(inner_method, V, q_grid, pi, I, psi, k, b, P, mp, gp)
        V, C, pol_k, pol_b = inner["V"], inner["C"], inner["pol_k"], inner["pol_b"]

        q_comp = _update_q(C, R, P, b, mp)
        q_next = gp.damping * q_comp + (1.0 - gp.damping) * q_grid
        err = np.max(np.abs(q_next - q_grid))
        q_grid = q_next

        # ===== DEBUG START (copy-paste) =====
        if verbose and (it in [0, 1, 2, 5, 10, 20, 30] or it == gp.outer_max_iter - 1):
            # q statistics
            qmin = float(np.min(q_grid))
            qmax = float(np.max(q_grid))
            print(
                f"[DEBUG q_grid] min={qmin:.6g} max={qmax:.6g} q_rf={1.0/(1.0+mp.r):.6g}"
            )

            # default probability and q at b_max (last b grid point)
            n_max = len(b) - 1
            q_bmax = q_grid[:, :, n_max]  # (Nz,Nk)

            pdef_list = []
            for iz in range(P.shape[0]):
                Piz = P[iz, :]  # (Nz,)
                for m in range(len(b)):  # <- (intentional) will overwrite; fix below
                    pass

            pdef_list = []
            Nz = P.shape[0]
            Nk = q_grid.shape[1]
            for iz in range(Nz):
                Piz = P[iz, :]
                for m in range(Nk):
                    D_next = (C[m, n_max, :] < 0.0).astype(np.float64)  # (Nz,)
                    pdef = float(np.dot(Piz, D_next))
                    pdef_list.append(pdef)

            pdef_arr = np.array(pdef_list, dtype=np.float64)
            print(
                f"[DEBUG at b_max] q(min,max)=({float(np.min(q_bmax)):.6g},{float(np.max(q_bmax)):.6g}) "
                f"p_def(mean,min,max)=({float(np.mean(pdef_arr)):.6g},{float(np.min(pdef_arr)):.6g},{float(np.max(pdef_arr)):.6g})"
            )
        # ===== DEBUG END =====

        if verbose:
            print(f"[GridBenchmark-{inner_method}] outer {it:03d}  ||dq||inf={err:.3e}")
        if err < gp.tol_q:
            break

    return {
        "k_grid": k,
        "b_grid": b,
        "z_grid": z,
        "P": P,
        "V_star": V,
        "C_star": C,
        "pol_k_idx": pol_k,
        "pol_b_idx": pol_b,
        "q_star": q_grid,
    }

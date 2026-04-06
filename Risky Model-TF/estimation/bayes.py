"""Bayesian estimation for the risky-debt model.

This module implements a Bayesian estimation layer that is materially aligned
with the user's written plan:

* estimate only ``(theta, psi0, alpha)``;
* keep ``(rho, sigma_eps, r, tau, delta, eta0, eta1)`` fixed;
* hide productivity during estimation and treat it as latent on a fixed
  Rouwenhorst grid;
* evaluate a deterministic forward-filter likelihood; and
* sample the posterior with TensorFlow Probability kernels.

The exact structural solve remains grid-based and NumPy-heavy, so the default
target is Hamiltonian Monte Carlo with a finite-difference gradient bridge.
Random-walk Metropolis is retained only as an explicit fallback when a user
requests robustness over the gradient-based default.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Tuple

import json
import os

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from risky_debt.config import ModelParams
from risky_debt.grid_benchmark import GridBenchParams, solve_grid_benchmark
from risky_debt.grid_compare import interp_grid_3d

from .bayes_reporting import BayesianArtifactPaths, BayesianArtifactWriter
from .common import make_json_serializable
from .filters import FiniteStateForwardFilter
from .obs_model import (
    ObservationNoiseConfig,
    PanelObservations,
    PathObservations,
    bernoulli_logpmf,
    extract_panel_observations,
    gaussian_logpdf,
    sigmoid,
)

tfd = tfp.distributions


@dataclass(frozen=True)
class StructuralSolveConfig:
    """Configuration of the coarse structural benchmark used inside MCMC."""

    Nk: int = 12
    Nb: int = 13
    Nz: int = 5
    k_max: float = 8.0
    z_m: float = 4.5
    outer_max_iter: int = 10
    inner_max_iter: int = 120
    mpi_eval_sweeps: int = 5
    mpi_max_iter: int = 60
    tol_q: float = 5e-3
    tol_v: float = 1e-4
    damping: float = 0.9
    inner_method: str = "mpi"

    def to_grid_params(self) -> GridBenchParams:
        """Convert to the benchmark solver's configuration dataclass."""
        return GridBenchParams(
            Nk=self.Nk,
            Nb=self.Nb,
            Nz=self.Nz,
            k_max=self.k_max,
            z_m=self.z_m,
            outer_max_iter=self.outer_max_iter,
            tol_q=self.tol_q,
            damping=self.damping,
            inner_max_iter=self.inner_max_iter,
            tol_V=self.tol_v,
            mpi_eval_sweeps=self.mpi_eval_sweeps,
            mpi_max_iter=self.mpi_max_iter,
        )


@dataclass(frozen=True)
class MCMCConfig:
    """Sampling controls for the Bayesian block."""

    kernel: str = "hmc"
    num_results: int = 48
    num_burnin: int = 48
    num_chains: int = 1
    step_size: float = 0.04
    leapfrog_steps: int = 4
    finite_diff_eps: float = 2e-3
    max_obs: int = 0
    max_paths: int = 0
    seed: int = 123


class CandidateBenchmarkCache:
    """Memoized structural solves for candidate parameter vectors."""

    def __init__(self, mp_base: ModelParams, solve_cfg: StructuralSolveConfig) -> None:
        """Initialize CandidateBenchmarkCache."""
        self.mp_base = mp_base
        self.solve_cfg = solve_cfg
        self._cache: Dict[Tuple[float, float, float], Dict[str, np.ndarray]] = {}

    @staticmethod
    def _key(theta: float, psi0: float, alpha: float) -> Tuple[float, float, float]:
        """Return the cache key for the current parameter vector."""
        return (round(float(theta), 5), round(float(psi0), 5), round(float(alpha), 5))

    def get(self, theta: float, psi0: float, alpha: float) -> Dict[str, np.ndarray]:
        """Return the cached coarse benchmark for a candidate parameter vector."""
        key = self._key(theta, psi0, alpha)
        if key not in self._cache:
            mp_cand = replace(self.mp_base, theta=float(theta), psi0=float(psi0), alpha=float(alpha))
            gp = self.solve_cfg.to_grid_params()
            self._cache[key] = solve_grid_benchmark(
                mp=mp_cand,
                gp=gp,
                inner_method=self.solve_cfg.inner_method,
                verbose=False,
            )
        return self._cache[key]


class BayesianRiskyDebtEstimator:
    """Bayesian estimator using a deterministic finite-state forward filter."""

    def __init__(
        self,
        mp_base: ModelParams,
        observations: PanelObservations,
        solve_cfg: StructuralSolveConfig,
        noise_cfg: ObservationNoiseConfig,
        mcmc_cfg: MCMCConfig,
    ) -> None:
        """Initialize BayesianRiskyDebtEstimator."""
        self.mp_base = mp_base
        self.solve_cfg = solve_cfg
        self.noise_cfg = noise_cfg
        self.mcmc_cfg = mcmc_cfg
        self.observations = observations.subset(max_paths=mcmc_cfg.max_paths, max_obs=mcmc_cfg.max_obs)
        self.cache = CandidateBenchmarkCache(mp_base=mp_base, solve_cfg=solve_cfg)
        self.forward_filter = FiniteStateForwardFilter()
        self.priors = self._make_priors()
        self.series_length = int(self.observations.t_eff)
        self.num_paths_used = int(self.observations.n_paths)
        self.total_observations_used = int(self.series_length * self.num_paths_used)

    def _make_priors(self) -> Dict[str, tfd.Distribution]:
        """Create priors matching the written Bayesian plan."""
        return {
            "theta": tfd.Beta(concentration1=2.0, concentration0=2.0),
            "psi0": tfd.LogNormal(loc=np.log(float(self.mp_base.psi0)), scale=0.75),
            "alpha": tfd.Beta(concentration1=2.0, concentration0=2.0),
        }

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid helper."""
        return 1.0 / (1.0 + np.exp(-x))

    def unconstrained_from_model_params(self) -> np.ndarray:
        """Map baseline parameters to the unconstrained MCMC space."""
        theta = float(np.clip(self.mp_base.theta, 1e-6, 1.0 - 1e-6))
        alpha = float(np.clip(self.mp_base.alpha, 1e-6, 1.0 - 1e-6))
        psi0 = float(max(self.mp_base.psi0, 1e-6))
        return np.asarray(
            [
                np.log(theta / (1.0 - theta)),
                np.log(psi0),
                np.log(alpha / (1.0 - alpha)),
            ],
            dtype=np.float32,
        )

    def _constrained_from_u(self, u: np.ndarray) -> Tuple[float, float, float, float]:
        """Map unconstrained parameters into ``(theta, psi0, alpha, log|J|)``."""
        u = np.asarray(u, dtype=float)
        theta = float(self._sigmoid(u[0]))
        psi0 = float(np.exp(u[1]))
        alpha = float(self._sigmoid(u[2]))
        log_j = (
            np.log(max(theta * (1.0 - theta), 1e-12))
            + float(u[1])
            + np.log(max(alpha * (1.0 - alpha), 1e-12))
        )
        return theta, psi0, alpha, log_j

    def _prior_log_prob(self, theta: float, psi0: float, alpha: float) -> float:
        """Evaluate the structural-parameter prior density."""
        lp = 0.0
        lp += float(self.priors["theta"].log_prob(theta).numpy())
        lp += float(self.priors["psi0"].log_prob(psi0).numpy())
        lp += float(self.priors["alpha"].log_prob(alpha).numpy())
        return lp

    def _candidate_log_emissions(self, benchmark: Dict[str, np.ndarray], mp_cand: ModelParams, obs: PathObservations) -> np.ndarray:
        """Build emission log densities for one observed path and latent grid."""
        T = len(obs.k)
        k_grid = np.asarray(benchmark["k_grid"], dtype=float)
        b_grid = np.asarray(benchmark["b_grid"], dtype=float)
        z_grid = np.asarray(benchmark["z_grid"], dtype=float)
        P = np.asarray(benchmark["P"], dtype=float)
        kp_star = np.asarray(benchmark["policy_kp_star"], dtype=float)
        bp_star = np.asarray(benchmark["policy_bp_star"], dtype=float)
        c_star = np.asarray(benchmark["C_star"], dtype=float)
        q_star = np.asarray(benchmark["q_star"], dtype=float)
        log_emissions = np.zeros((T, z_grid.size), dtype=np.float32)

        beta = 1.0 / (1.0 + float(mp_cand.r))
        tau = float(mp_cand.tau)
        delta = float(mp_cand.delta)
        eta0 = float(mp_cand.eta0)
        eta1 = float(mp_cand.eta1)
        k_min = float(mp_cand.k_min)
        q_min = float(mp_cand.q_min)
        q_max = float(mp_cand.q_max)
        kappa_issue = float(self.noise_cfg.kappa_issue)
        kappa_value = float(self.noise_cfg.kappa_value)

        for t in range(T):
            k_t = float(max(obs.k[t], mp_cand.k_min))
            b_t = float(obs.b[t])
            for j, z_t in enumerate(z_grid):
                kp_pred = float(
                    interp_grid_3d(k_grid, b_grid, z_grid, kp_star, np.asarray([k_t]), np.asarray([b_t]), np.asarray([z_t]))[0]
                )
                bp_pred = float(
                    interp_grid_3d(k_grid, b_grid, z_grid, bp_star, np.asarray([k_t]), np.asarray([b_t]), np.asarray([z_t]))[0]
                )
                kp_pred = max(kp_pred, mp_cand.k_min)
                q_pred = float(
                    interp_grid_3d(z_grid, k_grid, b_grid, q_star, np.asarray([z_t]), np.asarray([kp_pred]), np.asarray([bp_pred]))[0]
                )
                q_pred = float(np.clip(q_pred, q_min, q_max))
                kp_vec = np.full_like(z_grid, kp_pred, dtype=float)
                bp_vec = np.full_like(z_grid, bp_pred, dtype=float)
                cont_next = interp_grid_3d(k_grid, b_grid, z_grid, c_star, kp_vec, bp_vec, z_grid)
                cont_weights = sigmoid(cont_next, kappa_value)
                p_continue = float(np.dot(P[j], cont_weights))
                p_continue = float(np.clip(p_continue, 1e-6, 1.0 - 1e-6))
                p_default = 1.0 - p_continue

                profit = float(z_t * (max(k_t, k_min) ** mp_cand.theta))
                investment = float(kp_pred - (1.0 - delta) * k_t)
                adj_cost = float(mp_cand.psi0 * investment * investment / (2.0 * max(k_t, k_min)))
                e_base = (1.0 - tau) * profit - adj_cost - investment + bp_pred * q_pred - b_t
                r_tilde = (1.0 / q_pred) - 1.0
                debt_mask = 1.0 if bp_pred > 0.0 else 0.0
                tax_shield = beta * tau * r_tilde * bp_pred * p_continue * debt_mask
                e_total = e_base + tax_shield
                issue_prob = float(sigmoid(np.asarray([-e_total]), kappa_issue)[0])
                eta = -(eta0 - eta1 * e_total) * issue_prob
                d_pred = e_total + eta

                ll = 0.0
                ll += gaussian_logpdf(obs.k_next[t], kp_pred, self.noise_cfg.sigma_k)
                ll += gaussian_logpdf(obs.b_next[t], bp_pred, self.noise_cfg.sigma_b)
                ll += gaussian_logpdf(obs.q[t], q_pred, self.noise_cfg.sigma_q)
                ll += gaussian_logpdf(obs.d[t], d_pred, self.noise_cfg.sigma_d)
                ll += bernoulli_logpmf(obs.default[t], p_default)
                log_emissions[t, j] = float(ll)
        return log_emissions.astype(np.float32)

    def _log_likelihood(self, theta: float, psi0: float, alpha: float) -> float:
        """Evaluate the deterministic forward-filter likelihood on all used paths."""
        benchmark = self.cache.get(theta=theta, psi0=psi0, alpha=alpha)
        mp_cand = replace(self.mp_base, theta=float(theta), psi0=float(psi0), alpha=float(alpha))
        P_np = np.asarray(benchmark["P"], dtype=np.float32)
        init_probs = self._stationary_probs_np(P_np)
        total_ll = 0.0
        for p in range(self.num_paths_used):
            path_obs = self.observations.path(p)
            log_emissions_np = self._candidate_log_emissions(benchmark=benchmark, mp_cand=mp_cand, obs=path_obs)
            result = self.forward_filter.run(
                log_emissions=tf.convert_to_tensor(log_emissions_np, dtype=tf.float32),
                init_probs=tf.convert_to_tensor(init_probs, dtype=tf.float32),
                transition_matrix=tf.convert_to_tensor(P_np, dtype=tf.float32),
            )
            total_ll += float(result.log_likelihood.numpy())
        return float(total_ll)

    @staticmethod
    def _stationary_probs_np(P: np.ndarray) -> np.ndarray:
        """Compute a stationary distribution for a transition matrix in NumPy."""
        w, v = np.linalg.eig(P.T)
        idx = int(np.argmin(np.abs(w - 1.0)))
        vec = np.real(v[:, idx])
        vec = np.maximum(vec, 0.0)
        if float(vec.sum()) <= 0.0:
            vec = np.ones(P.shape[0], dtype=float)
        vec = vec / vec.sum()
        return vec.astype(np.float32)

    def target_log_prob_numpy(self, u_batch: np.ndarray) -> np.ndarray:
        """Evaluate the unconstrained posterior log density for one or more points."""
        u_batch = np.asarray(u_batch, dtype=np.float32)
        was_vector = u_batch.ndim == 1
        if was_vector:
            u_batch = u_batch[None, :]

        out = np.full((u_batch.shape[0],), -1.0e30, dtype=np.float32)
        for i, u in enumerate(u_batch):
            theta, psi0, alpha, log_j = self._constrained_from_u(u)
            if not (0.0 < theta < 1.0 and psi0 > 0.0 and 0.0 < alpha < 1.0):
                continue
            try:
                lp = self._prior_log_prob(theta=theta, psi0=psi0, alpha=alpha)
                ll = self._log_likelihood(theta=theta, psi0=psi0, alpha=alpha)
                out[i] = np.float32(lp + ll + log_j)
            except Exception:
                out[i] = np.float32(-1.0e30)

        return out[0] if was_vector else out

    def target_log_prob_grad_numpy(self, u_batch: np.ndarray) -> np.ndarray:
        """Finite-difference gradient of the unconstrained posterior."""
        u_batch = np.asarray(u_batch, dtype=np.float32)
        if u_batch.ndim == 1:
            u_batch = u_batch[None, :]
        eps = float(self.mcmc_cfg.finite_diff_eps)
        grad = np.zeros_like(u_batch, dtype=np.float32)
        for r in range(u_batch.shape[0]):
            u0 = u_batch[r].copy()
            for d in range(u0.size):
                up = u0.copy()
                um = u0.copy()
                up[d] += eps
                um[d] -= eps
                fp = float(self.target_log_prob_numpy(up))
                fm = float(self.target_log_prob_numpy(um))
                grad[r, d] = np.float32((fp - fm) / (2.0 * eps))
        return grad

    def _tf_target_log_prob(self):
        """Return a TensorFlow-compatible target log-probability function."""
        if self.mcmc_cfg.kernel == "rwm":
            def _target(u: tf.Tensor) -> tf.Tensor:
                flat = tf.reshape(tf.convert_to_tensor(u, tf.float32), (-1, 3))
                vals = tf.numpy_function(self.target_log_prob_numpy, [flat], tf.float32)
                vals.set_shape([None])
                return tf.reshape(vals, tf.shape(u)[:-1])
            return _target

        @tf.custom_gradient
        def _target_with_grad(flat_u: tf.Tensor):
            vals = tf.numpy_function(self.target_log_prob_numpy, [flat_u], tf.float32)
            vals.set_shape([None])
            def grad(dy: tf.Tensor) -> tf.Tensor:
                """Return the posterior value and finite-difference gradient at one point."""
                g = tf.numpy_function(self.target_log_prob_grad_numpy, [flat_u], tf.float32)
                g.set_shape([None, 3])
                return g * tf.reshape(tf.cast(dy, tf.float32), (-1, 1))
            return vals, grad

        def _target(u: tf.Tensor) -> tf.Tensor:
            flat = tf.reshape(tf.convert_to_tensor(u, tf.float32), (-1, 3))
            vals = _target_with_grad(flat)
            return tf.reshape(vals, tf.shape(u)[:-1])
        return _target

    def run(self, out_dir: str, figures_dir: str | None = None) -> Dict[str, object]:
        """Run posterior sampling and save numerical and figure artifacts.

        Args:
            out_dir: Directory for numerical Bayesian artifacts.
            figures_dir: Optional directory for report-ready HMC figures. When
                omitted, figures are stored in ``<out_dir>/figures``.
        """
        os.makedirs(out_dir, exist_ok=True)
        figure_root = figures_dir or os.path.join(out_dir, "figures")
        target_log_prob_fn = self._tf_target_log_prob()
        init_u0 = self.unconstrained_from_model_params()
        rng = np.random.default_rng(self.mcmc_cfg.seed)
        init_u = np.tile(init_u0[None, :], (self.mcmc_cfg.num_chains, 1)).astype(np.float32)
        if self.mcmc_cfg.num_chains > 1:
            init_u += rng.normal(0.0, 0.05, size=init_u.shape).astype(np.float32)

        if self.mcmc_cfg.kernel == "hmc":
            base_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=target_log_prob_fn,
                step_size=self.mcmc_cfg.step_size,
                num_leapfrog_steps=self.mcmc_cfg.leapfrog_steps,
            )
            kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                inner_kernel=base_kernel,
                num_adaptation_steps=max(1, int(0.8 * self.mcmc_cfg.num_burnin)),
                target_accept_prob=0.65,
            )
            trace_fn = lambda _, pkr: {
                "is_accepted": pkr.inner_results.is_accepted,
                "target_log_prob": pkr.inner_results.accepted_results.target_log_prob,
            }
        else:
            kernel = tfp.mcmc.RandomWalkMetropolis(
                target_log_prob_fn=target_log_prob_fn,
                new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=self.mcmc_cfg.step_size),
            )
            trace_fn = lambda _, pkr: {
                "is_accepted": pkr.is_accepted,
                "target_log_prob": pkr.accepted_results.target_log_prob,
            }

        draws_u, trace = tfp.mcmc.sample_chain(
            num_results=self.mcmc_cfg.num_results,
            num_burnin_steps=self.mcmc_cfg.num_burnin,
            current_state=tf.convert_to_tensor(init_u, dtype=tf.float32),
            kernel=kernel,
            trace_fn=trace_fn,
            seed=self.mcmc_cfg.seed,
        )

        draws_u_np = draws_u.numpy()
        theta = self._sigmoid(draws_u_np[..., 0])
        psi0 = np.exp(draws_u_np[..., 1])
        alpha = self._sigmoid(draws_u_np[..., 2])

        theta_mean = float(np.mean(theta))
        psi0_mean = float(np.mean(psi0))
        alpha_mean = float(np.mean(alpha))
        ll_at_mean = self._log_likelihood(theta=theta_mean, psi0=psi0_mean, alpha=alpha_mean)

        diagnostics = self._diagnostics(theta=theta, psi0=psi0, alpha=alpha, trace=trace)
        results = {
            "method": "BayesianMCMC",
            "sampler": "HamiltonianMonteCarlo" if self.mcmc_cfg.kernel == "hmc" else "RandomWalkMetropolis",
            "kernel": self.mcmc_cfg.kernel,
            "gradient_mode": "finite_difference_custom_gradient" if self.mcmc_cfg.kernel == "hmc" else "not_applicable",
            "estimated_parameters": ["theta", "psi0", "alpha"],
            "fixed_parameters": {
                "rho": float(self.mp_base.rho),
                "sigma_eps": float(self.mp_base.sigma_eps),
                "r": float(self.mp_base.r),
                "tau": float(self.mp_base.tau),
                "delta": float(self.mp_base.delta),
                "eta0": float(self.mp_base.eta0),
                "eta1": float(self.mp_base.eta1),
            },
            "posterior_mean": {"theta": theta_mean, "psi0": psi0_mean, "alpha": alpha_mean},
            "posterior_summary": {
                "theta": self._summary_stats(theta),
                "psi0": self._summary_stats(psi0),
                "alpha": self._summary_stats(alpha),
            },
            "true_parameters": {
                "theta": float(self.mp_base.theta),
                "psi0": float(self.mp_base.psi0),
                "alpha": float(self.mp_base.alpha),
            },
            "absolute_error": {
                "theta": abs(theta_mean - float(self.mp_base.theta)),
                "psi0": abs(psi0_mean - float(self.mp_base.psi0)),
                "alpha": abs(alpha_mean - float(self.mp_base.alpha)),
            },
            "filter": "deterministic_forward_filter",
            "observation_usage": {
                "num_paths_used": self.num_paths_used,
                "observations_per_path": self.series_length,
                "total_observations_used": self.total_observations_used,
            },
            "log_likelihood_at_posterior_mean": ll_at_mean,
            "noise_config": make_json_serializable(self.noise_cfg.__dict__),
            "solve_config": make_json_serializable(self.solve_cfg.__dict__),
            "diagnostics": diagnostics,
        }

        accepted_np = np.asarray(trace["is_accepted"], dtype=bool)
        target_log_prob_np = np.asarray(trace["target_log_prob"], dtype=np.float32)
        writer = BayesianArtifactWriter(
            BayesianArtifactPaths.from_dirs(
                estimation_dir=out_dir,
                figures_dir=figure_root,
            )
        )
        artifact_paths = writer.write_all(
            results=results,
            theta=theta,
            psi0=psi0,
            alpha=alpha,
            draws_u=draws_u_np,
            accepted=accepted_np,
            target_log_prob=target_log_prob_np,
            true_parameters=results["true_parameters"],
        )
        results["artifact_paths"] = make_json_serializable(artifact_paths)
        return results

    @staticmethod
    def _summary_stats(x: np.ndarray) -> Dict[str, float]:
        """Posterior summary statistics for a sampled parameter array."""
        flat = np.asarray(x, dtype=float).reshape(-1)
        return {
            "mean": float(np.mean(flat)),
            "median": float(np.median(flat)),
            "std": float(np.std(flat, ddof=0)),
            "p05": float(np.quantile(flat, 0.05)),
            "p95": float(np.quantile(flat, 0.95)),
        }

    def _diagnostics(self, theta: np.ndarray, psi0: np.ndarray, alpha: np.ndarray, trace: Dict[str, tf.Tensor]) -> Dict[str, object]:
        """Compute convergence and sampling diagnostics."""
        accepted = np.asarray(trace["is_accepted"].numpy(), dtype=float)
        target_log_prob = np.asarray(trace["target_log_prob"].numpy(), dtype=float)
        diag: Dict[str, object] = {
            "acceptance_rate": float(np.mean(accepted)),
            "target_log_prob_mean": float(np.mean(target_log_prob)),
            "target_log_prob_last": float(np.mean(target_log_prob[-1])),
        }
        if self.mcmc_cfg.num_chains > 1:
            diag["rhat"] = {
                "theta": float(tfp.mcmc.potential_scale_reduction(np.transpose(theta, (1, 0)), independent_chain_ndims=1).numpy()),
                "psi0": float(tfp.mcmc.potential_scale_reduction(np.transpose(psi0, (1, 0)), independent_chain_ndims=1).numpy()),
                "alpha": float(tfp.mcmc.potential_scale_reduction(np.transpose(alpha, (1, 0)), independent_chain_ndims=1).numpy()),
            }
        else:
            diag["rhat"] = {"theta": None, "psi0": None, "alpha": None}

        ess_kwargs = {"cross_chain_dims": [1]} if self.mcmc_cfg.num_chains > 1 else {}
        diag["effective_sample_size"] = {
            "theta": float(np.min(tfp.mcmc.effective_sample_size(theta, **ess_kwargs).numpy())),
            "psi0": float(np.min(tfp.mcmc.effective_sample_size(psi0, **ess_kwargs).numpy())),
            "alpha": float(np.min(tfp.mcmc.effective_sample_size(alpha, **ess_kwargs).numpy())),
        }
        return diag


def estimate_bayesian_posterior(
    out_dir: str,
    mp_true: ModelParams,
    data: Dict[str, np.ndarray],
    kernel: str = "hmc",
    num_results: int = 48,
    num_burnin: int = 48,
    num_chains: int = 1,
    step_size: float = 0.04,
    max_obs: int = 0,
    max_paths: int = 0,
    seed: int = 123,
    figures_dir: str | None = None,
) -> Dict[str, object]:
    """Run Bayesian estimation for the risky-debt model.

    Args:
        out_dir: Directory where posterior artifacts are written.
        mp_true: Baseline parameter block used to generate the synthetic sample.
        data: Synthetic observed dataset in the flattened estimation format.
        kernel: ``"hmc"`` for the default finite-difference Hamiltonian Monte
            Carlo path or ``"rwm"`` for the explicit fallback.
        num_results: Number of retained MCMC draws.
        num_burnin: Number of burn-in draws.
        num_chains: Number of parallel chains.
        step_size: Proposal scale or HMC step size.
        max_obs: Maximum time observations per path. Non-positive means use the
            full available length.
        max_paths: Maximum number of simulated paths. Non-positive means use all
            available paths.
        seed: Random seed.
        figures_dir: Optional directory for Bayesian diagnostic figures.
    """
    observations = extract_panel_observations(data)
    solve_cfg = StructuralSolveConfig()
    noise_cfg = ObservationNoiseConfig()
    mcmc_cfg = MCMCConfig(
        kernel=str(kernel),
        num_results=int(num_results),
        num_burnin=int(num_burnin),
        num_chains=int(num_chains),
        step_size=float(step_size),
        max_obs=int(max_obs),
        max_paths=int(max_paths),
        seed=int(seed),
    )
    estimator = BayesianRiskyDebtEstimator(
        mp_base=mp_true,
        observations=observations,
        solve_cfg=solve_cfg,
        noise_cfg=noise_cfg,
        mcmc_cfg=mcmc_cfg,
    )
    return estimator.run(out_dir=out_dir, figures_dir=figures_dir)


def estimate_hmc(
    out_dir: str,
    mp_true: ModelParams,
    data: Dict[str, np.ndarray],
    kernel: str = "hmc",
    num_results: int = 48,
    num_burnin: int = 48,
    num_chains: int = 1,
    step_size: float = 0.04,
    num_particles: int = 0,
    obs_sigma_lnz: float = 0.02,
    seed: int = 123,
    figures_dir: str | None = None,
) -> Dict[str, object]:
    """Compatibility wrapper around :func:`estimate_bayesian_posterior`.

    The legacy arguments ``num_particles`` and ``obs_sigma_lnz`` are no longer
    part of the deterministic forward-filter implementation and are ignored here
    only to preserve older imports.
    """
    del num_particles, obs_sigma_lnz
    return estimate_bayesian_posterior(
        out_dir=out_dir,
        mp_true=mp_true,
        data=data,
        kernel=kernel,
        num_results=num_results,
        num_burnin=num_burnin,
        num_chains=num_chains,
        step_size=step_size,
        seed=seed,
        figures_dir=figures_dir,
    )

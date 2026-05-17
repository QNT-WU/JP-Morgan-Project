"""Two-step SMM estimators and policy-training helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize

from ..config import ModelParams, NetParams, Obj2Params, TrainParams
from ..networks import MultiplierNet, PolicyNet
from ..objectives import obj2_batch_loss
from ..simulation import set_global_seed, simulate_ergodic_dataset
from .common import PARAMETER_NAMES, params_from_vector, structural_params_from_model, transform_params_to_tilde, transform_tilde_to_params, update_model_params, vector_from_params
from .moments import CRNDesign, MomentSpec, PathDataset, build_default_moment_spec, compute_moments, compute_smm_moment_series, path_sample_size, simulate_paths_crn, summarize_smm_moments
from .weighting import CovarianceEstimate, estimate_weighting_matrix, numerical_jacobian, sandwich_parameter_covariance
from .gmm import jacobian_identification_diagnostics


def _clip_and_apply(opt: tf.keras.optimizers.Optimizer, grads, vars_, clip: float) -> None:
    """Clip gradients to a finite norm and apply them with ``optimizer``."""
    grads, _ = tf.clip_by_global_norm(grads, clip)
    opt.apply_gradients(zip(grads, vars_))


def _train_policy_obj2_inner_with_diagnostics(
    *,
    mp: ModelParams,
    npol: NetParams,
    tp: TrainParams,
    warm_start_policy: Optional[PolicyNet] = None,
) -> tuple[PolicyNet, Dict[str, Any]]:
    """Train an Objective 2 inner policy and return solver diagnostics.

    SMM repeatedly solves the structural model at candidate parameter vectors.
    This helper records the final loss, iteration budget, and convergence flag
    so the SMM solution cache has an audit trail rather than only a policy.
    """
    set_global_seed(tp.seed)
    policy = PolicyNet(npol, mp.k_min, mp.k_max)
    _ = policy(tf.zeros((1, 2), dtype=tf.float32))
    if warm_start_policy is not None:
        policy.set_weights(warm_start_policy.get_weights())
    multiplier = MultiplierNet(npol)
    _ = multiplier(tf.zeros((1, 2), dtype=tf.float32))
    opt_policy = tf.keras.optimizers.Adam(tp.lr_policy)
    opt_multiplier = tf.keras.optimizers.Adam(tp.lr_policy)
    op2 = Obj2Params()
    k_buf, z_buf = simulate_ergodic_dataset(policy, mp, tp, seed=tp.seed + 11)
    rng = np.random.default_rng(tp.seed + 99)
    loss_history: List[float] = []
    failure_reason = ""
    convergence_flag = "completed"
    for epoch in range(1, tp.epochs + 1):
        if epoch == 1 or (epoch % tp.ergodic_refresh_every == 0):
            k_buf, z_buf = simulate_ergodic_dataset(policy, mp, tp, seed=tp.seed + 110 + epoch)
        for _ in range(tp.steps_per_epoch):
            idx = rng.choice(len(k_buf), size=tp.batch_size, replace=True)
            k = tf.convert_to_tensor(k_buf[idx], tf.float32)
            z = tf.convert_to_tensor(z_buf[idx], tf.float32)
            variables = policy.trainable_variables + multiplier.trainable_variables
            with tf.GradientTape() as tape:
                loss = obj2_batch_loss(policy, multiplier, mp, op2, k, z)
            if not bool(tf.reduce_all(tf.math.is_finite(loss)).numpy()):
                convergence_flag = "failed"
                failure_reason = "non-finite inner Objective 2 loss"
                break
            grads = tape.gradient(loss, variables)
            n_policy = len(policy.trainable_variables)
            _clip_and_apply(opt_policy, grads[:n_policy], policy.trainable_variables, tp.grad_clip)
            _clip_and_apply(opt_multiplier, grads[n_policy:], multiplier.trainable_variables, tp.grad_clip)
            loss_history.append(float(loss.numpy()))
        if convergence_flag == "failed":
            break
    final_loss = float(loss_history[-1]) if loss_history else float("inf")
    diagnostics: Dict[str, Any] = {
        "convergence_flag": convergence_flag,
        "failure_reason": failure_reason,
        "final_training_loss": final_loss,
        "min_training_loss": float(np.min(loss_history)) if loss_history else float("inf"),
        "mean_last_10_loss": float(np.mean(loss_history[-10:])) if loss_history else float("inf"),
        "epochs": int(tp.epochs),
        "steps_per_epoch": int(tp.steps_per_epoch),
        "training_steps_completed": int(len(loss_history)),
    }
    return policy, diagnostics


def _train_policy_obj2_inner(
    *,
    mp: ModelParams,
    npol: NetParams,
    tp: TrainParams,
    warm_start_policy: Optional[PolicyNet] = None,
) -> PolicyNet:
    """Train an Objective 2 policy model used inside SMM candidate evaluation."""
    policy, _ = _train_policy_obj2_inner_with_diagnostics(
        mp=mp, npol=npol, tp=tp, warm_start_policy=warm_start_policy
    )
    return policy


def path_sample_size_from_design(design: CRNDesign) -> int:
    """Return the effective sample size implied by a simulation design."""
    return int(design.n_paths * design.T)


@dataclass
class StartRecord:
    """Record optimization diagnostics for one starting value."""
    stage: str
    start_id: int
    x0: List[float]
    x_hat: List[float]
    success: bool
    status: int
    message: str
    nfev: int
    nit: int
    final_loss: float
    elapsed_seconds: float
    is_best: bool = False


@dataclass
class StageSummary:
    """Summarize one stage of a two-step estimation routine."""
    stage: str
    starts: List[StartRecord]
    best_start_id: int
    best_x: Dict[str, float]
    best_loss: float
    success: bool
    status: int
    message: str
    nfev: int
    nit: int

    @property
    def n_starts(self) -> int:
        """Return the number of starting values used in this stage."""
        return len(self.starts)


@dataclass
class SMMMethodResult:
    """Store the final outputs and diagnostics for one SMM variant."""
    method_name: str
    weight_method: str
    stage1: StageSummary
    stage2: StageSummary
    final_params: Dict[str, float]
    observed_moments: Dict[str, float]
    simulated_moments: Dict[str, float]
    moment_gaps: Dict[str, float]
    summarized_observed_moments: Dict[str, float]
    summarized_simulated_moments: Dict[str, float]
    covariance_info: CovarianceEstimate
    parameter_covariance: np.ndarray
    standard_errors: Dict[str, float]
    simulation_adjustment: float
    elapsed_seconds: float
    cache_hits: int = 0
    cache_misses: int = 0
    unique_structural_solves: int = 0
    identification_diagnostics: Dict[str, float | str | int] = None

    def to_flat_dict(self) -> Dict[str, float | str | bool]:
        """Flatten the result object into a serializable dictionary."""
        out: Dict[str, float | str | bool] = {
            "method": self.method_name,
            "weight_method": self.weight_method,
            "stage1_success": bool(self.stage1.success),
            "stage2_success": bool(self.stage2.success),
            "final_success": bool(self.stage2.success),
            "stage1_best_start_id": int(self.stage1.best_start_id),
            "stage2_best_start_id": int(self.stage2.best_start_id),
            "stage1_best_loss": float(self.stage1.best_loss),
            "stage2_best_loss": float(self.stage2.best_loss),
            "stage1_n_starts": int(self.stage1.n_starts),
            "stage2_n_starts": int(self.stage2.n_starts),
            "stage1_status": int(self.stage1.status),
            "stage2_status": int(self.stage2.status),
            "stage1_message": str(self.stage1.message),
            "stage2_message": str(self.stage2.message),
            "stage1_nfev": int(self.stage1.nfev),
            "stage2_nfev": int(self.stage2.nfev),
            "elapsed_seconds": float(self.elapsed_seconds),
            "condition_number": float(self.covariance_info.condition_number),
            "ridge": float(self.covariance_info.ridge),
            "min_eigenvalue": float(self.covariance_info.min_eigenvalue),
            "lags": int(self.covariance_info.lags),
            "simulation_adjustment": float(self.simulation_adjustment),
            "cache_hits": int(self.cache_hits),
            "cache_misses": int(self.cache_misses),
            "unique_structural_solves": int(self.unique_structural_solves),
        }
        for name in PARAMETER_NAMES:
            out[f"{name}_hat"] = float(self.final_params[name])
            out[f"se_{name}"] = float(self.standard_errors[name])
            out[f"{name}_stage1"] = float(self.stage1.best_x[name])
        if self.identification_diagnostics:
            for key, val in self.identification_diagnostics.items():
                out[key] = val
        for name, gap in self.moment_gaps.items():
            out[f"gap_{name}"] = float(gap)
        return out




@dataclass
class CandidateSolveRecord:
    """Cache one solved-and-simulated SMM candidate.

    The basic model cache stores the candidate structural parameters, the
    trained policy used for simulation, the simulated dataset, the resulting
    moment vector, and minimal solver diagnostics. This implements the
    document requirement that SMM should not silently re-solve repeated or
    numerically equivalent candidates.
    """

    key: Tuple[float, ...]
    params: Dict[str, float]
    policy: PolicyNet
    dataset: PathDataset
    model_params: ModelParams
    moment_vector: np.ndarray
    elapsed_seconds: float
    convergence_flag: str
    solver_diagnostics: Dict[str, Any]
    failure_reason: str = ""


class TwoStepSMMEstimator:
    """Estimate structural parameters with the two-step SMM workflow."""
    def __init__(self, *, mp_template: ModelParams, npol: NetParams, inner_tp: TrainParams, observed_dataset: PathDataset, simulation_design: CRNDesign, moment_spec: Optional[MomentSpec] = None, ridge: float = 1e-8, hac_lags: int | None = None, seed: int = 123):
        """Initialize the two-step SMM estimator and cache observed moments.

        The constructor stores observed synthetic data, moment definitions,
        optimization controls, and package hyperparameters required to solve the
        model repeatedly at candidate parameter values.
        """
        self.mp_template = mp_template
        self.npol = npol
        self.inner_tp = inner_tp
        self.observed_dataset = observed_dataset
        self.simulation_design = simulation_design
        self.spec = moment_spec or build_default_moment_spec()
        self.ridge = float(ridge)
        self.hac_lags = hac_lags
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.observed_moment_series = compute_smm_moment_series(observed_dataset, mp_template, self.spec)
        self.observed_moments = compute_moments(observed_dataset, mp_template, self.spec)
        self.observed_moment_vector = np.asarray([self.observed_moments[name] for name in self.spec.names], dtype=np.float64)
        self.observed_summary = summarize_smm_moments(self.observed_moments)
        self.n_obs_moments = int(self.observed_moment_series.shape[0])
        self.solve_cache: Dict[Tuple[float, ...], CandidateSolveRecord] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_round_digits = 6
        J = path_sample_size_from_design(simulation_design) / max(path_sample_size(observed_dataset), 1)
        self.simulation_adjustment = 1.0 + (1.0 / max(J, 1e-12))

    def _candidate_model_params(self, theta_tilde: np.ndarray) -> ModelParams:
        """Convert an unconstrained optimization vector into feasible model parameters."""
        return update_model_params(self.mp_template, transform_tilde_to_params(theta_tilde))

    def _cache_key(self, theta_tilde: np.ndarray) -> Tuple[float, ...]:
        """Return the rounded cache key for a candidate parameter vector."""
        return tuple(np.round(np.asarray(theta_tilde, dtype=np.float64), self.cache_round_digits))

    def _solve_candidate_record(self, theta_tilde: np.ndarray) -> CandidateSolveRecord:
        """Solve, simulate, and cache one SMM candidate parameter vector."""
        key = self._cache_key(theta_tilde)
        if key in self.solve_cache:
            self.cache_hits += 1
            return self.solve_cache[key]

        self.cache_misses += 1
        t0 = time.perf_counter()
        mp_candidate = self._candidate_model_params(theta_tilde)
        policy, solver_diagnostics = _train_policy_obj2_inner_with_diagnostics(
            mp=mp_candidate,
            npol=self.npol,
            tp=self.inner_tp,
            warm_start_policy=None,
        )
        if solver_diagnostics.get("convergence_flag") == "failed":
            # Keep the failed candidate visible to the optimizer by returning a
            # large moment vector; the resulting objective is heavily penalized.
            moment_vector = np.full(len(self.spec.names), 1e6, dtype=np.float64)
            ds_candidate = simulate_paths_crn(policy=policy, mp=mp_candidate, design=self.simulation_design, burn_in=0)
            record = CandidateSolveRecord(
                key=key,
                params=structural_params_from_model(mp_candidate),
                policy=policy,
                dataset=ds_candidate,
                model_params=mp_candidate,
                moment_vector=moment_vector,
                elapsed_seconds=float(time.perf_counter() - t0),
                convergence_flag="failed",
                solver_diagnostics=solver_diagnostics,
                failure_reason=str(solver_diagnostics.get("failure_reason", "inner solve failed")),
            )
            self.solve_cache[key] = record
            return record
        ds_candidate = simulate_paths_crn(policy=policy, mp=mp_candidate, design=self.simulation_design, burn_in=0)
        m_dict = compute_moments(ds_candidate, mp_candidate, self.spec)
        moment_vector = np.asarray([m_dict[name] for name in self.spec.names], dtype=np.float64)
        record = CandidateSolveRecord(
            key=key,
            params=structural_params_from_model(mp_candidate),
            policy=policy,
            dataset=ds_candidate,
            model_params=mp_candidate,
            moment_vector=moment_vector,
            elapsed_seconds=float(time.perf_counter() - t0),
            convergence_flag=str(solver_diagnostics.get("convergence_flag", "completed")),
            solver_diagnostics=solver_diagnostics,
        )
        self.solve_cache[key] = record
        return record

    def _simulate_candidate(self, theta_tilde: np.ndarray) -> tuple[PathDataset, ModelParams]:
        """Return the cached simulated dataset and model parameters for a candidate."""
        record = self._solve_candidate_record(theta_tilde)
        return record.dataset, record.model_params

    def _moment_vector_from_tilde(self, theta_tilde: np.ndarray) -> np.ndarray:
        """Compute or retrieve simulated SMM target moments for a candidate vector."""
        return self._solve_candidate_record(theta_tilde).moment_vector.copy()

    def _objective(self, theta_tilde: np.ndarray, W: np.ndarray) -> float:
        """Evaluate the weighted SMM loss for an unconstrained parameter vector."""
        g = self.observed_moment_vector - self._moment_vector_from_tilde(theta_tilde)
        value = float(g.T @ W @ g)
        return value if np.isfinite(value) else 1e12

    def _build_starts(self, x_center: np.ndarray, n_starts: int, start_scale: float) -> List[np.ndarray]:
        """Create deterministic transformed starting points for SMM optimization."""
        x_center = np.asarray(x_center, dtype=np.float64)
        starts = [x_center.copy()]
        for _ in range(max(0, n_starts - 1)):
            starts.append(x_center + self.rng.normal(scale=start_scale, size=x_center.shape))
        return starts

    def _run_multistart(self, stage: str, starts: List[np.ndarray], W: np.ndarray, max_evals: int) -> StageSummary:
        """Run the local SMM optimizer from each start and retain diagnostics."""
        records: List[StartRecord] = []
        best_idx = 0
        best_loss = np.inf
        best_x = np.asarray(starts[0], dtype=np.float64)
        best_res = {"success": False, "status": -1, "message": "not run", "nfev": 0, "nit": 0}
        for sid, x0 in enumerate(starts):
            t0 = time.perf_counter()
            res = minimize(lambda x: self._objective(x, W), x0=np.asarray(x0, dtype=np.float64), method="Nelder-Mead", options={"maxfev": int(max_evals), "maxiter": int(max_evals), "disp": False})
            elapsed = float(time.perf_counter() - t0)
            loss = float(res.fun) if np.isfinite(res.fun) else 1e12
            rec = StartRecord(stage, sid, list(np.asarray(x0, dtype=np.float64)), list(np.asarray(res.x, dtype=np.float64)), bool(res.success), int(res.status), str(res.message), int(res.nfev), int(getattr(res, "nit", 0)), loss, elapsed, False)
            records.append(rec)
            if loss < best_loss:
                best_loss = loss
                best_x = np.asarray(res.x, dtype=np.float64)
                best_idx = sid
                best_res = {"success": bool(res.success), "status": int(res.status), "message": str(res.message), "nfev": int(res.nfev), "nit": int(getattr(res, "nit", 0))}
        records[best_idx].is_best = True
        return StageSummary(stage, records, best_idx, transform_tilde_to_params(best_x), float(best_loss), bool(best_res["success"]), int(best_res["status"]), str(best_res["message"]), int(best_res["nfev"]), int(best_res["nit"]))

    def fit(self, *, x0: np.ndarray | None = None, max_evals: int = 40, weight_methods: Iterable[str] = ("standard", "newey_west"), n_starts: int = 1, start_scale: float = 0.15) -> Dict[str, SMMMethodResult]:
        """Run the full estimation workflow and return method results."""
        if x0 is None:
            x0 = transform_params_to_tilde(**structural_params_from_model(self.mp_template))
        W1 = np.eye(len(self.spec.names), dtype=np.float64)
        starts1 = self._build_starts(np.asarray(x0, dtype=np.float64), n_starts=n_starts, start_scale=start_scale)
        stage1 = self._run_multistart("stage1", starts1, W1, max_evals=max_evals)
        total_t0 = time.perf_counter()
        results: Dict[str, SMMMethodResult] = {}
        for idx, method in enumerate(weight_methods):
            cov_info = estimate_weighting_matrix(self.observed_moment_series, method=method, ridge=self.ridge, lags=self.hac_lags)
            starts2 = self._build_starts(transform_params_to_tilde(**stage1.best_x), n_starts=n_starts, start_scale=start_scale)
            stage2 = self._run_multistart("stage2", starts2, cov_info.weight_matrix, max_evals=max_evals)
            x_stage2 = transform_params_to_tilde(**stage2.best_x)
            ds_final, mp_final = self._simulate_candidate(x_stage2)
            sim_moments = compute_moments(ds_final, mp_final, self.spec)
            sim_vec = np.asarray([sim_moments[name] for name in self.spec.names], dtype=np.float64)
            gaps = self.observed_moment_vector - sim_vec
            D = numerical_jacobian(self._moment_vector_from_tilde, x_stage2)
            V_tilde = sandwich_parameter_covariance(D, cov_info.covariance, cov_info.weight_matrix, self.n_obs_moments, simulation_adjustment=self.simulation_adjustment, ridge=self.ridge)
            params_vec = vector_from_params(stage2.best_x)
            t_vec = np.asarray(x_stage2, dtype=np.float64)
            delta = np.diag([params_vec[0] * (1.0 - params_vec[0]), params_vec[1] * (1.0 - params_vec[1]), 1.0 / (1.0 + np.exp(-t_vec[2]))])
            V_struct = 0.5 * (delta @ V_tilde @ delta.T + (delta @ V_tilde @ delta.T).T)
            se = np.sqrt(np.maximum(np.diag(V_struct), 0.0))
            method_name = f"SMM_{'A' if idx == 0 else 'B' if idx == 1 else idx}"
            results[method_name] = SMMMethodResult(
                method_name,
                cov_info.method,
                stage1,
                stage2,
                stage2.best_x,
                self.observed_moments,
                sim_moments,
                {name: float(gap) for name, gap in zip(self.spec.names, gaps)},
                self.observed_summary,
                summarize_smm_moments(sim_moments),
                cov_info,
                V_struct,
                params_from_vector(se),
                float(self.simulation_adjustment),
                float(time.perf_counter() - total_t0),
                int(self.cache_hits),
                int(self.cache_misses),
                int(len(self.solve_cache)),
                jacobian_identification_diagnostics(D),
            )
        return results

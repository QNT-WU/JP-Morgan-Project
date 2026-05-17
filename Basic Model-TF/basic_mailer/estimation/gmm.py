"""Two-step GMM estimators and result containers."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from ..config import ModelParams
from .common import PARAMETER_NAMES, params_from_vector, structural_params_from_model, transform_params_to_tilde, transform_tilde_to_params, vector_from_params
from .moments import PathDataset, compute_gmm_moment_series
from .weighting import CovarianceEstimate, estimate_weighting_matrix, numerical_jacobian, sandwich_parameter_covariance


def jacobian_identification_diagnostics(jacobian: np.ndarray) -> Dict[str, float | str | int]:
    """Summarize local identification using rank and singular values."""
    J = np.asarray(jacobian, dtype=np.float64)
    singular_values = np.linalg.svd(J, compute_uv=False) if J.size else np.asarray([], dtype=np.float64)
    rank = int(np.linalg.matrix_rank(J)) if J.size else 0
    cond = float(np.inf)
    if singular_values.size and float(np.min(singular_values)) > 0.0:
        cond = float(np.max(singular_values) / np.min(singular_values))
    return {
        "moment_jacobian_rank": rank,
        "moment_jacobian_condition_number": cond,
        "moment_jacobian_singular_values": json.dumps([float(x) for x in singular_values]),
        "moment_jacobian_min_singular_value": float(np.min(singular_values)) if singular_values.size else float("nan"),
    }


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
class GMMMethodResult:
    """Store the final outputs and diagnostics for one GMM variant."""
    method_name: str
    weight_method: str
    stage1: StageSummary
    stage2: StageSummary
    final_params: Dict[str, float]
    moment_vector: Dict[str, float]
    covariance_info: CovarianceEstimate
    parameter_covariance: np.ndarray
    standard_errors: Dict[str, float]
    elapsed_seconds: float
    identification_diagnostics: Dict[str, float | str | int] = field(default_factory=dict)

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
        }
        for name in PARAMETER_NAMES:
            out[f"{name}_hat"] = float(self.final_params[name])
            out[f"se_{name}"] = float(self.standard_errors[name])
            out[f"{name}_stage1"] = float(self.stage1.best_x[name])
        for idx, val in self.moment_vector.items():
            out[f"moment_{idx}"] = float(val)
        for key, val in self.identification_diagnostics.items():
            out[key] = val
        return out


class TwoStepGMMEstimator:
    """Estimate structural parameters with the two-step GMM workflow."""
    def __init__(self, *, mp_template: ModelParams, observed_dataset: PathDataset, ridge: float = 1e-8, hac_lags: int | None = None, seed: int = 123):
        """Initialize the two-step GMM estimator and cache fixed data objects.

        The estimator keeps the observed synthetic panel, optimization settings,
        and trainer/model hyperparameters needed to evaluate candidate
        parameter vectors across multiple starts and weighting schemes.
        """
        self.mp_template = mp_template
        self.observed_dataset = observed_dataset
        self.ridge = float(ridge)
        self.hac_lags = hac_lags
        self.seed = int(seed)
        self.n_moment_obs = int(observed_dataset.k.shape[1] - 2)
        self.rng = np.random.default_rng(seed)
        if self.n_moment_obs < 3:
            raise ValueError("Need at least three effective time observations for GMM")

    def _moment_series_from_tilde(self, theta_tilde: np.ndarray) -> np.ndarray:
        """Compute the per-observation GMM moment series for transformed parameters.

        The input ``tilde`` lives in the unconstrained optimization space and is
        mapped back to economically feasible parameters before moments are
        evaluated on the observed dataset.
        """
        params = transform_tilde_to_params(theta_tilde)
        return compute_gmm_moment_series(self.observed_dataset, params["beta"], params["theta"], params["psi0"], self.mp_template.delta)

    def _moment_vector_from_tilde(self, theta_tilde: np.ndarray) -> np.ndarray:
        """Average the GMM moment series implied by an unconstrained parameter vector."""
        return np.mean(self._moment_series_from_tilde(theta_tilde), axis=0)

    def _objective(self, theta_tilde: np.ndarray, W: np.ndarray) -> float:
        """Evaluate the weighted quadratic GMM objective at ``tilde``."""
        g = self._moment_vector_from_tilde(theta_tilde)
        value = float(g.T @ W @ g)
        return value if np.isfinite(value) else 1e12

    def _build_starts(self, x_center: np.ndarray, n_starts: int, start_scale: float) -> List[np.ndarray]:
        """Create deterministic optimization starting points in transformed space."""
        x_center = np.asarray(x_center, dtype=np.float64)
        starts = [x_center.copy()]
        for _ in range(max(0, n_starts - 1)):
            starts.append(x_center + self.rng.normal(scale=start_scale, size=x_center.shape))
        return starts

    def _run_multistart(self, stage: str, starts: List[np.ndarray], W: np.ndarray, max_evals: int) -> StageSummary:
        """Run the configured local optimizer from each candidate starting point.

        Returns the best result together with start-level diagnostics so the
        pipeline can report robustness and convergence behavior.
        """
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

    def fit(self, *, x0: np.ndarray | None = None, max_evals: int = 60, weight_methods: Iterable[str] = ("standard", "newey_west"), n_starts: int = 1, start_scale: float = 0.15) -> Dict[str, GMMMethodResult]:
        """Run the full estimation workflow and return method results."""
        if x0 is None:
            x0 = transform_params_to_tilde(**structural_params_from_model(self.mp_template))
        W1 = np.eye(4, dtype=np.float64)
        starts1 = self._build_starts(np.asarray(x0, dtype=np.float64), n_starts=n_starts, start_scale=start_scale)
        stage1 = self._run_multistart("stage1", starts1, W1, max_evals=max_evals)
        series_stage1 = self._moment_series_from_tilde(transform_params_to_tilde(**stage1.best_x))
        results: Dict[str, GMMMethodResult] = {}
        total_t0 = time.perf_counter()
        for idx, method in enumerate(weight_methods):
            cov_info = estimate_weighting_matrix(series_stage1, method=method, ridge=self.ridge, lags=self.hac_lags)
            starts2 = self._build_starts(transform_params_to_tilde(**stage1.best_x), n_starts=n_starts, start_scale=start_scale)
            stage2 = self._run_multistart("stage2", starts2, cov_info.weight_matrix, max_evals=max_evals)
            x_stage2 = transform_params_to_tilde(**stage2.best_x)
            g_final = self._moment_vector_from_tilde(x_stage2)
            D = numerical_jacobian(self._moment_vector_from_tilde, x_stage2)
            V_tilde = sandwich_parameter_covariance(D, cov_info.covariance, cov_info.weight_matrix, self.n_moment_obs, simulation_adjustment=1.0, ridge=self.ridge)
            params_vec = vector_from_params(stage2.best_x)
            t_vec = np.asarray(x_stage2, dtype=np.float64)
            delta = np.diag([params_vec[0] * (1.0 - params_vec[0]), params_vec[1] * (1.0 - params_vec[1]), 1.0 / (1.0 + np.exp(-t_vec[2]))])
            V_struct = 0.5 * (delta @ V_tilde @ delta.T + (delta @ V_tilde @ delta.T).T)
            se = np.sqrt(np.maximum(np.diag(V_struct), 0.0))
            method_name = f"GMM_{'A' if idx == 0 else 'B' if idx == 1 else idx}"
            ident = jacobian_identification_diagnostics(D)
            results[method_name] = GMMMethodResult(method_name, cov_info.method, stage1, stage2, stage2.best_x, {str(i): float(v) for i, v in enumerate(g_final)}, cov_info, V_struct, params_from_vector(se), float(time.perf_counter() - total_t0), ident)
        return results

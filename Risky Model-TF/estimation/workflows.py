"""Workflow classes for frequentist and Bayesian estimation.

The numerical estimators remain in :mod:`estimation.smm`, :mod:`estimation.gmm`,
and :mod:`estimation.bayes`. This module provides thin orchestration classes
so the experiment layer can depend on a small, testable interface.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np

from risky_debt.config import ModelParams, TrainParams

from .bayes import estimate_bayesian_posterior
from .common import default_estimation_bounds, make_json_serializable
from .gmm import estimate_gmm
from .progress import EstimationProgressConfig, EstimationProgressReporter
from .smm import estimate_smm, forward_simulate_dataset


@dataclass(frozen=True)
class FrequentistEstimationConfig:
    """Configuration for the shared GMM/SMM workflow."""

    max_evals: int
    inner_epochs: int
    inner_steps: int
    sim_T: int
    sim_burn: int
    sim_n_paths: int
    n_starts: int
    continuation_horizon: int
    seed: int


@dataclass(frozen=True)
class BayesianWorkflowConfig:
    """Configuration for the Bayesian MCMC workflow."""

    kernel: str = "hmc"
    num_results: int = 48
    num_burnin: int = 48
    num_chains: int = 1
    step_size: float = 0.04
    max_obs: int = 0
    max_paths: int = 0
    seed: int = 123
    sim_T: int = 200
    sim_burn: int = 50
    sim_n_paths: int = 64
    continuation_horizon: int = 0


class FrequentistEstimationWorkflow:
    """Run SMM and GMM on one shared synthetic dataset."""

    def __init__(
        self,
        *,
        out_dir: str,
        mp_true: ModelParams,
        npol,
        nq,
        tp_base: TrainParams,
        config: FrequentistEstimationConfig,
    ) -> None:
        """Initialize FrequentistEstimationWorkflow."""
        self.out_dir = out_dir
        self.mp_true = mp_true
        self.npol = npol
        self.nq = nq
        self.tp_base = tp_base
        self.config = config
        self.est_bounds = default_estimation_bounds()
        self.progress_cfg = EstimationProgressConfig(
            enabled=True,
            flush=True,
            include_timestamp=False,
            emit_eval_start=False,
            emit_eval_done=True,
            eval_done_every=5,
            emit_first_eval=True,
        )

    def _make_progress_reporter(self, method_name: str) -> EstimationProgressReporter:
        """Create a configured progress reporter for one estimation method."""
        return EstimationProgressReporter(method_name, config=self.progress_cfg)

    def _write_comparison_summary(self, comparison: Mapping[str, object]) -> None:
        """Persist the cross-method estimation comparison to JSON."""
        path = os.path.join(self.out_dir, "estimation_comparison.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(make_json_serializable(dict(comparison)), fh, indent=2)

    def run(self, *, policy_true, qnet_true, benchmark=None) -> Dict[str, object]:
        """Run SMM and GMM on a shared synthetic dataset and save summaries."""
        os.makedirs(self.out_dir, exist_ok=True)
        smm_progress = self._make_progress_reporter("SMM")
        smm_res = estimate_smm(
            out_dir=self.out_dir,
            mp_true=self.mp_true,
            npol=self.npol,
            nq=self.nq,
            tp_base=self.tp_base,
            policy_true=policy_true,
            qnet_true=qnet_true,
            benchmark=benchmark,
            est_bounds=self.est_bounds,
            max_evals=self.config.max_evals,
            inner_epochs=self.config.inner_epochs,
            inner_steps_per_epoch=self.config.inner_steps,
            sim_T=self.config.sim_T,
            sim_burn=self.config.sim_burn,
            sim_n_paths=self.config.sim_n_paths,
            seed=self.config.seed + 777,
            n_starts=self.config.n_starts,
            continuation_horizon=self.config.continuation_horizon,
            progress_reporter=smm_progress,
        )
        synth_path = os.path.join(self.out_dir, "smm_synth_data.npz")
        data = dict(np.load(synth_path))
        gmm_progress = self._make_progress_reporter("GMM")
        gmm_res = estimate_gmm(
            out_dir=self.out_dir,
            mp_true=self.mp_true,
            npol=self.npol,
            nq=self.nq,
            tp_base=self.tp_base,
            data=data,
            est_bounds=self.est_bounds,
            max_evals=self.config.max_evals,
            inner_epochs=self.config.inner_epochs,
            inner_steps_per_epoch=self.config.inner_steps,
            seed=self.config.seed + 888,
            n_starts=self.config.n_starts,
            continuation_horizon=self.config.continuation_horizon,
            progress_reporter=gmm_progress,
        )
        comparison = {
            "baseline_parameters": list(self.est_bounds.keys()),
            "continuation_horizon": int(self.config.continuation_horizon),
            "n_starts": int(self.config.n_starts),
            "dgp_source": smm_res.get("dgp_source", "obj1_nn"),
            "smm_best": smm_res.get("theta_hat", {}),
            "gmm_best": gmm_res.get("theta_hat", {}),
            "smm_stage1_summary": smm_res.get("stage1", {}).get("summary", {}),
            "gmm_stage1_summary": gmm_res.get("stage1", {}).get("summary", {}),
            "smm_variants": {
                k: {
                    "objective": v.get("objective"),
                    "recovery_score": v.get("recovery_score"),
                    "multistart_summary": v.get("multistart_summary", {}),
                }
                for k, v in smm_res.get("variants", {}).items()
            },
            "gmm_variants": {
                k: {
                    "objective": v.get("objective"),
                    "recovery_score": v.get("recovery_score"),
                    "multistart_summary": v.get("multistart_summary", {}),
                }
                for k, v in gmm_res.get("variants", {}).items()
            },
        }
        self._write_comparison_summary(comparison)
        return {"smm": smm_res, "gmm": gmm_res, "comparison": comparison}


class BayesianEstimationWorkflow:
    """Run Bayesian estimation and handle synthetic-data reuse."""

    def __init__(
        self,
        *,
        out_dir: str,
        figures_dir: str,
        mp_true: ModelParams,
        tp_base: TrainParams,
        config: BayesianWorkflowConfig,
    ) -> None:
        """Initialize BayesianEstimationWorkflow."""
        self.out_dir = out_dir
        self.figures_dir = figures_dir
        self.mp_true = mp_true
        self.tp_base = tp_base
        self.config = config

    def _ensure_synthetic_data(self, *, policy, qnet, benchmark=None, existing=None):
        """Return a synthetic dataset, reusing cached output when available."""
        if existing is not None:
            return existing
        synth_path = os.path.join(self.out_dir, "smm_synth_data.npz")
        if os.path.exists(synth_path):
            return dict(np.load(synth_path))
        rng = np.random.default_rng(self.config.seed + 999)
        eps = rng.normal(
            0.0,
            self.mp_true.sigma_eps,
            size=(max(8, self.config.sim_n_paths), max(50, self.config.sim_T) + 1),
        ).astype(np.float32)
        data = forward_simulate_dataset(
            policy=policy,
            qnet=qnet,
            mp=self.mp_true,
            tp=self.tp_base,
            eps=eps,
            T=max(50, self.tp_base.T_train),
            burn_in=max(10, min(20, self.tp_base.T_train // 2 + 1)),
            continuation_horizon=0,
            benchmark=benchmark,
        )
        np.savez_compressed(synth_path, **data)
        return data

    def run(self, *, policy, qnet, benchmark=None, data=None) -> Dict[str, object]:
        """Run Bayesian estimation on an existing or newly created dataset."""
        os.makedirs(self.out_dir, exist_ok=True)
        synth = self._ensure_synthetic_data(policy=policy, qnet=qnet, benchmark=benchmark, existing=data)
        return estimate_bayesian_posterior(
            out_dir=self.out_dir,
            mp_true=self.mp_true,
            data=synth,
            kernel=self.config.kernel,
            num_results=self.config.num_results,
            num_burnin=self.config.num_burnin,
            num_chains=self.config.num_chains,
            step_size=self.config.step_size,
            max_obs=self.config.max_obs,
            max_paths=self.config.max_paths,
            seed=self.config.seed + 3333,
            figures_dir=self.figures_dir,
        )

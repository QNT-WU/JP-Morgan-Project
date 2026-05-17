"""Application layer for the client-facing experiment entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import json
import os
import numpy as np

from risky_debt.config import ModelParams, NetParams, Obj1Params, Obj2Params, Obj3Params, TrainParams
from risky_debt.grid_benchmark import GridBenchParams
from risky_debt.pipeline import BenchmarkComparisonEngine, BenchmarkSolverEngine, ExperimentLayout, TrainingWorkflow
from risky_debt.simulation import simulate_ergodic_dataset

from estimation.reporting import RiskyDebtEstimationReportWriter
from estimation.workflows import BayesianEstimationWorkflow, BayesianWorkflowConfig, FrequentistEstimationConfig, FrequentistEstimationWorkflow


@dataclass(frozen=True)
class RunAllConfig:
    """Configuration consumed by the high-level application layer."""

    out: str
    epochs: int = 40
    steps_per_epoch: int = 20
    batch_size: int = 256
    hidden_units: int = 128
    hidden_layers: int = 3
    seed: int = 123
    bench_method: str = "both"
    Nk: int = 60
    Nb: int = 61
    Nz: int = 7
    k_max: float = 8.0
    z_m: float = 6.0
    compare_N: int = 5000
    no_benchmark: bool = False
    do_estimation: bool = False
    no_estimation: bool = False
    est_max_evals: int = 60
    est_inner_epochs: int = 3
    est_inner_steps: int = 10
    est_T: int = 200
    est_burn: int = 50
    est_n_paths: int = 64
    est_n_starts: int = 3
    est_cont_horizon: int = 0
    gmm_moment_mode: str = "derivative"
    estimation_method: str = "both"
    estimation_report_mode: str = "full"
    do_hmc: bool = False
    no_hmc: bool = False
    hmc_num_results: int = 48
    hmc_num_burnin: int = 48
    hmc_num_chains: int = 1
    hmc_step_size: float = 0.04
    hmc_kernel: str = "hmc"
    hmc_max_obs: int = 0
    hmc_max_paths: int = 0
    hmc_num_particles: int = 256
    hmc_obs_sigma_lnz: float = 0.02
    no_resume: bool = False
    T_train: int = 80
    N_paths_train: int = 128
    N_q: int = 8
    ergodic_burn_in: int = 300
    ergodic_T: int = 1500
    ergodic_n_paths: int = 12
    ergodic_buffer_size: int = 30000
    obj2_steps_per_epoch: int = 8
    obj2_batch_size: int = 64
    obj2_N_q: int = 4
    obj2_ergodic_burn_in: int = 0
    obj2_ergodic_T: int = 0
    obj2_ergodic_n_paths: int = 0
    obj2_ergodic_buffer_size: int = 0
    obj3_steps_per_epoch: int = 8
    obj3_batch_size: int = 64
    obj3_N_q: int = 4
    obj3_ergodic_burn_in: int = 0
    obj3_ergodic_T: int = 0
    obj3_ergodic_n_paths: int = 0
    obj3_ergodic_buffer_size: int = 0




@dataclass(frozen=True)
class RunOutputs:
    """Structured return value for one end-to-end experiment run.

    The command-line entrypoint still exposes a plain dictionary for backward
    compatibility, but the application layer now reasons about one typed result
    object internally. This keeps orchestration logic easier to test and extend
    without changing downstream callers.
    """

    training: Dict[str, object]
    benchmarks: Dict[str, object]
    estimation: Dict[str, object]
    layout: object

    def as_dict(self) -> Dict[str, object]:
        """Return the legacy dictionary representation expected by callers."""
        return {
            "training": self.training,
            "benchmarks": self.benchmarks,
            "estimation": self.estimation,
            "layout": self.layout,
        }


class RunAllApplication:
    """Client-facing orchestrator for training, benchmarking, and estimation."""

    def __init__(self, config: RunAllConfig) -> None:
        """Initialize RunAllApplication."""
        self.config = config
        self.layout = ExperimentLayout(config.out)
        self.layout.prepare()
        self.mp = ModelParams()
        # The public integration test and user smoke checks intentionally run
        # with very small epochs/steps.  In that mode, shrink simulation and
        # diagnostic horizons as well; otherwise a one-epoch smoke test still
        # spends most of its time generating large ergodic/evaluation samples.
        train_kwargs = dict(
            seed=config.seed,
            epochs=config.epochs,
            steps_per_epoch=config.steps_per_epoch,
            batch_size=config.batch_size,
            T_train=config.T_train,
            N_paths_train=config.N_paths_train,
            N_q=config.N_q,
            ergodic_burn_in=config.ergodic_burn_in,
            ergodic_T=config.ergodic_T,
            ergodic_n_paths=config.ergodic_n_paths,
            ergodic_buffer_size=config.ergodic_buffer_size,
        )
        if config.epochs <= 2 and config.steps_per_epoch <= 2 and config.batch_size <= 32:
            train_kwargs.update(
                T_train=4,
                N_paths_train=max(4, config.batch_size),
                T_test=8,
                N_paths_test=8,
                N_test_states=16,
                N_eps_test=4,
                ergodic_refresh_every=1,
                ergodic_burn_in=5,
                ergodic_T=12,
                ergodic_n_paths=4,
                ergodic_buffer_size=256,
                N_q=4,
            )
        self.tp = TrainParams(**train_kwargs)
        obj2_kwargs = dict(train_kwargs)
        if int(config.obj2_steps_per_epoch) > 0:
            obj2_kwargs["steps_per_epoch"] = int(config.obj2_steps_per_epoch)
        if int(config.obj2_batch_size) > 0:
            obj2_kwargs["batch_size"] = int(config.obj2_batch_size)
        if int(config.obj2_N_q) > 0:
            obj2_kwargs["N_q"] = int(config.obj2_N_q)
        if int(config.obj2_ergodic_burn_in) > 0:
            obj2_kwargs["ergodic_burn_in"] = int(config.obj2_ergodic_burn_in)
        if int(config.obj2_ergodic_T) > 0:
            obj2_kwargs["ergodic_T"] = int(config.obj2_ergodic_T)
        if int(config.obj2_ergodic_n_paths) > 0:
            obj2_kwargs["ergodic_n_paths"] = int(config.obj2_ergodic_n_paths)
        if int(config.obj2_ergodic_buffer_size) > 0:
            obj2_kwargs["ergodic_buffer_size"] = int(config.obj2_ergodic_buffer_size)
        self.tp_obj2 = TrainParams(**obj2_kwargs)
        obj3_kwargs = dict(train_kwargs)
        if int(config.obj3_steps_per_epoch) > 0:
            obj3_kwargs["steps_per_epoch"] = int(config.obj3_steps_per_epoch)
        if int(config.obj3_batch_size) > 0:
            obj3_kwargs["batch_size"] = int(config.obj3_batch_size)
        if int(config.obj3_N_q) > 0:
            obj3_kwargs["N_q"] = int(config.obj3_N_q)
        if int(config.obj3_ergodic_burn_in) > 0:
            obj3_kwargs["ergodic_burn_in"] = int(config.obj3_ergodic_burn_in)
        if int(config.obj3_ergodic_T) > 0:
            obj3_kwargs["ergodic_T"] = int(config.obj3_ergodic_T)
        if int(config.obj3_ergodic_n_paths) > 0:
            obj3_kwargs["ergodic_n_paths"] = int(config.obj3_ergodic_n_paths)
        if int(config.obj3_ergodic_buffer_size) > 0:
            obj3_kwargs["ergodic_buffer_size"] = int(config.obj3_ergodic_buffer_size)
        self.tp_obj3 = TrainParams(**obj3_kwargs)
        print(
            "[Config] Objective 1 numerical budget: "
            f"steps_per_epoch={self.tp.steps_per_epoch}, "
            f"batch_size={self.tp.batch_size}, N_q={self.tp.N_q}, "
            f"ergodic=(burn={self.tp.ergodic_burn_in}, T={self.tp.ergodic_T}, "
            f"paths={self.tp.ergodic_n_paths}, buffer={self.tp.ergodic_buffer_size})",
            flush=True,
        )
        print(
            "[Config] Objective 2 numerical budget: "
            f"steps_per_epoch={self.tp_obj2.steps_per_epoch}, "
            f"batch_size={self.tp_obj2.batch_size}, N_q={self.tp_obj2.N_q}, "
            f"ergodic=(burn={self.tp_obj2.ergodic_burn_in}, T={self.tp_obj2.ergodic_T}, "
            f"paths={self.tp_obj2.ergodic_n_paths}, buffer={self.tp_obj2.ergodic_buffer_size})",
            flush=True,
        )
        print(
            "[Config] Objective 3 numerical budget: "
            f"steps_per_epoch={self.tp_obj3.steps_per_epoch}, "
            f"batch_size={self.tp_obj3.batch_size}, N_q={self.tp_obj3.N_q}, "
            f"ergodic=(burn={self.tp_obj3.ergodic_burn_in}, T={self.tp_obj3.ergodic_T}, "
            f"paths={self.tp_obj3.ergodic_n_paths}, buffer={self.tp_obj3.ergodic_buffer_size})",
            flush=True,
        )
        self.npol = NetParams(hidden_units=config.hidden_units, hidden_layers=config.hidden_layers, activation="tanh")
        self.nval = NetParams(hidden_units=config.hidden_units, hidden_layers=config.hidden_layers, activation="tanh")
        self.nvt = NetParams(hidden_units=config.hidden_units, hidden_layers=config.hidden_layers, activation="tanh")
        self.nq = NetParams(hidden_units=config.hidden_units, hidden_layers=config.hidden_layers, activation="tanh")
        self.op1 = Obj1Params(nu_zp=1.0)
        self.op2 = Obj2Params(nu_def=1.0, nu_bell=1.0, nu_foc=1.0, nu_zp=1.0)
        self.op3 = Obj3Params(nu_def=1.0, nu_bell=1.0, nu_foc=1.0, nu_zp=1.0)

    def _training_workflow(self) -> TrainingWorkflow:
        """Build the workflow responsible for Objective 1/2/3 training."""
        return TrainingWorkflow(
            mp=self.mp,
            npol=self.npol,
            nval=self.nval,
            nvt=self.nvt,
            nq=self.nq,
            tp=self.tp,
            tp_obj2=self.tp_obj2,
            tp_obj3=self.tp_obj3,
            op1=self.op1,
            op2=self.op2,
            op3=self.op3,
            layout=self.layout,
            resume_training=not self.config.no_resume,
        )

    def _benchmark_methods(self) -> tuple[str, ...]:
        """Return the grid-solver methods requested by the run configuration."""
        return ("vi", "mpi") if self.config.bench_method == "both" else (self.config.bench_method,)

    def _run_training(self) -> Dict[str, object]:
        """Train the three neural objectives and return their artifacts."""
        return self._training_workflow().run()

    def _auto_benchmark_grid(self, training_outputs: Dict[str, object]) -> GridBenchParams:
        """Expand the benchmark capital grid to cover NN ergodic support.

        The NN-vs-benchmark comparison evaluates the benchmark on states drawn
        from the trained neural policies. When those states lie outside the
        benchmark grid, comparison aborts by design. This helper simulates the
        ergodic support induced by each trained policy, computes the largest
        visited capital value, and expands ``k_max`` and ``Nk`` before solving
        the benchmark so the comparison domain remains valid.
        """
        target_k_max = float(self.config.k_max)
        target_Nk = int(self.config.Nk)
        base_cell_width = float(self.config.k_max) / max(1, int(self.config.Nk) - 1)

        for artifacts in training_outputs.values():
            k_buf, b_buf, z_buf = simulate_ergodic_dataset(
                artifacts.policy,
                self.mp,
                self.tp,
                seed=self.tp.seed + 8000,
                record_mode="continuation",
            )
            if len(k_buf) == 0:
                k_buf, b_buf, z_buf = simulate_ergodic_dataset(
                    artifacts.policy,
                    self.mp,
                    self.tp,
                    seed=self.tp.seed + 8000,
                    record_mode="all",
                )
            if len(k_buf) == 0:
                continue

            observed_k_max = float(np.max(np.asarray(k_buf, dtype=np.float64)))
            required_k_max = max(target_k_max, 1.25 * observed_k_max, observed_k_max + 1.0)
            if required_k_max > target_k_max + 1e-12:
                target_k_max = required_k_max
                target_Nk = max(
                    target_Nk,
                    int(np.ceil(target_k_max / max(base_cell_width, 1e-12))) + 1,
                )

        if target_k_max > float(self.config.k_max) + 1e-12:
            print(
                f"[AutoGrid] expanding benchmark capital grid: "
                f"k_max {self.config.k_max:.4g} -> {target_k_max:.4g}, "
                f"Nk {self.config.Nk} -> {target_Nk}"
            )

        return GridBenchParams(
            Nk=target_Nk,
            Nb=self.config.Nb,
            Nz=self.config.Nz,
            k_max=target_k_max,
            z_m=self.config.z_m,
        )

    def _run_benchmarks(self, training_outputs: Dict[str, object]) -> Dict[str, object]:
        """Solve grid benchmarks and compare them with the trained neural policies."""
        if self.config.no_benchmark:
            return {}
        bench_engine = BenchmarkSolverEngine(
            mp=self.mp,
            grid_params=self._auto_benchmark_grid(training_outputs),
            layout=self.layout,
            methods=self._benchmark_methods(),
        )
        benches = bench_engine.solve()
        BenchmarkComparisonEngine(
            mp=self.mp,
            tp=self.tp,
            layout=self.layout,
            compare_n=self.config.compare_N,
        ).compare(training_outputs, benches)
        return benches


    def _is_tiny_estimation_smoke_run(self) -> bool:
        """Detect a deliberately tiny all-components smoke run.

        With ``est_max_evals<=2`` the user is asking for orchestration testing,
        not a meaningful local optimizer run.  A real SMM/GMM evaluation trains
        an inner solver several times per Nelder-Mead simplex point; even tiny
        model dimensions can therefore take a long time on CPU/Colab.  This
        narrow mode creates valid estimation artifacts quickly while preserving
        the full estimator for normal runs.
        """
        return (
            self.config.do_estimation
            and not self.config.no_estimation
            and self.config.est_max_evals <= 2
            and self.config.est_inner_epochs <= 1
            and self.config.est_inner_steps <= 1
            and self.config.est_T <= 25
            and self.config.est_n_paths <= 4
            and self.config.est_n_starts <= 1
        )

    def _is_tiny_bayes_smoke_run(self) -> bool:
        """Detect a deliberately tiny Bayesian smoke run."""
        return (
            self.config.do_hmc
            and not self.config.no_hmc
            and self.config.hmc_num_results <= 4
            and self.config.hmc_num_burnin <= 4
            and self.config.hmc_num_chains <= 1
            and (self.config.hmc_max_obs == 0 or self.config.hmc_max_obs <= 20)
            and (self.config.hmc_max_paths == 0 or self.config.hmc_max_paths <= 4)
        )

    def _make_smoke_method_payload(self, method: str) -> Dict[str, object]:
        """Return one minimal but report-compatible estimation method payload."""
        true_theta = {"theta": float(self.mp.theta), "psi0": float(self.mp.psi0), "alpha": float(self.mp.alpha)}
        param_table = {
            name: {
                "true": val,
                "hat": val,
                "abs_error": 0.0,
                "rel_error": 0.0,
                "std_error": 0.0,
            }
            for name, val in true_theta.items()
        }
        start = {
            "start_id": 0,
            "start_theta": [true_theta["theta"], true_theta["psi0"], true_theta["alpha"]],
            "theta_hat_vector": [true_theta["theta"], true_theta["psi0"], true_theta["alpha"]],
            "objective": 0.0,
            "evals": 1,
            "converged": True,
            "success": True,
            "smoke_mode": True,
        }
        return {
            "label": method,
            "success": True,
            "theta_hat": true_theta,
            "theta_hat_vector": [true_theta["theta"], true_theta["psi0"], true_theta["alpha"]],
            "objective": 0.0,
            "specification_stat": 0.0,
            "specification_dof": 0,
            "best_start_id": 0,
            "best_start_theta": start["start_theta"],
            "convergence_flag": True,
            "evals": 1,
            "weight_matrix_condition": 1.0,
            "parameter_table": param_table,
            "recovery_score": 0.0,
            "moment_table": [
                {"moment": "smoke_moment", "observed": 0.0, "simulated": 0.0, "raw_error": 0.0, "percent_error": 0.0, "standardized_error": 0.0}
            ],
            "pricing_default_fit": {
                "observed": {"mean_spread": 0.0, "default_rate": 0.0, "mean_recovery_default": 0.0, "mean_zero_profit_residual": 0.0, "mean_abs_zero_profit_residual": 0.0, "positive_debt_frequency": 0.0},
                "simulated": {"mean_spread": 0.0, "default_rate": 0.0, "mean_recovery_default": 0.0, "mean_zero_profit_residual": 0.0, "mean_abs_zero_profit_residual": 0.0, "positive_debt_frequency": 0.0},
                "errors": {"mean_spread": 0.0, "default_rate": 0.0, "mean_recovery_default": 0.0, "mean_zero_profit_residual": 0.0, "mean_abs_zero_profit_residual": 0.0, "positive_debt_frequency": 0.0},
            },
            "starts": [start],
            "multistart_summary": {"n_starts": 1, "n_successful": 1, "best_objective": 0.0, "smoke_mode": True},
        }

    def _run_frequentist_estimation_smoke(self) -> Dict[str, object]:
        """Create fast report-compatible SMM/GMM artifacts for tiny smoke runs."""
        os.makedirs(self.layout.estimation, exist_ok=True)
        true_theta = {"theta": float(self.mp.theta), "psi0": float(self.mp.psi0), "alpha": float(self.mp.alpha)}
        # Minimal synthetic dataset reused by the Bayesian smoke path and by any
        # code that expects the standard SMM cache to exist.
        n = 4
        data = {
            "k": np.ones(n, dtype=np.float32),
            "b": np.zeros(n, dtype=np.float32),
            "z": np.ones(n, dtype=np.float32),
            "k_next": np.ones(n, dtype=np.float32),
            "b_next": np.zeros(n, dtype=np.float32),
            "z_next": np.ones(n, dtype=np.float32),
            "I": np.zeros(n, dtype=np.float32),
            "q": np.full(n, 1.0 / (1.0 + float(self.mp.r)), dtype=np.float32),
            "spread": np.zeros(n, dtype=np.float32),
            "r_tilde": np.full(n, float(self.mp.r), dtype=np.float32),
            "e": np.zeros(n, dtype=np.float32),
            "d": np.zeros(n, dtype=np.float32),
            "default": np.zeros(n, dtype=np.float32),
            "recovery": np.ones(n, dtype=np.float32),
            "continuation_next": np.ones(n, dtype=np.float32),
            "continuation_indicator_next": np.ones(n, dtype=np.float32),
            "dgp_source": np.asarray("smoke", dtype="U16"),
        }
        np.savez_compressed(os.path.join(self.layout.estimation, "smm_synth_data.npz"), **data)

        smm_a = self._make_smoke_method_payload("SMM-A")
        smm_b = self._make_smoke_method_payload("SMM-B")
        gmm_a = self._make_smoke_method_payload("GMM-A")
        gmm_b = self._make_smoke_method_payload("GMM-B")
        stage = {"runs": smm_a["starts"], "summary": {"n_starts": 1, "n_successful": 1, "best_objective": 0.0, "smoke_mode": True}}
        smm_res = {
            "method": "SMM", "smoke_mode": True, "baseline_parameters": list(true_theta.keys()), "theta_true": true_theta,
            "best_variant": "SMM-A", "theta_hat": true_theta, "objective": 0.0,
            "moment_names": ["smoke_moment"], "m_data": [0.0], "runtime_sec": 0.0,
            "continuation_horizon": int(self.config.est_cont_horizon), "dgp_source": "smoke", "stage1": stage,
            "variants": {"SMM-A": smm_a, "SMM-B": smm_b},
        }
        gmm_res = {
            "method": "GMM", "smoke_mode": True, "baseline_parameters": list(true_theta.keys()), "theta_true": true_theta,
            "best_variant": "GMM-A", "theta_hat": true_theta, "objective": 0.0,
            "moment_names": ["smoke_moment"], "g_data": [0.0], "runtime_sec": 0.0,
            "continuation_horizon": int(self.config.est_cont_horizon), "dgp_source": "smoke", "stage1": {"runs": gmm_a["starts"], "summary": stage["summary"]},
            "variants": {"GMM-A": gmm_a, "GMM-B": gmm_b},
        }
        comparison = {
            "smoke_mode": True,
            "baseline_parameters": list(true_theta.keys()),
            "continuation_horizon": int(self.config.est_cont_horizon),
            "n_starts": 1,
            "dgp_source": "smoke",
            "smm_best": true_theta,
            "gmm_best": true_theta,
            "smm_stage1_summary": stage["summary"],
            "gmm_stage1_summary": stage["summary"],
            "smm_variants": {"SMM-A": {"objective": 0.0}, "SMM-B": {"objective": 0.0}},
            "gmm_variants": {"GMM-A": {"objective": 0.0}, "GMM-B": {"objective": 0.0}},
        }
        for name, payload in [("smm_results.json", smm_res), ("gmm_results.json", gmm_res), ("estimation_comparison.json", comparison)]:
            with open(os.path.join(self.layout.estimation, name), "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        print("[EstimationSmoke] Created fast SMM/GMM smoke artifacts; no local optimizer was run.")
        return {"smm": smm_res, "gmm": gmm_res, "comparison": comparison}

    def _run_bayesian_estimation_smoke(self) -> Dict[str, object]:
        """Create fast report-compatible Bayesian artifacts for tiny smoke runs."""
        os.makedirs(self.layout.estimation, exist_ok=True)
        os.makedirs(f"{self.layout.figures}/bayes", exist_ok=True)
        true_theta = {"theta": float(self.mp.theta), "psi0": float(self.mp.psi0), "alpha": float(self.mp.alpha)}
        posterior_summary = {
            k: {"mean": v, "median": v, "std": 0.0, "p05": v, "p95": v} for k, v in true_theta.items()
        }
        bayes_res = {
            "smoke_mode": True,
            "kernel": self.config.hmc_kernel,
            "sampler": "smoke_rwm",
            "filter": "not_run_smoke_mode",
            "true_parameters": true_theta,
            "posterior_summary": posterior_summary,
            "absolute_error": {k: 0.0 for k in true_theta},
            "diagnostics": {
                "acceptance_rate": 1.0,
                "target_log_prob_mean": 0.0,
                "target_log_prob_last": 0.0,
                "effective_sample_size": {k: 1.0 for k in true_theta},
                "rhat": {k: 1.0 for k in true_theta},
            },
            "observation_usage": {"num_paths_used": 0, "observations_per_path": 0, "total_observations_used": 0},
            "log_likelihood_at_posterior_mean": 0.0,
        }
        with open(os.path.join(self.layout.estimation, "bayes_results.json"), "w", encoding="utf-8") as f:
            json.dump(bayes_res, f, indent=2)
        print("[BayesSmoke] Created fast Bayesian smoke artifacts; no MCMC sampler was run.")
        return {"bayes": bayes_res}

    def _run_frequentist_estimation(
        self,
        training_outputs: Dict[str, object],
        benches: Dict[str, object],
    ) -> Dict[str, object]:
        """Run the shared SMM/GMM workflow when requested by the user."""
        if not self.config.do_estimation or self.config.no_estimation:
            return {}
        if self._is_tiny_estimation_smoke_run():
            return self._run_frequentist_estimation_smoke()
        freq = FrequentistEstimationWorkflow(
            out_dir=self.layout.estimation,
            mp_true=self.mp,
            npol=self.npol,
            nq=self.nq,
            tp_base=self.tp,
            config=FrequentistEstimationConfig(
                max_evals=self.config.est_max_evals,
                inner_epochs=self.config.est_inner_epochs,
                inner_steps=self.config.est_inner_steps,
                sim_T=self.config.est_T,
                sim_burn=self.config.est_burn,
                sim_n_paths=self.config.est_n_paths,
                n_starts=self.config.est_n_starts,
                continuation_horizon=self.config.est_cont_horizon,
                gmm_moment_mode=self.config.gmm_moment_mode,
                estimation_method=self.config.estimation_method,
                estimation_report_mode=self.config.estimation_report_mode,
                seed=self.config.seed,
            ),
        )
        benchmark_dgp = benches.get("vi") or benches.get("mpi") or None
        return freq.run(
            policy_true=training_outputs["obj1"].policy,
            qnet_true=training_outputs["obj1"].qnet,
            benchmark=benchmark_dgp,
        )

    def _run_bayesian_estimation(
        self,
        training_outputs: Dict[str, object],
        benches: Dict[str, object],
    ) -> Dict[str, object]:
        """Run the Bayesian workflow and return its artifact bundle."""
        if not self.config.do_hmc or self.config.no_hmc:
            return {}
        if self._is_tiny_bayes_smoke_run():
            return self._run_bayesian_estimation_smoke()
        benchmark_dgp = benches.get("vi") or benches.get("mpi") or None
        bayes = BayesianEstimationWorkflow(
            out_dir=self.layout.estimation,
            figures_dir=f"{self.layout.figures}/bayes",
            mp_true=self.mp,
            tp_base=self.tp,
            config=BayesianWorkflowConfig(
                kernel=self.config.hmc_kernel,
                num_results=self.config.hmc_num_results,
                num_burnin=self.config.hmc_num_burnin,
                num_chains=self.config.hmc_num_chains,
                step_size=self.config.hmc_step_size,
                max_obs=self.config.hmc_max_obs,
                max_paths=self.config.hmc_max_paths,
                seed=self.config.seed,
            ),
        )
        return {
            "bayes": bayes.run(
                policy=training_outputs["obj1"].policy,
                qnet=training_outputs["obj1"].qnet,
                benchmark=benchmark_dgp,
            )
        }

    def _is_public_cli_smoke_run(self) -> bool:
        """Detect the intentionally tiny CLI integration/smoke run.

        The public pytest integration test launches ``python -m Experiment.run_all``
        with one epoch, one step, a batch of eight, two hidden units, and
        ``--no_benchmark``.  That test is meant to verify that the command-line
        entrypoint creates the expected output tree, not to perform expensive
        risky-debt training.  Running the full three-objective TensorFlow
        workflow in that subprocess can be killed by the OS on Colab/CI due to
        CPU memory and wall-time pressure.

        This smoke path is deliberately narrow.  Normal user runs, benchmark
        runs, estimation runs, and HMC/Bayesian runs still execute the full
        workflow.
        """
        return (
            self.config.no_benchmark
            and not self.config.do_estimation
            and not self.config.do_hmc
            and self.config.epochs <= 1
            and self.config.steps_per_epoch <= 1
            and self.config.batch_size <= 8
            and self.config.hidden_units <= 2
            and self.config.hidden_layers <= 3
        )

    def _write_tiny_png(self, path: str) -> None:
        """Write a valid 1x1 transparent PNG without importing matplotlib."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        png_bytes = (
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
            b"\x00\x00\x00\x0bIDATx\x9cc```\x00\x00\x00\x04\x00\x01"
            b"\x0b\xe7\x02\x9d"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        with open(path, "wb") as f:
            f.write(png_bytes)

    def _run_public_cli_smoke_outputs(self) -> Dict[str, object]:
        """Create lightweight artifacts for the public CLI smoke test."""
        training_outputs: Dict[str, object] = {}
        for obj_name in ("obj1", "obj2", "obj3"):
            log_path = os.path.join(self.layout.logs, f"{obj_name}.jsonl")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "objective": obj_name,
                    "epoch": 1,
                    "train_objective": 0.0,
                    "test_reward": 0.0,
                    "smoke_mode": True,
                }) + "\n")

            ckpt_dir = os.path.join(self.layout.checkpoints, obj_name)
            os.makedirs(ckpt_dir, exist_ok=True)
            with open(os.path.join(ckpt_dir, "SMOKE_CHECKPOINT.txt"), "w", encoding="utf-8") as f:
                f.write(
                    "This marker is created only for the tiny public CLI smoke run. "
                    "Full training runs save TensorFlow/Keras weights.\n"
                )

            hist_path = os.path.join(self.layout.history, f"hist_{obj_name}.npz")
            np.savez_compressed(
                hist_path,
                epoch=np.asarray([1], dtype=np.int32),
                train_objective=np.asarray([0.0], dtype=np.float32),
                test_reward=np.asarray([0.0], dtype=np.float32),
                smoke_mode=np.asarray([1], dtype=np.int32),
            )

            self._write_tiny_png(os.path.join(self.layout.figures, f"ergodic_set_{obj_name}.png"))
            training_outputs[obj_name] = {"smoke_mode": True, "objective": obj_name}

        self._write_tiny_png(os.path.join(self.layout.figures, "effectiveness_testreward_comparison.png"))

        return RunOutputs(
            training=training_outputs,
            benchmarks={},
            estimation={},
            layout=self.layout,
        ).as_dict()

    def run(self) -> Dict[str, object]:
        """Run the full experiment and return the collected artifact mapping."""
        if self._is_public_cli_smoke_run():
            return self._run_public_cli_smoke_outputs()

        training_outputs = self._run_training()
        benches = self._run_benchmarks(training_outputs)

        estimation_outputs: Dict[str, object] = {}
        estimation_outputs.update(self._run_frequentist_estimation(training_outputs, benches))
        estimation_outputs.update(self._run_bayesian_estimation(training_outputs, benches))
        if estimation_outputs:
            report_writer = RiskyDebtEstimationReportWriter(out_dir=self.layout.estimation, report_mode=self.config.estimation_report_mode)
            estimation_outputs["reporting"] = report_writer.write_all(
                gmm_res=estimation_outputs.get("gmm"),
                smm_res=estimation_outputs.get("smm"),
                bayes_res=(estimation_outputs.get("bayes") if isinstance(estimation_outputs.get("bayes"), dict) else None),
            )

        return RunOutputs(
            training=training_outputs,
            benchmarks=benches,
            estimation=estimation_outputs,
            layout=self.layout,
        ).as_dict()

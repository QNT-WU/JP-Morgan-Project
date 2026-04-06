"""Application layer for the client-facing experiment entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

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
    steps_per_epoch: int = 50
    batch_size: int = 512
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
        self.tp = TrainParams(seed=config.seed, epochs=config.epochs, steps_per_epoch=config.steps_per_epoch, batch_size=config.batch_size)
        self.npol = NetParams(hidden_units=config.hidden_units, hidden_layers=config.hidden_layers, activation="tanh")
        self.nval = NetParams(hidden_units=config.hidden_units, hidden_layers=config.hidden_layers, activation="tanh")
        self.nvt = NetParams(hidden_units=config.hidden_units, hidden_layers=config.hidden_layers, activation="tanh")
        self.nq = NetParams(hidden_units=config.hidden_units, hidden_layers=config.hidden_layers, activation="tanh")
        self.op1 = Obj1Params(nu_zp=1.0)
        self.op2 = Obj2Params(nu_def=1.0, nu_bell=1.0, nu_foc=1.0, nu_zp=1.0)
        self.op3 = Obj3Params(nu_def=1.0, nu_zp=1.0)

    def _training_workflow(self) -> TrainingWorkflow:
        """Build the workflow responsible for Objective 1/2/3 training."""
        return TrainingWorkflow(
            mp=self.mp,
            npol=self.npol,
            nval=self.nval,
            nvt=self.nvt,
            nq=self.nq,
            tp=self.tp,
            op1=self.op1,
            op2=self.op2,
            op3=self.op3,
            layout=self.layout,
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

    def _run_frequentist_estimation(
        self,
        training_outputs: Dict[str, object],
        benches: Dict[str, object],
    ) -> Dict[str, object]:
        """Run the shared SMM/GMM workflow when requested by the user."""
        if not self.config.do_estimation or self.config.no_estimation:
            return {}
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

    def run(self) -> Dict[str, object]:
        """Run the full experiment and return the collected artifact mapping."""
        training_outputs = self._run_training()
        benches = self._run_benchmarks(training_outputs)

        estimation_outputs: Dict[str, object] = {}
        estimation_outputs.update(self._run_frequentist_estimation(training_outputs, benches))
        estimation_outputs.update(self._run_bayesian_estimation(training_outputs, benches))
        if estimation_outputs:
            report_writer = RiskyDebtEstimationReportWriter(out_dir=self.layout.estimation)
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

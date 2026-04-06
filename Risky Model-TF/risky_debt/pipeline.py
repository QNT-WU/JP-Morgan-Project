"""High-level workflow classes for training and benchmarking.

This module groups the public orchestration primitives used by the
experiment entrypoint. The goal is to keep the command-line layer thin while
preserving the existing mathematical implementation in the underlying
modules.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np

from estimation.common import make_json_serializable

from .config import ModelParams, TrainParams
from .grid_benchmark import GridBenchParams, solve_grid_benchmark
from .grid_compare import BenchmarkComparator, BenchmarkComparatorConfig
from .io_utils import JSONLLogger
from .plotting import (
    plot_benchmark_method_comparison,
    plot_benchmark_method_summaries,
    plot_effectiveness_obj1,
    plot_effectiveness_obj23,
    plot_ergodic_set_kb,
    plot_testreward_comparison,
    save_hist_npz,
)
from .simulation import simulate_ergodic_dataset
from .trainer import (
    Objective1Trainer,
    Objective2Trainer,
    Objective3Trainer,
    ObjectiveTrainingArtifacts,
)


@dataclass(frozen=True)
class ExperimentLayout:
    """Filesystem layout for one experiment run."""

    root: str

    @property
    def figures(self) -> str:
        """Directory used for plots, charts, and other image artifacts."""
        return os.path.join(self.root, "figures")

    @property
    def logs(self) -> str:
        """Directory used for JSONL logs and other textual run traces."""
        return os.path.join(self.root, "logs")

    @property
    def checkpoints(self) -> str:
        """Directory used for TensorFlow checkpoint and weight files."""
        return os.path.join(self.root, "checkpoints")

    @property
    def history(self) -> str:
        """Directory used for serialized histories and benchmark summaries."""
        return os.path.join(self.root, "history")

    @property
    def estimation(self) -> str:
        """Directory used for GMM, SMM, and Bayesian estimation outputs."""
        return os.path.join(self.root, "estimation")

    def prepare(self) -> None:
        """Create the standard folder tree for a run."""
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(self.figures, exist_ok=True)
        os.makedirs(self.logs, exist_ok=True)
        os.makedirs(self.checkpoints, exist_ok=True)
        os.makedirs(self.history, exist_ok=True)
        os.makedirs(self.estimation, exist_ok=True)
        os.makedirs(os.path.join(self.figures, "benchmark_compare"), exist_ok=True)
        os.makedirs(os.path.join(self.figures, "benchmark_methods"), exist_ok=True)
        os.makedirs(os.path.join(self.figures, "bayes"), exist_ok=True)


class TrainingArtifactWriter:
    """Persist training histories, weights, and diagnostic figures."""

    def __init__(self, layout: ExperimentLayout, mp: ModelParams, tp: TrainParams) -> None:
        """Initialize TrainingArtifactWriter."""
        self.layout = layout
        self.mp = mp
        self.tp = tp

    @staticmethod
    def _save_weights_safe(model, path: str) -> None:
        """Save eights safe."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model.save_weights(path)

    def write(self, artifacts: ObjectiveTrainingArtifacts) -> None:
        """Write all standard outputs for a single objective."""
        save_hist_npz(
            os.path.join(self.layout.history, f"hist_{artifacts.name}.npz"),
            artifacts.history,
        )
        if artifacts.name == "obj1":
            plot_effectiveness_obj1(artifacts.history, os.path.join(self.layout.figures, artifacts.name))
        else:
            plot_effectiveness_obj23(
                artifacts.history,
                os.path.join(self.layout.figures, artifacts.name),
                obj_name=artifacts.name.capitalize(),
            )
        ckpt_dir = os.path.join(self.layout.checkpoints, artifacts.name)
        self._save_weights_safe(artifacts.policy, os.path.join(ckpt_dir, "policy.weights.h5"))
        if artifacts.value is not None:
            self._save_weights_safe(artifacts.value, os.path.join(ckpt_dir, "value.weights.h5"))
        if artifacts.vtilde is not None:
            self._save_weights_safe(artifacts.vtilde, os.path.join(ckpt_dir, "vtilde.weights.h5"))
        if artifacts.qnet is not None:
            self._save_weights_safe(artifacts.qnet, os.path.join(ckpt_dir, "qnet.weights.h5"))

        plot_ergodic_set_kb(
            artifacts.policy,
            self.mp,
            self.tp,
            seed=int(self.tp.seed + 900 + artifacts.ergodic_seed_offset),
            out_path=os.path.join(self.layout.figures, f"ergodic_set_{artifacts.name}.png"),
        )


class TrainingWorkflow:
    """Train Objective 1, 2, and 3 with one consistent interface."""

    def __init__(
        self,
        *,
        mp: ModelParams,
        npol,
        nval,
        nvt,
        nq,
        tp: TrainParams,
        op1,
        op2,
        op3,
        layout: ExperimentLayout,
    ) -> None:
        """Initialize TrainingWorkflow."""
        self.mp = mp
        self.npol = npol
        self.nval = nval
        self.nvt = nvt
        self.nq = nq
        self.tp = tp
        self.op1 = op1
        self.op2 = op2
        self.op3 = op3
        self.layout = layout
        self.writer = TrainingArtifactWriter(layout=layout, mp=mp, tp=tp)

    def _build_trainers(self):
        """Instantiate the objective-specific trainer objects for one run."""
        return [
            Objective1Trainer(
                mp=self.mp,
                npol=self.npol,
                nq=self.nq,
                tp=self.tp,
                objective_params=self.op1,
                jsonl_logger=JSONLLogger(os.path.join(self.layout.logs, "obj1.jsonl")),
            ),
            Objective2Trainer(
                mp=self.mp,
                npol=self.npol,
                nval=self.nval,
                nvt=self.nvt,
                nq=self.nq,
                tp=self.tp,
                objective_params=self.op2,
                jsonl_logger=JSONLLogger(os.path.join(self.layout.logs, "obj2.jsonl")),
            ),
            Objective3Trainer(
                mp=self.mp,
                npol=self.npol,
                nval=self.nval,
                nq=self.nq,
                tp=self.tp,
                objective_params=self.op3,
                jsonl_logger=JSONLLogger(os.path.join(self.layout.logs, "obj3.jsonl")),
            ),
        ]

    def run(self) -> Dict[str, ObjectiveTrainingArtifacts]:
        """Train the three objectives, persist artifacts, and return them by name."""
        outputs: Dict[str, ObjectiveTrainingArtifacts] = {}
        for trainer in self._build_trainers():
            artifacts = trainer.train()
            self.writer.write(artifacts)
            outputs[artifacts.name] = artifacts
        plot_testreward_comparison(
            {name.capitalize(): artifacts.history for name, artifacts in outputs.items()},
            os.path.join(self.layout.figures, "effectiveness_testreward_comparison.png"),
        )
        return outputs


class BenchmarkSolverEngine:
    """Solve and summarize classical dynamic-programming benchmarks."""

    def __init__(
        self,
        *,
        mp: ModelParams,
        grid_params: GridBenchParams,
        layout: ExperimentLayout,
        methods: tuple[str, ...],
    ) -> None:
        """Initialize BenchmarkSolverEngine."""
        self.mp = mp
        self.grid_params = grid_params
        self.layout = layout
        self.methods = methods

    def solve(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Compute all requested benchmark methods and save their summaries."""
        benches: Dict[str, Dict[str, np.ndarray]] = {}
        for method in self.methods:
            bench = solve_grid_benchmark(self.mp, self.grid_params, inner_method=method, verbose=True)
            benches[method] = bench
            np.savez_compressed(
                os.path.join(self.layout.history, f"grid_benchmark_{method}.npz"),
                **bench,
            )
            plot_benchmark_method_summaries(
                bench,
                os.path.join(self.layout.figures, "benchmark_methods"),
                method,
            )
        if "vi" in benches and "mpi" in benches:
            metrics = plot_benchmark_method_comparison(
                benches["vi"],
                benches["mpi"],
                os.path.join(self.layout.figures, "benchmark_methods"),
            )
            with open(os.path.join(self.layout.history, "benchmark_vi_vs_mpi.json"), "w", encoding="utf-8") as f:
                json.dump(make_json_serializable(metrics), f, indent=2)
        return benches


class BenchmarkComparisonEngine:
    """Compare trained neural policies against solved grid benchmarks."""

    def __init__(
        self,
        *,
        mp: ModelParams,
        tp: TrainParams,
        layout: ExperimentLayout,
        compare_n: int,
    ) -> None:
        """Initialize BenchmarkComparisonEngine."""
        self.mp = mp
        self.tp = tp
        self.layout = layout
        self.compare_n = int(compare_n)

    def compare(
        self,
        trained: Mapping[str, ObjectiveTrainingArtifacts],
        benches: Mapping[str, Dict[str, np.ndarray]],
    ) -> None:
        """Run NN-vs-benchmark comparisons for each objective and method."""
        bench_fig_root = os.path.join(self.layout.figures, "benchmark_compare")
        os.makedirs(bench_fig_root, exist_ok=True)

        for objective_name, artifacts in trained.items():
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
            n_keep = min(self.compare_n, len(k_buf))
            idx = np.random.choice(
                len(k_buf),
                size=n_keep,
                replace=False if len(k_buf) >= n_keep else True,
            )
            k_e, b_e, z_e = k_buf[idx], b_buf[idx], z_buf[idx]

            for bench_method, bench in benches.items():
                comparator = BenchmarkComparator(
                    mp=self.mp,
                    kappa_issue=getattr(self.tp, "kappa_issue", 0.0),
                    config=BenchmarkComparatorConfig(
                        out_dir=os.path.join(bench_fig_root, bench_method),
                        tag=f"{objective_name}_{bench_method}",
                        objective_name=objective_name,
                        benchmark_ergodic_seed=self.tp.seed + 8500,
                    ),
                )
                metrics = comparator.compare(
                    policy=artifacts.policy,
                    value=artifacts.value,
                    qnet=artifacts.qnet,
                    k_e=k_e,
                    b_e=b_e,
                    z_e=z_e,
                    bench=bench,
                )
                with open(
                    os.path.join(self.layout.history, f"compare_{objective_name}_{bench_method}.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(make_json_serializable(metrics), f, indent=2)

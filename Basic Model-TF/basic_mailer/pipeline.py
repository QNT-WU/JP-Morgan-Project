"""Object-oriented pipeline orchestration for the basic Mailer package.

This module concentrates the end-to-end experiment workflow inside package
classes so the client-facing codebase remains installable, testable, and easy
to extend while preserving the legacy command-line interface.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, replace
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from basic_mailer.benchmark.comparison import (
    compare_on_state_sample,
    full_grid_state_sample,
    simulate_benchmark_ergodic_dataset,
)
from basic_mailer.benchmark.solvers import (
    GridSpec,
    policy_from_idx,
    precompute_arrays,
    solve_howard_pi,
    solve_vfi,
)
from basic_mailer.config import ModelParams, NetParams, Obj2Params, Obj3Params, TrainParams
from basic_mailer.estimation import (
    TwoStepGMMEstimator,
    TwoStepSMMEstimator,
    _train_policy_obj2_inner,
    build_default_moment_spec,
    make_crn_design,
    path_sample_size,
    save_estimation_report,
    save_json,
    simulate_paths_crn,
    structural_params_from_model,
    transform_params_to_tilde,
    update_model_params,
    write_csv,
    write_latex_table,
)
from basic_mailer.io_utils import JSONLLogger, TFCheckpointIO
from basic_mailer.networks import MultiplierNet, PolicyNet, ValueNet
from basic_mailer.plotting import (
    plot_effectiveness_obj1,
    plot_effectiveness_obj23,
    plot_ergodic_set,
    plot_testreward_comparison,
    save_effectiveness_report,
)
from basic_mailer.simulation import simulate_ergodic_dataset
from basic_mailer.training import Objective1Trainer, Objective2Trainer, Objective3Trainer


@dataclass(frozen=True)
class PipelineArgs:
    """Typed configuration for the end-to-end experiment pipeline."""

    out: str
    epochs: int = 40
    steps_per_epoch: int = 50
    batch_size: int = 512
    hidden_units: int = 64
    seed: int = 123
    resume: bool = False
    no_benchmark: bool = False
    no_estimation: bool = False
    do_estimation: bool = False
    est_max_evals: int = 140
    est_inner_epochs: int = 3
    est_inner_steps: int = 10
    est_data_paths: int = 64
    est_data_T: int = 200
    est_data_burn: int = 50
    est_simulation_ratio: float = 1.0
    est_truth_obj: str = "obj1"

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> "PipelineArgs":
        """Build typed pipeline arguments from ``argparse`` output."""
        return cls(**vars(namespace))


@dataclass(frozen=True)
class OutputLayout:
    """Filesystem layout used by the pipeline."""

    out_dir: str
    fig_dir: str
    log_dir: str
    ckpt_dir: str

    @classmethod
    def from_out_dir(cls, out_dir: str) -> "OutputLayout":
        """Create the directory layout rooted at ``out_dir``."""
        return cls(
            out_dir=out_dir,
            fig_dir=os.path.join(out_dir, "figures"),
            log_dir=os.path.join(out_dir, "logs"),
            ckpt_dir=os.path.join(out_dir, "checkpoints"),
        )

    def create(self) -> None:
        """Materialize all pipeline directories on disk."""
        os.makedirs(self.fig_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)


@dataclass
class ObjectiveArtifacts:
    """Store the trained objects and history for one objective."""

    name: str
    policy: PolicyNet
    history: dict[str, list[float]]
    value: ValueNet | None = None
    multiplier: MultiplierNet | None = None


class BasicMailerPipeline:
    """Run training, benchmarking, and estimation for the basic model package."""

    def __init__(self, args: PipelineArgs) -> None:
        """Initialize the experiment pipeline and derive runtime configuration objects.

        The constructor materializes the package-level configuration used across
        training, benchmarking, evaluation, logging, and estimation while
        preserving the existing numerical defaults exposed by the CLI.
        """
        self.args = args
        self.layout = OutputLayout.from_out_dir(args.out)
        self.mp = ModelParams()
        self.npol = NetParams(hidden_units=args.hidden_units, hidden_layers=2, activation="tanh")
        self.nval = NetParams(hidden_units=args.hidden_units, hidden_layers=2, activation="tanh")
        self.tp = TrainParams(
            seed=args.seed,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
        )
        self.op2 = Obj2Params(nu_lambda=1.0)
        self.op3 = Obj3Params(nu_fb=1.0, nu_lambda=1.0)
        self.obj1: ObjectiveArtifacts | None = None
        self.obj2: ObjectiveArtifacts | None = None
        self.obj3: ObjectiveArtifacts | None = None

    @property
    def progress_stride(self) -> int:
        """Return the epoch interval used for console progress reporting."""
        return max(1, self.tp.epochs // 10)

    def run(self) -> None:
        """Execute the full experiment pipeline."""
        self.layout.create()
        self.obj1 = self._train_objective_1()
        self.obj2 = self._train_objective_2()
        self.obj3 = self._train_objective_3()
        plot_testreward_comparison(
            self.obj1.history,
            self.obj2.history,
            self.obj3.history,
            self.layout.fig_dir,
        )
        save_effectiveness_report(
            {
                "obj1": self.obj1.history,
                "obj2": self.obj2.history,
                "obj3": self.obj3.history,
            },
            self.layout.out_dir,
        )
        self._run_benchmark()
        self._run_estimation()

    def _train_objective_1(self) -> ObjectiveArtifacts:
        """Train the Objective 1 policy and save diagnostics."""
        print("\n========================")
        print("Train Objective 1")
        print("========================")
        logger = JSONLLogger(os.path.join(self.layout.log_dir, "obj1.jsonl"))
        policy = PolicyNet(self.npol, self.mp.k_min, self.mp.k_max)
        _ = policy(tf.zeros((1, 2), dtype=tf.float32))
        optimizer = tf.keras.optimizers.Adam(self.tp.lr_policy)
        ckptio = TFCheckpointIO(
            directory=os.path.join(self.layout.ckpt_dir, "obj1"),
            policy=policy,
            opt_policy=optimizer,
            value=None,
            opt_value=None,
            max_to_keep=3,
        )
        if self.args.resume:
            restored = ckptio.restore_latest()
            print(
                f"[Obj1] resume={self.args.resume}, restored={restored}, "
                f"latest={ckptio.latest_checkpoint}"
            )
        trainer = Objective1Trainer(
            mp=self.mp,
            tp=self.tp,
            policy=policy,
            optimizer=optimizer,
            jsonl_logger=logger,
            ckptio=ckptio,
            verbose=True,
            progress_stride=self.progress_stride,
        )
        policy, history = trainer.train()
        plot_effectiveness_obj1(history, self.layout.fig_dir)
        plot_ergodic_set(
            policy,
            self.mp,
            self.tp,
            seed=self.tp.seed + 900,
            out_path=os.path.join(self.layout.fig_dir, "ergodic_set_obj1.png"),
        )
        return ObjectiveArtifacts(name="obj1", policy=policy, history=history)

    def _train_objective_2(self) -> ObjectiveArtifacts:
        """Train the Objective 2 policy and save diagnostics."""
        print("\n========================")
        print("Train Objective 2")
        print("========================")
        logger = JSONLLogger(os.path.join(self.layout.log_dir, "obj2.jsonl"))
        policy = PolicyNet(self.npol, self.mp.k_min, self.mp.k_max)
        multiplier = MultiplierNet(self.npol)
        _ = policy(tf.zeros((1, 2), dtype=tf.float32))
        _ = multiplier(tf.zeros((1, 2), dtype=tf.float32))
        policy_optimizer = tf.keras.optimizers.Adam(self.tp.lr_policy)
        multiplier_optimizer = tf.keras.optimizers.Adam(self.tp.lr_policy)
        ckptio = TFCheckpointIO(
            directory=os.path.join(self.layout.ckpt_dir, "obj2"),
            policy=policy,
            opt_policy=policy_optimizer,
            value=None,
            opt_value=None,
            multiplier=multiplier,
            opt_multiplier=multiplier_optimizer,
            max_to_keep=3,
        )
        if self.args.resume:
            restored = ckptio.restore_latest()
            print(
                f"[Obj2] resume={self.args.resume}, restored={restored}, "
                f"latest={ckptio.latest_checkpoint}"
            )
        trainer = Objective2Trainer(
            mp=self.mp,
            tp=self.tp,
            op2=self.op2,
            policy=policy,
            multiplier=multiplier,
            policy_optimizer=policy_optimizer,
            multiplier_optimizer=multiplier_optimizer,
            jsonl_logger=logger,
            ckptio=ckptio,
            verbose=True,
            progress_stride=self.progress_stride,
        )
        policy, multiplier, history = trainer.train()
        plot_effectiveness_obj23(history, self.layout.fig_dir, obj_name="Obj2")
        plot_ergodic_set(
            policy,
            self.mp,
            self.tp,
            seed=self.tp.seed + 901,
            out_path=os.path.join(self.layout.fig_dir, "ergodic_set_obj2.png"),
        )
        return ObjectiveArtifacts(name="obj2", policy=policy, multiplier=multiplier, history=history)

    def _train_objective_3(self) -> ObjectiveArtifacts:
        """Train the Objective 3 policy/value pair and save diagnostics."""
        print("\n========================")
        print("Train Objective 3")
        print("========================")
        logger = JSONLLogger(os.path.join(self.layout.log_dir, "obj3.jsonl"))
        policy = PolicyNet(self.npol, self.mp.k_min, self.mp.k_max)
        value = ValueNet(self.nval)
        multiplier = MultiplierNet(self.npol)
        _ = policy(tf.zeros((1, 2), dtype=tf.float32))
        _ = value(tf.zeros((1, 2), dtype=tf.float32))
        _ = multiplier(tf.zeros((1, 2), dtype=tf.float32))
        policy_optimizer = tf.keras.optimizers.Adam(self.tp.lr_policy)
        value_optimizer = tf.keras.optimizers.Adam(self.tp.lr_value)
        multiplier_optimizer = tf.keras.optimizers.Adam(self.tp.lr_policy)
        ckptio = TFCheckpointIO(
            directory=os.path.join(self.layout.ckpt_dir, "obj3"),
            policy=policy,
            opt_policy=policy_optimizer,
            value=value,
            opt_value=value_optimizer,
            multiplier=multiplier,
            opt_multiplier=multiplier_optimizer,
            max_to_keep=3,
        )
        if self.args.resume:
            restored = ckptio.restore_latest()
            print(
                f"[Obj3] resume={self.args.resume}, restored={restored}, "
                f"latest={ckptio.latest_checkpoint}"
            )
        trainer = Objective3Trainer(
            mp=self.mp,
            tp=self.tp,
            op3=self.op3,
            policy=policy,
            value=value,
            multiplier=multiplier,
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer,
            multiplier_optimizer=multiplier_optimizer,
            jsonl_logger=logger,
            ckptio=ckptio,
            verbose=True,
            progress_stride=self.progress_stride,
        )
        policy, value, multiplier, history = trainer.train()
        plot_effectiveness_obj23(history, self.layout.fig_dir, obj_name="Obj3")
        plot_ergodic_set(
            policy,
            self.mp,
            self.tp,
            seed=self.tp.seed + 902,
            out_path=os.path.join(self.layout.fig_dir, "ergodic_set_obj3.png"),
        )
        return ObjectiveArtifacts(name="obj3", policy=policy, value=value, multiplier=multiplier, history=history)

    @staticmethod
    def _save_curve(y: Sequence[float], ylabel: str, title: str, out_path: str) -> None:
        """Persist a one-dimensional convergence curve as a PNG figure."""
        plt.figure(figsize=(6, 4))
        xs = list(range(1, len(y) + 1))
        plt.plot(xs, y)
        plt.xlabel("Iteration")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

    def _run_benchmark(self) -> None:
        """Run the grid benchmark and compare trained policies to it."""
        assert self.obj1 is not None and self.obj2 is not None and self.obj3 is not None
        if self.args.no_benchmark:
            print("\n[Skip] --no_benchmark provided: skipping grid benchmark + comparison.")
            return

        print("\n========================")
        print("Grid Benchmark (Step 1): VFI / Howard PI")
        print("========================")

        gs = GridSpec(Nk=200, Nz=7, k_min=0.05, k_max=8.0, tauchen_m=3.0)
        pre = precompute_arrays(self.mp, gs)

        vfi_t0 = time.perf_counter()
        V_star_vfi, pi_idx_star_vfi, vfi_info = solve_vfi(
            pre,
            tol=1e-7,
            max_iter=3000,
            return_info=True,
        )
        vfi_info["runtime_seconds"] = float(time.perf_counter() - vfi_t0)

        pi_t0 = time.perf_counter()
        V_star_pi, pi_idx_star_pi, pi_info = solve_howard_pi(
            pre,
            eval_sweeps=80,
            max_outer=200,
            return_info=True,
        )
        pi_info["runtime_seconds"] = float(time.perf_counter() - pi_t0)

        V_star = V_star_vfi
        pi_idx_star = pi_idx_star_vfi
        pi_star = policy_from_idx(pre.k_grid, pi_idx_star)
        pi_star_pi = policy_from_idx(pre.k_grid, pi_idx_star_pi)

        bench_diag_dir = os.path.join(self.layout.fig_dir, "benchmark_methods")
        os.makedirs(bench_diag_dir, exist_ok=True)
        self._save_curve(
            vfi_info.get("diff_history", []),
            ylabel="sup |V^(n+1)-V^(n)|",
            title="VFI convergence",
            out_path=os.path.join(bench_diag_dir, "vfi_convergence.png"),
        )
        self._save_curve(
            pi_info.get("policy_changes_history", []),
            ylabel="# changed policy states",
            title="Howard PI convergence (policy changes)",
            out_path=os.path.join(bench_diag_dir, "howard_pi_policy_changes.png"),
        )
        self._save_curve(
            pi_info.get("value_diff_history", []),
            ylabel="proxy sup value change",
            title="Howard PI convergence (value proxy)",
            out_path=os.path.join(bench_diag_dir, "howard_pi_value_proxy.png"),
        )

        z_indices = sorted({0, len(pre.z_grid) // 2, len(pre.z_grid) - 1})
        plt.figure(figsize=(6, 4))
        for m in z_indices:
            plt.plot(pre.k_grid, V_star_vfi[:, m], label=f"z[{m}]={pre.z_grid[m]:.3f}")
        plt.xlabel("k")
        plt.ylabel("V*(k,z)")
        plt.title("Benchmark value function (VFI)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(bench_diag_dir, "benchmark_value_function_vfi.png"),
            dpi=150,
        )
        plt.close()

        plt.figure(figsize=(6, 4))
        for m in z_indices:
            plt.plot(pre.k_grid, pi_star[:, m], label=f"z[{m}]={pre.z_grid[m]:.3f}")
        plt.xlabel("k")
        plt.ylabel("k'(k,z)")
        plt.title("Benchmark policy function (VFI)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(bench_diag_dir, "benchmark_policy_function_vfi.png"),
            dpi=150,
        )
        plt.close()

        solver_compare = {
            "value_sup_diff_vfi_vs_pi": float(np.max(np.abs(V_star_vfi - V_star_pi))),
            "value_rmse_vfi_vs_pi": float(np.sqrt(np.mean((V_star_vfi - V_star_pi) ** 2))),
            "policy_sup_diff_vfi_vs_pi": float(np.max(np.abs(pi_star - pi_star_pi))),
            "policy_rmse_vfi_vs_pi": float(np.sqrt(np.mean((pi_star - pi_star_pi) ** 2))),
            "policy_index_agreement_rate": float(np.mean(pi_idx_star_vfi == pi_idx_star_pi)),
        }
        benchmark_methods_summary = {
            "vfi": vfi_info,
            "howard_pi": pi_info,
            "vfi_vs_pi": solver_compare,
            "benchmark_used_for_nn_comparison": "vfi_primary_with_howard_pi_overlay_in_plots",
        }
        with open(
            os.path.join(self.layout.log_dir, "benchmark_methods_summary.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(benchmark_methods_summary, handle, indent=2)

        from basic_mailer.estimation.reporting import write_csv, write_latex_table
        table_dir = os.path.join(self.layout.out_dir, "tables")
        os.makedirs(table_dir, exist_ok=True)
        benchmark_method_rows = [
            {
                "method": "VFI",
                "converged": vfi_info.get("converged"),
                "iterations_or_outer_iterations": vfi_info.get("iterations"),
                "runtime_seconds": vfi_info.get("runtime_seconds"),
                "final_diff_or_policy_changes": vfi_info.get("final_diff"),
            },
            {
                "method": "Howard_PI",
                "converged": pi_info.get("converged"),
                "iterations_or_outer_iterations": pi_info.get("outer_iterations"),
                "runtime_seconds": pi_info.get("runtime_seconds"),
                "final_diff_or_policy_changes": pi_info.get("final_policy_changes"),
                "eval_sweeps": pi_info.get("eval_sweeps"),
            },
        ]
        benchmark_agreement_rows = [{"diagnostic": key, "value": value} for key, value in solver_compare.items()]
        write_csv(benchmark_method_rows, os.path.join(table_dir, "benchmark_methods_summary.csv"))
        write_latex_table(
            benchmark_method_rows,
            os.path.join(table_dir, "benchmark_methods_summary.tex"),
            caption="Classical benchmark solver convergence and runtime",
            label="tab:benchmark_methods_summary",
        )
        write_csv(benchmark_agreement_rows, os.path.join(table_dir, "benchmark_solver_agreement.csv"))
        write_latex_table(
            benchmark_agreement_rows,
            os.path.join(table_dir, "benchmark_solver_agreement.tex"),
            caption="Agreement between VFI and Howard policy-iteration benchmark solutions",
            label="tab:benchmark_solver_agreement",
        )

        print(
            "[Benchmark methods] VFI:",
            {k: v for k, v in vfi_info.items() if k != "diff_history"},
        )
        print(
            "[Benchmark methods] Howard PI:",
            {
                k: v
                for k, v in pi_info.items()
                if k not in ["policy_changes_history", "value_diff_history"]
            },
        )
        print("[Benchmark methods] VFI vs PI:", solver_compare)

        print("\n========================")
        print(
            "Comparison (Step 3): NN vs Benchmark on Full Grid / Benchmark Ergodic / "
            "NN Ergodic"
        )
        print("========================")

        cmp_dir = os.path.join(self.layout.fig_dir, "benchmark_compare")
        os.makedirs(cmp_dir, exist_ok=True)

        k_full, z_full = full_grid_state_sample(pre.k_grid, pre.z_grid)
        k_bench_erg, z_bench_erg = simulate_benchmark_ergodic_dataset(
            policy_star=pi_star,
            k_grid=pre.k_grid,
            z_grid=pre.z_grid,
            mp=self.mp,
            tp=self.tp,
            seed=self.tp.seed + 500,
        )
        k1_nn, z1_nn = simulate_ergodic_dataset(
            self.obj1.policy,
            self.mp,
            self.tp,
            seed=self.tp.seed + 501,
        )
        k2_nn, z2_nn = simulate_ergodic_dataset(
            self.obj2.policy,
            self.mp,
            self.tp,
            seed=self.tp.seed + 502,
        )
        k3_nn, z3_nn = simulate_ergodic_dataset(
            self.obj3.policy,
            self.mp,
            self.tp,
            seed=self.tp.seed + 503,
        )

        region_specs_common = [
            ("full_grid", k_full, z_full),
            ("benchmark_ergodic", k_bench_erg, z_bench_erg),
        ]
        objective_specs = [
            ("obj1", self.obj1.policy, None, (k1_nn, z1_nn)),
            ("obj2", self.obj2.policy, None, (k2_nn, z2_nn)),
            ("obj3", self.obj3.policy, self.obj3.value, (k3_nn, z3_nn)),
        ]
        bench_logger = JSONLLogger(
            os.path.join(self.layout.log_dir, "benchmark_compare.jsonl")
        )

        print("\n--- Benchmark comparison summary ---")
        benchmark_rows = []
        for obj_name, policy_nn, value_nn_or_none, (k_nn_erg, z_nn_erg) in objective_specs:
            local_regions = list(region_specs_common) + [("nn_ergodic", k_nn_erg, z_nn_erg)]
            for region_name, k_states, z_states in local_regions:
                out_subdir = os.path.join(cmp_dir, region_name)
                _, summary = compare_on_state_sample(
                    policy_nn=policy_nn,
                    value_nn_or_none=value_nn_or_none,
                    k_states=k_states,
                    z_states=z_states,
                    k_grid=pre.k_grid,
                    z_grid=pre.z_grid,
                    V_star=V_star,
                    policy_star=pi_star,
                    u=pre.u,
                    Pz=pre.Pz,
                    beta=pre.beta,
                    out_dir=out_subdir,
                    tag=f"{obj_name}_{region_name}",
                    V_star_alt=V_star_pi,
                    policy_star_alt=pi_star_pi,
                    alt_label="Howard PI",
                )
                print(f"{obj_name} [{region_name}]:", summary)
                row = {"objective": obj_name, "region": region_name, **summary}
                benchmark_rows.append(row)
                bench_logger.log(row)

        write_csv(benchmark_rows, os.path.join(table_dir, "benchmark_comparison_summary.csv"))
        write_latex_table(
            benchmark_rows,
            os.path.join(table_dir, "benchmark_comparison_summary.tex"),
            caption="Neural-network versus benchmark comparison across state regions",
            label="tab:benchmark_comparison_summary",
        )

    def _run_estimation(self) -> None:
        """Run the corrected two-step GMM and SMM estimators."""
        assert self.obj1 is not None and self.obj2 is not None and self.obj3 is not None
        if not self.args.do_estimation or self.args.no_estimation:
            print("\n[Skip] --no_estimation provided: skipping SMM/GMM estimation.")
            return

        print("\n========================")
        print("Estimation (Part 2): corrected two-step SMM + GMM")
        print("========================")

        est_dir = os.path.join(self.layout.out_dir, "estimation")
        os.makedirs(est_dir, exist_ok=True)

        if self.args.est_truth_obj == "obj1":
            truth_policy = self.obj1.policy
        elif self.args.est_truth_obj == "obj2":
            truth_policy = self.obj2.policy
        else:
            truth_policy = self.obj3.policy

        obs_design = make_crn_design(
            n_paths=self.args.est_data_paths,
            T=self.args.est_data_T,
            seed=self.args.seed + 777,
        )
        ds_truth = simulate_paths_crn(
            policy=truth_policy,
            mp=self.mp,
            design=obs_design,
            burn_in=self.args.est_data_burn,
        )
        np.savez(
            os.path.join(est_dir, "synthetic_truth_paths.npz"),
            k=ds_truth.k,
            z=ds_truth.z,
        )

        sim_paths = max(
            int(round(self.args.est_data_paths * max(self.args.est_simulation_ratio, 1e-6))),
            1,
        )
        sim_design = make_crn_design(
            n_paths=sim_paths,
            T=self.args.est_data_T - self.args.est_data_burn,
            seed=self.args.seed + 778,
        )

        tp_est = replace(
            self.tp,
            epochs=self.args.est_inner_epochs,
            steps_per_epoch=self.args.est_inner_steps,
            ergodic_burn_in=300,
            ergodic_T=1500,
            ergodic_n_paths=16,
            ergodic_buffer_size=50000,
        )
        true_params = structural_params_from_model(self.mp)
        x0 = transform_params_to_tilde(**true_params)
        spec = build_default_moment_spec()

        gmm = TwoStepGMMEstimator(
            mp_template=self.mp,
            observed_dataset=ds_truth,
            seed=self.args.seed + 900,
        )
        gmm_results = gmm.fit(x0=x0, max_evals=self.args.est_max_evals)

        smm = TwoStepSMMEstimator(
            mp_template=self.mp,
            npol=self.npol,
            inner_tp=tp_est,
            observed_dataset=ds_truth,
            simulation_design=sim_design,
            moment_spec=spec,
            seed=self.args.seed + 901,
        )
        smm_results = smm.fit(x0=x0, max_evals=self.args.est_max_evals)

        combined_results = {}
        combined_results.update(gmm_results)
        combined_results.update(smm_results)
        save_estimation_report(combined_results, true_params=true_params, out_dir=est_dir)

        k_common = ds_truth.k_curr.reshape(-1)[:5000]
        z_common = ds_truth.z_curr.reshape(-1)[:5000]
        x_common = tf.convert_to_tensor(np.stack([k_common, z_common], axis=1), tf.float32)
        kprime_truth = tf.clip_by_value(
            truth_policy(x_common),
            self.mp.k_min,
            self.mp.k_max,
        ).numpy()
        policy_rows = []
        for method_name, result in combined_results.items():
            mp_hat = update_model_params(self.mp, result.final_params)
            pol_hat = _train_policy_obj2_inner(mp=mp_hat, npol=self.npol, tp=tp_est)
            kprime_hat = tf.clip_by_value(
                pol_hat(x_common),
                self.mp.k_min,
                self.mp.k_max,
            ).numpy()
            policy_rows.append(
                {
                    "method": method_name,
                    "weight_method": result.weight_method,
                    "policy_mse_vs_truth": float(np.mean((kprime_hat - kprime_truth) ** 2)),
                    "final_success": bool(result.stage2.success),
                }
            )
        write_csv(policy_rows, os.path.join(est_dir, "table_policy_distance.csv"))
        write_latex_table(
            policy_rows,
            os.path.join(est_dir, "table_policy_distance.tex"),
            caption="Policy-function distance versus synthetic truth",
            label="tab:policy_distance",
        )

        save_json(
            {
                "true_params": true_params,
                "observed_sample_size": path_sample_size(ds_truth),
                "simulated_sample_size": sim_paths * (self.args.est_data_T - self.args.est_data_burn),
                "methods": {key: value.to_flat_dict() for key, value in combined_results.items()},
                "policy_distance": policy_rows,
            },
            os.path.join(est_dir, "estimation_summary.json"),
        )

        est_logger = JSONLLogger(os.path.join(self.layout.log_dir, "estimation.jsonl"))
        for row in policy_rows:
            method = row["method"]
            est_logger.log({**combined_results[method].to_flat_dict(), **row})

        print("\n--- Estimation summary (saved under outputs/.../estimation/) ---")
        for method in ["GMM_A", "GMM_B", "SMM_A", "SMM_B"]:
            if method in combined_results:
                print(method, combined_results[method].final_params)
                print(
                    "  stage1_success =",
                    combined_results[method].stage1.success,
                    "stage2_success =",
                    combined_results[method].stage2.success,
                    "final_success =",
                    combined_results[method].stage2.success,
                )
                print(
                    "  stage2_best_loss =",
                    combined_results[method].stage2.best_loss,
                    "winner_start =",
                    combined_results[method].stage2.best_start_id,
                )


def build_argument_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser used by both entrypoints."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory, e.g. outputs/run1",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--steps_per_epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--hidden_units", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Restore latest checkpoints if they exist",
    )
    parser.add_argument(
        "--no_benchmark",
        action="store_true",
        help="Skip grid benchmark + NN-vs-grid comparison",
    )
    parser.add_argument(
        "--no_estimation",
        action="store_true",
        help="Skip estimation (SMM + GMM)",
    )
    parser.add_argument(
        "--do_estimation",
        action="store_true",
        help="Run estimation (SMM + GMM) after NN and (optionally) benchmark",
    )
    parser.add_argument(
        "--est_max_evals",
        type=int,
        default=140,
        help="Maximum number of outer evaluations (nested solves) for Nelder-Mead",
    )
    parser.add_argument("--est_inner_epochs", type=int, default=3)
    parser.add_argument("--est_inner_steps", type=int, default=10)
    parser.add_argument("--est_data_paths", type=int, default=64)
    parser.add_argument("--est_data_T", type=int, default=200)
    parser.add_argument("--est_data_burn", type=int, default=50)
    parser.add_argument("--est_simulation_ratio", type=float, default=1.0)
    parser.add_argument(
        "--est_truth_obj",
        type=str,
        default="obj1",
        choices=["obj1", "obj2", "obj3"],
        help="Which trained NN to treat as truth solver for synthetic data",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> PipelineArgs:
    """Parse CLI arguments into a typed ``PipelineArgs`` instance."""
    return PipelineArgs.from_namespace(build_argument_parser().parse_args(argv))


def main(argv: Sequence[str] | None = None) -> None:
    """Run the package pipeline from the command line."""
    BasicMailerPipeline(parse_args(argv)).run()

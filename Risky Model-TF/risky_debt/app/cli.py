"""Command-line interface for ``python -m Experiment.run_all`` and direct package use."""

from __future__ import annotations

import argparse

from .application import RunAllApplication, RunAllConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the full experiment workflow."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True, help="Output directory, e.g. outputs/run1")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--steps_per_epoch", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_units", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--T_train", type=int, default=80, help="Objective 1 rollout horizon used during training.")
    parser.add_argument("--N_paths_train", type=int, default=128, help="Objective 1 rollout paths used during training.")
    parser.add_argument("--N_q", type=int, default=8, help="Inner Monte Carlo draws used in constructed risky-debt pricing.")
    parser.add_argument("--ergodic_burn_in", type=int, default=300, help="Burn-in used when refreshing policy-induced training state buffers.")
    parser.add_argument("--ergodic_T", type=int, default=1500, help="Horizon used when refreshing policy-induced training state buffers.")
    parser.add_argument("--ergodic_n_paths", type=int, default=12, help="Number of paths used when refreshing policy-induced training state buffers.")
    parser.add_argument("--ergodic_buffer_size", type=int, default=30000, help="Maximum retained states in the ergodic training buffer.")
    parser.add_argument("--obj2_steps_per_epoch", type=int, default=0, help="Objective 2 steps per epoch. Use 0 to inherit --steps_per_epoch.")
    parser.add_argument("--obj2_batch_size", type=int, default=0, help="Objective 2 batch size. Use 0 to inherit --batch_size.")
    parser.add_argument("--obj2_N_q", type=int, default=0, help="Objective 2 inner pricing draws. Use 0 to inherit --N_q.")
    parser.add_argument("--obj2_ergodic_burn_in", type=int, default=0, help="Objective 2 ergodic-buffer burn-in. Use 0 to inherit --ergodic_burn_in.")
    parser.add_argument("--obj2_ergodic_T", type=int, default=0, help="Objective 2 ergodic-buffer horizon. Use 0 to inherit --ergodic_T.")
    parser.add_argument("--obj2_ergodic_n_paths", type=int, default=0, help="Objective 2 ergodic-buffer paths. Use 0 to inherit --ergodic_n_paths.")
    parser.add_argument("--obj2_ergodic_buffer_size", type=int, default=0, help="Objective 2 retained ergodic states. Use 0 to inherit --ergodic_buffer_size.")
    parser.add_argument("--obj3_steps_per_epoch", type=int, default=0, help="Objective 3 steps per epoch. Use 0 to inherit --steps_per_epoch.")
    parser.add_argument("--obj3_batch_size", type=int, default=0, help="Objective 3 batch size. Use 0 to inherit --batch_size.")
    parser.add_argument("--obj3_N_q", type=int, default=0, help="Objective 3 inner pricing draws. Use 0 to inherit --N_q.")
    parser.add_argument("--obj3_ergodic_burn_in", type=int, default=0, help="Objective 3 ergodic-buffer burn-in. Use 0 to inherit --ergodic_burn_in.")
    parser.add_argument("--obj3_ergodic_T", type=int, default=0, help="Objective 3 ergodic-buffer horizon. Use 0 to inherit --ergodic_T.")
    parser.add_argument("--obj3_ergodic_n_paths", type=int, default=0, help="Objective 3 ergodic-buffer paths. Use 0 to inherit --ergodic_n_paths.")
    parser.add_argument("--obj3_ergodic_buffer_size", type=int, default=0, help="Objective 3 retained ergodic states. Use 0 to inherit --ergodic_buffer_size.")
    parser.add_argument("--bench_method", type=str, default="both", choices=["vi", "mpi", "both"])
    parser.add_argument("--Nk", type=int, default=60)
    parser.add_argument("--Nb", type=int, default=61)
    parser.add_argument("--Nz", type=int, default=7)
    parser.add_argument("--k_max", type=float, default=8)
    parser.add_argument("--z_m", type=float, default=6.0)
    parser.add_argument("--compare_N", type=int, default=5000)
    parser.add_argument("--no_benchmark", action="store_true", help="Skip grid benchmark and NN-vs-grid comparison")
    parser.add_argument("--do_estimation", action="store_true", help="Run frequentist estimation")
    parser.add_argument("--no_estimation", action="store_true", help="Skip frequentist estimation")
    parser.add_argument(
        "--estimation_method",
        type=str,
        default="both",
        choices=["both", "smm", "gmm"],
        help=(
            "Frequentist estimator(s) to run. 'both' runs SMM then GMM; "
            "'smm' runs only SMM; 'gmm' loads finalized SMM/synthetic-data outputs and runs only GMM."
        ),
    )
    parser.add_argument("--est_max_evals", type=int, default=60)
    parser.add_argument("--est_inner_epochs", type=int, default=3)
    parser.add_argument("--est_inner_steps", type=int, default=10)
    parser.add_argument("--est_T", type=int, default=200)
    parser.add_argument("--est_burn", type=int, default=50)
    parser.add_argument("--est_n_paths", type=int, default=64)
    parser.add_argument("--est_n_starts", type=int, default=3)
    parser.add_argument("--est_cont_horizon", type=int, default=0)
    parser.add_argument(
        "--gmm_moment_mode",
        type=str,
        default="derivative",
        choices=["derivative", "pricing_only"],
        help=(
            "GMM moment system. 'derivative' is the written-summary FOC/ZP system; "
            "'pricing_only' is a faster optional GMM mode that keeps only instrumented zero-profit pricing moments."
        ),
    )
    parser.add_argument(
        "--estimation_report_mode",
        type=str,
        default="full",
        choices=["full", "light"],
        help=(
            "Estimation report mode. 'full' writes all diagnostics; 'light' writes core SMM/GMM "
            "results and lightweight comparison tables while skipping heavy post-GMM diagnostics/plots."
        ),
    )
    parser.add_argument("--do_hmc", action="store_true", help="Run Bayesian estimation")
    parser.add_argument("--no_hmc", action="store_true", help="Skip Bayesian estimation")
    parser.add_argument("--hmc_num_results", type=int, default=48)
    parser.add_argument("--hmc_num_burnin", type=int, default=48)
    parser.add_argument("--hmc_num_chains", type=int, default=1)
    parser.add_argument("--hmc_step_size", type=float, default=0.04)
    parser.add_argument("--hmc_kernel", type=str, default="hmc", choices=["hmc", "rwm"])
    parser.add_argument("--hmc_max_obs", type=int, default=0)
    parser.add_argument("--hmc_max_paths", type=int, default=0)
    parser.add_argument("--hmc_num_particles", type=int, default=256, help="Legacy unsupported flag retained for compatibility.")
    parser.add_argument("--hmc_obs_sigma_lnz", type=float, default=0.02, help="Legacy unsupported flag retained for compatibility.")
    parser.add_argument("--no_resume", action="store_true", help="Disable automatic per-epoch training resume from outputs/<run>/checkpoints/*/latest.")
    args = parser.parse_args()
    if args.hmc_num_particles != 256:
        parser.error("--hmc_num_particles is no longer supported by the deterministic forward-filter Bayesian implementation.")
    if abs(args.hmc_obs_sigma_lnz - 0.02) > 1e-12:
        parser.error("--hmc_obs_sigma_lnz is no longer supported by the deterministic forward-filter Bayesian implementation.")
    return args


def main() -> None:
    """Instantiate the application layer and run the experiment."""
    args = parse_args()
    outputs = RunAllApplication(RunAllConfig(**vars(args))).run()
    layout = outputs["layout"]
    print("\nDONE.")
    print(f"Logs:    {layout.logs}")
    print(f"History: {layout.history}")
    print(f"Figures: {layout.figures}")
    print(f"Estimation: {layout.estimation}")
    print(f"Tables:  {layout.tables}")
    print(f"Bayes figures: {layout.figures}/bayes")
    print(f"Ckpts:   {layout.checkpoints}")

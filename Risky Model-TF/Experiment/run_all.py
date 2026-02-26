# Experiment/run_all.py
# run_all.py is the “one-button script”:
# It reads command-line options (epochs, batch size, seed, etc.).
# It creates the model parameters and training parameters.
# It trains Objective 1, then Objective 2, then Objective 3.
# After each objective, it saves:
# training history (.npz)
# plots (.png)
# model weights (.weights.h5)
# an ergodic-set scatter plot (k vs b)
# So: it’s a driver / orchestrator, not the model math itself.


from __future__ import annotations
import json
import numpy as np

import argparse
import os

from risky_debt.simulation import simulate_ergodic_dataset
from risky_debt.grid_benchmark import GridBenchParams, solve_grid_benchmark
from risky_debt.grid_compare import compare_nn_to_benchmark_on_ergodic


# Import config dataclasses (your parameter blocks)
# ModelParams (economics primitives)
# Matches your model environment:
# shock law (rho, sigma_eps)
# technology (theta, taxes tau)
# adjustment cost (psi0, depreciation delta)
# interest rate r, recovery haircut alpha, issuance costs eta0, eta1
# safe bounds like k_min, z_min

# NetParams (NN architecture)
# hidden layers, hidden units, activation

# TrainParams (training setup)
# epochs, batch size, learning rates
# ergodic simulation settings
# evaluation horizon, etc.

# Obj1Params / Obj2Params / Obj3Params
# the weights on residual terms for each objective.
# These are not “Test Euler residuals of Objective 1” by themselves.
# They are just knobs that the trainer uses when it computes the loss
from risky_debt.config import (
    ModelParams,
    NetParams,
    TrainParams,
    Obj1Params,
    Obj2Params,
    Obj3Params,
)

from risky_debt.io_utils import JSONLLogger
from risky_debt.trainer import train_objective_1, train_objective_2, train_objective_3
from risky_debt.plotting import (
    plot_effectiveness_obj1,
    plot_effectiveness_obj23,
    plot_ergodic_set_kb,
    save_hist_npz,
)


def domain_check(k_erg, b_erg, z_erg, bench, name=""):
    k_grid = bench["k_grid"]
    b_grid = bench["b_grid"]
    z_grid = bench["z_grid"]

    kmin, kmax = float(k_grid.min()), float(k_grid.max())
    bmin, bmax = float(b_grid.min()), float(b_grid.max())
    zmin, zmax = float(z_grid.min()), float(z_grid.max())

    print("\n" + "=" * 70)
    print(f"[DOMAIN CHECK] {name}")
    print(
        f"ergodic k: min={k_erg.min():.6g}, max={k_erg.max():.6g} | grid [{kmin:.6g}, {kmax:.6g}]"
    )
    print(
        f"ergodic b: min={b_erg.min():.6g}, max={b_erg.max():.6g} | grid [{bmin:.6g}, {bmax:.6g}]"
    )
    print(
        f"ergodic z: min={z_erg.min():.6g}, max={z_erg.max():.6g} | grid [{zmin:.6g}, {zmax:.6g}]"
    )

    bad_k = (k_erg < kmin) | (k_erg > kmax)
    bad_b = (b_erg < bmin) | (b_erg > bmax)
    bad_z = (z_erg < zmin) | (z_erg > zmax)

    nk, nb, nz = int(bad_k.sum()), int(bad_b.sum()), int(bad_z.sum())
    n = len(k_erg)
    print(f"out-of-domain counts: k={nk}/{n}, b={nb}/{n}, z={nz}/{n}")
    print("=" * 70 + "\n")


# (how you control runs without editing code)
# When the test runs, it executes:
# python -m Experiment.run_all --out /tmp/... --epochs 2 --steps_per_epoch 2 ...
# So parse_args is where those values get read.
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out", type=str, required=True, help="Output directory, e.g. outputs/run1"
    )
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--steps_per_epoch", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--hidden_units", type=int, default=128)
    p.add_argument("--hidden_layers", type=int, default=3)
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--bench_method", type=str, default="vi", choices=["vi", "mpi"])
    p.add_argument("--Nk", type=int, default=60)
    p.add_argument("--Nb", type=int, default=61)
    p.add_argument("--Nz", type=int, default=7)
    p.add_argument("--k_max", type=float, default=8)
    p.add_argument("--z_m", type=float, default=6.0)
    p.add_argument("--compare_N", type=int, default=5000)

    # ---------------- Estimation (SMM / GMM) ----------------
    # IMPORTANT (for Colab + pytest speed):
    # Estimation is OFF by default. Turn it on explicitly with --do_estimation.
    # We keep --no_estimation as a compatibility flag (it always forces skip).
    p.add_argument(
        "--no_benchmark",
        action="store_true",
        help="Skip grid benchmark and NN-vs-grid comparison",
    )
    p.add_argument(
        "--do_estimation",
        action="store_true",
        help="Run SMM + GMM estimation (OFF by default)",
    )
    p.add_argument(
        "--no_estimation",
        action="store_true",
        help="Skip SMM + GMM (compatibility flag)",
    )
    p.add_argument(
        "--est_max_evals", type=int, default=60, help="Outer optimizer max evaluations"
    )
    p.add_argument(
        "--est_inner_epochs",
        type=int,
        default=3,
        help="Inner solve epochs per outer eval",
    )
    p.add_argument(
        "--est_inner_steps",
        type=int,
        default=10,
        help="Inner solve steps/epoch per outer eval",
    )
    p.add_argument(
        "--est_T",
        type=int,
        default=200,
        help="Forward simulation horizon for synthetic data",
    )
    p.add_argument(
        "--est_burn", type=int, default=50, help="Burn-in for forward simulation"
    )
    p.add_argument(
        "--est_n_paths",
        type=int,
        default=64,
        help="Number of parallel paths for simulation",
    )

    # ---------------- Bayesian estimation (HMC/MCMC + filtering) ----------------
    # ON/OFF switch. Default is OFF to keep Colab + pytest fast.
    p.add_argument(
        "--do_hmc",
        action="store_true",
        help="Run Bayesian estimation (HMC/MCMC + filtering) (OFF by default)",
    )
    p.add_argument(
        "--no_hmc",
        action="store_true",
        help="Skip Bayesian estimation (compatibility flag)",
    )

    # Budget controls (keep defaults small; increase for real runs)
    p.add_argument("--hmc_num_results", type=int, default=200)
    p.add_argument("--hmc_num_burnin", type=int, default=200)
    p.add_argument("--hmc_num_chains", type=int, default=2)
    p.add_argument("--hmc_step_size", type=float, default=0.03)
    p.add_argument(
        "--hmc_kernel",
        type=str,
        default="rwm",
        choices=["rwm", "hmc"],
        help="MCMC kernel: 'rwm' (robust, no gradients) or 'hmc' (needs gradients)",
    )
    p.add_argument(
        "--hmc_num_particles",
        type=int,
        default=256,
        help="Particle count for likelihood estimation",
    )
    p.add_argument(
        "--hmc_obs_sigma_lnz",
        type=float,
        default=0.02,
        help="Measurement noise std for observed ln z (likelihood)",
    )

    return p.parse_args()


# (saving neural network weights)
def _save_weights_safe(model, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Keras will infer format from suffix; .weights.h5 is the simplest for Colab
    model.save_weights(path)


# main() starts: set output folders
def main() -> None:
    args = parse_args()
    out_dir = args.out

    fig_dir = os.path.join(out_dir, "figures")
    log_dir = os.path.join(out_dir, "logs")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    hist_dir = os.path.join(out_dir, "history")

    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(hist_dir, exist_ok=True)

    # Create parameters (model, training, network)
    mp = ModelParams()
    # Training params
    tp = TrainParams(
        seed=args.seed,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
    )

    # Network params
    npol = NetParams(
        hidden_units=args.hidden_units,
        hidden_layers=args.hidden_layers,
        activation="tanh",
    )
    nval = NetParams(
        hidden_units=args.hidden_units,
        hidden_layers=args.hidden_layers,
        activation="tanh",
    )
    nvt = NetParams(
        hidden_units=args.hidden_units,
        hidden_layers=args.hidden_layers,
        activation="tanh",
    )
    nq = NetParams(
        hidden_units=args.hidden_units,
        hidden_layers=args.hidden_layers,
        activation="tanh",
    )

    # Objective weight blocks
    op1 = Obj1Params(nu_zp=1.0)
    op2 = Obj2Params(nu_def=1.0, nu_bell=1.0, nu_foc=1.0, nu_zp=1.0)
    op3 = Obj3Params(nu_def=1.0, nu_zp=1.0)

    # ---------------- Obj1 ----------------
    print("\n========================")
    print("Train Objective 1 (Reward + ZP discipline)")
    print("========================")
    obj1_logger = JSONLLogger(os.path.join(log_dir, "obj1.jsonl"))

    # policy1: your NN for (k', b') = φ(k,b,z)
    # qnet1: your NN for q(z,k',b') (pricing)
    # hist1: recorded train_reward, test_reward, test_euler_mse
    policy1, qnet1, hist1 = train_objective_1(
        mp,
        npol,
        nq,
        tp,
        op1,
        jsonl_logger=obj1_logger,
        ckptio=None,
    )

    save_hist_npz(os.path.join(hist_dir, "hist_obj1.npz"), hist1)
    plot_effectiveness_obj1(hist1, os.path.join(fig_dir, "obj1"))

    # REQUIRED by integration test: checkpoints/obj1 contains files
    _save_weights_safe(policy1, os.path.join(ckpt_dir, "obj1", "policy.weights.h5"))
    _save_weights_safe(qnet1, os.path.join(ckpt_dir, "obj1", "qnet.weights.h5"))

    # REQUIRED by integration test: figures/ergodic_set_obj1.png
    plot_ergodic_set_kb(
        policy1,
        mp,
        tp,
        seed=tp.seed + 900,
        out_path=os.path.join(fig_dir, "ergodic_set_obj1.png"),
    )

    # ---------------- Obj2 ----------------
    print("\n========================")
    print("Train Objective 2 (Residual system)")
    print("========================")
    obj2_logger = JSONLLogger(os.path.join(log_dir, "obj2.jsonl"))

    # Same structure, but it trains 4 objects:
    # value2 = V(k,b,z) (equity value, constrained ≥0)
    # vtilde2 = \tilde V(k,b,z) (continuation value, should be unconstrained — meaning you want VtildeNet, not ValueNet, as we discussed)
    # qnet2 = q(z,k',b')
    policy2, value2, vtilde2, qnet2, hist2 = train_objective_2(
        mp,
        npol,
        nval,
        nvt,
        nq,
        tp,
        op2,
        jsonl_logger=obj2_logger,
        ckptio=None,
    )

    save_hist_npz(os.path.join(hist_dir, "hist_obj2.npz"), hist2)
    plot_effectiveness_obj23(hist2, os.path.join(fig_dir, "obj2"), obj_name="Obj2")

    _save_weights_safe(policy2, os.path.join(ckpt_dir, "obj2", "policy.weights.h5"))
    _save_weights_safe(value2, os.path.join(ckpt_dir, "obj2", "value.weights.h5"))
    _save_weights_safe(vtilde2, os.path.join(ckpt_dir, "obj2", "vtilde.weights.h5"))
    _save_weights_safe(qnet2, os.path.join(ckpt_dir, "obj2", "qnet.weights.h5"))

    plot_ergodic_set_kb(
        policy2,
        mp,
        tp,
        seed=tp.seed + 901,
        out_path=os.path.join(fig_dir, "ergodic_set_obj2.png"),
    )

    # ---------------- Obj3 ----------------
    print("\n========================")
    print("Train Objective 3 (Bellman/default + ZP)")
    print("========================")
    obj3_logger = JSONLLogger(os.path.join(log_dir, "obj3.jsonl"))

    # No vtilde net here because Obj3 directly uses:
    # Vtilde_eval = d + β V(next) inside the loss.
    policy3, value3, qnet3, hist3 = train_objective_3(
        mp,
        npol,
        nval,
        nq,
        tp,
        op3,
        jsonl_logger=obj3_logger,
        ckptio=None,
    )

    save_hist_npz(os.path.join(hist_dir, "hist_obj3.npz"), hist3)
    plot_effectiveness_obj23(hist3, os.path.join(fig_dir, "obj3"), obj_name="Obj3")

    _save_weights_safe(policy3, os.path.join(ckpt_dir, "obj3", "policy.weights.h5"))
    _save_weights_safe(value3, os.path.join(ckpt_dir, "obj3", "value.weights.h5"))
    _save_weights_safe(qnet3, os.path.join(ckpt_dir, "obj3", "qnet.weights.h5"))

    plot_ergodic_set_kb(
        policy3,
        mp,
        tp,
        seed=tp.seed + 902,
        out_path=os.path.join(fig_dir, "ergodic_set_obj3.png"),
    )

    # ---------------- Step 1 + Step 3: Benchmark + Compare ----------------
    if not args.no_benchmark:
        print("\n========================")
        print("Step 1 + Step 3: Grid Benchmark + Compare on Ergodic Set")
        print("========================")

        # bench_params = GridBenchParams(Nk=args.Nk, Nb=args.Nb, Nz=args.Nz, k_max=args.k_max)
        bench_params = GridBenchParams(
            Nk=args.Nk, Nb=args.Nb, Nz=args.Nz, k_max=args.k_max, z_m=args.z_m
        )

        bench = solve_grid_benchmark(
            mp, bench_params, inner_method=args.bench_method, verbose=True
        )

        bench_path = os.path.join(hist_dir, f"grid_benchmark_{args.bench_method}.npz")
        np.savez_compressed(bench_path, **bench)
        print(f"Saved benchmark to: {bench_path}")

        bench_fig_dir = os.path.join(fig_dir, "benchmark_compare")
        os.makedirs(bench_fig_dir, exist_ok=True)

        def _compare_one(obj_name: str, policy, value):
            k_buf, b_buf, z_buf = simulate_ergodic_dataset(
                policy, mp, tp, seed=tp.seed + 8000
            )
            N = min(args.compare_N, len(k_buf))
            idx = np.random.choice(
                len(k_buf), size=N, replace=False if len(k_buf) >= N else True
            )
            k_e, b_e, z_e = k_buf[idx], b_buf[idx], z_buf[idx]

            metrics = compare_nn_to_benchmark_on_ergodic(
                policy=policy,
                value=value,
                k_e=k_e,
                b_e=b_e,
                z_e=z_e,
                bench=bench,
                mp=mp,
                kappa_issue=getattr(mp, "kappa_issue", 0.0),
                out_dir=bench_fig_dir,
                tag=f"{obj_name}_{args.bench_method}",
            )

            out_json = os.path.join(
                hist_dir, f"compare_{obj_name}_{args.bench_method}.json"
            )
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            print(f"[{obj_name}] saved compare metrics: {out_json}")
            print(metrics)

        _compare_one("obj1", policy1, value=None)
        _compare_one("obj2", policy2, value=value2)
        _compare_one("obj3", policy3, value=value3)
    else:
        print("\n========================")
        print("Skipped benchmark + compare because --no_benchmark was passed.")
        print("========================")

    # ---------------- Step 4: Estimation (SMM + GMM) ----------------
    # OFF by default. Enable with --do_estimation.
    if args.do_estimation and (not args.no_estimation):
        print("\n========================")
        print("Step 4: Estimation (SMM + GMM) on Synthetic Data")
        print("========================")

        from estimation.smm import estimate_smm
        from estimation.gmm import estimate_gmm

        est_dir = os.path.join(out_dir, "estimation")
        os.makedirs(est_dir, exist_ok=True)

        # Which structural parameters to estimate (your 5D core set)
        # You can tighten/widen bounds anytime; these are safe defaults.
        est_bounds = {
            "theta": (0.10, 0.90),
            "rho": (0.50, 0.995),
            "sigma_eps": (0.005, 0.20),
            "psi0": (0.10, 10.0),
            "alpha": (0.01, 0.95),
        }

        # SMM uses policy/q from Obj1 as the DGP for synthetic data.
        smm_res = estimate_smm(
            out_dir=est_dir,
            mp_true=mp,
            npol=npol,
            nq=nq,
            tp_base=tp,
            policy_true=policy1,
            qnet_true=qnet1,
            est_bounds=est_bounds,
            max_evals=args.est_max_evals,
            inner_epochs=args.est_inner_epochs,
            inner_steps_per_epoch=args.est_inner_steps,
            sim_T=args.est_T,
            sim_burn=args.est_burn,
            sim_n_paths=args.est_n_paths,
            seed=args.seed + 777,
        )

        print("\n[SMM] Results (saved to outputs/.../estimation/smm_results.json)")
        print(smm_res)

        # GMM runs on the fixed synthetic dataset created above
        synth_path = os.path.join(est_dir, "smm_synth_data.npz")
        d = dict(np.load(synth_path))
        gmm_res = estimate_gmm(
            out_dir=est_dir,
            mp_true=mp,
            npol=npol,
            nq=nq,
            tp_base=tp,
            data=d,
            est_bounds=est_bounds,
            max_evals=args.est_max_evals,
            inner_epochs=args.est_inner_epochs,
            inner_steps_per_epoch=args.est_inner_steps,
            seed=args.seed + 888,
        )
        print("\n[GMM] Results (saved to outputs/.../estimation/gmm_results.json)")
        print(gmm_res)

    else:
        print("\n========================")
        print("Skipped estimation (default). Use --do_estimation to enable.")
        print("========================")

    # ---------------- Step 5: Bayesian Estimation (HMC/MCMC + filtering) ----------------
    # OFF by default. Enable with --do_hmc.
    if args.do_hmc and (not args.no_hmc):
        print("\n========================")
        print("Step 5: Bayesian Estimation (MCMC + filtering)")
        print("========================")

        from estimation.bayes import estimate_hmc

        est_dir = os.path.join(out_dir, "estimation")
        os.makedirs(est_dir, exist_ok=True)

        # Reuse the same synthetic dataset generated by SMM if it exists;
        # otherwise, fall back to a short forward simulation using Obj1.
        synth_path = os.path.join(est_dir, "smm_synth_data.npz")
        if os.path.exists(synth_path):
            data = dict(np.load(synth_path))
        else:
            # If user runs --do_hmc without --do_estimation, we still create a small dataset.
            from estimation.smm import forward_simulate_dataset

            rng = np.random.default_rng(args.seed + 999)
            eps = rng.normal(
                0.0,
                mp.sigma_eps,
                size=(max(8, args.est_n_paths), max(50, args.est_T) + 1),
            ).astype(np.float32)
            data = forward_simulate_dataset(
                policy=policy1,
                qnet=qnet1,
                mp=mp,
                tp=tp,
                eps=eps,
                T=max(50, args.est_T),
                burn_in=max(10, args.est_burn),
            )
            np.savez_compressed(synth_path, **data)
            print(f"[HMC] Created synthetic data at: {synth_path}")

        hmc_res = estimate_hmc(
            out_dir=est_dir,
            mp_true=mp,
            data=data,
            kernel=args.hmc_kernel,
            num_results=args.hmc_num_results,
            num_burnin=args.hmc_num_burnin,
            num_chains=args.hmc_num_chains,
            step_size=args.hmc_step_size,
            num_particles=args.hmc_num_particles,
            obs_sigma_lnz=args.hmc_obs_sigma_lnz,
            seed=args.seed + 3333,
        )

        print("\n[HMC] Results (saved to outputs/.../estimation/hmc_results.json)")
        print(hmc_res)
    else:
        print("\n========================")
        print("Skipped Bayesian estimation (default). Use --do_hmc to enable.")
        print("========================")

    print("\nDONE.")
    print(f"Logs:    {log_dir}")
    print(f"History: {hist_dir}")
    print(f"Figures: {fig_dir}")
    print(f"Ckpts:   {ckpt_dir}")


if __name__ == "__main__":
    main()

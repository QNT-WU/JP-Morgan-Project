from __future__ import annotations
from basic_mailer.grid_benchmark import (
    GridSpec,
    precompute_arrays,
    solve_vfi,
    solve_howard_pi,
    policy_from_idx,
)
from basic_mailer.grid_compare import compare_on_ergodic_states

# argparse makes it runnable as a script with flags like --out outputs/run1
import argparse
import os

import tensorflow as tf


from basic_mailer.config import ModelParams, NetParams, TrainParams, Obj3Params
from basic_mailer.networks import PolicyNet, ValueNet
from basic_mailer.io_utils import JSONLLogger, TFCheckpointIO
from basic_mailer.trainer import train_objective_1, train_objective_2, train_objective_3
from basic_mailer.plotting import (
    plot_effectiveness_obj1,
    plot_effectiveness_obj23,
    plot_testreward_comparison,
    plot_ergodic_set,
)


# Defines command line flags:
# --out (required): output directory
# --epochs, --steps_per_epoch, --batch_size: training loop controls
# --hidden_units: network width
# --seed: random seed
# --resume: if set, restore latest checkpoints
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out", type=str, required=True, help="Output directory, e.g. outputs/run1"
    )
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--steps_per_epoch", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--hidden_units", type=int, default=64)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument(
        "--resume", action="store_true", help="Restore latest checkpoints if they exist"
    )

    # -----------------------
    # Optional pipeline skips
    # -----------------------
    # By default, we run: NN training -> grid benchmark -> estimation (SMM+GMM).
    # You can skip later stages using these flags.
    p.add_argument(
        "--no_benchmark",
        action="store_true",
        help="Skip grid benchmark + NN-vs-grid comparison",
    )
    p.add_argument(
        "--no_estimation",
        action="store_true",
        help="Skip estimation (SMM + GMM)",
    )
    # -----------------------
    # Estimation (Part 2)
    # -----------------------
    # By default, estimation is OFF (so pytest/integration runs fast).
    # Enable it explicitly with --do_estimation. You can also force it OFF with --no_estimation.
    p.add_argument(
        "--do_estimation",
        action="store_true",
        help="Run estimation (SMM + GMM) after NN and (optionally) benchmark",
    )
    p.add_argument(
        "--est_max_evals",
        type=int,
        default=120,
        help="Maximum number of outer evaluations (nested solves) for Nelder-Mead",
    )
    p.add_argument("--est_inner_epochs", type=int, default=3)
    p.add_argument("--est_inner_steps", type=int, default=10)
    p.add_argument("--est_data_paths", type=int, default=64)
    p.add_argument("--est_data_T", type=int, default=200)
    p.add_argument("--est_data_burn", type=int, default=50)
    p.add_argument(
        "--est_truth_obj",
        type=str,
        default="obj1",
        choices=["obj1", "obj2", "obj3"],
        help="Which trained NN to treat as truth solver for synthetic data",
    )
    return p.parse_args()


# main(): prepare folders
# This ensures the output tree exists:
def main() -> None:
    args = parse_args()
    out_dir = args.out
    fig_dir = os.path.join(out_dir, "figures")
    log_dir = os.path.join(out_dir, "logs")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Build parameter objects
    # mp uses defaults (rho, sigma_eps, psi0, delta, r, k_min…)
    # npol, nval set the NN architecture
    mp = ModelParams()
    npol = NetParams(hidden_units=args.hidden_units, hidden_layers=2, activation="tanh")
    nval = NetParams(hidden_units=args.hidden_units, hidden_layers=2, activation="tanh")

    tp = TrainParams(
        seed=args.seed,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
    )
    op3 = Obj3Params(nu=1.0)

    # ---------------------------
    # Objective 1: build nets/opt for checkpointing + logging
    # ---------------------------
    print("\n========================")
    print("Train Objective 1")
    print("========================")
    # Console header + logger
    # This creates:outputs/run1/logs/obj1.jsonl
    # Each epoch writes one JSON record.
    obj1_logger = JSONLLogger(os.path.join(log_dir, "obj1.jsonl"))

    # Create a policy + optimizer here so checkpoint restore can happen BEFORE training.

    # policy1 = PolicyNet(npol, mp.k_min)
    policy1 = PolicyNet(npol, mp.k_min, mp.k_max)

    _ = policy1(tf.zeros((1, 2), dtype=tf.float32))

    opt1 = tf.keras.optimizers.Adam(tp.lr_policy)

    # Create checkpoint manager
    # So checkpoints go to:outputs/run1/checkpoints/obj1/
    ckptio1 = TFCheckpointIO(
        directory=os.path.join(ckpt_dir, "obj1"),
        policy=policy1,
        opt_policy=opt1,
        value=None,
        opt_value=None,
        max_to_keep=3,
    )

    # If a checkpoint exists, it loads policy weights and optimizer state
    if args.resume:
        restored = ckptio1.restore_latest()
        print(
            f"[Obj1] resume={args.resume}, restored={restored}, latest={ckptio1.latest_checkpoint}"
        )

    # Now train using the same objects by calling trainer directly would require trainer to accept existing objects.
    # To keep your trainer signature stable, we simply train via trainer which creates its own objects.
    # So here is the clean approach:
    #   - checkpointing/logging happens inside trainer, BUT we need ckptio built from the same policy/opt.
    # Therefore, use the trainer as written (it expects ckptio already bound to its own objects).
    # EASIEST: run training via trainer and let it create policy/opt and then checkpoint those objects.
    #
    # So we do it this way: call trainer with jsonl_logger and a ckptio constructed *after* trainer creates objects.
    # That requires a tiny wrapper approach: we will not pre-restore for Obj1 here (to avoid mismatched objects).
    #
    # If you truly need resume-from-checkpoint with trainer-created objects, we can add "train_objective_1_from_objects".
    # For now, we implement resume in a robust way by using "from objects" below.

    # ---- Robust resume-capable path: train using explicit objects ----
    # We'll implement training locally here with the same logic, so policy1/opt1 are the ones checkpointed/restored.
    # But to avoid rewriting all loops, the simplest is: you tell me if you want "from objects" trainers.
    #
    # Since you asked "do both", I'll provide the cleanest engineering approach now:
    # -> We add three small "from objects" wrappers inside this file.

    # ============== Obj1 training (from objects) ==============
    from basic_mailer.objectives import obj1_loss
    from basic_mailer.simulation import simulate_ergodic_dataset
    from basic_mailer.evaluation import (
        eval_test_reward,
        eval_test_euler_mse_policy_only,
    )
    import numpy as np

    # Training loop “from objects”
    def clip_and_apply(opt, grads, vars_, clip):
        grads, _ = tf.clip_by_global_norm(grads, clip)
        opt.apply_gradients(zip(grads, vars_))

    # generate ergodic buffer:
    k_buf, z_buf = simulate_ergodic_dataset(policy1, mp, tp, seed=tp.seed + 10)
    hist1 = {
        "epoch": [],
        "train_reward": [],
        "train_loss": [],
        "test_reward": [],
        "test_euler_mse": [],
    }

    # per epoch:
    # do steps_per_epoch gradient updates using obj1_loss
    # refresh ergodic buffer periodically
    # sample test states from buffer
    # compute: test_reward; test_euler_mse (policy-only Euler diagnostic)
    # log JSONL + save ckpt
    for epoch in range(1, tp.epochs + 1):
        tr = []
        for _ in range(tp.steps_per_epoch):
            with tf.GradientTape() as tape:
                loss, train_reward = obj1_loss(policy1, mp, tp)
            grads = tape.gradient(loss, policy1.trainable_variables)
            clip_and_apply(opt1, grads, policy1.trainable_variables, tp.grad_clip)
            tr.append(float(train_reward.numpy()))

        if epoch == 1 or (epoch % tp.ergodic_refresh_every == 0):
            k_buf, z_buf = simulate_ergodic_dataset(
                policy1, mp, tp, seed=tp.seed + 100 + epoch
            )

        idx = np.random.choice(len(k_buf), size=tp.N_test_states, replace=True)
        k_test = k_buf[idx]
        z_test = z_buf[idx]

        test_reward = eval_test_reward(policy1, mp, tp, seed=tp.seed + 200 + epoch)
        test_euler_mse = eval_test_euler_mse_policy_only(
            policy1, mp, k_test, z_test, N_eps=tp.N_eps_test, seed=tp.seed + 300 + epoch
        )
        train_reward_ep = float(np.mean(tr))
        train_loss_ep = -train_reward_ep  # ← ADD THIS LINE

        hist1["epoch"].append(epoch)
        hist1["train_reward"].append(train_reward_ep)
        hist1["train_loss"].append(train_loss_ep)  # ← ADD THIS LINE
        hist1["test_reward"].append(test_reward)
        hist1["test_euler_mse"].append(test_euler_mse)

        obj1_logger.log(
            {
                "objective": "obj1",
                "epoch": epoch,
                "train_reward": train_reward_ep,
                "train_loss": train_loss_ep,  # ← ADD THIS LINE
                "test_reward": test_reward,
                "test_euler_mse": test_euler_mse,
            }
        )
        ckptio1.save(step=epoch)

        if epoch == 1 or epoch % max(1, tp.epochs // 10) == 0:
            print(
                f"[Obj1][{epoch:03d}] TrainReward={train_reward_ep:.4f} "
                f"TestReward={test_reward:.4f} TestEulerMSE={test_euler_mse:.6f}"
            )

    plot_effectiveness_obj1(hist1, fig_dir)
    plot_ergodic_set(
        policy1,
        mp,
        tp,
        seed=tp.seed + 900,
        out_path=os.path.join(fig_dir, "ergodic_set_obj1.png"),
    )

    # ---------------------------
    # Objective 2
    # ---------------------------
    # Same structure:
    # creates obj2_logger → logs/obj2.jsonl
    # creates policy2, opt2
    # creates ckptio2 → checkpoints/obj2/
    # optional restore
    # build ergodic buffer
    # per epoch:
    # refresh buffer occasionally
    # sample minibatches from buffer
    # train with obj2_batch_loss
    # evaluate:
    # test_euler_mse using policy-only f
    # test_reward rollout
    # log + checkpoint
    print("\n========================")
    print("Train Objective 2")
    print("========================")
    obj2_logger = JSONLLogger(os.path.join(log_dir, "obj2.jsonl"))

    # policy2 = PolicyNet(npol, mp.k_min)

    policy2 = PolicyNet(npol, mp.k_min, mp.k_max)

    _ = policy2(tf.zeros((1, 2), dtype=tf.float32))

    opt2 = tf.keras.optimizers.Adam(tp.lr_policy)

    ckptio2 = TFCheckpointIO(
        directory=os.path.join(ckpt_dir, "obj2"),
        policy=policy2,
        opt_policy=opt2,
        value=None,
        opt_value=None,
        max_to_keep=3,
    )
    if args.resume:
        restored = ckptio2.restore_latest()
        print(
            f"[Obj2] resume={args.resume}, restored={restored}, latest={ckptio2.latest_checkpoint}"
        )

    from basic_mailer.objectives import obj2_batch_loss

    k_buf, z_buf = simulate_ergodic_dataset(policy2, mp, tp, seed=tp.seed + 11)
    hist2 = {"epoch": [], "train_loss": [], "test_euler_mse": [], "test_reward": []}

    for epoch in range(1, tp.epochs + 1):
        if epoch == 1 or (epoch % tp.ergodic_refresh_every == 0):
            k_buf, z_buf = simulate_ergodic_dataset(
                policy2, mp, tp, seed=tp.seed + 110 + epoch
            )

        losses = []
        for _ in range(tp.steps_per_epoch):
            idx = np.random.choice(len(k_buf), size=tp.batch_size, replace=True)
            k = tf.convert_to_tensor(k_buf[idx], tf.float32)
            z = tf.convert_to_tensor(z_buf[idx], tf.float32)

            with tf.GradientTape() as tape:
                loss = obj2_batch_loss(policy2, mp, k, z)
            grads = tape.gradient(loss, policy2.trainable_variables)
            clip_and_apply(opt2, grads, policy2.trainable_variables, tp.grad_clip)
            losses.append(float(loss.numpy()))

        idx_t = np.random.choice(len(k_buf), size=tp.N_test_states, replace=True)
        k_test = k_buf[idx_t]
        z_test = z_buf[idx_t]

        test_euler_mse = eval_test_euler_mse_policy_only(
            policy2, mp, k_test, z_test, N_eps=tp.N_eps_test, seed=tp.seed + 301 + epoch
        )
        test_reward = eval_test_reward(policy2, mp, tp, seed=tp.seed + 201 + epoch)

        train_loss_ep = float(np.mean(losses))
        hist2["epoch"].append(epoch)
        hist2["train_loss"].append(train_loss_ep)
        hist2["test_euler_mse"].append(test_euler_mse)
        hist2["test_reward"].append(test_reward)

        obj2_logger.log(
            {
                "objective": "obj2",
                "epoch": epoch,
                "train_loss": train_loss_ep,
                "test_euler_mse": test_euler_mse,
                "test_reward": test_reward,
            }
        )
        ckptio2.save(step=epoch)

        if epoch == 1 or epoch % max(1, tp.epochs // 10) == 0:
            print(
                f"[Obj2][{epoch:03d}] TrainLoss={train_loss_ep:.6f} "
                f"TestEulerMSE={test_euler_mse:.6f} TestReward={test_reward:.4f}"
            )

    plot_effectiveness_obj23(hist2, fig_dir, obj_name="Obj2")
    plot_ergodic_set(
        policy2,
        mp,
        tp,
        seed=tp.seed + 901,
        out_path=os.path.join(fig_dir, "ergodic_set_obj2.png"),
    )

    # ---------------------------
    # Objective 3
    # ---------------------------
    print("\n========================")
    print("Train Objective 3")
    print("========================")
    obj3_logger = JSONLLogger(os.path.join(log_dir, "obj3.jsonl"))

    # policy3 = PolicyNet(npol, mp.k_min)
    policy3 = PolicyNet(npol, mp.k_min, mp.k_max)

    value3 = ValueNet(nval)

    _ = policy3(tf.zeros((1, 2), dtype=tf.float32))
    _ = value3(tf.zeros((1, 2), dtype=tf.float32))

    optp3 = tf.keras.optimizers.Adam(tp.lr_policy)
    optv3 = tf.keras.optimizers.Adam(tp.lr_value)

    ckptio3 = TFCheckpointIO(
        directory=os.path.join(ckpt_dir, "obj3"),
        policy=policy3,
        opt_policy=optp3,
        value=value3,
        opt_value=optv3,
        max_to_keep=3,
    )
    if args.resume:
        restored = ckptio3.restore_latest()
        print(
            f"[Obj3] resume={args.resume}, restored={restored}, latest={ckptio3.latest_checkpoint}"
        )

    from basic_mailer.objectives import obj3_batch_loss

    k_buf, z_buf = simulate_ergodic_dataset(policy3, mp, tp, seed=tp.seed + 12)
    hist3 = {"epoch": [], "train_loss": [], "test_euler_mse": [], "test_reward": []}

    from basic_mailer.evaluation import eval_test_euler_mse_obj3

    for epoch in range(1, tp.epochs + 1):
        if epoch == 1 or (epoch % tp.ergodic_refresh_every == 0):
            k_buf, z_buf = simulate_ergodic_dataset(
                policy3, mp, tp, seed=tp.seed + 120 + epoch
            )

        losses = []
        for _ in range(tp.steps_per_epoch):
            idx = np.random.choice(len(k_buf), size=tp.batch_size, replace=True)
            k = tf.convert_to_tensor(k_buf[idx], tf.float32)
            z = tf.convert_to_tensor(z_buf[idx], tf.float32)

            with tf.GradientTape(persistent=True) as tape:
                loss = obj3_batch_loss(policy3, value3, mp, op3, k, z)
            grads_p = tape.gradient(loss, policy3.trainable_variables)
            grads_v = tape.gradient(loss, value3.trainable_variables)
            del tape

            clip_and_apply(optp3, grads_p, policy3.trainable_variables, tp.grad_clip)
            clip_and_apply(optv3, grads_v, value3.trainable_variables, tp.grad_clip)

            losses.append(float(loss.numpy()))

        idx_t = np.random.choice(len(k_buf), size=tp.N_test_states, replace=True)
        k_test = k_buf[idx_t]
        z_test = z_buf[idx_t]

        test_euler_mse = eval_test_euler_mse_obj3(
            policy3,
            value3,
            mp,
            k_test,
            z_test,
            N_eps=tp.N_eps_test,
            seed=tp.seed + 302 + epoch,
        )
        test_reward = eval_test_reward(policy3, mp, tp, seed=tp.seed + 202 + epoch)

        train_loss_ep = float(np.mean(losses))
        hist3["epoch"].append(epoch)
        hist3["train_loss"].append(train_loss_ep)
        hist3["test_euler_mse"].append(test_euler_mse)
        hist3["test_reward"].append(test_reward)

        obj3_logger.log(
            {
                "objective": "obj3",
                "epoch": epoch,
                "train_loss": train_loss_ep,
                "test_euler_mse": test_euler_mse,
                "test_reward": test_reward,
            }
        )
        ckptio3.save(step=epoch)

        if epoch == 1 or epoch % max(1, tp.epochs // 10) == 0:
            print(
                f"[Obj3][{epoch:03d}] TrainLoss={train_loss_ep:.6f} "
                f"TestEulerMSE={test_euler_mse:.6f} TestReward={test_reward:.4f}"
            )

    plot_effectiveness_obj23(hist3, fig_dir, obj_name="Obj3")
    plot_ergodic_set(
        policy3,
        mp,
        tp,
        seed=tp.seed + 902,
        out_path=os.path.join(fig_dir, "ergodic_set_obj3.png"),
    )

    # combined comparison plot (uses in-memory hists)
    plot_testreward_comparison(hist1, hist2, hist3, fig_dir)

    # ======================================================
    # Step 1 + Step 3: Benchmark + NN-vs-grid comparison
    # ======================================================
    if not args.no_benchmark:
        print("\n========================")
        print("Grid Benchmark (Step 1): VFI / Howard PI")
        print("========================")

        gs = GridSpec(Nk=200, Nz=7, k_min=0.05, k_max=8.0, tauchen_m=3.0)
        pre = precompute_arrays(mp, gs)

        # Choose ONE benchmark solver:
        # (A) VFI:
        V_star, pi_idx_star = solve_vfi(pre, tol=1e-7, max_iter=3000)

        # (B) Howard PI (uncomment to use instead of VFI):
        # V_star, pi_idx_star = solve_howard_pi(pre, eval_sweeps=80, max_outer=200)

        pi_star = policy_from_idx(pre.k_grid, pi_idx_star)

        print("\n========================")
        print("Comparison (Step 3): NN vs Benchmark on Ergodic Set")
        print("========================")

        cmp_dir = os.path.join(fig_dir, "benchmark_compare")
        os.makedirs(cmp_dir, exist_ok=True)

        # --- Obj1 ergodic sample + compare (policy-only NN)
        k1, z1 = simulate_ergodic_dataset(policy1, mp, tp, seed=tp.seed + 501)
        _, summary1 = compare_on_ergodic_states(
            policy_nn=policy1,
            value_nn_or_none=None,
            k_erg=k1,
            z_erg=z1,
            k_grid=pre.k_grid,
            z_grid=pre.z_grid,
            V_star=V_star,
            policy_star=pi_star,
            u=pre.u,
            Pz=pre.Pz,
            beta=pre.beta,
            out_dir=cmp_dir,
            tag="obj1",
        )

        # --- Obj2 ergodic sample + compare (policy-only NN)
        k2, z2 = simulate_ergodic_dataset(policy2, mp, tp, seed=tp.seed + 502)
        _, summary2 = compare_on_ergodic_states(
            policy_nn=policy2,
            value_nn_or_none=None,
            k_erg=k2,
            z_erg=z2,
            k_grid=pre.k_grid,
            z_grid=pre.z_grid,
            V_star=V_star,
            policy_star=pi_star,
            u=pre.u,
            Pz=pre.Pz,
            beta=pre.beta,
            out_dir=cmp_dir,
            tag="obj2",
        )

        # --- Obj3 ergodic sample + compare (policy + value NN)
        k3, z3 = simulate_ergodic_dataset(policy3, mp, tp, seed=tp.seed + 503)
        _, summary3 = compare_on_ergodic_states(
            policy_nn=policy3,
            value_nn_or_none=value3,
            k_erg=k3,
            z_erg=z3,
            k_grid=pre.k_grid,
            z_grid=pre.z_grid,
            V_star=V_star,
            policy_star=pi_star,
            u=pre.u,
            Pz=pre.Pz,
            beta=pre.beta,
            out_dir=cmp_dir,
            tag="obj3",
        )

        print("\n--- Benchmark comparison summary ---")
        print("Obj1:", summary1)
        print("Obj2:", summary2)
        print("Obj3:", summary3)

        bench_logger = JSONLLogger(os.path.join(log_dir, "benchmark_compare.jsonl"))
        bench_logger.log({"objective": "obj1", **summary1})
        bench_logger.log({"objective": "obj2", **summary2})
        bench_logger.log({"objective": "obj3", **summary3})
    else:
        print("\n[Skip] --no_benchmark provided: skipping grid benchmark + comparison.")

    # ======================================================
    # Part 2: Estimation (SMM + GMM)
    # Default: OFF. Enable with --do_estimation. Force OFF with --no_estimation.
    # ======================================================
    if args.do_estimation and (not args.no_estimation):
        print("\n========================")
        print("Estimation (Part 2): SMM + GMM")
        print("========================")

        from dataclasses import replace
        import json as _json
        import numpy as _np

        from basic_mailer.estimation.moments import (
            build_default_moment_spec,
            compute_moments,
            make_crn_design,
            make_identity_weight_matrix,
            simulate_paths_crn,
        )
        from basic_mailer.estimation.smm import (
            SMMEstimator,
            transform_tilde_to_theta,
            _train_policy_obj2_inner,
        )
        from basic_mailer.estimation.gmm import GMMEstimator

        est_dir = os.path.join(out_dir, "estimation")
        os.makedirs(est_dir, exist_ok=True)

        # 1) choose "truth policy" from the NN stage
        if args.est_truth_obj == "obj1":
            truth_policy = policy1
        elif args.est_truth_obj == "obj2":
            truth_policy = policy2
        else:
            truth_policy = policy3

        # 2) forward policy simulation under truth (synthetic dataset)
        #    IMPORTANT: this is forward simulation, not a grid benchmark.
        design = make_crn_design(
            n_paths=args.est_data_paths,
            T=args.est_data_T,
            seed=args.seed + 777,
        )
        ds_truth = simulate_paths_crn(
            policy=truth_policy,
            mp=mp,
            design=design,
            burn_in=args.est_data_burn,
        )

        _np.savez(
            os.path.join(est_dir, "synthetic_truth_paths.npz"),
            k=ds_truth.k,
            z=ds_truth.z,
            meta=_np.array(
                [
                    _json.dumps(
                        {
                            "truth_obj": args.est_truth_obj,
                            "n_paths": args.est_data_paths,
                            "T": args.est_data_T,
                            "burn_in": args.est_data_burn,
                        }
                    )
                ],
                dtype=object,
            ),
        )

        # target moments from synthetic data
        spec = build_default_moment_spec()
        m_hat = compute_moments(ds_truth, mp, spec)
        Wm = make_identity_weight_matrix(len(spec.names))

        # moments computed from the *truth* synthetic dataset.
        # compute these from the same forward-simulated ds_truth.
        k_t_truth = ds_truth.k_curr.reshape(-1)
        z_t_truth = ds_truth.z_curr.reshape(-1)
        x_truth = tf.convert_to_tensor(
            _np.stack([k_t_truth, z_t_truth], axis=1), tf.float32
        )
        k1_truth = tf.clip_by_value(truth_policy(x_truth), mp.k_min, mp.k_max).numpy()
        I_truth = k1_truth - (1.0 - mp.delta) * k_t_truth
        I_over_k_truth = I_truth / _np.maximum(k_t_truth, mp.k_min)
        invest_targets = {
            "mean_I_over_k": float(_np.mean(I_over_k_truth)),
            "var_I_over_k": float(_np.var(I_over_k_truth)),
        }

        # inner solve settings for nested estimation (set it small by default)
        tp_est = replace(
            tp,
            epochs=args.est_inner_epochs,
            steps_per_epoch=args.est_inner_steps,
            ergodic_burn_in=300,
            ergodic_T=1500,
            ergodic_n_paths=16,
            ergodic_buffer_size=50000,
        )

        # IMPORTANT: initialize outer search at CURRENT mp parameters.
        # If it starts from zeros, it maps to (0.5, 0.5, log2, log2),
        # can make Nelder-Mead stall.
        def _inv_sigmoid(p: float) -> float:
            p = float(_np.clip(p, 1e-6, 1.0 - 1e-6))
            return float(_np.log(p / (1.0 - p)))

        def _inv_softplus(y: float) -> float:
            y = float(max(y, 1e-8))
            return float(_np.log(_np.expm1(y)))

        x0 = _np.array(
            [
                _inv_sigmoid(mp.theta),
                _inv_sigmoid(mp.rho),
                _inv_softplus(mp.sigma_eps),
                _inv_softplus(mp.psi0),
            ],
            dtype=_np.float64,
        )

        # ---- SMM ----
        smm = SMMEstimator(
            mp_template=mp,
            npol=npol,
            inner_tp=tp_est,
            moment_spec=spec,
            W=Wm,
            crn_design=design,
            target_moments=m_hat,
            burn_in=args.est_data_burn,
        )
        x_smm, diag_smm = smm.fit(x0=x0, max_evals=args.est_max_evals)
        theta_smm = transform_tilde_to_theta(x_smm)

        mp_smm = replace(mp, **theta_smm)
        pol_smm = _train_policy_obj2_inner(mp=mp_smm, npol=npol, tp=tp_est)

        # common states for policy distance
        k_common = ds_truth.k_curr.reshape(-1)[:5000]
        z_common = ds_truth.z_curr.reshape(-1)[:5000]
        x_common = tf.convert_to_tensor(
            _np.stack([k_common, z_common], axis=1), tf.float32
        )
        kprime_truth = tf.clip_by_value(
            truth_policy(x_common), mp.k_min, mp.k_max
        ).numpy()
        kprime_smm = tf.clip_by_value(pol_smm(x_common), mp.k_min, mp.k_max).numpy()
        policy_mse_smm = float(_np.mean((kprime_smm - kprime_truth) ** 2))

        ds_smm = simulate_paths_crn(
            policy=pol_smm, mp=mp_smm, design=design, burn_in=args.est_data_burn
        )
        m_smm = compute_moments(ds_smm, mp_smm, spec)
        moment_gap_smm = {k: float(m_hat[k] - m_smm[k]) for k in spec.names}

        # ---- GMM ----
        gmm = GMMEstimator(
            mp_template=mp,
            npol=npol,
            inner_tp=tp_est,
            crn_design=design,
            n_states=2000,
            n_shocks=32,
            invest_targets=invest_targets,
            burn_in=args.est_data_burn,
            seed=args.seed + 888,
        )
        x_gmm, diag_gmm = gmm.fit(x0=x0, max_evals=args.est_max_evals)
        theta_gmm = transform_tilde_to_theta(x_gmm)

        mp_gmm = replace(mp, **theta_gmm)
        pol_gmm = _train_policy_obj2_inner(mp=mp_gmm, npol=npol, tp=tp_est)

        kprime_gmm = tf.clip_by_value(pol_gmm(x_common), mp.k_min, mp.k_max).numpy()
        policy_mse_gmm = float(_np.mean((kprime_gmm - kprime_truth) ** 2))

        ds_gmm = simulate_paths_crn(
            policy=pol_gmm, mp=mp_gmm, design=design, burn_in=args.est_data_burn
        )
        m_gmm = compute_moments(ds_gmm, mp_gmm, spec)
        moment_gap_gmm = {k: float(m_hat[k] - m_gmm[k]) for k in spec.names}

        # Investment moments gap (truth - candidate) for reporting
        k_t_gmm = ds_gmm.k_curr.reshape(-1)
        z_t_gmm = ds_gmm.z_curr.reshape(-1)
        x_gmm_states = tf.convert_to_tensor(
            _np.stack([k_t_gmm, z_t_gmm], axis=1), tf.float32
        )
        k1_gmm_states = tf.clip_by_value(
            pol_gmm(x_gmm_states), mp.k_min, mp.k_max
        ).numpy()
        I_gmm_states = k1_gmm_states - (1.0 - mp.delta) * k_t_gmm
        I_over_k_gmm = I_gmm_states / _np.maximum(k_t_gmm, mp.k_min)
        invest_gap_gmm = {
            "mean_I_over_k": float(
                invest_targets["mean_I_over_k"] - _np.mean(I_over_k_gmm)
            ),
            "var_I_over_k": float(
                invest_targets["var_I_over_k"] - _np.var(I_over_k_gmm)
            ),
        }

        # save report
        est_logger = JSONLLogger(os.path.join(log_dir, "estimation.jsonl"))

        def _param_errors(theta_hat: dict) -> dict:
            return {
                "err_theta": float(theta_hat["theta"] - mp.theta),
                "err_rho": float(theta_hat["rho"] - mp.rho),
                "err_sigma_eps": float(theta_hat["sigma_eps"] - mp.sigma_eps),
                "err_psi0": float(theta_hat["psi0"] - mp.psi0),
            }

        est_logger.log(
            {
                "method": "SMM",
                "theta_hat": theta_smm,
                **_param_errors(theta_smm),
                "policy_mse_vs_truth": policy_mse_smm,
                "moment_gap": moment_gap_smm,
                "outer": diag_smm,
            }
        )
        est_logger.log(
            {
                "method": "GMM",
                "theta_hat": theta_gmm,
                **_param_errors(theta_gmm),
                "policy_mse_vs_truth": policy_mse_gmm,
                "moment_gap": moment_gap_gmm,
                "invest_targets": invest_targets,
                "invest_gap": invest_gap_gmm,
                "outer": diag_gmm,
            }
        )

        print("\n--- Estimation summary (saved to logs/estimation.jsonl) ---")
        print("SMM theta_hat:", theta_smm)
        print("SMM outer:", diag_smm)
        print("SMM policy MSE vs truth:", policy_mse_smm)
        print("GMM theta_hat:", theta_gmm)
        print("GMM outer:", diag_gmm)
        print("GMM policy MSE vs truth:", policy_mse_gmm)

    else:
        print("\n[Skip] --no_estimation provided: skipping SMM/GMM estimation.")


if __name__ == "__main__":
    main()

from __future__ import annotations

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
    return p.parse_args()


# main(): prepare folders
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
    policy1 = PolicyNet(npol, mp.k_min)

    _ = policy1(tf.zeros((1, 2), dtype=tf.float32))

    opt1 = tf.keras.optimizers.Adam(tp.lr_policy)

    # Create checkpoint manager
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

    policy2 = PolicyNet(npol, mp.k_min)

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

    policy3 = PolicyNet(npol, mp.k_min)
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

    print("\nDONE.")
    print(f"Logs:        {log_dir} (obj1.jsonl / obj2.jsonl / obj3.jsonl)")
    print(f"Checkpoints: {ckpt_dir} (obj1/ obj2/ obj3/)")
    print(f"Figures:     {fig_dir}")


if __name__ == "__main__":
    main()

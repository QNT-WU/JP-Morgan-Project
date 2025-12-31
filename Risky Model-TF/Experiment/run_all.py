# Experiment/run_all.py
# run_all.py is the “one-button script”:


from __future__ import annotations

import argparse
import os


# Obj1Params / Obj2Params / Obj3Params

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


# When the test runs, it executes:
# python -m Experiment.run_all --out /tmp/... --epochs 2 --steps_per_epoch 2 ...
# parse_args is where those values get read.
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

    print("\nDONE.")
    print(f"Logs:    {log_dir}")
    print(f"History: {hist_dir}")
    print(f"Figures: {fig_dir}")
    print(f"Ckpts:   {ckpt_dir}")


if __name__ == "__main__":
    main()

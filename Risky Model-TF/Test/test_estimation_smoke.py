# Test/test_estimation_smoke.py

"""Smoke tests for the new estimation/ package.

Goal: ensure SMM + GMM entrypoints run end-to-end (tiny settings) and
produce the expected output dict structure.

I deliberately keep everything small so `pytest` is faster.
"""

import os
import tempfile

import numpy as np
import tensorflow as tf

from risky_debt.config import ModelParams, NetParams, TrainParams, Obj1Params
from risky_debt.networks import PolicyNet, PricingNet
from risky_debt.simulation import set_global_seed

from estimation.smm import estimate_smm, _InnerObjective1Solver
from estimation.gmm import estimate_gmm


def _tiny_tp() -> TrainParams:
    return TrainParams(
        seed=1,
        # for objective 1 rollouts
        T_train=5,
        N_paths_train=8,
        # training budget
        epochs=1,
        steps_per_epoch=1,
        batch_size=16,
    )


def test_smm_and_gmm_smoke():
    mp = ModelParams()
    tp = _tiny_tp()

    npol = NetParams(hidden_units=8, hidden_layers=1, activation="tanh")
    nq = NetParams(hidden_units=8, hidden_layers=1, activation="tanh")

    # create a small "true" policy/qnet via a tiny inner solve
    op1 = Obj1Params()
    inner = _InnerObjective1Solver(mp, npol, nq, tp, nu_zp=op1.nu_zp)
    set_global_seed(1)
    policy_true, qnet_true = inner.solve(mp, op1)

    est_bounds = {
        "theta": (0.10, 0.90),
        "rho": (0.50, 0.995),
        "sigma_eps": (0.005, 0.20),
        "psi0": (0.10, 10.0),
        "alpha": (0.01, 0.95),
    }

    with tempfile.TemporaryDirectory() as td:
        smm = estimate_smm(
            out_dir=td,
            mp_true=mp,
            npol=npol,
            nq=nq,
            tp_base=tp,
            policy_true=policy_true,
            qnet_true=qnet_true,
            est_bounds=est_bounds,
            max_evals=8,
            inner_epochs=1,
            inner_steps_per_epoch=1,
            sim_T=30,
            sim_burn=5,
            sim_n_paths=16,
            seed=111,
        )

        assert smm["method"] == "SMM"
        assert "theta_hat" in smm
        assert os.path.exists(os.path.join(td, "smm_results.json"))
        assert os.path.exists(os.path.join(td, "smm_synth_data.npz"))

        d = dict(np.load(os.path.join(td, "smm_synth_data.npz")))
        gmm = estimate_gmm(
            out_dir=td,
            mp_true=mp,
            npol=npol,
            nq=nq,
            tp_base=tp,
            data=d,
            est_bounds=est_bounds,
            max_evals=8,
            inner_epochs=1,
            inner_steps_per_epoch=1,
            seed=222,
        )

        assert gmm["method"] == "GMM"
        assert "theta_hat" in gmm
        assert os.path.exists(os.path.join(td, "gmm_results.json"))

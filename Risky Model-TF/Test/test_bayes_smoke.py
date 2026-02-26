# Test/test_bayes_smoke.py

"""Minimal Bayesian pipeline smoke test.

This test is intentionally tiny:
  * builds a short synthetic dataset by using the existing SMM forward simulator
  * runs the Bayesian estimator with very small budgets
  * checks that outputs are saved

"""

import os
import tempfile

import numpy as np

from risky_debt.config import ModelParams, NetParams, TrainParams, Obj1Params
from risky_debt.simulation import set_global_seed

from estimation.smm import _InnerObjective1Solver, forward_simulate_dataset
from estimation.bayes import estimate_hmc


def _tiny_tp() -> TrainParams:
    return TrainParams(
        seed=2,
        # for objective 1 rollouts
        T_train=5,
        N_paths_train=8,
        # training budget
        epochs=1,
        steps_per_epoch=1,
        batch_size=16,
    )


def test_bayes_hmc_smoke():
    mp = ModelParams()
    tp = _tiny_tp()
    npol = NetParams(hidden_units=8, hidden_layers=1, activation="tanh")
    nq = NetParams(hidden_units=8, hidden_layers=1, activation="tanh")
    op1 = Obj1Params()

    # tiny inner solve to get a policy/qnet to generate data
    inner = _InnerObjective1Solver(mp, npol, nq, tp, nu_zp=op1.nu_zp)
    set_global_seed(2)
    policy_true, qnet_true = inner.solve(mp, op1)

    rng = np.random.default_rng(999)
    eps = rng.normal(0.0, mp.sigma_eps, size=(16, 40 + 1)).astype(np.float32)
    data = forward_simulate_dataset(
        policy=policy_true,
        qnet=qnet_true,
        mp=mp,
        tp=tp,
        eps=eps,
        T=40,
        burn_in=5,
    )

    with tempfile.TemporaryDirectory() as td:
        res = estimate_hmc(
            out_dir=td,
            mp_true=mp,
            data=data,
            kernel="rwm",
            num_results=20,
            num_burnin=20,
            num_chains=1,
            step_size=0.05,
            num_particles=64,
            obs_sigma_lnz=0.05,
            seed=123,
        )

        assert res["method"] == "HMC"
        assert os.path.exists(os.path.join(td, "hmc_results.json"))
        assert os.path.exists(os.path.join(td, "hmc_draws.npz"))

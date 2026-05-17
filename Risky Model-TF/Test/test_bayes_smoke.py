"""Minimal Bayesian pipeline smoke test.

Marked slow because it exercises a real posterior-sampling code path, even though
all settings are aggressively downscaled for smoke testing.
"""

import os
import tempfile

import numpy as np
import pytest
import tensorflow as tf

from risky_debt.config import ModelParams, NetParams, TrainParams, Obj1Params
from risky_debt.networks import PolicyNet, PricingNet
from risky_debt.simulation import set_global_seed

from estimation.smm import _InnerObjective1Solver, forward_simulate_dataset
from estimation.bayes import estimate_hmc


def _tiny_tp() -> TrainParams:
    """Build a tiny training-parameter configuration for smoke tests."""
    return TrainParams(
        seed=2,
        T_train=4,
        N_paths_train=4,
        epochs=1,
        steps_per_epoch=1,
        batch_size=16,
    )


def _quick_inner_solve(self, mp, op1):
    """Run a tiny inner solve used by the Bayesian smoke test."""
    set_global_seed(self.seed)
    policy = PolicyNet(self.npol, mp.k_min, mp.b_min, mp.b_max)
    qnet = PricingNet(self.nq, mp.q_min, mp.q_max)
    _ = policy(tf.zeros((1, 3), tf.float32))
    _ = qnet(tf.zeros((1, 3), tf.float32))
    return policy, qnet


@pytest.mark.slow
def test_bayes_hmc_smoke(monkeypatch):
    """Run a tiny Bayesian smoke test and verify the expected artifacts."""
    monkeypatch.setattr(_InnerObjective1Solver, "solve", _quick_inner_solve)

    mp = ModelParams()
    tp = _tiny_tp()
    npol = NetParams(hidden_units=8, hidden_layers=1, activation="tanh")
    nq = NetParams(hidden_units=8, hidden_layers=1, activation="tanh")
    op1 = Obj1Params()

    inner = _InnerObjective1Solver(mp, npol, nq, tp, seed=2)
    set_global_seed(2)
    policy_true, qnet_true = inner.solve(mp, op1)

    rng = np.random.default_rng(999)
    eps = rng.normal(0.0, mp.sigma_eps, size=(1, 6 + 1)).astype(np.float32)
    data = forward_simulate_dataset(
        policy=policy_true,
        qnet=qnet_true,
        mp=mp,
        tp=tp,
        eps=eps,
        T=6,
        burn_in=1,
        continuation_horizon=1,
    )

    with tempfile.TemporaryDirectory() as td:
        res = estimate_hmc(
            out_dir=td,
            mp_true=mp,
            data=data,
            kernel="rwm",
            num_results=4,
            num_burnin=4,
            num_chains=1,
            step_size=0.05,
            num_particles=8,
            obs_sigma_lnz=0.05,
            seed=123,
        )

        assert res["method"] == "BayesianMCMC"
        assert res["sampler"] == "RandomWalkMetropolis"
        assert os.path.exists(os.path.join(td, "bayes_results.json"))
        assert os.path.exists(os.path.join(td, "bayes_draws.npz"))
        assert os.path.exists(os.path.join(td, "bayes_posterior_summary.csv"))
        assert os.path.exists(os.path.join(td, "bayes_artifacts.json"))
        assert os.path.exists(os.path.join(td, "figures", "bayes_trace_theta.png"))
        assert os.path.exists(os.path.join(td, "figures", "bayes_posterior_alpha.png"))
        assert os.path.exists(os.path.join(td, "figures", "bayes_target_log_prob.png"))

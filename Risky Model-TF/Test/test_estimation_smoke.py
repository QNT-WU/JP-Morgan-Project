"""Smoke tests for the estimation package.

These tests intentionally use very small settings so ordinary ``pytest`` runs stay
lightweight while still exercising the estimation code paths.
"""

import os
import tempfile

import numpy as np
import pytest
import tensorflow as tf

from risky_debt.config import ModelParams, NetParams, TrainParams, Obj1Params
from risky_debt.networks import PolicyNet, PricingNet
from risky_debt.simulation import set_global_seed

from estimation.smm import estimate_smm, _InnerObjective1Solver
from estimation.gmm import estimate_gmm
import estimation.smm as smm_mod
import estimation.gmm as gmm_mod


def _tiny_tp() -> TrainParams:
    """Build a tiny training-parameter configuration for smoke tests."""
    return TrainParams(
        seed=1,
        T_train=4,
        N_paths_train=4,
        epochs=1,
        steps_per_epoch=1,
        batch_size=16,
    )




def _fast_nelder_mead(f, x0, step, bounds, max_evals=60, tol=1e-6):
    """Single-evaluation optimizer stub for smoke speed."""
    x0 = np.asarray(x0, dtype=float)
    return {
        "x": x0,
        "objective": float(f(x0)),
        "evals": 1,
        "converged": True,
        "simplex_diameter": 0.0,
        "f_spread": 0.0,
    }


def _true_param_start(bounds, n_starts, seed, include_midpoint=True):
    """Deterministic stable start for smoke tests."""
    return [np.asarray([0.33, 1.0, 0.35], dtype=float)]



def _tiny_identification_report(*args, **kwargs):
    """Return a minimal identification report fixture for estimation tests."""
    return {"parameter_names": ["theta", "psi0", "alpha"], "true_theta": {}, "sweeps": {}}



def _quick_gmm_evaluate(self, x):
    """Return a deterministic finite GMM evaluation for smoke stability.

    The production GMM path uses gradient-based continuation calculations that are
    too brittle when the smoke test also monkeypatches the inner solver to return
    untrained random networks. For smoke purposes we only need to exercise the
    estimator orchestration and file-writing path with finite objects.
    """
    key = self._key(x)
    if key in self.cache:
        return self.cache[key]

    mp_cand = gmm_mod.apply_params(self.mp_true, self.param_names, x)
    policy_c, qnet_c = self.inner.solve(mp_cand, op1=self._obj1_params())

    n_obs = self.dataset.n_obs
    L = self.dataset.L
    x_arr = np.asarray(x, dtype=float)
    x_true = np.asarray([float(getattr(self.mp_true, n)) for n in self.param_names], dtype=float)
    # Small finite dependence on x keeps objective/jacobian well-defined in smoke mode.
    level = float(np.sum(x_arr - x_true)) * 1.0e-4
    m_series = np.full((n_obs, 3 * L), level, dtype=float)
    g = np.mean(m_series, axis=0)
    out = {"mp": mp_cand, "policy": policy_c, "qnet": qnet_c, "m_series": m_series, "g": g}
    self.cache[key] = out
    return out

def _quick_inner_solve(self, mp, op1):
    """Return initialized networks without gradient training for smoke speed."""
    set_global_seed(self.seed)
    policy = PolicyNet(self.npol, mp.k_min, mp.b_min, mp.b_max)
    qnet = PricingNet(self.nq, mp.q_min, mp.q_max)
    _ = policy(tf.zeros((1, 3), tf.float32))
    _ = qnet(tf.zeros((1, 3), tf.float32))
    return policy, qnet


@pytest.mark.slow
def test_smm_and_gmm_smoke(monkeypatch):
    """Run tiny SMM and GMM smoke tests and verify key output files."""
    monkeypatch.setattr(_InnerObjective1Solver, "solve", _quick_inner_solve)
    monkeypatch.setattr(smm_mod, "_nelder_mead", _fast_nelder_mead)
    monkeypatch.setattr(gmm_mod, "_nelder_mead", _fast_nelder_mead)
    monkeypatch.setattr(smm_mod, "generate_parameter_starts", _true_param_start)
    monkeypatch.setattr(gmm_mod, "generate_parameter_starts", _true_param_start)
    monkeypatch.setattr(smm_mod, "_build_smm_identification_report", _tiny_identification_report)
    monkeypatch.setattr(gmm_mod, "_build_gmm_identification_report", _tiny_identification_report)
    monkeypatch.setattr(gmm_mod._GMMEvaluator, "evaluate", _quick_gmm_evaluate)

    mp = ModelParams()
    tp = _tiny_tp()
    npol = NetParams(hidden_units=8, hidden_layers=1, activation="tanh")
    nq = NetParams(hidden_units=8, hidden_layers=1, activation="tanh")

    op1 = Obj1Params()
    inner = _InnerObjective1Solver(mp, npol, nq, tp, seed=1)
    set_global_seed(1)
    policy_true, qnet_true = inner.solve(mp, op1)

    est_bounds = {
        "theta": (0.10, 0.90),
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
            max_evals=1,
            inner_epochs=1,
            inner_steps_per_epoch=1,
            sim_T=4,
            sim_burn=1,
            sim_n_paths=1,
            seed=111,
            n_starts=1,
            continuation_horizon=1,
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
            max_evals=1,
            inner_epochs=1,
            inner_steps_per_epoch=1,
            seed=222,
            n_starts=1,
            continuation_horizon=1,
        )

        assert gmm["method"] == "GMM"
        assert "theta_hat" in gmm
        assert os.path.exists(os.path.join(td, "gmm_results.json"))

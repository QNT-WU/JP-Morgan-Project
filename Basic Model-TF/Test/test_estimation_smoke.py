"""Smoke test for the estimation pipeline."""

import numpy as np


def test_corrected_estimation_pipeline_smoke():
    """Smoke-test the corrected estimation workflow."""
    from basic_mailer.config import ModelParams, NetParams, TrainParams
    from basic_mailer.networks import PolicyNet
    from basic_mailer.estimation import (
        TwoStepGMMEstimator,
        TwoStepSMMEstimator,
        build_default_moment_spec,
        make_crn_design,
        simulate_paths_crn,
        structural_params_from_model,
        transform_params_to_tilde,
    )

    mp = ModelParams()
    npol = NetParams(hidden_units=8, hidden_layers=1, activation="tanh")
    tp_inner = TrainParams(epochs=1, steps_per_epoch=1, batch_size=32)

    truth_policy = PolicyNet(npol, mp.k_min, mp.k_max)
    _ = truth_policy(np.zeros((1, 2), dtype=np.float32))

    obs_design = make_crn_design(n_paths=4, T=12, seed=0)
    ds_obs = simulate_paths_crn(policy=truth_policy, mp=mp, design=obs_design, burn_in=2)

    sim_design = make_crn_design(n_paths=4, T=10, seed=1)
    x0 = transform_params_to_tilde(**structural_params_from_model(mp))

    gmm = TwoStepGMMEstimator(mp_template=mp, observed_dataset=ds_obs)
    gmm_results = gmm.fit(x0=x0, max_evals=5, weight_methods=("standard",), n_starts=1)
    assert "GMM_A" in gmm_results
    assert np.isfinite(gmm_results["GMM_A"].stage2.best_loss)

    smm = TwoStepSMMEstimator(
        mp_template=mp,
        npol=npol,
        inner_tp=tp_inner,
        observed_dataset=ds_obs,
        simulation_design=sim_design,
        moment_spec=build_default_moment_spec(),
    )
    smm_results = smm.fit(x0=x0, max_evals=5, weight_methods=("standard",), n_starts=1)
    assert "SMM_A" in smm_results
    assert np.isfinite(smm_results["SMM_A"].stage2.best_loss)

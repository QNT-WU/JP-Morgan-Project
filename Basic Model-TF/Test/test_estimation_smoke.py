import numpy as np


def test_moments_and_estimators_smoke():
    """Tiny smoke test so pytest stays fast."""
    from basic_mailer.config import ModelParams, NetParams, TrainParams
    from basic_mailer.networks import PolicyNet
    from basic_mailer.estimation.moments import (
        make_crn_design,
        simulate_paths_crn,
        compute_moments,
        build_default_moment_spec,
        make_identity_weight_matrix,
    )
    from basic_mailer.estimation.smm import SMMEstimator
    from basic_mailer.estimation.gmm import GMMEstimator

    mp = ModelParams()
    npol = NetParams(hidden_units=8, hidden_layers=1, activation="tanh")
    tp_inner = TrainParams(epochs=1, steps_per_epoch=1, batch_size=32)

    # Build a trivial policy net
    policy = PolicyNet(npol, mp.k_min, mp.k_max)
    _ = policy(np.zeros((1, 2), dtype=np.float32))

    design = make_crn_design(n_paths=4, T=10, seed=0)
    ds = simulate_paths_crn(policy=policy, mp=mp, design=design, burn_in=0)

    spec = build_default_moment_spec()
    m = compute_moments(ds, mp, spec)
    assert set(m.keys()) == set(spec.names)

    W = make_identity_weight_matrix(len(spec.names))

    smm = SMMEstimator(
        mp_template=mp,
        npol=npol,
        inner_tp=tp_inner,
        moment_spec=spec,
        W=W,
        crn_design=design,
        target_moments=m,
    )
    q_smm = smm.evaluate(np.zeros(4))
    assert np.isfinite(q_smm)

    gmm = GMMEstimator(
        mp_template=mp,
        npol=npol,
        inner_tp=tp_inner,
        crn_design=design,
        n_states=16,
        n_shocks=4,
        seed=0,
    )
    q_gmm = gmm.evaluate(np.zeros(4))
    assert np.isfinite(q_gmm)

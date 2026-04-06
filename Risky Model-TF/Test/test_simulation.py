"""Unit tests for simulation helpers."""

# Test/test_simulation.py
# simulate_ergodic_dataset(...) generates policy-induced ergodic sample of states (k_t, b_t, z_t)
import numpy as np

from risky_debt.config import ModelParams, NetParams, TrainParams
from risky_debt.networks import PolicyNet
from risky_debt.simulation import simulate_ergodic_dataset


def test_simulate_ergodic_dataset_shapes_and_basic_sanity():
    """Verify simulate ergodic dataset shapes and basic sanity."""
    mp = ModelParams()
    npol = NetParams(hidden_units=8, hidden_layers=1, activation="tanh")

    # Keep simulation tiny for speed
    tp = TrainParams(
        seed=123,
        ergodic_burn_in=50,
        ergodic_T=200,
        ergodic_n_paths=4,
        ergodic_buffer_size=10_000,
        epochs=1,
        steps_per_epoch=1,
        batch_size=16,
    )

    policy = PolicyNet(npol, mp.k_min, mp.b_min, mp.b_max)

    k, b, z = simulate_ergodic_dataset(policy, mp, tp, seed=999)

    assert isinstance(k, np.ndarray)
    assert isinstance(b, np.ndarray)
    assert isinstance(z, np.ndarray)

    assert k.shape == b.shape == z.shape
    assert k.ndim == 1
    assert len(k) > 0

    # basic feasibility
    assert (k > 0).all()
    assert np.isfinite(b).all()
    assert (z > 0).all()

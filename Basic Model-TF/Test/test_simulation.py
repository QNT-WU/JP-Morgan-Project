# Test/test_simulation.py
# simulate_ergodic_dataset(...) is the function that generates the policy-induced ergodic sample of states (𝑘𝑡,𝑧𝑡)
import numpy as np

from basic_mailer.config import ModelParams, NetParams, TrainParams
from basic_mailer.networks import PolicyNet
from basic_mailer.simulation import simulate_ergodic_dataset


# Name tells you exactly what it tests:shapes,positivity
def test_simulate_ergodic_dataset_shapes_and_positive():
    mp = ModelParams()
    npol = NetParams(hidden_units=8, hidden_layers=1, activation="tanh")
    # This is important: this test intentionally shrinks ergodic simulation
    tp = TrainParams(
        seed=123,
        # keep simulation small for unit test speed
        ergodic_burn_in=50,
        ergodic_T=200,
        ergodic_n_paths=4,
        ergodic_buffer_size=10_000,
        epochs=1,
        steps_per_epoch=1,
        batch_size=16,
    )

    policy = PolicyNet(npol, mp.k_min, mp.k_max)
    k, z = simulate_ergodic_dataset(policy, mp, tp, seed=999)

    # his verifies that simulate_ergodic_dataset returns NumPy arrays (not TF tensors)
    assert isinstance(k, np.ndarray)
    assert isinstance(z, np.ndarray)
    # Shapes match
    # Every recorded k_t must correspond to the same-time z_t
    assert k.shape == z.shape
    # This enforces a design decision: ergodic sample is stored as a flat buffer rather than a matrix
    # Internally the simulator produces something like:
    # k_mat: shape (n_paths, T_kept+1)
    # z_mat: shape (n_paths, T_kept+1)
    # Then it flattens them to:
    # k_flat: shape (n_paths*(T_kept+1),)
    # z_flat: same
    assert k.ndim == 1
    # Non-empty dataset
    assert len(k) > 0
    # Positivity constraints
    assert (k > 0).all()
    assert (z > 0).all()

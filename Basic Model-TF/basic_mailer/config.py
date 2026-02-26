from __future__ import annotations

from dataclasses import dataclass


# prevents accidental mutation during training (very common bug source)
# makes runs reproducible and easier to debug
# configs become “constants” for a run
@dataclass(frozen=True)  # frozen=True means: immutable after construction.
class ModelParams:
    # shock: ln z' = rho ln z + eps
    rho: float = 0.9
    # AR(1) persistence in log productivity: 0.9 means shocks are presistent
    sigma_eps: float = 0.02
    # Standard deviation of the AR(1) innovation
    # Small value = mild volatility in productivity.

    # production: pi(k,z) = z k^theta
    # 0 < theta < 1 gives diminishing returns to capital
    theta: float = 0.33

    # adjustment cost & depreciation
    # Larger psi0 penalizes investment changes more strongly.
    # delta is the Depreciation rate. (1-delta)k is what remains after depreciation.
    psi0: float = 1.0
    delta: float = 0.06

    # discount: Risk-free interest rate used to compute discount factor:
    r: float = 0.04  # beta = 1/(1+r)

    # numerical safety
    # Numerical safety: enforce k > 0.
    # If k is near 0, I will get exploding values or NaNs.
    # k_min: float = 1e-6
    k_min: float = 0.05
    k_max: float = 8.0


@dataclass(frozen=True)
# This holds neural network architecture choices.
class NetParams:
    hidden_units: int = 64
    hidden_layers: int = 2
    activation: str = "tanh"  # smooth recommended
    # tanh is smooth → helps stability for Euler/Bellman residual training (gradients depend on derivatives).


@dataclass(frozen=True)
# This holds training/evaluation configuration, i.e. “experiment settings”.
# RNG: Random Number Generator
class TrainParams:
    seed: int = 123
    # Random seed for reproducibility.
    # Should seed: TensorFlow RNG, NumPy RNG, simulation RNG streams

    epochs: int = 40
    # Total epochs.
    steps_per_epoch: int = 50
    # Each epoch has 50 gradient updates.
    # Total updates = epochs * steps_per_epoch
    batch_size: int = 512
    # How many training states per gradient step (for Obj2/Obj3).
    # For Obj1, you also use N_paths_train (see below).

    lr_policy: float = 3e-4
    # Learning rate for policy network.
    lr_value: float = 3e-4
    # Learning rate for value network (Obj3 only).
    grad_clip: float = 5.0
    # Global norm gradient clipping threshold.
    # Prevents gradient explosion: grads, _ = tf.clip_by_global_norm(grads, tp.grad_clip)

    # Obj1 rollout horizon
    T_train: int = 100
    # Horizon length used for objective-1 training rollouts.
    # Larger = closer to infinite horizon but noisier/unstable gradients.
    N_paths_train: int = 128
    # Number of Monte Carlo paths in Obj1 training objective.
    # More paths reduces Monte Carlo noise, but slower.

    # evaluation rollout horizon
    T_test: int = 300
    # Longer horizon for evaluation to approximate “true welfare”.
    N_paths_test: int = 128
    # How many test rollouts to average over.

    # Euler-test
    N_test_states: int = 512
    # Number of test states (k,z) for Euler diagnostics.
    N_eps_test: int = 64
    # For each test state, draw 64 shocks to approximate conditional expectation.
    # Larger = less noisy Euler MSE, but slower.

    # ergodic data refresh
    # These are for generating the “ergodic dataset” (states visited under current policy).
    ergodic_refresh_every: int = 5
    # Regenerate ergodic dataset every 5 epochs because policy changes over training.
    ergodic_burn_in: int = 2000
    # Number of initial simulated steps to discard so the Markov chain reaches its stationary region.
    ergodic_T: int = 10000
    # Number of steps to keep after burn-in per path.
    ergodic_n_paths: int = 32
    # Number of independent simulation paths.
    ergodic_buffer_size: int = 200000
    # Maximum size of stored ergodic states buffer.
    # If simulation produces more than this, you downsample or truncate.
    # The ergodic distribution sampling controls are not model parameters,not network parameters,and not optimization parameters.
    # They are data-generation controls.
    # They only affect how you sample states (k,z), not:the economic model,the neural networks,the objective functions themselves


@dataclass(frozen=True)
class Obj3Params:
    nu: float = 1.0
    # Weight for Euler residual term relative to Bellman residual in objective 3:
    # Keeping it separate is clean: it’s not a global “training param”, it’s objective-specific.

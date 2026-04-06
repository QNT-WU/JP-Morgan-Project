"""Configuration objects for the basic Mailer package."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelParams:
    """Structural and numerical parameters for the basic model.

    Attributes:
        rho: Persistence of the log productivity shock.
        sigma_eps: Standard deviation of the shock innovation.
        theta: Capital share in production.
        psi0: Scale parameter for convex adjustment costs.
        delta: Depreciation rate of capital.
        r: Risk-free interest rate used to define the discount factor.
        k_min: Lower numerical bound for capital.
        k_max: Upper numerical bound for capital.
    """

    rho: float = 0.9
    sigma_eps: float = 0.02
    theta: float = 0.33
    psi0: float = 1.0
    delta: float = 0.06
    r: float = 0.04
    k_min: float = 0.05
    k_max: float = 8.0


@dataclass(frozen=True)
class NetParams:
    """Neural-network architecture choices."""

    hidden_units: int = 64
    hidden_layers: int = 2
    activation: str = "tanh"


@dataclass(frozen=True)
class TrainParams:
    """Training, evaluation, and simulation controls."""

    seed: int = 123
    epochs: int = 40
    steps_per_epoch: int = 50
    batch_size: int = 512
    lr_policy: float = 3e-4
    lr_value: float = 3e-4
    grad_clip: float = 5.0
    T_train: int = 100
    N_paths_train: int = 128
    T_test: int = 300
    N_paths_test: int = 128
    N_test_states: int = 512
    N_eps_test: int = 64
    ergodic_refresh_every: int = 5
    ergodic_burn_in: int = 2000
    ergodic_T: int = 10000
    ergodic_n_paths: int = 32
    ergodic_buffer_size: int = 200000


@dataclass(frozen=True)
class Obj3Params:
    """Objective-3-specific hyperparameters."""

    nu: float = 1.0

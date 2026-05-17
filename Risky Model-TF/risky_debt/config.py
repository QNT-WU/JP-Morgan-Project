"""Configuration dataclasses for the risky-debt model package."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelParams:
    """Structural and numerical parameters of the risky-debt model.

    The fields combine economic primitives such as the productivity process,
    adjustment costs, and recovery rate with numerical guardrails used by the
    TensorFlow implementation.
    """

    rho: float = 0.9
    sigma_eps: float = 0.02
    theta: float = 0.33
    tau: float = 0.2
    psi0: float = 1.0
    delta: float = 0.06
    r: float = 0.04
    r_c: float = 0.04
    phi_borrow: float = 0.0
    alpha: float = 0.35
    eta0: float = 0.02
    eta1: float = 0.02
    k_min: float = 1e-6
    z_min: float = 1e-12
    b_min: float = -2.0
    b_max: float = 10.0
    q_min: float = 1e-4
    q_max: float = 0.999


@dataclass(frozen=True)
class NetParams:
    """Hyperparameters for one feed-forward neural network module."""

    hidden_units: int = 128
    hidden_layers: int = 3
    activation: str = "tanh"


@dataclass(frozen=True)
class TrainParams:
    """Training hyperparameters shared across neural objectives."""

    seed: int = 123
    epochs: int = 40
    steps_per_epoch: int = 20
    batch_size: int = 256
    lr_policy: float = 3e-4
    lr_value: float = 3e-4
    lr_vtilde: float = 3e-4
    lr_q: float = 3e-4
    grad_clip: float = 5.0
    T_train: int = 80
    N_paths_train: int = 128
    T_test: int = 250
    N_paths_test: int = 128
    N_test_states: int = 512
    N_eps_test: int = 64
    ergodic_refresh_every: int = 5
    ergodic_burn_in: int = 300
    ergodic_T: int = 1500
    ergodic_n_paths: int = 12
    ergodic_buffer_size: int = 30000
    k0_low: float = 0.5
    k0_high: float = 2.0
    b0_low: float = -0.5
    b0_high: float = 0.5
    z0_low: float = 0.5
    z0_high: float = 2.0
    kappa_issue: float = 0.02
    kappa_solv: float = 0.25
    kappa_b: float = 0.02
    eps_b: float = 1e-6
    eps_den: float = 1e-6
    eps_q: float = 1e-5
    omega_q: float = 1.0
    N_q: int = 8


@dataclass(frozen=True)
class Obj1Params:
    """Penalty weights used by Objective 1."""

    nu_zp: float = 1.0
    nu_critic: float = 1.0


@dataclass(frozen=True)
class Obj2Params:
    """Penalty weights used by Objective 2."""

    nu_def: float = 1.0
    nu_bell: float = 1.0
    nu_foc: float = 1.0
    nu_zp: float = 1.0


@dataclass(frozen=True)
class Obj3Params:
    """Penalty weights used by Objective 3."""

    nu_def: float = 1.0
    nu_bell: float = 1.0
    nu_foc: float = 1.0
    nu_zp: float = 1.0

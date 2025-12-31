from __future__ import annotations
from dataclasses import dataclass


# makes it immutable after creation. So you can’t accidentally do mp.r = 0.2 in the middle of training.
@dataclass(frozen=True)
class ModelParams:
    # shock process: ln z' = rho ln z + eps
    # lnz′=ρlnz+ε′,ε′∼N(0,σ_ε^2​)
    rho: float = 0.9
    sigma_eps: float = 0.02

    # production: pi(k,z) = z k^theta
    # π(k,z)=zkθ;after-tax operating profit: (1−𝜏)𝜋(𝑘,𝑧)
    theta: float = 0.33
    tau: float = 0.2  # corporate tax rate in (0,1)

    # adjustment cost & depreciation
    # I=k′−(1−δ)k, ψ(I,k)=ψ_0​ I^2​/2k
    psi0: float = 1.0
    delta: float = 0.06

    # risk-free rate and discounting
    r: float = 0.04  # beta = 1/(1+r)

    # default recovery haircut
    # R(k′,z′)=(1−α)((1−τ)π(k′,z′)+(1−δ)k′
    alpha: float = 0.35  # in (0,1)

    # equity issuance cost: eta(e) = (eta0 + eta1*e) 1_{e<0}
    # η(e)=(η0​+η1​e)1{e<0}​
    eta0: float = 0.02
    eta1: float = 0.02

    # numerical safety, Numerical safety floors
    # Ensures you never take log(z) with 𝑧≤0
    # Ensures 𝑘 never hits zero (division by 𝑘 appears in 𝜓(𝐼,𝑘))
    k_min: float = 1e-6
    z_min: float = 1e-12

    # policy bounds for b'
    # The model allows 𝑏∈𝑅.
    # The code chooses to cap the network output for stability(avoid huge debt exploding 𝑏′/𝑞)
    b_min: float = -2.0
    b_max: float = 4.0

    # pricing bounds
    # The model requires 𝑞∈(0,1)
    # 𝑞→0 making repayment 𝑏′/𝑞 blow up
    # q→1 sometimes creates flat gradients/weird corners depending on parameterization
    q_min: float = 1e-4
    q_max: float = 0.999


# each network (policy/value/pricing) will typically be an MLP with
# 3 hidden layers
# 128 units each
# tanh activation
@dataclass(frozen=True)
class NetParams:
    hidden_units: int = 128
    hidden_layers: int = 3
    activation: str = "tanh"


# Training loop shape
# “epoch” = one outer loop iteration
# each epoch has 50 gradient steps
# each step uses 512 states (for objectives 2/3 typically from ergodic buffer)
@dataclass(frozen=True)
class TrainParams:
    seed: int = 123

    epochs: int = 40
    steps_per_epoch: int = 50
    batch_size: int = 512

    # Learning rates per network
    # Objective 1 trains policy + q (and maybe only these)
    # Objective 2 trains policy + value + vtilde + q
    # Objective 3 trains policy + value + q
    lr_policy: float = 3e-4
    lr_value: float = 3e-4
    lr_vtilde: float = 3e-4
    lr_q: float = 3e-4

    # Gradient clipping
    grad_clip: float = 5.0

    # Obj1 rollout horizon
    # Objective 1 uses rollouts to estimate lifetime reward
    # Evaluation uses longer rollouts for more stable “TestReward”
    T_train: int = 80
    N_paths_train: int = 128

    # evaluation rollout horizon
    T_test: int = 250
    N_paths_test: int = 128

    # N_test_states: how many test states from ergodic buffer
    # N_eps_test: how many shock draws (or pairs) for Euler diagnostic estimation
    N_test_states: int = 512
    N_eps_test: int = 64

    # ergodic sampling controls
    # simulate the policy-induced Markov chain for long horizon ergodic_T
    # discard first ergodic_burn_in
    # run ergodic_n_paths independent paths
    # store up to ergodic_buffer_size states in a replay buffer
    # refresh every ergodic_refresh_every epochs because the policy changes
    ergodic_refresh_every: int = 5
    ergodic_burn_in: int = 2000
    ergodic_T: int = 10000
    ergodic_n_paths: int = 32
    ergodic_buffer_size: int = 200000

    # initial state ranges (broad support)
    # This defines 𝜇0​, the initial distribution for simulation/rollouts
    k0_low: float = 0.5
    k0_high: float = 2.0
    b0_low: float = -0.5
    b0_high: float = 0.5
    z0_low: float = 0.5
    z0_high: float = 2.0

    # Smoothness knobs(important for your model’s kinks)
    # issuance indicator 1{𝑒<0}:replaced by a smooth approximation(sigmoid-like)controlled by kappa_issue
    # continuation/default gating 𝑠′=𝜎(𝑉~/𝜅𝑠): kappa_solv is that 𝜅𝑠​​
    kappa_issue: float = 0.02  # smoothing for 1_{e<0}
    kappa_solv: float = 0.05  # smoothing for solvency gating


# Objective-weight parameter blocks
@dataclass(frozen=True)
class Obj1Params:
    # Train objective: maximize lifetime reward
    # plus pricing discipline via ZP residual
    nu_zp: float = 1.0


# Matches your residual blocks:
# default/complementarity
# Bellman consistency for 𝑉~
# FOC/Euler residuals
# Zzro profit pricing residual
@dataclass(frozen=True)
class Obj2Params:
    nu_def: float = 1.0
    nu_bell: float = 1.0
    nu_foc: float = 1.0
    nu_zp: float = 1.0


# Objective 3 focuses on:
# default/Bellman residual block
# Zzeo profit pricing residual block
@dataclass(frozen=True)
class Obj3Params:
    nu_def: float = 1.0
    nu_zp: float = 1.0

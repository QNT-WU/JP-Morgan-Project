"""basic_mailer.estimation

Estimation layer (SMM + GMM) that reuses your existing pipeline:

- simulation.py (forward policy simulation / ergodic generators)
- objectives.py (Obj2 Euler AiO loss for inner solve)
- evaluation.py (Euler residual function for GMM)

Folder layout (added):
    basic_mailer/estimation/
        moments.py
        smm.py
        gmm.py
"""

from .moments import (
    CRNDesign,
    PathDataset,
    MomentSpec,
    build_default_moment_spec,
    compute_moments,
    make_crn_design,
    make_identity_weight_matrix,
    simulate_paths_crn,
)
from .smm import SMMEstimator
from .gmm import GMMEstimator

__all__ = [
    "CRNDesign",
    "PathDataset",
    "MomentSpec",
    "build_default_moment_spec",
    "compute_moments",
    "make_crn_design",
    "make_identity_weight_matrix",
    "simulate_paths_crn",
    "SMMEstimator",
    "GMMEstimator",
]

"""Shared parameter transformations for the estimation workflow."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict

import numpy as np

from ..config import ModelParams
from ..primitives import beta_from_r

PARAMETER_NAMES = ("beta", "theta", "psi0")


def _sigmoid(x):
    """Map a real number to ``(0, 1)`` for constrained optimization."""
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x):
    """Map a real number to a strictly positive value."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _inv_sigmoid(p):
    """Apply the inverse-logit transform used for bounded parameters."""
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _inv_softplus(y):
    """Apply the inverse softplus transform used for positive parameters."""
    y = np.maximum(y, 1e-10)
    return np.log(np.expm1(y))


def transform_tilde_to_params(theta_tilde: np.ndarray) -> Dict[str, float]:
    """Transform unconstrained optimizer coordinates to structural parameters."""
    x = np.asarray(theta_tilde, dtype=np.float64).reshape(-1)
    if x.shape != (3,):
        raise ValueError("theta_tilde must be length 3")
    return {
        "beta": float(_sigmoid(x[0])),
        "theta": float(_sigmoid(x[1])),
        "psi0": float(_softplus(x[2])),
    }


def transform_params_to_tilde(beta: float, theta: float, psi0: float) -> np.ndarray:
    """Transform structural parameters to unconstrained optimizer coordinates."""
    return np.asarray([
        _inv_sigmoid(float(beta)),
        _inv_sigmoid(float(theta)),
        _inv_softplus(float(psi0)),
    ], dtype=np.float64)


def update_model_params(mp_template: ModelParams, params: Dict[str, float]) -> ModelParams:
    """Return a model-parameter object updated with structural estimates."""
    beta = float(params["beta"])
    r = (1.0 / beta) - 1.0
    return replace(mp_template, r=r, theta=float(params["theta"]), psi0=float(params["psi0"]))


def structural_params_from_model(mp: ModelParams) -> Dict[str, float]:
    """Extract the structural parameter block from a model-parameter object."""
    return {
        "beta": float(beta_from_r(mp.r)),
        "theta": float(mp.theta),
        "psi0": float(mp.psi0),
    }


def vector_from_params(params: Dict[str, float]) -> np.ndarray:
    """Convert named structural parameters to a dense vector."""
    return np.asarray([params[name] for name in PARAMETER_NAMES], dtype=np.float64)


def params_from_vector(x: np.ndarray) -> Dict[str, float]:
    """Convert a dense parameter vector to a named dictionary."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.shape != (3,):
        raise ValueError("parameter vector must have length 3")
    return {name: float(val) for name, val in zip(PARAMETER_NAMES, x)}

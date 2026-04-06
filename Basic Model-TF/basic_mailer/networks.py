"""Backward-compatible neural-network exports.

This module keeps the historical import path ``basic_mailer.networks`` while the
actual TensorFlow models now live under :mod:`basic_mailer.nn`.
"""

from __future__ import annotations

from .nn import (
    ActivationFactory,
    BoundedPolicyHead,
    MLPBlock,
    PolicyModel,
    PolicyNet,
    StateNormalization,
    ValueModel,
    ValueNet,
)


_get_activation = ActivationFactory.get

__all__ = [
    "_get_activation",
    "ActivationFactory",
    "BoundedPolicyHead",
    "MLPBlock",
    "PolicyModel",
    "PolicyNet",
    "StateNormalization",
    "ValueModel",
    "ValueNet",
]

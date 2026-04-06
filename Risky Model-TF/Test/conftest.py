"""Pytest configuration for the repository test suite.

The project depends on TensorFlow and TensorFlow Probability. In environments
where those libraries are not installed, the TensorFlow-backed tests are not
collected. A lightweight environment check remains so ``pytest`` exits cleanly
instead of failing during import collection.
"""

from __future__ import annotations

import importlib.util

_HAS_TF = importlib.util.find_spec("tensorflow") is not None
_HAS_TFP = importlib.util.find_spec("tensorflow_probability") is not None
_HAS_FULL_STACK = _HAS_TF and _HAS_TFP

collect_ignore_glob = []
if not _HAS_FULL_STACK:
    collect_ignore_glob = [
        "test_bayes_smoke.py",
        "test_estimation_smoke.py",
        "test_integration_run_all.py",
        "test_networks.py",
        "test_objectives_smoke.py",
        "test_primitives.py",
        "test_simulation.py",
    ]

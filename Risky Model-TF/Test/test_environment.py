"""Environment checks for the repository test suite."""

from __future__ import annotations

import importlib.util

import pytest


def test_tensorflow_test_dependencies_available() -> None:
    """Skip cleanly when TensorFlow-based test dependencies are unavailable."""
    missing = []
    if importlib.util.find_spec("tensorflow") is None:
        missing.append("tensorflow")
    if importlib.util.find_spec("tensorflow_probability") is None:
        missing.append("tensorflow_probability")
    if missing:
        pytest.skip("Missing optional test dependencies: " + ", ".join(missing))

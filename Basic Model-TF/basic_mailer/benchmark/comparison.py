"""Object-oriented wrappers around benchmark-comparison utilities.

This module exposes both a thin object-oriented façade and the lower-level
comparison helpers used by the experiment pipeline. It keeps the historical
comparison protocol available through stable package imports while centralizing
all benchmark-versus-neural-network logic under the :mod:`basic_mailer.benchmark`
namespace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from basic_mailer.grid_benchmark import policy_from_idx
from basic_mailer.grid_compare import (
    compare_on_state_sample,
    full_grid_state_sample,
    simulate_benchmark_ergodic_dataset,
)


@dataclass
class ComparisonOutput:
    """Structured result for an NN-versus-benchmark comparison.

    Attributes:
        details: Rich comparison object returned by the lower-level utility.
        summary: Flat metrics dictionary suitable for JSON logging.
    """

    details: Any
    summary: Dict[str, Any]


class BenchmarkComparator:
    """Evaluate neural-network solutions against a solved grid benchmark.

    The comparator stores the solved benchmark arrays once and then applies the
    standard comparison protocol on arbitrary state samples, such as the full
    benchmark grid, the benchmark ergodic region, or a neural-network ergodic
    region.
    """

    def __init__(
        self,
        *,
        precomputed,
        value_star,
        policy_idx_star,
        value_star_alt=None,
        policy_idx_star_alt=None,
        alt_label: str = "Howard PI",
    ) -> None:
        """Store solved benchmark objects for repeated comparisons.

        Args:
            precomputed: Shared benchmark arrays returned by
                :func:`basic_mailer.grid_benchmark.precompute_arrays`.
            value_star: Primary benchmark value function on the grid.
            policy_idx_star: Primary benchmark policy in action-index form.
            value_star_alt: Optional secondary benchmark value function.
            policy_idx_star_alt: Optional secondary benchmark policy in
                action-index form.
            alt_label: Human-readable label for the secondary benchmark.
        """
        self.precomputed = precomputed
        self.value_star = value_star
        self.policy_idx_star = policy_idx_star
        self.value_star_alt = value_star_alt
        self.policy_idx_star_alt = policy_idx_star_alt
        self.alt_label = alt_label

    @property
    def policy_star(self):
        """Return the primary benchmark policy in capital-level form."""
        return policy_from_idx(self.precomputed.k_grid, self.policy_idx_star)

    @property
    def policy_star_alt(self):
        """Return the secondary benchmark policy in capital-level form."""
        if self.policy_idx_star_alt is None:
            return None
        return policy_from_idx(self.precomputed.k_grid, self.policy_idx_star_alt)

    def compare(
        self,
        *,
        policy_nn,
        states_k,
        states_z,
        value_nn=None,
        out_dir: str,
        tag: str,
    ) -> ComparisonOutput:
        """Run the standard comparison protocol on a supplied state sample.

        Args:
            policy_nn: Neural-network policy object.
            states_k: Sampled capital states.
            states_z: Sampled productivity states.
            value_nn: Optional neural-network value object for Objective 3.
            out_dir: Output directory for plots.
            tag: Suffix used in artifact filenames.

        Returns:
            A :class:`ComparisonOutput` instance with detailed and summary
            comparison results.
        """
        details, summary = compare_on_state_sample(
            policy_nn=policy_nn,
            value_nn_or_none=value_nn,
            k_states=states_k,
            z_states=states_z,
            k_grid=self.precomputed.k_grid,
            z_grid=self.precomputed.z_grid,
            V_star=self.value_star,
            policy_star=self.policy_star,
            u=self.precomputed.u,
            Pz=self.precomputed.Pz,
            beta=self.precomputed.beta,
            out_dir=out_dir,
            tag=tag,
            V_star_alt=self.value_star_alt,
            policy_star_alt=self.policy_star_alt,
            alt_label=self.alt_label,
        )
        return ComparisonOutput(details=details, summary=summary)


__all__ = [
    "BenchmarkComparator",
    "ComparisonOutput",
    "compare_on_state_sample",
    "full_grid_state_sample",
    "simulate_benchmark_ergodic_dataset",
]

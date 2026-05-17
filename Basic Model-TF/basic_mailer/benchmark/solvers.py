"""Object-oriented wrappers around grid-based benchmark solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from basic_mailer.grid_benchmark import (
    GridSpec,
    policy_from_idx,
    precompute_arrays,
    solve_howard_pi,
    solve_vfi,
)


@dataclass
class SolverResult:
    """Container for a solved grid benchmark.

    Attributes:
        value: Benchmark value function on the full discretized grid.
        policy_idx: Benchmark policy in action-index form.
        policy_level: Benchmark policy converted to next-period capital levels.
        diagnostics: Solver metadata such as iterations and convergence status.
    """

    value: Any
    policy_idx: Any
    policy_level: Any
    diagnostics: Dict[str, Any]


class GridBenchmarkSolver:
    """High-level interface for VFI and Howard policy iteration."""

    def __init__(self, mp, grid_spec: GridSpec) -> None:
        """Precompute arrays shared by the benchmark solvers.

        Args:
            mp: Structural model parameters.
            grid_spec: Benchmark grid specification.
        """
        self.mp = mp
        self.grid_spec = grid_spec
        self.precomputed = precompute_arrays(mp, grid_spec)

    def solve(self, method: str = "vfi", **kwargs) -> SolverResult:
        """Solve the discretized model using the requested method.

        Args:
            method: Either ``"vfi"`` or one of ``{"pi", "howard", "howard_pi"}``.
            **kwargs: Keyword arguments forwarded to the underlying solver.

        Returns:
            A :class:`SolverResult` containing the solved benchmark objects.
        """
        method_key = method.lower().strip()
        if method_key == "vfi":
            value, policy_idx, diagnostics = solve_vfi(
                self.precomputed,
                return_info=True,
                **kwargs,
            )
        elif method_key in {"pi", "howard", "howard_pi"}:
            value, policy_idx, diagnostics = solve_howard_pi(
                self.precomputed,
                return_info=True,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown benchmark method '{method}'.")
        policy_level = policy_from_idx(self.precomputed.k_grid, policy_idx)
        return SolverResult(
            value=value,
            policy_idx=policy_idx,
            policy_level=policy_level,
            diagnostics=diagnostics,
        )


__all__ = ["GridBenchmarkSolver", "GridSpec", "SolverResult"]

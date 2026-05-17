"""Structured progress reporting for estimation routines.

The estimation layer can be slow because it combines synthetic-data generation,
inner model solves, and multi-start optimization. This module keeps runtime
tracking encapsulated in a small OOP helper so callers can enable informative
stdout progress without scattering ad hoc print statements across the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional
import time


@dataclass
class EstimationProgressConfig:
    """Configuration for estimation progress reporting.

    The defaults are intentionally notebook-friendly: they show stage boundaries,
    local-start boundaries, and only periodic evaluation updates so stdout does
    not get flooded during long optimization loops.
    """

    enabled: bool = True
    flush: bool = True
    include_timestamp: bool = False
    emit_eval_start: bool = False
    emit_eval_done: bool = True
    eval_done_every: int = 5
    emit_first_eval: bool = True


class EstimationProgressReporter:
    """Emit structured progress messages for long GMM/SMM routines.

    The reporter is intentionally lightweight: it keeps phase start times,
    formats metadata consistently, and prints only when explicitly enabled by
    the caller.
    """

    def __init__(
        self,
        method_name: str,
        config: Optional[EstimationProgressConfig] = None,
    ) -> None:
        """Initialize EstimationProgressReporter."""
        self.method_name = str(method_name).upper().strip()
        self.config = config or EstimationProgressConfig()
        self._phase_starts: Dict[str, float] = {}

    def _fmt_meta(self, meta: Mapping[str, object]) -> str:
        """Format metadata fields for progress-log output."""
        if not meta:
            return ""
        parts = []
        for key, value in meta.items():
            if isinstance(value, float):
                parts.append(f"{key}={value:.6g}")
            else:
                parts.append(f"{key}={value}")
        return " | " + ", ".join(parts)

    def _emit(self, message: str) -> None:
        """Emit one formatted progress-log line."""
        if not self.config.enabled:
            return
        if self.config.include_timestamp:
            stamp = time.strftime("%H:%M:%S")
            prefix = f"[{stamp}][{self.method_name}]"
        else:
            prefix = f"[{self.method_name}]"
        print(f"{prefix} {message}", flush=self.config.flush)

    def _should_emit_eval(self, eval_id: int) -> bool:
        """Return whether the current evaluation should be logged."""
        if eval_id <= 0:
            return False
        if self.config.emit_first_eval and eval_id == 1:
            return True
        every = max(int(self.config.eval_done_every), 1)
        return (eval_id % every) == 0

    def info(self, message: str, **meta) -> None:
        """Emit an informational progress line with optional metadata."""
        self._emit(f"{message}{self._fmt_meta(meta)}")

    def start_phase(self, phase_name: str, **meta) -> None:
        """Record the start time of one estimation phase and print it."""
        self._phase_starts[phase_name] = time.time()
        self._emit(f"START {phase_name}{self._fmt_meta(meta)}")

    def finish_phase(self, phase_name: str, **meta) -> None:
        """Close a named phase, append elapsed time, and print it."""
        start = self._phase_starts.pop(phase_name, None)
        if start is not None:
            meta = {**meta, "elapsed_sec": time.time() - start}
        self._emit(f"DONE {phase_name}{self._fmt_meta(meta)}")

    def start_multistart(self, phase_name: str, n_starts: int, max_evals: int) -> None:
        """Announce the start of a multi-start optimization block."""
        self.info(f"{phase_name}: multi-start begins", n_starts=n_starts, max_evals=max_evals)

    def start_local_run(self, phase_name: str, start_id: int, start_theta) -> None:
        """Announce the beginning of one local optimization start."""
        self.info(f"{phase_name}: local run starts", start_id=start_id, theta=start_theta)

    def local_eval_start(self, phase_name: str, start_id: int, eval_id: int, theta) -> None:
        """Optionally emit a progress line before one objective evaluation."""
        if self.config.emit_eval_start and self._should_emit_eval(eval_id):
            self.info(f"{phase_name}: objective eval", start_id=start_id, eval_id=eval_id, theta=theta)

    def local_eval_done(self, phase_name: str, start_id: int, eval_id: int, objective: float) -> None:
        """Optionally emit a progress line after one objective evaluation."""
        if self.config.emit_eval_done and self._should_emit_eval(eval_id):
            self.info(
                f"{phase_name}: objective eval done",
                start_id=start_id,
                eval_id=eval_id,
                objective=objective,
            )

    def finish_local_run(self, phase_name: str, start_id: int, **meta) -> None:
        """Emit the completion line for one local optimization start."""
        self.info(f"{phase_name}: local run finished", start_id=start_id, **meta)

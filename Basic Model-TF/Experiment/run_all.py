"""Legacy command-line wrapper for the package pipeline.

This module preserves the historical ``python -m Experiment.run_all`` entrypoint
while delegating the real orchestration to :mod:`basic_mailer.pipeline`.
"""

from __future__ import annotations

from collections.abc import Sequence

from basic_mailer.pipeline import main as _pipeline_main
from basic_mailer.pipeline import parse_args


def main(argv: Sequence[str] | None = None) -> None:
    """Run the package pipeline through the legacy module path."""
    _pipeline_main(argv)


if __name__ == "__main__":  # pragma: no cover
    main()

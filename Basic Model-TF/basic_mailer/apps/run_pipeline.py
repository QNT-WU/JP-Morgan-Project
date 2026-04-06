"""Package-level command-line entrypoint for the basic Mailer pipeline."""

from __future__ import annotations

from basic_mailer.pipeline import main, parse_args

__all__ = ["main", "parse_args"]

if __name__ == "__main__":  # pragma: no cover
    main()

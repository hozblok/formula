"""Backward-compatible alias of trace_v3 (top-up is the same command)."""

import sys

from .trace_v3 import LEGACY_SCHEME, main, tail_rng, trace

topup = trace

__all__ = ["LEGACY_SCHEME", "main", "tail_rng", "topup", "trace"]

if __name__ == "__main__":
    sys.exit(main())

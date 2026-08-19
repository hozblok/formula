"""Shared progress logging: stderr for stages, timestamped stdout for tracers."""

import sys
import time


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def tlog(msg: str) -> None:
    print(time.strftime("%H:%M:%S ") + msg, flush=True)

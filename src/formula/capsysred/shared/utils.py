"""Small structure helpers with no domain knowledge."""

import os
from contextlib import contextmanager


@contextmanager
def durable_open(path, mode="x", **kwargs):
    """open() whose writes are flushed and fsynced before close.

    The default mode "x" keeps the no-clobber contract of every canonical
    artifact; pass mode="w" for tmp files that an os.replace publishes."""
    with open(path, mode, **kwargs) as fh:
        yield fh
        fh.flush()
        os.fsync(fh.fileno())


def zeros(nx: int, ny: int):
    """ny×nx row-major grid of 0.0."""
    return [[0.0] * nx for _ in range(ny)]


def flat(grid):
    """Row-major grid -> flat list."""
    return [v for row in grid for v in row]

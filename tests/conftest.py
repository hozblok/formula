"""Shared archive fixtures."""

import pytest


@pytest.fixture
def write_v3_archive():
    from .rays_archive import write_v3_archive

    return write_v3_archive

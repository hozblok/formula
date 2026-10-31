"""Shared archive fixtures; the tests listed in slow.txt run only with --runslow."""

import warnings
from pathlib import Path

import pytest

HERE = Path(__file__).parent
SLOW = {
    line.split("#", 1)[0].strip()
    for line in (HERE / "slow.txt").read_text(encoding="utf-8").splitlines()
} - {""}


@pytest.fixture
def write_v3_archive():
    from .rays_archive import write_v3_archive

    return write_v3_archive


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="also run the tests in tests/slow.txt")


def _module(item):
    return item.path.relative_to(HERE).as_posix()


def _is_slow(item):
    return f"{_module(item)}::{getattr(item, 'originalname', item.name)}" in SLOW


def pytest_collection_modifyitems(config, items):
    modules = {_module(item): item.module for item in items if hasattr(item, "module")}
    # Entries of collected modules that name no function there.
    entries = (entry.split("::") for entry in SLOW)
    stale = sorted(
        f"{path}::{name}" for path, name in entries
        if path in modules and not hasattr(modules[path], name)
    )
    if stale:
        warnings.warn(f"tests/slow.txt names missing tests: {', '.join(stale)}")
    if config.getoption("--runslow"):
        return
    slow = [item for item in items if _is_slow(item)]
    config.hook.pytest_deselected(items=slow)
    items[:] = [item for item in items if not _is_slow(item)]

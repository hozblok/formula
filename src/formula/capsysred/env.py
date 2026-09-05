"""Process environment: every CAPSYSRED_* variable the package reads."""

import os


class Env:
    PYTHON_TRACE = "CAPSYSRED_PYTHON_TRACE"
    STAGE14_CACHE = "CAPSYSRED_STAGE14_CACHE"
    STAGE14_JOBS = "CAPSYSRED_STAGE14_JOBS"

    @staticmethod
    def python_trace() -> bool:
        """Force the Python reference tracer over the C++ twin."""
        return os.environ.get(Env.PYTHON_TRACE, "0") not in ("", "0")

    @staticmethod
    def stage14_cache() -> str | None:
        """Stage-14 cache root override; None = beside the archive."""
        return os.environ.get(Env.STAGE14_CACHE) or None

    @staticmethod
    def stage14_jobs() -> int:
        """Parallel stage-14 cache builders; unset/empty = 1."""
        value = os.environ.get(Env.STAGE14_JOBS, "")
        if not value.strip():
            return 1
        try:
            jobs = int(value)
        except ValueError:
            jobs = 0
        if jobs < 1:
            raise ValueError(
                f"{Env.STAGE14_JOBS} must be a positive integer, got {value!r}"
            )
        return jobs

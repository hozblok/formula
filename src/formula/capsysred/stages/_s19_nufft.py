"""Optional native FINUFFT ABI; no Python FINUFFT or NumPy imports."""

import ctypes as ct
import importlib.metadata
import os
from pathlib import Path
from functools import lru_cache


class _Options(ct.Structure):
    _fields_ = [(name, ct.c_int) for name in (
        "modeord", "spreadinterponly", "debug", "spread_debug", "showwarn",
        "nthreads", "fftw", "spread_sort", "spread_kerevalmeth", "spread_kerpad",
    )] + [("upsampfac", ct.c_double)] + [(name, ct.c_int) for name in (
        "spread_thread", "maxbatchsize", "spread_nthr_atomic", "spread_max_sp_size",
        "spread_kerformula",
    )] + [(name, ct.c_void_p) for name in (
        "fftw_lock_fun", "fftw_unlock_fun", "fftw_lock_data",
    )]


@lru_cache(maxsize=1)
def native_library():
    override = os.environ.get("FORMULA_FINUFFT_LIBRARY")
    if override:
        path = Path(override)
    else:
        distribution = importlib.metadata.distribution("finufft")
        if not distribution.version.startswith("2.5."):
            raise ValueError("Stage 19 native ABI currently supports FINUFFT 2.5.x")
        suffix = "libfinufft.dll" if os.name == "nt" else "libfinufft.so"
        path = Path(distribution.locate_file("finufft")) / suffix
    dependencies = path.parent.parent / "finufft.libs"
    if os.name == "nt" and dependencies.is_dir():
        with os.add_dll_directory(str(dependencies)):
            lib = ct.CDLL(str(path))
    else:
        lib = ct.CDLL(str(path))
    lib.finufft_default_opts.argtypes = [ct.POINTER(_Options)]
    lib.finufft_default_opts.restype = None
    lib.finufft_makeplan.argtypes = [
        ct.c_int, ct.c_int, ct.POINTER(ct.c_longlong), ct.c_int, ct.c_int,
        ct.c_double, ct.POINTER(ct.c_void_p), ct.POINTER(_Options),
    ]
    lib.finufft_makeplan.restype = ct.c_int
    lib.finufft_setpts.argtypes = [ct.c_void_p, ct.c_longlong, ct.c_void_p,
        ct.c_void_p, ct.c_void_p, ct.c_longlong, ct.c_void_p, ct.c_void_p, ct.c_void_p]
    lib.finufft_setpts.restype = ct.c_int
    lib.finufft_execute.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p]
    lib.finufft_execute.restype = ct.c_int
    lib.finufft_destroy.argtypes = [ct.c_void_p]
    lib.finufft_destroy.restype = ct.c_int
    return lib, str(path)


def _pointer(buffer):
    if isinstance(buffer, bytes):
        return ct.cast(ct.c_char_p(buffer), ct.c_void_p)
    return ct.cast((ct.c_char * len(buffer)).from_buffer(buffer), ct.c_void_p)


class Plan:
    """Centered type-1 modes, x-fast row-major complex128 output."""

    def __init__(self, nx, ny, *, eps=1e-10, nthreads=1):
        self.lib, self.library_path = native_library()
        self.nx, self.ny = nx, ny
        self.plan = ct.c_void_p()
        opts = _Options()
        self.lib.finufft_default_opts(ct.byref(opts))
        opts.nthreads = nthreads
        opts.fftw = 64
        opts.showwarn = 0
        modes = (ct.c_longlong * 2)(nx, ny)
        self._check(self.lib.finufft_makeplan(
            1, 2, modes, -1, 1, eps, ct.byref(self.plan), ct.byref(opts)))
        self.points = None

    @staticmethod
    def _check(status):
        if status:
            raise RuntimeError(f"Native FINUFFT returned status {status}")

    def set_points(self, x, y):
        if len(x) != len(y) or len(x) % 8:
            raise ValueError("Coordinate buffers must contain matching float64 arrays")
        self.points = (x, y)
        self._check(self.lib.finufft_setpts(
            self.plan, len(x) // 8, _pointer(x), _pointer(y), None, 0, None, None, None))

    def execute(self, strengths):
        if self.points is None or len(strengths) != 2 * len(self.points[0]):
            raise ValueError("Strength buffer does not match source nodes")
        output = bytearray(16 * self.nx * self.ny)
        self._check(self.lib.finufft_execute(
            self.plan, _pointer(strengths), _pointer(output)))
        return output

    def close(self):
        if self.plan.value:
            self.lib.finufft_destroy(self.plan)
            self.plan = ct.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

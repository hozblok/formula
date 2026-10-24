"""Optional surface-quadrature acceleration of the same 18LITE integral."""

from array import array
import math
import time

from formula import _stage19 as native
from ._s19_nufft import Plan


def evaluate_field(records, k, spec, observable, options):
    from .stage19 import _grid
    xs, ys, width, _ = _grid(spec)
    distance = float(spec["distance"])
    kappa = k / distance
    nx, ny = len(xs), len(ys)
    dx = float(spec["edge_x"]) / nx
    dy = float(spec["edge_y"]) / ny
    minimum = int(options.get("source_order", 6))
    safety = float(options.get("source_safety", 1.0))
    maximum = int(options.get("max_source_order", 512))
    limit = int(options.get("max_source_points", 10000000))
    pixel_order = int(options.get("pixel_order", 4))
    nthreads = int(options.get("nthreads", 1))
    eps = float(options.get("nufft_eps", 1e-10))
    cell = width if observable == "coherent_cell" else 0.0
    if cell:
        nodes, weights = native.gauss_rule(pixel_order)
        offsets = [(cell * (x-.5), cell * (y-.5), wx*wy)
                   for y, wy in zip(nodes, weights) for x, wx in zip(nodes, weights)]
    else:
        offsets = [(0., 0., 1.)]
    output = bytearray(16 * nx * ny)
    started = time.perf_counter()
    statistics = dict(backend="native-finufft-type1",
        physical_representation="constant complex amplitude and central linear OPL",
        integration="positive Duffy surface quadrature; checked by independent contour",
        numpy_used=False, nufft_eps=eps, source_order=minimum, source_safety=safety,
        pixel_order=pixel_order if cell else 1, source_points=0, source_batches=0,
        maximum_source_order=0, contour_check=None)
    batches = [records]
    with Plan(nx, ny, eps=eps, nthreads=nthreads) as plan:
        while batches:
            batch = batches.pop()
            if not batch:
                continue
            try:
                data = native.quadrature_sources(batch, kappa,
                    xs[0]-cell/2, xs[-1]+cell/2, ys[0]-cell/2, ys[-1]+cell/2,
                    minimum, safety, maximum, limit)
            except RuntimeError as exc:
                if "point limit" not in str(exc) or len(batch) < 2:
                    raise
                half = len(batch)//2
                batches.extend((batch[half:], batch[:half]))
                continue
            prepared = native.nufft_prepare(data["x"], data["y"], data["coefficients"],
                                            kappa*dx, kappa*dy, 0., 0.)
            plan.set_points(prepared["x"], prepared["y"])
            del prepared
            for ox, oy, weight in offsets:
                strengths = native.nufft_strengths(
                    data["x"], data["y"], data["coefficients"],
                    kappa*(xs[nx//2]+ox), kappa*(ys[ny//2]+oy))
                spectrum = plan.execute(strengths)
                native.nufft_postprocess(spectrum, output, xs, ys, kappa, ox, oy, weight)
            statistics["source_points"] += data["points"]
            statistics["source_batches"] += 1
            statistics["maximum_source_order"] = max(statistics["maximum_source_order"], data["max_order"])
        statistics["native_library"] = plan.library_path
    if options.get("contour_check", True):
        ix = sorted({nx//4, nx//2, 3*nx//4})
        iy = sorted({ny//4, ny//2, 3*ny//4})
        checked = native.eval_contour(records, k, distance, [xs[i] for i in ix],
            [ys[j] for j in iy], pixel_width=cell, order=16,
            pixel_order=max(pixel_order, 6), nthreads=nthreads)
        values = array("d"); values.frombytes(checked["field"])
        actual = memoryview(output).cast("d")
        err, norm = 0., 0.
        for a, (j, i) in enumerate((j, i) for j in iy for i in ix):
            value = complex(values[2*a], values[2*a+1])
            pos = j*nx+i
            observed = complex(actual[2*pos], actual[2*pos+1])
            err += abs(value-observed)**2; norm += abs(value)**2
        relative = math.sqrt(err/norm) if norm else (0. if err == 0 else None)
        tolerance = float(options.get("integration_tolerance", 0.002))
        passed = relative is not None and relative <= tolerance
        statistics["contour_check"] = dict(targets=len(ix)*len(iy),
            relative_complex_l2=relative, tolerance=tolerance, passed=passed,
            interpretation="Selected receiver check; not a bound over the whole screen")
        if not passed:
            raise RuntimeError(f"Stage 19 accelerated/contour disagreement {relative}; refine quadratures")
    statistics["seconds"] = time.perf_counter()-started
    return dict(field=bytes(output), shape=[ny,nx], diagnostics=statistics)

"""Positive quadrature of a prescribed exit amplitude and real phase.

Unlike the contour helpers, this path does not interpolate complex exponentials.
It integrates an affine complex amplitude and a P1/P2 *real* phase on every
triangle. It does not repair the exit model, omitted domains, or caustics.
"""

from __future__ import annotations

import time

import numpy as np


_MAX_TARGET_NODES = 65536
_MAX_DIRECT_PAIRS = 2_000_000
_EDGES = ((0, 1), (1, 2), (2, 0))


def _integer(value, name, minimum=1):
    try:
        valid = (not isinstance(value, (bool, np.bool_)) and np.isscalar(value)
                 and np.isreal(value) and np.isfinite(value)
                 and int(value) == value and value >= minimum)
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _real(value, name, *, zero=False):
    try:
        valid = (not isinstance(value, (bool, np.bool_)) and np.isscalar(value)
                 and not np.iscomplexobj(value) and np.isfinite(value)
                 and (value >= 0 if zero else value > 0))
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return float(value)


def _real_array(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real array") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _mesh(mesh, degree, k):
    triangles = _real_array(mesh["triangles"], "triangles")
    phase = _real_array(mesh["vertex_phase"], "vertex_phase")
    amplitude = np.asarray(mesh["vertex_amplitude"], dtype=complex)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 2):
        raise ValueError("triangles must have shape (N, 3, 2)")
    if phase.shape != triangles.shape[:2] or amplitude.shape != phase.shape:
        raise ValueError("vertex_phase and vertex_amplitude must have shape (N, 3)")
    if not np.isfinite(amplitude).all():
        raise ValueError("vertex_amplitude must be finite")
    directions = mesh.get("vertex_directions")
    if directions is not None:
        directions = _real_array(directions, "vertex_directions")
        if directions.shape != triangles.shape:
            raise ValueError("vertex_directions must have shape (N, 3, 2)")
    if len(triangles) and degree == 2 and directions is None:
        raise ValueError("phase_degree=2 requires vertex_directions")
    edges = triangles[:, 1:]-triangles[:, :1]
    determinant = edges[:, 0, 0]*edges[:, 1, 1]-edges[:, 0, 1]*edges[:, 1, 0]
    if not np.isfinite(determinant).all() or np.any(determinant == 0):
        raise ValueError("exit mesh contains a degenerate or non-finite triangle")
    correction = np.zeros(phase.shape)
    if degree == 2 and len(triangles):
        for index, (i, j) in enumerate(_EDGES):
            correction[:, index] = (-k/8)*np.sum(
                (directions[:, j]-directions[:, i])*(triangles[:, j]-triangles[:, i]), axis=-1)
    if not np.isfinite(correction).all() or not np.isfinite(phase[:, 1:]-phase[:, :1]).all():
        raise ValueError("exit phase differences or midpoint corrections are not finite")
    return triangles, amplitude, phase, directions, determinant, correction


def _phase_fit(triangles, phase, directions, correction, determinant, k):
    """Consistency indicators, not a quadrature or physical field error bound."""
    result = dict(available=directions is not None,
                  interpretation="Consistency of supplied vertex phase/directions with the selected real-phase polynomial; not an error bound.")
    if directions is None:
        return result
    accum = {name: [0, 0., 0.] for name in
             ("edge_trapezoid_residual_rad", "vertex_gradient_times_diameter_residual_rad",
              "midpoint_correction_rad")}

    def record(name, values):
        values = np.abs(values).ravel()
        if not np.isfinite(values).all():
            raise ValueError("phase-fit diagnostics are not finite; exit triangle is ill-conditioned")
        row = accum[name]
        row[0] += len(values)
        maximum = max(row[2], float(np.max(values, initial=0.)))
        if maximum:
            row[1] = row[1]*(row[2]/maximum)**2+float(np.sum((values/maximum)**2))
        row[2] = maximum

    for start in range(0, len(triangles), 16384):
        stop = start+16384
        t, p, u, c = (v[start:stop] for v in (triangles, phase, directions, correction))
        d = determinant[start:stop]
        e1, e2 = t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]
        g1 = np.column_stack((e2[:, 1], -e2[:, 0]))/d[:, None]
        g2 = np.column_stack((-e1[:, 1], e1[:, 0]))/d[:, None]
        g0 = -g1-g2
        affine = (p[:, 1]-p[:, 0])[:, None]*g1+(p[:, 2]-p[:, 0])[:, None]*g2
        gradients = np.repeat(affine[:, None], 3, axis=1)
        gradients[:, 0] += 4*(c[:, 0, None]*g1+c[:, 2, None]*g2)
        gradients[:, 1] += 4*(c[:, 0, None]*g0+c[:, 1, None]*g2)
        gradients[:, 2] += 4*(c[:, 1, None]*g1+c[:, 2, None]*g0)
        diameter = np.zeros(len(t))
        for i, j in _EDGES:
            edge = t[:, j]-t[:, i]
            diameter = np.maximum(diameter, np.linalg.norm(edge, axis=-1))
            record("edge_trapezoid_residual_rad", p[:, j]-p[:, i]
                   -(k/2)*np.sum((u[:, i]+u[:, j])*edge, axis=-1))
        record("vertex_gradient_times_diameter_residual_rad",
               np.linalg.norm(gradients-k*u, axis=-1)*diameter[:, None])
        record("midpoint_correction_rad", c)
    for name, (count, scaled_square_sum, maximum) in accum.items():
        result[name] = dict(count=count, maximum=maximum,
                            rms=float(maximum*np.sqrt(scaled_square_sum/count)) if count else 0.)
    return result


def _point_transform(points, coefficients, frequencies, *, backend, eps, nthreads):
    """Unnormalized Fourier sum, sign minus; arrays contain centered coordinates."""
    if backend == "direct":
        return coefficients @ np.exp(-1j*(points @ frequencies.T))
    import finufft
    return finufft.nufft2d3(np.ascontiguousarray(points[:, 0]),
                           np.ascontiguousarray(points[:, 1]),
                           np.ascontiguousarray(coefficients),
                           np.ascontiguousarray(frequencies[:, 0]),
                           np.ascontiguousarray(frequencies[:, 1]),
                           isign=-1, eps=eps, nthreads=nthreads)


def phase_field(mesh, *, k, distance, x, y, cell_width=0., pixel_order=4,
                quadrature_order=8, phase_degree=2, backend="finufft", nthreads=1,
                eps=1e-10, max_nodes_per_batch=500000, return_stats=False):
    """Propagate an affine amplitude and P1/P2 phase; return shape ``(len(y),len(x))``.

    ``vertex_directions`` are the transverse outgoing unit-ray components, so
    the phase gradients used here are ``k*u``. P2 uses three vertex phases and
    three directional-Hermite edge-midpoint values. It does not in general fit
    all six supplied gradient components. Equal phase/direction endpoints give
    equal P2 traces on a shared straight edge.

    Positive Duffy Gauss quadrature samples the exponential of this real phase.
    The receiver is the coherent square-cell mean, including its Fresnel chirp
    at every Gauss node. No longitudinal common phase ``exp(i*k*distance)`` is
    included, matching the existing stage18 convention.

    Source and receiver node arrays are batched. FINUFFT workspace additionally
    depends on spatial bandwidth; the batch limit is not a strict memory cap.
    ``eps`` controls NUFFT accuracy only: source and receiver quadrature orders
    and the prescribed phase model need independent convergence checks.
    """
    started = time.perf_counter()
    k, distance = _real(k, "k"), _real(distance, "distance")
    cell_width, eps = _real(cell_width, "cell_width", zero=True), _real(eps, "eps")
    if eps >= 1:
        raise ValueError("eps must be smaller than one")
    pixel_order = _integer(pixel_order, "pixel_order")
    quadrature_order = _integer(quadrature_order, "quadrature_order")
    phase_degree = _integer(phase_degree, "phase_degree")
    nthreads = _integer(nthreads, "nthreads")
    max_nodes_per_batch = _integer(max_nodes_per_batch, "max_nodes_per_batch")
    if phase_degree not in (1, 2):
        raise ValueError("phase_degree must be 1 or 2")
    if backend not in ("direct", "finufft", "auto"):
        raise ValueError("backend must be direct, finufft, or auto")
    x, y = _real_array(x, "x"), _real_array(y, "y")
    if x.ndim != 1 or y.ndim != 1 or not len(x) or not len(y):
        raise ValueError("x and y must be nonempty one-dimensional arrays")
    triangles, amplitude, phase, directions, determinant, correction = _mesh(mesh, phase_degree, k)
    kappa = k/distance
    if not np.isfinite(kappa):
        raise ValueError("k/distance is not finite")
    count, cells = len(triangles), len(x)*len(y)
    per_triangle = quadrature_order**2
    effective_pixel_order = pixel_order if cell_width else 1
    per_cell = effective_pixel_order**2
    source_count, target_count = count*per_triangle, cells*per_cell
    selected_backend = ("direct" if source_count*target_count <= _MAX_DIRECT_PAIRS
                        else "finufft") if backend == "auto" else backend
    stats = dict(method="positive-Duffy real-phase Fresnel integration", phase_degree=phase_degree,
                 phase_rule=("affine vertex phase" if phase_degree == 1 else
                             "P2 vertex phase plus directional-Hermite edge midpoints"),
                 amplitude_rule="affine complex vertex amplitude", quadrature_rule="positive tensor Gauss-Legendre through Duffy map",
                 quadrature_order=quadrature_order, pixel_order=pixel_order,
                 effective_pixel_order=effective_pixel_order, receiver="coherent square-cell mean",
                 triangles=count, source_nodes=source_count, target_nodes=target_count,
                 source_batches=0, transform_calls=0, maximum_source_batch=0, maximum_target_batch=0,
                 max_nodes_per_batch=max_nodes_per_batch, backend=selected_backend, eps=eps, nthreads=nthreads,
                 phase_fit=_phase_fit(triangles, phase, directions, correction, determinant, k),
                 limitation="Quadrature of the prescribed exit model; not a caustic repair or certified physical field error.")
    field = np.zeros(cells, dtype=complex)
    if not count:
        stats["seconds"] = time.perf_counter()-started
        field = field.reshape(len(y), len(x))
        return (field, stats) if return_stats else field
    lower, upper = triangles.min(axis=(0, 1)), triangles.max(axis=(0, 1))
    origin = lower/2+upper/2
    local_triangles = triangles-origin
    scale = float(np.max(np.abs(local_triangles)))
    stats.update(coordinate_origin=origin.tolist(), coordinate_scale=scale)
    gauss, weights = np.polynomial.legendre.leggauss(quadrature_order)
    gauss, weights = (gauss+1)/2, weights/2
    if cell_width:
        receiver_nodes, receiver_weights = np.polynomial.legendre.leggauss(pixel_order)
        receiver_nodes, receiver_weights = receiver_nodes*cell_width/2, receiver_weights/2
    else:
        receiver_nodes, receiver_weights = np.array([0.]), np.array([1.])
    source_limit = min(max_nodes_per_batch, _MAX_DIRECT_PAIRS) if selected_backend == "direct" else max_nodes_per_batch
    for start in range(0, source_count, source_limit):
        flat = np.arange(start, min(start+source_limit, source_count))
        parent, node = flat//per_triangle, flat % per_triangle
        i, j = node//quadrature_order, node % quadrature_order
        u, v = gauss[i], gauss[j]
        l0, l1, l2 = (1-u)*(1-v), u, (1-u)*v
        t = local_triangles[parent]
        points = t[:, 0]+l1[:, None]*(t[:, 1]-t[:, 0])+l2[:, None]*(t[:, 2]-t[:, 0])
        p = phase[parent]
        local_phase = l1*(p[:, 1]-p[:, 0])+l2*(p[:, 2]-p[:, 0])
        c = correction[parent]
        local_phase += 4*(c[:, 0]*l0*l1+c[:, 1]*l1*l2+c[:, 2]*l2*l0)
        a = amplitude[parent]
        values = l0*a[:, 0]+l1*a[:, 1]+l2*a[:, 2]
        source_phase = local_phase+.5*kappa*np.sum(points*points, axis=-1)
        coefficients = (np.abs(determinant[parent])*weights[i]*weights[j]*(1-u)*values
                        *np.exp(1j*p[:, 0])*np.exp(1j*source_phase))
        if not np.isfinite(source_phase).all() or not np.isfinite(coefficients).all():
            raise ValueError("source quadrature produced non-finite phase or coefficients")
        points /= scale
        stats["source_batches"] += 1
        stats["maximum_source_batch"] = max(stats["maximum_source_batch"], len(points))
        target_limit = _MAX_TARGET_NODES
        if selected_backend == "direct":
            target_limit = min(target_limit, max(1, _MAX_DIRECT_PAIRS//len(points)))
        for target_start in range(0, target_count, target_limit):
            target_flat = np.arange(target_start, min(target_start+target_limit, target_count))
            cell, node = target_flat//per_cell, target_flat % per_cell
            ix, iy = node % effective_pixel_order, node//effective_pixel_order
            targets = np.column_stack((x[cell % len(x)]-origin[0]+receiver_nodes[ix],
                                       y[cell//len(x)]-origin[1]+receiver_nodes[iy]))
            target_phase = .5*kappa*np.sum(targets*targets, axis=-1)
            frequencies = (kappa*scale)*targets
            if not np.isfinite(target_phase).all() or not np.isfinite(frequencies).all():
                raise ValueError("receiver quadrature produced non-finite phase or frequencies")
            spectrum = _point_transform(points, coefficients, frequencies, backend=selected_backend,
                                        eps=eps, nthreads=nthreads)
            contribution = (kappa/(2j*np.pi))*np.exp(1j*target_phase)*spectrum
            np.add.at(field, cell, receiver_weights[ix]*receiver_weights[iy]*contribution)
            stats["transform_calls"] += 1
            stats["maximum_target_batch"] = max(stats["maximum_target_batch"], len(targets))
    if not np.isfinite(field).all():
        raise ValueError("propagated field is not finite")
    stats["seconds"] = time.perf_counter()-started
    field = field.reshape(len(y), len(x))
    return (field, stats) if return_stats else field

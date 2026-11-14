"""Opt-in type-1 acceleration of prescribed-phase Fresnel cell means.

Each receiver Gauss offset defines an exact regular frequency lattice. Carriers
shift that lattice onto centered integer modes; no receiver interpolation or
change of the exit-field/pixel observable is involved. The original type-3
``phase_field`` remains the independent reference path.
"""

from __future__ import annotations

import time

import numpy as np

from ._b9_phase import _integer, _mesh, _phase_fit, _real, _real_array


def _axis(value, name):
    values = _real_array(value, name)
    if values.ndim != 1 or not len(values):
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    center = float(values[len(values)//2])
    if len(values) == 1:
        return values, center, 0., 0.
    spacing = float((values[-1]-values[0])/(len(values)-1))
    if not np.isfinite(spacing) or spacing == 0:
        raise ValueError(f"{name} must be a strictly monotone regular grid")
    exact = center+(np.arange(len(values))-len(values)//2)*spacing
    defect = float(np.max(abs(values-exact)))
    tolerance = 32*np.finfo(float).eps*max(float(np.max(abs(values))), abs(spacing))
    if defect > tolerance:
        raise ValueError(f"{name} must be a regular grid; maximum lattice defect is {defect:g}")
    return values, center, spacing, defect


def _source_batches(triangles, amplitude, phase, determinant, correction, origin,
                    kappa, order, limit):
    """Same positive Duffy rule/real-phase polynomial as the frozen type-3 path."""
    gauss, weights = np.polynomial.legendre.leggauss(order)
    gauss, weights = (gauss+1)/2, weights/2
    local_triangles = triangles-origin
    per_triangle = order**2
    for start in range(0, len(triangles)*per_triangle, limit):
        flat = np.arange(start, min(start+limit, len(triangles)*per_triangle))
        parent, node = flat//per_triangle, flat % per_triangle
        i, j = node//order, node % order
        u, v = gauss[i], gauss[j]
        l0, l1, l2 = (1-u)*(1-v), u, (1-u)*v
        t = local_triangles[parent]
        points = t[:, 0]+l1[:, None]*(t[:, 1]-t[:, 0])+l2[:, None]*(t[:, 2]-t[:, 0])
        p, c = phase[parent], correction[parent]
        local_phase = l1*(p[:, 1]-p[:, 0])+l2*(p[:, 2]-p[:, 0])
        local_phase += 4*(c[:, 0]*l0*l1+c[:, 1]*l1*l2+c[:, 2]*l2*l0)
        a = amplitude[parent]
        values = l0*a[:, 0]+l1*a[:, 1]+l2*a[:, 2]
        source_phase = local_phase+.5*kappa*np.sum(points*points, axis=-1)
        coefficients = (np.abs(determinant[parent])*weights[i]*weights[j]*(1-u)*values
                        *np.exp(1j*p[:, 0])*np.exp(1j*source_phase))
        if not np.isfinite(source_phase).all() or not np.isfinite(coefficients).all():
            raise ValueError("source quadrature produced non-finite phase or coefficients")
        yield points, coefficients


def regular_phase_field(mesh, *, k, distance, x, y, cell_width=0., pixel_order=4,
                        quadrature_order=8, phase_degree=2, eps=1e-10, nthreads=1,
                        max_nodes_per_batch=500000, receiver_channels_per_batch=4,
                        return_stats=False):
    """Integrate the same P1/P2 exit model on regular receiver axes using type 1.

    ``m=j-floor(nx/2)`` and ``X_j=X_center+m*dx``. For each pixel Gauss offset
    ``a`` the Fourier coefficient is multiplied by
    ``exp(-i*k/d*(X_center+a-origin).(q-origin))``. FINUFFT then evaluates the
    remaining integer modes at ``theta=k/d*dx*(q-origin)`` with sign minus.
    Only theta is wrapped modulo 2*pi; physical source coordinates are retained
    in every carrier. Receiver chirps are applied at individual Gauss points.

    Source quadrature and the coherent native-cell mean match ``phase_field``.
    Differences should be NUFFT/roundoff errors, not a new physical model.
    ``eps`` is not a source/receiver quadrature tolerance or a field-error bound.
    Batched channels reuse a FINUFFT plan and its sorted source points; internal
    workspace also depends on grid size, so batch limits are not memory bounds.
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
    receiver_channels_per_batch = _integer(receiver_channels_per_batch, "receiver_channels_per_batch")
    if phase_degree not in (1, 2):
        raise ValueError("phase_degree must be 1 or 2")
    x, cx, dx, x_defect = _axis(x, "x")
    y, cy, dy, y_defect = _axis(y, "y")
    triangles, amplitude, phase, directions, determinant, correction = _mesh(mesh, phase_degree, k)
    kappa = k/distance
    if not np.isfinite(kappa):
        raise ValueError("k/distance is not finite")
    effective_pixel_order = pixel_order if cell_width else 1
    channel_count = effective_pixel_order**2
    channels = min(receiver_channels_per_batch, channel_count)
    stats = dict(method="regular receiver type-1 positive-Duffy real-phase Fresnel integration",
        backend="finufft-type1", phase_degree=phase_degree,
        phase_rule="affine vertex phase" if phase_degree == 1 else "P2 vertex phase plus directional-Hermite edge midpoints",
        amplitude_rule="affine complex vertex amplitude", quadrature_rule="positive tensor Gauss-Legendre through Duffy map",
        quadrature_order=quadrature_order, pixel_order=pixel_order,
        effective_pixel_order=effective_pixel_order, receiver="coherent square-cell mean",
        triangles=len(triangles), source_nodes=len(triangles)*quadrature_order**2,
        receiver_cells=len(x)*len(y), receiver_channels=channel_count,
        receiver_channels_per_batch=channels, source_batches=0, transform_calls=0,
        maximum_source_batch=0, plans=0, padded_channel_transforms=0,
        max_nodes_per_batch=max_nodes_per_batch, eps=eps, nthreads=nthreads,
        receiver_lattice=dict(nx=len(x), ny=len(y), center=[cx, cy], spacing=[dx, dy],
                              maximum_float_lattice_defect=[x_defect, y_defect]),
        phase_fit=_phase_fit(triangles, phase, directions, correction, determinant, k),
        limitation="Same prescribed exit model and pixel quadrature as type3; NUFFT tolerance does not bound model or quadrature error.")
    field = np.zeros((len(y), len(x)), complex)
    if not len(triangles):
        stats["seconds"] = time.perf_counter()-started
        return (field, stats) if return_stats else field
    import finufft
    lower, upper = triangles.min(axis=(0, 1)), triangles.max(axis=(0, 1))
    origin = lower/2+upper/2
    stats["coordinate_origin"] = origin.tolist()
    stats["maximum_lattice_fourier_phase_defect_bound_rad"] = float(
        kappa*np.sum(np.max(abs(triangles-origin), axis=(0, 1))*np.array([x_defect, y_defect])))
    if cell_width:
        gauss, weights = np.polynomial.legendre.leggauss(pixel_order)
        gauss, weights = gauss*cell_width/2, weights/2
    else:
        gauss, weights = np.array([0.]), np.array([1.])
    offsets = np.array([(a, b) for b in gauss for a in gauss])
    receiver_weights = np.array([a*b for b in weights for a in weights])
    carrier = kappa*(np.array([cx, cy])-origin+offsets)
    if not np.isfinite(carrier).all():
        raise ValueError("receiver carriers are not finite")
    plan = finufft.Plan(1, (len(x), len(y)), n_trans=channels, eps=eps,
                       isign=-1, dtype="complex128", nthreads=nthreads, modeord=0)
    stats["plans"] = 1
    for points, coefficients in _source_batches(triangles, amplitude, phase, determinant,
            correction, origin, kappa, quadrature_order, max_nodes_per_batch):
        theta = points*(kappa*np.array([dx, dy]))
        if not np.isfinite(theta).all():
            raise ValueError("type1 source coordinates are not finite")
        theta = np.remainder(theta+np.pi, 2*np.pi)-np.pi
        plan.setpts(np.ascontiguousarray(theta[:, 0]), np.ascontiguousarray(theta[:, 1]))
        stats["source_batches"] += 1
        stats["maximum_source_batch"] = max(stats["maximum_source_batch"], len(points))
        for start in range(0, channel_count, channels):
            stop = min(start+channels, channel_count)
            physical_phase = carrier[start:stop]@points.T
            if not np.isfinite(physical_phase).all():
                raise ValueError("type1 carrier phase is not finite")
            strengths = np.zeros((channels, len(points)), complex)
            strengths[:stop-start] = coefficients*np.exp(-1j*physical_phase)
            spectrum = plan.execute(strengths)
            if channels == 1:
                spectrum = np.asarray(spectrum).reshape(1, len(x), len(y))
            stats["transform_calls"] += 1
            stats["padded_channel_transforms"] += channels-(stop-start)
            for local, channel in enumerate(range(start, stop)):
                rx = x-origin[0]+offsets[channel, 0]
                ry = y-origin[1]+offsets[channel, 1]
                chirp = .5*kappa*(rx[None, :]**2+ry[:, None]**2)
                if not np.isfinite(chirp).all():
                    raise ValueError("receiver chirp is not finite")
                field += receiver_weights[channel]*(kappa/(2j*np.pi))*np.exp(1j*chirp)*spectrum[local].T
    if not np.isfinite(field).all():
        raise ValueError("propagated field is not finite")
    stats["seconds"] = time.perf_counter()-started
    return (field, stats) if return_stats else field


def regular_phase_order_pair(mesh, *, quadrature_orders=(8, 12), **kwargs):
    """Return the higher-order field and paired-order sensitivity diagnostics.

    This is a measured difference of two positive Gauss rules. It is not a
    guaranteed bound, and coincident values do not exclude shared aliasing.
    Use a separate small receiver grid for an inexpensive convergence probe.
    """
    if "quadrature_order" in kwargs or "return_stats" in kwargs:
        raise ValueError("order-pair wrapper controls quadrature_order and return_stats")
    if len(quadrature_orders) != 2:
        raise ValueError("quadrature_orders must contain exactly two increasing orders")
    low, high = [_integer(q, "quadrature_orders") for q in quadrature_orders]
    if low >= high:
        raise ValueError("quadrature_orders must be strictly increasing")
    a, sa = regular_phase_field(mesh, quadrature_order=low, return_stats=True, **kwargs)
    b, sb = regular_phase_field(mesh, quadrature_order=high, return_stats=True, **kwargs)
    absolute = abs(a-b)
    norm = float(np.linalg.norm(b))
    intensity = abs(b)**2
    mask = intensity > 1e-4*float(np.max(intensity))
    relative = absolute[mask]/abs(b[mask])
    return b, dict(quadrature_orders=[low, high], runs=[sa, sb],
        comparison=dict(absolute_complex_l2=float(np.linalg.norm(a-b)),
            relative_complex_l2=float(np.linalg.norm(a-b)/norm) if norm else None,
            maximum_absolute_complex_difference=float(np.max(absolute)),
            reference_peak_intensity=float(np.max(intensity)), intensity_mask_relative_threshold=1e-4,
            masked_cells=int(mask.sum()), masked_relative_p95=float(np.quantile(relative, .95)) if len(relative) else None,
            masked_relative_maximum=float(np.max(relative)) if len(relative) else None),
        interpretation="Paired-order sensitivity of the specified sampled-cell vector, not an upper bound or a test of the physical exit model.")

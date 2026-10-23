"""Carrier-demodulated P1 contour propagation of a prescribed exit mesh.

The carriers change the representation only. They do not correct the supplied
geometrical-optics exit field, omitted triangles, or source sampling errors.
"""

from __future__ import annotations

from functools import lru_cache
import time
import warnings

import numpy as np
from scipy.cluster.vq import kmeans2

from ._b9_contour import triangle_fourier


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value) or int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _mesh(mesh):
    triangles = np.asarray(mesh["triangles"], float)
    amplitude = np.asarray(mesh["vertex_amplitude"], complex)
    phase = np.asarray(mesh["vertex_phase"], float)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 2) or not len(triangles):
        raise ValueError("triangles must have shape (N, 3, 2), with N > 0")
    if amplitude.shape != triangles.shape[:2] or phase.shape != amplitude.shape:
        raise ValueError("vertex amplitudes and phases must have shape (N, 3)")
    if not all(np.isfinite(a).all() for a in (triangles, amplitude, phase)):
        raise ValueError("exit mesh must be finite")
    edge = triangles[:, 1:]-triangles[:, :1]
    determinant = edge[:, 0, 0]*edge[:, 1, 1]-edge[:, 0, 1]*edge[:, 1, 0]
    if np.any(determinant == 0):
        raise ValueError("exit mesh contains a degenerate triangle")
    gradient = np.linalg.solve(edge, (phase[:, 1:]-phase[:, :1])[..., None])[..., 0]
    if not np.isfinite(gradient).all():
        raise ValueError("exit phase gradient is not finite")
    return triangles, amplitude, phase, gradient


def _select_carriers(triangles, phase, gradient, kappa, groups):
    chirped_gradient = gradient+kappa*triangles.mean(axis=1)
    if not np.isfinite(chirped_gradient).all():
        raise ValueError("chirped phase gradient is not finite")
    if len(triangles) < 128:
        low, high = chirped_gradient.min(axis=0), chirped_gradient.max(axis=0)
    else:
        low, high = np.quantile(chirped_gradient, [.005, .995], axis=0)
    clipped = np.clip(chirped_gradient, low, high)
    unique, inverse = np.unique(clipped, axis=0, return_inverse=True)
    count = min(groups, len(unique))
    if count == len(unique):
        carriers, labels = unique, inverse
    elif count == 1:
        carriers, labels = clipped.mean(axis=0, keepdims=True), np.zeros(len(triangles), int)
    else:
        scale = max(float(np.max(abs(clipped))), 1.)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="One of the clusters is empty.*", category=UserWarning)
            carriers, labels = kmeans2(clipped/scale, count, minit="++", iter=20, seed=1729, missing="warn")
        present, labels = np.unique(labels, return_inverse=True)
        carriers = carriers[present]*scale
    if not np.isfinite(carriers).all():
        raise ValueError("carrier selection produced non-finite coefficients")
    original = phase+.5*kappa*np.sum(triangles**2, axis=-1)
    residual = original-np.einsum("tvi,ti->tv", triangles, carriers[labels])
    return labels, carriers, dict(
        carrier_groups_requested=groups, carrier_groups_effective=len(carriers), seed=1729,
        carrier_selection="20 deterministic Lloyd iterations on chirped phase gradients; componentwise 0.5%-99.5% winsorization for meshes with at least 128 triangles",
        clipping_scope="Only grouping/carrier selection; no amplitude, phase, triangle or ray is clipped or removed.",
        gradient_clip_bounds_per_m=[low.tolist(), high.tolist()],
        carriers_per_m=carriers.tolist(), triangle_counts=np.bincount(labels, minlength=len(carriers)).tolist(),
        original_phase_span_p50_p95_max_rad=np.quantile(np.ptp(original, axis=1), [.5, .95, 1]).tolist(),
        demodulated_phase_span_p50_p95_max_rad=np.quantile(np.ptp(residual, axis=1), [.5, .95, 1]).tolist())


@lru_cache(maxsize=16)
def _subdivision_rule(n):
    rows = []
    for i in range(n):
        for j in range(n-i):
            a, b, c = np.array([[i, j], [i+1, j], [i, j+1]], float)/n
            rows.append([a, b, c])
            if i+j < n-1:
                rows.append([b, np.array([i+1, j+1])/n, c])
    uv = np.asarray(rows)
    return np.concatenate((1-uv.sum(axis=-1, keepdims=True), uv), axis=-1)


def carrier_field(mesh, subdivisions, groups, *, k, distance, x, y, cell_width=0.,
                  pixel_order=4, edge_order=8, eps=1e-10, nthreads=1,
                  max_triangles_per_batch=40000, backend="auto"):
    """Return a coherently averaged field and representation diagnostics.

    g_j=exp(i c_j.q) gtilde_j gives F[g_j](tau)=F[gtilde_j](tau-c_j).
    Every source field retains its original amplitude and unwrapped exit phase.
    """
    started = time.perf_counter()
    subdivisions = _integer(subdivisions, "subdivisions")
    groups, pixel_order = _integer(groups, "groups"), _integer(pixel_order, "pixel_order")
    edge_order, nthreads = _integer(edge_order, "edge_order", 2), _integer(nthreads, "nthreads")
    batch = _integer(max_triangles_per_batch, "max_triangles_per_batch")
    if not np.isfinite([k, distance, cell_width, eps]).all() or k <= 0 or distance <= 0 or cell_width < 0 or not 0 < eps < 1:
        raise ValueError("invalid propagation or precision parameters")
    x, y = np.asarray(x, float), np.asarray(y, float)
    if any(a.ndim != 1 or not len(a) or not np.isfinite(a).all() for a in (x, y)):
        raise ValueError("receiver axes must be nonempty finite vectors")
    if backend not in ("auto", "direct", "finufft", "finufft3"):
        raise ValueError("invalid Fourier backend")
    triangles, amplitude, phase, gradient = _mesh(mesh)
    kappa = k/distance
    labels, carriers, stats = _select_carriers(triangles, phase, gradient, kappa, groups)
    if cell_width:
        nodes, weights = np.polynomial.legendre.leggauss(pixel_order)
        nodes, weights = nodes*cell_width/2, weights/2
    else:
        nodes, weights = np.array([0.]), np.array([1.])
    order = len(nodes)
    tx, ty = (x[:, None]+nodes).ravel(), (y[:, None]+nodes).ravel()
    spectrum = np.zeros((len(ty), len(tx)), complex)
    bary = _subdivision_rule(subdivisions)
    parent_batch = max(1, batch//(subdivisions**2))
    group_stats = []
    for group, carrier in enumerate(carriers):
        tick = time.perf_counter()
        ids = np.flatnonzero(labels == group)
        calls = 0
        for begin in range(0, len(ids), parent_batch):
            selected = ids[begin:begin+parent_batch]
            parent = triangles[selected]
            demodulated = phase[selected]-parent@carrier
            q = np.einsum("svj,tjk->tsvk", bary, parent).reshape(-1, 3, 2)
            phi = np.einsum("svj,tj->tsv", bary, demodulated).reshape(-1, 3)
            a = np.einsum("svj,tj->tsv", bary, amplitude[selected]).reshape(-1, 3)
            values = a*np.exp(1j*(phi+.5*kappa*np.sum(q*q, axis=-1)))
            for offset in range(0, len(q), batch):
                spectrum += triangle_fourier(q[offset:offset+batch], values[offset:offset+batch],
                    kappa*tx-carrier[0], kappa*ty-carrier[1], edge_order=edge_order,
                    eps=eps, nthreads=nthreads, backend=backend)
                calls += 1
        group_stats.append(dict(group=group, parent_triangles=len(ids),
            subtriangles=len(ids)*subdivisions**2, contour_batches=calls, seconds=time.perf_counter()-tick))
    chirp = np.exp(.5j*kappa*(tx[None, :]**2+ty[:, None]**2))
    field = kappa/(2j*np.pi)*np.einsum("iajb,a,b->ij", (chirp*spectrum).reshape(len(y), order, len(x), order), weights, weights)
    if not np.isfinite(field).all():
        raise ValueError("carrier reconstruction produced a non-finite field")
    stats.update(subdivisions=subdivisions, pixel_order=pixel_order, effective_pixel_order=order,
        edge_order=edge_order, nufft_eps=eps, backend=backend, max_triangles_per_batch=batch,
        group_runs=group_stats, seconds=time.perf_counter()-started,
        receiver="Original output chirp is evaluated at every Gauss node before coherent averaging.",
        scope="Numerical representation of the prescribed affine exit model; no correction of GO physics or missing areas.")
    return field, stats

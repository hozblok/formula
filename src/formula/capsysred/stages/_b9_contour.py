"""Fresnel propagation of a piecewise affine field using triangle contours.

The prescribed vertex values belong to the chirped exit field.  This module
does not infer that field, its reflection history, or its transport amplitude.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=32)
def _gauss(order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return (nodes + 1.0) / 2.0, weights / 2.0


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _axes(kx, ky):
    kx, ky = np.asarray(kx, float), np.asarray(ky, float)
    if any(a.ndim != 1 or not len(a) or not np.isfinite(a).all() for a in (kx, ky)):
        raise ValueError("frequency axes must be nonempty finite vectors")
    return kx, ky


@dataclass
class Contour:
    """Dimensionless oriented-edge quadrature and affine triangle moments."""

    triangles: np.ndarray
    values: np.ndarray
    origin: np.ndarray
    scale: float
    area: np.ndarray
    points: np.ndarray
    coefficients: np.ndarray
    max_frequency: np.ndarray
    edge_order: int
    subdivisions: np.ndarray

    @property
    def stats(self):
        return {
            "triangles": len(self.triangles),
            "edge_nodes": len(self.points),
            "edge_order": self.edge_order,
            "maximum_edge_subdivisions": int(self.subdivisions.max(initial=0)),
            "area": float(self.area.sum() * self.scale**2),
        }


def prepare_contour(triangles, vertex_values, *, max_frequency,
                    edge_order=8, phase_step=4.0):
    """Prepare all edge jumps; shared edges are never silently discarded.

    ``max_frequency`` contains bounds on the absolute x/y angular frequencies.
    ``phase_step`` limits the phase change on each Gauss integration subedge.
    """
    triangles, values = np.asarray(triangles, float), np.asarray(vertex_values, complex)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 2) or not len(triangles):
        raise ValueError("triangles must have shape (N, 3, 2), with N > 0")
    if values.shape != triangles.shape[:2] or not np.isfinite(triangles).all() or not np.isfinite(values).all():
        raise ValueError("finite vertex values must have shape (N, 3)")
    maximum = np.asarray(max_frequency, float)
    if maximum.shape != (2,) or not np.isfinite(maximum).all() or np.any(maximum < 0):
        raise ValueError("max_frequency must contain two finite nonnegative bounds")
    edge_order = _integer(edge_order, "edge_order", 2)
    if not np.isfinite(phase_step) or phase_step <= 0:
        raise ValueError("phase_step must be positive and finite")
    low, high = triangles.min(axis=(0, 1)), triangles.max(axis=(0, 1))
    origin, scale = (low + high) / 2.0, float(np.max(high - low) / 2.0)
    if scale == 0:
        raise ValueError("triangle mesh has zero extent")
    q = (triangles - origin) / scale
    e1, e2 = q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]
    determinant = e1[:, 0]*e2[:, 1] - e1[:, 1]*e2[:, 0]
    if np.any(determinant == 0):
        raise ValueError("triangle mesh contains a degenerate triangle")
    delta1, delta2 = values[:, 1] - values[:, 0], values[:, 2] - values[:, 0]
    gradient = np.column_stack(((delta1*e2[:, 1] - delta2*e1[:, 1])/determinant,
                                (e1[:, 0]*delta2 - e2[:, 0]*delta1)/determinant))
    start = q.reshape(-1, 2)
    edge = (np.roll(q, -1, axis=1) - q).reshape(-1, 2)
    normal = np.column_stack((edge[:, 1], -edge[:, 0])) * np.repeat(np.sign(determinant), 3)[:, None]
    first, difference = values.ravel(), (np.roll(values, -1, axis=1) - values).ravel()
    gradients = np.repeat(gradient, 3, axis=0)
    subdivisions = np.maximum(1, np.ceil(np.abs(edge) @ (maximum*scale) / phase_step)).astype(int)
    nodes, weights = _gauss(edge_order)
    positions, strengths = [], []
    for count in np.unique(subdivisions):
        ids = np.flatnonzero(subdivisions == count)
        t = ((np.arange(count)[:, None] + nodes) / count).ravel()
        w = np.tile(weights / count, count)
        points = start[ids, None, :] + t[None, :, None] * edge[ids, None, :]
        u = first[ids, None] + difference[ids, None]*t[None, :]
        nx, ny = normal[ids, 0, None], normal[ids, 1, None]
        gx, gy = gradients[ids, 0, None], gradients[ids, 1, None]
        coefficients = np.stack((u*nx, u*ny, np.broadcast_to(gx*nx, u.shape),
                                 np.broadcast_to(gx*ny + gy*nx, u.shape),
                                 np.broadcast_to(gy*ny, u.shape))) * w[None, None, :]
        positions.append(points.reshape(-1, 2))
        strengths.append(coefficients.reshape(5, -1))
    return Contour(q, values.copy(), origin, scale, abs(determinant)/2.0,
                   np.concatenate(positions), np.ascontiguousarray(np.concatenate(strengths, axis=1)),
                   maximum, edge_order, subdivisions)


def _uniform_axis(axis):
    if len(axis) == 1:
        return 1.0, float(axis[0])
    step = float(axis[1] - axis[0])
    if step == 0 or not np.allclose(np.diff(axis), step, rtol=1e-10, atol=abs(step)*1e-10):
        return None
    return step, float(axis[len(axis)//2])


def _edge_sums(contour, kx, ky, *, backend, eps, nthreads):
    ax, ay = _uniform_axis(kx), _uniform_axis(ky)
    if backend not in {"auto", "direct", "finufft", "finufft3"}:
        raise ValueError("backend must be auto, direct, finufft or finufft3")
    finufft = None
    if backend != "direct":
        try:
            import finufft
        except ImportError:
            if backend in {"finufft", "finufft3"}:
                raise ImportError("the finufft backend requires the optional finufft package") from None
    if finufft is not None and ax is not None and ay is not None and backend != "finufft3":
        dx, x0 = ax
        dy, y0 = ay
        q = contour.points
        c = np.ascontiguousarray(contour.coefficients * np.exp(-1j*(x0*q[:, 0] + y0*q[:, 1])))
        result = finufft.nufft2d1(np.ascontiguousarray((dx*q[:, 0]+np.pi) % (2*np.pi)-np.pi),
                                 np.ascontiguousarray((dy*q[:, 1]+np.pi) % (2*np.pi)-np.pi),
                                 c, (len(kx), len(ky)), eps=eps, isign=-1, nthreads=nthreads)
        return result.transpose(0, 2, 1)
    if finufft is not None:
        wx, wy = np.meshgrid(kx, ky)
        q = contour.points
        result = finufft.nufft2d3(np.ascontiguousarray(q[:, 0]), np.ascontiguousarray(q[:, 1]),
                                 contour.coefficients, wx.ravel(), wy.ravel(), eps=eps,
                                 isign=-1, nthreads=nthreads)
        return result.reshape(5, len(ky), len(kx))
    wx, wy = np.meshgrid(kx, ky)
    frequency = np.column_stack((wx.ravel(), wy.ravel()))
    result = np.empty((5, len(frequency)), complex)
    chunk = max(1, min(256, 2_000_000 // len(contour.points)))
    for start in range(0, len(frequency), chunk):
        f = frequency[start:start + chunk]
        phase = contour.points[:, 0, None]*f[None, :, 0] + contour.points[:, 1, None]*f[None, :, 1]
        result[:, start:start + chunk] = contour.coefficients @ np.exp(-1j*phase)
    return result.reshape(5, len(ky), len(kx))


def _small_frequency(contour, frequencies):
    """Stable triangle moments near DC, where edge terms subtract strongly."""
    nodes, weights = _gauss(8)
    t, s = np.meshgrid(nodes, nodes, indexing="ij")
    barycentric = np.column_stack(((1-t).ravel()*(1-s).ravel(),
                                   t.ravel(), ((1-t)*s).ravel()))
    weights2 = (weights[:, None]*weights[None, :]*(1-t)).ravel()
    result = np.zeros(len(frequencies), complex)
    for start in range(0, len(contour.triangles), 1024):
        tri = contour.triangles[start:start + 1024]
        q = np.einsum("av,tvd->tad", barycentric, tri).reshape(-1, 2)
        u = (np.einsum("av,tv->ta", barycentric, contour.values[start:start + 1024])
             * weights2[None, :] * (2*contour.area[start:start + 1024, None])).ravel()
        for i, f in enumerate(frequencies):
            result[i] += np.sum(u*np.exp(-1j*(q @ f)))
    return result


def contour_fourier(contour, kx, ky, *, eps=1e-11, backend="auto", nthreads=1):
    """Fourier integral of the prescribed P1 field, including its gradients.

    Five contour sums implement F = i(k.B_u)/|k|² + (k.C.k)/|k|⁴.
    DC is an exact moment; small frequencies use a stable moment evaluation.
    """
    kx, ky = _axes(kx, ky)
    nthreads = _integer(nthreads, "nthreads")
    if not np.isfinite(eps) or not 0 < eps < 1:
        raise ValueError("eps must lie between zero and one")
    maximum = np.array([np.max(abs(kx)), np.max(abs(ky))])
    if np.any(maximum > contour.max_frequency*(1+1e-12) + np.finfo(float).tiny):
        raise ValueError("requested frequencies exceed the prepared quadrature bounds")
    sx, sy = kx*contour.scale, ky*contour.scale
    wx, wy = np.meshgrid(sx, sy)
    radius2 = wx**2 + wy**2
    small = abs(wx) + abs(wy) < 0.5
    result = np.empty(radius2.shape, complex)
    if np.any(~small):
        sums = _edge_sums(contour, sx, sy, backend=backend, eps=eps, nthreads=nthreads)
        safe = np.where(small, 1.0, radius2)
        result[:] = (1j*(wx*sums[0]+wy*sums[1])/safe
                     + (wx**2*sums[2]+wx*wy*sums[3]+wy**2*sums[4])/safe**2)
    if np.any(small):
        zero = radius2 == 0
        nonzero = small & ~zero
        result[zero] = np.sum(contour.area * contour.values.mean(axis=1))
        if np.any(nonzero):
            result[nonzero] = _small_frequency(contour, np.column_stack((wx[nonzero], wy[nonzero])))
    shift = np.exp(-1j*(kx[None, :]*contour.origin[0] + ky[:, None]*contour.origin[1]))
    return result * (contour.scale**2) * shift


def triangle_fourier(triangles, vertex_values, kx, ky, *, edge_order=8,
                     phase_step=4.0, eps=1e-11, backend="auto", nthreads=1):
    """Integrate a discontinuous affine complex field on triangles."""
    kx, ky = _axes(kx, ky)
    contour = prepare_contour(triangles, vertex_values,
                              max_frequency=[np.max(abs(kx)), np.max(abs(ky))],
                              edge_order=edge_order, phase_step=phase_step)
    return contour_fourier(contour, kx, ky, eps=eps, backend=backend, nthreads=nthreads)


def fresnel_from_chirped(triangles, vertex_values, *, k, distance, x, y,
                         cell_width=0.0, pixel_order=1, edge_order=8,
                         phase_step=4.0, eps=1e-11, backend="auto", nthreads=1,
                         return_stats=False):
    """Propagate P1 samples of E_exit(q) exp(i k |q|²/(2 distance)).

    This is an isolated Fresnel integral, without FFT-periodic replicas. The
    longitudinal carrier exp(i k distance) is omitted consistently everywhere.
    """
    x, y = _axes(x, y)
    if not np.isfinite([k, distance, cell_width]).all() or k <= 0 or distance <= 0 or cell_width < 0:
        raise ValueError("k and distance must be positive; cell_width nonnegative")
    pixel_order = _integer(pixel_order, "pixel_order")
    if cell_width == 0:
        nodes, weights = np.array([0.0]), np.array([1.0])
    else:
        nodes, weights = _gauss(pixel_order)
        nodes = (nodes-0.5)*cell_width
    bound = k/distance*np.array([np.max(abs(x))+np.max(abs(nodes)),
                                np.max(abs(y))+np.max(abs(nodes))])
    contour = prepare_contour(triangles, vertex_values, max_frequency=bound,
                              edge_order=edge_order, phase_step=phase_step)
    if pixel_order > 1 and cell_width > 0 and backend != "direct":
        tx, ty = (x[:, None]+nodes).ravel(), (y[:, None]+nodes).ravel()
        spectrum = contour_fourier(contour, k/distance*tx, k/distance*ty,
                                   eps=eps, backend=backend, nthreads=nthreads)
        chirp = np.exp(0.5j*k/distance*(tx[None, :]**2 + ty[:, None]**2))
        result = np.einsum("iajb,a,b->ij", (chirp*spectrum).reshape(len(y), pixel_order, len(x), pixel_order),
                           weights, weights)
        result *= k/(2j*np.pi*distance)
        stats = {**contour.stats, "pixel_order": pixel_order, "transform": "joint_nonuniform_pixel_nodes"}
        return (result, stats) if return_stats else result
    result = np.zeros((len(y), len(x)), complex)
    for dy, wy in zip(nodes, weights):
        for dx, wx in zip(nodes, weights):
            tx, ty = x+dx, y+dy
            spectrum = contour_fourier(contour, k/distance*tx, k/distance*ty,
                                       eps=eps, backend=backend, nthreads=nthreads)
            chirp = np.exp(0.5j*k/distance*(tx[None, :]**2 + ty[:, None]**2))
            result += wx*wy*chirp*spectrum
    result *= k/(2j*np.pi*distance)
    stats = {**contour.stats, "pixel_order": pixel_order, "transform": "separate_pixel_nodes"}
    return (result, stats) if return_stats else result

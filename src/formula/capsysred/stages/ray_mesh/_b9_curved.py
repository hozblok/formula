"""Curved P2 ray-tube pullback integral over entrance triangles.

Positions, real optical phase and complex pullback density are interpolated
on the entrance triangle. The density already contains the geometric measure
conversion and Fresnel/Maslov factors. No exit Jacobian division, Gaussian
window, branch merging or fitted normalization occurs here.
"""
from __future__ import annotations

from math import comb
import time

import numpy as np

from ._b9_phase import _integer, _real, _real_array
from ._b9_regular import _axis


def _curved_mesh(mesh):
    entrance = _real_array(mesh['entrance_triangles'], 'entrance_triangles')
    positions = _real_array(mesh['position_nodes'], 'position_nodes')
    phase = _real_array(mesh['phase_nodes'], 'phase_nodes')
    density = np.asarray(mesh['weight_nodes'], dtype=complex)
    if entrance.ndim != 3 or entrance.shape[1:] != (3, 2):
        raise ValueError('entrance_triangles must have shape (N,3,2)')
    count = len(entrance)
    if positions.shape != (count, 6, 2) or phase.shape != (count, 6) or density.shape != (count, 6):
        raise ValueError('curved P2 node arrays must have shapes (N,6,2), (N,6), (N,6)')
    if not np.isfinite(density).all():
        raise ValueError('weight_nodes must be finite')
    e1, e2 = entrance[:, 1]-entrance[:, 0], entrance[:, 2]-entrance[:, 0]
    determinant = e1[:, 0]*e2[:, 1]-e1[:, 1]*e2[:, 0]
    if not np.isfinite(determinant).all() or np.any(determinant == 0):
        raise ValueError('entrance triangles must be finite and nondegenerate')
    if (not np.isfinite(phase-phase[:, :1]).all()
            or not np.isfinite(positions-positions[:, :1]).all()):
        raise ValueError('curved nodal differences are not finite')
    return entrance, positions, phase, density, determinant


def _duffy_power(values):
    """P2 nodal scalar/vector values -> powers u^i v^j, s=u,t=(1-u)v."""
    p0, p1, p2, m01, m12, m20 = (values[:, i] for i in range(6))
    c01, c12, c20 = m01-(p0+p1)/2, m12-(p1+p2)/2, m20-(p2+p0)/2
    b, c = p1-p0+4*c01, p2-p0+4*c20
    d, e, f = -4*c01, 4*(c12-c01-c20), -4*c20
    result = np.zeros((len(values), 3, 3)+values.shape[2:])
    result[:, 0, 0], result[:, 1, 0], result[:, 0, 1] = p0, b, c
    result[:, 1, 1], result[:, 2, 1], result[:, 2, 0] = e-c, -e, d
    result[:, 0, 2], result[:, 1, 2], result[:, 2, 2] = f, -2*f, f
    return result


def _bernstein_matrix(degree):
    return np.array([[comb(i, j)/comb(degree, j) if j <= i else 0.
                      for j in range(degree+1)] for i in range(degree+1)])


def _derivative_bounds(positions, phase, kappa, receiver_box):
    """Convex-hull bounds of tensor Bernstein derivative coefficients.

    The complete Duffy phase has degree <=(4,4). Derivatives have degrees
    (3,4) and (4,3). Power -> Bernstein conversion on [0,1]^2 gives bounds
    for the prescribed polynomial, without inverse exit-map derivatives.
    Receiver dependence is affine, so four box corners suffice.
    """
    bounds = np.zeros((len(positions), 2))
    matrix3, matrix4 = _bernstein_matrix(3), _bernstein_matrix(4)
    for start in range(0, len(positions), 8192):
        stop = min(start+8192, len(positions))
        x, phi = positions[start:stop], phase[start:stop]
        anchor = x[:, 0]
        xp = _duffy_power(x-anchor[:, None])
        pp = _duffy_power(phi-phi[:, :1])
        base = np.zeros((len(x), 5, 5))
        base[:, :3, :3] = pp
        for i in range(3):
            for j in range(3):
                for a in range(3):
                    for b in range(3):
                        base[:, i+a, j+b] += .5*kappa*np.sum(xp[:, i, j]*xp[:, a, b], axis=1)
        local_bound = np.zeros((len(x), 2))
        for receiver in receiver_box:
            polynomial = base.copy()
            polynomial[:, :3, :3] -= kappa*np.einsum('tijc,tc->tij', xp, receiver-anchor)
            du = polynomial[:, 1:, :]*np.arange(1, 5)[None, :, None]
            dv = polynomial[:, :, 1:]*np.arange(1, 5)[None, None, :]
            bu = np.einsum('ia,tab,jb->tij', matrix3, du, matrix4, optimize=True)
            bv = np.einsum('ia,tab,jb->tij', matrix4, dv, matrix3, optimize=True)
            guard_u = 128*np.finfo(float).eps*np.sum(abs(du), axis=(1, 2))
            guard_v = 128*np.finfo(float).eps*np.sum(abs(dv), axis=(1, 2))
            local_bound[:, 0] = np.maximum(local_bound[:, 0], np.max(abs(bu), axis=(1, 2))+guard_u)
            local_bound[:, 1] = np.maximum(local_bound[:, 1], np.max(abs(bv), axis=(1, 2))+guard_v)
        bounds[start:stop] = local_bound
    if not np.isfinite(bounds).all():
        raise ValueError('curved phase derivative bounds are not finite')
    return bounds


def curved_quadrature_orders(mesh, *, k, distance, x, y, cell_width=0.,
                             min_order=8, safety=1.5, max_order=512, order_multiple=4):
    """Bound curved-phase derivatives; suggest heuristic per-triangle Gauss orders.

    Bounds enclose the real P2 optical phase plus the Fresnel phase composed
    with P2 X(q), over the Duffy square and entire receiver cell bounding box.
    Bernstein convex hull bounds are valid in real arithmetic; floating-point
    guards are not interval arithmetic. Converting B to q=min_order+safety*B/4
    remains a heuristic that requires paired-order/safety checks. The reported
    (q_u,q_v) pairs are executed with max(q_u,q_v) in both directions.
    """
    k, distance = _real(k, 'k'), _real(distance, 'distance')
    cell_width, safety = _real(cell_width, 'cell_width', zero=True), _real(safety, 'safety')
    min_order = _integer(min_order, 'min_order', 2)
    max_order, multiple = _integer(max_order, 'max_order', min_order), _integer(order_multiple, 'order_multiple')
    x, y = _real_array(x, 'x'), _real_array(y, 'y')
    if x.ndim != 1 or y.ndim != 1 or not len(x) or not len(y):
        raise ValueError('x and y must be nonempty one-dimensional arrays')
    entrance, positions, phase, density, determinant = _curved_mesh(mesh)
    kappa = k/distance
    if not np.isfinite(kappa):
        raise ValueError('k/distance is not finite')
    box = np.array([[x.min()-cell_width/2, y.min()-cell_width/2],
                    [x.max()+cell_width/2, y.min()-cell_width/2],
                    [x.min()-cell_width/2, y.max()+cell_width/2],
                    [x.max()+cell_width/2, y.max()+cell_width/2]])
    bounds = _derivative_bounds(positions, phase, kappa, box)
    floating = multiple*np.ceil((min_order+safety*bounds/4)/multiple)
    maximum = floating.max(initial=min_order)
    if not np.isfinite(floating).all() or maximum > max_order:
        raise ValueError(f'curved phase quadrature requires order {maximum:g}, greater than max_order={max_order}; do not silently truncate')
    orders = floating.astype(np.int64)
    executed = orders.max(axis=1) if len(orders) else np.empty(0, np.int64)
    unique, counts = np.unique(executed, return_counts=True)
    stats = dict(method='tensor Bernstein bounds of curved real-phase Duffy derivatives',
        triangles=len(entrance), receiver_box=box.tolist(), min_order=min_order, safety=safety,
        max_order=max_order, order_multiple=multiple,
        derivative_bound_maximum=bounds.max(axis=0, initial=0).tolist(),
        maximum_executed_order=int(executed.max()) if len(executed) else None,
        minimum_executed_order=int(executed.min()) if len(executed) else None,
        executed_isotropic_source_nodes=sum(int(q)**2*int(n) for q, n in zip(unique, counts)),
        suggested_anisotropic_source_nodes=sum(int(a)*int(b) for a, b in orders),
        groups=[dict(order=int(q), triangles=int(n), source_nodes=int(q)**2*int(n)) for q, n in zip(unique, counts)],
        polynomial_degree=[4, 4], derivative_degrees=[[3, 4], [4, 3]],
        limitation='Polynomial derivative bound plus heuristic Gauss-order selection; not a certified quadrature or physical-field error bound.')
    return dict(orders=orders, phase_derivative_bounds=bounds, stats=stats)


def _source_batches(indices, positions, phase, density, determinant, origin, kappa, order, limit):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes, weights = (nodes+1)/2, weights/2
    per_triangle = order**2
    for start in range(0, len(indices)*per_triangle, limit):
        flat = np.arange(start, min(start+limit, len(indices)*per_triangle))
        parent, node = indices[flat//per_triangle], flat % per_triangle
        i, j = node//order, node % order
        u, v = nodes[i], nodes[j]
        l0, l1, l2 = (1-u)*(1-v), u, (1-u)*v
        basis = (l0*(2*l0-1), l1*(2*l1-1), l2*(2*l2-1),
                 4*l0*l1, 4*l1*l2, 4*l2*l0)
        points = positions[parent, 0]-origin
        local_phase = np.zeros(len(parent))
        rho = basis[0]*density[parent, 0]
        for a in range(1, 6):
            points += basis[a][:, None]*(positions[parent, a]-positions[parent, 0])
            local_phase += basis[a]*(phase[parent, a]-phase[parent, 0])
            rho += basis[a]*density[parent, a]
        source_phase = local_phase+.5*kappa*np.sum(points*points, axis=1)
        coefficients = (abs(determinant[parent])*weights[i]*weights[j]*(1-u)*rho
                        *np.exp(1j*phase[parent, 0])*np.exp(1j*source_phase))
        if not np.isfinite(points).all() or not np.isfinite(coefficients).all() or not np.isfinite(source_phase).all():
            raise ValueError('curved source quadrature produced non-finite data')
        yield points, coefficients


class _Receiver:
    """Regular type-1 lattice receiver: per-Gauss-node carriers, chirps and the coherent
    square-cell mean; `add` accumulates weighted point sources (coefficients already carry
    the source phase and the half-chirp about `origin`)."""

    def __init__(self, *, k, distance, x, y, cell_width, pixel_order, eps, nthreads, channels_per_batch, origin):
        import finufft
        self.x, self.cx, self.dx, self.x_defect = _axis(x, 'x')
        self.y, self.cy, self.dy, self.y_defect = _axis(y, 'y')
        self.kappa, self.origin = k/distance, np.asarray(origin, float)
        pixel_q = pixel_order if cell_width else 1
        self.channel_count = pixel_q**2
        self.channels = min(channels_per_batch, self.channel_count)
        if cell_width:
            gauss, weights = np.polynomial.legendre.leggauss(pixel_order)
            gauss, weights = gauss*cell_width/2, weights/2
        else:
            gauss, weights = np.array([0.]), np.array([1.])
        self.offsets = np.array([(a, b) for b in gauss for a in gauss])
        self.receiver_weights = np.array([a*b for b in weights for a in weights])
        self.carrier = self.kappa*(np.array([self.cx, self.cy])-self.origin+self.offsets)
        if not np.isfinite(self.carrier).all():
            raise ValueError('receiver carriers are not finite')
        self.plan = finufft.Plan(1, (len(self.x), len(self.y)), n_trans=self.channels, eps=eps,
                                 isign=-1, dtype='complex128', nthreads=nthreads, modeord=0)
        self.effective_pixel_order = pixel_q
        self.maximum_local_position = np.zeros(2)

    def add(self, points, coefficients, field, stats):
        x, y, kappa, channels = self.x, self.y, self.kappa, self.channels
        self.maximum_local_position = np.maximum(self.maximum_local_position, abs(points).max(axis=0))
        theta = points*(kappa*np.array([self.dx, self.dy]))
        if not np.isfinite(theta).all():
            raise ValueError('type1 source coordinates are not finite')
        theta = np.remainder(theta+np.pi, 2*np.pi)-np.pi
        self.plan.setpts(np.ascontiguousarray(theta[:, 0]), np.ascontiguousarray(theta[:, 1]))
        stats['source_batches'] += 1
        stats['maximum_source_batch'] = max(stats['maximum_source_batch'], len(points))
        for begin in range(0, self.channel_count, channels):
            end = min(begin+channels, self.channel_count)
            physical_phase = self.carrier[begin:end]@points.T
            if not np.isfinite(physical_phase).all():
                raise ValueError('type1 carrier phase is not finite')
            strengths = np.zeros((channels, len(points)), complex)
            strengths[:end-begin] = coefficients*np.exp(-1j*physical_phase)
            spectrum = self.plan.execute(strengths)
            if channels == 1:
                spectrum = np.asarray(spectrum).reshape(1, len(x), len(y))
            stats['transform_calls'] += 1
            stats['padded_channel_transforms'] += channels-(end-begin)
            for local, channel in enumerate(range(begin, end)):
                rx, ry = x-self.origin[0]+self.offsets[channel, 0], y-self.origin[1]+self.offsets[channel, 1]
                chirp = .5*kappa*(rx[None, :]**2+ry[:, None]**2)
                if not np.isfinite(chirp).all():
                    raise ValueError('receiver chirp is not finite')
                field += self.receiver_weights[channel]*(kappa/(2j*np.pi))*np.exp(1j*chirp)*spectrum[local].T

    def lattice_defect_bound(self):
        return float(self.kappa*np.sum(self.maximum_local_position*[self.x_defect, self.y_defect]))


def _receiver_arguments(k, distance, cell_width, pixel_order, eps, nthreads, max_nodes_per_batch, receiver_channels_per_batch):
    k, distance = _real(k, 'k'), _real(distance, 'distance')
    cell_width, eps = _real(cell_width, 'cell_width', zero=True), _real(eps, 'eps')
    if eps >= 1:
        raise ValueError('eps must be smaller than one')
    pixel_order, nthreads = _integer(pixel_order, 'pixel_order'), _integer(nthreads, 'nthreads')
    max_nodes_per_batch = _integer(max_nodes_per_batch, 'max_nodes_per_batch')
    receiver_channels_per_batch = _integer(receiver_channels_per_batch, 'receiver_channels_per_batch')
    return k, distance, cell_width, eps, pixel_order, nthreads, max_nodes_per_batch, receiver_channels_per_batch


def curved_phase_field(mesh, *, k, distance, x, y, cell_width=0., pixel_order=4,
                        min_order=8, safety=1.5, max_order=512, eps=1e-10, nthreads=1,
                        max_nodes_per_batch=500000, receiver_channels_per_batch=4,
                        return_stats=False):
    """Integrate curved entrance-P2 tubes and coherently average native cells.

    Nodes are [v0,v1,v2,m01,m12,m20]. ``weight_nodes`` are complex pullback
    density rho(q), including geometric measure conversion and Fresnel/Maslov
    exactly once; ``phase_nodes`` contain the real optical phase. The measure
    is d^2q from entrance_triangles, so rho is not multiplied or divided by an
    exit Jacobian. Degenerate or folded exit maps are allowed algebraically;
    this integrator does not make a GO model uniform or physically exact there.

    The regular type1 lattice and per-Gauss-node receiver chirps follow the
    frozen regular helper. Distinct entrance branches sum without deduplication
    even when their exit positions coincide. No common exp(i*k*distance) is
    included, matching stage18. Quadrature and model convergence remain needed.
    """
    started = time.perf_counter()
    (k, distance, cell_width, eps, pixel_order, nthreads, max_nodes_per_batch,
     receiver_channels_per_batch) = _receiver_arguments(k, distance, cell_width, pixel_order, eps, nthreads,
                                                         max_nodes_per_batch, receiver_channels_per_batch)
    x, cx, dx, x_defect = _axis(x, 'x')
    y, cy, dy, y_defect = _axis(y, 'y')
    entrance, positions, phase, density, determinant = _curved_mesh(mesh)
    selection = curved_quadrature_orders(mesh, k=k, distance=distance, x=x, y=y,
        cell_width=cell_width, min_order=min_order, safety=safety, max_order=max_order)
    kappa = k/distance
    pixel_q = pixel_order if cell_width else 1
    channel_count = pixel_q**2
    channels = min(receiver_channels_per_batch, channel_count)
    stats = dict(method='curved P2 entrance pullback with positive Duffy integration and regular type1 receiver',
        representation='P2 exit position, real optical phase and complex pullback density on each entrance triangle',
        node_order=['v0', 'v1', 'v2', 'm01', 'm12', 'm20'],
        measure='absolute entrance determinant times positive Duffy weights; no extra exit Jacobian factor',
        receiver='coherent square-cell mean', pixel_order=pixel_order, effective_pixel_order=pixel_q,
        triangles=len(entrance), selection=selection['stats'],
        source_nodes=selection['stats']['executed_isotropic_source_nodes'],
        source_batches=0, maximum_source_batch=0, transform_calls=0, padded_channel_transforms=0,
        receiver_channels_per_batch=channels, plans=0, max_nodes_per_batch=max_nodes_per_batch,
        receiver_lattice=dict(nx=len(x), ny=len(y), center=[cx, cy], spacing=[dx, dy],
            maximum_float_lattice_defect=[x_defect, y_defect]), eps=eps, nthreads=nthreads,
        limitation='Quadrature of the prescribed curved GO exit integral; no caustic uniformization, missing-support repair, or certified physical-field error.')
    field = np.zeros((len(y), len(x)), complex)
    if not len(entrance):
        stats['seconds'] = time.perf_counter()-started
        return (field, stats) if return_stats else field
    origin = positions.min(axis=(0, 1))/2+positions.max(axis=(0, 1))/2
    stats['coordinate_origin'] = origin.tolist()
    receiver = _Receiver(k=k, distance=distance, x=x, y=y, cell_width=cell_width, pixel_order=pixel_order,
                         eps=eps, nthreads=nthreads, channels_per_batch=receiver_channels_per_batch, origin=origin)
    stats['plans'] = 1
    executed = selection['orders'].max(axis=1)
    for order in np.unique(executed):
        indices = np.flatnonzero(executed == order)
        for points, coefficients in _source_batches(indices, positions, phase, density, determinant,
                origin, kappa, int(order), max_nodes_per_batch):
            receiver.add(points, coefficients, field, stats)
    if not np.isfinite(field).all():
        raise ValueError('curved propagated field is not finite')
    stats['sampled_lattice_fourier_phase_defect_bound_rad'] = receiver.lattice_defect_bound()
    stats['seconds'] = time.perf_counter()-started
    return (field, stats) if return_stats else field


def curved_residual_fields(mesh, *, k, distance, x, y, cell_width=0., pixel_order=4, eps=1e-10, nthreads=1,
                           max_nodes_per_batch=500000, receiver_channels_per_batch=4, return_stats=False):
    """Fields of the stratified residual samples, one per batch.

    Batch b sums A_T rho(q_Tb) exp(i phi(q_Tb)) K(P, X(q_Tb)) over the unresolved
    faces T with the receiver of curved_phase_field; samples of zero density
    (non-screen fate, invalid transport) contribute nothing. The batch mean is
    the stratified estimate of the omitted integral; the spread of the batches
    is its sampling error, not a GO-model error.
    """
    started = time.perf_counter()
    (k, distance, cell_width, eps, pixel_order, nthreads, max_nodes_per_batch,
     receiver_channels_per_batch) = _receiver_arguments(k, distance, cell_width, pixel_order, eps, nthreads,
                                                         max_nodes_per_batch, receiver_channels_per_batch)
    batches = _integer(mesh.get('residual_batches', 0), 'residual_batches', 0)
    x, cx, dx, x_defect = _axis(x, 'x')
    y, cy, dy, y_defect = _axis(y, 'y')
    fields = np.zeros((batches, len(y), len(x)), complex)
    stats = dict(method='stratified residual samples propagated as weighted point sources on the regular type1 receiver',
                 batches=batches, faces=0, source_batches=0, maximum_source_batch=0, transform_calls=0,
                 padded_channel_transforms=0, plans=0, zero_weight_samples=0)
    if not batches:
        stats['seconds'] = time.perf_counter()-started
        return (fields, stats) if return_stats else fields
    area = _real_array(mesh['residual_area'], 'residual_area')
    points = _real_array(mesh['residual_points'], 'residual_points')
    phase = _real_array(mesh['residual_phase'], 'residual_phase')
    density = np.asarray(mesh['residual_density'], dtype=complex)
    count = len(area)
    if points.shape != (batches, count, 2) or phase.shape != (batches, count) or density.shape != (batches, count):
        raise ValueError('residual arrays must have shapes (B,N,2), (B,N), (B,N) for N residual faces')
    if np.any(area < 0) or not np.isfinite(density).all():
        raise ValueError('residual areas must be nonnegative and densities finite')
    stats['faces'] = count
    if not count:
        stats['seconds'] = time.perf_counter()-started
        return (fields, stats) if return_stats else fields
    kappa = k/distance
    origin = points.reshape(-1, 2).min(axis=0)/2+points.reshape(-1, 2).max(axis=0)/2
    stats['coordinate_origin'] = origin.tolist()
    receiver = _Receiver(k=k, distance=distance, x=x, y=y, cell_width=cell_width, pixel_order=pixel_order,
                         eps=eps, nthreads=nthreads, channels_per_batch=receiver_channels_per_batch, origin=origin)
    stats['plans'] = 1
    for b in range(batches):
        coefficients = area*density[b]*np.exp(1j*phase[b])
        keep = coefficients != 0
        stats['zero_weight_samples'] += int(np.sum(~keep))
        local = points[b][keep]-origin
        weights = coefficients[keep]*np.exp(.5j*kappa*np.sum(local*local, axis=1))
        for start in range(0, len(local), max_nodes_per_batch):
            receiver.add(local[start:start+max_nodes_per_batch], weights[start:start+max_nodes_per_batch], fields[b], stats)
    if not np.isfinite(fields).all():
        raise ValueError('residual propagated fields are not finite')
    stats['sampled_lattice_fourier_phase_defect_bound_rad'] = receiver.lattice_defect_bound()
    stats['seconds'] = time.perf_counter()-started
    return (fields, stats) if return_stats else fields

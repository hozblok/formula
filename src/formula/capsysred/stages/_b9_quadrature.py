"""Opt-in per-triangle source orders for the prescribed real-phase model.

Derivative bounds refer to the actual P1/P2 plus Fresnel polynomial on the
Duffy square and the entire receiver bounding box. The conversion from those
bounds to Gauss orders is a heuristic. Paired orders/safety factors must still
be tested; neither a smooth ray mesh nor NUFFT eps controls this error.
"""
from __future__ import annotations

import time

import numpy as np

from ._b9_phase import _integer, _mesh, _real, _real_array
from ._b9_regular import regular_phase_field


def _quadratic_abs_max(a, b, c):
    """Maximum of |a+b*t+c*t**2| over [0,1], elementwise."""
    a, b, c = np.broadcast_arrays(a, b, c)
    extremum = np.zeros(a.shape, float)
    np.divide(-b, 2*c, out=extremum, where=c != 0)
    inside = (c != 0) & (extremum > 0) & (extremum < 1)
    extremum = np.where(inside, extremum, 0.)
    stationary = a+extremum*(b+extremum*c)
    return np.maximum(np.maximum(abs(a), abs(a+b+c)), np.where(inside, abs(stationary), 0.))


def phase_quadrature_orders(mesh, *, k, distance, x, y, cell_width=0., phase_degree=2,
                            min_order=8, safety_factor=1., order_multiple=4,
                            max_order=512):
    """Return anisotropic order suggestions and Duffy phase-derivative bounds.

    Write s=lambda1=u and t=lambda2=(1-u)*v. Up to a constant, the full
    integrand phase is a*s+b*t+C*s**2+D*s*t+E*t**2. Its u derivative is affine
    in u and quadratic in v; the v derivative is affine in v and quadratic
    in u. Thus endpoint and stationary-point evaluation gives exact extrema
    in real arithmetic. Receiver coordinates enter affinely, so their box
    corners suffice. Bounds cover every point of the square native receiver
    cells, rather than only selected Gauss nodes. Phase differences and edge
    midpoint corrections avoid inverting ill-conditioned exit triangles.

    Orders round up ``min_order+safety_factor*bound/4`` to order_multiple.
    This is a resolution heuristic, NOT an error bound or a universal Gauss
    convergence criterion. A requested order over max_order raises, never
    silently clips. The returned pairs are currently executed isotropically
    by mixed_phase_field; anisotropic integration is not implemented.
    """
    k, distance = _real(k, 'k'), _real(distance, 'distance')
    cell_width = _real(cell_width, 'cell_width', zero=True)
    safety_factor = _real(safety_factor, 'safety_factor')
    phase_degree = _integer(phase_degree, 'phase_degree')
    min_order, order_multiple = _integer(min_order, 'min_order', 2), _integer(order_multiple, 'order_multiple')
    max_order = _integer(max_order, 'max_order', min_order)
    if phase_degree not in (1, 2):
        raise ValueError('phase_degree must be 1 or 2')
    x, y = _real_array(x, 'x'), _real_array(y, 'y')
    if x.ndim != 1 or y.ndim != 1 or not len(x) or not len(y):
        raise ValueError('x and y must be nonempty one-dimensional arrays')
    triangles, amplitude, phase, directions, determinant, correction = _mesh(mesh, phase_degree, k)
    kappa = k/distance
    if not np.isfinite(kappa):
        raise ValueError('k/distance is not finite')
    e1, e2 = triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
    c0, c1, c2 = correction.T
    C = -4*c0+.5*kappa*np.sum(e1*e1, axis=1)
    D = 4*(c1-c0-c2)+kappa*np.sum(e1*e2, axis=1)
    E = -4*c2+.5*kappa*np.sum(e2*e2, axis=1)
    offset1, offset2 = phase[:, 1]-phase[:, 0]+4*c0, phase[:, 2]-phase[:, 0]+4*c2
    box = np.array([[x.min()-cell_width/2, y.min()-cell_width/2],
                    [x.max()+cell_width/2, y.min()-cell_width/2],
                    [x.min()-cell_width/2, y.max()+cell_width/2],
                    [x.max()+cell_width/2, y.max()+cell_width/2]])
    bounds = np.zeros((len(triangles), 2))
    for receiver in box:
        delta = triangles[:, 0]-receiver
        a = offset1+kappa*np.sum(e1*delta, axis=1)
        b = offset2+kappa*np.sum(e2*delta, axis=1)
        du0 = _quadratic_abs_max(a, D-b, -2*E)
        du1 = _quadratic_abs_max(a+2*C, -b-D, np.zeros_like(C))
        dv0 = _quadratic_abs_max(b, D-b, -D)
        dv1 = _quadratic_abs_max(b+2*E, D-b-4*E, 2*E-D)
        # A small arithmetic guard; not an interval-arithmetic certificate.
        guard = 64*np.finfo(float).eps*(abs(a)+abs(b)+2*abs(C)+4*abs(D)+8*abs(E))
        bounds[:, 0] = np.maximum(bounds[:, 0], np.maximum(du0, du1)+guard)
        bounds[:, 1] = np.maximum(bounds[:, 1], np.maximum(dv0, dv1)+guard)
    if not np.isfinite(bounds).all():
        raise ValueError('Duffy phase-derivative bounds are not finite')
    floating_orders = order_multiple*np.ceil((min_order+safety_factor*bounds/4)/order_multiple)
    maximum_required = float(floating_orders.max(initial=min_order))
    if not np.isfinite(floating_orders).all() or maximum_required > max_order:
        raise ValueError(f'phase quadrature requires order {maximum_required:g}, greater than max_order={max_order}; do not silently truncate')
    orders = floating_orders.astype(np.int64)
    executed = orders.max(axis=1) if len(orders) else np.empty(0, np.int64)
    unique, counts = np.unique(executed, return_counts=True)
    node_count = sum(int(q)**2*int(n) for q, n in zip(unique, counts))
    pair_node_count = sum(int(a)*int(b) for a, b in orders)
    stats = dict(method='full real-phase Duffy derivative extrema with heuristic Gauss orders',
        phase_degree=phase_degree, triangles=len(triangles), receiver_box=box.tolist(),
        min_order=min_order, safety_factor=safety_factor, order_multiple=order_multiple, max_order=max_order,
        order_rule='round_up_multiple(min_order+safety_factor*derivative_bound/4)',
        derivative_bound_maximum=bounds.max(axis=0, initial=0).tolist(),
        suggested_anisotropic_source_nodes=pair_node_count, executed_isotropic_source_nodes=node_count,
        minimum_executed_order=int(executed.min()) if len(executed) else None,
        maximum_executed_order=int(executed.max()) if len(executed) else None,
        groups=[dict(order=int(q), triangles=int(n), source_nodes=int(q)**2*int(n)) for q, n in zip(unique, counts)],
        limitation='Bounds control derivatives of the prescribed polynomial only. Order selection is heuristic; use paired sensitivity checks. No caustic, omitted-area, or physical-field error bound.')
    return dict(orders=orders, phase_derivative_bounds=bounds, stats=stats)


def mixed_phase_field(mesh, *, k, distance, x, y, cell_width=0., pixel_order=4,
                              phase_degree=2, min_order=8, safety=1.5,
                              order_multiple=4, max_order=512, eps=1e-10, nthreads=1,
                              max_nodes_per_batch=500000, receiver_channels_per_batch=4,
                              return_stats=False):
    """Coherently sum groups of the unchanged regular-grid phase integrator.

    Pairs (q_u,q_v) are selected for each triangle, then max(q_u,q_v) is used
    on both Duffy axes. Every group uses the identical receiver coordinates,
    full physical phase convention, and coherent native-cell mean. Different
    local origins inside the unchanged helper are therefore fully restored
    before summation. No patch, amplitude, phase, or receiver is dropped.
    """
    started = time.perf_counter()
    selection = phase_quadrature_orders(mesh, k=k, distance=distance, x=x, y=y,
        cell_width=cell_width, phase_degree=phase_degree, min_order=min_order,
        safety_factor=safety, order_multiple=order_multiple, max_order=max_order)
    common = dict(k=k, distance=distance, x=x, y=y, cell_width=cell_width,
        pixel_order=pixel_order, phase_degree=phase_degree, eps=eps, nthreads=nthreads,
        max_nodes_per_batch=max_nodes_per_batch, receiver_channels_per_batch=receiver_channels_per_batch,
        return_stats=True)
    executed = selection['orders'].max(axis=1) if len(selection['orders']) else np.empty(0, np.int64)
    fields, runs = None, []
    names = ('triangles', 'vertex_phase', 'vertex_amplitude', 'vertex_directions')
    if not len(executed):
        fields, run = regular_phase_field(mesh, quadrature_order=min_order, **common)
        runs.append(run)
    else:
        for order in np.unique(executed):
            mask = executed == order
            subset = {name: np.asarray(mesh[name])[mask] for name in names if mesh.get(name) is not None}
            field, run = regular_phase_field(subset, quadrature_order=int(order), **common)
            fields = field if fields is None else fields+field
            runs.append(run)
    if not np.isfinite(fields).all():
        raise ValueError('mixed-order propagated field is not finite')
    stats = dict(method='mixed per-triangle positive Duffy orders; unchanged type1 receiver integration',
                 receiver='coherent square-cell mean', selection=selection['stats'], runs=runs,
                 seconds=time.perf_counter()-started,
                 source_nodes=sum(run['source_nodes'] for run in runs),
                 transform_calls=sum(run['transform_calls'] for run in runs),
                 limitation=selection['stats']['limitation'])
    return (fields, stats) if return_stats else fields

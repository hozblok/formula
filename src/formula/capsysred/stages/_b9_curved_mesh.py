"""MP-traced quadratic entrance charts with independent interior probes."""

from collections import Counter, defaultdict
import heapq
from time import perf_counter

import numpy as np

from ..trace import trace_ray
from ._b9_adaptive_mesh import _BudgetReached, _Partition, _TraceCache, _bore_spec, _det, _disk_partition
from ._b9_phase import _integer, _real, _real_array


_EDGES = ((0, 1), (1, 2), (2, 0))
_HOLDOUTS = np.array([[1/3, 1/3, 1/3], [.5, .25, .25], [.25, .5, .25], [.25, .25, .5]])
_METRICS = ("phase_error_rad", "density_relative_error", "geometry_phase_error_rad", "position_error_m",
            "fresnel_each_variation", "fresnel_each_probe_error", "fresnel_cumulative_variation",
            "fresnel_cumulative_probe_error", "map_jacobian_relative_error", "map_determinant_relative_error")


def p2_basis(bary):
    bary = np.asarray(bary)
    return np.concatenate((bary*(2*bary-1), np.stack([4*bary[..., i]*bary[..., j] for i,j in _EDGES], axis=-1)), axis=-1)


def p2_jacobian(entrance, positions, bary):
    """Derivative of six-node position interpolation with respect to entrance q."""
    matrix = np.stack((entrance[:, 1]-entrance[:, 0], entrance[:, 2]-entrance[:, 0]), axis=-1)
    grad = np.linalg.inv(matrix)
    grad = np.concatenate((-grad.sum(axis=1, keepdims=True), grad), axis=1)
    linear = np.einsum("tvi,tvj->tij", positions[:, :3], grad)
    result = np.broadcast_to(linear[:, None], (len(entrance), len(bary), 2, 2)).copy()
    for edge, (i, j) in enumerate(_EDGES):
        correction = positions[:, 3+edge]-(positions[:, i]+positions[:, j])/2
        term = bary[:, j][None, :, None]*grad[:, None, i]+bary[:, i][None, :, None]*grad[:, None, j]
        result += 4*correction[:, None, :, None]*term[:, :, None, :]
    return result


def pullback_density(nodes):
    """GO amplitude times absolute exit Jacobian, excluding the real OPL phase."""
    return (nodes["fresnel"]/nodes["source_distance"]*np.sqrt(nodes["uz0"]/nodes["uzexit"])
            *np.sqrt(abs(nodes["determinant"]))*np.exp(-.5j*np.pi*nodes["maslov"]))


def _receiver_box(x_bounds, y_bounds, cell_width):
    x, y = _real_array(x_bounds, "x_bounds"), _real_array(y_bounds, "y_bounds")
    width = _real_array(cell_width, "cell_width")
    if width.ndim == 0:
        width = np.repeat(width, 2)
    if x.shape != (2,) or y.shape != (2,) or x[0] > x[1] or y[0] > y[1]:
        raise ValueError("x_bounds and y_bounds must be ordered pairs")
    if width.shape != (2,) or np.any(width < 0):
        raise ValueError("cell_width must be a nonnegative scalar or pair")
    corners = np.array([[a, b] for a in (x[0]-width[0]/2, x[1]+width[0]/2)
                        for b in (y[0]-width[1]/2, y[1]+width[1]/2)])
    return corners, width


def _curved_check(cache, vertices, tolerances, corners, distance):
    q = np.asarray([cache.coordinates[i] for i in vertices])
    midpoint = [np.mean(np.asarray([cache.coordinates[v] for v in sorted((vertices[i], vertices[j]))]), axis=0)
                for i, j in _EDGES]
    holdout_q = _HOLDOUTS@q
    holdout_q[0] = np.mean(np.asarray([cache.coordinates[v] for v in sorted(vertices)]), axis=0)
    additional = cache.get(np.vstack((midpoint, holdout_q)))
    fit, probes = np.r_[vertices, additional[:3]], additional[3:]
    ids = np.r_[fit, probes]
    rows = [cache.records[i] for i in ids]
    result = dict(fit=fit, probes=probes, reasons=[], area=.5*abs(_det(q)))
    reasons = result["reasons"]
    if any(r["fate"] != "screen" for r in rows):
        reasons.append("non_screen_fate")
    for field, reason in (("reflections", "mixed_reflections"), ("maslov", "mixed_maslov")):
        if len({r[field] for r in rows}) != 1:
            reasons.append(reason)
    if any(not r["valid"] for r in rows):
        reasons.append("invalid_transport")
    determinant = np.asarray([r["determinant"] for r in rows])
    if np.any(~np.isfinite(determinant)) or np.any(abs(determinant) <= cache.floor):
        reasons.append("near_zero_jacobian")
    if len(set(np.sign(determinant))) != 1:
        reasons.append("mixed_orientation")
    if result["area"] <= 64*np.finfo(float).eps*cache.cap.bores[0]["radius"]**2:
        reasons.append("degenerate_entrance")
    if reasons:
        return result

    basis = p2_basis(_HOLDOUTS)
    x = np.asarray([r["points"] for r in rows])
    predicted_x = basis@x[:6]
    dx = predicted_x-x[6:]
    # The kernel phase difference is affine in receiver position.
    kernel_difference = cache.k/(2*distance)*(np.sum(dx*(predicted_x+x[6:]), axis=1)[:, None]-2*dx@corners.T)
    geometry_error = float(np.max(abs(kernel_difference)))
    phase = np.asarray([r["phase"] for r in rows])
    phase_error = float(np.max(abs(basis@(phase[:6]-phase[0])-(phase[6:]-phase[0]))))
    density = pullback_density({key: np.asarray([r[key] for r in rows]) for key in
        ("fresnel", "source_distance", "uz0", "uzexit", "determinant", "maslov")})
    density_error = float(np.max(abs(basis@density[:6]-density[6:]))/max(np.max(abs(density)), 1e-300))
    jacobian = p2_jacobian(q[None], x[None, :6], _HOLDOUTS)[0]
    mapped_det = np.linalg.det(jacobian)
    if np.any(~np.isfinite(mapped_det)) or np.any(abs(mapped_det) <= cache.floor):
        reasons.append("singular_p2_map")
    if np.any(np.sign(mapped_det) != np.sign(determinant[6:])):
        reasons.append("folded_p2_map")
    true_q = np.asarray([r["Q"] for r in rows[6:]])
    q_error = float(np.max(np.linalg.norm(jacobian-true_q, axis=(1, 2))/np.maximum(np.linalg.norm(true_q, axis=(1, 2)), 1e-300)))
    det_error = float(np.max(abs(mapped_det-determinant[6:])/np.maximum(abs(determinant[6:]), 1e-300)))
    factors = np.asarray([r["fresnel_each"] for r in rows])
    product = np.asarray([r["fresnel"] for r in rows])
    cumulative_variation = float(np.max(abs(product-product[:3].mean()))/max(np.max(abs(product)), 1e-300))
    cumulative_defect = float(np.max(abs(product[6:]-basis@product[:6]))/max(np.max(abs(product)), 1e-300))
    each_variation = each_defect = 0.
    if factors.shape[1]:
        scale = np.maximum(np.max(abs(factors), axis=0), 1e-300)
        each_variation = float(np.max(abs(factors-factors[:3].mean(axis=0))/scale))
        each_defect = float(np.max(abs(factors[6:]-basis@factors[:6])/scale))
    result.update(phase_error_rad=phase_error, density_relative_error=density_error,
        geometry_phase_error_rad=geometry_error, position_error_m=float(np.max(np.linalg.norm(dx, axis=1))),
        fresnel_each_variation=each_variation, fresnel_each_probe_error=each_defect,
        fresnel_cumulative_variation=cumulative_variation, fresnel_cumulative_probe_error=cumulative_defect,
        map_jacobian_relative_error=q_error, map_determinant_relative_error=det_error)
    for defect, tolerance, reason in ((phase_error, tolerances[0], "phase_probe"),
            (density_error, tolerances[1], "density_probe"), (geometry_error, tolerances[2], "geometry_phase_probe"),
            (max(each_variation, each_defect), tolerances[3], "fresnel_each"),
            (max(cumulative_variation, cumulative_defect), tolerances[3], "fresnel_cumulative")):
        if not np.isfinite(defect) or defect > tolerance:
            reasons.append(reason)
    return result


def curved_exit_mesh(source_origin_decimal, bore, z0, z1, k, delta, beta, *,
                     x_bounds, y_bounds, cell_width, distance, angles=128, radial_rings=4,
                     max_nodes=12000, max_depth=16, phase_tolerance_rad=.05,
                     density_relative_tolerance=.02, geometry_phase_tolerance_rad=.05,
                     fresnel_relative_tolerance=.05, precision=64, determinant_floor=1e-10,
                     entrance_relative_inset=2e-6):
    """Return P2 position, OPL and pullback density on accepted entrance faces.

    Ten rays per face supply six fitting nodes and four independent holdouts.
    Endpoint GO caustics remain excluded; the chart is not a uniform wave model.
    """
    started = perf_counter()
    angles, rings = _integer(angles, "angles", 8), _integer(radial_rings, "radial_rings")
    depth_limit, node_limit = _integer(max_depth, "max_depth", 0), _integer(max_nodes, "max_nodes", 3)
    precision = _integer(precision, "precision", 32)
    k, distance = _real(k, "k"), _real(distance, "distance")
    floor, delta, beta = (_real(v, name, zero=True) for v, name in
                           ((determinant_floor, "determinant_floor"), (delta, "delta"), (beta, "beta")))
    inset = _real(entrance_relative_inset, "entrance_relative_inset")
    tolerances = tuple(_real(v, name) for v, name in zip(
        (phase_tolerance_rad, density_relative_tolerance, geometry_phase_tolerance_rad, fresnel_relative_tolerance),
        ("phase_tolerance_rad", "density_relative_tolerance", "geometry_phase_tolerance_rad", "fresnel_relative_tolerance")))
    corners, width = _receiver_box(x_bounds, y_bounds, cell_width)
    source = np.asarray(source_origin_decimal, float)
    if (inset >= .01 or source.shape != (3,) or not np.isfinite(source).all()
            or not np.isfinite([float(z0), float(z1)]).all() or not source[2] < float(z0) < float(z1)):
        raise ValueError("finite source before entrance, positive length and inset below .01 required")
    regular, mp = _bore_spec(bore, precision)
    cache = _TraceCache(source_origin_decimal, regular, mp, z0, z1, k, delta, beta, precision, floor, node_limit)
    q, triangles = _disk_partition(np.asarray(regular["center"]), regular["radius"], angles, rings, inset)
    if len(q) > node_limit:
        raise ValueError("max_nodes must fit the initial polar vertices")
    vertices = cache.get(q)
    partition = _Partition(vertices[triangles])
    failed = []
    budget_reached = False
    evaluations = bisections = closure_splits = 0

    def evaluate(index):
        nonlocal evaluations, budget_reached
        leaf = partition.leaves[index]
        try:
            checked = _curved_check(cache, leaf["vertices"], tolerances, corners, distance)
        except _BudgetReached:
            checked = dict(reasons=["node_budget"], probes=None, fit=None,
                area=.5*abs(_det(np.asarray([cache.coordinates[v] for v in leaf["vertices"]]))))
            budget_reached = True
        leaf.update(checked)
        evaluations += 1
        if leaf["reasons"]:
            heapq.heappush(failed, (-leaf["area"], index))

    for index in list(partition.leaves):
        evaluate(index)
    while failed and not budget_reached:
        _, index = heapq.heappop(failed)
        if index not in partition.leaves:
            continue
        leaf = partition.leaves[index]
        v = leaf["vertices"]
        edge = max((tuple(sorted((v[i], v[j]))) for i, j in _EDGES),
                   key=lambda ij: sum((np.asarray(cache.coordinates[ij[0]])-cache.coordinates[ij[1]])**2))
        neighbours = partition.edges[edge]
        if any(partition.leaves[i]["depth"] >= depth_limit for i in neighbours):
            leaf["reasons"] = list(dict.fromkeys([*leaf["reasons"], "max_depth"]))
            continue
        midpoint = cache.get([np.mean(np.asarray([cache.coordinates[i] for i in edge]), axis=0)])[0]
        closure_splits += len(neighbours)-1
        children = partition.split(edge, int(midpoint))
        bisections += 1
        for child in children:
            evaluate(child)

    leaves = list(partition.leaves.values())
    faces = np.asarray([leaf["vertices"] for leaf in leaves], np.int64)
    accepted = np.array([not leaf["reasons"] for leaf in leaves], bool)
    all_q = np.asarray(cache.coordinates)[faces]
    a, b = all_q[:, 1]-all_q[:, 0], all_q[:, 2]-all_q[:, 0]
    areas = .5*abs(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0])
    fit_ids = np.asarray([leaf["fit"] if leaf.get("fit") is not None else [-1]*6 for leaf in leaves])
    nodes = cache.arrays()
    density = pullback_density(nodes)
    aperture_area = np.pi*regular["radius"]**2
    polygon_area = angles/2*(regular["radius"]*(1-inset))**2*np.sin(2*np.pi/angles)
    missing = aperture_area-polygon_area
    primary, by_reason = defaultdict(float), defaultdict(float)
    for leaf, area in zip(leaves, areas):
        if leaf["reasons"]:
            primary[leaf["reasons"][0]] += float(area)
            for reason in leaf["reasons"]:
                by_reason[reason] += float(area)
    maxima = {name: max((leaf.get(name, 0.) for leaf, ok in zip(leaves, accepted) if ok), default=0.) for name in _METRICS}
    diagnostic = dict(status="controls_partial" if not accepted.all() else "probe_controls_complete",
        source_origin_decimal=cache.source_strings, bore=regular, z0=float(z0), z1=float(z1), k=k, delta=delta, beta=beta,
        parameters=dict(angles=angles, radial_rings=rings, max_nodes=node_limit, max_depth=depth_limit,
            precision=precision, determinant_floor=floor, entrance_relative_inset=inset,
            phase_tolerance_rad=tolerances[0], density_relative_tolerance=tolerances[1],
            geometry_phase_tolerance_rad=tolerances[2], fresnel_relative_tolerance=tolerances[3],
            x_bounds=list(map(float, x_bounds)), y_bounds=list(map(float, y_bounds)), cell_width=width.tolist(), distance=distance),
        trace=dict(emitted_nodes=len(cache.records), screen_nodes=sum(r["fate"] == "screen" for r in cache.records),
            fate_counts=dict(Counter(r["fate"] for r in cache.records)),
            reflection_histogram=dict(Counter(str(r["reflections"]) for r in cache.records)),
            backend="python multiprecision" if cache.tracer is trace_ray else "native multiprecision", precision=precision,
            trace_seconds=cache.tracing_seconds, transport_seconds=cache.transport_seconds,
            maximum_recovered_entrance_defect_m=cache.recovery_maximum),
        topology=dict(partition_triangles=len(leaves), accepted_triangles=int(accepted.sum()),
            unresolved_triangles=int((~accepted).sum()), bisections=bisections, closure_neighbour_splits=closure_splits,
            maximum_depth=max(leaf["depth"] for leaf in leaves), evaluations=evaluations,
            conforming_entrance_partition=True, maximum_edge_incidence=max(map(len, partition.edges.values()))),
        coverage=dict(aperture_area_m2=aperture_area, polygon_area_m2=polygon_area,
            accepted_area_m2=float(areas[accepted].sum()), accepted_area_fraction=float(areas[accepted].sum()/aperture_area),
            unresolved_area_m2=float(areas[~accepted].sum()), unresolved_area_fraction=float(areas[~accepted].sum()/aperture_area),
            outer_chord_and_inset_deficit_m2=missing, outer_chord_and_inset_deficit_fraction=missing/aperture_area,
            area_closure_relative=float((areas.sum()+missing-aperture_area)/aperture_area),
            unresolved_primary_reason_area_m2=dict(primary), unresolved_each_reason_area_m2=dict(by_reason),
            reason_convention="Primary reasons are disjoint; each-reason areas overlap and cannot be added."),
        accepted_probe_maxima=maxima, budget_reached=budget_reached, seconds=perf_counter()-started,
        limitations=["Four interior holdouts are independent of the six fitting nodes; finite probes do not certify unsampled structure.",
            "The kernel geometry defect is tested over the whole requested receiver bounding box including native cell widths.",
            "The outer polygon deficit and unresolved entrance faces contribute no field and remain explicitly reported.",
            "The square-root absolute Jacobian density avoids division at a fold, but invalid/near-zero endpoint Jacobians remain excluded.",
            "This change of coordinates is not a uniform caustic solution; collapsed whole families and GO modeling error are not repaired.",
            "P2 map derivative signs are checked at independent probes; global injectivity of the polynomial chart is not certified."])
    metadata = dict(triangles=int(accepted.sum()), representation="quadratic_entrance_pullback",
        total_entrance_area_m2=aperture_area, accepted_entrance_area_m2=float(areas[accepted].sum()),
        accepted_entrance_area_fraction=float(areas[accepted].sum()/aperture_area),
        accepted_entrance_area_by_bore_m2=[float(areas[accepted].sum())], curved_controls=diagnostic,
        node_order=["v0", "v1", "v2", "m01", "m12", "m20"],
        density_convention="Fresnel/r*sqrt(uz0/uzexit)*sqrt(abs(detQ))*exp(-i*pi*Maslov/2); excludes exp(i*OPLphase); measure d2q.")
    ids = fit_ids[accepted]
    mesh = dict(entrance_triangles=all_q[accepted], position_nodes=nodes["points"][ids],
        phase_nodes=nodes["phase"][ids], weight_nodes=density[ids], entrance_area=areas[accepted],
        ray_indices=faces[accepted], fit_indices=ids, triangle_reflections=nodes["reflections"][faces[accepted, 0]],
        metadata=metadata, partition_entrance_triangles=all_q, partition_ray_indices=faces,
        partition_accepted=accepted, partition_reasons=np.asarray([";".join(leaf["reasons"]) for leaf in leaves]),
        partition_depth=np.asarray([leaf["depth"] for leaf in leaves]), partition_fit_indices=fit_ids,
        partition_probe_indices=np.asarray([leaf["probes"] if leaf["probes"] is not None else [-1]*4 for leaf in leaves]),
        traced_entrance_points=np.asarray(cache.coordinates), trace_nodes=nodes,
        partition_probe_metrics={name: np.asarray([leaf.get(name, np.nan) for leaf in leaves]) for name in _METRICS})
    return mesh, diagnostic

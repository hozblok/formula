"""Conforming entrance partitions with cached MP traces and four ray probes."""

from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal, localcontext
import heapq
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from ..native import make_tracer
from ..shared.nums import lift, vsub, vunit
from ..surfaces import CapillaryBundle
from ..trace import trace_ray
from ._b9_archive import apply_tube_flux, exit_nodes, mesh_from_triangles, _p1_triangle_power
from ._b9_phase import _integer, _real
from .stage17 import _fresnel_product


_EDGES = ((0, 1), (1, 2), (2, 0))
_PROBES = np.array([[.5, .5, 0.], [0., .5, .5], [.5, 0., .5], [1/3, 1/3, 1/3]])
_FIELDS = ("points", "phase", "amplitude", "directions", "entrance", "bore", "Q", "P",
           "determinant", "maslov", "valid", "reflections", "ray_ids", "fresnel",
           "source_distance", "uz0", "uzexit")


class _BudgetReached(Exception):
    pass


def _bore_spec(bore, precision):
    if not isinstance(bore, dict) or bore.keys()-{"kind", "center", "radius", "bend"}:
        raise ValueError("adaptive mesh supports straight cylinders and circular torus bores only")
    kind = bore.get("kind", "torus" if bore.get("bend") else "cylinder")
    if kind == "cylinder" and bore.get("bend"):
        kind = "torus"
    if kind not in ("cylinder", "torus"):
        raise ValueError("adaptive mesh supports straight cylinders and circular torus bores only")
    center = np.asarray(bore.get("center"), float)
    radius = _real(float(bore.get("radius", 0)), "bore.radius")
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("bore.center must have two finite coordinates")
    regular = dict(kind=kind, center=center.tolist(), radius=radius)
    mp = dict(kind=kind, center=tuple(lift(str(v), precision) for v in bore["center"]),
              radius=lift(str(bore["radius"]), precision))
    if kind == "torus":
        bend = bore.get("bend", {})
        toward = np.asarray(bend.get("toward"), float)
        bend_radius = _real(float(bend.get("radius", 0)), "bend.radius")
        if toward.shape != (2,) or not np.isfinite(toward).all() or not np.linalg.norm(toward):
            raise ValueError("bend.toward must be a finite nonzero transverse vector")
        if bend_radius <= radius:
            raise ValueError("torus major radius must exceed bore radius")
        regular["bend"] = dict(radius=bend_radius, toward=toward.tolist())
        mp["bend"] = dict(radius=lift(str(bend["radius"]), precision),
                           toward=tuple(lift(str(v), precision) for v in bend["toward"]))
    return regular, mp


def _disk_partition(center, radius, angles, rings, inset):
    theta = np.arange(angles)*2*np.pi/angles
    unit = np.column_stack((np.cos(theta), np.sin(theta)))
    radii = radius*(1-inset)*np.arange(1, rings+1)/rings
    q = np.vstack((center, (center+radii[:, None, None]*unit).reshape(-1, 2)))
    index = lambda ring, angle: 1+(ring-1)*angles+angle % angles
    faces = [[0, index(1, j), index(1, j+1)] for j in range(angles)]
    for ring in range(1, rings):
        for j in range(angles):
            low, high = index(ring, j), index(ring+1, j)
            faces.extend(([low, high, index(ring+1, j+1)], [low, index(ring+1, j+1), index(ring, j+1)]))
    return q, np.asarray(faces, np.int64)


def _det(t):
    a, b = t[1]-t[0], t[2]-t[0]
    return a[0]*b[1]-a[1]*b[0]


class _TraceCache:
    def __init__(self, source, bore, mp_bore, z0, z1, k, delta, beta, precision, floor, maximum):
        self.source_strings = [str(v) for v in source]
        self.source = np.asarray(self.source_strings, float)
        self.precision, self.k, self.delta, self.beta = precision, k, delta, beta
        self.floor, self.maximum = floor, maximum
        self.cap = SimpleNamespace(z0=float(z0), z1=float(z1), bores=[bore])
        self.optic = CapillaryBundle([mp_bore], lift(str(z0), precision), lift(str(z1), precision))
        self.tracer = make_tracer(self.optic)
        self.origin = tuple(lift(v, precision) for v in self.source_strings)
        with localcontext() as context:
            context.prec = max(100, precision+30)
            self.axial = Decimal(str(z1))-Decimal(self.source_strings[2])
        self.lookup, self.records, self.coordinates = {}, [], []
        self.tracing_seconds = self.transport_seconds = 0.
        self.transport_counts, self.transport_maxima = Counter(), {}
        self.recovery_maximum = 0.

    def get(self, coordinates):
        keys = [tuple(map(float, q)) for q in coordinates]
        needed = list(dict.fromkeys(key for key in keys if key not in self.lookup))
        if len(self.records)+len(needed) > self.maximum:
            raise _BudgetReached
        if needed:
            self._trace(needed)
        return np.asarray([self.lookup[key] for key in keys], np.int64)

    def _trace(self, keys):
        started = perf_counter()
        rays, phases, paths, sins = [], [], [], []
        with localcontext() as context:
            context.prec = max(100, self.precision+30)
            for q in keys:
                target = (lift(q[0], self.precision), lift(q[1], self.precision), self.optic.z0)
                ray = self.tracer(self.origin, vunit(vsub(target, self.origin)), self.optic, self.optic.z1, 200)
                rays.append(ray)
                phases.append(self.k*float(Decimal(str(ray.opl))-self.axial))
                paths.append([[float(v) for v in hit] for hit, _ in ray.reflections])
                sins.append([float(s) for _, s in ray.reflections])
        self.tracing_seconds += perf_counter()-started
        good = np.array([i for i, ray in enumerate(rays) if ray.fate == "screen"], int)
        nodes = None
        started = perf_counter()
        if len(good):
            mode = dict(origin=self.source, points=np.array([[float(v) for v in rays[i].point[:2]] for i in good]),
                        directions=np.array([[float(v) for v in rays[i].direction[:2]] for i in good]),
                        phase_opl=np.asarray(phases)[good], refl=[paths[i] for i in good],
                        ray_ids=good+len(self.records))
            nodes = exit_nodes(mode, self.cap, mode_z=self.cap.z1, exit_z=self.cap.z1, k=self.k,
                fresnel=_fresnel_product([sins[i] for i in good], self.delta, self.beta), determinant_floor=self.floor)
            defect = np.max(np.linalg.norm(nodes["entrance"]-np.asarray(keys)[good], axis=1))
            self.recovery_maximum = max(self.recovery_maximum, float(defect))
            if defect > 1e-8*self.cap.bores[0]["radius"]:
                raise ValueError("MP trace does not recover prescribed entrance coordinates")
            for key, value in nodes["transport"].items():
                if key.endswith("_count"):
                    self.transport_counts[key] += value
                elif "max" in key:
                    self.transport_maxima[key] = max(self.transport_maxima.get(key, 0.), value)
        self.transport_seconds += perf_counter()-started
        node_index = {int(original): j for j, original in enumerate(good)}
        for i, (q, ray) in enumerate(zip(keys, rays)):
            index = len(self.records)
            if i in node_index:
                record = {key: nodes[key][node_index[i]] for key in _FIELDS}
            else:
                record = dict(points=np.array([float(v) for v in ray.point[:2]]), phase=phases[i],
                    amplitude=0j, directions=np.zeros(2), entrance=np.asarray(q), bore=0,
                    Q=np.full((2, 2), np.nan), P=np.full((2, 2), np.nan), determinant=np.nan,
                    maslov=0, valid=False, reflections=len(paths[i]), ray_ids=index, fresnel=0j,
                    source_distance=np.linalg.norm(np.r_[q, self.cap.z0]-self.source), uz0=1., uzexit=1.)
            s = np.asarray(sins[i], float)
            root = np.sqrt(s*s-2*self.delta+2j*self.beta)
            record.update(entrance=np.asarray(q), fate=ray.fate,
                          fresnel_each=(s-root)/(s+root) if len(s) else np.empty(0, complex))
            self.lookup[q] = index
            self.coordinates.append(q)
            self.records.append(record)

    def arrays(self):
        result = {key: np.asarray([row[key] for row in self.records]) for key in _FIELDS}
        result.update(k=self.k, exit_z=self.cap.z1, determinant_floor=self.floor,
                      transport=dict(self.transport_counts, **self.transport_maxima))
        return result


class _Partition:
    def __init__(self, triangles):
        self.leaves, self.edges, self.serial = {}, defaultdict(set), 0
        for triangle in triangles:
            self.add(tuple(map(int, triangle)), 0)

    def add(self, vertices, depth):
        index, self.serial = self.serial, self.serial+1
        self.leaves[index] = dict(vertices=vertices, depth=depth, reasons=["not_checked"], probes=None)
        for i, j in _EDGES:
            self.edges[tuple(sorted((vertices[i], vertices[j])))].add(index)
        return index

    def split(self, edge, midpoint):
        parents = list(self.edges[edge])
        children = []
        for parent in parents:
            old = self.leaves.pop(parent)
            v = old["vertices"]
            for i, j in _EDGES:
                key = tuple(sorted((v[i], v[j])))
                self.edges[key].remove(parent)
                if not self.edges[key]:
                    del self.edges[key]
            for i, j in _EDGES:
                if tuple(sorted((v[i], v[j]))) == edge:
                    opposite = v[3-i-j]
                    children.extend((self.add((v[i], midpoint, opposite), old["depth"]+1),
                                     self.add((midpoint, v[j], opposite), old["depth"]+1)))
                    break
        return children


def _check(cache, vertices, tolerances, amplitude_mode):
    q = np.asarray([cache.coordinates[i] for i in vertices])
    # Sorted vertex arithmetic makes shared midpoint cache keys identical.
    probe_q = [np.mean(np.asarray([cache.coordinates[v] for v in sorted((vertices[i], vertices[j]))]), axis=0)
               for i, j in _EDGES]
    probe_q.append(np.mean(np.asarray([cache.coordinates[v] for v in sorted(vertices)]), axis=0))
    probes = cache.get(probe_q)
    rows = [cache.records[i] for i in (*vertices, *probes)]
    result = dict(probes=probes, reasons=[], area=.5*abs(_det(q)))
    reasons = result["reasons"]
    if any(r["fate"] != "screen" for r in rows):
        reasons.append("non_screen_fate")
    for field, reason in (("reflections", "mixed_reflections"), ("maslov", "mixed_maslov")):
        if len({r[field] for r in rows}) != 1:
            reasons.append(reason)
    if any(not r["valid"] for r in rows):
        reasons.append("invalid_transport")
    determinants = np.asarray([r["determinant"] for r in rows])
    if np.any(np.isfinite(determinants) & (abs(determinants) <= cache.floor)):
        reasons.append("near_zero_jacobian")
    if len(set(np.sign(determinants))) != 1:
        reasons.append("mixed_orientation")
    x = np.asarray([r["points"] for r in rows])
    determinant = _det(x[:3])
    threshold = 128*np.finfo(float).eps*cache.cap.bores[0]["radius"]**2
    if not np.isfinite(determinant) or abs(determinant) <= threshold or 2*result["area"] <= threshold:
        reasons.append("degenerate_exit")
    elif np.sign(determinant*_det(q)) != np.sign(determinants[0]):
        reasons.append("folded_exit")
    if reasons:
        return result

    jacobians = np.asarray([r["Q"] for r in rows])
    sigma_min = float(np.min(np.linalg.svd(jacobians, compute_uv=False)))
    q_variation = float(np.max(np.linalg.norm(jacobians-jacobians[:3].mean(axis=0), axis=(1, 2))))
    predicted_area = result["area"]*float(np.mean(abs(determinants[:3])))
    result.update(minimum_jacobian_singular_value=sigma_min,
                  jacobian_variation_over_minimum_singular=q_variation/max(sigma_min, 1e-300),
                  triangle_area_jacobian_relative_defect=abs(.5*abs(determinant)-predicted_area)/max(predicted_area, 1e-300))
    edge_matrix = (x[1:3]-x[0]).T
    uv = np.linalg.solve(edge_matrix, (x[3:]-x[0]).T).T
    bary = np.column_stack((1-uv.sum(axis=1), uv))
    phase = np.asarray([r["phase"] for r in rows])
    directions = np.asarray([r["directions"] for r in rows[:3]])
    corrections = np.array([-cache.k/8*np.dot(directions[j]-directions[i], x[j]-x[i]) for i, j in _EDGES])
    predicted = bary[:, 1]*(phase[1]-phase[0])+bary[:, 2]*(phase[2]-phase[0])
    for index, (i, j) in enumerate(_EDGES):
        predicted += 4*corrections[index]*bary[:, i]*bary[:, j]
    phase_error = float(np.max(abs(predicted-(phase[3:]-phase[0]))))
    amplitude = np.asarray([r["amplitude"] for r in rows])
    vertex_amplitude = amplitude[:3].copy()
    incoming = np.array([r["fresnel"]*np.sqrt(r["uz0"])/r["source_distance"] for r in rows[:3]])
    outgoing = vertex_amplitude*np.sqrt([r["uzexit"] for r in rows[:3]])
    pin = _p1_triangle_power(incoming[None], np.array([result["area"]]))[0]
    pout = _p1_triangle_power(outgoing[None], np.array([.5*abs(determinant)]))[0]
    flux_error = float(abs(pout-pin)/max(pin, 1e-300))
    if amplitude_mode == "tube_flux":
        if pout <= 0 < pin:
            reasons.append("zero_exit_flux")
            return result
        if pout > 0:
            vertex_amplitude *= np.sqrt(pin/pout)
    amp_error = float(np.max(abs(bary@vertex_amplitude-amplitude[3:]))/max(np.max(abs(amplitude)), 1e-300))
    geometry_error = float(np.max(np.linalg.norm(x[3:]-_PROBES@x[:3], axis=1))/cache.cap.bores[0]["radius"])
    factors = np.asarray([r["fresnel_each"] for r in rows])
    product = np.asarray([r["fresnel"] for r in rows])
    cumulative_variation = float(np.max(abs(product-product[:3].mean()))/max(np.max(abs(product)), 1e-300))
    cumulative_defect = float(np.max(abs(product[3:]-_PROBES@product[:3]))/max(np.max(abs(product)), 1e-300))
    each_variation = each_defect = 0.
    if factors.shape[1]:
        scale = np.maximum(np.max(abs(factors), axis=0), 1e-300)
        each_variation = float(np.max(abs(factors-factors[:3].mean(axis=0))/scale))
        each_defect = float(np.max(abs(factors[3:]-_PROBES@factors[:3])/scale))
    result.update(phase_error_rad=phase_error, amplitude_relative_error=amp_error,
                  geometry_relative_error=geometry_error, fresnel_each_variation=each_variation,
                  fresnel_each_probe_error=each_defect, fresnel_cumulative_variation=cumulative_variation,
                  fresnel_cumulative_probe_error=cumulative_defect, point_flux_relative_defect=flux_error)
    for defect, tolerance, reason in ((phase_error, tolerances[0], "phase_probe"),
            (amp_error, tolerances[1], "amplitude_probe"), (geometry_error, tolerances[2], "geometry_probe"),
            (max(each_variation, each_defect), tolerances[3], "fresnel_each"),
            (max(cumulative_variation, cumulative_defect), tolerances[3], "fresnel_cumulative"),
            (flux_error, tolerances[4], "point_flux")):
        if not np.isfinite(defect) or defect > tolerance:
            reasons.append(reason)
    return result


def adaptive_exit_mesh(source_origin_decimal, bore, z0, z1, k, delta, beta, *,
                       angles=128, radial_rings=4, max_depth=16, max_nodes=12000,
                       phase_tolerance_rad=.05, amplitude_relative_tolerance=.05,
                       geometry_relative_tolerance=.002, fresnel_relative_tolerance=.05,
                       flux_relative_tolerance=.05,
                       precision=64, determinant_floor=1e-10, amplitude_mode="point_jacobian",
                       entrance_relative_inset=2e-6):
    """Return accepted exit elements and an explicit complete/partial audit.

    The returned full entrance partition includes unresolved faces. Finite probes
    and area accounting are diagnostics, not certified physical field bounds.
    """
    started = perf_counter()
    angles, rings = _integer(angles, "angles", 8), _integer(radial_rings, "radial_rings")
    depth_limit, node_limit = _integer(max_depth, "max_depth", 0), _integer(max_nodes, "max_nodes", 3)
    precision = _integer(precision, "precision", 32)
    k, floor = _real(k, "k"), _real(determinant_floor, "determinant_floor", zero=True)
    delta, beta = _real(delta, "delta", zero=True), _real(beta, "beta", zero=True)
    inset = _real(entrance_relative_inset, "entrance_relative_inset")
    tolerances = tuple(_real(v, name) for v, name in zip(
        (phase_tolerance_rad, amplitude_relative_tolerance, geometry_relative_tolerance, fresnel_relative_tolerance,
         flux_relative_tolerance),
        ("phase_tolerance_rad", "amplitude_relative_tolerance", "geometry_relative_tolerance", "fresnel_relative_tolerance",
         "flux_relative_tolerance")))
    if inset >= .01 or amplitude_mode not in ("point_jacobian", "tube_flux"):
        raise ValueError("inset must be below .01; amplitude_mode must be point_jacobian or tube_flux")
    source = np.asarray(source_origin_decimal, float)
    if (source.shape != (3,) or not np.isfinite(source).all() or not np.isfinite([float(z0), float(z1)]).all()
            or not source[2] < float(z0) < float(z1)):
        raise ValueError("finite source before entrance and positive cylinder length required")
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
            checked = _check(cache, leaf["vertices"], tolerances, amplitude_mode)
        except _BudgetReached:
            checked = dict(reasons=["node_budget"], probes=None,
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
    areas = .5*np.abs(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0])
    nodes = cache.arrays()
    if accepted.any():
        mesh = mesh_from_triangles(nodes, cache.cap, faces[accepted])
        if amplitude_mode == "tube_flux":
            mesh = apply_tube_flux(mesh, nodes)
    else:
        mesh = dict(triangles=np.empty((0, 3, 2)), entrance_triangles=np.empty((0, 3, 2)),
                    vertex_phase=np.empty((0, 3)), vertex_amplitude=np.empty((0, 3), complex),
                    vertex_directions=np.empty((0, 3, 2)), ray_indices=np.empty((0, 3), np.int64),
                    triangle_reflections=np.empty(0, int), entrance_area=np.empty(0), exit_area=np.empty(0), metadata={})
    aperture_area = np.pi*regular["radius"]**2
    polygon_area = angles/2*(regular["radius"]*(1-inset))**2*np.sin(2*np.pi/angles)
    missing = aperture_area-polygon_area
    by_reason, primary = defaultdict(float), defaultdict(float)
    for leaf, area in zip(leaves, areas):
        if leaf["reasons"]:
            primary[leaf["reasons"][0]] += float(area)
            for reason in leaf["reasons"]:
                by_reason[reason] += float(area)
    unresolved = float(areas[~accepted].sum())
    metrics = ("phase_error_rad", "amplitude_relative_error", "geometry_relative_error", "fresnel_each_variation",
               "fresnel_each_probe_error", "fresnel_cumulative_variation", "fresnel_cumulative_probe_error",
               "jacobian_variation_over_minimum_singular", "triangle_area_jacobian_relative_defect", "point_flux_relative_defect")
    maxima = {name: max((leaf.get(name, 0.) for leaf, ok in zip(leaves, accepted) if ok), default=0.) for name in metrics}
    diagnostic = dict(status="controls_partial" if not accepted.all() else "probe_controls_complete",
        source_origin_decimal=cache.source_strings, bore=regular, z0=float(z0), z1=float(z1), k=k, delta=delta, beta=beta,
        parameters=dict(angles=angles, radial_rings=rings, max_depth=depth_limit, max_nodes=node_limit,
            precision=precision, determinant_floor=floor, amplitude_mode=amplitude_mode,
            entrance_relative_inset=inset, phase_tolerance_rad=tolerances[0], amplitude_relative_tolerance=tolerances[1],
            geometry_relative_tolerance=tolerances[2], fresnel_relative_tolerance=tolerances[3],
            flux_relative_tolerance=tolerances[4]),
        trace=dict(emitted_nodes=len(cache.records), screen_nodes=sum(r["fate"] == "screen" for r in cache.records),
            fate_counts=dict(Counter(r["fate"] for r in cache.records)),
            reflection_histogram=dict(Counter(str(r["reflections"]) for r in cache.records)),
            backend="python multiprecision" if cache.tracer is trace_ray else "native multiprecision", precision=precision,
            trace_seconds=cache.tracing_seconds, transport_seconds=cache.transport_seconds,
            maximum_recovered_entrance_defect_m=cache.recovery_maximum),
        topology=dict(partition_triangles=len(leaves), accepted_triangles=int(accepted.sum()),
            unresolved_triangles=int((~accepted).sum()), bisections=bisections, closure_neighbour_splits=closure_splits,
            maximum_depth=max(leaf["depth"] for leaf in leaves), evaluations=evaluations,
            conforming_entrance_partition=True, maximum_edge_incidence=max(map(len, partition.edges.values())),
            rule="Longest-edge bisection of a failing face and its shared-edge neighbour; no hidden deletion or overlapping entrance patches."),
        coverage=dict(aperture_area_m2=aperture_area, polygon_area_m2=polygon_area,
            accepted_area_m2=float(areas[accepted].sum()), accepted_area_fraction=float(areas[accepted].sum()/aperture_area),
            unresolved_area_m2=unresolved, unresolved_area_fraction=unresolved/aperture_area,
            outer_chord_and_inset_deficit_m2=missing, outer_chord_and_inset_deficit_fraction=missing/aperture_area,
            area_closure_relative=float((areas.sum()+missing-aperture_area)/aperture_area),
            unresolved_primary_reason_area_m2=dict(primary), unresolved_each_reason_area_m2=dict(by_reason),
            reason_convention="Primary reasons form a disjoint partition; each-reason areas overlap and must not be added."),
        accepted_probe_maxima=maxima,
        accepted_minimum_jacobian_singular_value=min((leaf["minimum_jacobian_singular_value"] for leaf, ok in zip(leaves, accepted) if ok), default=None),
        budget_reached=budget_reached, seconds=perf_counter()-started,
        limitations=["Three midpoint probes and a centroid can miss unsampled interior structure; not a certified error bound.",
                     "The outer polygon deficit remains when only interior triangles are refined.",
                     "Unresolved entrance faces contribute no field; their area is retained explicitly and does not bound the missing field.",
                     "GO endpoint caustics remain unresolved; invalid nodes are never divided by a vanishing Jacobian.",
                     "P2 phase and affine complex amplitude are checked at actual probe exit points; curved-edge geometry is an independent defect.",
                     "Fresnel checks cover each reflection and the cumulative product on seven sampled rays, not every ray in the element.",
                     "Point-flux consistency uses P1 mass matrices before optional amplitude correction; it is not a bound on coherent field error.",
                     "Optional tube_flux rescales each patch separately; branch flux is not total coherent power."])
    mesh["metadata"].update(adaptive_controls=diagnostic, triangles=int(accepted.sum()),
        total_entrance_area_m2=aperture_area, accepted_entrance_area_m2=float(areas[accepted].sum()),
        accepted_entrance_area_fraction=float(areas[accepted].sum()/aperture_area),
        accepted_entrance_area_by_bore_m2=[float(areas[accepted].sum())])
    mesh.update(partition_entrance_triangles=all_q, partition_ray_indices=faces, partition_accepted=accepted,
                partition_reasons=np.asarray([";".join(leaf["reasons"]) for leaf in leaves]),
                partition_depth=np.asarray([leaf["depth"] for leaf in leaves]),
                partition_probe_indices=np.asarray([leaf["probes"] if leaf["probes"] is not None else [-1]*4 for leaf in leaves]),
                traced_entrance_points=np.asarray(cache.coordinates), trace_nodes=nodes,
                partition_probe_metrics={name: np.asarray([leaf.get(name, np.nan) for leaf in leaves])
                    for name in (*metrics, "minimum_jacobian_singular_value")})
    return mesh, diagnostic


def audit_amplitude_probes(mesh, *, tolerance=None):
    """Check the returned amplitude, including a later shared-node correction."""
    nodes = mesh["trace_nodes"]
    probes = mesh["partition_probe_indices"][mesh["partition_accepted"]]
    triangles = mesh["triangles"]
    if len(probes) != len(triangles) or np.any(probes < 0):
        raise ValueError("accepted mesh and cached four-probe indices disagree")
    if tolerance is None:
        tolerance = mesh["metadata"]["adaptive_controls"]["parameters"]["amplitude_relative_tolerance"]
    tolerance = _real(tolerance, "tolerance")
    if not len(triangles):
        return dict(checked_triangles=0, failing_triangles=0, maximum=None, p95=None, tolerance=tolerance)
    matrix = np.stack((triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=-1)
    uv = np.linalg.solve(matrix[:, None], (nodes["points"][probes]-triangles[:, None, 0])[..., None])[..., 0]
    bary = np.concatenate((1-uv.sum(axis=-1, keepdims=True), uv), axis=-1)
    candidate = np.einsum("tpi,ti->tp", bary, mesh["vertex_amplitude"])
    reference = nodes["amplitude"][probes]
    scale = np.maximum(np.max(abs(nodes["amplitude"][mesh["ray_indices"]]), axis=1), np.max(abs(reference), axis=1))
    error = np.max(abs(candidate-reference), axis=1)/np.maximum(scale, 1e-300)
    return dict(checked_triangles=len(triangles), failing_triangles=int(np.sum(error > tolerance)),
        maximum=float(np.max(error)), p95=float(np.quantile(error, .95)), tolerance=tolerance,
        amplitude_model=mesh["metadata"].get("amplitude_model", "unknown"),
        interpretation="Actual corrected affine complex amplitude against four independently traced regular-GO probe values; not a physical field bound.")

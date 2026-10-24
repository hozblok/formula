"""Conforming Stage 19 entrance elements with cached multiprecision traces."""

from collections import Counter, defaultdict, deque
from collections.abc import MutableMapping
from contextlib import ExitStack
from decimal import Decimal, localcontext
from functools import lru_cache
from time import perf_counter
from types import SimpleNamespace

from ...formula import Number
from ..native import make_tracer
from ..shared.nums import lift, vsub, vunit
from ..surfaces import CapillaryBundle
from ._s19_domain import _area, _sign, _simple, domain_indicator
from ._s19_transport import decimal, determinant, dot, norm, transport_ray


D = Decimal
EDGES = ((0, 1), (1, 2), (2, 0))
DEFAULTS = dict(initial_rings=8, aperture_inset="1e-6", min_angle_deg="20",
                max_nodes=20000, max_triangles=20000, max_depth=16,
                record_cache_directory=None, record_cache_capacity=4096,
                progress=False, transport_backend="decimal", lazy_physics=False,
                check_individual_fresnel=True,
                phase_coordinates="pullback", domain_relative="0.02", target_accepted_fraction=None,
                phase_tolerance="0.05", geometry_relative="0.02", area_relative="0.02",
                eps_r="0.02", eps_F="0.02", eps_A="0.02", eps_rho="0.02",
                eta_r="1e-6", eta_F="1e-6", eta_A="1e-6", eta_rho="1e-6",
                jacobian_floor="1e-16", caustic_endpoint_tolerance="1e-20")


class _BudgetReached(Exception):
    pass


def signed_area(triangle):
    a, b, c = triangle
    return ((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))/2


def _midpoint(a, b):
    return tuple((x+y)/2 for x, y in zip(a, b))


def _center(triangle):
    # The same element may be encountered with a cyclically permuted vertex
    # order. Fix the summation order so its cache key is identical even in
    # the last Decimal digit; this is not tolerance-based coordinate merging.
    vertices = sorted(triangle)
    return tuple(sum((p[i] for p in vertices), D(0))/3 for i in range(2))


def _probe_points(triangle):
    return [*triangle, *(_midpoint(triangle[a], triangle[b]) for a, b in EDGES),
            _center(triangle)]


def _longest(triangle):
    return max(EDGES, key=lambda edge: sum((triangle[edge[0]][i]-triangle[edge[1]][i])**2
                                         for i in range(2)))


def _children(triangle):
    a, b = _longest(triangle)
    c = 3-a-b
    middle = _midpoint(triangle[a], triangle[b])
    return [(triangle[a], middle, triangle[c]), (middle, triangle[b], triangle[c])]


def _well_shaped(triangle, cosine_squared):
    if not signed_area(triangle):
        return False
    for i in range(3):
        a = [triangle[(i+1)%3][j]-triangle[i][j] for j in range(2)]
        b = [triangle[(i+2)%3][j]-triangle[i][j] for j in range(2)]
        scalar = dot(a, b)
        if scalar > 0 and scalar*scalar > cosine_squared*dot(a, a)*dot(b, b):
            return False
    return True


@lru_cache(maxsize=64)
def _unit_partition(rings, precision):
    with localcontext() as context:
        context.prec = precision
        points = [(D(0), D(0))]
        levels = []
        for ring in range(1, rings+1):
            ids = []
            for i in range(8*ring):
                angle = f"2*pi*{i}/{8*ring}"
                x = D(str(Number(f"cos({angle})", precision)))*ring/rings
                y = D(str(Number(f"sin({angle})", precision)))*ring/rings
                ids.append(len(points))
                points.append((x, y))
            levels.append(ids)
        faces = [(0, levels[0][j], levels[0][(j+1)%8]) for j in range(8)]
        for inner, outer in zip(levels, levels[1:]):
            i = j = 0
            while i < len(inner) or j < len(outer):
                a, b = inner[i%len(inner)], outer[j%len(outer)]
                if j < len(outer) and (i == len(inner) or (j+1)*len(inner) <= (i+1)*len(outer)):
                    faces.append((a, b, outer[(j+1)%len(outer)]))
                    j += 1
                else:
                    faces.append((a, b, inner[(i+1)%len(inner)]))
                    i += 1
        return tuple(points), tuple(faces)


def disk_partition(center, radius, *, rings=8, inset="1e-6", precision=64):
    """Nested polygons with approximately equal radial and tangential steps."""
    if int(rings) != rings or rings < 1:
        raise ValueError("initial_rings must be a positive integer")
    with localcontext() as context:
        context.prec = precision
        center, radius, inset = tuple(map(decimal, center)), decimal(radius), decimal(inset)
        if radius <= 0 or not 0 <= inset < 1:
            raise ValueError("positive radius and 0 <= aperture_inset < 1 required")
        points, faces = _unit_partition(int(rings), precision)
        actual = [tuple(center[j]+radius*(1-inset)*p[j] for j in range(2)) for p in points]
        return actual, list(faces)


class _Partition:
    def __init__(self, points, faces):
        self.points = list(points)
        self.faces, self.depth, self.edges = {}, {}, defaultdict(set)
        self.next_id = 0
        for face in faces:
            self.add(face, 0)

    def add(self, face, depth):
        face = tuple(face)
        if signed_area([self.points[i] for i in face]) < 0:
            face = (face[0], face[2], face[1])
        ident = self.next_id
        self.next_id += 1
        self.faces[ident], self.depth[ident] = face, depth
        for a, b in EDGES:
            self.edges[tuple(sorted((face[a], face[b])))].add(ident)
        return ident

    def split(self, ident, max_depth, max_triangles):
        # Propagate to a shared longest edge before bisecting either owner.
        first_id, point_count = self.next_id, len(self.points)
        removed, created, pending = {}, set(), [ident]
        while pending:
            current = pending[-1]
            if current not in self.faces:
                pending.pop()
                continue
            edge = self._longest_edge(current)
            neighbors = sorted(self.edges[edge])
            length = self._edge_length_squared(edge)
            incompatible = next((n for n in neighbors
                                 if self._edge_length_squared(self._longest_edge(n)) > length), None)
            if incompatible is not None:
                pending.append(incompatible)
                continue
            limit = ("max_depth" if any(self.depth[n] >= max_depth for n in neighbors) else
                     "triangle_budget" if len(self.faces)+len(neighbors) > max_triangles else None)
            if limit:
                for new in created:
                    self._remove(new)
                for old, (face, depth) in removed.items():
                    self.faces[old], self.depth[old] = face, depth
                    for a, b in EDGES:
                        self.edges[tuple(sorted((face[a], face[b])))].add(old)
                del self.points[point_count:]
                self.next_id = first_id
                return None, limit
            mid = len(self.points)
            self.points.append(_midpoint(self.points[edge[0]], self.points[edge[1]]))
            for neighbor in neighbors:
                old, depth = self._remove(neighbor)
                if neighbor < first_id:
                    removed[neighbor] = old, depth
                else:
                    created.remove(neighbor)
                third = next(v for v in old if v not in edge)
                created.update((self.add((edge[0], mid, third), depth+1),
                                self.add((mid, edge[1], third), depth+1)))
        return (sorted(removed), sorted(created)), None

    def _longest_edge(self, ident):
        face = self.faces[ident]
        edges = [tuple(sorted((face[a], face[b]))) for a, b in EDGES]
        return max(edges, key=lambda edge: (self._edge_length_squared(edge), edge))

    def _edge_length_squared(self, edge):
        return sum((self.points[edge[0]][i]-self.points[edge[1]][i])**2 for i in range(2))

    def _remove(self, ident):
        face, depth = self.faces.pop(ident), self.depth.pop(ident)
        for a, b in EDGES:
            edge = tuple(sorted((face[a], face[b])))
            self.edges[edge].remove(ident)
            if not self.edges[edge]:
                del self.edges[edge]
        return face, depth


def _complex_mod(z):
    return (z[0]*z[0]+z[1]*z[1]).sqrt()


def _complex_close(a, b, epsilon, absolute):
    error = _complex_mod((a[0]-b[0], a[1]-b[1]))
    return error <= max(absolute, epsilon*max(_complex_mod(a), _complex_mod(b)))


def _amplitude(node, jacobian):
    scale = (node["uz_in"]/node["direction"][2]/jacobian).sqrt()/node["source_distance"]
    real, imag = (scale*v for v in node["fresnel"])
    for _ in range(node["maslov"]%4):
        real, imag = imag, -real
    return real, imag


def _complex_enclosure(low, high, reference, epsilon, absolute):
    """Check a segment of positive multiples of one phasor, or request refinement."""
    if low == high:
        error = _complex_mod(tuple(a-b for a, b in zip(low, reference)))
        scale = max(_complex_mod(low), _complex_mod(reference))
        return _complex_close(low, reference, epsilon, absolute), error, error/scale if scale else D(0)
    middle = tuple((a+b)/2 for a, b in zip(low, high))
    radius = _complex_mod(tuple((a-b)/2 for a, b in zip(low, high)))
    error = _complex_mod(tuple(a-b for a, b in zip(middle, reference)))
    reference_norm = _complex_mod(reference)
    norms = (_complex_mod(low), _complex_mod(high))
    tolerance_low = max(absolute, epsilon*max(reference_norm, min(norms)))
    tolerance_high = max(absolute, epsilon*max(reference_norm, max(norms)))
    upper_error = error+radius
    relative_scale = max(reference_norm, min(norms))
    relative = upper_error/relative_scale if relative_scale else D(0)
    if upper_error <= tolerance_low:
        return True, upper_error, relative
    if error-radius > tolerance_high:
        return False, upper_error, relative
    return None, upper_error, relative


def _point_field_checks(node, model, options, source_scale):
    bounds = node.get("determinant_bounds", (node["determinant"],)*2)
    if (len(bounds) != 2 or not all(value is not None and value.is_finite() for value in bounds) or
            bounds[0] > bounds[1] or bounds[0]*bounds[1] <= 0 or
            min(map(abs, bounds)) <= D(options["jacobian_floor"])):
        return dict(amplitude=(None, D(0), D(0)), density=(None, D(0), D(0)))
    low, high = sorted(map(abs, bounds))
    amplitudes = [_amplitude(node, jacobian) for jacobian in (low, high)]
    densities = [tuple(v*jacobian for v in amplitude)
                 for amplitude, jacobian in zip(amplitudes, (low, high))]
    rho = tuple(v*model["jacobian"] for v in model["amplitude"])
    return dict(amplitude=_complex_enclosure(*amplitudes, model["amplitude"],
                    D(options["eps_A"]), D(options["eta_A"])*source_scale),
                density=_complex_enclosure(*densities, rho,
                    D(options["eps_rho"]), D(options["eta_rho"])*source_scale))


def _mp_bore(bore, precision):
    if bore.get("kind", "cylinder") not in ("cylinder", "torus") or "sides" in bore:
        raise ValueError("Stage 19 supports circular cylinder and torus bores")
    result = dict(kind="torus" if bore.get("bend") else "cylinder",
                  center=tuple(lift(str(v), precision) for v in bore["center"]),
                  radius=lift(str(bore["radius"]), precision))
    if bore.get("bend"):
        bend = bore["bend"]
        result["bend"] = dict(radius=lift(str(bend["radius"]), precision),
                              toward=tuple(lift(str(v), precision) for v in bend["toward"]))
    return result


class _TraceCache:
    def __init__(self, sim, origin, bore, options, precision):
        self.origin = tuple(map(decimal, origin))
        self.precision, self.bore, self.options = precision, bore, options
        cap = sim.cfg.capillary
        self.z0, self.z1 = decimal(cap.z0), decimal(cap.z1)
        self.optic = CapillaryBundle([_mp_bore(bore, precision)], lift(str(cap.z0), precision),
                                    lift(str(cap.z1), precision))
        self.tracer = make_tracer(self.optic)
        self.source = tuple(lift(str(v), precision) for v in self.origin)
        self.fresnel = sim.fresnel
        self.max_bounces = sim.cfg.max_bounces
        self.records, self.fates, self.reasons = {}, Counter(), Counter()
        self.pending_rays = {}
        self.transport_backends = Counter()
        self.determinant_retraces = 0
        self.trace_seconds = self.transport_seconds = 0.
        self.store = None
        if options.get("record_cache_directory") is not None:
            from ._s19_cache import TraceStore
            self.store = TraceStore(options["record_cache_directory"],
                                    options.get("record_cache_capacity", 4096))

    def close(self):
        if self.store is not None:
            self.store.close()

    def get_raw(self, q):
        """Trace MP geometry once; pending transport is neither valid nor invalid."""
        q = tuple(map(decimal, q))
        if q in self.records:
            return self.records[q]
        if len(self.records) >= self.options["max_nodes"]:
            raise _BudgetReached
        target = (lift(str(q[0]), self.precision), lift(str(q[1]), self.precision), self.optic.z0)
        initial = vunit(vsub(target, self.source))
        begin = perf_counter()
        ray = self.tracer(self.source, initial, self.optic, self.optic.z1, self.max_bounces)
        self.trace_seconds += perf_counter()-begin
        self.fates[ray.fate] += 1
        create = self.store.create if self.store is not None else dict
        record = create(id=len(self.records), q=q, fate=ray.fate, valid=None,
                      point=tuple(map(decimal, ray.point)), direction=tuple(map(decimal, ray.direction)),
                      opl=decimal(ray.opl), reflections=len(ray.reflections), maslov=None,
                      source_distance=norm([q[0]-self.origin[0], q[1]-self.origin[1], self.z0-self.origin[2]]))
        record["uz_in"] = (self.z0-self.origin[2])/record["source_distance"]
        if ray.fate == "screen":
            record.update(physics_status="pending", reason="physics_pending")
            # Transport already converts these positions/directions to Decimal.
            # Keep the original MP incidence Number for bit-identical Fresnel.
            # Release the full native TraceResult; ordinary enrichment does not retrace.
            reflections = [(tuple(map(decimal, point)), sine) for point, sine in ray.reflections]
            self.pending_rays[q] = (self.store.pending(record, reflections) if self.store is not None else
                SimpleNamespace(point=record["point"], direction=record["direction"], reflections=reflections))
        else:
            record.update(valid=False, physics_status="not_applicable", reason="ray_"+ray.fate)
        self.reasons[record["reason"]] += 1
        self.records[q] = record
        return record

    def get(self, q):
        """Return a fully checked record, enriching a cached raw trace if needed."""
        record = self.get_raw(q)
        if record["physics_status"] == "pending":
            q = record["q"]
            ray = self.pending_rays.pop(q)
            if self.store is not None:
                ray = ray.materialize()
            self.reasons["physics_pending"] -= 1
            if not self.reasons["physics_pending"]:
                del self.reasons["physics_pending"]
            begin = perf_counter()
            try:
                if self.options["transport_backend"] == "interval":
                    from ._s19_fast_transport import transport_with_fallback
                    transport = transport_with_fallback(self.origin, q, ray, self.bore, self.z0,
                        precision=self.precision,
                        endpoint_tolerance=self.options["caustic_endpoint_tolerance"],
                        jacobian_floor=self.options["jacobian_floor"])
                else:
                    transport = transport_ray(self.origin, q, ray, self.bore, self.z0,
                        precision=self.precision,
                        endpoint_tolerance=self.options["caustic_endpoint_tolerance"])
                self.transport_backends[transport.get("backend", "decimal")] += 1
                coefficients = [self.fresnel(sine) for _, sine in ray.reflections]
                product = Number(1, self.source[0].precision)
                for coefficient in coefficients:
                    product = product*coefficient
                record.update(fresnel=tuple(map(D, product.parts)),
                              fresnel_each=[tuple(map(D, value.parts)) for value in coefficients],
                              maslov=transport["maslov"], valid=transport["valid"],
                              reason=transport["reason"], determinant=transport.get("determinant"),
                              determinant_bounds=transport.get("determinant_bounds",
                                  (transport.get("determinant"),)*2),
                              physics_status="computed")
                if abs(record["point"][2]-self.z1) > D("1e-24"):
                    record.update(valid=False, reason="exit_plane_residual")
            except (ArithmeticError, ValueError) as error:
                record.update(valid=False, reason="transport_failure", error=str(error), physics_status="failed")
            self.transport_seconds += perf_counter()-begin
            self.reasons[record["reason"]] += 1
        return record

    def refine_determinant(self, q):
        """Resolve an interval-ambiguous field gate using the same MP ray."""
        record = self.get(q)
        bounds = record["determinant_bounds"]
        previous_determinant = record["determinant"]
        target = (lift(str(q[0]), self.precision), lift(str(q[1]), self.precision), self.optic.z0)
        initial = vunit(vsub(target, self.source))
        begin = perf_counter()
        ray = self.tracer(self.source, initial, self.optic, self.optic.z1, self.max_bounces)
        self.trace_seconds += perf_counter()-begin
        self.determinant_retraces += 1
        self.reasons[record["reason"]] -= 1
        if not self.reasons[record["reason"]]:
            del self.reasons[record["reason"]]
        begin = perf_counter()
        try:
            if (ray.fate != record["fate"] or len(ray.reflections) != record["reflections"] or
                    tuple(map(decimal, ray.point)) != record["point"] or
                    tuple(map(decimal, ray.direction)) != record["direction"] or
                    decimal(ray.opl) != record["opl"]):
                raise ValueError("Repeated determinant trace differs from cached geometry")
            coefficients = [self.fresnel(sine) for _, sine in ray.reflections]
            if [tuple(map(D, value.parts)) for value in coefficients] != record["fresnel_each"]:
                raise ValueError("Repeated determinant trace changes cached Fresnel coefficients")
            product = Number(1, self.precision)
            for coefficient in coefficients:
                product = product*coefficient
            if tuple(map(D, product.parts)) != record["fresnel"]:
                raise ValueError("Repeated determinant trace changes cached Fresnel product")
            transport = transport_ray(self.origin, q, ray, self.bore, self.z0,
                precision=self.precision, endpoint_tolerance=self.options["caustic_endpoint_tolerance"])
            self.transport_backends["decimal_field_fallback"] += 1
            if transport["valid"] and transport["maslov"] != record["maslov"]:
                raise ValueError("Exact determinant transport changes cached Maslov index")
            value = transport.get("determinant")
            if transport["valid"]:
                if value*previous_determinant <= 0:
                    raise ValueError("Exact determinant transport changes cached orientation")
                if (all(v is not None and v.is_finite() for v in bounds) and
                        bounds[0] <= bounds[1] and bounds[0]*bounds[1] > 0 and
                        not bounds[0] <= value <= bounds[1]):
                    raise ValueError("Exact determinant is outside the cached certificate")
            record.update(valid=transport["valid"], reason=transport["reason"],
                          determinant=value, determinant_bounds=(value, value))
        except (ArithmeticError, ValueError) as error:
            record.update(valid=False, reason="determinant_refinement_failure", error=str(error))
        self.transport_seconds += perf_counter()-begin
        self.reasons[record["reason"]] += 1
        return record


def _model(triangle, cache, jacobian_floor):
    vertices = [cache.get(q) for q in triangle]
    center = cache.get(_center(triangle))
    if not all(node["valid"] for node in [*vertices, center]):
        return None, "invalid_ray"
    histories = {(node["reflections"], node["maslov"]) for node in [*vertices, center]}
    if len(histories) != 1:
        return None, "mixed_history"
    output = [node["point"][:2] for node in vertices]
    input_area, output_signed = abs(signed_area(triangle)), signed_area(output)
    jacobian = abs(output_signed)/input_area
    if jacobian <= jacobian_floor or center["direction"][2] <= 0:
        return None, "exit_degeneracy"
    for node in [*vertices, center]:
        if abs(node["determinant"]) <= jacobian_floor:
            return None, "exit_degeneracy"
        if node["determinant"]*output_signed <= 0:
            return None, "exit_orientation"
    return dict(input=triangle, output=output, vertices=vertices, center=center,
                amplitude=_amplitude(center, jacobian), jacobian=jacobian, output_signed=output_signed,
                input_area=input_area), None


def _domain_geometry(model, probes):
    polygon = [probes[i]["point"][:2] for i in (0, 3, 1, 4, 2, 5)]
    center = model["center"]["point"][:2]
    orientation = _sign(model["output_signed"])
    if not _simple(polygon):
        return "domain_non_simple"
    if any(_sign(_area(center, polygon[i], polygon[(i+1)%6])) != orientation for i in range(6)):
        return "domain_fan_orientation"
    return None


def _check_eager(triangle, cache, options, k, screens, cosine_squared):
    if not _well_shaped(triangle, cosine_squared):
        return None, "shape"
    model, reason = _model(triangle, cache, D(options["jacobian_floor"]))
    if model is None:
        return None, reason
    center = model["center"]
    output = model["output"]
    physical_exit = options.get("phase_coordinates", "pullback") == "physical_exit"
    if physical_exit:
        screens = []
    longest = max(norm([output[a][i]-output[b][i] for i in range(2)]) for a, b in EDGES)
    phase_limit = D(options["phase_tolerance"])
    phase_max, geometry_max = D(0), D(0)
    individual_fresnel_exceedances = 0
    optical_max = D(0)
    screen_phase = [dict(z=str(screen["z"]), max_geometry_center_rad=D(0),
                         max_full_rad=D(0)) for screen in screens]
    probes, probe_nodes = [], []
    source_scale = 1/(cache.z0-cache.origin[2])
    point_field_max = dict(amplitude_absolute=D(0), amplitude_relative=D(0),
                           density_absolute=D(0), density_relative=D(0))
    for q, mapped in zip(_probe_points(triangle), _probe_points(output)):
        node = cache.get(q)
        probes.append(node["id"])
        probe_nodes.append(node)
        if not node["valid"]:
            return None, node["reason"]
        if (node["reflections"], node["maslov"]) != (center["reflections"], center["maslov"]):
            return None, "mixed_history"
        if (abs(node["determinant"]) <= D(options["jacobian_floor"]) or
                node["direction"][2] <= 0):
            return None, "exit_degeneracy"
        if node["determinant"]*model["output_signed"] <= 0:
            return None, "exit_orientation"
        for value, reference in zip(node["fresnel_each"], center["fresnel_each"]):
            if not _complex_close(value, reference, D(options["eps_r"]), D(options["eta_r"])):
                individual_fresnel_exceedances += 1
                if options.get("check_individual_fresnel", True):
                    return None, "fresnel_each"
        if not _complex_close(node["fresnel"], center["fresnel"], D(options["eps_F"]), D(options["eta_F"])):
            return None, "fresnel_product"
        error = tuple(mapped[i]-node["point"][i] for i in range(2))
        geometry_max = max(geometry_max, norm(error))
        if norm(error) > D(options["geometry_relative"])*longest:
            return None, "geometry"
        phase_point = node["point"] if physical_exit else mapped
        path = center["opl"]+sum((center["direction"][i]*(phase_point[i]-center["point"][i])
                                  for i in range(2)), D(0))
        phase = k*(path-node["opl"])
        optical_max = max(optical_max, abs(phase))
        phase_max = max(phase_max, abs(phase))
        if abs(phase) > phase_limit:
            return None, "exit_phase"
        for screen_index, screen in enumerate(screens):
            distance = decimal(screen["z"])-cache.z1
            if distance <= 0:
                continue
            pc = tuple(map(decimal, screen.get("center", (0, 0))))
            geometric = sum(((pc[i]-mapped[i])**2-(pc[i]-node["point"][i])**2 for i in range(2)), D(0))
            geometry_phase = k*geometric/(2*distance)
            full = phase+geometry_phase
            extent = abs(error[0])*decimal(screen["edge_x"])/2+abs(error[1])*decimal(screen["edge_y"])/2
            maximum = abs(full)+k*extent/distance
            summary = screen_phase[screen_index]
            summary["max_geometry_center_rad"] = max(summary["max_geometry_center_rad"], abs(geometry_phase))
            summary["max_full_rad"] = max(summary["max_full_rad"], maximum)
            phase_max = max(phase_max, maximum)
            if maximum > phase_limit:
                return None, "screen_phase"
        checks = _point_field_checks(node, model, options, source_scale)
        for name, result in checks.items():
            if result[0] is False:
                return None, "point_"+name
        if any(result[0] is None for result in checks.values()):
            node = cache.refine_determinant(q)
            if not node["valid"]:
                return None, node["reason"]
            if (abs(node["determinant"]) <= D(options["jacobian_floor"]) or
                    node["direction"][2] <= 0):
                return None, "exit_degeneracy"
            if node["determinant"]*model["output_signed"] <= 0:
                return None, "exit_orientation"
            checks = _point_field_checks(node, model, options, source_scale)
        for name, (passed, absolute, relative) in checks.items():
            if not passed:
                return None, "point_"+name
            point_field_max[name+"_absolute"] = max(point_field_max[name+"_absolute"], absolute)
            point_field_max[name+"_relative"] = max(point_field_max[name+"_relative"], relative)
    if physical_exit:
        reason = _domain_geometry(model, probe_nodes)
        if reason:
            return None, reason
    for child in _children(triangle):
        small, reason = _model(child, cache, D(options["jacobian_floor"]))
        if small is None:
            return None, "child_"+reason
        if small["output_signed"]*model["output_signed"] <= 0:
            return None, "child_orientation"
        if (small["center"]["reflections"], small["center"]["maslov"]) != (center["reflections"], center["maslov"]):
            return None, "child_history"
        if abs(small["jacobian"]-model["jacobian"]) > D(options["area_relative"])*max(small["jacobian"], model["jacobian"]):
            return None, "child_area"
        if not _complex_close(small["amplitude"], model["amplitude"],
                              D(options["eps_A"]), D(options["eta_A"])*source_scale):
            return None, "child_amplitude"
        rho = tuple(v*model["jacobian"] for v in model["amplitude"])
        child_rho = tuple(v*small["jacobian"] for v in small["amplitude"])
        if not _complex_close(child_rho, rho, D(options["eps_rho"]), D(options["eta_rho"])*source_scale):
            return None, "child_density"
    model.update(probe_node_ids=probes, phase_max=phase_max, geometry_max=geometry_max,
                 individual_fresnel_exceedances=individual_fresnel_exceedances,
                 individual_fresnel_enforced=options.get("check_individual_fresnel", True),
                 point_field_max=point_field_max,
                 optical_phase_max=optical_max,
                 screen_phase=[row for row in screen_phase if decimal(row["z"]) > cache.z1])
    if physical_exit:
        model["phase_coordinates"] = "physical_exit"
    return model, None


def _raw_model(triangle, cache, jacobian_floor):
    """Only quantities used by physics-independent rejection gates."""
    vertices = [cache.get_raw(q) for q in triangle]
    center = cache.get_raw(_center(triangle))
    if any(node["fate"] != "screen" or node["valid"] is False for node in [*vertices, center]):
        return None, "invalid_ray"
    if len({node["reflections"] for node in [*vertices, center]}) != 1:
        return None, "mixed_history"
    output = [node["point"][:2] for node in vertices]
    input_area, output_signed = abs(signed_area(triangle)), signed_area(output)
    jacobian = abs(output_signed)/input_area
    if jacobian <= jacobian_floor or center["direction"][2] <= 0:
        return None, "exit_degeneracy"
    return dict(input=triangle, output=output, vertices=vertices, center=center,
                jacobian=jacobian, output_signed=output_signed, input_area=input_area), None


def _cheap_rejection(triangle, cache, options, k, screens, cosine_squared):
    """Return a sufficient rejection, never an acceptance or a Maslov verdict.

    Every comparison below also occurs in _check_eager with the same MP
    arithmetic. A surviving element still passes through _check_eager in full.
    Earlier rejection can change cache insertion order and budget-limited mesh.
    """
    if not _well_shaped(triangle, cosine_squared):
        return "shape"
    floor = D(options["jacobian_floor"])
    model, reason = _raw_model(triangle, cache, floor)
    if model is None:
        return reason
    center, output = model["center"], model["output"]
    physical_exit = options.get("phase_coordinates", "pullback") == "physical_exit"
    if physical_exit:
        screens = []
    longest = max(norm([output[a][i]-output[b][i] for i in range(2)]) for a, b in EDGES)
    phase_limit = D(options["phase_tolerance"])
    probe_nodes = []
    for q, mapped in zip(_probe_points(triangle), _probe_points(output)):
        node = cache.get_raw(q)
        probe_nodes.append(node)
        if node["fate"] != "screen" or node["valid"] is False:
            return node["reason"]
        if node["reflections"] != center["reflections"]:
            return "mixed_history"
        error = tuple(mapped[i]-node["point"][i] for i in range(2))
        if norm(error) > D(options["geometry_relative"])*longest:
            return "geometry"
        phase_point = node["point"] if physical_exit else mapped
        path = center["opl"]+sum((center["direction"][i]*(phase_point[i]-center["point"][i])
                                  for i in range(2)), D(0))
        phase = k*(path-node["opl"])
        if abs(phase) > phase_limit:
            return "exit_phase"
        for screen in screens:
            distance = decimal(screen["z"])-cache.z1
            if distance <= 0:
                continue
            pc = tuple(map(decimal, screen.get("center", (0, 0))))
            geometric = sum(((pc[i]-mapped[i])**2-(pc[i]-node["point"][i])**2 for i in range(2)), D(0))
            full = phase+k*geometric/(2*distance)
            extent = abs(error[0])*decimal(screen["edge_x"])/2+abs(error[1])*decimal(screen["edge_y"])/2
            if abs(full)+k*extent/distance > phase_limit:
                return "screen_phase"
    if physical_exit:
        reason = _domain_geometry(model, probe_nodes)
        if reason:
            return reason
    for child in _children(triangle):
        small, reason = _raw_model(child, cache, floor)
        if small is None:
            return "child_"+reason
        if small["output_signed"]*model["output_signed"] <= 0:
            return "child_orientation"
        if small["center"]["reflections"] != center["reflections"]:
            return "child_history"
        if abs(small["jacobian"]-model["jacobian"]) > D(options["area_relative"])*max(small["jacobian"], model["jacobian"]):
            return "child_area"
    return None


def _check(triangle, cache, options, k, screens, cosine_squared):
    if options.get("lazy_physics", False):
        reason = _cheap_rejection(triangle, cache, options, k, screens, cosine_squared)
        if reason is not None:
            return None, reason
    return _check_eager(triangle, cache, options, k, screens, cosine_squared)


def _record(model, bore):
    center = model["center"]
    record = dict(input=[[str(v) for v in q] for q in model["input"]],
                exit=[[str(v) for v in q] for q in model["output"]],
                center_exit=[str(v) for v in center["point"][:2]],
                direction=[str(v) for v in center["direction"]], opl=str(center["opl"]),
                amplitude=[str(v) for v in model["amplitude"]], maslov=center["maslov"],
                reflections=center["reflections"], bore=bore,
                input_area=str(model["input_area"]), exit_area=str(abs(model["output_signed"])),
                jacobian=str(model["jacobian"]), orientation=1 if model["output_signed"] > 0 else -1,
                center_node_id=center["id"], probe_node_ids=model["probe_node_ids"],
                phase_probe_max_rad=str(model["phase_max"]), geometry_probe_max_m=str(model["geometry_max"]),
                fresnel_diagnostics=dict(individual_exceedances=model["individual_fresnel_exceedances"],
                    individual_check_enforced=model["individual_fresnel_enforced"]),
                point_field_diagnostics={key: str(value) for key, value in model["point_field_max"].items()},
                phase_diagnostics=dict(max_optical_rad=float(model["optical_phase_max"]),
                    screens=[dict(z=row["z"], max_geometry_center_rad=float(row["max_geometry_center_rad"]),
                                  max_full_rad=float(row["max_full_rad"])) for row in model["screen_phase"]]))
    if model.get("phase_coordinates") == "physical_exit":
        record.update(phase_coordinates="physical_exit", domain_geometry="simple_P6_consistent_center_fan",
                      domain_indicator_certified=False)
    return record


def _serialize(value):
    if isinstance(value, D):
        return str(value)
    if isinstance(value, MutableMapping):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


def build_source_mesh(sim, origin3decimal, options=None, *, retain_nodes=True):
    """Build all bores; unresolved elements remain explicit entrance records."""
    options = {**DEFAULTS, **(options or {})}
    precision = int(sim.cfg.precision)
    if not isinstance(options["progress"], bool):
        raise ValueError("progress must be boolean")
    if not isinstance(options["lazy_physics"], bool):
        raise ValueError("lazy_physics must be boolean")
    if not isinstance(options["check_individual_fresnel"], bool):
        raise ValueError("check_individual_fresnel must be boolean")
    if options["transport_backend"] not in ("decimal", "interval"):
        raise ValueError("transport_backend must be decimal or interval")
    if options["phase_coordinates"] not in ("pullback", "physical_exit"):
        raise ValueError("phase_coordinates must be pullback or physical_exit")
    target = options["target_accepted_fraction"]
    if target is not None and (isinstance(target, bool) or not decimal(target).is_finite() or not 0 < decimal(target) <= 1):
        raise ValueError("target_accepted_fraction must be null or in (0, 1]")
    if not isinstance(retain_nodes, bool):
        raise ValueError("retain_nodes must be boolean")
    for name in ("initial_rings", "max_nodes", "max_triangles", "max_depth", "record_cache_capacity"):
        if isinstance(options[name], bool) or int(options[name]) != options[name] or options[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if options["record_cache_directory"] is not None and (not isinstance(options["record_cache_directory"], str)
            or not options["record_cache_directory"].strip()):
        raise ValueError("record_cache_directory must be a nonempty path string or null")
    if not 0 < decimal(options["min_angle_deg"]) <= 60:
        raise ValueError("min_angle_deg must be in (0, 60]")
    for name in ("phase_tolerance", "geometry_relative", "area_relative", "eps_r", "eps_F", "eps_A", "eps_rho",
                 "eta_r", "eta_F", "eta_A", "eta_rho", "jacobian_floor", "caustic_endpoint_tolerance", "domain_relative"):
        if not decimal(options[name]).is_finite() or decimal(options[name]) <= 0:
            raise ValueError(f"{name} must be positive")
    cap = sim.cfg.capillary
    if cap is None or len(sim.lines) != 1:
        raise ValueError("Stage 19 requires a monochromatic capillary scene")
    started = perf_counter()
    with localcontext() as context, ExitStack() as cleanup:
        context.prec = precision
        origin = tuple(map(decimal, origin3decimal))
        if len(origin) != 3:
            raise ValueError("source must contain three coordinates")
        if origin[2] >= decimal(cap.z0):
            raise ValueError("source must precede entrance")
        k = D(str(Number("2*pi", precision)/sim.lam))
        cosine = D(str(Number(f"cos(pi*{options['min_angle_deg']}/180)", precision)))
        screens = options.get("screens")
        if screens is None:
            screens = [dict(z=str(s.z), center=list(map(str, s.center)), edge_x=str(s.edge_x), edge_y=str(s.edge_y))
                       for s in [cap.screen, *cap.screens]]
        triangles, unresolved, nodes, bore_stats = [], [], [], []
        selected = options.get("bores", list(range(len(cap.bores))))
        if (len(set(selected)) != len(selected) or
                any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(cap.bores)
                    for i in selected)):
            raise ValueError("bores must contain distinct valid integer indices")
        for bore_id in selected:
            bore_started = perf_counter()
            bore = cap.bores[bore_id]
            cache = _TraceCache(sim, origin, bore, options, precision)
            if cache.store is not None:
                # Register only the store; keeping the whole cache here would
                # retain all trace records during final domain processing.
                cleanup.callback(cache.store.close)
            points, faces = disk_partition(bore["center"], bore["radius"], rings=options["initial_rings"],
                                          inset=options["aperture_inset"], precision=precision)
            partition = _Partition(points, faces)
            initial_area = sum((abs(signed_area([points[v] for v in face])) for face in faces), D(0))
            physical_area = D(str(Number("pi", precision)))*decimal(bore["radius"])**2
            queue, accepted, rejected = deque(partition.faces), {}, {}
            failures, splits = Counter(), 0
            running_accepted_area = D(0)
            coverage_target = decimal(target) if target is not None else None
            budget_draining, drain_checked, drain_accepted = False, 0, 0
            next_report = 20000
            while queue:
                if options["progress"] and len(cache.records) >= next_report:
                    print(f"stage19 mesh progress bore={bore_id} nodes={len(cache.records)} "
                          f"accepted={len(accepted)} coverage={float(running_accepted_area/physical_area):.6f} "
                          f"seconds={perf_counter()-bore_started:.2f}", flush=True)
                    next_report = (len(cache.records)//20000+1)*20000
                if coverage_target is not None and running_accepted_area >= coverage_target*physical_area:
                    for other in partition.faces:
                        if other not in accepted:
                            rejected.setdefault(other, "coverage_target")
                    break
                ident = queue.popleft()
                if ident not in partition.faces or ident in accepted:
                    continue
                triangle = tuple(partition.points[v] for v in partition.faces[ident])
                drain_checked += budget_draining
                try:
                    model, reason = _check(triangle, cache, options, k, screens, cosine*cosine)
                except _BudgetReached:
                    budget_draining = True
                    rejected[ident] = "node_budget"
                    continue
                if model is not None:
                    accepted[ident] = model
                    running_accepted_area += model["input_area"]
                    drain_accepted += budget_draining
                    continue
                failures[reason] += 1
                if budget_draining:
                    rejected[ident] = "node_budget:"+reason
                    continue
                result, limit = partition.split(ident, options["max_depth"], options["max_triangles"])
                if result is None:
                    rejected[ident] = limit+":"+reason
                else:
                    removed, created = result
                    splits += 1
                    for old in removed:
                        old_model = accepted.pop(old, None)
                        if old_model is not None:
                            running_accepted_area -= old_model["input_area"]
                        rejected.pop(old, None)
                    queue.extend(created)
            accepted_area = sum((model["input_area"] for model in accepted.values()), D(0))
            unresolved_area, reasons = D(0), Counter()
            bore_unresolved = []
            for ident, face in partition.faces.items():
                if ident in accepted:
                    continue
                triangle = [partition.points[v] for v in face]
                area = abs(signed_area(triangle))
                reason = rejected.get(ident, "unprocessed")
                reasons[reason] += 1
                unresolved_area += area
                bore_unresolved.append(dict(bore=bore_id, input=[[str(v) for v in q] for q in triangle],
                                       input_area=str(area), reason=reason, depth=partition.depth[ident]))
            unresolved.extend(bore_unresolved)
            bore_triangles = [_record(model, bore_id) for model in accepted.values()]
            triangles.extend(bore_triangles)
            domain_points = {}
            if options["phase_coordinates"] == "physical_exit":
                for model in accepted.values():
                    for a, b in EDGES:
                        q = _midpoint(model["input"][a], model["input"][b])
                        if q not in domain_points:
                            domain_points[q] = cache.records[q]["point"][:2]
            if retain_nodes:
                nodes.extend(dict(bore=bore_id, **_serialize(record)) for record in cache.records.values())
            bore_stats.append(dict(bore=bore_id, accepted_triangles=len(accepted), unresolved_triangles=sum(reasons.values()),
                domain_indicator=None, target_accepted_fraction=str(coverage_target) if coverage_target is not None else None,
                target_reached=coverage_target is not None and accepted_area >= coverage_target*physical_area,
                incremental_area_defect_m2=str(running_accepted_area-accepted_area),
                budget_drain_checked=drain_checked, budget_drain_accepted=drain_accepted,
                nodes=len(cache.records), total_triangles=len(partition.faces), splits=splits,
                physical_area_m2=str(physical_area), partition_area_m2=str(initial_area),
                accepted_area_m2=str(accepted_area), unresolved_area_m2=str(unresolved_area),
                aperture_deficit_m2=str(physical_area-initial_area),
                aperture_deficit_fraction=str((physical_area-initial_area)/physical_area),
                accepted_fraction=str(accepted_area/physical_area),
                partition_balance_defect_m2=str(accepted_area+unresolved_area-initial_area),
                fates=dict(cache.fates), transport_reasons=dict(cache.reasons),
                transport_backends=dict(cache.transport_backends),
                physics_pending_nodes=len(cache.pending_rays),
                determinant_retraces=cache.determinant_retraces,
                physics_computed_nodes=sum(row["physics_status"] == "computed" for row in cache.records.values()),
                rejection_counts=dict(failures), unresolved_reasons=dict(reasons),
                tracing_seconds=cache.trace_seconds, transport_seconds=cache.transport_seconds))
            if cache.store is not None:
                bore_stats[-1]["record_cache"] = cache.store.statistics()
                cache.close()
            # Final field records and counters no longer depend on the ray cache.
            del cache, accepted, partition, points, faces, queue, rejected
            if options["phase_coordinates"] == "physical_exit":
                bore_stats[-1]["domain_indicator"] = domain_indicator(bore_triangles,
                    lambda bore_index, q: domain_points[q], k=k, precision=precision,
                    unresolved=bore_unresolved, safety="2", tolerance=options["domain_relative"])
            del domain_points
            if options["progress"]:
                print(f"stage19 mesh source=({origin[0]},{origin[1]}) bore={bore_id} "
                      f"accepted={bore_stats[-1]['accepted_triangles']} nodes={bore_stats[-1]['nodes']} "
                      f"coverage={float(accepted_area/physical_area):.6f} "
                      f"seconds={perf_counter()-bore_started:.2f}", flush=True)
        return dict(triangles=triangles, unresolved=unresolved, nodes=nodes,
                    stats=dict(precision=precision, source=list(map(str, origin)), bores=bore_stats,
                               options=_serialize(options),
                               seconds=perf_counter()-started, accepted_triangles=len(triangles),
                               unresolved_triangles=len(unresolved),
                               nodes=sum(row["nodes"] for row in bore_stats),
                               retained_nodes=len(nodes),
                               physics_pending_nodes=sum(row["physics_pending_nodes"] for row in bore_stats),
                               accuracy_validated=False, physical_model="regular-GO central amplitude and linear OPL",
                               limitations=["Finite probes do not certify regularity between nodes.",
                                   "Exit caustics and unresolved input elements have no wave error bound.",
                                   "Inscribed polygon aperture deficit is recorded separately.",
                                   "Flux quadrature acceptance is not yet enabled."]))

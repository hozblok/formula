"""Shared-edge finite-curve indicator for accepted straight field elements."""

from collections import Counter, defaultdict
from decimal import Decimal as D, localcontext

from ...formula import Number


def _area(a, b, c):
    return ((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))/2


def _sign(value):
    return (value > 0)-(value < 0)


def _magnitude(value):
    return (value[0]*value[0]+value[1]*value[1]).sqrt()


def _edge(bore, a, b):
    return (bore, *sorted((a, b)))


def _intersects(a, b, c, d):
    sides = (_area(a, b, c), _area(a, b, d), _area(c, d, a), _area(c, d, b))
    if _sign(sides[0])*_sign(sides[1]) < 0 and _sign(sides[2])*_sign(sides[3]) < 0:
        return True
    return any(value == 0 and all(min(p[i], q[i]) <= x[i] <= max(p[i], q[i]) for i in range(2))
               for value, x, p, q in ((sides[0], c, a, b), (sides[1], d, a, b),
                                      (sides[2], a, c, d), (sides[3], b, c, d)))


def _simple(polygon):
    size = len(polygon)
    return not any(_intersects(polygon[i], polygon[(i+1) % size],
                               polygon[j], polygon[(j+1) % size])
                   for i in range(size) for j in range(i+1, size)
                   if j not in ((i+1) % size, (i-1) % size))


def _jump(left, right, origin, points, k, turn, precision):
    # Remove one common unit carrier after subtracting both full anchored OPLs.
    delta = k*((left["opl"]-right["opl"])
               + sum(left["direction"][n]*(origin[n]-left["center"][n])
                     - right["direction"][n]*(origin[n]-right["center"][n]) for n in range(2)))
    reduced = delta % turn
    cosine = D(str(Number(f"cos({reduced})", precision)))
    sine = D(str(Number(f"sin({reduced})", precision)))
    ar, ai = left["amplitude"]
    br, bi = right["amplitude"]
    base = _magnitude((ar*cosine-ai*sine-br, ar*sine+ai*cosine-bi))
    variation = max(abs(k*sum((left["direction"][n]-right["direction"][n])
                              * (point[n]-origin[n]) for n in range(2))) for point in points)
    return min(left["magnitude"]+right["magnitude"],
               base+min(left["magnitude"], right["magnitude"])*min(D(2), variation))


def domain_indicator(records, midpoint, *, k, precision, unresolved=(),
                     safety="2", tolerance="0.02"):
    """Call midpoint(bore, input_point) for its actual 2D exit coordinates.

    The callback may reuse the trace cache. This tests a finite-curve indicator,
    P6 simplicity and center-fan orientation; it does not certify a curved map.
    """
    with localcontext() as context:
        context.prec = precision
        k, safety, tolerance = D(str(k)), D(str(safety)), D(str(tolerance))
        if not all(value.is_finite() and value > 0 for value in (k, safety, tolerance)):
            raise ValueError("Domain indicator needs finite positive k, safety and tolerance")
        turn = D(str(Number("2*pi", precision)))
        models, adjacency, unresolved_edges = [], defaultdict(list), set()
        output_vertices, vertex_disagreements = {}, 0
        for index, row in enumerate(records):
            entrance = [tuple(map(D, point)) for point in row["input"]]
            output = [tuple(map(D, point[:2])) for point in row["exit"]]
            amplitude = tuple(map(D, row["amplitude"]))
            bore = row["bore"]
            model = dict(entrance=entrance, output=output, bore=bore,
                         amplitude=amplitude, magnitude=_magnitude(amplitude),
                         center=tuple(map(D, row["center_exit"][:2])),
                         direction=tuple(map(D, row["direction"][:2])), opl=D(row["opl"]),
                         branch=(bore, row["reflections"], row["maslov"], row["orientation"]),
                         orientation=row["orientation"])
            models.append(model)
            for side in range(3):
                key = (bore, entrance[side])
                vertex_disagreements += output_vertices.setdefault(key, output[side]) != output[side]
                adjacency[_edge(bore, entrance[side], entrance[(side+1) % 3])].append((index, side))
        for row in unresolved:
            points = [tuple(map(D, point)) for point in row["input"]]
            unresolved_edges.update(_edge(row["bore"], points[i], points[(i+1) % 3]) for i in range(3))
        normalizer = sum((model["magnitude"]*abs(_area(*model["output"])) for model in models), D(0))
        area_normalizer = sum((abs(_area(*model["output"])) for model in models), D(0))
        weighted = amplitude_envelope = area_envelope = D(0)
        midpoints, counts, contributions = {}, Counter(), defaultdict(lambda: D(0))
        for key, owners in adjacency.items():
            bore, a, b = key
            q = tuple((a[n]+b[n])/2 for n in range(2))
            point = tuple(map(D, midpoint(bore, q)))[:2]
            if len(point) != 2 or not all(value.is_finite() for value in point):
                raise ValueError("Domain indicator midpoint needs two finite exit coordinates")
            midpoints[key] = point
            envelope = D(0)
            for index, side in owners:
                model = models[index]
                ear = abs(_area(model["output"][side], point, model["output"][(side+1) % 3]))
                envelope += safety*ear*model["magnitude"]
                area_envelope += safety*ear
            amplitude_envelope += envelope
            if len(owners) == 2 and models[owners[0][0]]["branch"] == models[owners[1][0]]["branch"]:
                left, right = (models[owner[0]] for owner in owners)
                start, end = output_vertices[(bore, a)], output_vertices[(bore, b)]
                contribution = safety*abs(_area(start, point, end))*_jump(
                    left, right, start, (start, point, end), k, turn, precision)
                kind = "internal_same_branch"
            else:
                contribution = envelope
                kind = ("nonmatching_branch" if len(owners) == 2 else "nonmanifold" if len(owners) > 2
                        else "unresolved_neighbor" if key in unresolved_edges else "missing_or_aperture_neighbor")
            weighted += contribution
            counts[kind] += 1
            contributions[kind] += contribution
        fans = Counter(simple=0, non_simple=0, all_fan_orientation_consistent=0,
                       bad_fan_orientation=0, zero_fan_triangles=0)
        for model in models:
            polygon = []
            for side in range(3):
                polygon.extend((model["output"][side], midpoints[_edge(
                    model["bore"], model["entrance"][side], model["entrance"][(side+1) % 3])]))
            simple = _simple(polygon)
            areas = [_area(model["center"], polygon[i], polygon[(i+1) % 6]) for i in range(6)]
            oriented = all(_sign(value) == model["orientation"] for value in areas)
            fans["simple"] += simple
            fans["non_simple"] += not simple
            fans["all_fan_orientation_consistent"] += oriented
            fans["bad_fan_orientation"] += not oriented
            fans["zero_fan_triangles"] += sum(value == 0 for value in areas)
        relative = weighted/normalizer if normalizer else None
        passed = (relative is not None and relative <= tolerance and not vertex_disagreements
                  and not counts["nonmanifold"] and not fans["non_simple"] and not fans["bad_fan_orientation"])
        return dict(passed=passed, safety=str(safety), tolerance=str(tolerance),
                    accepted_triangles=len(models), unique_edges=len(adjacency),
                    weighted_domain_indicator=str(weighted), weighted_domain_relative=str(relative) if relative is not None else None,
                    normalizer_sum_abs_A_exit_area=str(normalizer), normalizer_sum_exit_area=str(area_normalizer),
                    independent_triangle_amplitude_envelope=str(amplitude_envelope),
                    independent_triangle_amplitude_relative=str(amplitude_envelope/normalizer) if normalizer else None,
                    independent_triangle_area_relative=str(area_envelope/area_normalizer) if area_normalizer else None,
                    counts=dict(counts), contributions={key: str(value) for key, value in contributions.items()},
                    exit_vertex_disagreements=vertex_disagreements, p6_fan_checks=dict(fans),
                    interpretation="Finite midpoint-ear indicator of absolute L1 domain error, normalized by sum |A| times straight-element area; not a certified curved-boundary or coherence error bound.")

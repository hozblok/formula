"""Optional interval DRT certificate with independent MP geometry verification.

This prototype does not change the existing mesh or Decimal transport. A caller
may use ``transport_with_fallback`` instead of the slow DRT only after validating
the compiled backend. Field coordinates, OPL, Fresnel coefficients and triangle
areas remain in their existing MP representation.
"""

from decimal import Decimal as D, localcontext
import math

from ._s19_transport import decimal, dot, norm, transport_ray


def enclosing_interval(value):
    """Binary64 endpoints provably enclosing this finite Decimal input.

    Conversion is checked against Decimal.from_float, which represents the
    binary float exactly; the Decimal context does not round this comparison.
    """
    exact = decimal(value)
    if not exact.is_finite():
        raise ValueError("nonfinite interval input")
    rounded = float(exact)
    if not math.isfinite(rounded):
        raise ValueError("input outside binary64 range")
    represented = D.from_float(rounded)
    low = math.nextafter(rounded, -math.inf) if represented > exact else rounded
    high = math.nextafter(rounded, math.inf) if represented < exact else rounded
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError("interval bound overflow")
    return [low, high]


def interval_payload(origin, entrance, ray, bore, z0, *,
                     endpoint_tolerance="1e-20", jacobian_floor="1e-16"):
    vector = lambda items: [enclosing_interval(v) for v in items]
    geometry = {"center": vector([*bore["center"], 0]),
                "radius": enclosing_interval(bore["radius"])}
    if bore.get("bend"):
        geometry["bend_radius"] = enclosing_interval(bore["bend"]["radius"])
        geometry["toward"] = vector([*bore["bend"]["toward"], 0])
    elif bore.get("kind", "cylinder") != "cylinder":
        raise ValueError("unsupported wall geometry")
    return {"source": vector(origin), "entrance": vector([*entrance, z0]),
            "points": [vector(point) for point, _ in ray.reflections]+[vector(ray.point)],
            "direction": vector(ray.direction), "bore": geometry,
            "z0": enclosing_interval(z0),
            "endpoint_tolerance": enclosing_interval(endpoint_tolerance),
            "jacobian_floor": enclosing_interval(jacobian_floor)}


def _wall_normal(point, bore, z0):
    """Same stable geometry as Decimal DRT; omit the Hessian and derivative."""
    center = tuple(map(decimal, bore["center"]))
    bend = bore.get("bend")
    if not bend and bore.get("kind", "cylinder") == "cylinder":
        raw = [point[0]-center[0], point[1]-center[1], D(0)]
    elif bend:
        radius = decimal(bend["radius"])
        toward = tuple(map(decimal, bend["toward"]))
        toward_norm = norm(toward)
        tangent = [toward[0]/toward_norm, toward[1]/toward_norm, D(0)]
        binormal = [-tangent[1], tangent[0], D(0)]
        local = [point[0]-center[0], point[1]-center[1], point[2]-decimal(z0)]
        xi, eta, zeta = dot(local, tangent), dot(local, binormal), local[2]
        rho = ((radius-xi)**2+zeta*zeta).sqrt()
        dr = (-2*radius*xi+xi*xi+zeta*zeta)/(rho+radius)
        radial = [(xi-radius)*tangent[0]/rho, (xi-radius)*tangent[1]/rho, zeta/rho]
        raw = [dr*radial[i]+eta*binormal[i] for i in range(3)]
    else:
        raise ValueError("unsupported wall geometry")
    length = norm(raw)
    if not length:
        raise ValueError("undefined wall normal")
    return [v/length for v in raw], length


def verify_ray_geometry(origin, entrance, ray, bore, z0, *, precision=64,
                        endpoint_tolerance="1e-20"):
    """Retain the old MP position/wall/direction/grazing residual checks.

    This intentionally does not compute Q/P or claim that the previous
    symplectic residual was evaluated. The native interval certificate checks
    the differential formulas separately.
    """
    with localcontext() as context:
        context.prec = precision
        source = list(map(decimal, origin))
        start = [decimal(entrance[0]), decimal(entrance[1]), decimal(z0)]
        initial = [start[i]-source[i] for i in range(3)]
        distance = norm(initial)
        if not distance:
            return {"valid": False, "reason": "source_on_entrance"}
        u = [x/distance for x in initial]
        reflections = [list(map(decimal, p)) for p, _ in ray.reflections]
        destinations = reflections+[list(map(decimal, ray.point))]
        position_error = wall_error = D(0)
        grazing = D(1)
        edge = decimal(endpoint_tolerance)
        for index, destination in enumerate(destinations):
            dz = destination[2]-start[2]
            if dz < -edge or u[2] <= 0:
                return {"valid": False, "reason": "backward_path"}
            flight = dz/u[2]
            position_error = max(position_error,
                norm([start[i]+flight*u[i]-destination[i] for i in range(3)]))
            if index < len(reflections):
                normal, wall_radius = _wall_normal(destination, bore, z0)
                wall_error = max(wall_error, abs(wall_radius-decimal(bore["radius"])))
                udot = dot(u, normal)
                grazing = min(grazing, abs(udot))
                if abs(udot) <= edge:
                    return {"valid": False, "reason": "grazing"}
                u = [u[i]-2*udot*normal[i] for i in range(3)]
                if u[2] <= 0:
                    return {"valid": False, "reason": "backward_path"}
            start = destination
        direction_error = norm([u[i]-decimal(ray.direction[i]) for i in range(3)])
        tolerance = max(D("1e-24"), D(10)**(-(precision-12)))
        valid = (position_error <= tolerance and wall_error <= tolerance and
                 direction_error <= tolerance)
        return {"valid": valid, "reason": "ok" if valid else "transport_residual",
                "position_error": position_error, "wall_error": wall_error,
                "direction_error": direction_error, "minimum_grazing": grazing,
                "geometry_tolerance": tolerance}


def certify_transport(origin, entrance, ray, bore, z0, *, precision=64,
                      endpoint_tolerance="1e-20", jacobian_floor="1e-16"):
    """Try the interval certificate; no hidden Decimal DRT fallback here."""
    from formula import _stage19_transport as native
    try:
        geometry = verify_ray_geometry(origin, entrance, ray, bore, z0,
                                      precision=precision, endpoint_tolerance=endpoint_tolerance)
    except (ArithmeticError, ValueError, OverflowError) as error:
        return {"certified": False, "reason": "mp_geometry_failure", "error": str(error),
                "mp_geometry_verified": False}
    if not geometry["valid"]:
        return {"certified": False, "reason": "mp_"+geometry["reason"], "geometry": geometry,
                "mp_geometry_verified": False}
    try:
        payload = interval_payload(origin, entrance, ray, bore, z0,
            endpoint_tolerance=endpoint_tolerance, jacobian_floor=jacobian_floor)
    except (ArithmeticError, ValueError, OverflowError) as error:
        return {"certified": False, "reason": "interval_input", "error": str(error),
                "geometry": geometry, "mp_geometry_verified": True}
    result = native.certify(payload)
    result.update(geometry=geometry, mp_geometry_verified=True)
    return result


def transport_with_fallback(origin, entrance, ray, bore, z0, *, precision=64,
                            endpoint_tolerance="1e-20", jacobian_floor="1e-16"):
    """Return regularity, Maslov and bounds for the fixed-z determinant.

    The successful fast result has no Decimal Q/P and no old symplectic
    residual. The midpoint is diagnostic; magnitude-sensitive decisions use
    determinant_bounds and request Decimal refinement when ambiguous.
    """
    if not ray.reflections:
        # A point source in front of two parallel z planes gives the exact
        # affine transverse map X=s+(q-s)*(z_exit-z_s)/(z_in-z_s).
        # Preserve the MP path/direction validation before using this identity.
        geometry = verify_ray_geometry(origin, entrance, ray, bore, z0,
            precision=precision, endpoint_tolerance=endpoint_tolerance)
        with localcontext() as context:
            context.prec = precision
            source_z, entrance_z, exit_z = map(decimal, (origin[2], z0, ray.point[2]))
            if geometry["valid"] and source_z < entrance_z <= exit_z:
                ratio = (exit_z-source_z)/(entrance_z-source_z)
                determinant = ratio*ratio
                if determinant > decimal(jacobian_floor):
                    return {"valid": True, "reason": "ok", "maslov": 0,
                        "det_sign": 1, "determinant": determinant,
                        "determinant_bounds": (determinant, determinant),
                        "Q": [[ratio, D(0)], [D(0), ratio]],
                        "backend": "analytic_free",
                        "certificate": {"certified": True, "reason": "exact_free_fixed_z_map",
                                        "mp_geometry_verified": True}, **geometry}
    certificate = certify_transport(origin, entrance, ray, bore, z0, precision=precision,
        endpoint_tolerance=endpoint_tolerance, jacobian_floor=jacobian_floor)
    if certificate["certified"]:
        return {"valid": True, "reason": "ok", "maslov": certificate["maslov"],
                "det_sign": certificate["det_sign"],
                "determinant": D.from_float(certificate["determinant"]),
                "determinant_bounds": tuple(D.from_float(v) for v in certificate["determinant_interval"]),
                "backend": "interval_drt", "certificate": certificate,
                **certificate["geometry"]}
    result = transport_ray(origin, entrance, ray, bore, z0, precision=precision,
                           endpoint_tolerance=endpoint_tolerance)
    result.update(backend="decimal_fallback", certificate=certificate)
    if result.get("determinant") is not None:
        result["determinant_bounds"] = (result["determinant"], result["determinant"])
    return result

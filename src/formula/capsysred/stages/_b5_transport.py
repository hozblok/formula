"""Fixed-source ray derivatives and free-flight caustics from full paths."""

import math

import numpy as np

from ..walls.wall_cylinder import CylinderWall
from ..walls.wall_polygon import PolygonWall
from ..walls.wall_torus import TorusWall


def _det2(a):
    return a[..., 0, 0]*a[..., 1, 1]-a[..., 0, 1]*a[..., 1, 0]


def count_free_caustics(q, velocity_derivative, distance):
    """Count roots of det(Q + dz V), with multiplicity, inside a free flight."""
    q = np.asarray(q, dtype=float)
    d = np.asarray(distance, dtype=float)[..., None, None]*velocity_derivative
    a, c = _det2(d), _det2(q)
    b = q[..., 0, 0]*d[..., 1, 1]+d[..., 0, 0]*q[..., 1, 1]
    b -= q[..., 0, 1]*d[..., 1, 0]+d[..., 0, 1]*q[..., 1, 0]
    scale = np.maximum.reduce([np.abs(a), np.abs(b), np.abs(c)])
    tol = 128*np.finfo(float).eps*scale
    linear = np.abs(a) <= tol
    roots = np.full((*np.shape(a), 2), np.nan)
    simple = linear & (np.abs(b) > tol)
    np.divide(-c, b, out=roots[..., 0], where=simple)
    disc = b*b-4*a*c
    dtol = 256*np.finfo(float).eps*(b*b+np.abs(4*a*c))
    real = (~linear) & (disc >= -dtol)
    radical = np.sqrt(np.maximum(disc, 0))
    stable = -.5*(b+np.copysign(radical, b))
    np.divide(stable, a, out=roots[..., 0], where=real)
    np.divide(c, stable, out=roots[..., 1], where=real & (stable != 0))
    zero = real & (stable == 0)
    roots[..., 1] = np.where(zero, roots[..., 0], roots[..., 1])
    edge_tol = 1e-10
    inside = (roots > edge_tol) & (roots < 1-edge_tol)
    ambiguous = np.any((np.abs(roots) <= edge_tol) | (np.abs(roots-1) <= edge_tol), axis=-1)
    ambiguous |= (scale == 0) | (~np.isfinite(scale))
    return np.sum(inside, axis=-1).astype(np.int64), ambiguous


def _arg_ratio(num, den):
    """Principal argument of num / den, 0 where the ratio is undefined."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.angle(num/den)
    return np.where(np.isfinite(out), out, 0.0)


def det_path_increment(A, B, t=1.0):
    """Continuous change of arg det(A + s B) for s from 0 to t (complex 2x2 stacks): the
    sum over the roots r of the quadratic of the principal arg((t - r) / (-r)).  Returns
    (increment, a root lies on the segment)."""
    A, B = np.asarray(A, complex), np.asarray(B, complex)
    a, c = _det2(B), _det2(A)
    b = A[..., 0, 0]*B[..., 1, 1]+B[..., 0, 0]*A[..., 1, 1]-A[..., 0, 1]*B[..., 1, 0]-B[..., 0, 1]*A[..., 1, 0]
    t = np.broadcast_to(np.asarray(t, float), np.shape(c))
    tol = 64*np.finfo(float).eps*np.maximum.reduce([np.abs(a)*t*t, np.abs(b)*t, np.abs(c), np.full(np.shape(c), 1e-300)])
    quadratic = np.abs(a)*t*t > tol
    linear = ~quadratic & (np.abs(b)*t > tol)
    out = np.zeros(np.shape(c))
    hit = np.zeros(np.shape(c), bool)

    def add(root, where):
        nonlocal out, hit
        out = np.where(where, out+_arg_ratio(t-root, -root), out)
        on = (np.abs(root.imag) <= 1e-12*np.abs(root)) & (root.real >= 0) & (root.real <= t)
        hit |= where & on

    with np.errstate(divide="ignore", invalid="ignore"):
        add(-c/b, linear)
        disc = np.sqrt(b*b-4*a*c)
        big = np.where(np.abs(b+disc) >= np.abs(b-disc), -(b+disc), -(b-disc))/(2*a)
        add(big, quadratic)
        add(c/(a*big), quadratic)
    return out, hit


def det_phase_increment(Q, P, V, dz, gamma):
    """Continuous change of arg det(gamma (Q + s V) + i P) over a free flight s = 0..dz."""
    Q, P, V = (np.asarray(m, float) for m in (Q, P, V))
    return det_path_increment(gamma*Q+1j*P, gamma*V, dz)


def gamma_phase_shift(Q, P, gamma_from, gamma_to):
    """Continuous change of arg det(g Q + i P) for g from gamma_from to gamma_to."""
    Q, P = np.asarray(Q, float), np.asarray(P, float)
    return det_path_increment(gamma_from*Q+1j*P, (gamma_to-gamma_from)*Q, 1.0)[0]


def _fixed_z(r, u):
    return r[:, :2]-u[:, :2, None]*(r[:, 2, :]/u[:, 2, None])[:, None, :]


def _mirror(normal):
    """2x2 transverse reflection across the wall tangent (det -1)."""
    n = normal[:, :2]/np.linalg.norm(normal[:, :2], axis=1)[:, None]
    return np.eye(2)-2.0*n[:, :, None]*n[:, None, :]


def is_polygon(bore):
    """Regular-polygon bore: config dicts carry kind, raw YAML dicts only `sides`."""
    return bore.get("kind") == "polygon" or bore.get("sides") is not None


def polygon_faces(bore):
    """Outward unit normals (n, 2) of a regular polygon: face k at rotation + 2 pi k / n."""
    n = int(bore["sides"])
    rot = (float(bore["rotation"]) if bore.get("rotation") is not None
           else math.radians(float(bore.get("rotation_deg", 0.0))))
    phi = rot + 2.0 * np.pi * np.arange(n) / n
    return np.column_stack([np.cos(phi), np.sin(phi)])


def polygon_apothems(bore):
    """Per-face apothems (n,): `apothems` when given, else the radius on every face."""
    aps = bore.get("apothems")
    if aps is None:
        return np.full(int(bore["sides"]), float(bore["radius"]))
    return np.array([float(a) for a in aps])


def bore_inside(points, bore, tol=1e-8):
    """Transverse points (..., 2) inside the bore: circle test, or apothem test per face."""
    points = np.asarray(points, dtype=float)
    center = np.asarray(bore["center"], dtype=float)
    radius = float(bore["radius"])
    if is_polygon(bore):
        aps = polygon_apothems(bore)
        return np.max((points - center) @ polygon_faces(bore).T - aps, axis=-1) <= tol * aps.max()
    return np.linalg.norm(points - center, axis=-1) <= radius * (1 + tol)


def bore_area(bore):
    """Cross-section area: pi r^2, or the shoelace sum over the polygon vertices."""
    radius = float(bore["radius"])
    if is_polygon(bore):
        faces, aps = polygon_faces(bore), polygon_apothems(bore)
        n = len(faces)
        verts = np.array([np.linalg.solve(np.stack([faces[k], faces[(k + 1) % n]]), [aps[k], aps[(k + 1) % n]])
                          for k in range(n)])
        x, y = verts[:, 0], verts[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return math.pi * radius * radius


def wall_normal_derivative(points, bore, z0):
    """Outward unit normal and its Cartesian derivative for a circular, torus or polygon bore."""
    points = np.asarray(points, dtype=float)
    center = np.asarray(bore["center"], dtype=float)
    eye = np.eye(3)
    if is_polygon(bore):
        faces = polygon_faces(bore)
        dots = (points[..., :2] - center) @ faces.T
        k = np.argmax(dots - polygon_apothems(bore), axis=-1)
        scale = np.take_along_axis(dots, k[..., None], -1)[..., 0]
        grad, hess = PolygonWall.gradient_hessian(faces[k, 0], faces[k, 1], scale)
    elif bore.get("kind", "cylinder") == "cylinder" and not bore.get("bend"):
        grad, hess = CylinderWall.gradient_hessian(points[..., 0] - center[0], points[..., 1] - center[1])
    elif bore.get("kind") == "torus" or bore.get("bend"):
        bend = bore["bend"]
        radius = float(bend["radius"])
        toward = np.asarray(bend["toward"], dtype=float)
        toward /= np.linalg.norm(toward)
        tangent = np.r_[toward, 0.]
        binormal = np.array([-toward[1], toward[0], 0.])
        local = points-np.r_[center, float(z0)]
        xi, eta, zeta = local@tangent, local@binormal, local[..., 2]
        rho = np.hypot(radius-xi, zeta)
        dr = (-2*radius*xi+xi*xi+zeta*zeta)/(rho+radius)
        radial = (xi-radius)[..., None]*tangent
        radial[..., 2] = zeta
        grad, hess = TorusWall.gradient_hessian(
            radial[..., 0], radial[..., 1], radial[..., 2], eta,
            binormal[0], binormal[1], rho, dr / rho)
    else:
        raise ValueError("B5 transport supports cylinder, torus and polygon walls only")
    raw = np.empty_like(points)
    raw[..., 0], raw[..., 1], raw[..., 2] = grad
    hxx, hxy, hxz, hyy, hyz, hzz = hess
    shape = np.broadcast_shapes(*(np.shape(value) for value in hess))
    hessian = np.empty((*shape, 3, 3))
    hessian[..., 0, 0], hessian[..., 1, 1], hessian[..., 2, 2] = hxx, hyy, hzz
    hessian[..., 0, 1] = hessian[..., 1, 0] = hxy
    hessian[..., 0, 2] = hessian[..., 2, 0] = hxz
    hessian[..., 1, 2] = hessian[..., 2, 1] = hyz
    length = np.linalg.norm(raw, axis=-1)
    normal = raw/length[..., None]
    projector = eye-normal[..., :, None]*normal[..., None, :]
    derivative = np.einsum("...ij,...jk->...ik", projector, hessian)/length[..., None, None]
    return normal, derivative, length


def reflect_variations(direction, position_derivative, direction_derivative, normal, normal_derivative):
    """Differentiate the hit time and specular reflection at a smooth wall."""
    dot = np.sum(direction*normal, axis=-1)
    dt = -np.einsum("...i,...ij->...j", normal, position_derivative)/dot[..., None]
    hit = position_derivative+direction[..., :, None]*dt[..., None, :]
    dn = np.einsum("...ij,...jk->...ik", normal_derivative, hit)
    ddot = np.einsum("...i,...ij->...j", normal, direction_derivative)
    ddot += np.einsum("...i,...ij->...j", direction, dn)
    outgoing = direction-2*dot[..., None]*normal
    du = direction_derivative-2*(normal[..., :, None]*ddot[..., None, :]+dot[..., None, None]*dn)
    fixed = hit-outgoing[..., :, None]*(hit[..., 2, :]/outgoing[..., 2, None])[..., None, :]
    return outgoing, fixed, du


def _velocity_derivative(direction, derivative):
    slope = direction[..., :2]/direction[..., 2, None]
    return (derivative[..., :2, :]-slope[..., :, None]*derivative[..., 2, None, :])/direction[..., 2, None, None]


def transport_mode(mode, cap, target_z, gamma_ref=1.0):
    """Return Q=dX/dq, P=du_perp/dq, V = dQ/dz of the final flight, caustic count, and
    the continuous argument of (-1)^bounces det(gamma_ref Q + i P) along every ray: exact
    over free flights, and over a bounce along the path from the mirrored incident state
    (M Q, M P) to the reflected one (a thin lens on flat-in-z walls), so it stays defined
    on exact caustics."""
    source = np.asarray(mode["origin"], dtype=float)
    target = np.column_stack([mode["points"], np.full(len(mode["points"]), target_z)])
    count = len(target)
    refs = np.asarray([len(path) for path in mode["refl"]])
    first = np.array([path[0] if path else target[i] for i, path in enumerate(mode["refl"])], dtype=float)
    initial = first-source
    initial /= np.linalg.norm(initial, axis=1)[:, None]
    entrance = source[:2]+initial[:, :2]/initial[:, 2, None]*(float(cap.z0)-source[2])
    within = np.column_stack([bore_inside(entrance, b) for b in cap.bores])
    bore_ids = np.where(within.sum(axis=1) == 1, within.argmax(axis=1), -1)
    q_all = np.full((count, 2, 2), np.nan)
    p_all = np.full_like(q_all, np.nan)
    maslov = np.zeros(count, dtype=np.int64)
    phase = np.zeros(count)
    v_all = np.full_like(q_all, np.nan)
    phase_hit = np.zeros(count, dtype=bool)
    phase_jump = np.zeros(count)
    valid = bore_ids >= 0
    endpoint = np.zeros(count, dtype=bool)
    direction_error = np.zeros(count)
    position_error = np.zeros(count)
    wall_error = np.zeros(count)
    grazing = np.full(count, np.inf)
    for bore_index, bore in enumerate(cap.bores):
        for nref in np.unique(refs[bore_ids == bore_index]):
            ids = np.flatnonzero((bore_ids == bore_index) & (refs == nref))
            u = initial[ids].copy()
            current = np.column_stack([entrance[ids], np.full(len(ids), float(cap.z0))])
            r = np.broadcast_to(np.eye(3)[:, :2], (len(ids), 3, 2)).copy()
            distance = (float(cap.z0)-source[2])/u[:, 2]
            du = (r-u[:, :, None]*u[:, None, :2])/distance[:, None, None]
            path = np.asarray([mode["refl"][i] for i in ids], dtype=float) if nref else None
            theta = np.angle(_det2(gamma_ref*_fixed_z(r, u)+1j*du[:, :2]))
            for j in range(nref+1):
                destination = path[:, j] if j < nref else target[ids]
                dz = destination[:, 2]-current[:, 2]
                v = _velocity_derivative(u, du)
                crosses, ambiguous = count_free_caustics(r[:, :2], v, dz)
                maslov[ids] += crosses
                endpoint[ids] |= ambiguous
                step, hit = det_phase_increment(_fixed_z(r, u), du[:, :2], v, dz, gamma_ref)
                theta = theta+step
                phase_hit[ids] |= hit
                flight = dz/u[:, 2]
                reached = current+flight[:, None]*u
                position_error[ids] = np.maximum(position_error[ids], np.linalg.norm(reached-destination, axis=1))
                r = r+flight[:, None, None]*du
                if j < nref:
                    normal, dn, wall_distance = wall_normal_derivative(destination, bore, cap.z0)
                    wall_error[ids] = np.maximum(wall_error[ids], np.abs(wall_distance-float(bore["radius"])))
                    grazing[ids] = np.minimum(grazing[ids], np.abs(np.sum(u*normal, axis=1)))
                    mirror = _mirror(normal)
                    q_in, p_in = mirror@_fixed_z(r, u), mirror@du[:, :2]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        u, r, du = reflect_variations(u, r, du, normal, dn)
                    jump, hit = det_path_increment(gamma_ref*q_in+1j*p_in,
                                                   gamma_ref*(r[:, :2]-q_in)+1j*(du[:, :2]-p_in))
                    theta = theta+jump
                    phase_hit[ids] |= hit
                    phase_jump[ids] = np.maximum(phase_jump[ids], np.abs(jump))
                else:
                    r -= u[:, :, None]*(r[:, 2, :]/u[:, 2, None])[:, None, :]
                    v_all[ids] = v
                current = destination
            q_all[ids] = r[:, :2]
            p_all[ids] = du[:, :2]
            phase[ids] = theta
            direction_error[ids] = np.linalg.norm(u[:, :2]-mode["directions"][ids], axis=1)
    product = np.einsum("nji,njk->nik", q_all, p_all)
    defect = np.linalg.norm(product-product.swapaxes(1, 2), axis=(1, 2))
    defect /= np.maximum(np.linalg.norm(q_all, axis=(1, 2))*np.linalg.norm(p_all, axis=(1, 2)), 1e-300)
    finite = np.all(np.isfinite(q_all), axis=(1, 2)) & np.all(np.isfinite(p_all), axis=(1, 2))
    valid &= finite & (direction_error < 1e-8) & (position_error < 1e-8)
    valid &= (wall_error < 1e-9) & (grazing > 1e-10) & (defect < 1e-7)
    continuous_valid = valid & ~phase_hit
    valid &= ~endpoint
    diagnostics = {
        "ray_count": count, "valid_count": int(valid.sum()), "unassigned_bore_count": int((bore_ids < 0).sum()),
        "endpoint_caustic_count": int(endpoint.sum()), "nonfinite_count": int((~finite).sum()),
        "direction_error_max": float(np.max(direction_error, initial=0)),
        "position_error_max_m": float(np.max(position_error, initial=0)),
        "wall_error_max_m": float(np.max(wall_error, initial=0)),
        "lagrangian_defect_max": float(np.nanmax(defect, initial=0)),
        "maslov_histogram": {str(m): int(np.sum(maslov[valid] == m)) for m in np.unique(maslov[valid])},
        "maslov_convention": "positive free-flight focal multiplicities; wall orientation jumps excluded",
        "continuous_valid_count": int(continuous_valid.sum()),
        "phase_root_on_path_count": int(np.sum(phase_hit & finite)),
        "phase_bounce_jump_max_rad": float(np.max(phase_jump[finite], initial=0)),
    }
    return {"entrance": entrance, "bore": bore_ids, "Q": q_all, "P": p_all, "V": v_all,
            "maslov": maslov, "phase": phase, "gamma_ref": float(gamma_ref),
            "valid": valid, "continuous_valid": continuous_valid, "diagnostics": diagnostics}

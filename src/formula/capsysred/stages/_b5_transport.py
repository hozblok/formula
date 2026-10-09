"""Fixed-source ray derivatives and free-flight caustics from full paths."""

import numpy as np


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


def wall_normal_derivative(points, bore, z0):
    """Outward unit normal and its Cartesian derivative for a circular bore."""
    points = np.asarray(points, dtype=float)
    center = np.asarray(bore["center"], dtype=float)
    eye = np.eye(3)
    if bore.get("kind", "cylinder") == "cylinder" and not bore.get("bend"):
        raw = points-np.r_[center, 0.]
        raw[..., 2] = 0
        hessian = np.broadcast_to(np.diag([1., 1., 0.]), (*points.shape[:-1], 3, 3))
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
        radial /= rho[..., None]
        rr = radial[..., :, None]*radial[..., None, :]
        ring_plane = np.outer(tangent, tangent)+np.diag([0., 0., 1.])
        hessian = rr+(dr/rho)[..., None, None]*(ring_plane-rr)+np.outer(binormal, binormal)
        raw = dr[..., None]*radial+eta[..., None]*binormal
    else:
        raise ValueError("B5 transport supports cylinder and torus walls only")
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


def transport_mode(mode, cap, target_z):
    """Return Q=dX/dq, P=du_perp/dq and caustic count for every archived ray."""
    source = np.asarray(mode["origin"], dtype=float)
    target = np.column_stack([mode["points"], np.full(len(mode["points"]), target_z)])
    count = len(target)
    refs = np.asarray([len(path) for path in mode["refl"]])
    first = np.array([path[0] if path else target[i] for i, path in enumerate(mode["refl"])], dtype=float)
    initial = first-source
    initial /= np.linalg.norm(initial, axis=1)[:, None]
    entrance = source[:2]+initial[:, :2]/initial[:, 2, None]*(float(cap.z0)-source[2])
    centers = np.array([b["center"] for b in cap.bores], dtype=float)
    radii = np.array([b["radius"] for b in cap.bores], dtype=float)
    within = np.linalg.norm(entrance[:, None, :]-centers[None], axis=-1) <= radii[None]*(1+1e-8)
    bore_ids = np.where(within.sum(axis=1) == 1, within.argmax(axis=1), -1)
    q_all = np.full((count, 2, 2), np.nan)
    p_all = np.full_like(q_all, np.nan)
    maslov = np.zeros(count, dtype=np.int64)
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
            for j in range(nref+1):
                destination = path[:, j] if j < nref else target[ids]
                dz = destination[:, 2]-current[:, 2]
                v = _velocity_derivative(u, du)
                crosses, ambiguous = count_free_caustics(r[:, :2], v, dz)
                maslov[ids] += crosses
                endpoint[ids] |= ambiguous
                flight = dz/u[:, 2]
                reached = current+flight[:, None]*u
                position_error[ids] = np.maximum(position_error[ids], np.linalg.norm(reached-destination, axis=1))
                r = r+flight[:, None, None]*du
                if j < nref:
                    normal, dn, wall_distance = wall_normal_derivative(destination, bore, cap.z0)
                    wall_error[ids] = np.maximum(wall_error[ids], np.abs(wall_distance-float(bore["radius"])))
                    grazing[ids] = np.minimum(grazing[ids], np.abs(np.sum(u*normal, axis=1)))
                    with np.errstate(divide="ignore", invalid="ignore"):
                        u, r, du = reflect_variations(u, r, du, normal, dn)
                else:
                    r -= u[:, :, None]*(r[:, 2, :]/u[:, 2, None])[:, None, :]
                current = destination
            q_all[ids] = r[:, :2]
            p_all[ids] = du[:, :2]
            direction_error[ids] = np.linalg.norm(u[:, :2]-mode["directions"][ids], axis=1)
    product = np.einsum("nji,njk->nik", q_all, p_all)
    defect = np.linalg.norm(product-product.swapaxes(1, 2), axis=(1, 2))
    defect /= np.maximum(np.linalg.norm(q_all, axis=(1, 2))*np.linalg.norm(p_all, axis=(1, 2)), 1e-300)
    finite = np.all(np.isfinite(q_all), axis=(1, 2)) & np.all(np.isfinite(p_all), axis=(1, 2))
    valid &= finite & ~endpoint & (direction_error < 1e-8) & (position_error < 1e-8)
    valid &= (wall_error < 1e-9) & (grazing > 1e-10) & (defect < 1e-7)
    diagnostics = {
        "ray_count": count, "valid_count": int(valid.sum()), "unassigned_bore_count": int((bore_ids < 0).sum()),
        "endpoint_caustic_count": int(endpoint.sum()), "nonfinite_count": int((~finite).sum()),
        "direction_error_max": float(np.max(direction_error, initial=0)),
        "position_error_max_m": float(np.max(position_error, initial=0)),
        "wall_error_max_m": float(np.max(wall_error, initial=0)),
        "lagrangian_defect_max": float(np.nanmax(defect, initial=0)),
        "maslov_histogram": {str(m): int(np.sum(maslov[valid] == m)) for m in np.unique(maslov[valid])},
        "maslov_convention": "positive free-flight focal multiplicities; wall orientation jumps excluded",
    }
    return {"entrance": entrance, "bore": bore_ids, "Q": q_all, "P": p_all,
            "maslov": maslov, "valid": valid, "diagnostics": diagnostics}

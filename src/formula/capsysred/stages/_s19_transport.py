"""Small-matrix Decimal ray variations and Maslov history for Stage 19."""

from decimal import Decimal, localcontext


D = Decimal


def decimal(value):
    return value if isinstance(value, D) else D(str(value))


def dot(a, b):
    return sum((x*y for x, y in zip(a, b)), D(0))


def norm(a):
    return dot(a, a).sqrt()


def determinant(a):
    return a[0][0]*a[1][1] - a[0][1]*a[1][0]


def count_free_caustics(q, velocity_derivative, distance, *, precision=64,
                        endpoint_tolerance="1e-20"):
    """Count focal multiplicities strictly inside one forward free flight."""
    with localcontext() as context:
        context.prec = precision
        q = [[decimal(v) for v in row] for row in q]
        v = [[decimal(x)*decimal(distance) for x in row] for row in velocity_derivative]
        a, c = determinant(v), determinant(q)
        b = q[0][0]*v[1][1]+v[0][0]*q[1][1]-q[0][1]*v[1][0]-v[0][1]*q[1][0]
        scale = max(abs(a), abs(b), abs(c))
        if not scale:
            return 0, True
        numerical = D(10) ** (-(precision-8))
        edge = decimal(endpoint_tolerance)
        if abs(a) <= numerical*scale:
            roots = [-c/b] if abs(b) > numerical*scale else []
        else:
            discriminant = b*b-4*a*c
            if discriminant < 0:
                # A numerically split double focus outside this segment is
                # irrelevant (a diverging spherical wave has such a root
                # behind the entrance). Only an in-segment event is ambiguous.
                vertex = -b/(2*a)
                uncertain = (abs(discriminant) <= numerical*(b*b+abs(4*a*c))
                             and -edge <= vertex <= 1+edge)
                return 0, uncertain
            radical = discriminant.sqrt()
            stable = -(b+(radical if b >= 0 else -radical))/2
            roots = [stable/a, c/stable] if stable else [-b/(2*a)]*2
        ambiguous = any(abs(r) <= edge or abs(r-1) <= edge for r in roots)
        return sum(edge < r < 1-edge for r in roots), ambiguous


def wall_normal_derivative(point, bore, z0):
    """Unit normal and its spatial derivative for a cylinder or torus."""
    point = tuple(map(decimal, point))
    center = tuple(map(decimal, bore["center"]))
    zero, one = D(0), D(1)
    eye = [[one if i == j else zero for j in range(3)] for i in range(3)]
    bend = bore.get("bend")
    if not bend and bore.get("kind", "cylinder") == "cylinder":
        raw = [point[0]-center[0], point[1]-center[1], zero]
        hessian = [[one if i == j and i < 2 else zero for j in range(3)] for i in range(3)]
    elif bend:
        radius = decimal(bend["radius"])
        toward = tuple(map(decimal, bend["toward"]))
        length = norm(toward)
        tangent = [toward[0]/length, toward[1]/length, zero]
        binormal = [-tangent[1], tangent[0], zero]
        local = [point[0]-center[0], point[1]-center[1], point[2]-decimal(z0)]
        xi, eta, zeta = dot(local, tangent), dot(local, binormal), local[2]
        rho = ((radius-xi)**2+zeta*zeta).sqrt()
        dr = (-2*radius*xi+xi*xi+zeta*zeta)/(rho+radius)
        radial = [(xi-radius)*tangent[0]/rho, (xi-radius)*tangent[1]/rho, zeta/rho]
        raw = [dr*radial[i]+eta*binormal[i] for i in range(3)]
        hessian = []
        for i in range(3):
            row = []
            for j in range(3):
                rr = radial[i]*radial[j]
                plane = tangent[i]*tangent[j]+(one if i == j == 2 else zero)
                row.append(rr+(dr/rho)*(plane-rr)+binormal[i]*binormal[j])
            hessian.append(row)
    else:
        raise ValueError("Stage 19 transport supports circular cylinder and torus walls")
    length = norm(raw)
    if not length:
        raise ValueError("undefined wall normal")
    normal = [x/length for x in raw]
    projector = [[eye[i][j]-normal[i]*normal[j] for j in range(3)] for i in range(3)]
    derivative = [[sum((projector[i][l]*hessian[l][j] for l in range(3)), zero)/length
                   for j in range(3)] for i in range(3)]
    return normal, derivative, length


def reflect_variations(direction, positions, directions, normal, normal_derivative):
    """Differentiate the wall hit and reflection, then return to fixed z."""
    udot = dot(direction, normal)
    dt = [-sum((normal[i]*positions[i][j] for i in range(3)), D(0))/udot for j in range(2)]
    hit = [[positions[i][j]+direction[i]*dt[j] for j in range(2)] for i in range(3)]
    dn = [[sum((normal_derivative[i][l]*hit[l][j] for l in range(3)), D(0))
           for j in range(2)] for i in range(3)]
    ddot = [sum((normal[i]*directions[i][j]+direction[i]*dn[i][j] for i in range(3)), D(0))
            for j in range(2)]
    outgoing = [direction[i]-2*udot*normal[i] for i in range(3)]
    du = [[directions[i][j]-2*(normal[i]*ddot[j]+udot*dn[i][j]) for j in range(2)]
          for i in range(3)]
    if outgoing[2] <= 0:
        raise ValueError("backward ray after reflection")
    fixed = [[hit[i][j]-outgoing[i]*hit[2][j]/outgoing[2] for j in range(2)] for i in range(3)]
    return outgoing, fixed, du


def transport_ray(origin, entrance, ray, bore, z0, *, precision=64,
                  endpoint_tolerance="1e-20"):
    """Use a real traced path; endpoint focal events remain unresolved."""
    with localcontext() as context:
        context.prec = precision
        source = list(map(decimal, origin))
        start = [decimal(entrance[0]), decimal(entrance[1]), decimal(z0)]
        initial = [start[i]-source[i] for i in range(3)]
        distance = norm(initial)
        u = [x/distance for x in initial]
        zero, one = D(0), D(1)
        r = [[one if i == j else zero for j in range(2)] for i in range(3)]
        du = [[(r[i][j]-u[i]*u[j])/distance for j in range(2)] for i in range(3)]
        reflections = [list(map(decimal, p)) for p, _ in ray.reflections]
        destinations = reflections+[list(map(decimal, ray.point))]
        maslov, ambiguous, grazing = 0, False, one
        position_error, wall_error = zero, zero
        for index, destination in enumerate(destinations):
            dz = destination[2]-start[2]
            if dz < -decimal(endpoint_tolerance) or u[2] <= 0:
                return dict(valid=False, reason="backward_path", maslov=maslov)
            velocity = [[(du[i][j]-u[i]*du[2][j]/u[2])/u[2] for j in range(2)] for i in range(2)]
            crosses, uncertain = count_free_caustics(r[:2], velocity, dz, precision=precision,
                                                     endpoint_tolerance=endpoint_tolerance)
            maslov += crosses
            ambiguous |= uncertain
            flight = dz/u[2]
            position_error = max(position_error, norm([start[i]+flight*u[i]-destination[i] for i in range(3)]))
            r = [[r[i][j]+flight*du[i][j] for j in range(2)] for i in range(3)]
            if index < len(reflections):
                normal, derivative, wall_radius = wall_normal_derivative(destination, bore, z0)
                wall_error = max(wall_error, abs(wall_radius-decimal(bore["radius"])))
                incidence = abs(dot(u, normal))
                grazing = min(grazing, incidence)
                if incidence <= decimal(endpoint_tolerance):
                    return dict(valid=False, reason="grazing", maslov=maslov)
                try:
                    u, r, du = reflect_variations(u, r, du, normal, derivative)
                except ValueError:
                    return dict(valid=False, reason="backward_path", maslov=maslov)
            else:
                r = [[r[i][j]-u[i]*r[2][j]/u[2] for j in range(2)] for i in range(3)]
            start = destination
        direction_error = norm([u[i]-decimal(ray.direction[i]) for i in range(3)])
        q, p = r[:2], du[:2]
        sym01 = sum((q[i][0]*p[i][1] for i in range(2)), zero)
        sym10 = sum((q[i][1]*p[i][0] for i in range(2)), zero)
        scale = norm([x for row in q for x in row])*norm([x for row in p for x in row])
        defect = abs(sym01-sym10)/scale if scale else zero
        tolerance = max(D("1e-24"), D(10) ** (-(precision-12)))
        valid = not ambiguous and direction_error <= tolerance and defect <= tolerance
        valid &= position_error <= tolerance and wall_error <= tolerance
        reason = "ok" if valid else ("endpoint_caustic" if ambiguous else "transport_residual")
        return dict(valid=valid, reason=reason, maslov=maslov, Q=q, P=p,
                    determinant=determinant(q), direction_error=direction_error,
                    position_error=position_error, wall_error=wall_error,
                    lagrangian_defect=defect, minimum_grazing=grazing)

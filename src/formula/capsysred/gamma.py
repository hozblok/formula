"""General-astigmatism beam algebra (Arnaud-Kogelnik), float64.

The beamlet parameter is the complex symmetric 2x2 matrix Q = Gamma^-1
stored as (q_xx, q_xy, q_yy); the scalar q is Q = q*I. Tensor ABCD in the
lab transverse (x, y) frame: drift L is Q += L*I; a bounce is Gamma -= P,
then the mirror Q -> M Q M, M = I - 2 n n^T with n the transverse trace of
the wall normal N. P is the wall's second fundamental form II seen through
the lab displacements' footprints along the incident direction u:

    P_ij = 2 |N.u| II(d_i, d_j),    d_i = e_i - (N.e_i)/(N.u) * u

A meridional bounce keeps P diagonal in the (n, t) frame, 1/f_t = 2 k_m/s
and 1/f_s = 2 s k_s (s = |N.u|, k_m, k_s the meridional and azimuthal
curvatures) up to O(s^2); skew bounces add the off-diagonal twist.
The on-axis amplitude over a drift is sqrt(det(Q)/det(Q+L)) on the
continuous branch, factored through the roots of det(Q + s) (a thin lens
adds no on-axis factor). For the scalar case it reduces to q0/q — the
(w0/w)*exp(i*Gouy) of the stage-11a deposit.
"""

import cmath
import math


def inv2(m):
    xx, xy, yy = m
    det = xx * yy - xy * xy
    return (yy / det, -xy / det, xx / det)


def det2(m):
    return m[0] * m[2] - m[1] * m[1]


def reflect(q, phi, pxx, pxy, pyy):
    """One bounce: Gamma -= P (lab frame), then the mirror Q -> M Q M,
    M = I - 2 n n^T, n = (cos phi, sin phi); phi NaN = unknown normal, no-op."""
    if math.isnan(phi):
        return q
    if pxx or pxy or pyy:
        g = inv2(q)
        q = inv2((g[0] - pxx, g[1] - pxy, g[2] - pyy))
    c, s = math.cos(phi), math.sin(phi)
    c2, s2 = c * c - s * s, 2.0 * c * s
    a, b, d = q
    return (c2 * c2 * a + 2.0 * c2 * s2 * b + s2 * s2 * d,
            c2 * s2 * (a - d) + (s2 * s2 - c2 * c2) * b,
            s2 * s2 * a - 2.0 * c2 * s2 * b + c2 * c2 * d)


def propagate(zr, segments, lenses):
    """(Q at screen, on-axis amplitude factor) through the drift/bounce
    chain; len(segments) == len(lenses) + 1, waist i*zr*I at the source.

    zr is the isotropic launch Rayleigh range, or (zr_t, zr_s, psi) — an
    elliptic waist with the zr_t axis at azimuth psi.

    The drift amplitude det(Q)/det(Q+L) is exact in closed form:
    det(Q + s) = (s - s1)(s - s2), and over the real segment each linear
    factor's argument turns by less than pi (the root is off the axis),
    so the continuous branch of the square root is the product of
    principal roots 1/sqrt(1 - L/s_i) — exact through any focus."""
    if isinstance(zr, tuple):
        zrt, zrs, psi = zr
        if zrt == zrs:
            q = (complex(0.0, zrt), 0j, complex(0.0, zrt))
        else:
            c, sn = math.cos(psi), math.sin(psi)
            q = (complex(0.0, zrt * c * c + zrs * sn * sn),
                 complex(0.0, (zrt - zrs) * c * sn),
                 complex(0.0, zrt * sn * sn + zrs * c * c))
    else:
        q = (complex(0.0, zr), 0j, complex(0.0, zr))
    amp = 1.0 + 0j
    for j, seg in enumerate(segments):
        tr = q[0] + q[2]
        disc = cmath.sqrt(tr * tr - 4.0 * det2(q))
        for root in (0.5 * (disc - tr), -0.5 * (disc + tr)):
            amp *= 0.0 if root == 0 else 1.0 / cmath.sqrt(1.0 - seg / root)
        q = (q[0] + seg, q[1], q[2] + seg)
        if j < len(lenses):
            q = reflect(q, *lenses[j])
    return q, amp


# walls whose bounce lens here is exact (polygon: exactly flat); anything
# else — implicit, future kinds — deposits flat, the stage-11a scalar model
EXACT_KINDS = frozenset(("cylinder", "revolution", "torus", "funnel",
                         "polygon"))


def _curved_lens(grad, hess, out):
    """(phi, pxx, pxy, pyy) from the gradient and Hessian (xx, xy, xz, yy,
    yz, zz) of an implicit wall F growing outward, at the hit; out is the
    outgoing direction."""
    gx, gy, gz = grad
    hxx, hxy, hxz, hyy, hyz, hzz = hess
    ox, oy, oz = out
    gn = math.sqrt(gx * gx + gy * gy + gz * gz)
    nx, ny, nz = gx / gn, gy / gn, gz / gn
    on = math.sqrt(ox * ox + oy * oy + oz * oz)
    no = (nx * ox + ny * oy + nz * oz) / on
    # incident direction u: the outgoing one mirrored about N
    ux = ox / on - 2.0 * no * nx
    uy = oy / on - 2.0 * no * ny
    uz = oz / on - 2.0 * no * nz
    hx = hxx * ux + hxy * uy + hxz * uz
    hy = hxy * ux + hyy * uy + hyz * uz
    huu = ux * hx + uy * hy + uz * (hxz * ux + hyz * uy + hzz * uz)
    # H(d_i, d_j) with d_i = e_i - a_i*u, a_i = N_i/(N.u), N.u = -no
    ax, ay = -nx / no, -ny / no
    k = 2.0 * abs(no) / gn
    return (math.atan2(ny, nx),
            k * (hxx - 2.0 * ax * hx + ax * ax * huu),
            k * (hxy - ay * hx - ax * hy + ax * ay * huu),
            k * (hyy - 2.0 * ay * hy + ay * ay * huu))


def _wall_lens(wall, x, y, z, out):
    kind = wall.kind
    if kind == "polygon":   # exactly flat: the hit face's outward normal
        cx, cy = wall._cxf, wall._cyf
        mx, my = max(wall._facesf, key=lambda m: m[0] * (x - cx) + m[1] * (y - cy))
        return (math.atan2(my, mx), 0.0, 0.0, 0.0)
    if kind == "cylinder":
        grad = (x - wall._cxf, y - wall._cyf, 0.0)
        hess = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    elif kind == "revolution":
        grad = (x - wall._cxf, y - wall._cyf,
                -0.5 * wall._c1f - wall._c2f * z)
        hess = (1.0, 0.0, 0.0, 1.0, 0.0, -wall._c2f)
    elif kind == "funnel":
        # F/2 = (U^2 + V^2 - r0^2 f^2)/2, (U, V) = (x, y) - center*g(z)
        zr = z - wall._z0f
        gg = 1.0 + zr * (wall._agf + zr * wall._bgf)
        ff = 1.0 + zr * (wall._aff + zr * wall._bff)
        gp = wall._agf + 2.0 * zr * wall._bgf
        fp = wall._aff + 2.0 * zr * wall._bff
        cx, cy, r02 = wall._cxf, wall._cyf, wall._r0f ** 2
        uu, vv = x - cx * gg, y - cy * gg
        cn = uu * cx + vv * cy
        hzz = ((cx * cx + cy * cy) * gp * gp - 2.0 * cn * wall._bgf
               - r02 * (fp * fp + 2.0 * ff * wall._bff))
        grad = (uu, vv, -cn * gp - r02 * ff * fp)
        hess = (1.0, 0.0, -cx * gp, 1.0, -cy * gp, hzz)
    elif kind == "torus":
        # F/2 = ((rho - R)^2 + s^2)/2, s along the ring plane normal n:
        # Hessian r*I + (1 - r)*(m m^T + n n^T), m = q/rho, r = (rho - R)/rho
        cx3, cy3, cz3 = wall._Cf
        nx, ny = wall._nf
        vx, vy, vz = x - cx3, y - cy3, z - cz3
        s = vx * nx + vy * ny
        qx, qy = vx - s * nx, vy - s * ny
        rho = math.sqrt(qx * qx + qy * qy + vz * vz)
        r = (rho - wall._Rf) / rho
        grad = (r * qx + s * nx, r * qy + s * ny, r * vz)
        mx, my, mz = qx / rho, qy / rho, vz / rho
        w = 1.0 - r
        hess = (r + w * (mx * mx + nx * nx), w * (mx * my + nx * ny),
                w * mx * mz, r + w * (my * my + ny * ny), w * my * mz,
                r + w * mz * mz)
    else:
        return (math.nan, 0.0, 0.0, 0.0)   # implicit/unknown: normal unknown
    return _curved_lens(grad, hess, out)


def _axis_dist2(wall, x, y, z):
    """Squared distance from the hit to the bore axis AT that z: funnel
    axes scale with g(z), torus axes bend — the entrance center mispicks
    the wall once bores taper or bend past the packing pitch."""
    kind = wall.kind
    if kind == "funnel":
        zr = z - wall._z0f
        gg = 1.0 + zr * (wall._agf + zr * wall._bgf)
        return (x - wall._cxf * gg) ** 2 + (y - wall._cyf * gg) ** 2
    if kind == "torus":
        cx3, cy3, cz3 = wall._Cf
        nx, ny = wall._nf
        vx, vy, vz = x - cx3, y - cy3, z - cz3
        dot = vx * nx + vy * ny
        rho = math.hypot(math.hypot(vx - dot * nx, vy - dot * ny), vz)
        return (rho - wall._Rf) ** 2 + dot * dot
    return (x - wall._cxf) ** 2 + (y - wall._cyf) ** 2


def bounce_lenses(optic, pts, outs):
    """(phi, pxx, pxy, pyy) per bounce from the wall shape at each hit point
    and the outgoing direction there. Optics without a wall collection and
    unsupported kinds use the scalar-q flat-wall fallback."""
    if not pts:
        return []
    walls = getattr(optic, "walls", None)
    if walls is None:                      # Generic flat-wall fallback
        return [(math.nan, 0.0, 0.0, 0.0)] * len(pts)
    out = []
    for (x, y, z), o in zip(pts, outs):
        wall = walls[0] if len(walls) == 1 else min(
            walls, key=lambda w: _axis_dist2(w, x, y, z))
        out.append(_wall_lens(wall, x, y, z, o))
    return out

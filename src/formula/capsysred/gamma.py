"""General-astigmatism beam algebra (Arnaud-Kogelnik), float64.

The beamlet parameter is the complex symmetric 2x2 matrix Q = Gamma^-1
stored as (q_xx, q_xy, q_yy); the scalar q is Q = q*I. Tensor ABCD:
drift L is Q += L*I, a grazing bounce is Gamma -= P in the inverse frame,
P = (1/f_t)*n*n^T + (1/f_s)*t*t^T with n = (cos phi, sin phi) the transverse
trace of the incidence plane (bounce azimuth) and t its normal:

    f_t = R_merid * sin(theta) / 2      (wall curvature along the axis)
    f_s = R_sag / (2 * sin(theta))      (azimuthal bore curvature)

At theta = 90 deg both collapse to R/2 (no astigmatism) — the sign check.
The on-axis amplitude accumulates det(I + dL*G)^(-1/2) over drift sub-steps
(a thin lens adds no on-axis factor); sub-stepping keeps each principal
square root within its branch. For the scalar case the product telescopes
to q0/q — the (w0/w)*exp(i*Gouy) of the stage-11a deposit.
"""

import cmath
import math


def inv2(m):
    xx, xy, yy = m
    det = xx * yy - xy * xy
    return (yy / det, -xy / det, xx / det)


def det2(m):
    return m[0] * m[2] - m[1] * m[1]


def reflect(q, phi, inv_ft, inv_fs):
    """One grazing bounce: Gamma -= P rotated to the bounce azimuth."""
    if inv_ft == 0.0 and inv_fs == 0.0:
        return q
    c, s = math.cos(phi), math.sin(phi)
    pxx = inv_ft * c * c + inv_fs * s * s
    pxy = (inv_ft - inv_fs) * c * s
    pyy = inv_ft * s * s + inv_fs * c * c
    g = inv2(q)
    return inv2((g[0] - pxx, g[1] - pxy, g[2] - pyy))


def propagate(zr, segments, lenses, nsub=2):
    """(Q at screen, on-axis amplitude factor) through the drift/bounce
    chain; len(segments) == len(lenses) + 1, waist i*zr*I at the source."""
    q = (complex(0.0, zr), 0j, complex(0.0, zr))
    amp = 1.0 + 0j
    for j, seg in enumerate(segments):
        step = seg / nsub
        for _ in range(nsub):
            pre = det2(q)
            q = (q[0] + step, q[1], q[2] + step)
            amp *= cmath.sqrt(pre / det2(q))
        if j < len(lenses):
            q = reflect(q, *lenses[j])
    return q, amp


def _center(wall):
    if wall.kind == "torus":   # ring center minus R*toward, toward = (n_y, -n_x)
        return (wall._Cf[0] - wall._Rf * wall._nf[1],
                wall._Cf[1] + wall._Rf * wall._nf[0])
    return (wall._cxf, wall._cyf)


def _wall_lens(wall, x, y, z, s):
    kind = wall.kind
    if kind in ("polygon", "implicit"):   # flat face / no closed-form curvature
        return (0.0, 0.0, 0.0)
    cx, cy = _center(wall)
    if kind == "cylinder":
        phi = math.atan2(y - cy, x - cx)
        return (phi, 0.0, 2.0 * s / math.sqrt(wall._a2f))
    if kind == "revolution":
        phi = math.atan2(y - cy, x - cx)
        r2 = wall._c0f + wall._c1f * z + wall._c2f * z * z
        r = math.sqrt(max(r2, 1e-30))
        rp = (wall._c1f + 2.0 * wall._c2f * z) / (2.0 * r)
        rpp = (wall._c2f - rp * rp) / r
        # R_merid = -(1+r'^2)^1.5/r'': waists (r'' < 0) curve toward the ray
        inv_ft = -2.0 * rpp / ((1.0 + rp * rp) ** 1.5 * s) if rpp else 0.0
        return (phi, inv_ft, 2.0 * s / r)
    # torus: tube radius sagittal; the bend radius meridional, concave
    # (focusing) on the outer wall of the bend, convex on the inner
    cx3, cy3, cz3 = wall._Cf
    nx, ny = wall._nf
    vx, vy, vz = x - cx3, y - cy3, z - cz3
    dot = vx * nx + vy * ny
    px, py, pz = vx - dot * nx, vy - dot * ny, vz
    rho = math.sqrt(px * px + py * py + pz * pz)
    tcx = cx3 + wall._Rf * px / rho
    tcy = cy3 + wall._Rf * py / rho
    phi = math.atan2(y - tcy, x - tcx)
    r_mer = wall._Rf if rho > wall._Rf else -wall._Rf
    return (phi, 2.0 / (r_mer * s), 2.0 * s / wall._af)


def bounce_lenses(optic, pts, sins):
    """(phi, 1/f_t, 1/f_s) per bounce from the wall shape at each hit point
    (float triples). Mirror and unsupported kinds come out flat — the
    beamlet falls back to the scalar-q (flat wall) model there."""
    if not pts:
        return []
    walls = getattr(optic, "walls", None)
    if walls is None:                      # Mirror: flat wall
        return [(0.0, 0.0, 0.0)] * len(pts)
    out = []
    for (x, y, z), s in zip(pts, sins):
        wall = walls[0] if len(walls) == 1 else min(
            walls, key=lambda w: (x - _center(w)[0]) ** 2
                                 + (y - _center(w)[1]) ** 2)
        out.append(_wall_lens(wall, x, y, z, s))
    return out

"""Modal error bound of a paraxial field in a circular Dirichlet bore.

The error e = E - E_m of a method field E_m (E the exact paraxial Dirichlet field on the
bore r < a_w) solves, in the bore frame, i e_z = -Lap e / 2k + U e with e|_wall = -g
(g the method field's wall trace) and e(0) = E_in - E_m(0).  In the Dirichlet basis
b_mn = J_|m|(kappa_mn r) e^{i m phi} (wave_fb.Disc), Green's identity turns the trace into
a forcing of the Galerkin coefficients c_mn (field = sum c_mn b_mn):

    c' = -i H c + s(z),   s_mn(z) = -i kappa_mn ghat_m(z) / (k a_w J_|m|+1(j_mn)),

ghat_m the Fourier coefficients of g over phi, H = diag(kappa^2 / 2k) + U the operator of
wave_fb.FBChannel.  Hence, with the exact free-paraxial-equation residual R of E_m,

    ||e(L)|| <= ||e(0)|| + ||int_0^L e^{-iH(L-z)} s(z) dz|| + tail(N) + int ||R|| dz,

the boundary integral taken over the modes of the Disc with the phase resolved by the
sections, and tail(N) bounding the modes beyond it by two integrations by parts.  No
field is propagated: ||e(0)|| is a quadrature on the entrance disc and the boundary part
lives on the trace.  The exact Dirichlet field (g = 0, e(0) = 0) gives 0.
"""
import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import jv

from .wave_fb import Disc, FBChannel


def disc_for(a_w: float, k: float, L: float, n_sections: int, phase_step: float = 0.7) -> Disc:
    """The Dirichlet basis whose fastest mode phase kappa^2 L / 2k advances at most
    `phase_step` rad per section: jmax = a_w sqrt(2k phase_step n_sections / L)."""
    kappa_max = math.sqrt(2.0 * k * phase_step * n_sections / L)
    return Disc(a_w, max(8.0, kappa_max * a_w))


def frame_ring(chan: FBChannel, z: float):
    """Lab positions of the frame ring points (r = a_w, Disc.phi) at z and the lab -> frame
    phase factor exp(-i k slope.x' - i g) at them."""
    d = chan.disc
    t, tp = chan.frame_axes()
    xi, eta = d.aw * np.cos(d.phi), d.aw * np.sin(d.phi)
    c = chan.axis(z)
    px = c[0] + xi * t[0] + eta * tp[0]
    py = c[1] + xi * t[1] + eta * tp[1]
    sl = chan.slope(z)
    phase = np.exp(-1j * chan.k * (sl[0] * (px - c[0]) + sl[1] * (py - c[1])) - 1j * chan.lab_phase_g(z))
    return px, py, phase


def frame_disc(chan: FBChannel, z: float):
    """Lab positions [nr, nphi] of the frame polar grid at z and the lab -> frame phase."""
    d = chan.disc
    t, tp = chan.frame_axes()
    xi = d.r[:, None] * np.cos(d.phi)[None, :]
    eta = d.r[:, None] * np.sin(d.phi)[None, :]
    c = chan.axis(z)
    px = c[0] + xi * t[0] + eta * tp[0]
    py = c[1] + xi * t[1] + eta * tp[1]
    sl = chan.slope(z)
    phase = np.exp(-1j * chan.k * (sl[0] * (px - c[0]) + sl[1] * (py - c[1])) - 1j * chan.lab_phase_g(z))
    return px, py, phase


def ring_hat(disc: Disc, g_ring):
    """Fourier coefficients ghat_m (signed m of the Disc) of ring values [..., nphi]."""
    F = np.fft.fft(np.asarray(g_ring), axis=-1) / disc.nphi
    return F[..., disc.col]


def extra_hat(disc: Disc, g_ring):
    """Fourier coefficients of the ring values for |m| > M (not in the Disc): [..., n_extra]."""
    F = np.fft.fft(np.asarray(g_ring), axis=-1) / disc.nphi
    ms = np.fft.fftfreq(disc.nphi, 1.0 / disc.nphi).astype(int)
    keep = np.abs(ms) > disc.M
    return F[..., keep]


def _filon(th):
    """E0 = int_0^1 e^{-i th v} dv, E1 = int_0^1 v e^{-i th v} dv (series below |th| = 1e-3)."""
    th = np.asarray(th, complex)
    small = np.abs(th) < 1e-3
    ths = np.where(small, 1.0, th)
    ex = np.exp(-1j * ths)
    e0 = (1.0 - ex) / (1j * ths)
    e1 = (e0 - ex) / (1j * ths)
    e0 = np.where(small, 1.0 - 1j * th / 2.0 - th * th / 6.0, e0)
    e1 = np.where(small, 0.5 - 1j * th / 3.0 - th * th / 8.0, e1)
    return e0, e1


class ModalBound:
    """Boundary-forced modal evolution in one bore and the bound's components."""

    def __init__(self, chan: FBChannel, sub_phase: float = 0.1):
        self.chan, self.disc, self.k = chan, chan.disc, chan.k
        self.sub_phase = float(sub_phase)
        d = self.disc
        # J_|m|+1(j_mn) per signed m and radial index (0 where no mode)
        jp = np.zeros((d.M + 1, d.nmax))
        for mm, z in enumerate(d.zeros):
            jp[mm, :z.size] = jv(mm + 1, z)
        self.jp = jp[d.am]
        kappa = np.sqrt(d.kap2)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.coef = np.where(d.valid, -1j * kappa / (self.k * d.aw * self.jp), 0.0)   # s = coef * ghat_m

    def forcing(self, ghat_z):
        """s(z) [nz, 2M+1, nmax] from ghat [nz, 2M+1]."""
        return self.coef[None, :, :] * np.asarray(ghat_z)[:, :, None]

    def boundary_part(self, zs, ghat_z, L):
        """int_0^L e^{-iH(L-z)} s(z) dz over the sections zs (zs[0] = 0, zs[-1] = L), s linear on
        each section.  Straight bore: diagonal phases with the exact (Filon) weights
        h [E1 s_j + (E0 - E1) s_j+1]; bent bore: exponential trapezoid steps with the
        Chebyshev propagator.  Returns the coefficient array."""
        zs = np.asarray(zs, float)
        s = self.forcing(ghat_z)
        h = np.diff(zs)
        if self.chan.rb is None:
            a = self.chan.h0
            out = np.zeros_like(s[0])
            for j in range(len(h)):
                e0, e1 = _filon(a * h[j])
                out = out * np.exp(-1j * a * h[j]) + h[j] * (e1 * s[j] + (e0 - e1) * s[j + 1])
            return out
        # bent bore: exponential trapezoid on sub-steps short enough that the fastest mode
        # advances at most `sub_phase` rad per sub-step (error ~ phase^2), s linear in between
        emin, emax = self.chan._spectral_bounds()
        c = np.zeros_like(s[0])
        for j in range(len(h)):
            n_sub = max(1, int(math.ceil(max(abs(emin), abs(emax)) * h[j] / self.sub_phase)))
            d = h[j] / n_sub
            for q in range(n_sub):
                s_a = s[j] + (s[j + 1] - s[j]) * (q / n_sub)
                s_b = s[j] + (s[j + 1] - s[j]) * ((q + 1) / n_sub)
                c, _ = self.chan.propagate(c + 0.5 * d * s_a, d)
                c = c + 0.5 * d * s_b
        return c

    def norm(self, c):
        return math.sqrt(self.disc.norm(c))

    def tails(self, zs, ghat_z, extra_hat=None):
        """Modes beyond the Disc, two integrations by parts (alpha = kappa^2/2k):
        |c| sqrt(norm) <= 2 sqrt(pi) (|ghat(0)| + |ghat(L)|) / kappa + 4 sqrt(pi) k (|ghat'(0)| + |ghat'(L)|
        + int |ghat''|) / kappa^3, summed with kappa_mn >= (n - 1/4) pi / a_w; `extra_hat`
        [nz, n_extra] are coefficients of |m| beyond the Disc (every radial index is a tail)."""
        d = self.disc
        aw = d.aw
        zs = np.asarray(zs, float)
        gh = np.asarray(ghat_z)
        g1 = np.gradient(gh, zs, axis=0)
        g2 = np.gradient(g1, zs, axis=0)
        end = np.abs(gh[0]) + np.abs(gh[-1])
        der = np.abs(g1[0]) + np.abs(g1[-1]) + np.trapezoid(np.abs(g2), zs, axis=0)
        n_valid = d.valid.sum(axis=1)                                   # radial modes kept per m
        s2 = (aw / math.pi) ** 2 / np.maximum(n_valid - 0.25, 0.75)       # sum_{n>N} 1/kappa^2
        s6 = (aw / math.pi) ** 6 / (5.0 * np.maximum(n_valid - 0.25, 0.75) ** 5)
        t1 = math.sqrt(float(np.sum(4.0 * math.pi * end ** 2 * s2)))
        t2 = math.sqrt(float(np.sum(16.0 * math.pi * self.k ** 2 * der ** 2 * s6)))
        if extra_hat is not None and np.size(extra_hat):
            eh = np.asarray(extra_hat)
            e1 = np.gradient(eh, zs, axis=0); e2 = np.gradient(e1, zs, axis=0)
            end_x = np.abs(eh[0]) + np.abs(eh[-1])
            der_x = np.abs(e1[0]) + np.abs(e1[-1]) + np.trapezoid(np.abs(e2), zs, axis=0)
            all2 = (aw / math.pi) ** 2 * 2.2                               # sum_{n>=1} 1/(n-1/4)^2 < 2.2
            all6 = (aw / math.pi) ** 6 * 4.1                               # sum_{n>=1} 1/(n-1/4)^6 < 4.1
            t1 = math.hypot(t1, math.sqrt(float(np.sum(4.0 * math.pi * end_x ** 2 * all2))))
            t2 = math.hypot(t2, math.sqrt(float(np.sum(16.0 * math.pi * self.k ** 2 * der_x ** 2 * all6))))
        return t1, t2



def bores_bound(chans, L, sample, entrance, phase_step=0.7, t_abs2=0.0):
    """B of one mode's method field over circular bores (the FBChannels share one Disc).
    sample(b, z_rel, radii, phis_lab, cx, cy) -> the method field on polar points around the
    bore axis at z0 + z_rel (raw units); entrance(ch) -> E_in on the Disc polar grid of bore
    b.  C = least-squares complex scale of the fan against E_in over all bores; per bore
    e0 = ||E_in - E_m(0)/C||, the frame wall trace at sections resolving the fastest mode
    phase, its boundary part and tails; bores in quadrature, T (sqrt t_abs2 / |C|) by the
    triangle inequality.  Returns the relative components (divided by ||E_in||)."""
    disc, k = chans[0].disc, chans[0].k
    a_w = disc.aw
    alpha_max = max(disc.kap2max / (2.0 * k) + (0.0 if ch.rb is None else k * a_w / ch.rb) for ch in chans)
    n_sec = max(int(math.ceil(L * alpha_max / phase_step)), 20)
    zs = np.linspace(0.0, L, n_sec + 1)
    wq = (disc.wr * disc.r)[:, None] * (2.0 * math.pi / disc.nphi)
    ein, efan, ein2, num = [], [], 0.0, 0j
    for b, ch in enumerate(chans):
        t, _ = ch.frame_axes()
        phis = disc.phi + math.atan2(t[1], t[0])
        cx, cy = ch.axis(0.0)
        fan = sample(b, 0.0, disc.r, phis, cx, cy)
        e_in = entrance(ch)
        ein.append(e_in); efan.append(fan)
        ein2 += float(np.sum(wq * np.abs(e_in) ** 2))
        num += complex(np.sum(wq * np.conj(e_in) * fan))
    C = num / ein2 if ein2 > 0 else 1.0 + 0j
    per_bore, b_sq = [], 0.0
    for b, ch in enumerate(chans):
        t, _ = ch.frame_axes()
        phis = disc.phi + math.atan2(t[1], t[0])
        e0 = math.sqrt(float(np.sum(wq * np.abs(ein[b] - efan[b] / C) ** 2)))
        g = np.zeros((len(zs), disc.nphi), complex)
        for jz, zr in enumerate(zs):
            cx, cy = ch.axis(zr)
            _, _, phase = frame_ring(ch, zr)
            g[jz] = sample(b, zr, np.array([a_w]), phis, cx, cy)[0] * phase / C
        mb = ModalBound(ch)
        ghat = ring_hat(disc, g)
        bnd = mb.norm(mb.boundary_part(zs, ghat, L))
        t1, t2 = mb.tails(zs, ghat, extra_hat(disc, g))
        bb = e0 + bnd + t1 + t2
        b_sq += bb * bb
        per_bore.append({"bore": b, "e0": e0, "boundary": bnd, "tail": t1 + t2, "B": bb})
    norm_in = math.sqrt(ein2)
    t_abs = math.sqrt(max(t_abs2, 0.0)) / abs(C)
    return {"B": (math.sqrt(b_sq) + t_abs) / norm_in,
            "B_e0": math.sqrt(sum(r["e0"] ** 2 for r in per_bore)) / norm_in,
            "B_boundary": math.sqrt(sum(r["boundary"] ** 2 for r in per_bore)) / norm_in,
            "B_tail": math.sqrt(sum(r["tail"] ** 2 for r in per_bore)) / norm_in,
            "B_T": t_abs / norm_in,
            "bores": [{k_: (v / norm_in if k_ != "bore" else v) for k_, v in r.items()} for r in per_bore],
            "C_fit": [abs(C), math.atan2(C.imag, C.real)], "sections": n_sec, "jmax": float(disc.jmax),
            "modes": int(disc.valid.sum()), "E_in_norm": norm_in}


# ------------------------------------------------------------------ square bore
class Square:
    """Dirichlet sine basis of a straight square bore |x'|, |y'| < aw in the bore frame
    (rotation rot about the centre): phi_mn = sin(m pi (x'+aw)/2aw) sin(n pi (y'+aw)/2aw) / aw,
    kappa^2 = (pi/2aw)^2 (m^2 + n^2), 1 <= m, n <= M.  Outline nodes: nq midpoints per face,
    faces +x, +y, -x, -y in arc order from the corner (aw, -aw); sampled trace coefficients
    exist up to n = nq (the band limit of the sampling)."""

    def __init__(self, aw, M, rot=0.0, nq=None):
        self.aw, self.M, self.rot = float(aw), int(M), float(rot)
        self.nq = int(nq or 8 * self.M)
        m = np.arange(1, self.M + 1)
        self.kap2 = (math.pi / (2.0 * self.aw)) ** 2 * (m[:, None] ** 2 + m[None, :] ** 2)
        self.kap2max = float(self.kap2.max())
        self.h = 2.0 * self.aw / self.nq
        self.s = -self.aw + (np.arange(self.nq) + 0.5) * self.h
        aw, s = self.aw, self.s
        self.face_x = np.concatenate([np.full(self.nq, aw), -s, np.full(self.nq, -aw), s])
        self.face_y = np.concatenate([s, np.full(self.nq, aw), -s, np.full(self.nq, -aw)])
        self.cr, self.sr = math.cos(self.rot), math.sin(self.rot)
        # tensor Gauss grid of the cross-section (to_modes, tests)
        t, w = leggauss(self.nq)
        self.gq, self.gw = aw * t, aw * w
        self.Sg = np.sin(m[:, None] * math.pi * (self.gq[None, :] + aw) / (2.0 * aw))

    def lab(self, cx, cy, xf, yf):
        return cx + self.cr * xf - self.sr * yf, cy + self.sr * xf + self.cr * yf

    def frame(self, cx, cy, px, py):
        dx, dy = px - cx, py - cy
        return self.cr * dx + self.sr * dy, -self.sr * dx + self.cr * dy

    def arc_index(self, cx, cy, px, py):
        """Outline node met first along the ray from the centre through (px, py)."""
        xf, yf = self.frame(cx, cy, px, py)
        ax, ay = np.abs(xf), np.abs(yf)
        r = np.maximum(np.maximum(ax, ay), 1e-30)
        xh, yh = xf * (self.aw / r), yf * (self.aw / r)
        aw = self.aw
        u = np.where(ax >= ay, np.where(xf > 0, yh + aw, 5.0 * aw - yh),
                     np.where(yf > 0, 3.0 * aw - xh, 7.0 * aw + xh))
        return np.floor(u / self.h).astype(int) % (4 * self.nq)

    def grid(self, cx=0.0, cy=0.0):
        """Lab positions [nq, nq] of the tensor Gauss grid (x along axis 0)."""
        xf, yf = np.meshgrid(self.gq, self.gq, indexing="ij")
        return self.lab(cx, cy, xf, yf)

    def to_modes(self, E):
        """Coefficients [M, M] of values on the tensor Gauss grid (orthonormal basis)."""
        Sw = self.Sg * self.gw[None, :]
        return Sw @ np.asarray(E) @ Sw.T / self.aw


class SquareBound:
    """Boundary-forced sine-mode evolution in a straight square bore and the bound's tails."""

    def __init__(self, sq: Square, k: float):
        self.sq, self.k = sq, float(k)
        self.alpha = sq.kap2 / (2.0 * self.k)
        n = np.arange(1, sq.nq + 1)
        self.sgn = (-1.0) ** n                                  # (-1)^n, n <= nq
        self.mp = n * math.pi / (2.0 * sq.aw ** 2)              # n pi / 2 aw^2
        self.S = np.sin(n[:, None] * math.pi * (sq.s[None, :] + sq.aw) / (2.0 * sq.aw))   # [nq, nq]

    def face_coef(self, g_outline):
        """G^f_n = int_face g s_n ds [..., 4, nq] in the face coordinate (y on the x faces, x on
        the y faces), midpoint rule on the outline nodes."""
        g = np.asarray(g_outline)
        g = g.reshape(g.shape[:-1] + (4, self.sq.nq))
        G = np.einsum("...fq,nq->...fn", g, self.S) * self.sq.h
        flip = -self.sgn                                        # nodes run backwards on faces +y, -x
        G[..., 1, :] *= flip
        G[..., 2, :] *= flip
        return G

    def _forcing(self, G, n):
        """s_mn = (i/2k) int g d_n phi_mn ds for m, n <= n from the face coefficients [4, nq]."""
        gpx, gpy, gmx, gmy = (G[f, :n] for f in range(4))
        mp, sgn = self.mp[:n], self.sgn[:n]
        f = (mp[:, None] * (sgn[:, None] * gpx[None, :] - gmx[None, :])
             + mp[None, :] * (sgn[None, :] * gpy[:, None] - gmy[:, None]))
        return 1j / (2.0 * self.k) * f

    def boundary_part(self, zs, G, L):
        """int_0^L e^{-i alpha (L-z)} s(z) dz over the sections (Filon weights, s linear per
        section) for the M-box; G [nz, 4, nq]."""
        zs = np.asarray(zs, float)
        M = self.sq.M
        s = np.array([self._forcing(Gz, M) for Gz in G])
        h = np.diff(zs)
        out = np.zeros_like(s[0])
        for j in range(len(h)):
            e0, e1 = _filon(self.alpha * h[j])
            out = out * np.exp(-1j * self.alpha * h[j]) + h[j] * (e1 * s[j] + (e0 - e1) * s[j + 1])
        return out

    def norm(self, c):
        return float(np.sqrt(np.sum(np.abs(c) ** 2)))

    def tails(self, zs, G):
        """Modes outside the M-box up to the sampling band n <= nq, two integrations by parts:
        |c| <= (|s(0)| + |s(L)|)/alpha + (|s'(0)| + |s'(L)| + int |s''| dz)/alpha^2."""
        zs = np.asarray(zs, float)
        nq, M = self.sq.nq, self.sq.M
        n = np.arange(1, nq + 1)
        alpha = (math.pi / (2.0 * self.sq.aw)) ** 2 * (n[:, None] ** 2 + n[None, :] ** 2) / (2.0 * self.k)
        tail = np.ones((nq, nq), bool)
        tail[:M, :M] = False
        G1 = np.gradient(G, zs, axis=0)
        G2 = np.gradient(G1, zs, axis=0)
        f = lambda Gz: np.abs(self._forcing(Gz, nq))
        end = f(G[0]) + f(G[-1])
        w = np.gradient(zs)                                     # trapezoid weights
        w[0] *= 0.5; w[-1] *= 0.5
        der = f(G1[0]) + f(G1[-1])
        for j in range(len(zs)):
            der = der + w[j] * f(G2[j])
        t1 = math.sqrt(float(np.sum((end[tail] / alpha[tail]) ** 2)))
        t2 = math.sqrt(float(np.sum((der[tail] / alpha[tail] ** 2) ** 2)))
        return t1, t2


def squares_bound(sqs, centers, k, L, sample_disc, sample_outline, e_in, phase_step=0.7, t_abs2=0.0, nr=48, nphi=384):
    """B of one mode's method field over straight square bores (one Square per bore, same aw and M).
    sample_disc(b, radii, phis, stretch, cx, cy) -> the fan on the stretched polar grid of bore b
    at z0 (raw units); sample_outline(b, z_rel, sq, cx, cy) -> the fan on its outline nodes at
    z0 + z_rel; e_in(px, py) -> E_in at lab points.  Assembled as bores_bound."""
    aw = sqs[0].aw
    alpha_max = max(sq.kap2max for sq in sqs) / (2.0 * k)
    n_sec = max(int(math.ceil(L * alpha_max / phase_step)), 20)
    zs = np.linspace(0.0, L, n_sec + 1)
    t, wt = leggauss(nr)
    t, wt = 0.5 * (t + 1.0), 0.5 * wt
    ein, efan, wqs, ein2, num = [], [], [], 0.0, 0j
    for b, sq in enumerate(sqs):
        cx, cy = centers[b]
        phis = sq.rot - 0.25 * math.pi + (np.arange(nphi) + 0.5) * 2.0 * math.pi / nphi
        stretch = 1.0 / np.cos(((phis - sq.rot + 0.25 * math.pi) % (0.5 * math.pi)) - 0.25 * math.pi)
        wq = (aw * aw * t * wt)[:, None] * (stretch ** 2)[None, :] * (2.0 * math.pi / nphi)
        rr = (aw * t)[:, None] * stretch[None, :]
        e = e_in(cx + rr * np.cos(phis)[None, :], cy + rr * np.sin(phis)[None, :])
        fan = sample_disc(b, aw * t, phis, stretch, cx, cy)
        ein.append(e); efan.append(fan); wqs.append(wq)
        ein2 += float(np.sum(wq * np.abs(e) ** 2))
        num += complex(np.sum(wq * np.conj(e) * fan))
    C = num / ein2 if ein2 > 0 else 1.0 + 0j
    per_bore, b_sq = [], 0.0
    for b, sq in enumerate(sqs):
        cx, cy = centers[b]
        e0 = math.sqrt(float(np.sum(wqs[b] * np.abs(ein[b] - efan[b] / C) ** 2)))
        g = np.array([sample_outline(b, zr, sq, cx, cy) for zr in zs]) / C
        sb = SquareBound(sq, k)
        G = sb.face_coef(g)
        bnd = sb.norm(sb.boundary_part(zs, G, L))
        t1, t2 = sb.tails(zs, G)
        bb = e0 + bnd + t1 + t2
        b_sq += bb * bb
        per_bore.append({"bore": b, "e0": e0, "boundary": bnd, "tail": t1 + t2, "B": bb})
    norm_in = math.sqrt(ein2)
    t_abs = math.sqrt(max(t_abs2, 0.0)) / abs(C)
    return {"B": (math.sqrt(b_sq) + t_abs) / norm_in,
            "B_e0": math.sqrt(sum(r["e0"] ** 2 for r in per_bore)) / norm_in,
            "B_boundary": math.sqrt(sum(r["boundary"] ** 2 for r in per_bore)) / norm_in,
            "B_tail": math.sqrt(sum(r["tail"] ** 2 for r in per_bore)) / norm_in,
            "B_T": t_abs / norm_in,
            "bores": [{k_: (v / norm_in if k_ != "bore" else v) for k_, v in r.items()} for r in per_bore],
            "C_fit": [abs(C), math.atan2(C.imag, C.real)], "sections": n_sec, "jmax": float(sqs[0].M),
            "modes": int(sqs[0].M ** 2), "E_in_norm": norm_in, "basis": "square sines"}

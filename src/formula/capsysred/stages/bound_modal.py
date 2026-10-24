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

from .wave_fb import Disc, FBChannel, wall_offset


def _interp_error(zs, s_at, norm):
    """int ||s - s_lin|| dz for s sampled at zs (s_at(j) -> the forcing at section j): the
    linear interpolant misses at most h^2/8 max||s''|| per section, s'' by second differences."""
    zs = np.asarray(zs, float)
    n = len(zs)
    if n < 3:
        return 0.0
    h = np.diff(zs)
    d2 = np.zeros(n)
    prev, cur = s_at(0), s_at(1)
    for j in range(1, n - 1):
        nxt = s_at(j + 1)
        d2[j] = norm((nxt - cur) / h[j] - (cur - prev) / h[j - 1]) / (0.5 * (h[j] + h[j - 1]))
        prev, cur = cur, nxt
    d2[0], d2[-1] = d2[1], d2[-2]
    return float(np.sum(h ** 3 / 8.0 * np.maximum(d2[:-1], d2[1:])))


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

    def interp_error(self, zs, ghat_z):
        """int ||s - s_lin|| dz of the piecewise-linear forcing between the sections, from the
        second differences of the samples: sum_j h_j^3 / 8 max(||s''_j||, ||s''_j+1||)."""
        return _interp_error(zs, lambda j: self.forcing(np.asarray(ghat_z)[j:j + 1])[0], self.norm)

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



def _scale(num, ein2, e_in_dep):
    """Fan scale C: the least-squares phase; the magnitude from the deposit-unit norm of E_in
    when given (a wrong fan amplitude is then error), else least squares as well."""
    if ein2 <= 0.0:
        return 1.0 + 0j
    lsq = num / ein2
    if e_in_dep is None:
        return lsq if lsq != 0 else 1.0 + 0j
    mag = e_in_dep / math.sqrt(ein2)
    return mag * (lsq / abs(lsq) if lsq != 0 else 1.0)


def bores_bound(chans, L, sample, entrance, phase_step=0.7, t_abs2=0.0, e_in_dep=None):
    """B of one mode's method field over circular bores (the FBChannels share one Disc); the
    reference is the Dirichlet solution of the channels' operator H (lossless when the channels
    carry no gamma: a damped reference would add the source -Gamma E_method, not included here).
    sample(b, z_rel, radii, phis_lab, cx, cy) -> the method field on polar points around the
    bore axis at z0 + z_rel (raw units); entrance(ch) -> E_in on the Disc polar grid of bore
    b.  The fan scale C: least-squares phase, magnitude from e_in_dep (deposit-unit norm of
    E_in) or least squares; `rescale` re-assembles the result for another magnitude (the
    common scale of a launch).  Returns the relative components (divided by ||E_in||) and
    the C-independent "parts"."""
    disc, k = chans[0].disc, chans[0].k
    a_w = disc.aw
    alpha_max = max(disc.kap2max / (2.0 * k) + (0.0 if ch.rb is None else k * a_w / ch.rb) for ch in chans)
    n_sec = max(int(math.ceil(L * alpha_max / phase_step)), 20)
    zs = np.linspace(0.0, L, n_sec + 1)
    wq = (disc.wr * disc.r)[:, None] * (2.0 * math.pi / disc.nphi)
    bores, ein2, num = [], 0.0, 0j
    for b, ch in enumerate(chans):
        t, _ = ch.frame_axes()
        phis = disc.phi + math.atan2(t[1], t[0])
        cx, cy = ch.axis(0.0)
        fan = sample(b, 0.0, disc.r, phis, cx, cy)
        e_in = entrance(ch)
        g = np.zeros((len(zs), disc.nphi), complex)
        for jz, zr in enumerate(zs):
            cx, cy = ch.axis(zr)
            _, _, phase = frame_ring(ch, zr)
            g[jz] = sample(b, zr, np.array([a_w]), phis, cx, cy)[0] * phase
        mb = ModalBound(ch)
        ghat = ring_hat(disc, g)
        bores.append(_bore_parts(b, e_in, fan, wq, disc.to_modes, disc.to_grid, lambda c: ch.propagate(c, L)[0],
                                 mb.boundary_part(zs, ghat, L), mb.norm, mb.interp_error(zs, ghat),
                                 *mb.tails(zs, ghat, extra_hat(disc, g))))
        ein2 += bores[-1]["ein2"]
        num += bores[-1]["num"]
    parts = {"bores": bores, "ein2": ein2, "num": num, "t_abs2": float(t_abs2), "mnorm": disc.mnorm,
             "sections": n_sec, "jmax": float(disc.jmax), "modes": int(disc.valid.sum())}
    return _assemble(parts, _scale(num, ein2, e_in_dep))


def _bore_parts(b, e_in, fan, wq, to_modes, to_grid, propagate, vB, norm, itp1, t1, t2):
    """C-independent pieces of one bore: quadrature scalars of E_in and the fan (C = 1), their
    projections propagated to L, the boundary part of the fan's trace, remainders beyond the
    modes, interpolation and tail terms (all for C = 1)."""
    cE, cF = to_modes(e_in), to_modes(fan)
    rE, rF = e_in - to_grid(cE), fan - to_grid(cF)
    q = lambda a, c: complex(np.sum(wq * np.conj(a) * c))
    return {"bore": b, "ein2": q(e_in, e_in).real, "fan2": q(fan, fan).real, "num": q(e_in, fan),
            "rE2": q(rE, rE).real, "rF2": q(rF, rF).real, "rEF": q(rE, rF),
            "vE": propagate(cE), "vF": propagate(cF), "vB": vB, "bnd1": norm(vB), "itp1": itp1, "t1": t1, "t2": t2}


def _assemble(parts, C):
    """Relative components for the fan scale C from the parts: per bore e0 = ||E_in - fan/C||,
    exact = ||e^{-iHL} P e(0) + boundary part||, r0 = ||(1-P) e(0)||, B = min(exact + r0, e0 +
    ||boundary||) + interpolation + tail; bores in quadrature, T by the triangle inequality."""
    mn = parts["mnorm"]
    norm = (lambda v: math.sqrt(float(np.sum(np.abs(v) ** 2 * mn)))) if mn is not None else \
        (lambda v: math.sqrt(float(np.sum(np.abs(v) ** 2))))
    inv, ac = 1.0 / C, abs(C)
    per_bore = []
    for p in parts["bores"]:
        e0 = math.sqrt(max(p["ein2"] - 2.0 * (p["num"] * inv).real + p["fan2"] / ac ** 2, 0.0))
        r0 = math.sqrt(max(p["rE2"] - 2.0 * (p["rEF"] * inv).real + p["rF2"] / ac ** 2, 0.0))
        exact = norm(p["vE"] - (p["vF"] - p["vB"]) * inv)
        bnd, itp, tail = p["bnd1"] / ac, p["itp1"] / ac, (p["t1"] + p["t2"]) / ac
        tri = e0 + bnd + itp + tail
        per_bore.append({"bore": p["bore"], "e0": e0, "r0": r0, "exact": exact, "boundary": bnd, "interp": itp,
                         "tail": tail, "B": min(exact + r0 + itp + tail, tri), "B_tri": tri})
    ein2, num = parts["ein2"], parts["num"]
    norm_in = math.sqrt(ein2)
    t_abs = math.sqrt(max(parts["t_abs2"], 0.0)) / ac
    q = lambda key: math.sqrt(sum(r[key] ** 2 for r in per_bore)) / norm_in
    return {"B": (math.sqrt(sum(r["B"] ** 2 for r in per_bore)) + t_abs) / norm_in,
            "B_tri": (math.sqrt(sum(r["B_tri"] ** 2 for r in per_bore)) + t_abs) / norm_in,
            "B_e0": q("e0"), "B_r0": q("r0"), "B_exact": q("exact"), "B_boundary": q("boundary"), "B_interp": q("interp"),
            "B_tail": q("tail"), "B_T": t_abs / norm_in,
            "bores": [{k_: (v / norm_in if k_ != "bore" else v) for k_, v in r.items()} for r in per_bore],
            "C_fit": [ac, math.atan2(C.imag, C.real)],
            "scale_defect": (abs(num / ein2) / ac - 1.0) if ein2 > 0 else None,   # LSQ amplitude vs the scale used
            "sections": parts["sections"], "jmax": parts["jmax"], "modes": parts["modes"], "E_in_norm": norm_in,
            "E_dep": ac * norm_in,                                                 # ||E_in|| in the fan's (deposit) units
            "parts": parts}


def rescale(result, c_mag):
    """The same bound for the fan magnitude c_mag (least-squares phase kept)."""
    parts = result["parts"]
    num = parts["num"]
    C = c_mag * (num / abs(num) if num != 0 else 1.0)
    return _assemble(parts, C)


def common_scale(results):
    """One fan magnitude for the modes of a launch (the mean of their least-squares magnitudes):
    a mode with another amplitude is error; the map's |mu| is invariant to the common factor.
    Returns the rescaled results without the parts, with the common magnitude."""
    bounds = [r for r in results if r and r.get("parts")]
    if not bounds:
        return [({k: v for k, v in r.items() if k != "parts"} if r else r) for r in results]
    c_mag = float(np.mean([abs(r["parts"]["num"]) / r["parts"]["ein2"] for r in bounds if r["parts"]["ein2"] > 0]))
    out = []
    for r in results:
        if r and r.get("parts"):
            r = rescale(r, c_mag)
            r["C_common"] = c_mag
        out.append({k: v for k, v in r.items() if k != "parts"} if r else r)
    return out


# ------------------------------------------------------------------ square bore
class Square:
    """Dirichlet sine basis of a straight square bore |x'|, |y'| < aw in the bore frame
    (rotation rot about the centre): phi_mn = sin(m pi (x'+aw)/2aw) sin(n pi (y'+aw)/2aw) / aw,
    kappa^2 = (pi/2aw)^2 (m^2 + n^2), 1 <= m, n <= M.  Outline nodes: nq midpoints per face
    (default: 50 nm spacing, at least 2M), faces +x, +y, -x, -y in arc order from the corner
    (aw, -aw); sampled trace coefficients exist up to n = nq (the band limit of the sampling)."""

    def __init__(self, aw, M, rot=0.0, nq=None):
        self.aw, self.M, self.rot = float(aw), int(M), float(rot)
        self.nq = int(nq or max(2 * self.M, math.ceil(2.0 * self.aw / 5e-8)))
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

    def to_grid(self, c):
        """Values on the tensor Gauss grid of the coefficients c [M, M]."""
        return self.Sg.T @ np.asarray(c) @ self.Sg / self.aw

    def eval(self, c, xf, yf):
        """Values of sum c phi at arbitrary frame points."""
        m = np.arange(1, self.M + 1)
        sx = np.sin(m[:, None] * math.pi * (np.asarray(xf).ravel()[None, :] + self.aw) / (2.0 * self.aw))
        sy = np.sin(m[:, None] * math.pi * (np.asarray(yf).ravel()[None, :] + self.aw) / (2.0 * self.aw))
        return (np.einsum("pm,mp->p", sx.T @ np.asarray(c), sy) / self.aw).reshape(np.shape(xf))

    def project(self, E, wq, xf, yf):
        """Quadrature projection of values E on arbitrary frame points (weights wq) onto the
        M-box, and the pointwise remainder E - sum c phi."""
        m = np.arange(1, self.M + 1)
        sx = np.sin(m[:, None] * math.pi * (xf.ravel()[None, :] + self.aw) / (2.0 * self.aw))    # [M, P]
        sy = np.sin(m[:, None] * math.pi * (yf.ravel()[None, :] + self.aw) / (2.0 * self.aw))
        we = (np.asarray(wq) * np.asarray(E)).ravel()
        c = (sx * we[None, :]) @ sy.T / self.aw
        back = np.einsum("pm,mp->p", sx.T @ c, sy) / self.aw
        return c, np.asarray(E) - back.reshape(np.shape(E))


def slab_damping(kappa, k, delta, beta, aw):
    """DIR-ell amplitude damping of a slab mode of wall wavenumber kappa (wave_fb.mode_damping
    per wall pair): kappa^2 Im ell / (k a_w) times the exact Fresnel loss at the grazing angle
    kappa / k relative to its theta -> 0 form.  A diagnostic of the model gap, not part of B."""
    ell = wall_offset(k, delta, beta)
    thw = np.asarray(kappa, float) / k
    q = np.sqrt(thw * thw - 2.0 * delta + 2j * beta)
    q = np.where(q.imag < 0.0, -q, q)
    loss = 4.0 * thw * q.real / np.abs(thw + q) ** 2
    loss0 = 4.0 * thw * beta / (2.0 * delta) ** 1.5
    factor = np.divide(loss, loss0, out=np.ones_like(loss), where=loss0 > 0.0)
    return np.asarray(kappa) ** 2 * ell.imag / (aw * k) * factor


def square_damping(sq: Square, k, delta, beta):
    """Per-mode damping [M, M] of the square: the x faces see kappa_m, the y faces kappa_n."""
    g = slab_damping(np.arange(1, sq.M + 1) * math.pi / (2.0 * sq.aw), k, delta, beta, sq.aw)
    return g[:, None] + g[None, :]


class SquareBound:
    """Boundary-forced sine-mode evolution in a straight square bore and the bound's tails;
    `gamma` [M, M] = DIR-ell damping (square_damping), none = lossless Dirichlet."""

    def __init__(self, sq: Square, k: float, gamma=None):
        self.sq, self.k = sq, float(k)
        self.alpha = sq.kap2 / (2.0 * self.k) - (0.0 if gamma is None else 1j * np.asarray(gamma))
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

    def interp_error(self, zs, G):
        """int ||s - s_lin|| dz of the piecewise-linear forcing of the M-box between the sections."""
        return _interp_error(zs, lambda j: self._forcing(G[j], self.sq.M), self.norm)

    def tails(self, zs, G):
        """Modes outside the M-box, two integrations by parts: |c| <= (|s(0)| + |s(L)|)/alpha
        + (|s'(0)| + |s'(L)| + int |s''| dz)/alpha^2; the face harmonics up to the sampling band
        n <= nq (the rest taken as zero), the normal index to infinity."""
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
        t1_sq = float(np.sum((end[tail] / alpha[tail]) ** 2))
        t2_sq = float(np.sum((der[tail] / alpha[tail] ** 2) ** 2))
        # normal index beyond the sampling band (m > nq for the x faces, n > nq for the y faces):
        # |s_mn| <= (1/2k) m pi/(2 aw^2) X_n with X_n = |G+x_n| + |G-x_n|; (m pi/2aw^2)/alpha_mn
        # = (4 k aw m/pi)/(m^2+n^2), sum_{m>K} m^2/(m^2+n^2)^2 <= 1/K, sum_{m>K} m^2/(m^2+n^2)^4 <= 1/(5 K^5)
        aw, K = self.sq.aw, nq
        c1, c2 = 2.0 * aw / math.pi, 2.0 * self.k * (2.0 * aw / math.pi) ** 3
        for fa, fb in ((0, 2), (1, 3)):
            endf = np.abs(G[0, fa]) + np.abs(G[0, fb]) + np.abs(G[-1, fa]) + np.abs(G[-1, fb])
            derf = (np.abs(G1[0, fa]) + np.abs(G1[0, fb]) + np.abs(G1[-1, fa]) + np.abs(G1[-1, fb])
                    + np.trapezoid(np.abs(G2[:, fa]) + np.abs(G2[:, fb]), zs, axis=0))
            t1_sq += (c1 ** 2 / K) * float(np.sum(endf ** 2))
            t2_sq += (c2 ** 2 / (5.0 * K ** 5)) * float(np.sum(derf ** 2))
        return math.sqrt(t1_sq), math.sqrt(t2_sq)


def squares_bound(sqs, centers, k, L, sample_disc, sample_outline, e_in, phase_step=0.7, t_abs2=0.0, nr=48, nphi=384,
                  gamma=None, e_in_dep=None, sample_grid=None):
    """B of one mode's method field over straight square bores (one Square per bore, same aw and M).
    sample_grid(b, xf, yf, cx, cy) -> the fan on the tensor Gauss grid of bore b at z0 (frame
    coordinates xf x yf; None when the beams are not separable) -- exact projection; else
    sample_disc(b, radii, phis, stretch, cx, cy) on the stretched polar grid with a quadrature
    projection; sample_outline(b, z_rel, sq, cx, cy) -> the fan on the outline nodes at z0 + z_rel;
    e_in(px, py) -> E_in at lab points; gamma must stay None for a bound against the lossless
    Dirichlet solution (a damped reference needs the source -Gamma E_method, not included);
    e_in_dep = deposit-unit norm of E_in fixing the fan scale.  Assembled as bores_bound."""
    aw = sqs[0].aw
    alpha_max = max(sq.kap2max for sq in sqs) / (2.0 * k)
    n_sec = max(int(math.ceil(L * alpha_max / phase_step)), 20)
    zs = np.linspace(0.0, L, n_sec + 1)
    t, wt = leggauss(nr)
    t, wt = 0.5 * (t + 1.0), 0.5 * wt
    bores, ein2, num = [], 0.0, 0j
    for b, sq in enumerate(sqs):
        cx, cy = centers[b]
        fan = sample_grid(b, sq.gq, sq.gq, cx, cy) if sample_grid is not None else None
        if fan is not None:
            px, py = sq.grid(cx, cy)
            wq = sq.gw[:, None] * sq.gw[None, :]
            to_modes, to_grid = sq.to_modes, sq.to_grid
        else:
            phis = sq.rot - 0.25 * math.pi + (np.arange(nphi) + 0.5) * 2.0 * math.pi / nphi
            stretch = 1.0 / np.cos(((phis - sq.rot + 0.25 * math.pi) % (0.5 * math.pi)) - 0.25 * math.pi)
            wq = (aw * aw * t * wt)[:, None] * (stretch ** 2)[None, :] * (2.0 * math.pi / nphi)
            rr = (aw * t)[:, None] * stretch[None, :]
            px, py = cx + rr * np.cos(phis)[None, :], cy + rr * np.sin(phis)[None, :]
            fan = sample_disc(b, aw * t, phis, stretch, cx, cy)
            xf, yf = sq.frame(cx, cy, px, py)
            to_modes = lambda E, sq=sq, wq=wq, xf=xf, yf=yf: sq.project(E, wq, xf, yf)[0]
            to_grid = lambda c, sq=sq, xf=xf, yf=yf: sq.eval(c, xf, yf)
        e = e_in(px, py)
        g = np.array([sample_outline(b, zr, sq, cx, cy) for zr in zs])
        sb = SquareBound(sq, k, gamma)
        G = sb.face_coef(g)
        bores.append(_bore_parts(b, e, fan, wq, to_modes, to_grid, lambda c, sb=sb: c * np.exp(-1j * sb.alpha * L),
                                 sb.boundary_part(zs, G, L), sb.norm, sb.interp_error(zs, G), *sb.tails(zs, G)))
        ein2 += bores[-1]["ein2"]
        num += bores[-1]["num"]
    parts = {"bores": bores, "ein2": ein2, "num": num, "t_abs2": float(t_abs2), "mnorm": None,
             "sections": n_sec, "jmax": float(sqs[0].M), "modes": int(sqs[0].M ** 2)}
    out = _assemble(parts, _scale(num, ein2, e_in_dep))
    out["basis"] = "square sines"
    return out

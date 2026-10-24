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

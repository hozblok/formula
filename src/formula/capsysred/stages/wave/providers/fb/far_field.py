"""Far field of the provider fb from the exit modes and its ring statistics."""

import math

import numpy as np
from scipy import fft as sfft
from scipy.special import jv


def far_field_amplitude(disc, c, k, thetas, nphi):
    """Fraunhofer amplitude A(theta, phi) = int E exp(-i k theta.x) d^2x of the field sum_mn c_mn J_|m|(j_mn r/aw)
    e^{i m phi} (zero outside aw): A = sum_m e^{i m phi} (-i)^|m| sum_n c_mn G_mn(k theta), with the Lommel form
    G_mn(q) = 2 pi aw kappa J_{|m|+1}(j) J_|m|(q aw) / (kappa^2 - q^2) (J_|m|(j) = 0; pi aw^2 J_{|m|+1}(j)^2 at
    q = kappa); azimuths phi_j = 2 pi j / nphi (nphi >= 2M + 1) by an FFT over m. Shape (len(thetas), nphi)."""
    aw, M = disc.aw, disc.M
    if nphi < 2 * M + 1:
        raise ValueError(f"far field: {nphi} azimuths alias the basis |m| <= {M}")
    q = k * np.asarray(thetas, float)
    a = np.zeros((q.size, nphi), complex)
    for mm in range(M + 1):
        n = disc.zeros[mm].size
        kap, jp1 = disc.zp[mm, :n] / aw, disc.jp1p[mm, :n]
        den = kap[None, :] ** 2 - q[:, None] ** 2
        near = np.abs(den) < 1e-9 * kap[None, :] ** 2
        g = 2.0 * math.pi * aw * (kap * jp1)[None, :] * jv(mm, q * aw)[:, None] / np.where(near, 1.0, den)
        g = np.where(near, math.pi * aw * aw * jp1[None, :] ** 2, g)
        for m in sorted({mm, -mm}):
            a[:, m % nphi] += (-1j) ** mm * (g @ c[m + M, :n])
    return nphi * sfft.ifft(a, axis=1)


def far_field_profiles(P, thetas, k, band):
    """Azimuthal power density dP/dphi = (k / 2 pi)^2 int_band |A|^2 theta dtheta (trapezoid) and the
    azimuthal mean of |A|^2 per theta; with Parseval, sum(ring) dphi is the power inside the band."""
    t = np.asarray(thetas, float)
    w = np.zeros_like(t)
    dt = np.diff(t)
    w[:-1] += 0.5 * dt
    w[1:] += 0.5 * dt
    w *= t * (k / (2.0 * math.pi)) ** 2 * ((t >= band[0]) & (t <= band[1]))
    return w @ P, P.mean(axis=1)


def ring_summary(ring):
    """Ring statistics: mean density, values at phi = 0 and pi over the mean, azimuthal harmonics |c_q| / c_0."""
    n = ring.size
    mean = float(ring.mean())
    if mean <= 0.0:
        return {"mean": mean, "at_0": None, "at_pi": None, "harmonics": []}
    f = np.abs(sfft.rfft(ring)) / abs(sfft.rfft(ring)[0])
    return {"mean": mean, "at_0": float(ring[0] / mean), "at_pi": float(ring[n // 2] / mean) if n % 2 == 0 else None,
            "harmonics": [float(v) for v in f[1:33]]}

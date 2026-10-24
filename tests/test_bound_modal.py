"""Modal error bound: a free Gaussian beam taken as the error field e has a known wall
trace, entrance value and exit value; the boundary-forced modal evolution must return it."""
import math

import numpy as np
import pytest

from formula.capsysred.stages import beamlet_defect as D
from formula.capsysred.stages import bound_modal as B
from formula.capsysred.stages.wave_fb import Disc, FBChannel

LAM = 12.398419843320026e-10 / 8.048
K = 2.0 * math.pi / LAM
AW = 24.0e-6
L = 0.05          # default length of the zero-trace test


def _beam(w0, x0, y0):
    seg = D.Segments(1)
    seg.qxx[0] = seg.qyy[0] = complex(0.0, 0.5 * w0 * w0 * K)
    seg.a[0] = 1.0
    seg.fres[0] = 1.0
    seg.x0[0], seg.y0[0] = x0, y0
    seg.z1[0] = 1.0
    return seg


def _frame_values(chan, seg, z, px, py, phase):
    shape = px.shape
    vals = D.beam_values(seg, np.array([0]), z, px.reshape(1, -1), py.reshape(1, -1), K, 0.0)[0]
    return vals.reshape(shape) * phase


@pytest.mark.parametrize("rbend, n_sec, L, jmax", [(None, 200, 0.05, 80.0), (None, 12, 0.05, 80.0), (2156.25, 200, 0.02, 120.0)])
def test_modal_evolution_reproduces_a_gaussian_beam(rbend, n_sec, L, jmax):
    """e := a Gaussian beam of waist 4 um centred 18 um off axis (wall value 0.1 of the peak);
    its frame values at z = 0 (disc) and on the wall (sections) feed the modal evolution,
    which must reproduce the beam's frame field at L: a straight bore to 1e-6 of the norm
    (each mode evolves exactly), 12 sections (0.6 rad per section) to 1e-3; in a bent bore
    the Galerkin coupling truncated to the Disc leaves ~1e-2 (a lab-straight beam is tilted
    in the bend frame), still far below the error without the boundary part."""
    chan0 = FBChannel(np.zeros(2), np.array([1.0, 0.0]) if rbend else None, rbend, Disc(AW, jmax), K)
    disc = chan0.disc
    seg = _beam(4.0e-6, 18.0e-6, 0.0)
    seg.z1[0] = L
    zs = np.linspace(0.0, L, n_sec + 1)
    # entrance values of e on the polar grid -> coefficients
    px, py, ph = B.frame_disc(chan0, 0.0)
    c0 = disc.to_modes(_frame_values(chan0, seg, 0.0, px, py, ph))
    # wall trace of the method field at the sections: e = -g on the wall, so g = -e there
    g = np.zeros((len(zs), disc.nphi), complex)
    for j, z in enumerate(zs):
        rx, ry, rph = B.frame_ring(chan0, z)
        g[j] = -_frame_values(chan0, seg, z, rx, ry, rph)
    mb = B.ModalBound(chan0)
    ghat = B.ring_hat(disc, g)
    bnd = mb.boundary_part(zs, ghat, L)
    free, _ = chan0.propagate(c0, L)
    cL = free + bnd
    # the beam itself at L on the polar grid
    px, py, ph = B.frame_disc(chan0, L)
    eL = _frame_values(chan0, seg, L, px, py, ph)
    cL_true = disc.to_modes(eL)
    n_true = mb.norm(cL_true)
    assert n_true > 0.0
    tol = (1e-6 if n_sec >= 100 else 1e-3) if rbend is None else 1e-2
    err = mb.norm(cL - cL_true)
    assert err <= tol * n_true
    assert err < 0.3 * mb.norm(free - cL_true)        # the boundary part is what closes the gap
    assert mb.norm(c0) + mb.norm(bnd) >= n_true * (1.0 - 1e-9)
    if rbend is None:
        t1, t2 = mb.tails(zs, ghat, B.extra_hat(disc, g))
        assert 0.0 <= t1 + t2 < 0.1 * n_true


def test_zero_trace_leaves_the_initial_norm():
    """A beam far from the wall: no trace, the modal evolution is unitary."""
    chan0 = FBChannel(np.zeros(2), None, None, Disc(AW, 60.0), K)
    disc = chan0.disc
    seg = _beam(2.0e-6, 0.0, 0.0)
    zs = np.linspace(0.0, L, 101)
    px, py, ph = B.frame_disc(chan0, 0.0)
    c0 = disc.to_modes(_frame_values(chan0, seg, 0.0, px, py, ph))
    g = np.zeros((len(zs), disc.nphi), complex)
    for j, z in enumerate(zs):
        rx, ry, rph = B.frame_ring(chan0, z)
        g[j] = _frame_values(chan0, seg, z, rx, ry, rph)
    mb = B.ModalBound(chan0)
    bnd = mb.boundary_part(zs, B.ring_hat(disc, g), L)
    assert mb.norm(bnd) < 1e-6 * mb.norm(c0)
    free, _ = chan0.propagate(c0, L)
    assert abs(mb.norm(free) - mb.norm(c0)) < 1e-9 * mb.norm(c0)

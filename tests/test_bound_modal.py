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
    itp = mb.interp_error(zs, ghat)
    assert 0.0 <= itp < (1e-3 if n_sec >= 100 else 0.05) * n_true     # h^3 s''/8 per section


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


def test_square_modal_evolution_reproduces_a_gaussian_beam():
    """Square bore |x|, |y| < a_w: the same Gaussian beam as e (wall value 0.1 of the peak on
    the +x face), its entrance modes and face traces feed the sine-mode evolution, which must
    return the beam at L (straight bore: each mode exact; the face integrals are midpoint
    sums on the outline nodes, h = a_w/240)."""
    sq = B.Square(AW, 60)
    sb = B.SquareBound(sq, K)
    seg = _beam(4.0e-6, 18.0e-6, 0.0)
    L = 0.05
    seg.z1[0] = L
    zs = np.linspace(0.0, L, 201)
    gx, gy = sq.grid()
    vals = lambda z, px, py: D.beam_values(seg, np.array([0]), z, px.reshape(1, -1), py.reshape(1, -1), K, 0.0)[0].reshape(px.shape)
    c0 = sq.to_modes(vals(0.0, gx, gy))
    g = np.array([-vals(z, sq.face_x, sq.face_y) for z in zs])            # e = -g on the wall
    G = sb.face_coef(g)
    assert G.shape == (len(zs), 4, sq.nq)
    bnd = sb.boundary_part(zs, G, L)
    free = c0 * np.exp(-1j * sb.alpha * L)
    cL_true = sq.to_modes(vals(L, gx, gy))
    n_true = sb.norm(cL_true)
    err = sb.norm(free + bnd - cL_true)
    assert err <= 1e-3 * n_true
    assert err < 0.3 * sb.norm(free - cL_true)
    t1, t2 = sb.tails(zs, G)
    assert 0.0 <= t1 + t2 < 0.1 * n_true
    assert 0.0 <= sb.interp_error(zs, G) < 1e-3 * n_true
    # the outline order and the arc index agree: node j is met first along its own ray
    px, py = sq.lab(0.0, 0.0, sq.face_x, sq.face_y)
    assert np.array_equal(sq.arc_index(0.0, 0.0, 0.5 * px, 0.5 * py), np.arange(4 * sq.nq))


def test_square_face_coefficients_are_the_sine_transform():
    sq = B.Square(AW, 6, nq=16)
    sb = B.SquareBound(sq, K)
    g = np.random.default_rng(1).normal(size=4 * sq.nq) + 0j
    G = sb.face_coef(g)
    n = np.arange(1, sq.nq + 1)
    # face +x: nodes y = s; face +y: nodes x = -s (reversed), face -x: y = -s, face -y: x = s
    for f, coord in enumerate((sq.s, -sq.s, -sq.s, sq.s)):
        S = np.sin(n[:, None] * math.pi * (coord[None, :] + AW) / (2.0 * AW))
        assert np.allclose(G[f], (S @ g[f * sq.nq:(f + 1) * sq.nq]) * sq.h)


def test_damping_contracts_the_evolution_and_matches_the_slab_form():
    """DIR-ell damping: the square's per-mode rate is the sum of the two slab rates, the damped
    evolution contracts, and the Filon weights hold for the complex exponent."""
    delta, beta = 7.125763840840871e-06, 9.207952277589002e-08
    sq = B.Square(AW, 20, nq=64)
    gam = B.square_damping(sq, K, delta, beta)
    g1 = B.slab_damping(np.arange(1, 21) * math.pi / (2.0 * AW), K, delta, beta, AW)
    assert np.allclose(gam, g1[:, None] + g1[None, :]) and np.all(gam > 0.0)
    assert gam[-1, -1] > 10.0 * gam[0, 0]                     # steep modes lose more
    sb = B.SquareBound(sq, K, gam)
    c = np.ones((20, 20), complex)
    assert sb.norm(c * np.exp(-1j * sb.alpha * 0.05)) < sb.norm(c)
    th = (sb.alpha * 1e-3).ravel()[:5]
    e0, e1 = B._filon(th)
    t = np.linspace(0.0, 1.0, 20001)
    for j in range(5):
        w0 = np.trapezoid(np.exp(-1j * th[j] * (1.0 - t)), t)
        w1 = np.trapezoid(np.exp(-1j * th[j] * t) * t, t)
        assert abs(e0[j] - w0) < 1e-8 and abs(e1[j] - w1) < 1e-8


def test_square_projection_splits_the_field_into_the_box_and_a_remainder():
    sq = B.Square(AW, 12, nq=48)
    rng = np.random.default_rng(3)
    xf = rng.uniform(-AW, AW, 4000); yf = rng.uniform(-AW, AW, 4000)
    wq = np.full(4000, 4.0 * AW * AW / 4000)
    m = np.arange(1, 13)
    phi = lambda i, j: np.sin(i * math.pi * (xf + AW) / (2 * AW)) * np.sin(j * math.pi * (yf + AW) / (2 * AW)) / AW
    E = 2.0 * phi(3, 5) - 1j * phi(12, 1) + 0.5 * np.sin(20 * math.pi * (xf + AW) / (2 * AW)) * phi(1, 1) * AW
    c, resid = sq.project(E, wq, xf, yf)
    assert abs(c[2, 4] - 2.0) < 0.1 and abs(c[11, 0] + 1j) < 0.1      # Monte-Carlo quadrature: ~1/sqrt(P)
    assert np.allclose(E - resid, np.einsum("pm,mp->p", np.sin(m[:, None] * math.pi * (xf[None, :] + AW) / (2 * AW)).T @ c,
                                            np.sin(m[:, None] * math.pi * (yf[None, :] + AW) / (2 * AW))) / AW)


def _straight_bound(chan, mb, zs, e_init, g_ring, L):
    """B of a straight bore from the entrance defect e_init (polar grid) and the method's ring
    trace: the components of bound_modal.bores_bound for one bore."""
    disc = chan.disc
    wq = (disc.wr * disc.r)[:, None] * (2.0 * math.pi / disc.nphi)
    e0 = math.sqrt(float(np.sum(wq * np.abs(e_init) ** 2)))
    c0 = disc.to_modes(e_init)
    r0 = math.sqrt(float(np.sum(wq * np.abs(e_init - disc.to_grid(c0)) ** 2)))
    ghat = B.ring_hat(disc, g_ring)
    bnd_c = mb.boundary_part(zs, ghat, L)
    prop, _ = chan.propagate(c0, L)
    t1, t2 = mb.tails(zs, ghat, B.extra_hat(disc, g_ring))
    itp = mb.interp_error(zs, ghat)
    exact = mb.norm(prop + bnd_c) + r0 + itp + t1 + t2
    tri = e0 + mb.norm(bnd_c) + itp + t1 + t2
    return min(exact, tri), prop + bnd_c, c0


@pytest.mark.parametrize("wrong", ["none", "waist", "offset", "scale"])
def test_bound_covers_the_error_of_a_wrong_method_field(wrong):
    """Ladder step 1 in 3D: E := the Dirichlet evolution of a Gaussian beam A, the method field
    := a free beam that is A itself (wall leakage only) or a wrong one (waist 3 um, centre
    16 um, amplitude x1.3).  B must cover ||E(L) - method(L)|| within the Disc and stay within
    x1.6 of it (x2.5 for the leakage alone, where the mode tail is comparable); a wrong input
    raises both."""
    chan = FBChannel(np.zeros(2), None, None, Disc(AW, 80.0), K)
    disc, mb = chan.disc, B.ModalBound(chan)
    L = 0.05
    zs = np.linspace(0.0, L, 201)
    truth = _beam(4.0e-6, 18.0e-6, 0.0)
    truth.z1[0] = L
    method = {"none": _beam(4.0e-6, 18.0e-6, 0.0), "waist": _beam(3.0e-6, 18.0e-6, 0.0),
              "offset": _beam(4.0e-6, 16.0e-6, 0.0), "scale": _beam(4.0e-6, 18.0e-6, 0.0)}[wrong]
    method.z1[0] = L
    if wrong == "scale":
        method.a[0] = 1.3
    px, py, ph = B.frame_disc(chan, 0.0)
    a0 = _frame_values(chan, truth, 0.0, px, py, ph)
    m0 = _frame_values(chan, method, 0.0, px, py, ph)
    g = np.zeros((len(zs), disc.nphi), complex)
    for j, z in enumerate(zs):
        rx, ry, rph = B.frame_ring(chan, z)
        g[j] = _frame_values(chan, method, z, rx, ry, rph)            # the method's wall trace
    bound, eL_modal, _ = _straight_bound(chan, mb, zs, a0 - m0, g, L)
    # the actual error within the Disc: E(L) = Dirichlet evolution of A(0), method(L) = free beam
    eA, _ = chan.propagate(disc.to_modes(a0), L)
    px, py, ph = B.frame_disc(chan, L)
    cm_L = disc.to_modes(_frame_values(chan, method, L, px, py, ph))
    err = mb.norm(eA - cm_L)
    scale = mb.norm(disc.to_modes(a0))
    assert bound >= err * (1.0 - 1e-6)
    assert bound <= (2.5 if wrong == "none" else 1.6) * err          # leakage only: the mode tail is comparable
    assert mb.norm(eL_modal - (eA - cm_L)) < 1e-3 * scale             # the exact modal part is e(L) itself
    if wrong == "none":
        assert err < 0.05 * scale                                        # only the wall leakage of A
    else:
        assert err > 0.1 * scale and bound > 0.1 * scale                 # the wrong input is seen


def test_square_tail_covers_the_infinite_normal_index():
    """A stationary method field whose wall trace is one face harmonic (n = 9) excites the modes
    (m, 9) for every normal index m; after L = 8/pi each has phase -1, so the exact relative
    error is 2.  The tail must reach m -> infinity: B >= 2 (aw = k = 1, M = 8, nq = 64)."""
    sq = B.Square(1.0, 8, nq=64)
    f = lambda x, y: np.sinh(9 * math.pi * x / 2) / math.sinh(9 * math.pi / 2) * np.sin(9 * math.pi * (y + 1) / 2)
    sample_disc = lambda b, radii, phis, stretch, cx, cy: f(radii[:, None] * stretch[None, :] * np.cos(phis)[None, :],
                                                           radii[:, None] * stretch[None, :] * np.sin(phis)[None, :]) + 0j
    sample_outline = lambda b, zr, sq_, cx, cy: f(sq_.face_x, sq_.face_y) + 0j
    out = B.squares_bound([sq], [(0.0, 0.0)], 1.0, 8.0 / math.pi, sample_disc, sample_outline,
                          lambda px, py: f(px, py) + 0j, phase_step=0.7, nr=100, nphi=2048)
    assert abs(out["C_fit"][0] - 1.0) < 1e-9 and out["B_e0"] < 1e-9
    assert out["B"] >= 2.0 * (1.0 - 1e-6)
    assert out["B"] < 2.3


def test_fan_scale_from_the_deposit_norm_flags_an_amplitude_defect():
    """Method field = 2 x the lowest Disc mode: with the deposit-unit norm of E_in fixing the
    scale, the doubled amplitude is error (B ~ 1); the least-squares scale alone hides it."""
    chan = FBChannel(np.zeros(2), None, None, Disc(AW, 20.0), K)
    disc = chan.disc
    c = np.zeros(disc.valid.shape, complex)
    c[list(disc.ms).index(0), 0] = 1.0
    phi = disc.to_grid(c)
    norm_phi = math.sqrt(disc.norm(c))
    sample = lambda b, zr, radii, phis, cx, cy: (2.0 * phi if radii.size > 1 else np.zeros((1, disc.nphi), complex))
    fixed = B.bores_bound([chan], 0.01, sample, lambda ch: phi, 0.7, 0.0, e_in_dep=norm_phi)
    lsq = B.bores_bound([chan], 0.01, sample, lambda ch: phi, 0.7, 0.0)
    assert abs(fixed["C_fit"][0] - 1.0) < 1e-6 and abs(fixed["scale_defect"] - 1.0) < 1e-6
    assert abs(fixed["B"] - 1.0) < 1e-3
    assert abs(lsq["C_fit"][0] - 2.0) < 1e-6 and lsq["B"] < 1e-6 and abs(lsq["scale_defect"]) < 1e-9

"""Stage 11 defect indicator: beam chain and evaluation against the stage's own beam
algebra, the entrance-kernel defect, the image cancellation on a flat wall, and the
exit tails of a straight ray."""
import cmath
import math

import numpy as np
import pytest

from formula.capsysred.gamma import propagate
from formula.capsysred.stages import beamlet_defect as D
from formula.capsysred.surfaces import CapillaryBundle

LAM = 12.398419843320026e-10 / 8.048
K = 2.0 * math.pi / LAM


def test_drift_amplitude_matches_gamma_propagate():
    rng = np.random.default_rng(3)
    for _ in range(20):
        zr_t, zr_s = rng.uniform(1e-3, 0.1, 2)
        s = rng.uniform(0.0, 0.5)
        q = (complex(0.0, zr_t), 0j, complex(0.0, zr_s))
        _, amp = propagate((zr_t, zr_s, 0.0), [s], [])
        mine = D.drift_amp(np.array([q[0]]), np.array([q[1]]), np.array([q[2]]), np.array([s]))[0]
        assert abs(mine - amp) < 1e-12 * abs(amp)


def test_beam_values_reproduce_a_free_gaussian_beam():
    """One segment from z = 0 with waist w0 at the start: at z the field is the ABCD beam
    (q -> q + z, amplitude q0/q) in the deposit convention exp(ik d^2 / 2 q), q = z - i z_R."""
    w0 = 1.5e-6
    zr = 0.5 * w0 * w0 * K
    seg = D.Segments(1)
    seg.qxx[0] = seg.qyy[0] = complex(0.0, zr)
    seg.a[0] = 1.0
    seg.fres[0] = 1.0
    seg.z1[0] = 1.0
    z = 0.02
    x = np.linspace(-6e-6, 6e-6, 25)
    px, py = np.meshgrid(x, x)
    vals = D.beam_values(seg, np.array([0]), z, px.reshape(1, -1), py.reshape(1, -1), K, 0.0)[0].reshape(px.shape)
    q0, q = complex(0.0, -zr), complex(z, -zr)
    exact = (q0 / q) * np.exp(1j * K * (px * px + py * py) / (2.0 * q))
    assert np.max(np.abs(vals - exact)) < 1e-10 * np.max(np.abs(exact))


def test_entrance_defect_shrinks_with_the_kernel():
    a_w, z_in = 24e-6, 0.675
    d_narrow, w_n = D.entrance_defect(K, a_w, z_in, complex(0.0, -0.5 * (0.3e-6) ** 2 * K), complex(0.0, -0.5 * (0.3e-6) ** 2 * K))
    d_wide, w_w = D.entrance_defect(K, a_w, z_in, complex(0.0, -0.5 * (3e-6) ** 2 * K), complex(0.0, -0.5 * (3e-6) ** 2 * K))
    assert w_n == pytest.approx(0.3e-6, rel=1e-6) and w_w == pytest.approx(3e-6, rel=1e-6)
    assert 0.0 < d_narrow < d_wide < 0.5
    # a Gaussian of width w smooths a disc edge over ~w: the defect scales like sqrt(w / a)
    assert d_wide / d_narrow == pytest.approx(math.sqrt(10.0), rel=0.35)


def test_image_cancels_on_a_flat_wall():
    """A ray bouncing once on the flat face x = +a of a square bore: the incident beam
    continued past the hit and the reflected beam continued before it cancel on the face."""
    from formula.capsysred.simulation import Simulation
    raw = {"precision": 32, "energy_kev": 8.048, "seed": 1,
           "capillary": {"bores": [{"center": [0.0, 0.0], "radius": 5e-6, "sides": 4}], "z0": 0.0, "z1": 0.2,
                         "source": {"shape": "gaussian", "size": 1e-7, "position": [0.0, 0.0, -0.5],
                                    "n_modes": 1, "n_rays": 1},
                         "screen": {"z": 0.2, "center": [0.0, 0.0], "edge_x": 1e-5, "edge_y": 1e-5, "nx": 10, "ny": 10}},
           "beamlet": {"w0": 1e-6, "w0_t": None, "window_sigmas": 3.0}}
    sim = Simulation.from_dict(raw)
    cap = sim.cfg.capillary
    bundle = CapillaryBundle(cap.bores, cap.z0, cap.z1)
    a = 5e-6
    theta = 4e-5                                   # slope toward +x; the face x = +a is hit at z_hit
    x_ent = 1e-6
    z_src = -0.5
    origin = [x_ent - theta * 0.5, 0.0, z_src]     # entrance point (x_ent, 0) at z = 0
    z_hit = (a - x_ent) / theta
    hit = [a, 0.0, z_hit]
    z_rec = 0.2
    x_rec = a - theta * (z_rec - z_hit)
    rows = [(x_rec, 0.0, -theta, 0.0, 0.0, [theta], [hit])]
    rows[0] = (x_rec, 0.0, -theta / math.sqrt(1 + theta * theta), 0.0,
               math.dist(origin, hit) + math.dist(hit, (x_rec, 0.0, z_rec)), [theta], [hit])
    zr = 0.5 * (1e-6) ** 2 * K
    fres = (2.0 * sim.delta_f, 2.0 * sim.beta_f)
    seg = D.build_segments(rows, origin, z_rec, bundle, zr, zr, 0.0, fres, K,
                           np.array([[0.0, 0.0]]), np.array([a]), 0.0)
    assert seg.qxx.size == 2 and seg.final[1] and not seg.final[0]
    # on the face the mirror image equals the incident beam, so with the grazing Fresnel
    # factor r = -1 + O(theta / theta_c) the physical pair cancels to |1 + r|
    r = seg.fres[1]
    assert abs(r + 1.0) < 0.05
    y = np.linspace(-3e-6, 3e-6, 7)
    for z in (z_hit - 0.03, z_hit, z_hit + 0.03):
        px = np.full((1, y.size), a)
        inc = D.beam_values(seg, np.array([0]), z, px, y[None, :], K, z_src)[0]
        ref = D.beam_values(seg, np.array([1]), z, px, y[None, :], K, z_src)[0]
        assert np.max(np.abs(ref / r - inc)) < 1e-4 * np.max(np.abs(inc))      # phase rounding of k*opl ~ 1e-6 rad
        assert np.max(np.abs(inc + ref)) <= (abs(1.0 + r) + 1e-4) * np.max(np.abs(inc))


def test_exit_tails_vanish_without_bounces():
    """A straight ray has one (final) segment: no tails on the exit plane."""
    from formula.capsysred.simulation import Simulation
    raw = {"precision": 32, "energy_kev": 8.048, "seed": 1,
           "capillary": {"bores": [{"center": [0.0, 0.0], "radius": 5e-6}], "z0": 0.0, "z1": 0.2,
                         "source": {"shape": "gaussian", "size": 1e-7, "position": [0.0, 0.0, -0.5],
                                    "n_modes": 1, "n_rays": 1},
                         "screen": {"z": 0.2, "center": [0.0, 0.0], "edge_x": 1e-5, "edge_y": 1e-5, "nx": 10, "ny": 10}},
           "beamlet": {"w0": 1e-6, "w0_t": None, "window_sigmas": 3.0}}
    sim = Simulation.from_dict(raw)
    cap = sim.cfg.capillary
    bundle = CapillaryBundle(cap.bores, cap.z0, cap.z1)
    origin = [0.0, 0.0, -0.5]
    rows = [(1e-6, 0.0, 1e-6 / 0.7, 0.0, 0.7, [], [])]
    zr = 0.5 * (1e-6) ** 2 * K
    seg = D.build_segments(rows, origin, 0.2, bundle, zr, zr, 0.0, (2 * sim.delta_f, 2 * sim.beta_f), K,
                           np.array([[0.0, 0.0]]), np.array([5e-6]), 0.0)
    assert seg.qxx.size == 1 and seg.final[0]
    assert D.exit_tails(seg, sim.cfg.raw["capillary"]["bores"], 5e-6, 0.2, K, -0.5, 0.5e-6, 4.0) == (0.0, 0.0)
    trace, noise = D.wall_trace(seg, sim.cfg.raw["capillary"]["bores"], 5e-6, np.array([0.1]), K, -0.5, 0.5e-6, 4.0)
    assert trace.shape == (1,) and trace[0] >= 0.0 and noise[0] == trace[0]      # one ray: all noise


def test_polygon_outline_stretch():
    """A square of apothem a (no rotation): faces at angle 0 (stretch 1), corners at 45 deg (sqrt 2)."""
    phis = np.array([0.0, math.pi / 4, math.pi / 2, math.pi])
    s = D._stretch(D._shape({"sides": 4, "rotation_deg": 0}), phis)
    assert np.allclose(s, [1.0, math.sqrt(2.0), 1.0, 1.0])
    assert np.allclose(D._stretch(D._shape({"radius": 1.0}), phis), 1.0)

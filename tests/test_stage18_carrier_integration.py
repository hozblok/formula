"""Independent carrier audits and an integration guard for the optional route."""

from decimal import Decimal, localcontext
import json

import numpy as np
import pytest

from formula.capsysred.config import B9_DEFAULTS
from formula.capsysred.stages._b9_carrier import _mesh, _select_carriers, carrier_field


def shared_edge_diagnostics():
    triangles = np.array([[[0., 0.], [1., 0.], [0., 1.]],
                          [[1., 0.], [0., 0.], [.5, -.3]]])
    amplitude = 1+.2*triangles[..., 0]+1j*(.3-.1*triangles[..., 1])
    mesh = dict(triangles=triangles, vertex_amplitude=amplitude,
                vertex_phase=np.zeros((2, 3)))
    _, _, phase, gradient = _mesh(mesh)
    labels, carriers, _ = _select_carriers(triangles, phase, gradient, 2., 2)
    slopes = carriers[labels, 0]
    x = np.linspace(0., 1., 1001)
    exact = (1+.2*x+.3j)*np.exp(1j*x*x)
    rows = []
    for n in (1, 2, 4, 8):
        j = np.minimum((n*x).astype(int), n-1)
        left, right, weight = j/n, (j+1)/n, n*x-j
        traces = []
        for c in slopes:
            low = (1+.2*left+.3j)*np.exp(1j*(left*left-c*left))
            high = (1+.2*right+.3j)*np.exp(1j*(right*right-c*right))
            traces.append(np.exp(1j*c*x)*((1-weight)*low+weight*high))
        jump = abs(traces[0]-traces[1])
        rows.append(dict(subdivisions=n, maximum_jump=float(jump.max()),
                         midpoint_jump=float(jump[500]),
                         endpoint_jump=float(max(jump[0], jump[-1])),
                         maximum_trace_error=float(max(abs(t-exact).max() for t in traces))))
    return dict(kappa=2., carriers_per_m=carriers[labels].tolist(), rows=rows,
                interpretation="Continuous affine amplitude, zero exit phase; only carrier-demodulated P1 traces differ. The parent midpoint becomes an interpolation node for even subdivisions, so maximum edge jump is the meaningful refinement control.")


def test_continuous_exit_model_can_have_carrier_representation_edge_jumps():
    audit = shared_edge_diagnostics()
    rows = audit["rows"]
    assert all(row["endpoint_jump"] < 1e-14 for row in rows)
    assert rows[0]["midpoint_jump"] > .019
    jumps = [row["maximum_jump"] for row in rows]
    assert jumps[0] > jumps[1] > jumps[2] > jumps[3] > 0
    assert jumps[3] < .1*jumps[0]


def unit_amplitude_power_diagnostics():
    nodes, weights = np.polynomial.legendre.leggauss(32)
    nodes, weights = (nodes+1)/2, weights/2
    u, v = np.meshgrid(nodes, nodes, indexing="ij")
    bary = np.stack(((1-u)*(1-v), u, (1-u)*v), axis=-1)
    area_weights = 2*weights[:, None]*weights[None, :]*(1-u)
    rows = []
    for phase in (np.array([0., 2*np.pi/3, 4*np.pi/3]), np.array([0., 2*np.pi, 4*np.pi])):
        vertex = np.exp(1j*phase)
        approximation, exact = bary@vertex, np.exp(1j*(bary@phase))
        rows.append(dict(vertex_phase_rad=phase.tolist(), exact_normalized_power=1.,
            p1_power_integrated=float(np.sum(area_weights*abs(approximation)**2)),
            p1_power_formula=float((3+abs(vertex.sum())**2)/12),
            relative_field_L2_error=float(np.sqrt(np.sum(area_weights*abs(approximation-exact)**2)))))
    return rows


def test_unit_amplitude_vertex_samples_do_not_certify_power_or_phase():
    loss, alias = unit_amplitude_power_diagnostics()
    assert loss["p1_power_integrated"] == pytest.approx(.25, abs=2e-14)
    assert alias["p1_power_integrated"] == pytest.approx(1., abs=2e-14)
    assert alias["relative_field_L2_error"] == pytest.approx(np.sqrt(2), abs=2e-14)


def test_different_local_origins_translation_and_orientation_preserve_complex_sum():
    base = np.array([[0., 0.], [.06, 0.], [.02, .04]])
    triangles = base[None, :, :]+np.array([[-.4, .1], [.9, -.6]])[:, None, :]
    phase = np.einsum("tvi,ti->tv", triangles, [[9., -5.], [-11., 3.]])
    phase += np.array([.7, -.4])[:, None]
    mesh = dict(triangles=triangles, vertex_phase=phase,
                vertex_amplitude=np.array([[1+.4j, .5-.1j, .8+.2j], [.2-.5j, .9+.3j, -.1+.7j]]))
    args = dict(subdivisions=2, groups=2, k=7., distance=1., x=np.array([-.15, .21]),
                y=np.array([-.12, .08]), cell_width=.02, pixel_order=4,
                max_triangles_per_batch=2, edge_order=12, backend="direct")
    combined, _ = carrier_field(mesh, **args)
    pieces = [carrier_field({name: value[i:i+1] for name, value in mesh.items()},
                            **args)[0] for i in (0, 1)]
    np.testing.assert_allclose(combined, sum(pieces), rtol=2e-10, atol=2e-14)
    offset = np.array([8., -9.])
    translated = {**mesh, "triangles": triangles+offset}
    moved, _ = carrier_field(translated, **{**args, "x": args["x"]+offset[0], "y": args["y"]+offset[1]})
    np.testing.assert_allclose(moved, combined, rtol=2e-10, atol=2e-14)
    reversed_mesh = {name: value[:, ::-1] for name, value in mesh.items()}
    reversed_field, _ = carrier_field(reversed_mesh, **args)
    np.testing.assert_allclose(reversed_field, combined, rtol=2e-10, atol=2e-14)


@pytest.mark.skipif("carrier_groups" not in B9_DEFAULTS,
                    reason="Optional production carrier route awaits completion of frozen baseline runs")
@pytest.mark.parametrize("groups", [0, 32])
def test_optional_carrier_replay_preserves_native_cells_and_source_ensemble(tmp_path, monkeypatch, groups):
    from formula.capsysred import Simulation, rays_v3
    from formula.capsysred.rays import geometry_metadata
    from formula.capsysred.stages import _b9_carrier, stage18

    pytest.importorskip("finufft")
    captured = []
    original = _b9_carrier.carrier_field

    def audited(*args, **kwargs):
        assert groups > 0, "The default route must keep the baseline representation"
        field, stats = original(*args, **kwargs)
        captured.append((field.copy(), kwargs.copy()))
        return field, stats

    monkeypatch.setattr(_b9_carrier, "carrier_field", audited)
    monkeypatch.setattr(stage18, "carrier_field", audited, raising=False)
    sim = Simulation.from_dict({
        "precision": 32, "energy_kev": 8.048,
        "capillary": {
            "bores": [{"center": [0., 0.], "radius": 20e-6}], "z0": 0., "z1": .02,
            "source": {"shape": "gaussian", "size": .25e-6, "position": [0., 0., -.1],
                       "n_modes": 2, "n_rays": 36},
            "screen": {"z": .02, "edge_x": 2e-6, "edge_y": 2e-6, "nx": 9, "ny": 9},
            "screens": [{"z": .08}],
        },
        "b9_estimator": {"max_modes": 2, "rays_per_mode": 36, "screen_index": 1,
                         "amplitude_mode": "tube_flux", "map_stride": 2,
                         "phase_subdivisions": [2], "pixel_order": 2, "holdout_stride": 0,
                         "max_triangles_per_batch": 7, "carrier_groups": groups},
    })
    archive = tmp_path/"rays"
    rays_v3.write_fingerprint(archive, {"format": 3, "geometry": geometry_metadata(sim.cfg)})
    entries = []
    for mode, sx in enumerate((Decimal("-.0000002"), Decimal(".0000002"))):
        writer = rays_v3.SectionWriter(archive, "capillary", mode, 0, 36,
                                       origin=[str(sx), "0", "-.1"])
        with localcontext() as context:
            context.prec = 80
            for ray, (x, y) in enumerate((x, y) for x in np.linspace(-2e-6, 2e-6, 6)
                                        for y in np.linspace(-2e-6, 2e-6, 6)):
                path = ((Decimal(str(x))-sx)**2+Decimal(str(y))**2+Decimal(".12")**2).sqrt()
                writer.write_row({"stage": "capillary", "mode": mode, "ray": ray, "fate": "screen",
                                  "pixel": 0, "opl": str(path), "sins": [], "refl": [],
                                  "x": x, "y": y, "dx": (x-float(sx))/float(path), "dy": y/float(path)})
        entries.append(writer.close())
    rays_v3.write_index(archive, entries)
    output = tmp_path/"result"
    sim.replay(str(archive), str(output), stages=[18])
    folder = output/"stage18"
    meta = json.loads((folder/"meta.json").read_text())
    assert meta["options"]["carrier_groups"] == groups
    assert meta["completed_source_modes"] == 2
    assert meta["full_coherence_computed"] and not meta["accuracy_validated"]
    for mode in meta["modes"]:
        assert "timing_scope" in mode["variants"][0]
    with np.load(folder/"map-b36-s2-m2.npz") as result:
        assert int(result["carrier_groups"]) == groups
        width = float(result["receiver_width_m"])
        np.testing.assert_allclose(np.diff(result["x"]), 2*width, rtol=1e-13)
        assert np.isfinite(result["mu"]).all()
        assert np.max(abs(result["mu"])) <= 1+1e-12
        if groups:
            assert len(captured) == 2
            assert "_b9_carrier.py" in meta["implementation_sha256"]
            assert all(item[1]["cell_width"] == width for item in captured)
            fields = np.stack([item[0] for item in captured])
            ref = tuple(result["ref_index"])
            np.testing.assert_allclose(result["I"], np.mean(abs(fields)**2, axis=0), rtol=2e-13)
            cross = fields*fields[(slice(None), *ref)].conj()[:, None, None]
            np.testing.assert_allclose(result["W"], cross.mean(axis=0), rtol=2e-13, atol=1e-14)
            with np.load(folder/"field-b36-s2-mode0.npz") as raw:
                np.testing.assert_array_equal(raw["field"], fields[0])
                assert int(raw["carrier_groups"]) == groups
                assert int(raw["phase_subdivisions"]) == 2
        else:
            assert not captured

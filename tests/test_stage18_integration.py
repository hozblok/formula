"""Stage 18 routes archives without changing the existing stage defaults."""

from pathlib import Path
from types import SimpleNamespace
from decimal import Decimal, localcontext
import json

import pytest

from formula.capsysred import Simulation


def make_scene():
    return Simulation.from_dict({
        "precision": 32,
        "capillary": {
            "source": {"shape": "point", "size": 0, "position": [0, 0, -.1],
                       "n_modes": 1, "n_rays": 1},
            "screen": {"z": .2}, "screens": [{"z": .4}],
        },
    })


def test_stage18_replay_does_not_scan_a_generic_archive(tmp_path, monkeypatch):
    import formula.capsysred.simulation as module

    seen = []
    sim = make_scene()

    def refuse_scan(*args, **kwargs):
        raise AssertionError("Stage 18 must read selected v3 modes itself")

    def preflight_inputs(got_sim, options):
        assert got_sim is sim
        seen.append("inputs")

    def preflight_output(output):
        assert not Path(output).exists()
        seen.append("output")

    def run(got_sim, output, options, *, rays_paths, log):
        assert got_sim is sim
        assert Path(output).is_dir()
        assert rays_paths == ["selected-v3-archive"]
        assert callable(log)
        seen.append("run")
        return {"results": {"status": "experimental"}, "files": [],
                "report": ["Stage 18 test result"]}

    backend = SimpleNamespace(preflight_b9_inputs=preflight_inputs,
                              preflight_b9_output=preflight_output,
                              run_b9_stage=run)
    monkeypatch.setattr(module, "_b9_backend", lambda: backend)
    monkeypatch.setattr(module, "RaysReader", refuse_scan)
    monkeypatch.setattr(module, "MultiRaysReader", refuse_scan)
    result = sim.replay("selected-v3-archive", str(tmp_path / "result"), stages=[18])
    assert seen == ["inputs", "output", "run"]
    assert sim.results["b9"] == {"status": "experimental"}
    assert result["files"]


def test_stage18_input_preflight_precedes_output_creation(tmp_path, monkeypatch):
    import formula.capsysred.simulation as module

    def reject(*args, **kwargs):
        raise ValueError("injected Stage 18 preflight rejection")

    monkeypatch.setattr(module, "_b9_backend", lambda: SimpleNamespace(
        preflight_b9_inputs=reject,
        preflight_b9_output=lambda output: pytest.fail("output checked after rejection")))
    with pytest.raises(ValueError, match="preflight rejection"):
        make_scene().run(str(tmp_path / "result"), stages=[18])
    assert not (tmp_path / "result").exists()


def test_stage18_requires_capillary_before_backend_load(tmp_path, monkeypatch):
    import formula.capsysred.simulation as module

    monkeypatch.setattr(module, "_b9_backend", lambda: pytest.fail("backend loaded"))
    sim = Simulation.from_dict({
        "free": {"source": {"shape": "point", "size": 0, "position": [0, 0, -.1],
                            "n_modes": 1, "n_rays": 1}}})
    with pytest.raises(ValueError, match="capillary.source"):
        sim.run(str(tmp_path / "result"), stages=[18])
    assert not (tmp_path / "result").exists()


def test_stage18_cli_dispatches_explicit_stage(tmp_path, monkeypatch):
    from formula.capsysred import __main__ as cli

    calls = []

    def replay(paths, out, stages):
        calls.append((paths, out, stages))
        return {"out_dir": out, "files": []}

    monkeypatch.setattr(cli.Simulation, "from_yaml", lambda path: SimpleNamespace(replay=replay))
    out = str(tmp_path / "result")
    assert cli.main(["scene.yaml", "--stages", "18", "--replay", "archive", "-o", out]) == 0
    assert calls == [(["archive"], out, [18])]


@pytest.mark.parametrize("amplitude_mode", ["point_jacobian", "tube_flux"])
def test_stage18_full_archive_replay_writes_complex_coherence(tmp_path, amplitude_mode):
    import numpy as np
    from formula.capsysred import rays_v3
    from formula.capsysred.rays import geometry_metadata

    pytest.importorskip("finufft")
    sim = Simulation.from_dict({
        "precision": 32, "energy_kev": 8.048,
        "capillary": {
            "bores": [{"center": [0., 0.], "radius": 20e-6}], "z0": 0., "z1": .02,
            "source": {"shape": "point", "size": 0., "position": [0., 0., -.1],
                       "n_modes": 1, "n_rays": 144},
            "screen": {"z": .02, "edge_x": 2e-6, "edge_y": 2e-6, "nx": 9, "ny": 9},
            "screens": [{"z": .08}],
        },
        "b9_estimator": {"max_modes": 1, "rays_per_mode": 144, "screen_index": 1, "amplitude_mode": amplitude_mode,
                         "map_stride": 2, "phase_subdivisions": [1, 2], "pixel_order": 2},
    })
    archive = tmp_path / "rays"
    rays_v3.write_fingerprint(archive, {"format": 3, "geometry": geometry_metadata(sim.cfg)})
    writer = rays_v3.SectionWriter(archive, "capillary", 0, 0, 144, origin=["0", "0", "-.1"])
    with localcontext() as context:
        context.prec = 80
        for ray, (x, y) in enumerate((x, y) for x in np.linspace(-2e-6, 2e-6, 12)
                                    for y in np.linspace(-2e-6, 2e-6, 12)):
            path = (Decimal(str(x))**2 + Decimal(str(y))**2 + Decimal(".12")**2).sqrt()
            writer.write_row({"stage": "capillary", "mode": 0, "ray": ray, "fate": "screen",
                              "pixel": 0, "opl": str(path), "sins": [], "refl": [],
                              "x": float(x), "y": float(y), "dx": x/float(path), "dy": y/float(path)})
    rays_v3.write_index(archive, [writer.close()])
    output = tmp_path / "result"
    result = sim.replay(str(archive), str(output), stages=[18])
    meta = json.loads((output / "stage18/meta.json").read_text())
    assert meta["status"] == "experimental-GO-exit-contour-diffraction"
    assert meta["full_coherence_computed"] is True
    assert meta["accuracy_validated"] is False
    assert meta["completed_source_modes"] == 1
    assert meta["distance_after_exit_m"] == .06
    assert len(meta["outputs"]) == 2
    assert not (output / "stage18.partial").exists()
    for subdivisions in (1, 2):
        name = f"stage18/map-b144-s{subdivisions}-m1.npz"
        assert name in result["files"]
        with np.load(output / name) as data:
            assert data["mu"].shape == (5, 5)
            assert np.isfinite(data["I"]).all() and (data["I"] > 0).all()
            np.testing.assert_allclose(abs(data["mu"]), 1., atol=1e-12)
            assert np.isnan(data["mu_err"]).all()
            assert "matched_stage14_mu" in data
    sentinel = (output / "stage18/meta.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        sim.replay(str(archive), str(output), stages=[18])
    assert (output / "stage18/meta.json").read_bytes() == sentinel

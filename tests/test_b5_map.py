"""Canonical-map sampling normalization, reference binning and publication."""

from decimal import Decimal, localcontext
import itertools
import json

import numpy as np
import pytest

from formula.capsysred import Simulation, rays_v3
from formula.capsysred.rays import geometry_metadata
from formula.capsysred.stages import _b5_map as maps


def scene_and_archive(tmp_path):
    sim = Simulation.from_dict({"precision": 32, "energy_kev": 8.048,
        "capillary": {"bores": [{"center": [0., 0.], "radius": 2e-6}], "z0": 0., "z1": .02,
            "source": {"shape": "point", "size": 0., "position": [0., 0., -.01], "n_modes": 3, "n_rays": 16},
            "screen": {"z": .02, "edge_x": 8e-6, "edge_y": 8e-6, "nx": 21, "ny": 21}},
        "b5_estimator": {"provider": "archive_canonical", "max_modes": 3, "rays_per_mode": 16,
                         "map_stride": 3, "widths_m": [.3e-6], "map_snapshots": [1]}})
    archive = tmp_path/"rays"
    rays_v3.write_fingerprint(archive, {"format": 3, "geometry": geometry_metadata(sim.cfg)})
    entries = []
    for mode in range(3):
        writer = rays_v3.SectionWriter(archive, "capillary", mode, 0, 16, origin=["0", "0", "-.01"])
        with localcontext() as context:
            context.prec = 80
            for ray, (x, y) in enumerate(itertools.product(np.linspace(-1e-6, 1e-6, 4), repeat=2)):
                opl = (Decimal(str(x))**2+Decimal(str(y))**2+Decimal(".03")**2).sqrt()
                writer.write_row({"stage": "capillary", "mode": mode, "ray": ray,
                                  "fate": "screen", "pixel": 0, "opl": str(opl), "sins": [],
                                  "x": float(x), "y": float(y), "dx": x/float(opl), "dy": y/float(opl)})
        entries.append(writer.close())
    rays_v3.write_index(archive, entries)
    return sim, archive


def test_ustatistic_removes_self_bias_with_unequal_complex_rays_and_loss():
    population = np.array([[1+2j, 3-.5j], [-2+.3j, .2+1j], [0, 0]])
    debiased_i, debiased_w, plugin_i = [], [], []
    for first, second in itertools.product(population, repeat=2):
        individual = np.array([first, second])/2
        field = individual.sum(axis=0)
        result = maps.debiased_observables(field, np.sum(abs(individual)**2, axis=0),
                    np.sum(individual*individual[:, :1].conjugate(), axis=0), 2, 0)
        debiased_i.append(result["I"])
        debiased_w.append(result["W"])
        plugin_i.append(result["pluginI"])
    truth = population.mean(axis=0)
    np.testing.assert_allclose(np.mean(debiased_i, axis=0), abs(truth)**2, atol=1e-14)
    np.testing.assert_allclose(np.mean(debiased_w, axis=0), truth*truth[0].conjugate(), atol=1e-14)
    assert np.max(np.mean(plugin_i, axis=0)-abs(truth)**2) > .5


def test_native_binning_keeps_reference_and_drops_only_its_self_cross(tmp_path):
    sim, _ = scene_and_archive(tmp_path)
    grid = maps._map_grid(sim.cfg.capillary.screen, 3)
    reference = grid["reference"]
    left = (grid["x"][0], grid["y"][0])
    unselected = (grid["x0"]+2.5*grid["cell_width"], grid["y0"]+2.5*grid["cell_width"])
    points = np.array([reference, reference, left, unselected,
                       [grid["x0"]+grid["edge_x"], reference[1]]])
    result = maps.matched_ray_field(points, np.zeros(5), np.array([1, 1j, 2, 7, 8]), grid)
    assert result["I"][grid["ref_index"]] == pytest.approx(0.)
    assert result["W"][grid["ref_index"]] == pytest.approx(0.)
    assert result["W"][0, 0] == pytest.approx(2-2j)
    assert result["ray_count"].sum() == 3


def test_nonpositive_intensity_and_over_one_coherence_are_not_clipped():
    intensity = np.array([1., -1., 0., 1.])
    mu = maps.normalized_coherence(intensity, np.array([1., 2., 3., 1.2j]), 0)
    assert mu[0] == 1
    assert np.isnan(mu[1:3]).all()
    assert mu[3] == 1.2j
    assert maps._array_counts(intensity, mu)["over_one_mu"] == 1


def test_small_archive_map_and_snapshots_are_published(tmp_path):
    sim, archive = scene_and_archive(tmp_path)
    output = tmp_path/"output"
    result = maps.run_canonical_map(sim, output, sim.cfg.validate_b5_estimator(), rays_paths=[archive])
    meta = json.loads((output/"stage17/meta.json").read_text())
    assert meta["full_coherence_computed"] is True
    assert meta["accuracy_validated"] is False
    assert [v["source_modes"] for v in meta["outputs"]] == [1, 3]
    assert sum(v["transport"]["valid_count"] for v in meta["modes"]) == 48
    data = np.load(output/"stage17"/meta["outputs"][-1]["file"])
    assert data["I"].shape == (7, 7)
    assert data["mu_plugin"][tuple(data["ref_index"])].real == pytest.approx(1.)
    assert np.nanmax(abs(data["mu_plugin"])) <= 1+1e-12
    assert len(result["files"]) == 3
    assert not (output/"stage17.partial").exists()


def test_failed_worker_cleans_only_its_staging_directory(tmp_path, monkeypatch):
    sim, archive = scene_and_archive(tmp_path)
    def fail(job):
        raise RuntimeError("injected worker failure")
    monkeypatch.setattr(maps, "_mode_job", fail)
    output = tmp_path/"output"
    with pytest.raises(RuntimeError, match="injected"):
        maps.run_canonical_map(sim, output, sim.cfg.validate_b5_estimator(), rays_paths=[archive])
    assert not (output/"stage17.partial").exists()
    assert not (output/"stage17").exists()
    with pytest.raises(ValueError, match="exactly one"):
        maps.run_canonical_map(sim, output, sim.cfg.validate_b5_estimator(), rays_paths=None)


def test_jackknife_requires_all_positive_leave_one_intensities():
    rows_i = [np.array([1., 1.]), np.array([2., -5.]), np.array([4., 2.])]
    rows_w = [np.array([1., .3j]), np.array([2., .4j]), np.array([4., -.1j])]
    error, count = maps._jackknife(rows_i, rows_w, 0)
    assert error[0] == pytest.approx(0.)
    assert count[0] == 3
    assert np.isnan(error[1])
    assert count[1] == 1


def test_reflection_factor_fresnel_default_and_ideal_minus_one():
    from formula.capsysred.stages import stage17

    sins = [[], [.01], [.01, .02], [.01, .02, .03]]
    fresnel = stage17._reflection_factor(sins, 1e-5, 1e-7)
    assert np.array_equal(fresnel, stage17._fresnel_product(sins, 1e-5, 1e-7))
    ideal = stage17._reflection_factor(sins, 1e-5, 1e-7, "ideal_minus_one")
    assert ideal.dtype == complex and ideal.tolist() == [1, -1, 1, -1]
    with pytest.raises(ValueError, match="reflection"):
        stage17._reflection_factor(sins, 1e-5, 1e-7, "x")


def test_map_records_reflection_and_ideal_matches_default_without_reflections(tmp_path):
    sim, archive = scene_and_archive(tmp_path)
    sim.cfg.raw["b5_estimator"]["reflection"] = "ideal_minus_one"
    out = tmp_path/"ideal"
    maps.run_canonical_map(sim, out, sim.cfg.validate_b5_estimator(), rays_paths=[archive])
    meta = json.loads((out/"stage17/meta.json").read_text())
    assert meta["reflection"] == "ideal_minus_one" and meta["options"]["reflection"] == "ideal_minus_one"
    sim.cfg.raw["b5_estimator"]["reflection"] = "fresnel"
    out2 = tmp_path/"fresnel"
    maps.run_canonical_map(sim, out2, sim.cfg.validate_b5_estimator(), rays_paths=[archive])
    a = np.load(out/"stage17"/meta["outputs"][-1]["file"])
    b = np.load(out2/"stage17"/meta["outputs"][-1]["file"])
    assert np.array_equal(a["W"], b["W"])   # zero reflections: both factors are 1

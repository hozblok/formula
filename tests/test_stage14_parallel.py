"""Stage-14 parallel range builds (CAPSYSRED_STAGE14_JOBS): serial equivalence
and input pinning (workers take the config by value and the parent-fixed
archive identity).  Resume-after-kill is covered by the atomic per-range
files."""

from __future__ import annotations

import json
import math
import shutil

import pytest
import yaml

from formula.capsysred import Simulation, rays_v3
from formula.capsysred.screen import ScreenGrid


def _config() -> dict:
    return {
        "seed": 1402,
        "screen": {"nx": 3, "ny": 1, "edge_x": 3.0e-6, "edge_y": 1.0e-6},
        "capillary": {
            "source": {
                "shape": "point",
                "size": 3.0e-7,
                "position": [0.0, 0.0, -0.01],
                "n_modes": 4,
                "n_rays": 20,
            },
            "screen": {"nx": 3, "ny": 1, "edge_x": 3.0e-6, "edge_y": 1.0e-6},
            "screens": [{"nx": 5, "edge_x": 5.0e-6}],
        },
    }


def _write_v3(archive, sim: Simulation) -> None:
    """Smallest canonical v3 archive: one section per mode, two pixels lit."""
    from formula.capsysred.rays import geometry_metadata
    grid = ScreenGrid(sim.cfg.capillary.screen)
    ref = grid.ref_pixel(sim.cfg.capillary.screen.reference)
    x_ref, y_ref = grid.pixel_xy(ref)
    x_target, y_target = grid.pixel_xy(ref - 1)
    phase_flip = math.pi / float(sim.lines[0].k)
    n_modes, n_rays = sim.cfg.capillary.source.budget()
    rays_v3.write_fingerprint(archive, {
        "format": rays_v3.FORMAT,
        "geometry": geometry_metadata(sim.cfg),
    })
    entries = []
    for mode in range(n_modes):
        writer = rays_v3.SectionWriter(archive, "capillary", mode, 0, n_rays,
                                       origin=["0", "0", "-0.01"])
        for ray in range(n_rays):
            at_ref = ray < n_rays // 2
            x, y = (x_ref, y_ref) if at_ref else (x_target, y_target)
            opl = phase_flip if mode % 2 and not at_ref else 0.0
            writer.write_row({
                "stage": "capillary", "mode": mode, "ray": ray,
                "fate": "screen", "pixel": ref, "opl": repr(opl),
                "sins": [], "x": x, "y": y, "dx": 0.0, "dy": 0.0,
            })
        entries.append(writer.close())
    rays_v3.write_index(archive, entries)


def _scene(tmp_path):
    raw = _config()
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    sim = Simulation.from_yaml(str(cfg_path))
    archive = tmp_path / "arch"
    _write_v3(archive, sim)
    return cfg_path, archive


def _mu_rows(out_dir, sub: str = "") -> list[dict]:
    path = out_dir / "stage14" / sub / "mu-jack.jsonl"
    return [json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()]


def _assert_rows_match(serial: list[dict], parallel: list[dict]) -> None:
    assert len(serial) == len(parallel)
    for a, b in zip(serial, parallel):
        assert set(a) == set(b)
        for key in a:
            va, vb = a[key], b[key]
            if isinstance(va, float) and isinstance(vb, float):
                assert va == pytest.approx(vb, rel=1e-12, abs=1e-300, nan_ok=True)
            else:
                assert va == vb


def _assert_outputs_match(dir_a, dir_b) -> None:
    """Both stage-14 screens: the main one and the fan-out screen-1."""
    _assert_rows_match(_mu_rows(dir_a), _mu_rows(dir_b))
    _assert_rows_match(_mu_rows(dir_a, "screen-1"), _mu_rows(dir_b, "screen-1"))


def test_stage14_parallel_matches_serial(tmp_path, monkeypatch):
    """jobs=3 against the strict pass; the parallel run uses a dict config
    (workers need no yaml on disk)."""
    cfg_path, archive = _scene(tmp_path)
    monkeypatch.delenv("CAPSYSRED_STAGE14_JOBS", raising=False)
    Simulation.from_yaml(str(cfg_path)).replay(
        [str(archive)], str(tmp_path / "serial"), stages=[14])
    shutil.rmtree(tmp_path / "stage14-cache")
    monkeypatch.setenv("CAPSYSRED_STAGE14_JOBS", "3")
    Simulation.from_dict(_config()).replay(
        [str(archive)], str(tmp_path / "parallel"), stages=[14])
    _assert_outputs_match(tmp_path / "serial", tmp_path / "parallel")


def test_stage14_parallel_ignores_config_file_edit(tmp_path, monkeypatch):
    """A yaml edited under a running build must not leak into the workers."""
    cfg_path, archive = _scene(tmp_path)
    monkeypatch.delenv("CAPSYSRED_STAGE14_JOBS", raising=False)
    Simulation.from_yaml(str(cfg_path)).replay(
        [str(archive)], str(tmp_path / "serial"), stages=[14])
    shutil.rmtree(tmp_path / "stage14-cache")
    sim = Simulation.from_yaml(str(cfg_path))
    tampered = _config()
    tampered["trace"] = {"amplitude_min": 2.0}   # would absorb every ray
    cfg_path.write_text(yaml.safe_dump(tampered), encoding="utf-8")
    monkeypatch.setenv("CAPSYSRED_STAGE14_JOBS", "2")
    sim.replay([str(archive)], str(tmp_path / "parallel"), stages=[14])
    _assert_outputs_match(tmp_path / "serial", tmp_path / "parallel")


def _table_scene(tmp_path):
    """Scene with a three-line spectrum table (an external file reference)."""
    raw = _config()
    table = tmp_path / "spectrum.txt"
    table.write_text("8.0 0.6\n8.1 0.3\n7.9 0.1\n", encoding="utf-8")
    raw["spectrum"] = {"mode": "table", "file": str(table)}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    sim = Simulation.from_yaml(str(cfg_path))
    archive = tmp_path / "arch"
    _write_v3(archive, sim)
    return raw, cfg_path, archive, table


def test_stage14_parallel_matches_serial_table_spectrum(tmp_path, monkeypatch):
    """Multi-line physics resolved from a table file, through the workers."""
    raw, cfg_path, archive, _ = _table_scene(tmp_path)
    monkeypatch.delenv("CAPSYSRED_STAGE14_JOBS", raising=False)
    Simulation.from_yaml(str(cfg_path)).replay(
        [str(archive)], str(tmp_path / "serial"), stages=[14])
    shutil.rmtree(tmp_path / "stage14-cache")
    monkeypatch.setenv("CAPSYSRED_STAGE14_JOBS", "3")
    Simulation.from_dict(raw).replay(
        [str(archive)], str(tmp_path / "parallel"), stages=[14])
    _assert_outputs_match(tmp_path / "serial", tmp_path / "parallel")


def test_stage14_parallel_detects_spectrum_table_drift(tmp_path, monkeypatch):
    """A spectrum table rewritten under a running build must abort the
    workers, not deposit different physics under the parent analysis_id."""
    _, cfg_path, archive, table = _table_scene(tmp_path)
    sim = Simulation.from_yaml(str(cfg_path))
    table.write_text("8.0 0.5\n8.2 0.5\n", encoding="utf-8")
    monkeypatch.setenv("CAPSYSRED_STAGE14_JOBS", "2")
    with pytest.raises(ValueError, match="physics differs from the parent"):
        sim.replay([str(archive)], str(tmp_path / "out"), stages=[14])


def _boom(*_args, **_kwargs):
    raise RuntimeError("stop before assembly")


def _publish_ranges_then_abort(tmp_path, cfg_path, archive, out: str) -> None:
    """Run a parallel build that dies after the workers publish their ranges."""
    from formula.capsysred.stages import stage14
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(stage14, "_sum_range_aggregates", _boom)
        with pytest.raises(RuntimeError, match="stop before assembly"):
            Simulation.from_yaml(str(cfg_path)).replay(
                [str(archive)], str(tmp_path / out), stages=[14])


def test_stage14_checkpoint_corruption_detected(tmp_path, monkeypatch):
    """A size-preserving byte flip in a published range fails the resume
    instead of being laundered into the final sha."""
    cfg_path, archive = _scene(tmp_path)
    monkeypatch.setenv("CAPSYSRED_STAGE14_JOBS", "2")
    _publish_ranges_then_abort(tmp_path, cfg_path, archive, "p1")
    rows_files = sorted(tmp_path.rglob("rows-m*.f64"))
    assert rows_files
    data = bytearray(rows_files[0].read_bytes())
    data[len(data) // 2] ^= 0xFF
    rows_files[0].write_bytes(data)
    with pytest.raises(ValueError, match="range checkpoint corrupted"):
        Simulation.from_yaml(str(cfg_path)).replay(
            [str(archive)], str(tmp_path / "p2"), stages=[14])


def test_stage14_checkpoint_without_manifest_rebuilt(tmp_path, monkeypatch):
    cfg_path, archive = _scene(tmp_path)
    monkeypatch.delenv("CAPSYSRED_STAGE14_JOBS", raising=False)
    Simulation.from_yaml(str(cfg_path)).replay(
        [str(archive)], str(tmp_path / "serial"), stages=[14])
    shutil.rmtree(tmp_path / "stage14-cache")
    monkeypatch.setenv("CAPSYSRED_STAGE14_JOBS", "2")
    _publish_ranges_then_abort(tmp_path, cfg_path, archive, "p1")
    manifests = sorted(tmp_path.rglob("done-m*.json"))
    assert manifests
    manifests[0].unlink()
    Simulation.from_yaml(str(cfg_path)).replay(
        [str(archive)], str(tmp_path / "p2"), stages=[14])
    _assert_outputs_match(tmp_path / "serial", tmp_path / "p2")


def test_stage14_finalize_crash_resumes(tmp_path, monkeypatch):
    """A crash between the aggregates/meta writes and publication resumes in
    place: no FileExistsError wedge, ranges are still there and reused."""
    from formula.capsysred.stages import stage14
    cfg_path, archive = _scene(tmp_path)
    monkeypatch.delenv("CAPSYSRED_STAGE14_JOBS", raising=False)
    Simulation.from_yaml(str(cfg_path)).replay(
        [str(archive)], str(tmp_path / "serial"), stages=[14])
    shutil.rmtree(tmp_path / "stage14-cache")
    monkeypatch.setenv("CAPSYSRED_STAGE14_JOBS", "2")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(stage14, "_publish_directory", _boom)
        with pytest.raises(RuntimeError, match="stop before assembly"):
            Simulation.from_yaml(str(cfg_path)).replay(
                [str(archive)], str(tmp_path / "p1"), stages=[14])
    # canonical files are already in the partial, checkpoints survived
    assert list(tmp_path.rglob("aggregates.bin"))
    assert list(tmp_path.rglob("rows-m*.f64"))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(stage14, "_range_worker", _boom)   # ranges must be reused
        Simulation.from_yaml(str(cfg_path)).replay(
            [str(archive)], str(tmp_path / "p2"), stages=[14])
    _assert_outputs_match(tmp_path / "serial", tmp_path / "p2")
    assert not list(tmp_path.rglob("*.partial"))


def test_stage14_jobs_change_resumes(tmp_path, monkeypatch):
    """jobs=2 checkpoints resumed under jobs=3: the recorded partition wins,
    every range is reused, publication leaves no orphan partial."""
    from formula.capsysred.stages import stage14
    cfg_path, archive = _scene(tmp_path)
    monkeypatch.delenv("CAPSYSRED_STAGE14_JOBS", raising=False)
    Simulation.from_yaml(str(cfg_path)).replay(
        [str(archive)], str(tmp_path / "serial"), stages=[14])
    shutil.rmtree(tmp_path / "stage14-cache")
    monkeypatch.setenv("CAPSYSRED_STAGE14_JOBS", "2")
    _publish_ranges_then_abort(tmp_path, cfg_path, archive, "p1")
    monkeypatch.setenv("CAPSYSRED_STAGE14_JOBS", "3")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(stage14, "_range_worker", _boom)   # ranges must be reused
        Simulation.from_yaml(str(cfg_path)).replay(
            [str(archive)], str(tmp_path / "p2"), stages=[14])
    _assert_outputs_match(tmp_path / "serial", tmp_path / "p2")
    assert not list(tmp_path.rglob("*.partial"))


def test_stage14_stale_partial_removed_on_cache_hit(tmp_path, monkeypatch):
    cfg_path, archive = _scene(tmp_path)
    monkeypatch.setenv("CAPSYSRED_STAGE14_JOBS", "2")
    Simulation.from_yaml(str(cfg_path)).replay(
        [str(archive)], str(tmp_path / "p1"), stages=[14])
    caches = sorted((tmp_path / "stage14-cache").glob("*/meta.json"))
    assert caches
    stale = caches[0].parent.with_name(caches[0].parent.name + ".partial")
    stale.mkdir()
    (stale / "rows-m000000-000002.f64").write_bytes(b"junk")
    Simulation.from_yaml(str(cfg_path)).replay(
        [str(archive)], str(tmp_path / "p2"), stages=[14])
    assert not stale.exists()
    _assert_outputs_match(tmp_path / "p1", tmp_path / "p2")


def _cache_metas(tmp_path) -> dict[str, dict]:
    """Cache meta per analysis_id (one independently fingerprinted per screen)."""
    metas = sorted((tmp_path / "stage14-cache").glob("*/meta.json"))
    assert metas
    return {path.parent.name: json.loads(path.read_text(encoding="utf-8"))
            for path in metas}


def test_stage14_parallel_rows_identical_and_cache_hit(tmp_path, monkeypatch):
    """The doc's byte-identity claim, directly: the parallel rows file hashes
    the same as the serial one; afterwards a replay is a pure cache hit."""
    from formula.capsysred.stages import stage14
    cfg_path, archive = _scene(tmp_path)
    monkeypatch.delenv("CAPSYSRED_STAGE14_JOBS", raising=False)
    Simulation.from_yaml(str(cfg_path)).replay(
        [str(archive)], str(tmp_path / "serial"), stages=[14])
    serial = _cache_metas(tmp_path)
    shutil.rmtree(tmp_path / "stage14-cache")
    monkeypatch.setenv("CAPSYSRED_STAGE14_JOBS", "3")
    Simulation.from_dict(_config()).replay(
        [str(archive)], str(tmp_path / "parallel"), stages=[14])
    parallel = _cache_metas(tmp_path)
    assert set(parallel) == set(serial)          # same analysis_ids per screen
    for analysis_id, meta in parallel.items():
        expected = serial[analysis_id]["files"]
        assert meta["files"]["mode-rows.f64"] == expected["mode-rows.f64"]
        # summed aggregates differ in last-ulp order only, never in size
        assert (meta["files"]["aggregates.bin"]["bytes"]
                == expected["aggregates.bin"]["bytes"])
    # a second replay must not open any builder: pure cache hit
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(stage14, "_build_stream_fanout", _boom)
        mp.setattr(stage14, "_build_fanout_parallel", _boom)
        Simulation.from_dict(_config()).replay(
            [str(archive)], str(tmp_path / "hit"), stages=[14])
    _assert_outputs_match(tmp_path / "parallel", tmp_path / "hit")


def test_stage14_agg_checkpoint_corruption_detected(tmp_path, monkeypatch):
    cfg_path, archive = _scene(tmp_path)
    monkeypatch.setenv("CAPSYSRED_STAGE14_JOBS", "2")
    _publish_ranges_then_abort(tmp_path, cfg_path, archive, "p1")
    agg_files = sorted(tmp_path.rglob("agg-m*.bin"))
    assert agg_files
    data = bytearray(agg_files[0].read_bytes())
    data[-1] ^= 0xFF
    agg_files[0].write_bytes(data)
    with pytest.raises(ValueError, match="aggregates differ from the manifest"):
        Simulation.from_yaml(str(cfg_path)).replay(
            [str(archive)], str(tmp_path / "p2"), stages=[14])


def test_stage14_manifest_size_lie_rebuilds(tmp_path, monkeypatch):
    """A manifest whose recorded sizes contradict the files marks the range
    as not done; the rerun rebuilds it and matches the serial result."""
    cfg_path, archive = _scene(tmp_path)
    monkeypatch.delenv("CAPSYSRED_STAGE14_JOBS", raising=False)
    Simulation.from_yaml(str(cfg_path)).replay(
        [str(archive)], str(tmp_path / "serial"), stages=[14])
    shutil.rmtree(tmp_path / "stage14-cache")
    monkeypatch.setenv("CAPSYSRED_STAGE14_JOBS", "2")
    _publish_ranges_then_abort(tmp_path, cfg_path, archive, "p1")
    manifests = sorted(tmp_path.rglob("done-m*.json"))
    assert manifests
    lying = json.loads(manifests[0].read_text(encoding="utf-8"))
    lying["rows"]["bytes"] += 24
    manifests[0].write_text(json.dumps(lying), encoding="utf-8")
    Simulation.from_yaml(str(cfg_path)).replay(
        [str(archive)], str(tmp_path / "p2"), stages=[14])
    _assert_outputs_match(tmp_path / "serial", tmp_path / "p2")


@pytest.mark.parametrize("n_modes,jobs", [(4, 1), (4, 3), (4, 4), (4, 9),
                                          (5, 2), (7, 7), (1000, 48)])
def test_mode_ranges_partition(n_modes, jobs):
    from formula.capsysred.stages.stage14 import _mode_ranges
    ranges = _mode_ranges(n_modes, jobs)
    assert len(ranges) == min(jobs, n_modes)
    assert ranges[0][0] == 0 and ranges[-1][1] == n_modes
    assert all(m1 > m0 for m0, m1 in ranges)
    assert all(ranges[i][1] == ranges[i + 1][0]
               for i in range(len(ranges) - 1))
    sizes = [m1 - m0 for m0, m1 in ranges]
    assert max(sizes) - min(sizes) <= 1


def test_sum_range_aggregates_manual():
    from array import array
    from formula.capsysred.stages.stage14 import (STREAM_COUNTERS,
                                                  _sum_range_aggregates)

    def item(scale):
        return {
            "I": array("d", [1.5 * scale, 2.0 * scale]),
            "w_re": array("d", [0.25 * scale, -0.5 * scale]),
            "w_im": array("d", [-1.0 * scale, 0.125 * scale]),
            "ic": array("d", [3.0 * scale, 0.0]),
            "n_rays": array("Q", [10 * scale, 0]),
            "m_realizations": array("I", [2 * scale, 1]),
            "m_pair_realizations": array("I", [scale, 0]),
            "m_ref_realizations": array("I", [scale, scale]),
            "max_rays_per_realization": array("I", [5 * scale, 7]),
            "stats": {name: scale for name in STREAM_COUNTERS},
            "bounce_hist": [scale, 0, 3 * scale],
            "scatter": [scale, 2 * scale],
        }

    total = _sum_range_aggregates([item(1), item(4)], npix=2)
    assert list(total["I"]) == [7.5, 10.0]
    assert list(total["w_re"]) == [1.25, -2.5]
    assert list(total["w_im"]) == [-5.0, 0.625]
    assert list(total["ic"]) == [15.0, 0.0]
    assert list(total["n_rays"]) == [50, 0]
    assert list(total["m_realizations"]) == [10, 2]
    assert list(total["m_pair_realizations"]) == [5, 0]
    assert list(total["m_ref_realizations"]) == [5, 5]
    assert list(total["max_rays_per_realization"]) == [20, 7]  # max, not sum
    assert total["stats"] == {name: 5 for name in STREAM_COUNTERS}
    assert total["bounce_hist"] == [5, 0, 15]
    assert total["scatter"] == [5, 10]
    overflowing = item(1)
    overflowing["m_realizations"] = array("I", [0xFFFFFFFF, 0])
    with pytest.raises(ValueError, match="overflows uint32"):
        _sum_range_aggregates([overflowing, item(1)], npix=2)


def test_stage14_worker_pins_prepared_input(tmp_path):
    from formula.capsysred.stages import stage14
    _, archive = _scene(tmp_path)
    digest = rays_v3.index_digest(str(archive))
    physics = stage14._physics_contract(Simulation.from_dict(_config()))
    with pytest.raises(ValueError, match="physics differs from the parent"):
        stage14._range_worker(
            _config(), str(archive),
            {"index_sha256": digest, "n_modes": 4, "n_rays": 20,
             "source_z": -0.01,
             "physics": {**physics, "amplitude_min": 123.0}}, 0, 1, [])
    with pytest.raises(ValueError, match="changed since input preparation"):
        stage14._range_worker(
            _config(), str(archive),
            {"index_sha256": "0" * 64, "n_modes": 4, "n_rays": 20,
             "source_z": -0.01, "physics": physics}, 0, 1, [])
    with pytest.raises(ValueError, match="differs from the prepared input"):
        stage14._range_worker(
            _config(), str(archive),
            {"index_sha256": digest, "n_modes": 999, "n_rays": 20,
             "source_z": -0.01, "physics": physics}, 0, 1, [])


def test_stage14_jobs_env_validation(monkeypatch):
    from formula.capsysred.env import Env
    monkeypatch.delenv("CAPSYSRED_STAGE14_JOBS", raising=False)
    assert Env.stage14_jobs() == 1
    monkeypatch.setenv("CAPSYSRED_STAGE14_JOBS", " ")
    assert Env.stage14_jobs() == 1
    monkeypatch.setenv("CAPSYSRED_STAGE14_JOBS", "48")
    assert Env.stage14_jobs() == 48
    for bad in ("0", "-2", "three", "1.5"):
        monkeypatch.setenv("CAPSYSRED_STAGE14_JOBS", bad)
        with pytest.raises(ValueError, match="CAPSYSRED_STAGE14_JOBS"):
            Env.stage14_jobs()


def test_stage14_jobs_reject_v2_archive():
    from formula.capsysred.stages import stage14
    part = stage14.InputPart("r.jsonl.gz", {}, 4, 20, 0.0, "", None, "", "",
                             {}, 0, None)
    with pytest.raises(ValueError, match="needs a v3 rays archive"):
        stage14._build_fanout_parallel(None, part, [], 2, print)

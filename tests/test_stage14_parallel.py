"""Stage-14 parallel range builds (CAPSYSRED_STAGE14_JOBS): serial equivalence
and input pinning (workers take the config by value and the parent-fixed
archive identity).  Resume-after-kill is covered by the atomic per-range
files."""

from __future__ import annotations

import copy
import json
import math
import os
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
        "rng": {"scheme": "lattice-v1"},
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
        mp.setattr(stage14, "_fsum_double_fields", _boom)
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
        mp.setattr(stage14, "_publish_cache_tree", _boom)
        with pytest.raises(RuntimeError, match="stop before assembly"):
            Simulation.from_yaml(str(cfg_path)).replay(
                [str(archive)], str(tmp_path / "p1"), stages=[14])
    # canonical files already sit in ready.tmp, checkpoints survived
    assert list(tmp_path.rglob("ready.tmp/aggregates.bin"))
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


def test_stage14_metaless_final_fail_closed(tmp_path, monkeypatch):
    """A meta-less cache directory is never ours (the cache is published as
    one directory rename): fail closed, never auto-delete — with or without
    a .partial beside it."""
    cfg_path, archive = _scene(tmp_path)
    monkeypatch.setenv("CAPSYSRED_STAGE14_JOBS", "2")
    Simulation.from_yaml(str(cfg_path)).replay(
        [str(archive)], str(tmp_path / "p1"), stages=[14])
    metas = sorted((tmp_path / "stage14-cache").glob("*/meta.json"))
    assert metas
    metas[0].unlink()
    valuable = metas[0].parent / "mode-rows.f64"
    assert valuable.exists()
    with pytest.raises(ValueError, match="remove it manually"):
        Simulation.from_yaml(str(cfg_path)).replay(
            [str(archive)], str(tmp_path / "p2"), stages=[14])
    assert valuable.exists()                     # nothing was deleted
    stale = metas[0].parent.with_name(metas[0].parent.name + ".partial")
    stale.mkdir()
    (stale / "done-m000000-000001.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="remove it manually"):
        Simulation.from_yaml(str(cfg_path)).replay(
            [str(archive)], str(tmp_path / "p3"), stages=[14])
    assert valuable.exists()                     # a partial is no license


def test_stage14_worker_rejects_mutated_screen(tmp_path, monkeypatch):
    """A typed screen config mutated after construction must fail loudly:
    the parent meta would describe a screen the workers did not deposit."""
    cfg_path, archive = _scene(tmp_path)
    sim = Simulation.from_yaml(str(cfg_path))
    sim.cfg.capillary.screen.reference = (-1e-6, 0.0)
    monkeypatch.setenv("CAPSYSRED_STAGE14_JOBS", "2")
    with pytest.raises(ValueError, match="screen contract differs"):
        sim.replay([str(archive)], str(tmp_path / "out"), stages=[14])


def test_stage14_stale_partial_removed_on_cache_hit(tmp_path, monkeypatch):
    """A validated final wins over an orphan partial at any jobs value."""
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
    # the same orphan must not block the strict serial pass either
    monkeypatch.delenv("CAPSYSRED_STAGE14_JOBS", raising=False)
    stale.mkdir()
    (stale / "rows-m000000-000002.f64").write_bytes(b"junk")
    Simulation.from_yaml(str(cfg_path)).replay(
        [str(archive)], str(tmp_path / "p3"), stages=[14])
    assert not stale.exists()
    _assert_outputs_match(tmp_path / "p1", tmp_path / "p3")


def test_stage14_parallel_matches_serial_many_ranges(tmp_path, monkeypatch):
    """n_modes above the fold width exercises the bounded-memory reducer."""
    raw = _config()
    raw["capillary"]["source"]["n_modes"] = 12
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    sim = Simulation.from_yaml(str(cfg_path))
    archive = tmp_path / "arch"
    _write_v3(archive, sim)
    monkeypatch.delenv("CAPSYSRED_STAGE14_JOBS", raising=False)
    Simulation.from_yaml(str(cfg_path)).replay(
        [str(archive)], str(tmp_path / "serial"), stages=[14])
    shutil.rmtree(tmp_path / "stage14-cache")
    monkeypatch.setenv("CAPSYSRED_STAGE14_JOBS", "3")
    Simulation.from_dict(raw).replay(
        [str(archive)], str(tmp_path / "parallel"), stages=[14])
    _assert_outputs_match(tmp_path / "serial", tmp_path / "parallel")


def test_effective_workers_caps():
    from formula.capsysred.stages.stage14 import (_MAX_NT_WORKERS,
                                                  _effective_workers)
    assert _effective_workers(2, 128) == 2
    assert _effective_workers(50, 3) == 3
    assert _effective_workers(1, 0) == 1
    big = _effective_workers(128, 128)
    assert big == (_MAX_NT_WORKERS if os.name == "nt" else 128)


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


@pytest.mark.parametrize("n_modes", [1, 4, 5, 127, 128, 129, 1000, 2000])
def test_mode_ranges_partition(n_modes):
    from formula.capsysred.stages._stage14_checkpoints import (RANGE_SLOTS,
                                                               mode_ranges)
    ranges = mode_ranges(n_modes)
    assert len(ranges) == min(RANGE_SLOTS, n_modes)
    assert ranges[0][0] == 0 and ranges[-1][1] == n_modes
    assert all(m1 > m0 for m0, m1 in ranges)
    assert all(ranges[i][1] == ranges[i + 1][0]
               for i in range(len(ranges) - 1))
    sizes = [m1 - m0 for m0, m1 in ranges]
    assert max(sizes) - min(sizes) <= 1


def test_fold_int_aggregates_manual():
    from array import array
    from formula.capsysred.stages.stage14 import (STREAM_COUNTERS,
                                                  _fold_int_aggregates,
                                                  _seal_int_aggregates)

    def item(scale):
        return {
            "n_rays": array("Q", [10 * scale, 0]),
            "m_realizations": array("I", [2 * scale, 1]),
            "m_pair_realizations": array("I", [scale, 0]),
            "m_ref_realizations": array("I", [scale, scale]),
            "max_rays_per_realization": array("I", [5 * scale, 7]),
            "stats": {name: scale for name in STREAM_COUNTERS},
            "bounce_hist": [scale, 0, 3 * scale],
            "scatter": [scale, 2 * scale],
        }

    total = item(1)
    _fold_int_aggregates(total, item(4))
    _seal_int_aggregates(total)
    assert list(total["n_rays"]) == [50, 0]
    assert total["n_rays"].typecode == "Q"
    assert list(total["m_realizations"]) == [10, 2]
    assert total["m_realizations"].typecode == "I"
    assert list(total["m_pair_realizations"]) == [5, 0]
    assert list(total["m_ref_realizations"]) == [5, 5]
    assert list(total["max_rays_per_realization"]) == [20, 7]  # max, not sum
    assert total["stats"] == {name: 5 for name in STREAM_COUNTERS}
    assert total["bounce_hist"] == [5, 0, 15]
    assert total["scatter"] == [5, 10]
    total = item(1)
    total["m_realizations"] = array("I", [0xFFFFFFFF, 0])
    _fold_int_aggregates(total, item(1))
    with pytest.raises(ValueError, match="overflows uint32"):
        _seal_int_aggregates(total)


@pytest.mark.xfail(strict=True, reason=(
    "P1: the native store accumulates += per ray/mode, so range subtotals "
    "are already rounded before the union fsum; serial and parallel "
    "aggregates differ in last ulps (documented ordering class; the fix "
    "would be a compensated accumulator inside the C++ Stage14Store)"))
def test_stage14_union_matches_monolithic_store_bitwise(tmp_path):
    from formula.capsysred.stages.stage14 import _native_store, _typed_array
    contributions = [1.0, 1e-16, 1e-16, -1.0]

    def w_re(path, values):
        # ref=1 carries a unit ray, so pixel 0's W row is exactly `value`
        # (a ray at the ref pixel alone contributes zero: self-pairs are
        # excluded by the estimator).
        store = _native_store(str(path), len(values), 2, 1, [1.0], [1.0])
        for mode, value in enumerate(values):
            store.begin_mode(mode)
            store.add_ray(0, 0.0, [complex(value, 0.0)])
            store.add_ray(1, 0.0, [complex(1.0, 0.0)])
            store.fold_mode()
        return _typed_array("d", store.finish()["w_re"])[0]

    mono = w_re(tmp_path / "mono.f64", contributions)
    left = w_re(tmp_path / "left.f64", contributions[:2])
    right = w_re(tmp_path / "right.f64", contributions[2:])
    assert mono == 0.0                            # each +1e-16 is absorbed
    assert math.fsum([left, right]) == mono


def test_fsum_double_fields_adversarial(tmp_path):
    """Windowed folding is forbidden: cancellation across ranges must give
    the exact global fsum, with 1e16/1e300 outliers and 1-pixel tiles."""
    import struct
    from formula.capsysred.stages.stage14 import _fsum_double_fields
    spread = [1e16] + [1.0] * 126 + [-1e16]      # exact global fsum = 126
    huge = [1e300] + [1.0] * 126 + [-1e300]
    npix = 2
    paths = []
    for k, (a, b) in enumerate(zip(spread, huge)):
        path = tmp_path / f"agg-{k:03d}.bin"
        path.write_bytes(bytes(16) + struct.pack(
            "<8d",
            a, b,        # I
            b, a,        # w_re
            1.0, -1.0,   # w_im
            0.0, 0.0))   # ic
        paths.append(str(path))
    for budget in (8 * len(paths), 1 << 20):     # tile = 1 px, then one shot
        out = _fsum_double_fields(paths, npix, tile_budget=budget)
        assert list(out["I"]) == [126.0, 126.0]
        assert list(out["w_re"]) == [126.0, 126.0]
        assert list(out["w_im"]) == [128.0, -128.0]
        assert list(out["ic"]) == [0.0, 0.0]


@pytest.mark.xfail(strict=True, reason=(
    "P1: the native finalizer computes every LOO as rounded_total - row "
    "(bindings_stage14.cpp:1000-1002,1079-1081); with a dominating mode "
    "that is not the sum of the remaining mode rows — the fix is an "
    "error-free expansion of the totals with deletion from it"))
def test_stage14_loo_equals_sum_of_remaining_rows(tmp_path):
    from formula.capsysred.stages.stage14 import _native_store, _typed_array
    # W row of pixel 0 = ray amplitude at 0 times the unit ref ray; powers
    # of two keep every row exact, and 2.0 + 2**-53-scale rows are ties that
    # round to even (2.0), so the dominant-mode total provably absorbs all
    # small rows.
    path = tmp_path / "rows.f64"
    w_values = [2.0, 2.0 ** -52, 2.0 ** -52, 2.0 ** -52]
    store = _native_store(str(path), len(w_values), 2, 1, [1.0], [1.0])
    for mode, value in enumerate(w_values):
        store.begin_mode(mode)
        store.add_ray(0, 0.0, [complex(value, 0.0)])
        store.add_ray(1, 0.0, [complex(1.0, 0.0)])
        store.fold_mode()
    native = store.finish()
    total = _typed_array("d", native["w_re"])[0]
    data = path.read_bytes()
    rows = [_typed_array("d", data[mode * 48:mode * 48 + 8])[0]
            for mode in range(len(w_values))]
    assert rows == w_values                   # rows are stored exactly
    assert total == 2.0                       # every small row was absorbed
    native_loo = total - rows[0]              # the finalizer's delete-one
    assert native_loo == math.fsum(rows[1:])  # true remainder: 3 * 2**-52


@pytest.mark.xfail(strict=True, reason=(
    "P1: the main capillary screen is never validated against z1 (config "
    "checks only the extra screens) and the stage-14 re-projection happily "
    "steps backwards through the optic, ignoring reflections; the invariant "
    "source.z < z0 < z1 <= screen.z must hold for every screen"))
def test_stage14_rejects_main_screen_inside_optic(tmp_path, monkeypatch):
    raw = _config()
    raw["capillary"]["z0"] = 0.0
    raw["capillary"]["z1"] = 0.05
    raw["capillary"]["screen"]["z"] = 0.06
    raw["capillary"]["screens"][0]["z"] = 0.06
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    sim = Simulation.from_yaml(str(cfg_path))
    archive = tmp_path / "arch"
    _write_v3(archive, sim)
    monkeypatch.delenv("CAPSYSRED_STAGE14_JOBS", raising=False)
    inside = copy.deepcopy(raw)
    inside["capillary"]["screen"]["z"] = 0.025    # inside the optic (< z1)
    with pytest.raises(ValueError):
        Simulation.from_dict(inside).replay(
            [str(archive)], str(tmp_path / "out"), stages=[14])


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


def _set_fingerprint_rng(archive, rng) -> None:
    path = archive / "rays-fingerprint.yaml"
    meta = yaml.safe_load(path.read_text(encoding="utf-8"))
    if rng is None:
        meta.pop("rng", None)
    else:
        meta["rng"] = rng
    path.write_text(yaml.safe_dump(meta), encoding="utf-8")


def test_stage14_requires_known_rng_scheme(tmp_path):
    """Provenance-less archives must not feed the jackknife."""
    cfg_path, archive = _scene(tmp_path)
    _set_fingerprint_rng(archive, None)
    with pytest.raises(ValueError, match="known rng scheme"):
        Simulation.from_yaml(str(cfg_path)).replay(
            [str(archive)], str(tmp_path / "out"), stages=[14])


def _union_scene(tmp_path, seeds=(1402, 1403)):
    archives = []
    for k, seed in enumerate(seeds):
        raw = _config()
        raw["seed"] = seed
        sim = Simulation.from_dict(raw)
        archive = tmp_path / f"arch{k}"
        _write_v3(archive, sim)
        archives.append(str(archive))
    union = _config()
    union["capillary"]["source"]["n_modes"] = 4 * len(seeds)
    return union, archives


def test_stage14_union_lattice_parts(tmp_path, monkeypatch):
    """A fresh lattice-v1 union with distinct seeds is the supported case."""
    union, archives = _union_scene(tmp_path)
    monkeypatch.delenv("CAPSYSRED_STAGE14_JOBS", raising=False)
    Simulation.from_dict(union).replay(
        archives, str(tmp_path / "out"), stages=[14])
    assert _mu_rows(tmp_path / "out")


def test_stage14_union_requires_lattice_provenance(tmp_path):
    """Sequential shard seeds are base+k: distinct top-level seeds do not
    prove disjoint streams, so a non-lattice part must fail the union."""
    union, archives = _union_scene(tmp_path)
    _set_fingerprint_rng(tmp_path / "arch1",
                         {"scheme": "sequential-v2", "shards": [{"seed": 1403}]})
    with pytest.raises(ValueError, match="lattice-v1 provenance"):
        Simulation.from_dict(union).replay(
            archives, str(tmp_path / "out"), stages=[14])


def test_stage14_jobs_reject_v2_archive():
    from formula.capsysred.stages import stage14
    part = stage14.InputPart("r.jsonl.gz", {}, 4, 20, 0.0, "", None, "", "",
                             {}, 0, None)
    with pytest.raises(ValueError, match="needs a v3 rays archive"):
        stage14._build_fanout_parallel(None, part, [], 2, print)

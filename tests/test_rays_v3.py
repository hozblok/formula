"""v3 per-mode archives: top-up, parallel/serial parity, verification, rng skips."""

from __future__ import annotations

import gzip
import json
import os
import random
import shutil

import pytest
import yaml

from formula.capsysred import Simulation, rays_v3
from formula.capsysred.screen import ScreenGrid
from formula.capsysred.source import Source
from formula.capsysred.trace_v3 import AIM_DRAWS, SCENES, trace as trace_v3, verify


def _raw(n_modes=6, n_rays=60, seed=77, shape="disk"):
    return {
        "seed": seed,
        "capillary": {
            "source": {"shape": shape, "size": 2.0e-6, "position": [0.0, 0.0, -0.02],
                       "n_modes": n_modes, "n_rays": n_rays},
            "screen": {"nx": 9, "ny": 9, "edge_x": 1.6e-5, "edge_y": 1.6e-5},
            "screens": [{"nx": 5, "ny": 5, "edge_x": 1.0e-5, "edge_y": 1.0e-5}],
        },
        "trace": {"lean_rays": True},
    }


def _write_yaml(path, raw):
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(raw, fh, sort_keys=False)
    return str(path)


def _stage14(raw, archive, out):
    sim = Simulation.from_dict(raw)
    sim.replay(str(archive), str(out), stages=[14])
    return sim.results["stage14:capillary"]


def _mu_rows(out):
    with open(os.path.join(out, "stage14", "mu-jack.jsonl"), encoding="utf-8") as fh:
        return fh.read()


def _payload_hashes(result):
    return [cache.meta["files"]["mode-rows.f64"]["sha256"] for cache in result["cache_parts"]]


def test_origins_depend_on_seed_and_mode_only(tmp_path):
    """The origin of mode m is drawn before its rays: n_rays and jobs do not move it."""
    raw40 = _raw(n_modes=6, n_rays=40)
    long = str(tmp_path / "long")
    trace_v3(_write_yaml(tmp_path / "cfg-40.yaml", raw40), long, jobs=2,
             level=6, log=lambda m: None)
    short = str(tmp_path / "short")
    trace_v3(_write_yaml(tmp_path / "cfg-20.yaml", _raw(n_modes=6, n_rays=20)), short,
             jobs=1, level=6, log=lambda m: None)
    idx_long, idx_short = rays_v3.load_index(long), rays_v3.load_index(short)
    assert ([rays_v3.section_header(long, s[0])["origin"] for s in idx_long.modes("capillary")]
            == [rays_v3.section_header(short, s[0])["origin"]
                for s in idx_short.modes("capillary")])
    # lattice-v1: the shorter archive's rows are a prefix of the longer one's, mode by mode
    for m in range(6):
        rows_long = [l for e in idx_long.sections("capillary", m)
                     for l in rays_v3.iter_section_lines(long, e)]
        rows_short = [l for e in idx_short.sections("capillary", m)
                      for l in rays_v3.iter_section_lines(short, e)]
        assert rows_long[:20] == rows_short and len(rows_long) == 40


def test_topup_lattice_equals_single_piece(tmp_path):
    raw = _raw(n_modes=5, n_rays=40)
    archive = str(tmp_path / "head")
    trace_v3(_write_yaml(tmp_path / "cfg-40.yaml", raw), archive,
             jobs=1, level=6, log=lambda m: None)
    raw100 = json.loads(json.dumps(raw))
    raw100["capillary"]["source"]["n_rays"] = 100
    cfg100 = _write_yaml(tmp_path / "cfg-100.yaml", raw100)
    trace_v3(cfg100, archive, jobs=2, level=6, log=lambda m: None)
    full = str(tmp_path / "full")
    trace_v3(cfg100, full, jobs=1, level=6, log=lambda m: None)
    index = rays_v3.load_index(archive)
    full_index = rays_v3.load_index(full)
    assert (list(rays_v3.scene_lines(archive, index, "capillary"))
            == list(rays_v3.scene_lines(full, full_index, "capillary")))


def test_topup_chunk_invariance_and_merge_oracle(tmp_path):
    raw = _raw()
    one = str(tmp_path / "one")
    trace_v3(_write_yaml(tmp_path / "cfg.yaml", raw), one,
             jobs=1, level=6, log=lambda m: None)
    two = str(tmp_path / "two")
    shutil.copytree(one, two)

    def cfg_for(n):
        r = json.loads(json.dumps(raw))
        r["capillary"]["source"]["n_rays"] = n
        return r, _write_yaml(tmp_path / f"cfg-{n}.yaml", r)

    raw120, cfg120 = cfg_for(120)
    _, cfg90 = cfg_for(90)
    trace_v3(cfg120, one, jobs=1, level=6, log=lambda m: None)
    trace_v3(cfg90, two, jobs=2, level=6, log=lambda m: None)
    trace_v3(cfg120, two, jobs=2, level=6, log=lambda m: None)
    ia, ib = rays_v3.load_index(one), rays_v3.load_index(two)
    assert ia.budgets == ib.budgets == {"capillary": [6, 120]}
    for m in range(6):
        tail_a = b"".join(l for e in ia.sections("capillary", m)[1:]
                          for l in rays_v3.iter_section_lines(one, e))
        tail_b = b"".join(l for e in ib.sections("capillary", m)[1:]
                          for l in rays_v3.iter_section_lines(two, e))
        assert tail_a and tail_a == tail_b
        assert len(ib.sections("capillary", m)) == 3
    verify(two, jobs=1, log=lambda m: None)

    # Oracle: head + tail sections == one physically merged section per mode.
    merged = str(tmp_path / "merged")
    os.makedirs(os.path.join(merged, rays_v3.MODES_DIR))
    rays_v3.write_fingerprint(merged, rays_v3.read_fingerprint(one))
    entries = []
    for m in range(6):
        secs = ia.sections("capillary", m)
        origin = rays_v3.section_header(one, secs[0])["origin"]
        w = rays_v3.SectionWriter(merged, "capillary", m, 0, 120, origin)
        for e in secs:
            for line in rays_v3.iter_section_lines(one, e):
                w.write_line(line)
        entries.append(w.close())
    rays_v3.write_index(merged, entries)
    res_two = _stage14(raw120, two, tmp_path / "s14-two")
    res_merged = _stage14(raw120, merged, tmp_path / "s14-merged")
    assert _mu_rows(tmp_path / "s14-two") == _mu_rows(tmp_path / "s14-merged")
    assert _payload_hashes(res_two) == _payload_hashes(res_merged)
    assert res_two["stats"]["emitted"] == 720
    with pytest.raises(ValueError, match="replay budgets differ"):
        _stage14(raw, two, tmp_path / "s14-wrong-budget")
    with pytest.raises(ValueError, match="below the recorded"):
        trace_v3(cfg90, two, jobs=1, level=6, log=lambda m: None)


def test_verify_and_stage14_detect_corruption(tmp_path):
    raw = _raw(n_modes=3, n_rays=30)
    archive = str(tmp_path / "rays-modes")
    trace_v3(_write_yaml(tmp_path / "cfg.yaml", raw), archive,
             jobs=1, level=6, log=lambda m: None)
    entry = rays_v3.load_index(archive).sections("capillary", 1)[0]
    path = rays_v3.section_path(archive, entry)
    with gzip.open(path, "rb") as fh:
        lines = fh.read().split(b"\n")
    lines[2], lines[3] = lines[3], lines[2]      # swap two rows: same bytes, other order
    with open(path, "wb") as raw_fh:
        with gzip.GzipFile(fileobj=raw_fh, mode="wb", mtime=0) as gz:
            gz.write(b"\n".join(lines))
    with pytest.raises(ValueError, match="sha256|not mode"):
        verify(archive, jobs=1, log=lambda m: None)
    with pytest.raises(ValueError, match="sha256|non-canonical"):
        _stage14(raw, archive, tmp_path / "s14-bad")


def test_trace_v3_refuses_foreign_rng_scheme(tmp_path):
    raw = _raw(n_modes=2, n_rays=10)
    archive = str(tmp_path / "rays-modes")
    cfg = _write_yaml(tmp_path / "cfg.yaml", raw)
    trace_v3(cfg, archive, jobs=1, level=6, log=lambda m: None)
    fp_path = rays_v3.fingerprint_path(archive)
    with open(fp_path, encoding="utf-8") as fh:
        meta = yaml.safe_load(fh)
    meta["rng"] = {"scheme": "sequential-v2", "shards": [{"seed": 77}]}
    with open(fp_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(meta, fh)
    raw20 = json.loads(json.dumps(raw))
    raw20["capillary"]["source"]["n_rays"] = 20
    with pytest.raises(ValueError, match="unknown rng scheme"):
        trace_v3(_write_yaml(tmp_path / "cfg-20.yaml", raw20), archive,
                 jobs=1, level=6, log=lambda m: None)


@pytest.mark.parametrize("shape,draws", [("point", 0), ("grid", 1), ("gaussian", 2), ("disk", 2)])
def test_rng_skip_counts(shape, draws):
    raw = _raw(shape=shape)
    raw["capillary"]["source"].update(grid_n=3, grid_step=1.0e-6)
    sim = Simulation.from_dict(raw)
    cap = sim.cfg.capillary
    real, skip = random.Random(5), random.Random(5)
    source = Source(cap.source, real)
    aim = sim._aim_capillary(source, ScreenGrid(cap.screen), real)
    for _ in range(4):
        origin = source.mode_origin()
        for _ in range(7):
            aim(origin)
    for _ in range(4 * (draws + 7 * AIM_DRAWS)):
        skip.random()
    assert real.getstate() == skip.getstate()
    real, skip = random.Random(6), random.Random(6)
    free = sim._aim_free(None, ScreenGrid(cap.screen), real)
    for _ in range(5):
        free(origin)
    for _ in range(5 * SCENES["free"]["draws"]):
        skip.random()
    assert real.getstate() == skip.getstate()


def test_trace_v3_fresh_equals_parallel_and_two_steps(tmp_path):
    raw = _raw(n_modes=5, n_rays=40)
    cfg40 = _write_yaml(tmp_path / "cfg-40.yaml", raw)
    raw100 = json.loads(json.dumps(raw))
    raw100["capillary"]["source"]["n_rays"] = 100
    cfg100 = _write_yaml(tmp_path / "cfg-100.yaml", raw100)

    one = str(tmp_path / "one")
    trace_v3(cfg100, one, jobs=1, level=6, log=lambda m: None)
    parallel = str(tmp_path / "parallel")
    trace_v3(cfg100, parallel, jobs=2, level=6, log=lambda m: None)
    two = str(tmp_path / "two")
    trace_v3(cfg40, two, jobs=1, level=6, log=lambda m: None)
    trace_v3(cfg100, two, jobs=2, level=6, log=lambda m: None)

    one_index = rays_v3.load_index(one)
    expected = list(rays_v3.scene_lines(one, one_index, "capillary"))
    expected_origins = [rays_v3.section_header(one, s[0])["origin"]
                        for s in one_index.modes("capillary")]
    for archive in (parallel, two):
        index = rays_v3.load_index(archive)
        assert index.budgets == {"capillary": [5, 100]}
        got = list(rays_v3.scene_lines(archive, index, "capillary"))
        assert got == expected
        # The initial section in two used n_rays=40; origins must still agree.
        assert [rays_v3.section_header(archive, s[0])["origin"]
                for s in index.modes("capillary")] == expected_origins
    assert one_index.budgets == {"capillary": [5, 100]}
    assert len(rays_v3.load_index(two).sections("capillary", 0)) == 2
    fp = rays_v3.read_fingerprint(one)
    assert fp["rng"]["scheme"] == "lattice-v1" and fp["geometry"]["seed"] == 77
    verify(two, jobs=1, log=lambda m: None)
    result = _stage14(raw100, one, tmp_path / "s14")
    assert result["stats"]["emitted"] == 500 and result["cache_hits"] == 0
    before = open(rays_v3.index_path(one), "rb").read()
    trace_v3(cfg100, one, jobs=1, level=6, log=lambda m: None)   # no-op
    assert open(rays_v3.index_path(one), "rb").read() == before
    with pytest.raises(ValueError, match="below the recorded"):
        trace_v3(cfg40, one, jobs=1, level=6, log=lambda m: None)


def test_trace_v3_all_scenes_equals_parallel_and_completed_archive(tmp_path):
    raw = _raw(n_modes=4, n_rays=30)
    raw["free"] = {"source": {"shape": "disk", "size": 2.0e-6, "position": [0.0, 0.0, -0.02],
                              "n_modes": 3, "n_rays": 25},
                   "screen": {"nx": 7, "ny": 1, "edge_x": 1.4e-5, "edge_y": 2.0e-6}}
    raw["screen"] = {"nx": 7, "ny": 1, "edge_x": 1.4e-5, "edge_y": 2.0e-6}
    cfg = _write_yaml(tmp_path / "cfg.yaml", raw)
    archive = str(tmp_path / "v3")
    summary = trace_v3(cfg, archive, jobs=2, level=6, log=lambda m: None, scenes="all")
    assert summary["budgets"] == {"capillary": [4, 30], "free": [3, 25]}
    serial = str(tmp_path / "serial")
    trace_v3(cfg, serial, jobs=1, level=6, log=lambda m: None, scenes="all")
    index = rays_v3.load_index(archive)
    serial_index = rays_v3.load_index(serial)
    for scene in ("free", "capillary"):
        expected = list(rays_v3.scene_lines(serial, serial_index, scene))
        got = list(rays_v3.scene_lines(archive, index, scene))
        assert got == expected
    # a capillary-only archive completes itself on the next default run
    only_cap = str(tmp_path / "cap")
    trace_v3(cfg, only_cap, jobs=1, level=6, log=lambda m: None, scenes=("capillary",))
    assert rays_v3.load_index(only_cap).budgets == {"capillary": [4, 30]}
    trace_v3(cfg, only_cap, jobs=1, level=6, log=lambda m: None)
    completed_index = rays_v3.load_index(only_cap)
    assert completed_index.budgets == {"capillary": [4, 30], "free": [3, 25]}
    for scene in ("free", "capillary"):
        assert (list(rays_v3.scene_lines(only_cap, completed_index, scene))
                == list(rays_v3.scene_lines(serial, serial_index, scene)))
    a = Simulation.from_dict(raw)
    a.replay(only_cap, str(tmp_path / "s2-v3"), stages=[2])
    b = Simulation.from_dict(raw)
    b.replay(serial, str(tmp_path / "s2-serial"), stages=[2])
    assert a.results["free"]["stats"] == b.results["free"]["stats"]
    before = open(rays_v3.index_path(only_cap), "rb").read()
    trace_v3(cfg, only_cap, jobs=1, level=6, log=lambda m: None)   # both scenes: no-op
    assert open(rays_v3.index_path(only_cap), "rb").read() == before

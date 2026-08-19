"""v3 per-mode archives: conversion parity, top-up, verification, rng skips."""

from __future__ import annotations

import gzip
import json
import os
import random
import shutil

import pytest
import yaml

from formula.capsysred import Simulation, rays_v3
from formula.capsysred.convert_rays_v3 import CAPILLARY_AIM_DRAWS, convert, verify
from formula.capsysred.rays import RaysReader, SceneSeed, _SCENE_SEED_STRIDE
from formula.capsysred.screen import ScreenGrid
from formula.capsysred.source import Source
from formula.capsysred.topup_trace import topup
from formula.capsysred.trace_v3 import trace as trace_v3


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


def _v2_from_v3(raw, v2_dir):
    """A v2 recording of raw's scenes: trace_v3 into a scratch archive, then
    stream the rows into rays.jsonl.gz + sidecar (the legacy writer is gone)."""
    from formula.capsysred.rays import sidecar_metadata, write_metadata
    scratch = os.path.join(str(v2_dir), "scratch")
    cfg = os.path.join(str(v2_dir), "trace-config.yaml")
    os.makedirs(v2_dir, exist_ok=True)
    _write_yaml(cfg, raw)
    trace_v3(cfg, scratch, jobs=1, level=6, log=lambda m: None, scenes="all")
    index = rays_v3.load_index(scratch)
    path = os.path.join(str(v2_dir), "rays.jsonl.gz")
    with gzip.open(path, "wb") as fh:
        fh.write(b"{}\n")
        for scene in [sc for sc in ("free", "capillary") if sc in index.budgets]:
            n = 0
            for line in rays_v3.scene_lines(scratch, index, scene):
                fh.write(line)
                n += 1
            fh.write(json.dumps({"scene_end": scene, "rows": n}).encode() + b"\n")
    shutil.rmtree(scratch)
    write_metadata(path, sidecar_metadata(Simulation.from_dict(raw).cfg))
    return path


def _norm(rec):
    return rec._replace(point=None if rec.point is None else rec.point[:2])


def test_convert_parity_reader_and_stage14(tmp_path):
    raw = _raw()
    v2 = tmp_path / "v2"
    _v2_from_v3(raw, v2)
    archive = str(v2 / "rays-modes")
    summary = convert(str(v2), archive, jobs=2, level=6, shards=None, origins=True,
                      log=lambda m: None)
    assert summary["origin_check"] == {"modes": 6, "max_dxy_m": 0.0, "max_rel_dopl": 0.0}
    verify(archive, jobs=1, log=lambda m: None)

    a = RaysReader(str(v2 / "rays.jsonl.gz"))
    b = RaysReader(archive)
    assert a.meta["budgets"] == b.meta["budgets"] == {"capillary": [6, 60]}
    assert [_norm(r) for r in a.scene_records("capillary")] == \
        [_norm(r) for r in b.scene_records("capillary")]
    index = rays_v3.load_index(archive)
    for header in (rays_v3.section_header(archive, s[0]) for s in index.modes("capillary")):
        assert header["origin_check"] == "retrace-ray0" and len(header["origin"]) == 3

    old = _stage14(raw, v2 / "rays.jsonl.gz", tmp_path / "s14-v2")
    new = _stage14(raw, archive, tmp_path / "s14-v3")
    assert _mu_rows(tmp_path / "s14-v2") == _mu_rows(tmp_path / "s14-v3")
    assert _payload_hashes(old) == _payload_hashes(new)
    assert new["cache_hits"] == 0
    again = _stage14(raw, archive, tmp_path / "s14-v3-again")
    assert again["cache_hits"] == 1
    with pytest.raises(ValueError, match="already exists"):
        convert(str(v2), archive, 1, 6, None, True, log=lambda m: None)


def _v2_lines(path):
    with gzip.open(path, "rb") as fh:
        return fh.read().split(b"\n")[1:]        # drop the preamble


def test_lattice_convert_and_origins_independent_of_rays(tmp_path):
    raw = _raw(n_modes=6, n_rays=40)
    seq = tmp_path / "seq"
    _v2_from_v3(raw, seq)
    archive = str(seq / "rays-modes")
    summary = convert(str(seq), archive, 2, 6, None, True, log=lambda m: None)
    assert summary["origin_check"] == {"modes": 6, "max_dxy_m": 0.0, "max_rel_dopl": 0.0}
    assert rays_v3.read_fingerprint(archive)["rng"]["scheme"] == "lattice-v1"
    # trace_v3 with two processes writes the same rows as the sequential v2 trace
    direct = str(tmp_path / "direct")
    trace_v3(_write_yaml(tmp_path / "cfg.yaml", raw), direct, jobs=2,
             level=6, log=lambda m: None)
    idx_a, idx_b = rays_v3.load_index(archive), rays_v3.load_index(direct)
    assert (list(rays_v3.scene_lines(archive, idx_a, "capillary"))
            == list(rays_v3.scene_lines(direct, idx_b, "capillary")))
    # Origins are a property of (seed, mode) alone: n_rays does not move them.
    short = tmp_path / "short"
    _v2_from_v3(_raw(n_modes=6, n_rays=20), short)
    convert(str(short), str(short / "rays-modes"), 1, 6, None, True, log=lambda m: None)
    short_idx = rays_v3.load_index(str(short / "rays-modes"))
    assert ([rays_v3.section_header(archive, s[0])["origin"] for s in idx_a.modes("capillary")]
            == [rays_v3.section_header(str(short / "rays-modes"), s[0])["origin"]
                for s in short_idx.modes("capillary")])


def _legacy_trace(raw, out_dir):
    """A sequential-v2 recording: one rng stream per scene, as before lattice-v1."""
    from formula.capsysred.native import make_tracer
    from formula.capsysred.rays import (_SCENE_SEED_STRIDE, row_of, sidecar_metadata,
                                        write_metadata)
    from formula.capsysred.shared.types import ray_record
    from formula.capsysred.surfaces import CapillaryBundle
    sim = Simulation.from_dict(raw)
    cfg = sim.cfg
    cap = cfg.capillary
    os.makedirs(out_dir)
    path = os.path.join(out_dir, "rays.jsonl.gz")
    rng = random.Random(cfg.seed * _SCENE_SEED_STRIDE + SceneSeed.CAPILLARY)
    source = Source(cap.source, rng)
    screen = ScreenGrid(cap.screen)
    aim = sim._aim_capillary(source, screen, rng)
    optic = CapillaryBundle(cap.bores, cap.z0, cap.z1)
    tracer = make_tracer(optic)
    n = 0
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as fh:
        fh.write("{}\n")
        for mode in range(cap.source.n_modes):
            origin = source.mode_origin()
            for ray in range(cap.source.n_rays):
                tr = tracer(origin, aim(origin), optic, screen.z, cfg.max_bounces)
                rec = ray_record(tr, screen, mode, ray, tr.fate)
                fh.write(json.dumps(row_of("capillary", rec, cfg.lean_rays)) + "\n")
                n += 1
        fh.write(json.dumps({"scene_end": "capillary", "rows": n}) + "\n")
    meta = sidecar_metadata(cfg)
    del meta["rng"]
    write_metadata(path, meta)


def test_convert_legacy_sequential_origins(tmp_path):
    raw = _raw(n_modes=4, n_rays=30)
    run = tmp_path / "legacy"
    _legacy_trace(raw, str(run))
    archive = str(run / "rays-modes")
    summary = convert(str(run), archive, 1, 6, None, True, log=lambda m: None)
    assert summary["origin_check"] == {"modes": 4, "max_dxy_m": 0.0, "max_rel_dopl": 0.0}
    fp = rays_v3.read_fingerprint(archive)
    assert fp["rng"]["scheme"] == "sequential-v2"
    assert fp["rng"]["capillary"]["shards"] == [{"seed": 77, "modes": 4}]
    # A wrong layout is caught by the ray-0 re-trace before anything is published.
    with pytest.raises(ValueError, match="wrong seed/shard layout"):
        convert(str(run), str(run / "wrong"), 1, 6, 2, True, log=lambda m: None)
    assert not os.path.exists(rays_v3.index_path(run / "wrong"))


def test_topup_lattice_equals_single_piece(tmp_path):
    raw = _raw(n_modes=5, n_rays=40)
    head = tmp_path / "head"
    _v2_from_v3(raw, head)
    archive = str(head / "rays-modes")
    convert(str(head), archive, 1, 6, None, True, log=lambda m: None)
    raw100 = json.loads(json.dumps(raw))
    raw100["capillary"]["source"]["n_rays"] = 100
    topup(_write_yaml(tmp_path / "cfg-100.yaml", raw100), archive,
          jobs=1, level=6, log=lambda m: None)
    full = tmp_path / "full"
    _v2_from_v3(raw100, full)
    index = rays_v3.load_index(archive)
    expected = [l for l in _v2_lines(full / "rays.jsonl.gz") if l.startswith(b'{"stage"')]
    got = [l.rstrip(b"\n") for l in rays_v3.scene_lines(archive, index, "capillary")]
    assert got == expected


def test_topup_chunk_invariance_and_merge_oracle(tmp_path):
    raw = _raw()
    v2 = tmp_path / "v2"
    _v2_from_v3(raw, v2)
    one = str(v2 / "one")
    convert(str(v2), one, 1, 6, None, True, log=lambda m: None)
    two = str(tmp_path / "two")
    shutil.copytree(one, two)

    def cfg_for(n):
        r = json.loads(json.dumps(raw))
        r["capillary"]["source"]["n_rays"] = n
        return r, _write_yaml(tmp_path / f"cfg-{n}.yaml", r)

    raw120, cfg120 = cfg_for(120)
    _, cfg90 = cfg_for(90)
    topup(cfg120, one, jobs=1, level=6, log=lambda m: None)
    topup(cfg90, two, jobs=1, level=6, log=lambda m: None)
    topup(cfg120, two, jobs=1, level=6, log=lambda m: None)
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
        topup(cfg90, two, jobs=1, level=6, log=lambda m: None)


def test_verify_and_stage14_detect_corruption(tmp_path):
    raw = _raw(n_modes=3, n_rays=30)
    v2 = tmp_path / "v2"
    _v2_from_v3(raw, v2)
    archive = str(v2 / "rays-modes")
    convert(str(v2), archive, 1, 6, None, True, log=lambda m: None)
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
    for _ in range(4 * (draws + 7 * CAPILLARY_AIM_DRAWS)):
        skip.random()
    assert real.getstate() == skip.getstate()
    from formula.capsysred.trace_v3 import SCENES
    real, skip = random.Random(6), random.Random(6)
    free = sim._aim_free(None, ScreenGrid(cap.screen), real)
    for _ in range(5):
        free(origin)
    for _ in range(5 * SCENES["free"]["draws"]):
        skip.random()
    assert real.getstate() == skip.getstate()
    assert SceneSeed.CAPILLARY_TOPUP not in (SceneSeed.CAPILLARY, SceneSeed.FREE)
    assert _SCENE_SEED_STRIDE == 1000003


def test_trace_v3_fresh_equals_v2_and_two_steps(tmp_path):
    raw = _raw(n_modes=5, n_rays=40)
    cfg40 = _write_yaml(tmp_path / "cfg-40.yaml", raw)
    raw100 = json.loads(json.dumps(raw))
    raw100["capillary"]["source"]["n_rays"] = 100
    cfg100 = _write_yaml(tmp_path / "cfg-100.yaml", raw100)

    one = str(tmp_path / "one")
    trace_v3(cfg100, one, jobs=1, level=6, log=lambda m: None)
    two = str(tmp_path / "two")
    trace_v3(cfg40, two, jobs=1, level=6, log=lambda m: None)
    trace_v3(cfg100, two, jobs=1, level=6, log=lambda m: None)
    v2 = tmp_path / "v2"
    _v2_from_v3(raw100, v2)

    expected = [l for l in _v2_lines(v2 / "rays.jsonl.gz") if l.startswith(b'{"stage"')]
    for archive in (one, two):
        index = rays_v3.load_index(archive)
        assert index.budgets == {"capillary": [5, 100]}
        got = [l.rstrip(b"\n") for l in rays_v3.scene_lines(archive, index, "capillary")]
        assert got == expected
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


def test_trace_v3_all_scenes_equals_v2_trace(tmp_path):
    raw = _raw(n_modes=4, n_rays=30)
    raw["free"] = {"source": {"shape": "disk", "size": 2.0e-6, "position": [0.0, 0.0, -0.02],
                              "n_modes": 3, "n_rays": 25},
                   "screen": {"nx": 7, "ny": 1, "edge_x": 1.4e-5, "edge_y": 2.0e-6}}
    raw["screen"] = {"nx": 7, "ny": 1, "edge_x": 1.4e-5, "edge_y": 2.0e-6}
    cfg = _write_yaml(tmp_path / "cfg.yaml", raw)
    archive = str(tmp_path / "v3")
    summary = trace_v3(cfg, archive, jobs=2, level=6, log=lambda m: None, scenes="all")
    assert summary["budgets"] == {"capillary": [4, 30], "free": [3, 25]}
    v2 = tmp_path / "v2"
    _v2_from_v3(raw, v2)
    index = rays_v3.load_index(archive)
    for scene in ("free", "capillary"):
        expected = [l for l in _v2_lines(v2 / "rays.jsonl.gz")
                    if l.startswith(('{"stage": "%s"' % scene).encode())]
        got = [l.rstrip(b"\n") for l in rays_v3.scene_lines(archive, index, scene)]
        assert got == expected
    # a capillary-only archive completes itself on the next default run
    only_cap = str(tmp_path / "cap")
    trace_v3(cfg, only_cap, jobs=1, level=6, log=lambda m: None, scenes=("capillary",))
    assert rays_v3.load_index(only_cap).budgets == {"capillary": [4, 30]}
    trace_v3(cfg, only_cap, jobs=1, level=6, log=lambda m: None)
    assert rays_v3.load_index(only_cap).budgets == {"capillary": [4, 30], "free": [3, 25]}
    a = Simulation.from_dict(raw)
    a.replay(only_cap, str(tmp_path / "s2-v3"), stages=[2])
    b = Simulation.from_dict(raw)
    b.replay(str(v2 / "rays.jsonl.gz"), str(tmp_path / "s2-v2"), stages=[2])
    assert a.results["free"]["stats"] == b.results["free"]["stats"]
    before = open(rays_v3.index_path(only_cap), "rb").read()
    trace_v3(cfg, only_cap, jobs=1, level=6, log=lambda m: None)   # both scenes: no-op
    assert open(rays_v3.index_path(only_cap), "rb").read() == before


"""rays v3 archive primitives without tracing: writer, index, fingerprint,
section verification, rng streams, CLI argument guards."""

from __future__ import annotations

import gzip
import json
import os
import random

import pytest
import yaml

from formula.capsysred import rays_v3
from formula.capsysred.rays import (RNG_SCHEME, SceneSeed, geometry_core,
                                    metadata_equal, stream_rng)
from formula.capsysred.rays_v3 import Index, Section, SectionWriter, section_name

SHA = "0" * 64
FP = {"format": 3, "geometry": {"seed": 1, "screen": {"nx": 9}}, "rng": {"scheme": RNG_SCHEME}}


def _row(scene, mode, ray):
    return {"stage": scene, "mode": mode, "ray": ray, "fate": "lost",
            "pixel": None, "opl": 0.0, "sins": [0.0]}


def _write(archive, scene, mode, r0, r1, rows=None, origin=("0", "0", "-0.02")):
    w = SectionWriter(archive, scene, mode, r0, r1, list(origin), {"rng": {"x": 1}})
    for ray in (rows if rows is not None else range(r0, r1)):
        w.write_row(_row(scene, mode, ray))
    return w.close()


def _archive(tmp_path, n_modes=2, n_rays=3):
    archive = str(tmp_path / "arch")
    rays_v3.write_fingerprint(archive, FP)
    entries = [_write(archive, "capillary", m, 0, n_rays) for m in range(n_modes)]
    return archive, rays_v3.write_index(archive, entries)


def _rewrite(path, lines):
    with open(path, "wb") as raw:
        # match SectionWriter's gzip parameters so untouched bytes reproduce
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                           compresslevel=rays_v3.DEFAULT_LEVEL, mtime=0) as gz:
            gz.write(b"".join(lines))


def _lines(path):
    with gzip.open(path, "rb") as fh:
        return fh.read().splitlines(keepends=True)


# ------------------------------------------------------------------ writer

def test_section_writer_roundtrip(tmp_path):
    archive, index = _archive(tmp_path)
    assert index.budgets == {"capillary": [2, 3]}
    entry = index.sections("capillary", 1)[0]
    assert entry.file == section_name("capillary", 1, 0, 3)
    assert not os.path.exists(rays_v3.section_path(archive, entry) + ".tmp")
    header = rays_v3.section_header(archive, entry)
    assert header["origin"] == ["0", "0", "-0.02"] and header["rng"] == {"x": 1}
    lines = list(rays_v3.iter_section_lines(archive, entry))
    assert [json.loads(l)["ray"] for l in lines] == [0, 1, 2]
    assert rays_v3.verify_section(archive, entry) == header
    # Deterministic bytes: the same rows give the same sha256.
    again = str(tmp_path / "again")
    rays_v3.write_fingerprint(again, FP)
    assert _write(again, "capillary", 1, 0, 3).sha256 == entry.sha256
    assert rays_v3.metadata(archive)["budgets"] == {"capillary": [2, 3]}
    assert [r.ray for r in rays_v3.scene_records(archive, index, "capillary")] == [0, 1, 2] * 2


def test_section_writer_row_count_mismatch_aborts(tmp_path):
    archive = str(tmp_path / "a")
    w = SectionWriter(archive, "free", 0, 0, 3)
    w.write_row(_row("free", 0, 0))
    with pytest.raises(ValueError, match="wrote 1 rows, section promises 3"):
        w.close()
    assert os.listdir(os.path.join(archive, rays_v3.MODES_DIR)) == []
    with pytest.raises(ValueError, match="already closed"):
        w.close()


def test_section_writer_abort_and_no_clobber(tmp_path):
    archive = str(tmp_path / "a")
    w = SectionWriter(archive, "free", 0, 0, 2)
    w.write_row(_row("free", 0, 0))
    w.abort()
    assert os.listdir(os.path.join(archive, rays_v3.MODES_DIR)) == []
    _write(archive, "free", 0, 0, 2)
    with pytest.raises(ValueError, match="already exists"):
        SectionWriter(archive, "free", 0, 0, 2)
    for bad in ((0, 0, 0), (0, 3, 2), (-1, 0, 1), (0, -1, 1)):
        with pytest.raises(ValueError, match="non-empty and non-negative"):
            SectionWriter(archive, "free", *bad)


# ------------------------------------------------------------------- index

def _entry(scene, mode, r0, r1, rows=None, file=None):
    return Section(scene, mode, r0, r1, r1 - r0 if rows is None else rows, 10, SHA,
                   file or section_name(scene, mode, r0, r1))


def test_index_accepts_split_modes_and_sorts():
    entries = [_entry("free", 1, 0, 5), _entry("capillary", 0, 2, 5),
               _entry("capillary", 0, 0, 2), _entry("free", 0, 0, 5),
               _entry("capillary", 1, 0, 5)]
    index = Index(entries)
    assert index.budgets == {"capillary": [2, 5], "free": [2, 5]}
    assert [e.r0 for e in index.sections("capillary", 0)] == [0, 2]
    assert sorted(index.scenes()) == ["capillary", "free"]
    assert index.modes("missing") == []
    assert index.total_bytes() == 50


@pytest.mark.parametrize("entries,msg", [
    ([_entry("free", 0, 0, 5), _entry("free", 0, 0, 5)], "repeats section"),
    ([_entry("free", 0, 0, 5, file="free-x.jsonl.gz")], "non-canonical file name"),
    ([_entry("free", 0, 0, 5), _entry("free", 2, 0, 5)], "modes are not contiguous"),
    ([_entry("free", 1, 0, 5)], "modes are not contiguous"),
    ([_entry("free", 0, 1, 5)], "not contiguous from ray 0"),
    ([_entry("free", 0, 0, 2), _entry("free", 0, 3, 5)], "not contiguous from ray 0"),
    ([_entry("free", 0, 0, 5, rows=4)], "not contiguous from ray 0"),
    ([_entry("free", 0, 0, 5), _entry("free", 1, 0, 4)], "unequal ray counts"),
])
def test_index_rejects(entries, msg):
    with pytest.raises(ValueError, match=msg):
        Index(entries)


def test_load_and_write_index(tmp_path):
    archive = str(tmp_path / "a")
    os.makedirs(archive)
    with pytest.raises(ValueError, match="missing or unreadable"):
        rays_v3.load_index(archive)
    path = rays_v3.index_path(archive)
    open(path, "w").close()
    with pytest.raises(ValueError, match="index is empty"):
        rays_v3.load_index(archive)
    with open(path, "w") as fh:
        fh.write("{not json\n")
    with pytest.raises(ValueError, match=":1: invalid JSON"):
        rays_v3.load_index(archive)
    bad = _entry("free", 0, 0, 5).as_dict()
    bad["sha256"] = "abc"
    with open(path, "w") as fh:
        fh.write(json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match="invalid rays index entry"):
        rays_v3.load_index(archive)
    with open(path, "w") as fh:
        fh.write(json.dumps(dict(_entry("free", 0, 0, 5).as_dict(), mode="x")) + "\n")
    with pytest.raises(ValueError, match="invalid rays index entry"):
        rays_v3.load_index(archive)

    entries = [_entry("free", 1, 0, 5), _entry("free", 0, 0, 5)]
    index = rays_v3.write_index(archive, entries)
    assert not os.path.exists(path + ".tmp")
    assert [e.mode for e in index.entries] == [0, 1]
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    assert [r["mode"] for r in rows] == [0, 1]
    assert rays_v3.load_index(archive).entries == index.entries
    digest = rays_v3.index_digest(archive)
    with pytest.raises(ValueError, match="unequal ray counts"):
        rays_v3.write_index(archive, entries + [_entry("free", 2, 0, 4)])
    assert rays_v3.index_digest(archive) == digest       # failed write leaves it intact


# ------------------------------------------------------------- fingerprint

def test_fingerprint_no_clobber_and_validation(tmp_path):
    archive = str(tmp_path / "a")
    path = rays_v3.write_fingerprint(archive, FP)
    assert rays_v3.write_fingerprint(archive, json.loads(json.dumps(FP))) == path
    other = dict(FP, geometry={"seed": 2})
    with pytest.raises(ValueError, match="exists and differs"):
        rays_v3.write_fingerprint(archive, other)
    assert rays_v3.read_fingerprint(archive) == FP
    with pytest.raises(ValueError, match="must carry format 3"):
        rays_v3.write_fingerprint(str(tmp_path / "b"), dict(FP, format=2))

    def written(meta):
        p = str(tmp_path / "c")
        os.makedirs(p, exist_ok=True)
        with open(rays_v3.fingerprint_path(p), "w", encoding="utf-8") as fh:
            yaml.safe_dump(meta, fh)
        return p

    for meta, msg in ((dict(FP, format=2), "format must be 3"),
                      (dict(FP, geometry=[1]), "must be a mapping"),
                      (dict(FP, lean=False), "lean must be true"),
                      (dict(FP, budgets={"free": [1, 1]}), "budgets live in the index"),
                      ([1, 2], "format must be 3")):
        with pytest.raises(ValueError, match=msg):
            rays_v3.read_fingerprint(written(meta))
    with open(rays_v3.fingerprint_path(written(FP)), "w") as fh:
        fh.write("a: [unclosed\n")
    with pytest.raises(ValueError, match="unreadable"):
        rays_v3.read_fingerprint(str(tmp_path / "c"))
    with pytest.raises(ValueError, match="unreadable"):
        rays_v3.read_fingerprint(str(tmp_path / "nowhere"))


# ------------------------------------------------------------ verification

def test_section_read_resumes_after_transient_io_error(tmp_path, monkeypatch):
    archive, index = _archive(tmp_path, n_modes=1, n_rays=4)
    entry = index.sections("capillary", 0)[0]
    good = list(rays_v3.iter_section_lines(archive, entry))
    monkeypatch.setattr(rays_v3, "READ_BACKOFF_S", 0.0)
    failures = {"left": 0, "seen": 0}
    real_open = open

    class Flaky:
        def __init__(self, fh):
            self.fh = fh

        def readinto(self, b):
            if failures["left"]:
                failures["left"] -= 1
                failures["seen"] += 1
                raise OSError(22, "Invalid argument")
            return self.fh.readinto(b)

        def __getattr__(self, name):
            return getattr(self.fh, name)

    def flaky_open(path, mode="r", *args, **kwargs):
        fh = real_open(path, mode, *args, **kwargs)
        return Flaky(fh) if str(path).endswith(".jsonl.gz") else fh

    monkeypatch.setattr(rays_v3, "open", flaky_open, raising=False)
    failures["left"] = 2
    assert list(rays_v3.iter_section_lines(archive, entry)) == good
    assert failures["seen"] == 2
    failures["left"] = rays_v3.READ_ATTEMPTS
    with pytest.raises(ValueError, match="truncated or corrupt"):
        list(rays_v3.iter_section_lines(archive, entry))


def test_section_corruption_is_detected(tmp_path):
    archive, index = _archive(tmp_path, n_modes=1, n_rays=4)
    entry = index.sections("capillary", 0)[0]
    path = rays_v3.section_path(archive, entry)
    good = _lines(path)

    def expect(msg, lines=None, raw=None):
        if raw is not None:
            with open(path, "wb") as fh:
                fh.write(raw)
        else:
            _rewrite(path, lines)
        with pytest.raises(ValueError, match=msg):
            list(rays_v3.iter_section_lines(archive, entry))

    with open(path, "rb") as fh:
        blob = fh.read()
    expect("truncated or corrupt", raw=blob[: len(blob) // 2])
    expect("truncated or corrupt", raw=b"not gzip at all")
    expect("sha256 differ", raw=blob + b"\x00")
    expect("missing section trailer", good[:-1])
    expect("data after the section trailer", good + [good[1]])
    expect("trailer contradicts", good[:2] + good[3:])          # one row dropped
    expect("unterminated line", good[:-1] + [good[-1].rstrip(b"\n")])
    expect("contradicts the index", [good[0].replace(b'"mode": 0', b'"mode": 1')] + good[1:])
    expect("invalid section header", [b"garbage\n"] + good[1:])
    # Rows present but re-ordered: bytes change, so the index hash catches it.
    swapped = [good[0], good[2], good[1]] + good[3:]
    expect("sha256 differ", swapped)
    # The same rows under the right hash still fail the canonical-id check.
    entry2 = Section(*entry[:4], entry.rows, *rays_v3._hash_file(path), entry.file)
    with pytest.raises(ValueError, match="row 0 is not mode 0 ray 0"):
        rays_v3.verify_section(archive, entry2)
    wrong_scene = [good[0]] + [l.replace(b'"stage": "capillary"', b'"stage": "free"')
                               for l in good[1:-1]] + [good[-1]]
    _rewrite(path, wrong_scene)
    entry3 = Section(*entry[:4], entry.rows, *rays_v3._hash_file(path), entry.file)
    with pytest.raises(ValueError, match="is not mode 0 ray 0"):
        rays_v3.verify_section(archive, entry3)
    _rewrite(path, good)
    assert rays_v3.verify_section(archive, entry)["mode"] == 0


def test_scene_lines_follow_index_order(tmp_path):
    archive = str(tmp_path / "a")
    rays_v3.write_fingerprint(archive, FP)
    entries = [_write(archive, "free", 1, 0, 2), _write(archive, "free", 0, 2, 4),
               _write(archive, "free", 0, 0, 2), _write(archive, "free", 1, 2, 4)]
    index = rays_v3.write_index(archive, entries)
    ids = [(json.loads(l)["mode"], json.loads(l)["ray"])
           for l in rays_v3.scene_lines(archive, index, "free")]
    assert ids == [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
    assert rays_v3.origins(archive, index, "free") == [["0", "0", "-0.02"]] * 2
    assert rays_v3.scene_lines(archive, index, "capillary") is not None
    assert list(rays_v3.scene_lines(archive, index, "capillary")) == []


# ------------------------------------------------------------------- rng

def test_stream_rng_is_keyed_by_name():
    a = stream_rng(7, SceneSeed.CAPILLARY, 3)
    b = stream_rng(7, SceneSeed.CAPILLARY, 3)
    assert a.getstate() == b.getstate()
    assert a.getstate() == random.Random("7/capillary/3").getstate()
    distinct = {stream_rng(7, SceneSeed.CAPILLARY, 3).random(),
                stream_rng(7, SceneSeed.CAPILLARY, 4).random(),
                stream_rng(7, SceneSeed.FREE, 3).random(),
                stream_rng(8, SceneSeed.CAPILLARY, 3).random(),
                stream_rng(73, SceneSeed.CAPILLARY).random()}
    assert len(distinct) == 5
    # "73/capillary" and "7/capillary/3" must not alias by concatenation.
    assert stream_rng(73, SceneSeed.CAPILLARY).random() != stream_rng(7, SceneSeed.CAPILLARY, 3).random()


def test_geometry_core_and_metadata_equal():
    geo = {"seed": 1, "capillary": {"source": {"n_modes": 5, "n_rays": 9, "size": 1e-6}},
           "free": {"source": {"n_modes": 2, "n_rays": 3}}, "screen": {"nx": 3}}
    core = geometry_core(geo)
    assert core == {"seed": 1, "capillary": {"source": {"size": 1e-6}},
                    "free": {"source": {}}, "screen": {"nx": 3}}
    assert geo["capillary"]["source"]["n_modes"] == 5       # input untouched
    assert geometry_core({"seed": 1, "capillary": {}}) == {"seed": 1, "capillary": {}}
    assert metadata_equal({"a": 1, "b": [1, 2]}, {"b": [1, 2], "a": 1})
    assert not metadata_equal({"a": 1}, {"a": 1.0})
    assert not metadata_equal({"a": "1"}, {"a": 1})
    assert not metadata_equal({"a": [1, 2]}, {"a": [2, 1]})


# ------------------------------------------------------------------- cli

@pytest.mark.parametrize("jobs", ["0", "9", "-1"])
def test_trace_v3_cli_rejects_jobs(tmp_path, jobs):
    from formula.capsysred.trace_v3 import main
    with pytest.raises(SystemExit):
        main([str(tmp_path / "cfg.yaml"), "--archive", str(tmp_path / "a"), "--jobs", jobs])
    assert not os.path.exists(tmp_path / "a")


def test_convert_cli_rejects_jobs(tmp_path):
    from formula.capsysred.convert_rays_v3 import MAX_JOBS, main
    with pytest.raises(SystemExit):
        main([str(tmp_path), "--jobs", str(MAX_JOBS + 1)])
    with pytest.raises(SystemExit):
        main(["--verify", str(tmp_path), "--jobs", "0"])


def test_stages_cli_rejects_unknown_stage(tmp_path):
    from formula.capsysred.__main__ import main
    with pytest.raises(SystemExit):
        main([str(tmp_path / "cfg.yaml"), "--stages", "4"])
    with pytest.raises(SystemExit):
        main([str(tmp_path / "cfg.yaml"), "--stages", "6,13"])

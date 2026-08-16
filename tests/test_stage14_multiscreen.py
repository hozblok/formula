"""Stage-14 one-pass fan-out and per-screen cache contracts."""

from __future__ import annotations

from contextlib import contextmanager
import gzip
import json
import math

import pytest

from formula.capsysred import Simulation
from formula.capsysred import stage14
from formula.capsysred.rays import sidecar_metadata, write_metadata
from formula.capsysred.screen import ScreenGrid


_RESULT_FILES = {
    "meta.json",
    "mu-jack.jsonl",
    "14-capillary-jack-mu.svg",
    "14a-capillary-jack-slice.svg",
    "14b-capillary-jack-intensity.svg",
    "14c-capillary-jack-overlay.svg",
    "14d-capillary-ray-scatter.svg",
}

_RESULT_KEYS = (
    "stage14:capillary",
    "stage14:capillary-s1",
    "stage14:capillary-s2",
)


def _config(*, extra_screens: bool) -> dict:
    capillary = {
        "source": {
            "shape": "point",
            "size": 3.0e-7,
            "position": [0.0, 0.0, -0.01],
            "n_modes": 2,
            "n_rays": 20,
        },
        "screen": {
            "nx": 3,
            "ny": 1,
            "edge_x": 3.0e-6,
            "edge_y": 1.0e-6,
        },
    }
    if extra_screens:
        # Same physical plane is deliberate: only the target grid changes, so
        # all three stores can be compared without introducing projection
        # edge cases into the orchestration test.
        capillary["screens"] = [
            {"nx": 5, "edge_x": 5.0e-6},
            {"nx": 7, "edge_x": 7.0e-6},
        ]
    return {
        "seed": 1402,
        "screen": {
            "nx": 3,
            "ny": 1,
            "edge_x": 3.0e-6,
            "edge_y": 1.0e-6,
        },
        "capillary": capillary,
    }


def _write_rays(path, raw: dict) -> None:
    """Write the smallest canonical capillary stream accepted by Stage 14."""
    sim = Simulation.from_dict(raw)
    grid = ScreenGrid(sim.cfg.capillary.screen)
    ref = grid.ref_pixel(sim.cfg.capillary.screen.reference)
    target = ref - 1
    x_ref, y_ref = grid.pixel_xy(ref)
    x_target, y_target = grid.pixel_xy(target)
    phase_flip = math.pi / float(sim.lines[0].k)
    n_modes, n_rays = sim.cfg.capillary.source.budget(1)
    with gzip.open(path, "xt", encoding="utf-8", newline="\n") as fh:
        fh.write("{}\n")
        for mode in range(n_modes):
            for ray in range(n_rays):
                at_ref = ray < n_rays // 2
                x, y = (x_ref, y_ref) if at_ref else (x_target, y_target)
                opl = phase_flip if mode == 1 and not at_ref else 0.0
                fh.write(json.dumps({
                    "stage": "capillary",
                    "mode": mode,
                    "ray": ray,
                    "fate": "screen",
                    "pixel": ref,
                    "opl": repr(opl),
                    "sins": [],
                    "x": x,
                    "y": y,
                    "dx": 0.0,
                    "dy": 0.0,
                }) + "\n")
        fh.write(json.dumps({
            "scene_end": "capillary",
            "rows": n_modes * n_rays,
        }) + "\n")
    write_metadata(path, sidecar_metadata(sim.cfg, 1))


def _cache_metas(recording) -> list[dict]:
    root = recording / "stage14-cache"
    return [
        json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        for directory in sorted(root.iterdir())
        if directory.is_dir() and not directory.name.endswith(".partial")
    ]


def _assert_screen_results(sim: Simulation, out_dir) -> None:
    expected = (
        ("stage14:capillary", "capillary", out_dir / "stage14"),
        ("stage14:capillary-s1", "capillary-s1",
         out_dir / "stage14" / "screen-1"),
        ("stage14:capillary-s2", "capillary-s2",
         out_dir / "stage14" / "screen-2"),
    )
    for index, (key, label, directory) in enumerate(expected):
        result = sim.results[key]
        assert result["result_meta"]["screen"] == label
        assert all(row["screen"] == label for row in result["rows"])
        expected_names = (_RESULT_FILES | {"screen-1", "screen-2"}
                          if index == 0 else _RESULT_FILES)
        assert {path.name for path in directory.iterdir()} == expected_names


def test_stage14_cold_multiscreen_fanout_opens_archive_once_and_reuses_all(
        tmp_path, monkeypatch):
    raw = _config(extra_screens=True)
    recording = tmp_path / "recording"
    recording.mkdir()
    rays = recording / "rays.jsonl.gz"
    _write_rays(rays, raw)

    archive_opens = []
    original_archive_text = stage14._archive_text

    @contextmanager
    def counted_archive(part):
        archive_opens.append(part.path)
        with original_archive_text(part) as fh:
            yield fh

    monkeypatch.setattr(stage14, "_archive_text", counted_archive)
    cold = Simulation.from_dict(raw)
    cold.replay(str(rays), str(tmp_path / "cold"), stages=[14])

    assert archive_opens == [str(rays.resolve())]
    _assert_screen_results(cold, tmp_path / "cold")
    assert {key for key in cold.results if key.startswith("stage14:")} == {
        "stage14:capillary",
        "stage14:capillary-s1",
        "stage14:capillary-s2",
    }
    assert all(cold.results[key]["cache_hits"] == 0 for key in _RESULT_KEYS)
    root_meta = json.loads(
        (tmp_path / "cold" / "stage14" / "meta.json").read_text(
            encoding="utf-8"))
    assert [item["screen"] for item in root_meta["fanout"]["screens"]] == [
        "capillary", "capillary-s1", "capillary-s2"]
    assert root_meta["fanout"]["physical_ray_archive_bytes_read"] == rays.stat().st_size
    assert root_meta["performance"]["total_seconds"] == cold.results[
        "stage14:capillary"]["seconds"]
    assert sum(cold.results[key]["result_meta"]["performance"][
        "ray_archive_bytes_read"] for key in _RESULT_KEYS) == rays.stat().st_size

    metas = _cache_metas(recording)
    assert len(metas) == 3
    assert len({meta["analysis_id"] for meta in metas}) == 3
    assert {meta["analysis_signature"]["screen"]["nx"]
            for meta in metas} == {3, 5, 7}
    assert {meta["n_pixels"] for meta in metas} == {3, 5, 7}

    # A fully hot fan-out must not touch gzip at all.
    def no_gzip(*_args, **_kwargs):
        raise AssertionError("a Stage-14 all-cache-hit run opened gzip")

    monkeypatch.setattr(stage14.gzip, "GzipFile", no_gzip)
    hot = Simulation.from_dict(raw)
    hot.replay(str(rays), str(tmp_path / "hot"), stages=[14])
    _assert_screen_results(hot, tmp_path / "hot")
    assert archive_opens == [str(rays.resolve())]
    assert all(hot.results[key]["cache_hits"] == 1 for key in _RESULT_KEYS)
    assert all(
        hot.results[key]["result_meta"]["performance"]["ray_archive_bytes_read"]
        == 0
        for key in _RESULT_KEYS
    )

    # No-clobber is checked before either cache work or gzip I/O.
    with pytest.raises(ValueError, match="result already exists"):
        hot.replay(str(rays), str(tmp_path / "hot"), stages=[14])


def test_stage14_partial_hit_builds_two_missing_screens_in_one_pass(
        tmp_path, monkeypatch):
    base_raw = _config(extra_screens=False)
    recording = tmp_path / "recording"
    recording.mkdir()
    rays = recording / "rays.jsonl.gz"
    _write_rays(rays, base_raw)

    base = Simulation.from_dict(base_raw)
    base.replay(str(rays), str(tmp_path / "base"), stages=[14])
    assert set((tmp_path / "base" / "stage14").iterdir()) == {
        tmp_path / "base" / "stage14" / name for name in _RESULT_FILES
    }
    assert {key for key in base.results if key.startswith("stage14:")} == {
        "stage14:capillary"
    }
    assert len(_cache_metas(recording)) == 1

    archive_opens = []
    original_archive_text = stage14._archive_text

    @contextmanager
    def counted_archive(part):
        archive_opens.append(part.path)
        with original_archive_text(part) as fh:
            yield fh

    monkeypatch.setattr(stage14, "_archive_text", counted_archive)
    expanded = Simulation.from_dict(_config(extra_screens=True))
    expanded.replay(str(rays), str(tmp_path / "expanded"), stages=[14])

    assert archive_opens == [str(rays.resolve())]
    _assert_screen_results(expanded, tmp_path / "expanded")
    assert expanded.results["stage14:capillary"]["cache_hits"] == 1
    assert expanded.results["stage14:capillary-s1"]["cache_hits"] == 0
    assert expanded.results["stage14:capillary-s2"]["cache_hits"] == 0
    expanded_meta = json.loads(
        (tmp_path / "expanded" / "stage14" / "meta.json").read_text(
            encoding="utf-8"))
    assert expanded_meta["fanout"]["physical_ray_archive_bytes_read"] == rays.stat().st_size
    assert expanded.results["stage14:capillary"]["result_meta"][
        "performance"]["target_cache_miss_ray_archive_bytes"] == 0
    assert len(_cache_metas(recording)) == 3

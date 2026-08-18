"""End-to-end checks for the Stage-14 stream/cache/publication path."""

from __future__ import annotations

import gzip
import json
import math

import pytest

from formula.capsysred import Simulation
from formula.capsysred.rays import sidecar_metadata, write_metadata
from formula.capsysred.screen import ScreenGrid
from formula.capsysred.stage14_flags import FlagThresholds, validate_pixel_row


def _config(n_modes=10, seed=101, *, thresholds=None):
    raw = {
        "seed": seed,
        "screen": {
            "nx": 3, "ny": 1,
            "edge_x": 3.0e-6, "edge_y": 1.0e-6,
        },
        "capillary": {
            "source": {
                "shape": "point", "size": 3.0e-7,
                "position": [0.0, 0.0, -0.01],
                "n_modes": n_modes, "n_rays": 20,
            },
            "screen": {
                "nx": 3, "ny": 1,
                "edge_x": 3.0e-6, "edge_y": 1.0e-6,
            },
        },
    }
    if thresholds is not None:
        raw["stage14"] = {"flag_thresholds": thresholds}
    return raw


def _write_controlled_rays(path, raw, signs):
    """Two occupied pixels with a known cross-mode phase cancellation."""
    sim = Simulation.from_dict(raw)
    cfg = sim.cfg
    grid = ScreenGrid(cfg.capillary.screen)
    ref = grid.ref_pixel(cfg.capillary.screen.reference)
    target = ref - 1
    x_ref, y_ref = grid.pixel_xy(ref)
    x_target, y_target = grid.pixel_xy(target)
    phase_flip = math.pi / float(sim.lines[0].k)
    n_modes, n_rays = cfg.capillary.source.budget()
    assert n_modes == len(signs) and n_rays == 20
    with gzip.open(path, "xt", encoding="utf-8", newline="\n") as fh:
        # The preamble is deliberately neither {} nor legacy metadata.  It is
        # an ignored framing record under the current rays contract.
        fh.write(json.dumps(["ignored", {"geometry": "contradictory"}]) + "\n")
        for mode, sign in enumerate(signs):
            for ray in range(n_rays):
                at_ref = ray < n_rays // 2
                x, y = (x_ref, y_ref) if at_ref else (x_target, y_target)
                opl = 0.0 if at_ref or sign > 0 else phase_flip
                fh.write(json.dumps({
                    "stage": "capillary", "mode": mode, "ray": ray,
                    "fate": "screen",
                    # Deliberately wrong: Stage 14 and replay must re-bin x/y.
                    "pixel": ref + 1,
                    "opl": repr(opl), "sins": [],
                    "x": x, "y": y, "dx": 0.0, "dy": 0.0,
                }) + "\n")
        fh.write(json.dumps({
            "scene_end": "capillary", "rows": n_modes * n_rays,
        }) + "\n")
    write_metadata(path, sidecar_metadata(cfg))
    return target, ref


def _flatten(grid):
    return [value for row in grid for value in row]


def _rewrite_gzip_lines(path, transform):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        lines = list(fh)
    lines = transform(lines)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as fh:
        fh.writelines(lines)


def test_stage14_cache_hit_schema_and_stage10_projection(tmp_path, monkeypatch):
    raw = _config()
    rays_dir = tmp_path / "recording"
    rays_dir.mkdir()
    rays = rays_dir / "rays.jsonl.gz"
    target, ref = _write_controlled_rays(
        rays, raw, [1, 1, 1, 1, 1, 1, -1, -1, -1, -1])

    old = Simulation.from_dict(raw)
    old.replay(str(rays), str(tmp_path / "stage10"), stages=[10])
    modern = Simulation.from_dict(raw)
    result = modern.replay(str(rays), str(tmp_path / "stage14"), stages=[14])
    new = modern.results["stage14:capillary"]

    assert new["cache_hits"] == 0
    assert new["n_modes"] == 10
    assert new["ref_pixel"] == ref
    assert new["ref_status"] == "ok"
    assert len(new["rows"]) == 3
    assert new["rows"][target]["flag"] == "trusted"
    assert new["rows"][target]["n_mu_loo_valid"] == 10
    assert new["rows"][ref]["flag"] is None
    assert new["rows"][ref]["mu_raw"] is None
    assert new["rows"][ref + 1]["flag"] == "no-rays"
    thresholds = FlagThresholds(**modern.cfg.stage14_flag_thresholds)
    for row in new["rows"]:
        validate_pixel_row(row, thresholds)

    old_maps = old.results["jack:capillary"]["maps"]
    assert _flatten(old_maps["density"]) == _flatten(new["maps"]["density"])
    assert _flatten(old_maps["intensity"]) == pytest.approx(
        _flatten(new["maps"]["intensity"]), rel=1e-13, abs=1e-13)
    assert old_maps["solid"][0][target] == 1.0
    assert old_maps["mu"][0][target] == pytest.approx(
        min(new["rows"][target]["mu_raw"], 1.0), rel=1e-12, abs=1e-12)

    expected = {
        "meta.json", "mu-jack.jsonl", "14-capillary-jack-mu.svg",
        "14a-capillary-jack-slice.svg", "14b-capillary-jack-intensity.svg",
        "14c-capillary-jack-overlay.svg", "14d-capillary-ray-scatter.svg",
        "14e-capillary-ref-passport.svg",
    }
    assert {path.name for path in (tmp_path / "stage14" / "stage14").iterdir()} == expected
    rows = [json.loads(line) for line in
            (tmp_path / "stage14" / "stage14" / "mu-jack.jsonl")
            .read_text(encoding="utf-8").splitlines()]
    assert rows == new["rows"]
    assert not ({"mu", "mu_err", "solid", "dubious", "stage"} & rows[0].keys())

    # Threshold-only reclassification uses the same expensive cache.  Prove
    # the hot path does not even call gzip.open.
    changed = _config(thresholds={"ic_n_sigma": 2.5, "ref_ic_n_sigma": 3.0,
                                  "w_n_sigma": 3.0, "min_coherent_fraction": 0.08})

    def no_gzip(*_args, **_kwargs):
        raise AssertionError("cache hit opened the rays archive")

    monkeypatch.setattr("formula.capsysred.stage14.gzip.open", no_gzip)
    hit = Simulation.from_dict(changed)
    hit.replay(str(rays), str(tmp_path / "stage14-hit"), stages=[14])
    hit_result = hit.results["stage14:capillary"]
    assert hit_result["cache_hits"] == 1
    assert hit_result["result_meta"]["performance"]["ray_archive_bytes_read"] == 0
    with pytest.raises(ValueError, match="result already exists"):
        hit.replay(str(rays), str(tmp_path / "stage14-hit"), stages=[14])
    assert result["out_dir"] == str(tmp_path / "stage14")


def test_stage14_union_matches_monolithic_cache(tmp_path):
    union_raw = _config(n_modes=10, seed=999)
    signs_a = [1, 1, 1, -1, -1]
    signs_b = [1, 1, 1, -1, -1]
    part_paths = []
    for index, (seed, signs) in enumerate(((11, signs_a), (22, signs_b))):
        directory = tmp_path / f"part-{index}"
        directory.mkdir()
        path = directory / "rays.jsonl.gz"
        _write_controlled_rays(path, _config(n_modes=5, seed=seed), signs)
        part_paths.append(str(path))
    mono_dir = tmp_path / "mono-recording"
    mono_dir.mkdir()
    mono_path = mono_dir / "rays.jsonl.gz"
    _write_controlled_rays(mono_path, _config(n_modes=10, seed=33),
                           signs_a + signs_b)

    union = Simulation.from_dict(union_raw)
    union.replay(part_paths, str(tmp_path / "union-out"), stages=[14])
    mono = Simulation.from_dict(union_raw)
    mono.replay(str(mono_path), str(tmp_path / "mono-out"), stages=[14])
    a, b = union.results["stage14:capillary"], mono.results["stage14:capillary"]
    assert [row["flag"] for row in a["rows"]] == [row["flag"] for row in b["rows"]]
    for key in ("I", "ic", "ic_err", "w_abs", "w_err", "mu_raw", "mu_raw_err"):
        av = [row[key] for row in a["rows"]]
        bv = [row[key] for row in b["rows"]]
        for left, right in zip(av, bv):
            if left is None or right is None:
                assert left is right
            else:
                assert left == pytest.approx(right, rel=2e-12, abs=2e-12)
    parts = a["result_meta"]["parts"]
    assert [(part["global_mode_start"], part["global_mode_end"])
            for part in parts] == [(0, 5), (5, 10)]

    # Caller order remains explicit provenance, but compensated part totals
    # keep the scientific result invariant under a union permutation.
    reversed_sim = Simulation.from_dict(union_raw)
    reversed_sim.replay(list(reversed(part_paths)),
                        str(tmp_path / "reversed-out"), stages=[14])
    reversed_rows = reversed_sim.results["stage14:capillary"]["rows"]
    assert [row["flag"] for row in reversed_rows] == [
        row["flag"] for row in a["rows"]
    ]
    for key in ("I", "ic", "ic_err", "w_abs", "w_err", "mu_raw",
                "mu_raw_err"):
        for left, right in zip(
                (row[key] for row in reversed_rows),
                (row[key] for row in a["rows"])):
            if left is None or right is None:
                assert left is right
            else:
                assert left == pytest.approx(right, rel=2e-12, abs=2e-12)


def test_stage14_local_reuse_is_strict_but_explicit_replay_is_deliberate(tmp_path):
    recorded_raw = _config(seed=17)
    run_raw = _config(seed=18)
    local = tmp_path / "local"
    local.mkdir()
    rays = local / "rays.jsonl.gz"
    _write_controlled_rays(rays, recorded_raw,
                           [1, 1, 1, 1, 1, 1, -1, -1, -1, -1])
    with pytest.raises(ValueError, match="does not match this config"):
        Simulation.from_dict(run_raw).run(str(local), stages=[14])
    assert not (local / "stage14").exists()

    explicit = Simulation.from_dict(run_raw)
    explicit.replay(str(rays), str(tmp_path / "explicit"), stages=[14])
    assert explicit.results["stage14:capillary"]["n_modes"] == 10


def test_stage14_needs_a_recording_and_reads_v2_or_v3(tmp_path):
    import yaml
    from formula.capsysred.trace_v3 import trace as trace_v3
    raw = _config(n_modes=2, seed=71)
    with pytest.raises(ValueError, match="no rays recording"):
        Simulation.from_dict(raw).run(str(tmp_path / "empty"), stages=[14])
    assert not (tmp_path / "empty" / "stage14").exists()
    v2 = tmp_path / "v2"
    sim = Simulation.from_dict(raw)
    sim.trace(str(v2))
    result = sim.run(str(v2), stages=[14])
    assert "stage14:capillary" in sim.results
    assert any(name.endswith("14d-capillary-ray-scatter.svg")
               for name in result["files"])
    v3 = tmp_path / "v3"
    cfg = v3 / "cfg.yaml"
    v3.mkdir()
    with open(cfg, "w", encoding="utf-8") as fh:
        yaml.safe_dump(raw, fh, sort_keys=False)
    trace_v3(str(cfg), str(v3 / "rays-modes"), None, jobs=1, level=6, log=lambda m: None)
    other = Simulation.from_dict(raw)
    other.run(str(v3), stages=[14])
    assert (other.results["stage14:capillary"]["rows"]
            == sim.results["stage14:capillary"]["rows"])


@pytest.mark.parametrize("damage, match", [
    ("missing-target-field", "optical path length"),
    ("bad-trailer", "capillary trailer"),
    ("bad-foreign-row", "optical path length"),
])
def test_stage14_strict_stream_rejects_corruption(tmp_path, damage, match):
    raw = _config(n_modes=2, seed=81)
    recording = tmp_path / damage
    recording.mkdir()
    rays = recording / "rays.jsonl.gz"
    _write_controlled_rays(rays, raw, [1, -1])

    def corrupt(lines):
        if damage == "missing-target-field":
            row = json.loads(lines[1])
            del row["opl"]
            lines[1] = json.dumps(row) + "\n"
        elif damage == "bad-trailer":
            row = json.loads(lines[-1])
            row["rows"] -= 1
            lines[-1] = json.dumps(row) + "\n"
        else:
            foreign = {
                "stage": "free", "mode": 0, "ray": 0,
                "fate": "lost", "pixel": None, "sins": [],
            }
            lines.insert(1, json.dumps(foreign) + "\n")
        return lines

    _rewrite_gzip_lines(rays, corrupt)
    with pytest.raises(ValueError, match=match):
        Simulation.from_dict(raw).replay(
            str(rays), str(tmp_path / f"out-{damage}"), stages=[14]
        )
    cache_root = recording / "stage14-cache"
    assert list(cache_root.glob("*.partial"))
    assert not [path for path in cache_root.iterdir()
                if not path.name.endswith(".partial")]

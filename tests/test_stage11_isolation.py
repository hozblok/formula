"""Differential trace/Stage-14 isolation against an exported Git revision.

The differential is opt-in: CAPSYSRED_ISOLATION_BASELINE names a Git revision.
Set CAPSYSRED_ISOLATION_BASELINE_NATIVE to test an older extension as well;
otherwise both sides deliberately use the current extension. Parser and
metadata regressions always run and need neither Git nor a second extension.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
GEOMETRIES = {
    "cylinder": {"center": [0.0, 0.0], "radius": 6e-6},
    "hex": {"center": [0.0, 0.0], "radius": 6e-6, "sides": 6},
    "torus": {"center": [0.0, 0.0], "radius": 6e-6,
              "bend": {"radius": 1000.0, "toward": [1.0, 0.0]}},
    "funnel": {"center": [2e-6, 0.0], "radius": 6e-6,
               "funnel": {"g": [1.0, 5.0], "f": [-2.0, 5.0]}},
}
LEGACY_BEAMLET = {
    "string-waist": {"w0": "5e-7"},
    "zero-waist": {"w0": 0},
    "boolean-anisotropy": {"w0_t": True},
    "unused-waist-plane": {"waist_z": "unused-by-stage14"},
}
CASES = [f"{kind}-{spectrum}" for kind in GEOMETRIES
         for spectrum in ("mono", "band")]
CASES += list(LEGACY_BEAMLET)
CASES += ["controlled-band"]


def _config(case):
    kind, spectrum = ("cylinder", "mono") if case in LEGACY_BEAMLET else case.split("-")
    if kind == "controlled":
        kind = "cylinder"
    raw = {
        "precision": 32,
        "seed": 271828,
        "energy_kev": 8.048,
        "screen": {"nx": 3, "ny": 3},
        "capillary": {
            "z0": 0.0, "z1": 0.05,
            "bores": [GEOMETRIES[kind]],
            "source": {"shape": "disk", "size": 3e-7,
                       "position": [0.0, 0.0, -0.01],
                       "n_modes": 3, "n_rays": 80},
            "screen": {"z": 0.06, "edge_x": 24e-6, "edge_y": 24e-6,
                       "nx": 3, "ny": 3, "reference": [0.0, 0.0]},
            "screens": [{"z": 0.08, "nx": 5, "ny": 3,
                         "edge_x": 48e-6, "edge_y": 32e-6}],
        },
    }
    if spectrum == "band":
        raw["spectrum"] = {"mode": "lines", "per_line_fresnel": True,
                           "lines": [{"energy_kev": 7.2432, "weight": 1.0},
                                     {"energy_kev": 8.048, "weight": 2.0},
                                     {"energy_kev": 8.8528, "weight": 3.0}]}
    if case in LEGACY_BEAMLET:
        raw["beamlet"] = LEGACY_BEAMLET[case]
    return raw


@pytest.mark.parametrize("case", LEGACY_BEAMLET)
def test_unused_beamlet_values_preserve_trace_and_stage14_contracts(case, tmp_path):
    from formula.capsysred import Simulation
    from formula.capsysred.rays import geometry_metadata, sidecar_metadata
    from formula.capsysred.screen import ScreenGrid
    from formula.capsysred.stages.stage14 import _analysis_signature, _screen_contract

    base = Simulation.from_dict(_config("cylinder-mono"))
    other = Simulation.from_dict(_config(case))
    assert "waist_z" not in base.cfg.raw["beamlet"]
    assert geometry_metadata(other.cfg) == geometry_metadata(base.cfg)
    assert sidecar_metadata(other.cfg) == sidecar_metadata(base.cfg)
    for normal, legacy in zip([base.cfg.capillary.screen, *base.cfg.capillary.screens],
                              [other.cfg.capillary.screen, *other.cfg.capillary.screens]):
        a, b = ScreenGrid(normal), ScreenGrid(legacy)
        assert _analysis_signature(base, _screen_contract(a, a.ref_pixel(normal.reference))) == (
            _analysis_signature(other, _screen_contract(b, b.ref_pixel(legacy.reference))))
    with pytest.raises(ValueError, match="beamlet"):
        other._stage11(str(tmp_path))
    assert not list(tmp_path.iterdir())


def _controlled_rows(config, case_dir):
    import gzip
    from formula.capsysred import Simulation
    from formula.capsysred.rays import sidecar_metadata, write_metadata

    sim = Simulation.from_yaml(str(config))
    archive = case_dir / "rays.jsonl.gz"
    rows = []
    for mode in range(3):
        for ray in range(80):
            at_ref = ray < 50
            rows.append(json.dumps({
                "stage": "capillary", "mode": mode, "ray": ray, "fate": "screen",
                "pixel": 4 if at_ref else 5,
                "opl": repr(0.061 + mode * 1e-5 + (0.0 if at_ref else (mode + 1) * 1e-12)),
                "sins": [0.001, 0.002], "x": 0.0 if at_ref else 8e-6,
                "y": 0.0, "dx": 0.0, "dy": 0.0,
            }).encode("utf-8") + b"\n")
    raw = b"".join(rows)
    with gzip.open(archive, "wb") as stream:
        stream.write(b"{}\n" + raw + b'{"scene_end": "capillary", "rows": 240}\n')
    write_metadata(str(archive), sidecar_metadata(sim.cfg))
    return raw, case_dir / "rays-fingerprint.yaml"


def _worker(out):
    import contextlib
    import hashlib
    import math
    import traceback

    import yaml
    from formula import _formula
    from formula.capsysred import Simulation, rays_v3
    from formula.capsysred.rays import geometry_metadata
    from formula.capsysred.trace_v3 import trace

    out.mkdir(parents=True)
    extension = Path(_formula.__file__)
    runtime = {"extension": str(extension),
               "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
               "lens_stride": getattr(_formula.BeamletGrid, "lens_stride", None),
               "simulation_source": str(sys.modules[Simulation.__module__].__file__)}
    (out / "runtime.json").write_text(json.dumps(runtime), encoding="utf-8")
    for case in CASES:
        case_dir = out / case
        case_dir.mkdir()
        try:
            config = case_dir / "config.yaml"
            config.write_text(yaml.safe_dump(_config(case), sort_keys=False), encoding="utf-8")
            if case == "controlled-band":
                rows, fingerprint = _controlled_rows(config, case_dir)
            else:
                archive = case_dir / "rays-modes"
                trace(str(config), str(archive), jobs=1, level=1, log=lambda _: None,
                      scenes=("capillary",))
                index = rays_v3.load_index(str(archive))
                rows = b"".join(rays_v3.scene_lines(str(archive), index, "capillary"))
                fingerprint = archive / "rays-fingerprint.yaml"
            evidence = case_dir / "evidence"
            evidence.mkdir()
            (evidence / "trace-rows.jsonl").write_bytes(rows)
            shutil.copyfile(fingerprint, evidence / "rays-fingerprint.yaml")
            parsed = [json.loads(line) for line in rows.splitlines()]
            assert len(parsed) == 240
            assert any(row.get("sins") for row in parsed), "case never reflected"
            sim = Simulation.from_yaml(str(config))
            (evidence / "raw-config.json").write_text(
                json.dumps(sim.cfg.raw, sort_keys=True), encoding="utf-8")
            (evidence / "geometry.json").write_text(
                json.dumps(geometry_metadata(sim.cfg), sort_keys=True), encoding="utf-8")
            with (case_dir / "run.log").open("w", encoding="utf-8") as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    sim.run(str(case_dir), stages=[14])
            screen_summaries = []
            for name, result in sim.results.items():
                if not name.startswith("stage14:"):
                    continue
                if case == "controlled-band":
                    assert result["ref_status"] == "ok"
                    assert any(row.get("mu_raw") is not None
                               and math.isfinite(row["mu_raw"]) and row["mu_raw"] > 0
                               for row in result["rows"])
                label = name.split(":", 1)[1]
                subdir = "" if label == "capillary" else "screen-1"
                shutil.copyfile(case_dir / "stage14" / subdir / "mu-jack.jsonl",
                                evidence / f"{label}.mu-jack.jsonl")
                assert len(result["cache_parts"]) == 1
                cache = result["cache_parts"][0]
                cache_dir = Path(cache.rows_path).parent
                for filename in ("mode-rows.f64", "aggregates.bin"):
                    shutil.copyfile(cache_dir / filename, evidence / f"{label}.{filename}")
                contract = {key: cache.meta[key] for key in
                            ("analysis_signature", "stage_id", "screen", "source_screen_z",
                             "n_modes", "n_rays_per_mode", "n_pixels", "reference_pixel",
                             "max_bounces", "scatter", "files")}
                (evidence / f"{label}.contract.json").write_text(
                    json.dumps(contract, sort_keys=True), encoding="utf-8")
                screen_summaries.append({"screen": label, "ref_status": result["ref_status"],
                                         "rows": len(result["rows"]), "stats": result["stats"]})
            assert len(screen_summaries) == 2
            (case_dir / "summary.json").write_text(json.dumps({
                "trace_rows": len(parsed),
                "reflected_rows": sum(bool(row.get("sins")) for row in parsed),
                "screens": screen_summaries,
            }, indent=2), encoding="utf-8")
        except Exception:
            (case_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")


@pytest.fixture(scope="module")
def differential_outputs(tmp_path_factory):
    baseline_ref = os.environ.get("CAPSYSRED_ISOLATION_BASELINE")
    if not baseline_ref:
        pytest.skip("set CAPSYSRED_ISOLATION_BASELINE to opt into the Git differential")
    scratch = tmp_path_factory.mktemp("stage11-isolation")
    baseline = scratch / "baseline-source"
    baseline.mkdir()
    archive = subprocess.check_output(["git", "archive", baseline_ref, "src"], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        bundle.extractall(baseline, filter="data")
    from formula import _formula
    current_native = Path(_formula.__file__)
    baseline_native = Path(os.environ.get("CAPSYSRED_ISOLATION_BASELINE_NATIVE", current_native))
    shutil.copy2(baseline_native, baseline / "src" / "formula" / current_native.name)
    (scratch / "baseline-commit.txt").write_bytes(subprocess.check_output(
        ["git", "rev-parse", baseline_ref], cwd=ROOT))
    for label, source in (("baseline", baseline / "src"), ("current", ROOT / "src")):
        env = os.environ.copy()
        env.update(PYTHONPATH=str(source), PYTHONUTF8="1", CAPSYSRED_STAGE14_JOBS="1")
        env.pop("CAPSYSRED_PYTHON_TRACE", None)
        run = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker",
                              str(scratch / label)], cwd=scratch, env=env,
                             capture_output=True, text=True, timeout=240)
        assert run.returncode == 0, run.stdout + run.stderr
    # both sides must have loaded the native library they were given
    import hashlib
    expected = {"baseline": hashlib.sha256(baseline_native.read_bytes()).hexdigest(),
                "current": hashlib.sha256(current_native.read_bytes()).hexdigest()}
    for label, digest in expected.items():
        runtime = json.loads((scratch / label / "runtime.json").read_text(encoding="utf-8"))
        assert runtime["extension_sha256"] == digest, (
            f"{label} loaded another native library: {runtime['extension']}")
    print(f"\nDifferential evidence: {scratch}")
    return scratch


@pytest.mark.parametrize("case", CASES)
def test_trace_and_stage14_bytes_unchanged(differential_outputs, case):
    root = differential_outputs
    for label in ("baseline", "current"):
        error = root / label / case / "error.txt"
        assert not error.exists(), f"{label}/{case}: {error.read_text(encoding='utf-8')}"
    old = root / "baseline" / case / "evidence"
    new = root / "current" / case / "evidence"
    names = sorted(path.name for path in old.iterdir())
    assert names == sorted(path.name for path in new.iterdir())
    assert names
    for name in names:
        assert (old / name).read_bytes() == (new / name).read_bytes(), f"{case}/{name}"
    old_figures = root / "baseline" / case / "stage14"
    new_figures = root / "current" / case / "stage14"
    figures = sorted(path.relative_to(old_figures) for path in old_figures.rglob("*.svg"))
    assert figures == sorted(path.relative_to(new_figures) for path in new_figures.rglob("*.svg"))
    assert figures
    for name in figures:
        assert (old_figures / name).read_bytes() == (new_figures / name).read_bytes(), f"{case}/{name}"


if __name__ == "__main__":
    assert sys.argv[1] == "--worker"
    _worker(Path(sys.argv[2]))

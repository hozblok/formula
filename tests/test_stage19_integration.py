"""NumPy-free Stage 19 dispatch, source headers, precision and checkpoints."""

from array import array
from pathlib import Path
from types import SimpleNamespace
import gzip
import json
import lzma
import math
import subprocess
import sys

import pytest

from formula import Number
from formula.capsysred import Simulation, rays_v3
from formula.capsysred.rays import geometry_metadata
from formula.capsysred.stages import stage19 as S


def scene(modes=2):
    return Simulation.from_dict({"precision": 32, "energy_kev": 8.048,
        "capillary": {"z0": 0, "z1": .05,
            "source": {"shape": "point", "size": 0, "position": [0, 0, -.1],
                       "n_modes": modes, "n_rays": 1},
            "screen": {"z": .05, "nx": 3, "ny": 3, "edge_x": 6e-6,
                       "edge_y": 6e-6, "reference": [0, 0]},
            "screens": [{"z": .4}]}})


def archive(tmp_path, sim):
    target = tmp_path / "rays-modes"
    (target / "modes").mkdir(parents=True)
    rays_v3.write_fingerprint(target, {"format": 3, "geometry": geometry_metadata(sim.cfg)})
    entries = []
    for mode in range(sim.cfg.capillary.source.n_modes):
        name = rays_v3.section_name("capillary", mode, 0, 1)
        header = {"header": 3, "scene": "capillary", "mode": mode, "r0": 0, "r1": 1,
                  "origin": ["0.000000000000000000000000000001", "0", "-0.1"]}
        with gzip.open(target / "modes" / name, "wt", encoding="utf-8") as handle:
            handle.write(json.dumps(header) + "\nnot-a-ray-body\n")
        entries.append(rays_v3.Section("capillary", mode, 0, 1, 1, 1, "0" * 64, name))
    rays_v3.write_index(target, entries)
    return target


def test_import_path_excludes_array_packages():
    code = """
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy','scipy','finufft','matplotlib'}:
            raise RuntimeError('forbidden import: '+fullname)
sys.meta_path.insert(0, Block())
from formula.capsysred.stages import stage19
from formula.capsysred.simulation import KNOWN_STAGES
assert 19 in KNOWN_STAGES
"""
    subprocess.run([sys.executable, "-B", "-c", code], check=True, timeout=60)


def test_source_selection_reads_headers_not_bodies(tmp_path):
    sim = scene()
    path = archive(tmp_path, sim)
    opts = S.validate(sim.cfg)
    sources, provenance = S.read_sources(sim, opts, [str(path)])
    assert len(sources) == 2 and sources[0]["origin"][0] == str(Number("1e-30", 32))
    assert sources[0]["weight"] == .5
    assert provenance["ray_bodies_read"] is False
    assert provenance["ray_body_checksums_verified"] is False
    opts["max_modes"] = 3
    with pytest.raises(ValueError, match="archive contains"):
        S.read_sources(sim, opts, [str(path)])


def test_stage19_config_is_isolated_and_strict():
    sim = scene()
    assert "stage19" not in sim.cfg.raw
    assert S.validate(sim.cfg)["observables"] == ["point", "coherent_cell"]
    assert S.validate(sim.cfg)["archive_nodes"] == "none"
    assert S.validate(sim.cfg)["archive_compression"] == "gzip"
    assert S.validate(sim.cfg)["preview_modes"] == []
    for bad in ({"workers": True}, {"max_modes": 0}, {"numpify": True},
                {"observables": ["intensity"]}, {"screen_indices": [0, 0]},
                {"mesh": []}, {"resume": "yes"}, {"archive_nodes": True},
                {"archive_nodes": []}, {"archive_nodes": "parents"},
                {"archive_compression": True}, {"archive_compression": []},
                {"archive_compression": "../xz"}, {"archive_compression": "gzip;echo bad"},
                {"preview_modes": True}, {"preview_modes": [0]}, {"preview_modes": [True]},
                {"preview_modes": [1, 1]}, {"preview_modes": [1.5]}):
        sim.cfg.raw["stage19"] = bad
        with pytest.raises(ValueError, match="stage19"):
            S.validate(sim.cfg)


def test_preview_count_cannot_exceed_source_ensemble(tmp_path):
    sim = scene(2)
    sim.cfg.raw["stage19"] = {"source_archive": str(archive(tmp_path, sim)), "preview_modes": [3]}
    with pytest.raises(ValueError, match="preview_modes exceeds"):
        S.run_stage19(sim, tmp_path / "out", log=lambda value: None)
    assert not (tmp_path / "out/stage19/manifest.json").exists()


def test_physical_exit_configuration_and_coverage_target_are_explicit():
    sim = scene()
    assert S.validate(sim.cfg)["mesh"].get("phase_coordinates", "pullback") == "pullback"
    sim.cfg.raw["stage19"] = {"minimum_accepted_fraction": .95,
        "mesh": {"phase_coordinates": "physical_exit", "target_accepted_fraction": ".98"}}
    result = S.validate(sim.cfg)
    assert result["mesh"]["phase_coordinates"] == "physical_exit"
    assert result["mesh"]["target_accepted_fraction"] == ".98"
    for bad in ("unknown", True, [], None):
        sim.cfg.raw["stage19"]["mesh"]["phase_coordinates"] = bad
        with pytest.raises(ValueError, match="phase_coordinates"):
            S.validate(sim.cfg)
    sim.cfg.raw["stage19"]["mesh"]["phase_coordinates"] = "physical_exit"
    for bad in ("NaN", "Infinity", ".94", "1.01", True, []):
        sim.cfg.raw["stage19"]["mesh"]["target_accepted_fraction"] = bad
        with pytest.raises(ValueError, match="target_accepted_fraction"):
            S.validate(sim.cfg)
    for valid in (None, ".95", 1):
        sim.cfg.raw["stage19"]["mesh"]["target_accepted_fraction"] = valid
        S.validate(sim.cfg)


def test_stage19_replay_skips_generic_ray_reader(tmp_path, monkeypatch):
    from formula.capsysred import simulation
    sim = scene()
    seen = []
    def run(got, output, options, *, rays_paths, log):
        assert got is sim and rays_paths == ["header-only"]
        seen.append("run")
        return {"results": {"n_modes": 2}, "files": [], "report": []}
    backend = SimpleNamespace(validate=S.validate,
        preflight_stage19_inputs=lambda *args: seen.append("inputs"),
        preflight_stage19_output=lambda *args: seen.append("output"), run_stage19=run)
    monkeypatch.setattr(simulation, "_stage19_backend", lambda: backend)
    monkeypatch.setattr(simulation, "RaysReader", lambda *args: pytest.fail("ray body scan"))
    monkeypatch.setattr(simulation, "MultiRaysReader", lambda *args: pytest.fail("ray body scan"))
    sim.replay("header-only", str(tmp_path / "out"), stages=[19])
    assert seen == ["inputs", "output", "run"]
    assert sim.results["stage19"]["n_modes"] == 2


def test_stage19_requires_capillary_before_backend_load(tmp_path, monkeypatch):
    from formula.capsysred import simulation
    sim = Simulation.from_dict({"free": {"source": {"shape": "point", "size": 0,
        "position": [0, 0, -.1], "n_modes": 1, "n_rays": 1}}})
    monkeypatch.setattr(simulation, "_stage19_backend", lambda: pytest.fail("backend loaded"))
    with pytest.raises(ValueError, match="capillary.source"):
        sim.run(str(tmp_path / "out"), stages=[19])
    assert not (tmp_path / "out").exists()


def test_full_opl_phase_reduced_before_float_conversion():
    sim = scene()
    opl = "1234567.89012345678901234567890123"
    triangle = {"input": [["0", "0"], ["1", "0"], ["0", "1"]],
                "exit": [["0", "0"], ["1e-6", "0"], ["0", "1e-6"]],
                "center_exit": ["0.333333333333333333e-6", "0.333333333333333333e-6"],
                "direction": ["0", "0", "1"], "amplitude": ["1", "0"], "opl": opl}
    record = S._records(sim, [triangle])[0]
    phase = Number(f"({sim.lines[0].k})*({opl})", 32)
    expected = complex(Number(f"exp(i*({phase}))", 32))
    got = complex(math.cos(record[-1]), math.sin(record[-1]))
    assert abs(got - expected) < 1e-12
    naive = float(sim.lines[0].k) * float(opl)
    assert abs(complex(math.cos(naive), math.sin(naive)) - expected) > 1e-3


def test_accumulator_state_roundtrip_and_truncation(tmp_path):
    path = tmp_path / "state.bin"
    state = {"n_modes": 3, "ref_index": 1, "sum": array("d", [1, 2, 3]).tobytes()}
    S._write_state(path, state)
    assert S._read_state(path) == state
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="Truncated"):
        S._read_state(path)


@pytest.mark.parametrize("extension", ["gz", "xz"])
def test_mesh_archive_keeps_decimal_strings_and_unresolved(tmp_path, extension):
    source = {"mode": 0, "origin": ["1e-30", "0", "-.1"], "weight": 1}
    mesh = {"stats": {"accepted": 1, "missing_area": "1e-40"},
            "triangles": [{"opl": "0.12345678901234567890123456789"}],
            "nodes": [{"point": ["1e-30", "0"]}],
            "unresolved": [{"reason": "history"}]}
    path = tmp_path / f"mesh.jsonl.{extension}"
    storage = S.save_mesh(path, source, mesh)
    saved_source, saved_mesh = S.load_mesh(path)
    assert saved_mesh.pop("archive_storage") == storage
    assert (saved_source, saved_mesh) == (source, mesh)
    assert storage["archive_nodes"] == "all"


@pytest.mark.parametrize("policy", ["none", "accepted", "all"])
def test_xz_mesh_preserves_exact_gzip_jsonl_bytes_and_mp_fields(tmp_path, policy):
    source = {"mode": 7, "origin": ["1e-48", "-2e-49", "-.675"], "weight": .0016}
    triangle = {"bore": 0, "center_node_id": 2, "probe_node_ids": [1, 2],
                "input": [["0", "0"], ["1e-6", "0"], ["0", "1e-6"]],
                "exit": [["0", "0"], ["1e-6", "0"], ["0", "1e-6"]],
                "center_exit": ["3.333333333333333333333333333e-7"] * 2,
                "direction": ["1e-25", "-2e-25", "1"],
                "amplitude": ["1.0000000000000000000000000001", "-1e-49"],
                "opl": ".1234567890123456789012345678901234567890123456789"}
    mesh = {"stats": {"nodes": 3}, "triangles": [triangle],
            "nodes": [{"bore": 0, "id": i, "opl": triangle["opl"]} for i in (1, 2, 3)],
            "unresolved": [{"bore": 0, "input": [["1e-48", "-2e-49"]], "reason": "history"}]}
    gzip_path, xz_path = tmp_path / "mesh.jsonl.gz", tmp_path / "mesh.jsonl.xz"
    S.save_mesh(gzip_path, source, mesh, node_policy=policy)
    S.save_mesh(xz_path, source, mesh, node_policy=policy)
    with gzip.open(gzip_path, "rb") as compressed, lzma.open(xz_path, "rb") as compact:
        assert compact.read() == compressed.read()
    assert S._mesh_header(xz_path) == S._mesh_header(gzip_path)
    assert list(S._mesh_rows(xz_path)) == list(S._mesh_rows(gzip_path))
    assert S.load_mesh(xz_path) == S.load_mesh(gzip_path)
    assert S.load_mesh(xz_path, load_nodes=False) == S.load_mesh(gzip_path, load_nodes=False)
    sim = scene()
    assert S._records(sim, S._mesh_records([xz_path], "triangle")) == S._records(sim, [triangle])
    assert not list(tmp_path.glob("*.tmp"))


def test_mesh_archive_compression_suffix_is_explicit(tmp_path):
    sim = scene()
    sim.cfg.raw["stage19"] = {"archive_compression": "xz"}
    assert S._mesh_suffix(S.validate(sim.cfg)) == ".jsonl.xz"
    assert S._mesh_suffix({}) == ".jsonl.gz"
    for unknown in (None, [], True, "zip", "../xz"):
        with pytest.raises(ValueError, match="archive_compression"):
            S._mesh_suffix({"archive_compression": unknown})
    with pytest.raises(ValueError, match="suffix"):
        S._mesh_open(tmp_path / "mesh.jsonl.zip", "wt")
    assert not (tmp_path / "mesh.jsonl.zip").exists()


@pytest.mark.parametrize("policy,expected", [("none", []), ("accepted", [(0, 1), (0, 2), (1, 1)]),
                                             ("all", [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2)])])
def test_mesh_storage_policy_changes_only_node_provenance(tmp_path, policy, expected):
    source = {"mode": 3, "origin": ["1e-50", "0", "-.675"], "weight": .25}
    mesh = {"stats": {"nodes": 5, "accepted_triangles": 2},
            "triangles": [{"bore": 0, "opl": "0.123456789012345678901234567890123456789",
                           "amplitude": ["1.000000000000000000000001", "-1e-40"],
                           "center_node_id": 2, "probe_node_ids": [1, 2, 1]},
                          {"bore": 1, "center_node_id": 1, "probe_node_ids": [1]}],
            "nodes": [{"bore": bore, "id": node, "opl": "0.12345678901234567890123456789"}
                      for bore, node in [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2)]],
            "unresolved": [{"bore": 0, "input": [["1e-50", "2e-50"]], "reason": "history"}]}
    before = json.dumps(mesh, sort_keys=True)
    path = tmp_path / "mesh.jsonl.gz"
    storage = S.save_mesh(path, source, mesh, node_policy=policy)
    saved_source, saved = S.load_mesh(path)
    assert saved_source == source
    for key in ("triangles", "unresolved", "stats"):
        assert saved[key] == mesh[key]
    assert [(row["bore"], row["id"]) for row in saved["nodes"]] == expected
    assert storage["node_records_saved"] == len(expected)
    assert storage["node_records_omitted"] == 5 - len(expected)
    assert "trace-cache" in storage["node_id_namespace"]
    assert json.dumps(mesh, sort_keys=True) == before
    _, skipped = S.load_mesh(path, load_nodes=False)
    assert skipped["nodes"] == []
    assert skipped["archive_storage"] == storage
    assert skipped["triangles"] == mesh["triangles"]


def test_mesh_storage_counts_nodes_not_retained_by_builder(tmp_path):
    mesh = {"stats": {"nodes": 900}, "triangles": [], "nodes": [], "unresolved": []}
    storage = S.save_mesh(tmp_path / "mesh.jsonl.gz", {}, mesh, node_policy="none")
    assert storage["node_records_traced"] == 900
    assert storage["node_records_available"] == storage["node_records_saved"] == 0
    assert storage["node_records_omitted"] == 900
    for policy in ([], "invalid", True):
        with pytest.raises(ValueError, match="node_policy"):
            S.save_mesh(tmp_path / "invalid.gz", {}, mesh, node_policy=policy)


def test_legacy_mesh_load_can_skip_nodes_without_changing_field(tmp_path):
    path = tmp_path / "legacy.jsonl.gz"
    rows = [{"header": S.FORMAT, "source": {"mode": 0}, "stats": {"nodes": 1}},
            {"kind": "triangle", "record": {"opl": "0.12345678901234567890123456789"}},
            {"kind": "node", "record": {"id": 3, "point": ["1e-30", "0"]}}]
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    _, mesh = S.load_mesh(path, load_nodes=False)
    assert mesh["nodes"] == []
    assert mesh["triangles"] == [rows[1]["record"]]
    assert mesh["archive_storage"]["archive_nodes"] == "legacy_unspecified"
    assert mesh["archive_storage"]["node_records_saved"] == 1


def test_mesh_archive_node_count_is_verified_even_when_nodes_are_skipped(tmp_path):
    path = tmp_path / "mismatch.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"header": S.FORMAT, "source": {}, "stats": {},
                                 "archive_storage": {"node_records_saved": 1}}) + "\n")
    with pytest.raises(ValueError, match="node count"):
        S.load_mesh(path, load_nodes=False)


@pytest.mark.parametrize("policy", ["none", "all"])
def test_source_task_releases_nodes_before_fields_and_resume(tmp_path, monkeypatch, policy):
    from formula.capsysred.stages import _s19_mesh
    sim = scene()
    options = S.validate(sim.cfg)
    options.update(archive_nodes=policy, observables=["point"])
    screens = S._screen_specs(sim, options)[:1]
    source = {"mode": 1, "origin": ["0", "0", "-.1"], "weight": .5}
    mesh = {"stats": {"nodes": 3, "bores": []}, "triangles": [{"opl": "1e-30"}],
            "nodes": [{"id": 1}, {"id": 2}, {"id": 3}], "unresolved": []}
    calls = []
    def build(*args, retain_nodes):
        calls.append(retain_nodes)
        if not retain_nodes:
            mesh["nodes"] = []
        return mesh
    def evaluate(*args):
        assert mesh["nodes"] == []
        assert mesh["triangles"] == mesh["unresolved"] == []
        return {"field": bytes(16 * 9)}
    monkeypatch.setattr(_s19_mesh, "build_source_mesh", build)
    monkeypatch.setattr(S, "_records", lambda sim, records: [])
    monkeypatch.setattr(S, "_evaluate_field", evaluate)
    target = tmp_path / "mode-000001"
    task = (sim.cfg.raw, options, screens, source, str(target))
    result = json.loads(Path(S._source_task(task)).read_text())
    assert calls == [policy != "none"]
    assert result["archive_storage"]["node_records_traced"] == 3
    assert result["archive_storage"]["node_records_saved"] == (3 if policy == "all" else 0)
    (target / "exit-point.bin").unlink()
    monkeypatch.setattr(S, "load_mesh", lambda *args, **kwargs: pytest.fail("resume must stream MP models"))
    assert json.loads(Path(S._source_task(task)).read_text())["archive_storage"] == result["archive_storage"]
    assert calls == [policy != "none"]


@pytest.mark.parametrize("compression,extension", [("gzip", "gz"), ("xz", "xz")])
def test_bore_checkpoint_resume_preserves_combined_fields_and_rejects_changed_contract(tmp_path, monkeypatch, compression, extension):
    from copy import deepcopy
    from formula.capsysred.stages import _s19_mesh
    sim = scene()
    raw = deepcopy(sim.cfg.raw)
    raw["capillary"]["bores"] = [{"center": [0, 0], "radius": 6e-6},
                                 {"center": [2e-5, 0], "radius": 6e-6}]
    sim = Simulation.from_dict(raw)
    options = S.validate(sim.cfg)
    options.update(observables=["point"], screen_indices=[0], archive_compression=compression)
    screens = S._screen_specs(sim, options)
    source = {"mode": 1, "origin": ["0", "0", "-.1"], "weight": .5}
    monkeypatch.setattr(S, "_code_hashes", lambda options=None: {"test": "stable"})
    templates = []
    for bore, amplitude in enumerate((["1", ".4"], ["-.2", ".7"])):
        triangle = {"bore": bore, "input": [["-1e-5", "-1e-5"], ["1e-5", "-1e-5"], ["0", "1e-5"]],
                    "exit": [["-1e-5", "-1e-5"], ["1e-5", "-1e-5"], ["0", "1e-5"]],
                    "center_exit": ["0", "0"], "direction": ["0", "0", "1"],
                    "opl": ".15", "amplitude": amplitude}
        templates.append({"triangles": [triangle], "nodes": [], "unresolved": [],
            "stats": {"nodes": 4, "accepted_triangles": 1, "unresolved_triangles": 0,
                      "retained_nodes": 0, "physics_pending_nodes": 0, "seconds": 2+bore,
                      "bores": [{"bore": bore, "accepted_fraction": "1", "aperture_deficit_fraction": "0"}]}})
    calls, interrupted, built_parts = [], [False], []
    def build(sim, origin, mesh_options, *, retain_nodes):
        bore = mesh_options["bores"][0]
        calls.append(bore)
        assert retain_nodes is False
        assert all(part["triangles"] == part["nodes"] == part["unresolved"] == [] for part in built_parts)
        if bore == 1 and not interrupted[0]:
            interrupted[0] = True
            raise RuntimeError("bore interrupted")
        part = deepcopy(templates[bore])
        built_parts.append(part)
        return part
    monkeypatch.setattr(_s19_mesh, "build_source_mesh", build)
    directory = tmp_path / "mode-000001"
    task = (raw, options, screens, source, str(directory))
    with pytest.raises(RuntimeError, match="bore interrupted"):
        S._source_task(task)
    assert (directory / f"mesh-bore-0000.jsonl.{extension}").exists()
    assert not (directory / f"mesh.jsonl.{extension}").exists()
    altered = deepcopy(options)
    altered["mesh"]["eps_A"] = ".04"
    with pytest.raises(ValueError, match="bore checkpoint differs"):
        S._source_task((raw, altered, screens, source, str(directory)))
    assert calls == [0, 1]
    result = json.loads(Path(S._source_task(task)).read_text())
    assert calls == [0, 1, 1]
    assert result["stats"]["nodes"] == 8
    assert result["stats"]["accepted_triangles"] == 2
    assert result["stats"]["seconds"] == 5
    assert not list(directory.glob("mesh-bore-*.jsonl.*"))
    saved_source, combined = S.load_mesh(directory / f"mesh.jsonl.{extension}")
    expected = [part["triangles"][0] for part in templates]
    assert saved_source == source and combined["triangles"] == expected
    expected_field = S._evaluate_field(S._records(sim, expected), float(sim.lines[0].k),
                                        screens[0], "point", options["kernel"])["field"]
    assert (directory / "exit-point.bin").read_bytes() == expected_field
    (directory / "exit-point.bin").unlink()
    monkeypatch.setattr(S, "load_mesh", lambda *args, **kwargs: pytest.fail("resume must stream models"))
    S._source_task(task)
    assert calls == [0, 1, 1]
    assert (directory / "exit-point.bin").read_bytes() == expected_field


@pytest.mark.parametrize("compression", ["gzip", "xz"])
def test_resume_after_failed_source_preserves_native_moments(tmp_path, monkeypatch, compression):
    pytest.importorskip("formula._stage19")
    sim = scene()
    path = archive(tmp_path, sim)
    sim.cfg.raw["stage19"] = {"source_archive": str(path), "archive_compression": compression}
    monkeypatch.setattr(S, "_code_hashes", lambda options=None: {"test": "stable"})
    calls, fail = [], [True]
    def source_task(task):
        _, options, screens, source, target = task
        mode = source["mode"]
        calls.append(mode)
        if mode == 1 and fail[0]:
            fail[0] = False
            raise RuntimeError("source interrupted")
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        fields = []
        values = [complex(1, mode)] * 9
        values[0] = 1 if mode == 0 else 1j
        values[4] = 2 if mode == 0 else 1+1j
        data = array("d", [v for z in values for v in (z.real, z.imag)]).tobytes()
        for screen in screens:
            for observable in options["observables"]:
                name = f"{screen['name']}-{observable}"
                S._write_binary(target / (name + ".bin"), data)
                fields.append({"name": name, "file": name + ".bin", "sha256": S.sha256(data).hexdigest()})
        completed = target / "result.json"
        S._dump(completed, {"mode": mode, "fields": fields, "stats": {}, "seconds": 0})
        return str(completed)
    monkeypatch.setattr(S, "_source_task", source_task)
    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match="source interrupted"):
        S.run_stage19(sim, output, log=lambda value: None)
    checkpoint = json.loads((output / "stage19/checkpoint.json").read_text())
    assert checkpoint["n_modes"] == 1
    result = S.run_stage19(sim, output, log=lambda value: None)
    assert result["results"]["n_modes"] == 2
    assert result["results"]["archive_compression"] == compression
    with gzip.open(output / "stage19/exit-point/coherence.jsonl.gz", "rt") as handle:
        first = json.loads(next(handle))
    assert first["I"] == 1
    assert first["W"] == [1.5, .5]
    assert first["abs_mu"] == pytest.approx(math.sqrt(2.5/3))
    assert calls == [0, 1, 1]
    S.run_stage19(sim, output, log=lambda value: None)
    assert calls == [0, 1, 1]
    assert not list((output / "stage19/modes").rglob("*.bin"))
    sim.cfg.raw["stage19"]["archive_compression"] = "xz" if compression == "gzip" else "gzip"
    with pytest.raises(ValueError, match="different configuration"):
        S.run_stage19(sim, output, log=lambda value: None)


def test_mesh_coverage_gate_never_discards_a_source():
    options = S.validate(scene().cfg)
    with pytest.raises(ValueError, match="no accepted"):
        S._mesh_gate({"triangles": [], "stats": {"bores": []}}, options)
    mesh = {"triangles": [{}], "stats": {"bores": [{"bore": 0, "accepted_fraction": "0.94"}]}}
    with pytest.raises(ValueError, match="below"):
        S._mesh_gate(mesh, options)
    options["minimum_accepted_fraction"] = .9
    S._mesh_gate(mesh, options)


def test_physical_exit_requires_valid_finite_domain_evidence():
    from copy import deepcopy
    options = S.validate(scene().cfg)
    options["mesh"] = {"phase_coordinates": "physical_exit", "domain_relative": ".02"}
    valid = {"passed": True, "weighted_domain_relative": ".001", "tolerance": ".02",
             "normalizer_sum_abs_A_exit_area": "1e-9", "exit_vertex_disagreements": 0,
             "counts": {"internal_same_branch": 2},
             "p6_fan_checks": {"non_simple": 0, "bad_fan_orientation": 0}}
    mesh = {"triangles": [{}], "stats": {"bores": [
        {"bore": 0, "accepted_fraction": ".99", "domain_indicator": valid}]}}
    S._mesh_gate(mesh, options)
    for change in ({"passed": False}, {"passed": 1}, {"weighted_domain_relative": "NaN"},
                   {"weighted_domain_relative": "Infinity"}, {"weighted_domain_relative": ".03"},
                   {"weighted_domain_relative": "-.01"}, {"normalizer_sum_abs_A_exit_area": "0"},
                   {"tolerance": "NaN"}, {"tolerance": ".04"}, {"exit_vertex_disagreements": 1},
                   {"counts": {"nonmanifold": 1}}, {"p6_fan_checks": {"non_simple": 1, "bad_fan_orientation": 0}}):
        altered = deepcopy(mesh)
        altered["stats"]["bores"][0]["domain_indicator"].update(change)
        with pytest.raises(ValueError, match="domain"):
            S._mesh_gate(altered, options)
    for missing in (None, {}, {"passed": True}):
        altered = deepcopy(mesh)
        altered["stats"]["bores"][0]["domain_indicator"] = missing
        with pytest.raises(ValueError, match="domain"):
            S._mesh_gate(altered, options)
    mesh["stats"]["bores"][0]["accepted_fraction"] = "NaN"
    with pytest.raises(ValueError, match="accepted fraction"):
        S._mesh_gate(mesh, options)


def test_physical_exit_hash_includes_domain_implementation():
    options = S.validate(scene().cfg)
    assert "_s19_domain.py" not in S._code_hashes(options)
    options["mesh"]["phase_coordinates"] = "physical_exit"
    assert "_s19_domain.py" in S._code_hashes(options)


@pytest.mark.parametrize("interruption", ["source", "preview"])
def test_previews_keep_prefix_moments_and_resume_without_changing_final(tmp_path, monkeypatch, interruption):
    pytest.importorskip("formula._stage19")
    sim = scene(3)
    path = archive(tmp_path, sim)
    sim.cfg.raw["stage19"] = {"source_archive": str(path), "preview_modes": [2],
                             "observables": ["point"], "screen_indices": [0]}
    monkeypatch.setattr(S, "_code_hashes", lambda options=None: {"test": "stable"})
    samples = [(2+1j, 1+0j), (-.5+2j, .3-.4j), (1-3j, 2+.2j)]
    source_calls, exports, failed = [], [], [False]

    def source_task(task):
        _, options, screens, source, target = task
        mode = source["mode"]
        source_calls.append(mode)
        if interruption == "source" and mode == 2 and not failed[0]:
            failed[0] = True
            raise RuntimeError("source interrupted")
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        pixel, reference = samples[mode]
        values = [pixel] * 9
        values[4] = reference
        data = array("d", [v for z in values for v in (z.real, z.imag)]).tobytes()
        S._write_binary(target / "exit-point.bin", data)
        completed = target / "result.json"
        S._dump(completed, {"mode": mode, "fields": [{"name": "exit-point", "file": "exit-point.bin",
                     "sha256": S.sha256(data).hexdigest()}], "seconds": 0,
                     "stats": {"bores": [{"bore": 0, "accepted_fraction": ".99",
                                          "aperture_deficit_fraction": ".001"}]}})
        return str(completed)

    original_export = S._export_map
    def export(*args, **kwargs):
        exports.append(kwargs.get("target_modes"))
        result = original_export(*args, **kwargs)
        if interruption == "preview" and kwargs.get("target_modes") == 3 and not failed[0]:
            failed[0] = True
            raise RuntimeError("preview interrupted")
        return result
    monkeypatch.setattr(S, "_source_task", source_task)
    monkeypatch.setattr(S, "_export_map", export)
    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match=interruption + " interrupted"):
        S.run_stage19(sim, output, log=lambda value: None)
    directory = output / "stage19"
    assert json.loads((directory / "checkpoint.json").read_text())["n_modes"] == 2
    assert not (directory / "meta.json").exists()
    preview = directory / "previews/modes-000002"
    assert preview.exists() == (interruption == "source")
    result = S.run_stage19(sim, output, log=lambda value: None)
    assert result["results"]["n_modes"] == 3
    metadata = json.loads((preview / "meta.json").read_text())
    assert metadata["status"] == "preliminary"
    assert metadata["n_modes"] == 2 and metadata["target_modes"] == 3
    assert metadata["ensemble_complete"] is False
    assert metadata["mesh_coverage"]["accepted_fraction_by_bore"]["0"]["sources"] == 2
    assert metadata["maps"][0]["status"] == "preliminary"
    for target, count in ((preview, 2), (directory, 3)):
        with gzip.open(target / "exit-point/coherence.jsonl.gz", "rt") as handle:
            actual = json.loads(next(handle))
        intensity = sum(abs(p)**2 for p, r in samples[:count]) / count
        reference = sum(abs(r)**2 for p, r in samples[:count]) / count
        mutual = sum(p*r.conjugate() for p, r in samples[:count]) / count
        assert actual["I"] == pytest.approx(intensity)
        assert actual["W"] == pytest.approx([mutual.real, mutual.imag])
        assert actual["abs_mu"] == pytest.approx(abs(mutual) / math.sqrt(intensity*reference))
    assert source_calls == ([0, 1, 2, 2] if interruption == "source" else [0, 1, 2])
    assert exports.count(3) == (1 if interruption == "source" else 2)
    assert not list((directory / "modes").rglob("*.bin"))


def test_failed_mesh_keeps_partition_diagnostic_before_field_work(tmp_path, monkeypatch):
    from formula.capsysred.stages import _s19_mesh
    sim = scene(1)
    options = S.validate(sim.cfg)
    screens = S._screen_specs(sim, options)
    source = {"mode": 0, "origin": ["0", "0", "-.1"], "weight": 1.}
    mesh = {"stats": {"nodes": 0, "bores": [{"bore": 0, "accepted_fraction": "0"}]},
            "triangles": [], "nodes": [], "unresolved": [
                {"bore": 0, "input": [["0", "0"], ["1e-6", "0"], ["0", "1e-6"]],
                 "reason": "node_budget"}]}
    monkeypatch.setattr(_s19_mesh, "build_source_mesh", lambda *args, **kwargs: mesh)
    monkeypatch.setattr(S, "_evaluate_field", lambda *args: pytest.fail("unaccepted mesh must not be integrated"))
    directory = tmp_path / "mode-000000"
    with pytest.raises(ValueError, match="no accepted"):
        S._source_task((sim.cfg.raw, options, screens, source, str(directory)))
    assert (directory / "mesh-bore-0000.jsonl.gz").is_file()
    assert not (directory / "mesh.jsonl.gz").exists()
    assert "#D55E00" in (directory / "input-triangles.svg").read_text()
    assert "PARTIAL SOURCE" in (directory / "input-triangles.svg").read_text()
    status = json.loads((directory / "incomplete.json").read_text())
    assert status["fields_computed"] is False
    assert status["status"] == "mesh_acceptance_failed"
    assert not (directory / "result.json").exists()


def test_screen_global_phase_preserves_same_plane_unequal_mode_moments():
    fields = [(1+2j, 3-1j), (.2-.7j, -2+4j)]
    phases = (.3, -2.1)
    def moments(rows):
        return (sum(p*r.conjugate() for p, r in rows),
                sum(abs(p)**2 for p, _ in rows), sum(abs(r)**2 for _, r in rows))
    rotated = [(p*complex(math.cos(t), math.sin(t)), r*complex(math.cos(t), math.sin(t)))
               for (p, r), t in zip(fields, phases)]
    assert moments(rotated) == pytest.approx(moments(fields), rel=1e-14)


def test_stage19_run_lock_released_on_exception(tmp_path):
    with S._run_lock(tmp_path):
        with pytest.raises(ValueError, match="locked"):
            with S._run_lock(tmp_path):
                pass
    with pytest.raises(RuntimeError):
        with S._run_lock(tmp_path):
            raise RuntimeError("interrupted")
    with S._run_lock(tmp_path):
        pass


@pytest.mark.parametrize("policy", ["none", "accepted", "all"])
@pytest.mark.parametrize("input_extension,output_extension", [("gz", "gz"), ("xz", "xz"), ("gz", "xz"), ("xz", "gz")])
def test_streaming_bore_assembly_preserves_mp_rows_nodes_and_svg(tmp_path, policy, input_extension, output_extension):
    parts, triangles, unresolved, expected_nodes = [], [], [], []
    for bore in range(2):
        triangle = {"bore": bore, "center_node_id": 1,
                    "input": [["0", "0"], ["0.000001234567890123456789", "0"], ["0", ".000002"]],
                    "amplitude": [str(bore+1), ".12345678901234567890123456789"],
                    "opl": "0.123456789012345678901234567890123456789", "diagnostics": {"phase": "1e-48"}}
        missing = {"bore": bore, "input": [["0", "0"], ["-.000001", "0"], ["0", "-.000002"]],
                   "reason": "history", "probe": "0.000000000000000000000001"}
        nodes = [{"bore": bore, "id": 1, "opl": triangle["opl"]}, {"bore": bore, "id": 2, "unused": True}]
        part = {"triangles": [triangle], "unresolved": [missing], "nodes": nodes,
                "stats": {"nodes": 2, "retained_nodes": 2, "bores": [{"bore": bore}],
                          "options": {"bores": [bore], "precision": 64}}}
        path = tmp_path / f"bore-{bore}.{input_extension}"
        S.save_mesh(path, {"source": {"mode": 0}, "bore": bore}, part, node_policy=policy)
        parts.append((path, S._mesh_header(path)))
        triangles.append(triangle)
        unresolved.append(missing)
        expected_nodes.extend(nodes if policy == "all" else nodes[:1] if policy == "accepted" else [])
    combined = tmp_path / f"mesh.{output_extension}"
    header = S._combine_bore_archives(combined, {"mode": 0}, parts, {"phase_coordinates": "physical_exit"})
    source, loaded = S.load_mesh(combined)
    assert source == {"mode": 0}
    assert loaded["triangles"] == triangles and loaded["unresolved"] == unresolved
    assert loaded["nodes"] == expected_nodes
    assert header["stats"]["retained_nodes"] == 0
    assert header["stats"]["peak_bore_retained_nodes"] == 2
    assert header["stats"]["accepted_triangles"] == header["stats"]["unresolved_triangles"] == 2
    assert "bores" not in header["stats"]["options"]
    assert header["archive_storage"]["node_records_traced"] == 4
    assert header["archive_storage"]["node_records_saved"] == len(expected_nodes)
    assert list(S._mesh_records([combined], "triangle")) == triangles
    spec = S._screen_specs(scene(), S.validate(scene().cfg))[0]
    S.mesh_svg(tmp_path / "expected.svg", triangles, spec, 0, unresolved)
    S.mesh_svg(tmp_path / "streamed.svg", S._mesh_records([combined], "triangle"), spec, 0,
               S._mesh_records([combined], "unresolved"))
    assert (tmp_path / "expected.svg").read_bytes() == (tmp_path / "streamed.svg").read_bytes()


def test_streaming_bore_assembly_detects_truncation_before_publication(tmp_path):
    path = tmp_path / "bore.gz"
    mesh = {"triangles": [{"opl": "1"}], "nodes": [], "unresolved": [],
            "stats": {"nodes": 0, "bores": [{"bore": 0}]}}
    S.save_mesh(path, {}, mesh, node_policy="none")
    header = S._mesh_header(path)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(header) + "\n")
    combined = tmp_path / "mesh.gz"
    with pytest.raises(ValueError, match="triangle count"):
        S._combine_bore_archives(combined, {}, [(path, header)], {})
    assert not combined.exists()


def test_failed_bore_is_saved_and_stops_before_next_bore_or_field(tmp_path, monkeypatch):
    from copy import deepcopy
    from formula.capsysred.stages import _s19_mesh
    raw = deepcopy(scene().cfg.raw)
    raw["capillary"]["bores"] *= 2
    sim = Simulation.from_dict(raw)
    options = S.validate(sim.cfg)
    source = {"mode": 0, "origin": ["0", "0", "-.1"], "weight": .5}
    calls = []
    def build(sim, origin, opts, **kwargs):
        calls.append(opts["bores"][0])
        return {"triangles": [], "nodes": [], "unresolved": [],
                "stats": {"nodes": 3, "bores": [{"bore": 0, "accepted_fraction": ".5"}]}}
    monkeypatch.setattr(_s19_mesh, "build_source_mesh", build)
    monkeypatch.setattr(S, "_evaluate_field", lambda *args: pytest.fail("failed bore reached field kernel"))
    monkeypatch.setattr(S, "_records", lambda *args: pytest.fail("failed bore reached native packing"))
    target = tmp_path / "mode-0"
    with pytest.raises(ValueError, match="no accepted"):
        S._source_task((raw, options, S._screen_specs(sim, options), source, str(target)))
    assert calls == [0]
    assert (target / "mesh-bore-0000.jsonl.gz").exists()
    status = json.loads((target / "incomplete.json").read_text())
    assert status["saved_bores"] == [0] and status["missing_bores"] == [1]
    assert status["failed_bore"] == 0 and not status["source_partition_complete"]
    assert not (target / "mesh.jsonl.gz").exists()
    assert "PARTIAL SOURCE" in (target / "input-triangles.svg").read_text()

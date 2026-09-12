"""Equal yaml seed => bit-identical rays and bit-identical Stage-14 results,
for every wall kind; the source lattice is independent of the optic."""

from __future__ import annotations

import json
import os

import pytest
import yaml

from formula.capsysred import Simulation, rays_v3
from formula.capsysred.surfaces import entrance_disk
from formula.capsysred.trace_v3 import trace as trace_v3

# One bore per wall kind; cylinder, torus, funnel and implicit share the
# exact entrance aim radius 3 um, so their source-side draws coincide.
BORES = {
    "cylinder": {"center": [0.0, 0.0], "radius": 3.0e-6},
    # gentle bend: sag over z1 ~ 0.2 um < bore, some rays never bounce
    "torus": {"center": [0.0, 0.0], "radius": 3.0e-6,
              "bend": {"radius": 234.375, "toward": [1.0, 0.0]}},
    "polygon": {"center": [0.0, 0.0], "radius": 3.0e-6, "sides": 6},
    "revolution": {"center": [0.0, 0.0], "r2_poly": [9.0e-12, -1.0e-10]},
    "funnel": {"center": [0.0, 0.0], "radius": 3.0e-6,
               "funnel": {"g": [0.0, 0.0], "f": [-6.0, 0.0]}},
    "implicit": {"center": [0.0, 0.0], "surface": "x^2+y^2-9",
                 "aim_radius": 3.0e-6, "engine_method": "sturm"},
}
BUDGETS = {"implicit": (2, 5)}          # engine hits are expensive
DEFAULT_BUDGET = (3, 12)


def _raw(kind, seed=4242):
    n_modes, n_rays = BUDGETS.get(kind, DEFAULT_BUDGET)
    return {
        "seed": seed,
        "capillary": {
            "source": {"shape": "disk", "size": 1.0e-6,
                       "position": [0.0, 0.0, -0.01],
                       "n_modes": n_modes, "n_rays": n_rays},
            "screen": {"z": 0.011, "nx": 5, "ny": 5,
                       "edge_x": 1.2e-5, "edge_y": 1.2e-5},
            "bores": [BORES[kind]], "z0": 0.0, "z1": 0.01,
        },
        "trace": {"lean_rays": True},
    }


def _trace(raw, cfg_path, archive):
    with open(cfg_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(raw, fh, sort_keys=False)
    trace_v3(str(cfg_path), str(archive), jobs=1, level=6, log=lambda m: None)
    return str(archive)


def _tree_bytes(archive) -> dict:
    """Every file of the archive, path -> bytes."""
    out = {}
    for root, _, names in os.walk(archive):
        for name in names:
            path = os.path.join(root, name)
            with open(path, "rb") as fh:
                out[os.path.relpath(path, archive)] = fh.read()
    return out


def _rows(archive) -> dict:
    index = rays_v3.load_index(archive)
    return {(json.loads(l)["mode"], json.loads(l)["ray"]): l
            for l in rays_v3.scene_lines(archive, index, "capillary")}


def _stage14(raw, archive, out):
    sim = Simulation.from_dict(raw)
    sim.replay(str(archive), str(out), stages=[14])
    result = sim.results["stage14:capillary"]
    with open(os.path.join(out, "stage14", "mu-jack.jsonl"), "rb") as fh:
        return result, fh.read()


def _payload_hashes(result):
    return [cache.meta["files"]["mode-rows.f64"]["sha256"]
            for cache in result["cache_parts"]]


@pytest.fixture(scope="module")
def traced(tmp_path_factory):
    root = tmp_path_factory.mktemp("arch")
    return {kind: _trace(_raw(kind), root / f"{kind}.yaml", root / kind)
            for kind in BORES}


@pytest.mark.parametrize("kind", list(BORES))
def test_retrace_is_bit_identical(traced, tmp_path, kind):
    raw = _raw(kind)
    again = _trace(raw, tmp_path / "cfg.yaml", tmp_path / "again")
    first_tree, again_tree = _tree_bytes(traced[kind]), _tree_bytes(again)
    assert sorted(first_tree) == sorted(again_tree)
    assert first_tree == again_tree                     # rays byte for byte
    # Stage 14 on the twin archives: identical estimator rows and payloads.
    res_a, mu_a = _stage14(raw, traced[kind], tmp_path / "s14-a")
    res_b, mu_b = _stage14(raw, again, tmp_path / "s14-b")
    assert mu_a == mu_b and mu_a
    assert _payload_hashes(res_a) == _payload_hashes(res_b)
    assert res_a["stats"] == res_b["stats"]


def test_other_seed_changes_rays(traced, tmp_path):
    other = _trace(_raw("cylinder", seed=4243), tmp_path / "cfg.yaml",
                   tmp_path / "other")
    assert rays_v3.index_digest(other) != rays_v3.index_digest(traced["cylinder"])


def test_stage14_cache_replay_is_bit_identical(traced, tmp_path):
    raw = _raw("torus")
    res_a, mu_a = _stage14(raw, traced["torus"], tmp_path / "s14-a")
    res_b, mu_b = _stage14(raw, traced["torus"], tmp_path / "s14-b")
    # The cache lives beside the archive; the second run must reuse it and
    # still publish byte-identical results.
    assert res_b["cache_hits"] == 1
    assert mu_a == mu_b
    assert _payload_hashes(res_a) == _payload_hashes(res_b)


def test_source_lattice_is_independent_of_the_surface(traced):
    # Mode origins depend on (seed, mode) alone: every wall kind that shares
    # the mode count sees the very same origins.
    origins = {kind: rays_v3.origins(archive, rays_v3.load_index(archive),
                                     "capillary")
               for kind, archive in traced.items()}
    n_modes = DEFAULT_BUDGET[0]
    full = {k: o for k, o in origins.items() if len(o) == n_modes}
    assert len(set(map(json.dumps, full.values()))) == 1
    assert origins["implicit"] == full["cylinder"][: BUDGETS["implicit"][0]]

    # Rays that never touch a wall are bit-identical across kinds with the
    # same entrance aim disk: the lattice rng is keyed by (seed, mode) only.
    z0 = 0.0
    disk = {kind: entrance_disk(
        Simulation.from_dict(_raw(kind)).cfg.capillary.bores[0], z0)
        for kind in BORES}
    same_aim = [k for k in ("cylinder", "torus", "funnel", "implicit")
                if disk[k] == disk["cylinder"]]
    assert set(same_aim) >= {"cylinder", "torus", "funnel"}
    assert disk["polygon"] != disk["cylinder"]          # apothem vs aim disk
    rows = {kind: _rows(traced[kind]) for kind in same_aim}
    base = rows["cylinder"]
    unbounced = {key for key, line in base.items()
                 if b'"sins": []' in line and b'"fate": "screen"' in line}
    assert unbounced
    for kind in same_aim[1:]:
        shared = [key for key in unbounced
                  if b'"sins": []' in rows[kind].get(key, b"")]
        assert shared, f"{kind}: no unbounced rays shared with the cylinder"
        for key in shared:
            assert rows[kind][key] == base[key], (kind, key)

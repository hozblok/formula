"""Replay of stages 2, 7 and 8 onto another screen: arrivals re-project from
the recorded plane and re-bin on the target grid; union parts share a plane."""

from __future__ import annotations

import pytest
import yaml

from formula.capsysred import Simulation
from formula.capsysred.trace_v3 import trace as trace_v3

Z_SOURCE, Z_REC, Z_FAR = -0.08, 0.06, 0.09
SCALE = (Z_FAR - Z_SOURCE) / (Z_REC - Z_SOURCE)      # straight rays from a point source
EDGE_X, EDGE_Y = 1.2e-5, 2.0e-6
RESULT = {2: "free", 7: "alt:free", 8: "sketch:free"}


def _raw(z=Z_REC, nx=21, edge_x=EDGE_X, edge_y=EDGE_Y, seed=3, n_modes=4):
    return {
        "seed": seed,
        "screen": {"z": z, "nx": nx, "ny": 1, "edge_x": edge_x, "edge_y": edge_y,
                   "center": [0.0, 0.0]},
        "free": {"source": {"shape": "point", "size": 0.0,
                            "position": [0.0, 0.0, Z_SOURCE],
                            "n_modes": n_modes, "n_rays": 150}},
    }


def _record(tmp_path, raw, name="rec"):
    cfg = tmp_path / f"{name}.yaml"
    with open(cfg, "w", encoding="utf-8") as fh:
        yaml.safe_dump(raw, fh, sort_keys=False)
    archive = str(tmp_path / name / "rays-modes")
    trace_v3(str(cfg), archive, jobs=1, level=6, log=lambda m: None)
    return archive


def _flat(density):
    return [v for row in density for v in row] if isinstance(density[0], list) else list(density)


def _replay(raw, archive, out, stage):
    sim = Simulation.from_dict(raw)
    sim.replay(archive, str(out), stages=[stage])
    return sim.results[RESULT[stage]]


@pytest.mark.parametrize("stage", [2, 7, 8])
def test_replay_reprojects_free_arrivals_to_the_target_plane(tmp_path, stage):
    """A point source sends straight rays: on a farther plane with the window
    scaled by the same factor every ray lands in the same pixel, so counts and
    density equal those of the recording plane; an unscaled window loses rays."""
    archive = _record(tmp_path, _raw())
    a = _replay(_raw(), archive, tmp_path / "at-plane", stage)
    b = _replay(_raw(z=Z_FAR, edge_x=EDGE_X * SCALE, edge_y=EDGE_Y * SCALE), archive,
                tmp_path / "far-scaled", stage)
    assert b["stats"] == a["stats"]
    assert b["maps"]["density"] == a["maps"]["density"]
    assert sum(_flat(a["maps"]["density"])) == a["stats"]["screen"]
    assert 0 < a["stats"]["screen"] < 600          # the aim pads the window by 2 %
    c = _replay(_raw(z=Z_FAR), archive, tmp_path / "far-unscaled", stage)
    assert c["stats"]["off_window"] > a["stats"]["off_window"]
    assert sum(_flat(c["maps"]["density"])) == c["stats"]["screen"] < a["stats"]["screen"]


@pytest.mark.parametrize("stage", [2, 7, 8])
def test_replay_on_a_smaller_grid_rebins_the_arrivals(tmp_path, stage):
    archive = _record(tmp_path, _raw(nx=21))
    res = _replay(_raw(nx=5), archive, tmp_path / "small", stage)
    density = _flat(res["maps"]["density"])
    assert len(density) == 5 and sum(density) == res["stats"]["screen"] > 0
    assert res["stats"]["screen"] + res["stats"]["off_window"] == 600


def test_union_parts_must_share_the_recorded_plane(tmp_path):
    a = _record(tmp_path, _raw(seed=3), "a")
    far = _record(tmp_path, _raw(seed=4, z=Z_FAR), "far")
    union = Simulation.from_dict(_raw(n_modes=8))
    with pytest.raises(ValueError, match="different planes"):
        union.replay([a, far], str(tmp_path / "union-far"), stages=[2])
    same = _record(tmp_path, _raw(seed=4), "same")
    union.replay([a, same], str(tmp_path / "union"), stages=[2])
    assert union.results["free"]["stats"]["emitted"] == 8 * 150

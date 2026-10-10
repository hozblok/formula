"""Stage 17 configuration stays isolated until explicitly selected."""

import copy

import pytest

from formula.capsysred.config import B5_DEFAULTS, Config
from formula.capsysred.simulation import KNOWN_STAGES, Simulation


def _raw():
    return {
        "precision": 32,
        "capillary": {
            "source": {"shape": "point", "size": 0, "position": [0, 0, -0.1],
                       "n_modes": 1, "n_rays": 1},
            "screen": {"z": 0.2},
            "screens": [{"z": 0.4}],
        },
    }


def test_b5_defaults_are_lazy_and_do_not_change_raw_config():
    cfg = Config(_raw())
    before = copy.deepcopy(cfg.raw)
    assert "b5_estimator" not in cfg.raw
    assert cfg.validate_b5_estimator() == B5_DEFAULTS
    assert cfg.raw == before
    assert 17 in KNOWN_STAGES
    assert len(cfg.capillary.screens) == 1
    assert float(cfg.capillary.screen.z) == 0.2
    assert float(cfg.capillary.screens[0].z) == 0.4


def test_b5_custom_options_and_lazy_unknown_keys():
    raw = _raw()
    raw["b5_estimator"] = {"max_modes": 2, "degree": 5, "neighbors": 48,
                           "min_neighbors": 24, "screen_index": 1, "seed": 0,
                           "patch_space": "screen", "charts": "screen"}
    cfg = Config(raw)
    result = cfg.validate_b5_estimator()
    assert result == {**B5_DEFAULTS, **raw["b5_estimator"]}
    raw["b5_estimator"]["unknown"] = 1
    cfg = Config(raw)
    assert cfg.raw["b5_estimator"]["unknown"] == 1
    with pytest.raises(ValueError, match="b5_estimator has unknown keys"):
        cfg.validate_b5_estimator()


@pytest.mark.parametrize("bad", [
    [], "archive_phase", {"provider": "wave"}, {"provider": None},
    {"patch_space": "mixed"}, {"patch_space": None}, {"charts": "pp"}, {"charts": True},
    {"max_modes": 0}, {"max_modes": True}, {"rays_per_mode": 0}, {"rays_per_mode": 2.5},
    {"degree": 1}, {"degree": 6}, {"degree": 4.0}, {"degree": True},
    {"neighbors": 23}, {"min_neighbors": 23}, {"neighbors": 24, "min_neighbors": 48},
    {"reflection": "perfect"}, {"reflection": None}, {"reflection": True},
    {"patches_per_family": 0}, {"phase_tolerance": 0}, {"phase_tolerance": float("inf")},
    {"phase_tolerance": True}, {"max_condition": -1}, {"max_condition": float("nan")},
    {"screen_index": -1}, {"screen_index": 0.5}, {"seed": -1}, {"seed": False},
])
def test_b5_invalid_configuration(bad):
    raw = _raw()
    raw["b5_estimator"] = bad
    cfg = Config(raw)
    with pytest.raises(ValueError, match="b5_estimator"):
        cfg.validate_b5_estimator()


@pytest.mark.parametrize("bad", [
    {"widths_m": []}, {"widths_m": [0]}, {"widths_m": [float("nan")]},
    {"widths_m": [True]}, {"widths_m": [1e-6, 1e-6]}, {"map_jobs": 0},
    {"map_stride": 0}, {"map_ray_budgets": [0]}, {"map_ray_budgets": [20001]},
    {"map_ray_budgets": [1, 1]}, {"map_snapshots": [5]}, {"map_snapshots": True},
])
def test_canonical_invalid_configuration(bad):
    raw = _raw()
    raw["b5_estimator"] = {"provider": "archive_canonical", **bad}
    with pytest.raises(ValueError, match="b5_estimator"):
        Config(raw).validate_b5_estimator()


def test_b5_section_does_not_change_recording_contract_or_default_stages():
    from formula.capsysred.rays import geometry_metadata, sidecar_metadata

    base = Config(_raw())
    raw = _raw()
    raw["b5_estimator"] = {"degree": 5, "unknown": "ignored by other stages"}
    other = Config(raw)
    assert geometry_metadata(base) == geometry_metadata(other)
    assert sidecar_metadata(base) == sidecar_metadata(other)
    assert Simulation(base)._default_stages() == {1, 14}
    assert Simulation(other)._default_stages() == {1, 14}


def test_stage17_requires_a_capillary_before_loading_backend(tmp_path):
    raw = {"free": {"source": {"shape": "point", "size": 0,
                                "position": [0, 0, -0.1], "n_modes": 1, "n_rays": 1}}}
    with pytest.raises(ValueError, match="capillary.source"):
        Simulation.from_dict(raw).run(str(tmp_path / "result"), stages=[17])
    assert not (tmp_path / "result").exists()


def test_reflection_option_defaults_to_fresnel_and_accepts_ideal():
    assert B5_DEFAULTS["reflection"] == "fresnel"
    raw = _raw()
    raw["b5_estimator"] = {"reflection": "ideal_minus_one"}
    assert Config(raw).validate_b5_estimator()["reflection"] == "ideal_minus_one"

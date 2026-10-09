"""Stage 18 validates its options lazily and preserves recording identity."""

import copy

import pytest

from formula.capsysred.config import B9_DEFAULTS, Config
from formula.capsysred.simulation import KNOWN_STAGES, Simulation


def _raw():
    return {"precision": 32, "capillary": {
        "source": {"shape": "point", "size": 0, "position": [0, 0, -.1],
                   "n_modes": 1, "n_rays": 1}, "screen": {"z": .2}}}


def test_b9_defaults_are_lazy_and_do_not_change_raw_config():
    cfg = Config(_raw())
    before = copy.deepcopy(cfg.raw)
    assert "b9_estimator" not in cfg.raw
    assert cfg.validate_b9_estimator() == B9_DEFAULTS
    assert cfg.raw == before
    assert cfg.b9["carrier_groups"] == 0
    assert cfg.b9["max_triangles_per_batch"] == 40000
    assert 18 in KNOWN_STAGES


def test_b9_custom_options_and_lazy_unknown_keys():
    raw = _raw()
    raw["b9_estimator"] = {
        "amplitude_mode": "tube_flux", "carrier_groups": 32,
        "mode_start": 2, "max_modes": 8, "rays_per_mode": 40000,
        "map_ray_budgets": [10000, 40000], "map_snapshots": [4, 8],
        "phase_subdivisions": [1, 2, 4], "holdout_stride": 0,
        "determinant_floor": 0, "pixel_order": 8, "edge_order": 16,
    }
    assert Config(raw).validate_b9_estimator() == {**B9_DEFAULTS, **raw["b9_estimator"]}
    raw["b9_estimator"]["unknown"] = 1
    cfg = Config(raw)
    assert cfg.raw["b9_estimator"]["unknown"] == 1
    with pytest.raises(ValueError, match="b9_estimator has unknown keys"):
        cfg.validate_b9_estimator()


@pytest.mark.parametrize("bad", [
    [], "archive_contour", {"provider": "wave"}, {"provider": None},
    {"amplitude_mode": "fitted"}, {"amplitude_mode": None}, {"amplitude_mode": True},
    {"mode_start": -1}, {"mode_start": True}, {"max_modes": 0}, {"max_modes": True},
    {"rays_per_mode": 0}, {"rays_per_mode": 2}, {"rays_per_mode": 2.5}, {"screen_index": -1},
    {"map_stride": 0}, {"map_jobs": 1.5}, {"edge_order": 0}, {"edge_order": 1}, {"pixel_order": False},
    {"nufft_threads": -1}, {"holdout_stride": 1}, {"holdout_stride": -1},
    {"max_triangles_per_batch": 0}, {"max_triangles_per_batch": True},
    {"carrier_groups": -1}, {"carrier_groups": True}, {"carrier_groups": 1.5}, {"carrier_groups": "32"},
    {"nufft_eps": 0}, {"nufft_eps": 1}, {"nufft_eps": float("nan")},
    {"nufft_eps": True}, {"determinant_floor": -1}, {"determinant_floor": float("inf")},
    {"phase_subdivisions": []}, {"phase_subdivisions": [0]}, {"phase_subdivisions": [True]},
    {"phase_subdivisions": [1, 1]}, {"phase_subdivisions": [1.5]},
    {"map_ray_budgets": [20001]}, {"map_ray_budgets": [1, 1]}, {"map_ray_budgets": [2]},
    {"map_snapshots": [5]}, {"map_snapshots": True}, {"map_snapshots": [0]},
])
def test_b9_invalid_configuration(bad):
    raw = _raw()
    raw["b9_estimator"] = bad
    with pytest.raises(ValueError, match="b9_estimator"):
        Config(raw).validate_b9_estimator()


def test_b9_options_do_not_change_recording_contract_or_default_stages():
    from formula.capsysred.rays import geometry_metadata, sidecar_metadata

    base = Config(_raw())
    raw = _raw()
    raw["b9_estimator"] = {"phase_subdivisions": [4], "unknown": "unused"}
    other = Config(raw)
    assert geometry_metadata(base) == geometry_metadata(other)
    assert sidecar_metadata(base) == sidecar_metadata(other)
    assert Simulation(base)._default_stages() == Simulation(other)._default_stages() == {1, 14}

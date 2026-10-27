"""Focused configuration-contract tests for opt-in Stage 14."""

import math

import pytest

from formula.capsysred.config import load


DEFAULT_THRESHOLDS = {
    "ic_n_sigma": 3.0,
    "ref_ic_n_sigma": 3.0,
    "w_n_sigma": 3.0,
    "min_coherent_fraction": 0.05,
}


def test_stage14_threshold_defaults_have_exact_public_keys():
    cfg = load({})
    assert cfg.stage14_flag_thresholds == DEFAULT_THRESHOLDS
    assert set(cfg.raw["stage14"]) == {"flag_thresholds"}
    assert set(cfg.raw["stage14"]["flag_thresholds"]) == set(DEFAULT_THRESHOLDS)


def test_stage14_partial_threshold_override_merges_with_defaults():
    cfg = load({
        "stage14": {
            "flag_thresholds": {
                "ic_n_sigma": 4.5,
                "min_coherent_fraction": 1.0,
            },
        },
    })
    assert cfg.stage14_flag_thresholds == {
        "ic_n_sigma": 4.5,
        "ref_ic_n_sigma": 3.0,
        "w_n_sigma": 3.0,
        "min_coherent_fraction": 1.0,
    }


@pytest.mark.parametrize(
    "raw",
    [
        {"stage14": None},
        {"stage14": []},
        {"stage14": {"unknown": 1}},
        {"stage14": {"flag_thresholds": None}},
        {"stage14": {"flag_thresholds": []}},
        {"stage14": {"flag_thresholds": {"unknown": 1}}},
        {"stage14": {"flag_thresholds": {"ic_n_sigma": 0}}},
        {"stage14": {"flag_thresholds": {"ic_n_sigma": -1}}},
        {"stage14": {"flag_thresholds": {"ic_n_sigma": True}}},
        {"stage14": {"flag_thresholds": {"ref_ic_n_sigma": math.inf}}},
        {"stage14": {"flag_thresholds": {"w_n_sigma": math.nan}}},
        {"stage14": {"flag_thresholds": {"min_coherent_fraction": 0}}},
        {"stage14": {"flag_thresholds": {"min_coherent_fraction": -0.1}}},
        {"stage14": {"flag_thresholds": {"min_coherent_fraction": 1.0001}}},
        {"stage14": {"flag_thresholds": {"min_coherent_fraction": math.nan}}},
    ],
)
def test_stage14_invalid_schema_or_thresholds_fail_closed(raw):
    with pytest.raises((TypeError, ValueError)):
        load(raw)


def test_root_flag_thresholds_is_rejected_without_alias_or_fallback():
    with pytest.raises(ValueError, match="top-level flag_thresholds"):
        load({"flag_thresholds": DEFAULT_THRESHOLDS})

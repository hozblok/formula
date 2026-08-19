"""Normative unit tests for the pure Stage 14 flag taxonomy."""

from dataclasses import replace
import json
import math

import pytest

from formula.capsysred.stages.stage14_flags import (
    PIXEL_FLAGS,
    FlagThresholds,
    PixelCounters,
    PixelStatistics,
    Stage14InvariantError,
    classify_pixel,
    serialize_pixel,
    validate_counters,
    validate_pixel_row,
    validate_ref,
    w_signal_status,
)


THRESHOLDS = FlagThresholds()
N_JK = 4


def _counters(
    *,
    n_rays: int = 8,
    m_realizations: int = 4,
    m_pair_realizations: int = 3,
    m_ref_realizations: int = 2,
    max_rays_per_realization: int = 3,
) -> PixelCounters:
    return PixelCounters(
        n_rays=n_rays,
        m_realizations=m_realizations,
        m_pair_realizations=m_pair_realizations,
        m_ref_realizations=m_ref_realizations,
        max_rays_per_realization=max_rays_per_realization,
    )


def _stats(**changes: object) -> PixelStatistics:
    values: dict[str, object] = {
        "I": 100.0,
        "counters": _counters(),
        "ic": 20.0,
        "ic_err": 1.0,
        "w_abs": 10.0,
        "w_err": 2.0,
        "mu_raw": 0.5,
        "mu_raw_err": 0.2,
        "n_mu_loo_valid": N_JK,
    }
    values.update(changes)
    return PixelStatistics(**values)  # type: ignore[arg-type]


def _classify(stats: PixelStatistics, **changes: object) -> str | None:
    args: dict[str, object] = {
        "is_reference": False,
        "ref_status": "ok",
        "n_jackknife_units": N_JK,
        "thresholds": THRESHOLDS,
    }
    args.update(changes)
    return classify_pixel(stats, **args)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("expected", "stats"),
    [
        (
            "no-rays",
            _stats(
                I=0.0,
                counters=_counters(
                    n_rays=0,
                    m_realizations=0,
                    m_pair_realizations=0,
                    m_ref_realizations=0,
                    max_rays_per_realization=0,
                ),
                ic=None,
                ic_err=None,
                w_abs=None,
                w_err=None,
                mu_raw=None,
                mu_raw_err=None,
                n_mu_loo_valid=None,
            ),
        ),
        (
            "solo-rays-only",
            _stats(
                counters=_counters(
                    n_rays=2,
                    m_realizations=2,
                    m_pair_realizations=0,
                    m_ref_realizations=1,
                    max_rays_per_realization=1,
                ),
                ic=None,
                ic_err=None,
                mu_raw=None,
                mu_raw_err=None,
                n_mu_loo_valid=None,
            ),
        ),
        ("negative-Ic", _stats(ic=-4.0, ic_err=1.0, mu_raw=None,
                                mu_raw_err=None, n_mu_loo_valid=None)),
        ("null-Ic", _stats(ic=-1.0, ic_err=1.0, mu_raw=None,
                            mu_raw_err=None, n_mu_loo_valid=None)),
        ("noisy-Ic", _stats(ic=1.0, ic_err=2.0, mu_raw=None,
                             mu_raw_err=None, n_mu_loo_valid=None)),
        (
            "no-ref-realizations",
            _stats(
                counters=_counters(m_ref_realizations=0),
                w_abs=None,
                w_err=None,
                mu_raw=None,
                mu_raw_err=None,
                n_mu_loo_valid=None,
            ),
        ),
        ("over-mu", _stats(mu_raw=1.2)),
        ("noisy-mu", _stats(mu_raw_err=1.01)),
        ("trusted", _stats()),
    ],
)
def test_all_nine_pixel_classes(expected: str, stats: PixelStatistics):
    assert _classify(stats) == expected


def test_class_test_is_exhaustive():
    assert set(PIXEL_FLAGS) == {
        "no-rays",
        "solo-rays-only",
        "negative-Ic",
        "null-Ic",
        "noisy-Ic",
        "no-ref-realizations",
        "over-mu",
        "noisy-mu",
        "trusted",
    }


def test_reference_pixel_and_bad_reference_are_null_not_extra_classes():
    assert _classify(_stats(), is_reference=True) is None
    assert _classify(_stats(), ref_status="weak") is None
    # Reference-independent diagnoses remain available with a bad reference.
    assert _classify(_stats(ic=-4.0, ic_err=1.0), ref_status="weak") == "negative-Ic"
    with pytest.raises(Stage14InvariantError, match="forbids"):
        _classify(_stats(), ref_status="jackknife-unavailable")


@pytest.mark.parametrize(
    ("stats", "expected"),
    [
        # U == 0 is not negative-Ic.
        (_stats(ic=-3.0, ic_err=1.0), "null-Ic"),
        # L == 0 remains in the L <= 0 branch.
        (_stats(ic=3.0, ic_err=1.0), "noisy-Ic"),
        # U_star == delta goes to noisy-Ic, not null-Ic.
        (_stats(I=60.0, ic=-1.0, ic_err=1.0), "noisy-Ic"),
        # mu_raw == 1 is already over-mu.
        (_stats(mu_raw=1.0), "over-mu"),
        # mu_raw_err == 1 remains trusted.
        (_stats(mu_raw_err=1.0), "trusted"),
    ],
)
def test_normative_comparison_boundaries(stats: PixelStatistics, expected: str):
    assert _classify(stats) == expected


def test_strict_full_loo_semantics():
    incomplete = _stats(n_mu_loo_valid=N_JK - 1, mu_raw_err=None)
    assert _classify(incomplete) == "noisy-mu"
    assert _classify(replace(incomplete, mu_raw=1.1)) == "over-mu"

    with pytest.raises(Stage14InvariantError, match="incomplete LOO"):
        _classify(_stats(n_mu_loo_valid=N_JK - 1, mu_raw_err=0.1))
    with pytest.raises(Stage14InvariantError, match="mu_raw_err"):
        _classify(_stats(n_mu_loo_valid=N_JK, mu_raw_err=None))
    with pytest.raises(Stage14InvariantError, match="n_mu_loo_valid"):
        _classify(_stats(n_mu_loo_valid=N_JK + 1))


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"m_pair_realizations": 0, "ic_ref": math.nan,
          "ic_ref_err": math.nan}, "no-pairs-at-ref"),
        ({"ic_ref": math.nan}, "numeric-invalid"),
        ({"ic_ref_err": -1.0}, "numeric-invalid"),
        ({"ic_ref": 0.0}, "ic-nonpositive"),
        ({"ic_ref": -1.0}, "ic-nonpositive"),
        ({"ic_ref": 3.0, "ic_ref_err": 1.0}, "weak"),
        ({"ic_ref_loo": [2.0, 0.0, 2.0, 2.0]}, "loo-invalid"),
        ({"ic_ref_loo": [2.0, math.inf, 2.0, 2.0]}, "loo-invalid"),
        ({}, "ok"),
    ],
)
def test_validate_ref_exact_status_order(kwargs: dict[str, object], expected: str):
    args: dict[str, object] = {
        "m_pair_realizations": 2,
        "ic_ref": 10.0,
        "ic_ref_err": 1.0,
        "ic_ref_loo": [8.0, 9.0, 9.5, 8.5],
        "n_jackknife_units": N_JK,
        "thresholds": THRESHOLDS,
    }
    args.update(kwargs)
    assert validate_ref(**args) == expected  # type: ignore[arg-type]


def test_validate_ref_rejects_missing_loo_row_as_structural_error():
    with pytest.raises(Stage14InvariantError, match="row count"):
        validate_ref(
            m_pair_realizations=2,
            ic_ref=10.0,
            ic_ref_err=1.0,
            ic_ref_loo=[8.0, 9.0, 9.5],
            n_jackknife_units=N_JK,
            thresholds=THRESHOLDS,
        )


@pytest.mark.parametrize(
    "counters",
    [
        _counters(n_rays=True),
        _counters(n_rays=-1),
        _counters(m_realizations=2, m_pair_realizations=3),
        _counters(n_rays=3, m_realizations=4),
        _counters(m_realizations=2, m_pair_realizations=2,
                  m_ref_realizations=3),
        _counters(n_rays=2, m_realizations=2, m_pair_realizations=2,
                  max_rays_per_realization=3),
        _counters(n_rays=0, m_realizations=1, m_pair_realizations=0,
                  m_ref_realizations=0, max_rays_per_realization=0),
        _counters(n_rays=2, m_realizations=0, m_pair_realizations=0,
                  m_ref_realizations=0, max_rays_per_realization=1),
        _counters(m_pair_realizations=0, max_rays_per_realization=2),
        _counters(m_pair_realizations=1, max_rays_per_realization=1),
    ],
)
def test_invalid_counter_lattice_is_a_hard_error(counters: PixelCounters):
    with pytest.raises(Stage14InvariantError):
        validate_counters(counters)


@pytest.mark.parametrize(
    "changes",
    [
        {"I": math.nan},
        {"ic": math.inf},
        {"ic_err": -1.0},
        {"w_abs": math.nan},
        {"mu_raw": math.inf},
        {"mu_raw_err": -0.1},
    ],
)
def test_nonfinite_or_negative_required_numbers_are_hard_errors(
    changes: dict[str, object],
):
    with pytest.raises(Stage14InvariantError):
        _classify(_stats(**changes))


def test_serializer_emits_exact_schema_and_json_safe_values():
    row = serialize_pixel(
        _stats(),
        pixel=17,
        x_um=1.25,
        y_um=-2.5,
        is_reference=False,
        ref_status="ok",
        n_jackknife_units=N_JK,
        thresholds=THRESHOLDS,
    )
    assert row["stage_id"] == 14
    assert row["screen"] == "capillary"
    assert row["flag"] == "trusted"
    assert "mu" not in row and "dubious" not in row and "solid" not in row
    json.dumps(row, allow_nan=False)
    validate_pixel_row(row, THRESHOLDS)


def test_serializer_enforces_null_semantics():
    no_rays = _stats(
        I=0.0,
        counters=_counters(
            n_rays=0,
            m_realizations=0,
            m_pair_realizations=0,
            m_ref_realizations=0,
            max_rays_per_realization=0,
        ),
        ic=None,
        ic_err=None,
        w_abs=None,
        w_err=None,
        mu_raw=None,
        mu_raw_err=None,
        n_mu_loo_valid=None,
    )
    row = serialize_pixel(
        no_rays,
        pixel=0,
        x_um=0.0,
        y_um=0.0,
        is_reference=False,
        ref_status="ok",
        n_jackknife_units=N_JK,
        thresholds=THRESHOLDS,
    )
    assert row["flag"] == "no-rays"
    assert all(row[name] is None for name in
               ("ic", "ic_err", "w_abs", "w_err", "mu_raw",
                "mu_raw_err", "n_mu_loo_valid"))

    with pytest.raises(Stage14InvariantError, match="no-rays"):
        serialize_pixel(
            replace(no_rays, w_abs=0.0, w_err=0.0),
            pixel=0,
            x_um=0.0,
            y_um=0.0,
            is_reference=False,
            ref_status="ok",
            n_jackknife_units=N_JK,
            thresholds=THRESHOLDS,
        )


def test_serializer_keeps_ref_and_bad_ref_mu_fields_null():
    no_mu = replace(
        _stats(), mu_raw=None, mu_raw_err=None, n_mu_loo_valid=None
    )
    reference_stats = replace(
        no_mu,
        counters=replace(no_mu.counters, m_ref_realizations=no_mu.counters.m_realizations),
    )
    reference = serialize_pixel(
        reference_stats,
        pixel=3,
        x_um=0.0,
        y_um=0.0,
        is_reference=True,
        ref_status="ok",
        n_jackknife_units=N_JK,
        thresholds=THRESHOLDS,
    )
    assert reference["flag"] is None and reference["is_reference"] is True

    bad_ref = serialize_pixel(
        no_mu,
        pixel=4,
        x_um=1.0,
        y_um=0.0,
        is_reference=False,
        ref_status="weak",
        n_jackknife_units=N_JK,
        thresholds=THRESHOLDS,
    )
    assert bad_ref["flag"] is None


def test_reference_still_obeys_raw_field_missingness():
    empty = _stats(
        I=0.0,
        counters=_counters(n_rays=0, m_realizations=0,
                           m_pair_realizations=0, m_ref_realizations=0,
                           max_rays_per_realization=0),
        ic=None, ic_err=None, w_abs=None, w_err=None,
        mu_raw=None, mu_raw_err=None, n_mu_loo_valid=None,
    )
    serialize_pixel(empty, pixel=0, x_um=0.0, y_um=0.0,
                    is_reference=True, ref_status="no-pairs-at-ref",
                    n_jackknife_units=N_JK, thresholds=THRESHOLDS)
    with pytest.raises(Stage14InvariantError, match="no-rays"):
        serialize_pixel(replace(empty, I=1.0), pixel=0, x_um=0.0, y_um=0.0,
                        is_reference=True, ref_status="no-pairs-at-ref",
                        n_jackknife_units=N_JK, thresholds=THRESHOLDS)

    paired_without_ic = replace(
        _stats(), ic=None, ic_err=None, mu_raw=None, mu_raw_err=None,
        n_mu_loo_valid=None)
    with pytest.raises(Stage14InvariantError, match="require ic"):
        serialize_pixel(paired_without_ic, pixel=0, x_um=0.0, y_um=0.0,
                        is_reference=True, ref_status="ok",
                        n_jackknife_units=N_JK, thresholds=THRESHOLDS)


def test_w_availability_matches_shared_realization_counter():
    no_shared = replace(
        _stats(),
        counters=replace(_stats().counters, m_ref_realizations=0),
        w_abs=None, w_err=None, mu_raw=None, mu_raw_err=None,
        n_mu_loo_valid=None)
    serialize_pixel(no_shared, pixel=1, x_um=0.0, y_um=0.0,
                    is_reference=False, ref_status="ok",
                    n_jackknife_units=N_JK, thresholds=THRESHOLDS)
    with pytest.raises(Stage14InvariantError, match="availability"):
        serialize_pixel(replace(no_shared, w_abs=0.0, w_err=0.0),
                        pixel=1, x_um=0.0, y_um=0.0,
                        is_reference=False, ref_status="ok",
                        n_jackknife_units=N_JK, thresholds=THRESHOLDS)


def test_serializer_rejects_half_w_pair_and_nan_coordinate():
    with pytest.raises(Stage14InvariantError, match="w_abs and w_err"):
        serialize_pixel(
            _stats(w_err=None),
            pixel=2,
            x_um=0.0,
            y_um=0.0,
            is_reference=False,
            ref_status="ok",
            n_jackknife_units=N_JK,
            thresholds=THRESHOLDS,
        )

    row = serialize_pixel(
        _stats(),
        pixel=2,
        x_um=0.0,
        y_um=0.0,
        is_reference=False,
        ref_status="ok",
        n_jackknife_units=N_JK,
        thresholds=THRESHOLDS,
    )
    row["x_um"] = math.nan
    with pytest.raises(Stage14InvariantError, match="x_um"):
        validate_pixel_row(row, THRESHOLDS)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ic_n_sigma": 0.0},
        {"ref_ic_n_sigma": math.inf},
        {"w_n_sigma": math.nan},
        {"min_coherent_fraction": 0.0},
        {"min_coherent_fraction": 1.01},
    ],
)
def test_threshold_validation(kwargs: dict[str, float]):
    with pytest.raises(Stage14InvariantError):
        FlagThresholds(**kwargs)


def test_channel_b_is_independent_and_uses_inclusive_threshold():
    assert w_signal_status(None, None, THRESHOLDS) == "unknown"
    assert w_signal_status(1.0, 0.0, THRESHOLDS) == "unknown"
    assert w_signal_status(6.0, 2.0, THRESHOLDS) == "detected"
    assert w_signal_status(5.9, 2.0, THRESHOLDS) == "not-detected"

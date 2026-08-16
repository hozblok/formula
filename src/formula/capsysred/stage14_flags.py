"""Pure Stage 14 pixel taxonomy and serialization invariants.

This module deliberately contains no estimator or rendering code.  It is the
single small, testable implementation of the normative Stage 14 decision tree:
invalid structure raises :class:`Stage14InvariantError`, while statistically
unusable data is represented by an exact ``ref_status`` or pixel flag.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Literal, TypeAlias, cast


PixelFlag: TypeAlias = Literal[
    "no-rays",
    "solo-rays-only",
    "negative-Ic",
    "null-Ic",
    "noisy-Ic",
    "no-ref-realizations",
    "over-mu",
    "noisy-mu",
    "trusted",
]

RefStatus: TypeAlias = Literal[
    "ok",
    "jackknife-unavailable",
    "no-pairs-at-ref",
    "numeric-invalid",
    "ic-nonpositive",
    "weak",
    "loo-invalid",
]

WSignalStatus: TypeAlias = Literal["detected", "not-detected", "unknown"]

PIXEL_FLAGS: tuple[PixelFlag, ...] = (
    "no-rays",
    "solo-rays-only",
    "negative-Ic",
    "null-Ic",
    "noisy-Ic",
    "no-ref-realizations",
    "over-mu",
    "noisy-mu",
    "trusted",
)

REF_STATUSES: tuple[RefStatus, ...] = (
    "ok",
    "jackknife-unavailable",
    "no-pairs-at-ref",
    "numeric-invalid",
    "ic-nonpositive",
    "weak",
    "loo-invalid",
)


class Stage14InvariantError(ValueError):
    """A hard Stage 14 data/configuration error, not a scientific status."""


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_integer(name: str, value: object, *, minimum: int | None = None) -> int:
    if not _is_integer(value):
        raise Stage14InvariantError(f"{name} must be an integer")
    result = cast(int, value)
    if minimum is not None and result < minimum:
        raise Stage14InvariantError(f"{name} must be >= {minimum}")
    return result


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _require_finite(name: str, value: object) -> float:
    if not _is_finite_number(value):
        raise Stage14InvariantError(f"{name} must be a finite number")
    return float(cast(float, value))


def _require_nonnegative(name: str, value: object) -> float:
    result = _require_finite(name, value)
    if result < 0.0:
        raise Stage14InvariantError(f"{name} must be >= 0")
    return result


def _require_optional_finite(name: str, value: object | None) -> float | None:
    if value is None:
        return None
    return _require_finite(name, value)


def _require_optional_nonnegative(name: str, value: object | None) -> float | None:
    if value is None:
        return None
    return _require_nonnegative(name, value)


def _validate_n_jk(value: object) -> int:
    return _require_integer("n_jackknife_units", value, minimum=2)


@dataclass(frozen=True, slots=True)
class FlagThresholds:
    """Pre-registered thresholds of the Stage 14 estimator."""

    z: float = 3.0
    z_ref: float = 3.0
    z_w: float = 3.0
    f_min: float = 0.05

    def __post_init__(self) -> None:
        z = _require_finite("z", self.z)
        z_ref = _require_finite("z_ref", self.z_ref)
        z_w = _require_finite("z_w", self.z_w)
        f_min = _require_finite("f_min", self.f_min)
        if z <= 0.0:
            raise Stage14InvariantError("z must be > 0")
        if z_ref <= 0.0:
            raise Stage14InvariantError("z_ref must be > 0")
        if z_w <= 0.0:
            raise Stage14InvariantError("z_w must be > 0")
        if not 0.0 < f_min <= 1.0:
            raise Stage14InvariantError("f_min must satisfy 0 < f_min <= 1")


@dataclass(frozen=True, slots=True)
class PixelCounters:
    """Exact per-pixel counters accumulated before classification."""

    n_rays: int
    m_realizations: int
    m_pair_realizations: int
    m_ref_realizations: int
    max_rays_per_realization: int


@dataclass(frozen=True, slots=True)
class PixelStatistics:
    """Raw Stage 14 fields for one pixel; unavailable values are ``None``."""

    I: float
    counters: PixelCounters
    ic: float | None = None
    ic_err: float | None = None
    w_abs: float | None = None
    w_err: float | None = None
    mu_raw: float | None = None
    mu_raw_err: float | None = None
    n_mu_loo_valid: int | None = None


def validate_counters(counters: PixelCounters) -> None:
    """Validate the normative counter lattice for one pixel.

    The checks are intentionally independent of the scientific classifier so
    callers can preflight every pixel, including the reference pixel, first.
    """

    n_rays = _require_integer("n_rays", counters.n_rays, minimum=0)
    m_realizations = _require_integer(
        "m_realizations", counters.m_realizations, minimum=0
    )
    m_pair = _require_integer(
        "m_pair_realizations", counters.m_pair_realizations, minimum=0
    )
    m_ref = _require_integer(
        "m_ref_realizations", counters.m_ref_realizations, minimum=0
    )
    max_rays = _require_integer(
        "max_rays_per_realization",
        counters.max_rays_per_realization,
        minimum=0,
    )

    if not m_pair <= m_realizations <= n_rays:
        raise Stage14InvariantError(
            "counters must satisfy m_pair_realizations <= "
            "m_realizations <= n_rays"
        )
    if m_ref > m_realizations:
        raise Stage14InvariantError(
            "m_ref_realizations must be <= m_realizations"
        )
    if max_rays > n_rays:
        raise Stage14InvariantError(
            "max_rays_per_realization must be <= n_rays"
        )

    if n_rays == 0:
        if m_realizations != 0 or max_rays != 0:
            raise Stage14InvariantError(
                "an empty pixel must have zero realizations and zero maximum"
            )
    elif m_realizations < 1 or max_rays < 1:
        raise Stage14InvariantError(
            "a lit pixel must have a realization and a positive maximum"
        )

    if m_pair == 0 and max_rays > 1:
        raise Stage14InvariantError(
            "m_pair_realizations == 0 requires max_rays_per_realization <= 1"
        )
    if m_pair > 0 and max_rays < 2:
        raise Stage14InvariantError(
            "m_pair_realizations > 0 requires max_rays_per_realization >= 2"
        )


def validate_ref(
    *,
    m_pair_realizations: int,
    ic_ref: float | None,
    ic_ref_err: float | None,
    ic_ref_loo: Sequence[float],
    n_jackknife_units: int,
    thresholds: FlagThresholds,
) -> RefStatus:
    """Return the exact reference status in normative failure order.

    Structural problems (bad counts or a missing LOO entry) are hard errors;
    an invalid *numeric estimate* is instead the corresponding scientific
    reference status.
    """

    n_jk = _validate_n_jk(n_jackknife_units)
    m_pair = _require_integer(
        "m_pair_realizations(ref)", m_pair_realizations, minimum=0
    )
    if len(ic_ref_loo) != n_jk:
        raise Stage14InvariantError(
            "reference LOO row count must equal n_jackknife_units"
        )

    if m_pair == 0:
        return "no-pairs-at-ref"
    if (
        not _is_finite_number(ic_ref)
        or not _is_finite_number(ic_ref_err)
        or cast(float, ic_ref_err) < 0.0
    ):
        return "numeric-invalid"

    ic = float(cast(float, ic_ref))
    err = float(cast(float, ic_ref_err))
    if ic <= 0.0:
        return "ic-nonpositive"
    if ic - thresholds.z_ref * err <= 0.0:
        return "weak"
    if any(not _is_finite_number(value) or value <= 0.0 for value in ic_ref_loo):
        return "loo-invalid"
    return "ok"


def classify_pixel(
    stats: PixelStatistics,
    *,
    is_reference: bool,
    ref_status: RefStatus,
    n_jackknife_units: int,
    thresholds: FlagThresholds,
    jackknife_computed: bool = True,
) -> PixelFlag | None:
    """Classify one pixel using the normative first-match decision tree."""

    if not isinstance(jackknife_computed, bool):
        raise Stage14InvariantError("jackknife_computed must be a bool")
    if ref_status not in REF_STATUSES:
        raise Stage14InvariantError(f"unknown ref_status: {ref_status!r}")
    if not jackknife_computed:
        if ref_status != "jackknife-unavailable":
            raise Stage14InvariantError(
                "jackknife_computed=false requires jackknife-unavailable"
            )
        return None
    if ref_status == "jackknife-unavailable":
        raise Stage14InvariantError(
            "jackknife_computed=true forbids jackknife-unavailable"
        )

    n_jk = _validate_n_jk(n_jackknife_units)
    if not isinstance(is_reference, bool):
        raise Stage14InvariantError("is_reference must be a bool")
    validate_counters(stats.counters)

    if is_reference:
        return None

    counters = stats.counters
    if counters.n_rays == 0:
        return "no-rays"
    if counters.m_pair_realizations == 0:
        return "solo-rays-only"

    intensity = _require_finite("I", stats.I)
    ic = _require_finite("ic", stats.ic)
    ic_err = _require_nonnegative("ic_err", stats.ic_err)
    if intensity <= 0.0:
        raise Stage14InvariantError("I must be > 0 when pairs exist")

    lower = ic - thresholds.z * ic_err
    upper = ic + thresholds.z * ic_err
    upper_star = max(ic, 0.0) + thresholds.z * ic_err
    delta = thresholds.f_min * intensity

    if upper < 0.0:
        return "negative-Ic"
    if lower <= 0.0:
        if upper_star < delta:
            return "null-Ic"
        return "noisy-Ic"

    if ref_status != "ok":
        return None
    if counters.m_ref_realizations == 0:
        return "no-ref-realizations"

    _require_nonnegative("w_abs", stats.w_abs)
    mu_raw = _require_nonnegative("mu_raw", stats.mu_raw)
    n_valid = _require_integer(
        "n_mu_loo_valid", stats.n_mu_loo_valid, minimum=0
    )
    if n_valid > n_jk:
        raise Stage14InvariantError(
            "n_mu_loo_valid must be <= n_jackknife_units"
        )

    full_loo = n_valid == n_jk
    if full_loo:
        mu_raw_err = _require_nonnegative("mu_raw_err", stats.mu_raw_err)
    else:
        if stats.mu_raw_err is not None:
            raise Stage14InvariantError(
                "incomplete LOO requires mu_raw_err to be null"
            )
        mu_raw_err = None

    if mu_raw >= 1.0:
        return "over-mu"
    if not full_loo or cast(float, mu_raw_err) > 1.0:
        return "noisy-mu"
    return "trusted"


def validate_serializer_invariants(
    stats: PixelStatistics,
    *,
    flag: PixelFlag | None,
    is_reference: bool,
    ref_status: RefStatus,
    n_jackknife_units: int | None,
    thresholds: FlagThresholds,
    jackknife_computed: bool,
) -> None:
    """Validate numeric and strict-null invariants before JSON publication."""

    if flag is not None and flag not in PIXEL_FLAGS:
        raise Stage14InvariantError(f"unknown pixel flag: {flag!r}")
    if ref_status not in REF_STATUSES:
        raise Stage14InvariantError(f"unknown ref_status: {ref_status!r}")
    if not isinstance(is_reference, bool):
        raise Stage14InvariantError("is_reference must be a bool")
    if not isinstance(jackknife_computed, bool):
        raise Stage14InvariantError("jackknife_computed must be a bool")

    validate_counters(stats.counters)
    intensity = _require_nonnegative("I", stats.I)
    ic = _require_optional_finite("ic", stats.ic)
    ic_err = _require_optional_nonnegative("ic_err", stats.ic_err)
    w_abs = _require_optional_nonnegative("w_abs", stats.w_abs)
    w_err = _require_optional_nonnegative("w_err", stats.w_err)
    mu_raw = _require_optional_nonnegative("mu_raw", stats.mu_raw)
    mu_raw_err = _require_optional_nonnegative("mu_raw_err", stats.mu_raw_err)

    if (ic is None) != (ic_err is None):
        raise Stage14InvariantError("ic and ic_err must be null together")
    if (w_abs is None) != (w_err is None):
        raise Stage14InvariantError("w_abs and w_err must be null together")
    if mu_raw is None and (mu_raw_err is not None or stats.n_mu_loo_valid is not None):
        raise Stage14InvariantError(
            "null mu_raw requires null mu_raw_err and n_mu_loo_valid"
        )

    if stats.n_mu_loo_valid is not None:
        n_valid = _require_integer(
            "n_mu_loo_valid", stats.n_mu_loo_valid, minimum=0
        )
    else:
        n_valid = None

    if not jackknife_computed:
        if ref_status != "jackknife-unavailable":
            raise Stage14InvariantError(
                "jackknife_computed=false requires jackknife-unavailable"
            )
        if n_jackknife_units is not None:
            raise Stage14InvariantError(
                "uncomputed jackknife requires null n_jackknife_units"
            )
        if flag is not None:
            raise Stage14InvariantError(
                "uncomputed jackknife requires a null flag"
            )
        if any(value is not None for value in (ic_err, w_err, mu_raw_err, n_valid)):
            raise Stage14InvariantError(
                "uncomputed jackknife requires null error and LOO fields"
            )
        return

    if ref_status == "jackknife-unavailable":
        raise Stage14InvariantError(
            "jackknife_computed=true forbids jackknife-unavailable"
        )

    n_jk = _validate_n_jk(n_jackknife_units)
    if n_valid is not None and n_valid > n_jk:
        raise Stage14InvariantError(
            "n_mu_loo_valid must be <= n_jackknife_units"
        )

    counters = stats.counters
    # The reference is excluded only from the nonlinear μ classifier.  The
    # same availability contract for I/Ic/W still applies to its raw fields.
    if counters.n_rays == 0:
        if intensity != 0.0 or any(
            value is not None for value in (
                ic, ic_err, w_abs, w_err, mu_raw, mu_raw_err,
                stats.n_mu_loo_valid)
        ):
            raise Stage14InvariantError(
                "no-rays requires I == 0 and all derived fields to be null"
            )
    elif counters.m_pair_realizations == 0:
        if any(value is not None for value in
               (ic, ic_err, mu_raw, mu_raw_err, stats.n_mu_loo_valid)):
            raise Stage14InvariantError(
                "solo-rays-only requires null ic, mu and LOO fields"
            )
    elif ic is None or ic_err is None:
        raise Stage14InvariantError(
            "pixels with pair realizations require ic and ic_err"
        )

    has_shared_realization = counters.m_ref_realizations > 0
    if has_shared_realization != (w_abs is not None):
        raise Stage14InvariantError(
            "w_abs/w_err availability must match m_ref_realizations"
        )

    if is_reference:
        if counters.m_ref_realizations != counters.m_realizations:
            raise Stage14InvariantError(
                "the reference pixel requires m_ref_realizations == "
                "m_realizations"
            )
        if flag is not None:
            raise Stage14InvariantError("the reference pixel must have flag=null")
        if any(
            value is not None
            for value in (mu_raw, mu_raw_err, stats.n_mu_loo_valid)
        ):
            raise Stage14InvariantError(
                "the reference pixel must not publish mu or LOO fields"
            )
        return

    expected = classify_pixel(
        stats,
        is_reference=False,
        ref_status=ref_status,
        n_jackknife_units=n_jk,
        thresholds=thresholds,
    )
    if flag != expected:
        raise Stage14InvariantError(
            f"flag {flag!r} does not match classifier result {expected!r}"
        )

    if flag in {"no-rays", "solo-rays-only"}:
        # Their common raw-field invariants were checked above.  Neither
        # category enters the nonlinear-mu branch.
        pass
    elif flag in {"negative-Ic", "null-Ic", "noisy-Ic"}:
        if any(
            value is not None
            for value in (mu_raw, mu_raw_err, stats.n_mu_loo_valid)
        ):
            raise Stage14InvariantError(
                f"{flag} requires null mu and LOO fields"
            )
    elif flag == "no-ref-realizations":
        if any(
            value is not None
            for value in (
                w_abs,
                w_err,
                mu_raw,
                mu_raw_err,
                stats.n_mu_loo_valid,
            )
        ):
            raise Stage14InvariantError(
                "no-ref-realizations requires null w, mu and LOO fields"
            )
    elif flag is None:
        if ref_status == "ok":
            raise Stage14InvariantError(
                "a non-reference computed pixel may be null only for a bad ref"
            )
        if any(
            value is not None
            for value in (mu_raw, mu_raw_err, stats.n_mu_loo_valid)
        ):
            raise Stage14InvariantError(
                "a ref-inapplicable pixel requires null mu and LOO fields"
            )
    else:
        if w_abs is None or w_err is None or mu_raw is None or n_valid is None:
            raise Stage14InvariantError(
                "the mu branch requires w, mu_raw and n_mu_loo_valid"
            )


_ROW_KEYS = frozenset(
    {
        "stage_id",
        "screen",
        "pixel",
        "is_reference",
        "ref_status",
        "jackknife_computed",
        "n_jackknife_units",
        "jackknife_unit",
        "x_um",
        "y_um",
        "I",
        "n_rays",
        "mu_raw",
        "mu_raw_err",
        "ic",
        "ic_err",
        "w_abs",
        "w_err",
        "m_realizations",
        "m_pair_realizations",
        "m_ref_realizations",
        "max_rays_per_realization",
        "n_mu_loo_valid",
        "flag",
    }
)


def validate_pixel_row(
    row: Mapping[str, object], thresholds: FlagThresholds
) -> None:
    """Validate a complete Stage 14 ``mu-jack.jsonl`` pixel object."""

    missing = _ROW_KEYS - row.keys()
    extra = row.keys() - _ROW_KEYS
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={sorted(missing)}")
        if extra:
            details.append(f"extra={sorted(extra)}")
        raise Stage14InvariantError("invalid pixel-row schema: " + ", ".join(details))

    if row["stage_id"] != 14 or not _is_integer(row["stage_id"]):
        raise Stage14InvariantError("stage_id must be integer 14")
    screen = row["screen"]
    if (not isinstance(screen, str)
            or not (screen == "capillary"
                    or (screen.startswith("capillary-s")
                        and screen[11:].isdigit()
                        and int(screen[11:]) > 0))):
        raise Stage14InvariantError(
            "screen must be 'capillary' or 'capillary-sN' in schema v1"
        )
    _require_integer("pixel", row["pixel"], minimum=0)
    _require_finite("x_um", row["x_um"])
    _require_finite("y_um", row["y_um"])
    if not isinstance(row["is_reference"], bool):
        raise Stage14InvariantError("is_reference must be a bool")
    if not isinstance(row["jackknife_computed"], bool):
        raise Stage14InvariantError("jackknife_computed must be a bool")

    computed = cast(bool, row["jackknife_computed"])
    unit = row["jackknife_unit"]
    if computed and unit != "mode":
        raise Stage14InvariantError(
            "computed schema v1 requires jackknife_unit='mode'"
        )
    if not computed and unit is not None:
        raise Stage14InvariantError(
            "uncomputed jackknife requires jackknife_unit=null"
        )

    counters = PixelCounters(
        n_rays=cast(int, row["n_rays"]),
        m_realizations=cast(int, row["m_realizations"]),
        m_pair_realizations=cast(int, row["m_pair_realizations"]),
        m_ref_realizations=cast(int, row["m_ref_realizations"]),
        max_rays_per_realization=cast(
            int, row["max_rays_per_realization"]
        ),
    )
    stats = PixelStatistics(
        I=cast(float, row["I"]),
        counters=counters,
        ic=cast(float | None, row["ic"]),
        ic_err=cast(float | None, row["ic_err"]),
        w_abs=cast(float | None, row["w_abs"]),
        w_err=cast(float | None, row["w_err"]),
        mu_raw=cast(float | None, row["mu_raw"]),
        mu_raw_err=cast(float | None, row["mu_raw_err"]),
        n_mu_loo_valid=cast(int | None, row["n_mu_loo_valid"]),
    )
    validate_serializer_invariants(
        stats,
        flag=cast(PixelFlag | None, row["flag"]),
        is_reference=cast(bool, row["is_reference"]),
        ref_status=cast(RefStatus, row["ref_status"]),
        n_jackknife_units=cast(int | None, row["n_jackknife_units"]),
        thresholds=thresholds,
        jackknife_computed=computed,
    )


def serialize_pixel(
    stats: PixelStatistics,
    *,
    pixel: int,
    x_um: float,
    y_um: float,
    is_reference: bool,
    ref_status: RefStatus,
    n_jackknife_units: int,
    thresholds: FlagThresholds,
    screen: str = "capillary",
) -> dict[str, object]:
    """Build and validate one schema-v1 JSON object.

    The returned mapping contains only JSON-native values.  A caller should
    still use ``json.dumps(..., allow_nan=False)`` as a final defence.
    """

    flag = classify_pixel(
        stats,
        is_reference=is_reference,
        ref_status=ref_status,
        n_jackknife_units=n_jackknife_units,
        thresholds=thresholds,
    )
    counters = stats.counters
    row: dict[str, object] = {
        "stage_id": 14,
        "screen": screen,
        "pixel": pixel,
        "is_reference": is_reference,
        "ref_status": ref_status,
        "jackknife_computed": True,
        "n_jackknife_units": n_jackknife_units,
        "jackknife_unit": "mode",
        "x_um": x_um,
        "y_um": y_um,
        "I": stats.I,
        "n_rays": counters.n_rays,
        "mu_raw": stats.mu_raw,
        "mu_raw_err": stats.mu_raw_err,
        "ic": stats.ic,
        "ic_err": stats.ic_err,
        "w_abs": stats.w_abs,
        "w_err": stats.w_err,
        "m_realizations": counters.m_realizations,
        "m_pair_realizations": counters.m_pair_realizations,
        "m_ref_realizations": counters.m_ref_realizations,
        "max_rays_per_realization": counters.max_rays_per_realization,
        "n_mu_loo_valid": stats.n_mu_loo_valid,
        "flag": flag,
    }
    validate_pixel_row(row, thresholds)
    return row


def w_signal_status(
    w_abs: float | None,
    w_err: float | None,
    thresholds: FlagThresholds,
) -> WSignalStatus:
    """Return the independent diagnostic channel-B status."""

    if w_abs is None or w_err is None:
        return "unknown"
    value = _require_nonnegative("w_abs", w_abs)
    error = _require_nonnegative("w_err", w_err)
    if error == 0.0:
        return "unknown"
    if value / error >= thresholds.z_w:
        return "detected"
    return "not-detected"


__all__ = [
    "FlagThresholds",
    "PIXEL_FLAGS",
    "PixelCounters",
    "PixelFlag",
    "PixelStatistics",
    "REF_STATUSES",
    "RefStatus",
    "Stage14InvariantError",
    "WSignalStatus",
    "classify_pixel",
    "serialize_pixel",
    "validate_counters",
    "validate_pixel_row",
    "validate_ref",
    "validate_serializer_invariants",
    "w_signal_status",
]

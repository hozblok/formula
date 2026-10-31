"""Explicit Python-oracle tests for the native Stage 14 store/finalizer."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import struct

import pytest

from formula import _formula


Stage14Store = getattr(_formula, "Stage14Store", None)
stage14_finalize = getattr(_formula, "stage14_finalize", None)
native_stage14 = pytest.mark.skipif(
    Stage14Store is None or stage14_finalize is None,
    reason="formula extension was built without native Stage 14 bindings",
)

STORE_KEYS = {
    "I",
    "w_re",
    "w_im",
    "ic",
    "n_rays",
    "m_realizations",
    "m_pair_realizations",
    "m_ref_realizations",
    "max_rays_per_realization",
    "payload_bytes",
    "payload_sha256",
    "n_modes",
    "mode_ids",
}

FINAL_KEYS = {
    "ic_err",
    "w_err",
    "mu_raw",
    "mu_raw_err",
    "n_mu_loo_valid",
    "mu_raw_defined",
    "mu_raw_err_defined",
    "ic_ref_loo",
    "payload_sha256",
    "payload_bytes",
    "bytes_read",
    "pass1_seconds",
    "pass2_seconds",
}


def _unpack(raw: bytes, code: str, count: int):
    assert len(raw) == struct.calcsize(f"<{count}{code}")
    return list(struct.unpack(f"<{count}{code}", raw))


def _doubles(raw: bytes, count: int) -> list[float]:
    return _unpack(raw, "d", count)


def _pack_doubles(values) -> bytes:
    values = list(values)
    return struct.pack(f"<{len(values)}d", *values)


def _oracle(modes, n_pixels: int, ref: int, kms, weights):
    rows_w = []
    rows_ic = []
    total_i = [0.0] * n_pixels
    total_w = [0j] * n_pixels
    total_ic = [0.0] * n_pixels
    n_rays = [0] * n_pixels
    m_realizations = [0] * n_pixels
    m_pair_realizations = [0] * n_pixels
    m_ref_realizations = [0] * n_pixels
    max_rays_per_realization = [0] * n_pixels

    for rays in modes:
        g = [[0j] * n_pixels for _ in kms]
        sq = [[0.0] * n_pixels for _ in kms]
        counts = [0] * n_pixels
        for pixel, opl, amps in rays:
            counts[pixel] += 1
            for line, (km, amp) in enumerate(zip(kms, amps)):
                phase = km * opl
                term = amp * complex(math.cos(phase), math.sin(phase))
                g[line][pixel] += term
                sq[line][pixel] += amp.real * amp.real + amp.imag * amp.imag

        ref_occupied = counts[ref] != 0
        for pixel, count in enumerate(counts):
            n_rays[pixel] += count
            if count:
                m_realizations[pixel] += 1
                max_rays_per_realization[pixel] = max(
                    max_rays_per_realization[pixel], count
                )
                if count >= 2:
                    m_pair_realizations[pixel] += 1
                if ref_occupied:
                    m_ref_realizations[pixel] += 1

        row_w = [0j] * n_pixels
        row_ic = [0.0] * n_pixels
        for line, weight in enumerate(weights):
            ref_conj = g[line][ref].conjugate()
            for pixel in range(n_pixels):
                a2 = g[line][pixel].real ** 2 + g[line][pixel].imag ** 2
                total_i[pixel] += weight * a2
                row_ic[pixel] += weight * (a2 - sq[line][pixel])
                cross = weight * g[line][pixel] * ref_conj
                if pixel == ref:
                    cross -= weight * sq[line][pixel]
                row_w[pixel] += cross
        rows_w.append(row_w)
        rows_ic.append(row_ic)
        for pixel in range(n_pixels):
            total_w[pixel] += row_w[pixel]
            total_ic[pixel] += row_ic[pixel]

    return {
        "rows_w": rows_w,
        "rows_ic": rows_ic,
        "I": total_i,
        "W": total_w,
        "ic": total_ic,
        "n_rays": n_rays,
        "m_realizations": m_realizations,
        "m_pair_realizations": m_pair_realizations,
        "m_ref_realizations": m_ref_realizations,
        "max_rays_per_realization": max_rays_per_realization,
    }


def _oracle_finalize(oracle, ref: int):
    rows_w = oracle["rows_w"]
    rows_ic = oracle["rows_ic"]
    total_w = oracle["W"]
    total_ic = oracle["ic"]
    n_modes = len(rows_w)
    n_pixels = len(total_ic)
    factor = (n_modes - 1) / n_modes
    ic_err = []
    w_err = []
    mu_raw = []
    mu_raw_err = []
    n_mu_loo_valid = []
    ic_ref_loo = [total_ic[ref] - row[ref] for row in rows_ic]

    for pixel in range(n_pixels):
        ic_mean = total_ic[pixel] / n_modes
        w_mean = total_w[pixel] / n_modes
        ic_err.append(math.sqrt(
            factor * sum((row[pixel] - ic_mean) ** 2 for row in rows_ic)
        ))
        w_err.append(math.sqrt(
            factor * sum(abs(row[pixel] - w_mean) ** 2 for row in rows_w)
        ))
        if total_ic[pixel] > 0 and total_ic[ref] > 0:
            mu_raw.append(abs(total_w[pixel]) /
                          math.sqrt(total_ic[pixel] * total_ic[ref]))
        else:
            mu_raw.append(None)

        loo = []
        for mode in range(n_modes):
            ic_p = total_ic[pixel] - rows_ic[mode][pixel]
            ic_r = total_ic[ref] - rows_ic[mode][ref]
            w = total_w[pixel] - rows_w[mode][pixel]
            if ic_p > 0 and ic_r > 0:
                value = abs(w) / math.sqrt(ic_p * ic_r)
                if math.isfinite(value):
                    loo.append(value)
        n_mu_loo_valid.append(len(loo))
        if len(loo) == n_modes:
            mean = sum(loo) / n_modes
            mu_raw_err.append(math.sqrt(
                factor * sum((value - mean) ** 2 for value in loo)
            ))
        else:
            mu_raw_err.append(None)
    return {
        "ic_err": ic_err,
        "w_err": w_err,
        "mu_raw": mu_raw,
        "mu_raw_err": mu_raw_err,
        "n_mu_loo_valid": n_mu_loo_valid,
        "ic_ref_loo": ic_ref_loo,
    }


def _build_store(tmp_path: Path, modes, n_pixels: int, ref: int, kms, weights):
    path = tmp_path / "mode-rows.f64"
    store = Stage14Store(
        str(path), len(modes), n_pixels, ref, list(kms), list(weights)
    )
    for mode_id, rays in enumerate(modes, start=10):
        store.begin_mode(mode_id)
        for pixel, opl, amps in rays:
            store.add_ray(pixel, opl, list(amps))
        store.fold_mode()
    aggregate = store.finish()
    return path, aggregate


def _finalize(path: Path, aggregate, n_pixels: int, ref: int, n_modes: int):
    return stage14_finalize(
        [str(path)],
        [n_modes],
        n_pixels,
        ref,
        aggregate["I"],
        aggregate["w_re"],
        aggregate["w_im"],
        aggregate["ic"],
        [aggregate["payload_sha256"]],
    )


@native_stage14
def test_native_multiline_store_and_finalizer_match_explicit_oracle(tmp_path):
    kms = [0.7, 1.9]
    weights = [0.35, 0.65]
    n_pixels = 3
    ref = 0
    modes = []
    for mode in range(4):
        rays = []
        for pixel in range(n_pixels):
            for ray in range(2 + (mode + pixel) % 2):
                rays.append((
                    pixel,
                    0.05 * mode + 0.011 * pixel + 0.003 * ray,
                    (
                        complex(1.0 + 0.08 * mode + 0.03 * pixel,
                                0.02 * (ray + 1)),
                        complex(0.6 + 0.04 * pixel + 0.01 * ray,
                                -0.03 * mode + 0.015 * ray),
                    ),
                ))
        modes.append(rays)

    expected = _oracle(modes, n_pixels, ref, kms, weights)
    expected_final = _oracle_finalize(expected, ref)
    path, aggregate = _build_store(
        tmp_path, modes, n_pixels, ref, kms, weights
    )

    assert set(aggregate) == STORE_KEYS
    assert aggregate["n_modes"] == 4
    assert aggregate["mode_ids"] == [10, 11, 12, 13]
    assert aggregate["payload_bytes"] == 4 * n_pixels * 24
    assert aggregate["payload_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert _doubles(aggregate["I"], n_pixels) == pytest.approx(expected["I"])
    assert _doubles(aggregate["w_re"], n_pixels) == pytest.approx(
        [value.real for value in expected["W"]]
    )
    assert _doubles(aggregate["w_im"], n_pixels) == pytest.approx(
        [value.imag for value in expected["W"]]
    )
    assert _doubles(aggregate["ic"], n_pixels) == pytest.approx(expected["ic"])
    assert _unpack(aggregate["n_rays"], "Q", n_pixels) == expected["n_rays"]
    for name in (
        "m_realizations",
        "m_pair_realizations",
        "m_ref_realizations",
        "max_rays_per_realization",
    ):
        assert _unpack(aggregate[name], "I", n_pixels) == expected[name]

    # The payload itself is exactly mode-major interleaved W.real/W.imag/Ic.
    raw_cells = _doubles(path.read_bytes(), 4 * n_pixels * 3)
    expected_cells = []
    for row_w, row_ic in zip(expected["rows_w"], expected["rows_ic"]):
        for w, ic in zip(row_w, row_ic):
            expected_cells.extend((w.real, w.imag, ic))
    assert raw_cells == pytest.approx(expected_cells)

    final = _finalize(path, aggregate, n_pixels, ref, len(modes))
    assert set(final) == FINAL_KEYS
    assert final["payload_sha256"] == [aggregate["payload_sha256"]]
    assert final["payload_bytes"] == [aggregate["payload_bytes"]]
    assert final["bytes_read"] == 2 * aggregate["payload_bytes"]
    assert _doubles(final["ic_err"], n_pixels) == pytest.approx(
        expected_final["ic_err"]
    )
    assert _doubles(final["w_err"], n_pixels) == pytest.approx(
        expected_final["w_err"]
    )
    assert _doubles(final["mu_raw"], n_pixels) == pytest.approx(
        expected_final["mu_raw"]
    )
    assert _doubles(final["mu_raw_err"], n_pixels) == pytest.approx(
        expected_final["mu_raw_err"]
    )
    assert _unpack(final["n_mu_loo_valid"], "I", n_pixels) == [4, 4, 4]
    assert _unpack(final["mu_raw_defined"], "B", n_pixels) == [1, 1, 1]
    assert _unpack(final["mu_raw_err_defined"], "B", n_pixels) == [1, 1, 1]
    assert _doubles(final["ic_ref_loo"], 4) == pytest.approx(
        expected_final["ic_ref_loo"]
    )


@native_stage14
def test_native_finalize_uses_centered_ic_and_complex_w_moments(tmp_path):
    n_modes, n_pixels, ref = 4, 2, 0
    rows_ic = [
        [1e12 + 0.125, 2e12 + 0.25],
        [1e12 - 0.125, 2e12 - 0.25],
        [1e12 + 0.375, 2e12 + 0.75],
        [1e12 - 0.375, 2e12 - 0.75],
    ]
    rows_w = [
        [complex(1e11 + 0.125, -5e10 + 0.25),
         complex(2e11 + 0.25, 3e10 - 0.5)],
        [complex(1e11 - 0.125, -5e10 - 0.25),
         complex(2e11 - 0.25, 3e10 + 0.5)],
        [complex(1e11 + 0.375, -5e10 + 0.75),
         complex(2e11 + 0.75, 3e10 - 1.5)],
        [complex(1e11 - 0.375, -5e10 - 0.75),
         complex(2e11 - 0.75, 3e10 + 1.5)],
    ]
    total_ic = [sum(row[p] for row in rows_ic) for p in range(n_pixels)]
    total_w = [sum((row[p] for row in rows_w), 0j) for p in range(n_pixels)]
    path = tmp_path / "centered.f64"
    cells = []
    for row_w, row_ic in zip(rows_w, rows_ic):
        for w, ic in zip(row_w, row_ic):
            cells.extend((w.real, w.imag, ic))
    path.write_bytes(_pack_doubles(cells))

    final = stage14_finalize(
        [str(path)],
        [n_modes],
        n_pixels,
        ref,
        _pack_doubles([1.0, 1.0]),
        _pack_doubles([value.real for value in total_w]),
        _pack_doubles([value.imag for value in total_w]),
        _pack_doubles(total_ic),
        [hashlib.sha256(path.read_bytes()).hexdigest()],
    )
    factor = (n_modes - 1) / n_modes
    expected_ic = [
        math.sqrt(factor * sum(
            (row[p] - total_ic[p] / n_modes) ** 2 for row in rows_ic
        ))
        for p in range(n_pixels)
    ]
    expected_w = [
        math.sqrt(factor * sum(
            abs(row[p] - total_w[p] / n_modes) ** 2 for row in rows_w
        ))
        for p in range(n_pixels)
    ]
    assert _doubles(final["ic_err"], n_pixels) == pytest.approx(expected_ic)
    assert _doubles(final["w_err"], n_pixels) == pytest.approx(expected_w)

    # A raw-moment subtraction loses these sub-unit deviations at 1e12.
    unstable = factor * (
        sum(row[0] ** 2 for row in rows_ic) - total_ic[0] ** 2 / n_modes
    )
    assert unstable != pytest.approx(expected_ic[0] ** 2)


@native_stage14
def test_native_invalid_loo_is_counted_but_never_subset_rescaled(tmp_path):
    # Mode 0 carries all pair intensity at ref.  Deleting it gives Ic_ref=0;
    # the other two LOO replicas remain valid, but no sigma may be published.
    modes = [
        [(0, 0.0, (1 + 0j,)), (0, 0.0, (1 + 0j,)),
         (1, 0.0, (1 + 0j,)), (1, 0.0, (1 + 0j,))],
        [(0, 0.0, (1 + 0j,)),
         (1, 0.0, (1 + 0j,)), (1, 0.0, (1 + 0j,))],
        [(0, 0.0, (1 + 0j,)),
         (1, 0.0, (1 + 0j,)), (1, 0.0, (1 + 0j,))],
    ]
    path, aggregate = _build_store(tmp_path, modes, 2, 0, [0.0], [1.0])
    final = _finalize(path, aggregate, 2, 0, 3)

    assert _doubles(final["ic_ref_loo"], 3) == pytest.approx([0.0, 2.0, 2.0])
    assert _unpack(final["n_mu_loo_valid"], "I", 2) == [2, 2]
    assert _unpack(final["mu_raw_defined"], "B", 2) == [1, 1]
    assert _unpack(final["mu_raw_err_defined"], "B", 2) == [0, 0]
    assert all(math.isnan(value) for value in _doubles(final["mu_raw_err"], 2))


@native_stage14
def test_shared_occupancy_is_not_inferred_from_cancelling_cross_sum(tmp_path):
    modes = [
        # Ref is occupied but its field cancels exactly, so W_s(target) == 0.
        [(0, 0.0, (1 + 0j,)), (0, 0.0, (-1 + 0j,)),
         (1, 0.0, (1 + 0j,))],
        [(0, 0.0, (1 + 0j,)), (1, 0.0, (1 + 0j,))],
    ]
    path, aggregate = _build_store(tmp_path, modes, 2, 0, [0.0], [1.0])
    assert _unpack(aggregate["m_ref_realizations"], "I", 2) == [2, 2]
    first_mode = _doubles(path.read_bytes()[:2 * 3 * 8], 2 * 3)
    assert first_mode[3] == 0.0  # target W.real
    assert first_mode[4] == 0.0  # target W.imag


@native_stage14
@pytest.mark.parametrize(
    ("ic_ref", "ic_pixel", "w_abs", "expected"),
    [
        (1e300, 1e-300, 1e300, 1e300),
        (1e-300, 1e300, 1e-300, 1e-300),
    ],
)
def test_mu_denominator_is_range_safe(
    tmp_path, ic_ref, ic_pixel, w_abs, expected
):
    path = tmp_path / f"mu-{expected}.f64"
    cells = []
    for _ in range(2):
        cells.extend((0.0, 0.0, ic_ref / 2))
        cells.extend((w_abs / 2, 0.0, ic_pixel / 2))
    path.write_bytes(_pack_doubles(cells))
    final = stage14_finalize(
        [str(path)], [2], 2, 0,
        _pack_doubles([0.0, 0.0]),
        _pack_doubles([0.0, w_abs]),
        _pack_doubles([0.0, 0.0]),
        _pack_doubles([ic_ref, ic_pixel]),
        [hashlib.sha256(path.read_bytes()).hexdigest()],
    )
    values = _doubles(final["mu_raw"], 2)
    assert _unpack(final["mu_raw_defined"], "B", 2)[1] == 1
    assert values[1] == pytest.approx(expected, rel=2e-15, abs=0.0)
    assert values[1] > 0.0


@native_stage14
@pytest.mark.parametrize("magnitude", [1e200, 1e-200])
def test_centered_ic_and_w_sumsq_preserve_extreme_scale(tmp_path, magnitude):
    path = tmp_path / f"sumsq-{magnitude}.f64"
    path.write_bytes(_pack_doubles([
        magnitude, magnitude, magnitude,
        -magnitude, -magnitude, -magnitude,
    ]))
    final = stage14_finalize(
        [str(path)], [2], 1, 0,
        _pack_doubles([0.0]),
        _pack_doubles([0.0]),
        _pack_doubles([0.0]),
        _pack_doubles([0.0]),
        [hashlib.sha256(path.read_bytes()).hexdigest()],
    )
    assert _doubles(final["ic_err"], 1)[0] == pytest.approx(
        magnitude, rel=2e-15, abs=0.0
    )
    assert _doubles(final["w_err"], 1)[0] == pytest.approx(
        math.sqrt(2.0) * magnitude, rel=2e-15, abs=0.0
    )


@native_stage14
@pytest.mark.parametrize("magnitude", [1e200, 1e-200])
def test_raw_mu_sumsq_preserves_extreme_scale(tmp_path, magnitude):
    path = tmp_path / f"mu-sumsq-{magnitude}.f64"
    path.write_bytes(_pack_doubles([
        1.0, 0.0, 1.0,
        magnitude, 0.0, 1.0,
        1.0, 0.0, 1.0,
        2.0 * magnitude, 0.0, 1.0,
    ]))
    final = stage14_finalize(
        [str(path)], [2], 2, 0,
        _pack_doubles([0.0, 0.0]),
        _pack_doubles([2.0, 3.0 * magnitude]),
        _pack_doubles([0.0, 0.0]),
        _pack_doubles([2.0, 2.0]),
        [hashlib.sha256(path.read_bytes()).hexdigest()],
    )
    assert _unpack(final["n_mu_loo_valid"], "I", 2)[1] == 2
    assert _unpack(final["mu_raw_err_defined"], "B", 2)[1] == 1
    error = _doubles(final["mu_raw_err"], 2)[1]
    assert error == pytest.approx(0.5 * magnitude, rel=3e-15, abs=0.0)
    assert error > 0.0


@native_stage14
def test_store_small_and_zero_weights_do_not_overflow_intermediates(tmp_path):
    path = tmp_path / "weighted.f64"
    store = Stage14Store(str(path), 1, 1, 0, [0.0, 1.0], [1e-300, 0.0])
    store.begin_mode(0)
    for _ in range(2):
        store.add_ray(0, 0.0, [8e153 + 0j, complex(1e308, 1e308)])
    store.fold_mode()
    aggregate = store.finish()
    assert _doubles(aggregate["I"], 1)[0] == pytest.approx(
        2.56e8, rel=3e-15
    )
    assert _doubles(aggregate["ic"], 1)[0] == pytest.approx(
        1.28e8, rel=3e-15
    )
    assert _doubles(aggregate["w_re"], 1)[0] == pytest.approx(
        1.28e8, rel=3e-15
    )


@native_stage14
def test_add_ray_failure_is_atomic(tmp_path):
    path = tmp_path / "atomic-ray.f64"
    store = Stage14Store(str(path), 1, 1, 0, [0.0, 0.0], [0.5, 0.5])
    store.begin_mode(0)
    with pytest.raises(ValueError, match="non-finite"):
        store.add_ray(0, 0.0, [1 + 0j, complex(math.nan, 0.0)])
    store.add_ray(0, 0.0, [1 + 0j, 1 + 0j])
    store.fold_mode()
    aggregate = store.finish()
    assert _unpack(aggregate["n_rays"], "Q", 1) == [1]
    assert _doubles(aggregate["I"], 1) == pytest.approx([1.0])
    assert _doubles(aggregate["ic"], 1) == pytest.approx([0.0])
    assert _doubles(aggregate["w_re"], 1) == pytest.approx([0.0])


@native_stage14
def test_store_no_clobber_preserves_existing_unicode_path(tmp_path):
    path = tmp_path / "кэш-данные.f64"
    sentinel = b"do not replace"
    path.write_bytes(sentinel)
    with pytest.raises(RuntimeError, match="already exists"):
        Stage14Store(str(path), 1, 1, 0, [0.0], [1.0])
    assert path.read_bytes() == sentinel


@native_stage14
def test_store_no_clobber_does_not_follow_dangling_symlink(tmp_path):
    target = tmp_path / "missing-target.f64"
    link = tmp_path / "dangling-cache.f64"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")
    with pytest.raises(RuntimeError):
        Stage14Store(str(link), 1, 1, 0, [0.0], [1.0])
    assert link.is_symlink()
    assert not target.exists()

from decimal import Decimal, localcontext

import numpy as np
import pytest

from formula.capsysred import rays_v3
from formula.capsysred.stages import _b5_archive as archive_reader


def _ray(ray, *, dx=1e-8, opl="0.905000000000000050", sins=None, refl=None):
    row = dict(stage="capillary", mode=0, ray=ray, fate="screen", pixel=0,
               x=2e-6, y=-3e-6, dx=dx, dy=0.0, opl=opl,
               sins=[] if sins is None else sins)
    if refl is not None:
        row["refl"] = refl
    return row


def _archive(tmp_path, rows, *, lean=False, split=None):
    archive = tmp_path / "rays"
    meta = dict(format=3, geometry=dict(capillary=dict(screen=dict(z=.23))))
    if lean:
        meta["lean"] = True
    rays_v3.write_fingerprint(archive, meta)
    cuts = [0, len(rows)] if split is None else [0, split, len(rows)]
    entries = []
    for lo, hi in zip(cuts, cuts[1:]):
        writer = rays_v3.SectionWriter(archive, "capillary", 0, lo, hi,
                                       origin=["0", "0", "-0.675"])
        for row in rows[lo:hi]:
            writer.write_row(row)
        entries.append(writer.close())
    rays_v3.write_index(archive, entries)
    return archive, entries


def test_decimal_phase_and_stable_unit_direction_extension(tmp_path):
    archive, _ = _archive(tmp_path, [_ray(0)])
    modes, meta = archive_reader.read_sample(archive, target_z=.58, k=4e10)
    mode = modes[0]
    with localcontext() as context:
        context.prec = 80
        dx = Decimal("1e-8")
        expected = Decimal("4e10") * (Decimal("5e-17")
                    + Decimal(".35") * (1 / (1 - dx * dx).sqrt() - 1))
    assert mode["phase_opl"][0] == pytest.approx(float(expected), rel=2e-15)
    assert mode["points"][0] == pytest.approx([2e-6 + .35e-8, -3e-6])
    assert mode["directions"][0] == pytest.approx([1e-8, 0])
    assert meta["modes"][0]["screen_rows"] == 1
    assert meta["files"][0]["sha256"]
    assert meta["phase_convention"].endswith("no Fresnel phase")


def test_selected_mode_start_reads_only_requested_mode(tmp_path):
    archive, entries = _archive(tmp_path, [_ray(0)])
    writer = rays_v3.SectionWriter(archive, "capillary", 1, 0, 1,
                                   origin=["0.000001", "0", "-0.675"])
    row = _ray(0)
    row["mode"] = 1
    writer.write_row(row)
    rays_v3.write_index(archive, [*entries, writer.close()])
    modes, meta = archive_reader.read_sample(archive, mode_start=1, max_modes=1,
                                            target_z=.58, k=4e10)
    assert len(modes) == 1 and modes[0]["mode"] == 1
    assert modes[0]["origin"][0] == 1e-6
    assert meta["mode_start"] == 1 and len(meta["files"]) == 1
    with pytest.raises(ValueError, match="mode_start"):
        archive_reader.read_sample(archive, mode_start=-1, target_z=.58, k=4e10)


def test_prefix_counts_all_fates_and_retains_relative_phase(tmp_path):
    rows = [_ray(0), _ray(1), _ray(2, opl="0.905000000025000050"), _ray(3)]
    rows[1]["fate"] = "absorbed"
    archive, _ = _archive(tmp_path, rows, split=2)
    modes, meta = archive_reader.read_sample(archive, target_z=.23, k=4e10,
                                           max_rays_per_mode=3)
    assert modes[0]["ray_ids"].tolist() == [0, 2]
    assert np.diff(modes[0]["phase_opl"])[0] == pytest.approx(1.0)
    assert meta["modes"][0]["prefix_rows"] == 3
    assert meta["modes"][0]["fate_counts"] == dict(screen=2, absorbed=1)
    assert len(meta["files"]) == 2
    assert meta["files"][1]["sampled_ray_stop"] == 3


def test_hash_checks_unread_suffix(tmp_path):
    archive, entries = _archive(tmp_path, [_ray(0), _ray(1)])
    section = archive / "modes" / entries[0].file
    payload = bytearray(section.read_bytes())
    payload[-1] ^= 1
    section.write_bytes(payload)
    with pytest.raises(ValueError, match="SHA-256"):
        archive_reader.read_sample(archive, target_z=.58, k=1, max_rays_per_mode=1)


def test_rejects_lean_and_missing_reflections(tmp_path):
    archive, _ = _archive(tmp_path, [_ray(0)], lean=True)
    with pytest.raises(ValueError, match="non-lean"):
        archive_reader.read_sample(archive, target_z=.58, k=1)
    other = tmp_path / "other"
    archive, _ = _archive(other, [_ray(0, sins=["0.0001"])])
    with pytest.raises(ValueError, match="reflection points"):
        archive_reader.read_sample(archive, target_z=.58, k=1)


@pytest.mark.parametrize("update, message", [
    ({"dx": 1.0}, "unit vector"),
    ({"x": float("nan")}, "finite coordinates"),
    ({"opl": "NaN"}, "non-finite decimal"),
    ({"ray": 1}, "ray identity"),
    ({"sins": ["1.1"], "refl": [[0, 0, .1]]}, "outside"),
])
def test_rejects_invalid_selected_rows(tmp_path, update, message):
    row = _ray(0)
    row.update(update)
    archive, _ = _archive(tmp_path, [row])
    with pytest.raises(ValueError, match=message):
        archive_reader.read_sample(archive, target_z=.58, k=1)


@pytest.mark.parametrize("fate", ["screen ", "unknown", "", None, 1, [], {}])
def test_rejects_malformed_fate_in_sampled_prefix(tmp_path, fate):
    row = _ray(0)
    row["fate"] = fate
    archive, _ = _archive(tmp_path, [row])
    with pytest.raises(ValueError, match="invalid ray fate"):
        archive_reader.read_sample(archive, target_z=.58, k=1)


def test_lost_and_absorbed_remain_in_prefix_budget(tmp_path):
    rows = [_ray(0), _ray(1), _ray(2)]
    rows[0]["fate"], rows[1]["fate"] = "lost", "absorbed"
    archive, _ = _archive(tmp_path, rows)
    modes, meta = archive_reader.read_sample(archive, target_z=.58, k=1)
    assert modes[0]["ray_ids"].tolist() == [2]
    assert meta["modes"][0]["prefix_rows"] == 3
    assert meta["modes"][0]["fate_counts"] == dict(lost=1, absorbed=1, screen=1)


def test_change_during_verified_prefix_read_is_rejected(tmp_path, monkeypatch):
    archive, _ = _archive(tmp_path, [_ray(0), _ray(1)])
    original_line = archive_reader._line
    calls = 0

    def changed_line(stream, path):
        nonlocal calls
        result = original_line(stream, path)
        calls += 1
        if calls == 2:
            with path.open("ab") as output:
                output.write(b"changed")
        return result

    monkeypatch.setattr(archive_reader, "_line", changed_line)
    with pytest.raises(ValueError, match="changed"):
        archive_reader.read_sample(archive, target_z=.58, k=1, max_rays_per_mode=1)


def test_rejects_backward_replay(tmp_path):
    archive, _ = _archive(tmp_path, [_ray(0)])
    with pytest.raises(ValueError, match="downstream"):
        archive_reader.read_sample(archive, target_z=.1, k=1)

"""Direct unit tests of stage-14 _validate_row: every shape check without an
archive fixture."""

from __future__ import annotations

import pytest

from formula.capsysred.stages.stage14 import _validate_row

_DROP = object()


def _row(**overrides) -> dict:
    base = {"stage": "capillary", "mode": 3, "ray": 7, "fate": "screen",
            "pixel": 5, "opl": "0.25", "sins": [0.1, 0.2],
            "x": 1e-6, "y": -2e-6, "dx": 0.0, "dy": 1e-3}
    base.update(overrides)
    return {k: v for k, v in base.items() if v is not _DROP}


def test_validate_row_parses_canonical_rows():
    assert _validate_row(_row(), 2, "arch", {}) == (
        "capillary", 3, 7, [0.1, 0.2], "screen")
    scene, mode, ray, sins, fate = _validate_row(
        _row(fate="absorbed", pixel=None, sins=[],
             x=_DROP, y=_DROP, dx=_DROP, dy=_DROP), 2, "arch", {})
    assert (scene, mode, ray, sins, fate) == ("capillary", 3, 7, [], "absorbed")
    # numeric opl and int sins pass the float() boundary
    assert _validate_row(_row(opl=0.25, sins=[1]), 2, "arch", {})[3] == [1.0]


BAD = [
    (dict(stage=_DROP), "ray row lacks stage"),
    (dict(stage=5), "ray row lacks stage"),
    (dict(mode=-1), "nonnegative integers"),
    (dict(mode="3"), "nonnegative integers"),
    (dict(ray=True), "nonnegative integers"),
    (dict(pixel=_DROP), "invalid saved pixel id"),
    (dict(pixel=-2), "invalid saved pixel id"),
    (dict(pixel=True), "invalid saved pixel id"),
    (dict(pixel="5"), "invalid saved pixel id"),
    (dict(opl=_DROP), "invalid optical path length"),
    (dict(opl="abc"), "invalid optical path length"),
    (dict(opl=None), "invalid optical path length"),
    (dict(opl="inf"), "non-finite optical path length"),
    (dict(sins=_DROP), "sins must be a list"),
    (dict(sins="0.1"), "sins must be a list"),
    (dict(sins=["x"]), "invalid sins"),
    (dict(sins=[0.1, None]), "invalid sins"),
    (dict(sins=[float("nan")]), "non-finite sins"),
    (dict(fate=_DROP), "invalid ray fate"),
    (dict(fate="gone"), "invalid ray fate"),
    (dict(x=_DROP), "screen ray lacks finite"),
    (dict(dy="oops"), "screen ray lacks finite"),
    (dict(dx=float("inf")), "screen ray has non-finite"),
]


@pytest.mark.parametrize("overrides,match", BAD,
                         ids=[match for _, match in BAD])
def test_validate_row_rejects(overrides, match):
    with pytest.raises(ValueError, match=match):
        _validate_row(_row(**overrides), 9, "arch", {})


def test_validate_row_trailer_and_prefix():
    with pytest.raises(ValueError,
                       match="arch:41: capillary row follows its trailer"):
        _validate_row(_row(), 41, "arch", {"capillary": 100})
    # a foreign scene's trailer does not block this row
    assert _validate_row(_row(), 41, "arch", {"free": 10})[0] == "capillary"
    # non-screen fates skip the x/y/direction checks entirely
    assert _validate_row(
        _row(fate="lost", x=_DROP, y=_DROP, dx=_DROP, dy=_DROP),
        41, "arch", {})[4] == "lost"

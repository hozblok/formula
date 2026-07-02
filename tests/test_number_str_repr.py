"""Number.__str__ and __repr__ contract:

  * __str__ is value-centric (human form). Zero imag collapses to real;
    sign of imag sits outside the *i term, not inside parens.
  * __str__ agrees with __eq__: a == b ⇒ str(a) == str(b).
  * __str__ roundtrips through Number(...): keeps '*' before i so the
    Solver parser doesn't choke on juxtaposition.
  * __repr__ is debug form: Number('expr', precision=P), eval-able.
"""

from formula import Number


# ─── __str__: format ────────────────────────────────────────────────────────


def test_str_real_value():
    assert str(Number(5)) == "5"
    assert str(Number(-7)) == "-7"


def test_str_complex_math_sign_outside():
    # Sign of imag is at the top level, not buried as '+i*(-2)'.
    assert str(Number("3+4*i")) == "3+4*i"
    assert str(Number("-1-2*i")) == "-1-2*i"


def test_str_pure_imag_drops_zero_real():
    # Symmetric to zero-imag collapse: zero real disappears, leaving the
    # signed imag term alone. '5*i' and '-5*i' read as math, not as a
    # sum with an invisible zero.
    assert str(Number("0+5*i")) == "5*i"
    assert str(Number("0-5*i")) == "-5*i"
    assert str(Number("5*i")) == "5*i"
    assert str(Number("-i")) == "-1*i"


def test_str_keeps_star_for_solver_roundtrip():
    # If we wrote '3+4i' the Solver would try to read '4i' as a variable.
    # The '*' is structural and must remain.
    assert "*" in str(Number("3+4*i"))


# ─── __str__: zero imag collapses ──────────────────────────────────────────


def test_str_collapses_clean_zero_imag():
    # i*i = -1+0i exactly (mul, not pow) — string must show plain '-1'.
    assert str(Number("i*i")) == "-1"
    assert str(Number("1+0*i")) == "1"
    assert str(Number("(1+i)*(1-i)")) == "2"
    assert str(Number("(0+3*i)*(0-3*i)")) == "9"


def test_str_preserves_drift_imag():
    # i^4 = exp(4·log(i)) drifts: imag is ~7.6e-25, not 0. The drift is
    # real (== Number('1') is False), so __str__ must show it. This is
    # exactly what we want: __str__ never lies about value.
    n = Number("i^4")
    s = str(n)
    assert n != Number("1")
    assert s != "1"
    assert "*i" in s


# ─── __str__ ⇔ __eq__ invariant ─────────────────────────────────────────────


def test_str_matches_eq_across_kinds():
    # The whole point of step 1: equal values render equal.
    equal_pairs = [
        (Number("i*i"), Number("-1")),
        (Number("1+0*i"), Number("1")),
        (Number("(1+i)*(1-i)"), Number("2")),
        (Number("abs(4-3*i)"), Number("5")),
    ]
    for a, b in equal_pairs:
        assert a == b, f"setup broken: {a!r} != {b!r}"
        assert str(a) == str(b), f"a == b but str(a)={str(a)!r} != str(b)={str(b)!r}"


# ─── Roundtrip: Number(str(n)) reproduces n ────────────────────────────────


def test_roundtrip_real():
    n = Number("1/3", precision=24)
    assert Number(str(n), precision=24) == n


def test_roundtrip_complex():
    n = Number("3+4*i", precision=24)
    assert Number(str(n), precision=24) == n


def test_roundtrip_negative_imag():
    n = Number("-1-2*i", precision=24)
    assert Number(str(n), precision=24) == n


def test_roundtrip_pure_imaginary():
    n = Number("0+5*i", precision=24)
    assert Number(str(n), precision=24) == n


def test_roundtrip_high_precision():
    n = Number("1/7", precision=240)
    assert Number(str(n), precision=240) == n


def test_roundtrip_scientific_notation_in_imag():
    # Small imag triggers scientific output ('1e-05'). The '*' before i
    # is what keeps Solver from reading 'e-05i' as one token.
    n = Number("3+1e-5*i", precision=24)
    s = str(n)
    assert "e" in s.lower()
    assert Number(s, precision=24) == n


def test_roundtrip_drift_value():
    # Even drifted values must roundtrip — that's the point of __str__
    # being a valid Solver expression.
    n = Number("i^4", precision=24)
    assert Number(str(n), precision=24) == n


# ─── __repr__: debug form ───────────────────────────────────────────────────


def test_repr_includes_class_expression_and_precision():
    n = Number("3+4*i", precision=24)
    r = repr(n)
    assert r.startswith("Number(")
    assert "precision=24" in r
    assert "3+4*i" in r


def test_repr_is_evaluatable():
    # repr should produce a literal that reconstructs an equal Number.
    n = Number("3+4*i", precision=24)
    rt = eval(repr(n), {"Number": Number})
    assert rt == n


def test_repr_roundtrips_real_at_high_precision():
    n = Number("1/7", precision=240)
    rt = eval(repr(n), {"Number": Number})
    assert rt == n


def test_repr_loses_kind_by_design():
    # __repr__ delegates to str(self), which collapses zero imag. Eval
    # of the repr produces a value-equal Number with possibly different
    # _is_complex — that's correct for a value-centric debug form.
    n = Number("i*i", precision=24)  # mp_complex(-1, 0) internally
    rt = eval(repr(n), {"Number": Number})
    assert rt == n  # value identity preserved; kind may not be


# ─── Additional edge coverage ───────────────────────────────────────────────


def test_str_zero_complex_collapses_to_plain_zero():
    # 0+0i is value-equal to 0 — __str__ must show plain '0', not '0+0*i'.
    n = Number("0+0*i")
    assert str(n) == "0"
    assert n == Number("0")
    assert hash(n) == hash(Number("0"))


def test_str_negative_real_positive_imag():
    # Asymmetric sign combo not covered by the symmetric ++ / -- cases.
    n = Number("-3+4*i")
    assert str(n) == "-3+4*i"
    assert Number(str(n)) == n


def test_str_pure_imag_scientific_notation():
    # Pure imag with tiny magnitude hits the r=="0" branch + scientific form.
    # mag.lstrip('-') must not eat into the mantissa for either sign.
    pos = Number("0+1e-20*i")
    neg = Number("0-1e-20*i")
    assert str(pos) == "1e-20*i"
    assert str(neg) == "-1e-20*i"
    assert Number(str(pos)) == pos
    assert Number(str(neg)) == neg


def test_roundtrip_negative_pure_imaginary_via_solver():
    # The r=="0" sign=="-" branch was asserted string-only; lock the
    # Solver-roundtrip too so a parser drift doesn't slip through.
    n = Number("0-5*i", precision=24)
    assert str(n) == "-5*i"
    assert Number(str(n), precision=24) == n


def test_roundtrip_pure_imag_from_arithmetic():
    # i*i*i produces mp_complex(-0, -1) at the raw mp level; the C++ strip
    # turns the real "-0" into "0", and __str__ then takes the r=="0" branch.
    n = Number("i") * Number("i") * Number("i")
    assert n.parts() == ("0", "-1")
    assert str(n) == "-1*i"
    assert Number(str(n)) == n


def test_roundtrip_scientific_notation_in_real_part():
    # Mirror of the imag-side scientific test: scientific magnitude on the
    # LEFT of the '+' in 'NeM+K*i'. Solver must parse 'e-05' immediately
    # followed by '+'.
    n = Number("1e-5+3*i", precision=24)
    s = str(n)
    assert "e" in s.lower()
    assert s.endswith("+3*i")
    assert Number(s, precision=24) == n


def test_roundtrip_at_minimum_precision():
    # Precision floor: any value < 16 rounds up to 16. The string form
    # must still carry enough digits to reproduce the wrapped mp value.
    n = Number("1/7", precision=16)
    assert Number(str(n), precision=16) == n
    m = Number("1+1/7*i", precision=16)
    assert Number(str(m), precision=16) == m


def test_str_preserves_pure_imag_drift():
    # Mirror of test_str_preserves_drift_imag: drifted real, finite imag.
    # __str__ must NOT round the drift to "0" and emit a pure-imag form.
    n = Number("sin(pi) + i")
    r, i = n.parts()
    assert r != "0", f"setup: expected drifted real, got {r!r}"
    assert i == "1"
    s = str(n)
    assert s not in ("1*i", "-1*i"), f"str collapsed drifted real: {s!r}"
    assert "*i" in s and "e-" in s
    assert Number(s) == n
    assert n != Number("i")


def test_repr_precision_is_rounded_up_value():
    # __init__ rounds precision up to a supported value; __repr__ must emit
    # _precision (the rounded-up one), not the user's raw input.
    n = Number("1", precision=100)
    assert n.precision == 128  # sanity: rounding happened
    assert "precision=128" in repr(n)
    assert "precision=100" not in repr(n)
    rt = eval(repr(n), {"Number": Number})
    assert rt == n and rt.precision == 128


def test_repr_preserves_precision_on_exact_value():
    # For exact integer-valued complex like 3+4*i, _pair() is ("3","4")
    # regardless of precision, so value-equality alone wouldn't catch a
    # repr that dropped/hard-coded the precision arg. Lock it directly.
    n = Number("3+4*i", precision=64)
    assert n.precision == 64  # 64 is in AllowedPrecisions, no rounding
    rt = eval(repr(n), {"Number": Number})
    assert rt.precision == 64


def test_repr_eval_roundtrip_pure_imaginary():
    # __repr__ delegates to __str__; the r=="0" pure-imag branch (including
    # its sign sub-branches) must roundtrip through eval(repr(n)).
    for expr in ("0+5*i", "-i", "0-1e-10*i"):
        n = Number(expr)
        rt = eval(repr(n), {"Number": Number})
        assert rt == n, f"{expr!r}: repr={repr(n)!r} did not roundtrip"

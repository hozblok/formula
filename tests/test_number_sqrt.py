"""Number.sqrt: native mp sqrt for reals, ^0.5 fallback for complex."""

from formula import Number


def test_real_sqrt_exact():
    assert Number("9", 32).sqrt() == Number("3", 32)


def test_real_sqrt_matches_pow():
    a = Number("2", 32)
    assert float(abs(a.sqrt() - a ** Number("0.5", 32))) < 1e-30


def test_precision_and_kind_preserved():
    n = Number("2", 64).sqrt()
    assert n.precision == 64 and not n.is_complex


def test_complex_sqrt():
    z = Number("2*i", 32).sqrt()  # sqrt(2i) = 1+i
    assert float(abs(z - Number("1+i", 32))) < 1e-30

"""(re, im) pairs of Decimals from Formula results: parsing, arithmetic, comparison."""

import re
from decimal import Decimal


def parse(s):
    """(re, im) of a formatted real or complex result."""
    m = re.fullmatch(r"(.*)\+i\*\((.*)\)", s)
    return (Decimal(m[1]), Decimal(m[2])) if m else (Decimal(s), Decimal(0))


def add(*zs):
    return tuple(sum(parts) for parts in zip(*zs))


def scale(c, z):
    return tuple(c * part for part in z)


def times_i(z, power):
    """z * i^power."""
    for _ in range(power % 4):
        z = (-z[1], z[0])
    return z


def near(z, w, eps=Decimal("1e-50")):
    """Componentwise, relative to max(1, |w_k|)."""
    return all(abs(a - b) <= eps * max(1, abs(b)) for a, b in zip(z, w))


def rel_near(z, w, eps=Decimal("1e-55")):
    """Normwise relative: no floor at 1, so tiny values are compared too."""
    return max(abs(a - b) for a, b in zip(z, w)) <= eps * max(abs(v) for v in w)

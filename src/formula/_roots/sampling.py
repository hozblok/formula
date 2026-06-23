"""Reference backend: sample, bracket sign changes, refine (safeguarded Newton).

Not rigorous — even-multiplicity roots and sub-sample features can be missed.
Serves as the baseline/oracle that the rigorous backends are checked against.
"""

from typing import List

from ..formula import Number
from .utils import finite_sign, merge_close_roots, rtsafe


def find_all(
    func, t_min: Number, t_max: Number, precision: int, samples: int = 256, **_
) -> List[Number]:
    """All sign-change roots of g on [t_min, t_max], refined to precision."""
    xacc = Number(f"1e-{max(precision - 2, 1)}", precision)
    step = (t_max - t_min) / Number(samples, precision)
    roots: List[Number] = []
    t_prev = t_min
    s_prev = finite_sign(func.g(t_prev))
    if s_prev == 0:
        roots.append(t_prev)
    for i in range(1, samples + 1):
        t_cur = t_min + step * Number(i, precision)
        s_cur = finite_sign(func.g(t_cur))
        if s_cur == 0:
            roots.append(t_cur)
        elif s_prev is not None and s_cur is not None and s_prev * s_cur < 0:
            roots.append(rtsafe(func, t_prev, t_cur, xacc))
        t_prev, s_prev = t_cur, s_cur
    return merge_close_roots(roots, xacc)

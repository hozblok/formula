"""Reference backend: sample, bracket sign changes, refine (safeguarded Newton).

Not rigorous — even-multiplicity roots and sub-sample features can be missed.
Serves as the baseline/oracle that the rigorous backends are checked against.
"""

from ..formula import NumberContext
from .utils import finite_sign, merge_close_roots, rtsafe, tol_full


def find_all(
    func, t_min, t_max, precision: int, samples: int = 256, **_
) -> list:
    """All sign-change roots of g on [t_min, t_max], refined to precision."""
    xacc = tol_full(NumberContext(precision))
    step = (t_max - t_min) / samples
    roots = []
    t_prev = t_min
    s_prev = finite_sign(func.g(t_prev))
    if s_prev == 0:
        roots.append(t_prev)
    for i in range(1, samples + 1):
        t_cur = t_min + step * i
        s_cur = finite_sign(func.g(t_cur))
        if s_cur == 0:
            roots.append(t_cur)
        elif s_prev is not None and s_cur is not None and s_prev * s_cur < 0:
            root = rtsafe(func, t_prev, t_cur, xacc)  # None across a pole or a jump
            if root is not None:
                roots.append(root)
        t_prev, s_prev = t_cur, s_cur
    return merge_close_roots(roots, xacc)

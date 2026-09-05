"""Fresnel amplitude as a Formula expression with per-line optical constants.

The traced geometry is energy-independent, so a ray's amplitude re-evaluates
at any spectral line by substituting numeric 2*delta(E_m), 2*beta(E_m) into a
cached bounce-count template (variables s1..s_nb, dd, bb).
"""

from ..formula import Number
from .shared.nums import solver


def fresnel_expr(s: str = "s", dd: str = "dd", bb: str = "bb") -> str:
    """Complex r (s-pol) as a formula string; args are variable names or literals."""
    root = f"sqrt(({s})^2-({dd})+i*({bb}))"
    return f"((({s})-{root})/(({s})+{root}))"


def _product(nb: int) -> str:
    if nb < 1:
        raise ValueError("nb must be >= 1 (a bounce-free ray has amplitude 1)")
    return "*".join(fresnel_expr(f"s{j + 1}") for j in range(nb))


def ampl_template(nb: int, precision: int):
    """Cached Solver for prod_j r(s_j; dd, bb): variables s1..s_nb, dd, bb."""
    return solver(_product(nb), precision)


class LineAmplitudes:
    """Per-spectral-line amplitude products prod_j r(s_j; E_m) for one ray."""

    def __init__(self, material, lines, precision: int):
        two = Number("2", precision)
        # (2*delta, 2*beta) per line as full-precision strings for the template
        self.two_delta_beta = [
            (str(material.delta(l.e_kev, precision=precision) * two),
             str(material.beta(l.e_kev, precision=precision) * two))
            for l in lines]
        self.precision = precision

    def __call__(self, sins):
        """sins: sin(theta_j) per bounce (Number or full-precision str)."""
        if not sins:
            return [Number("1", self.precision)] * len(self.two_delta_beta)
        template = ampl_template(len(sins), self.precision)
        values = {f"s{j + 1}": str(s) for j, s in enumerate(sins)}
        out = []
        for dd, bb in self.two_delta_beta:
            values["dd"] = dd
            values["bb"] = bb
            out.append(template.number(values))
        return out

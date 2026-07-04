"""Fresnel amplitude as a Formula expression of the photon energy E [keV].

The traced geometry is energy-independent, so a ray's amplitude re-evaluates at
any energy: U(E) = exp(i*k(E)*L) * prod_j r(s_j; 2*delta(E), 2*beta(E)).
Templates are cached Solvers keyed by bounce count (variables s1..s_nb, E, L);
`ray_expression` bakes one ray's angles into a standalone expression of E.
"""

from ..xray import HC_KEV_ANGSTROM, R_E
from .nums import const, solver


def fresnel_expr(s: str = "s", dd: str = "dd", bb: str = "bb") -> str:
    """Complex r (s-pol) as a formula string; args are variable names or literals."""
    root = f"sqrt(({s})^2-({dd})-i*({bb}))"
    return f"((({s})-{root})/(({s})+{root}))"


def material_dd_bb(material) -> tuple:
    """(2*delta(E), 2*beta(E)) of a GlassMaterial as expression strings of E [keV]."""
    lam_m = f"(({HC_KEV_ANGSTROM})/E*1e-10)"
    dd = f"(({R_E})*{lam_m}^2*({material.electron_density})/pi)"
    bb = f"(2*({material.beta_ref})*(({material.energy_ref_kev})/E)^4)"
    return dd, bb


def _product(nb: int, material) -> str:
    if nb < 1:
        raise ValueError("nb must be >= 1 (a bounce-free ray has amplitude 1)")
    dd, bb = material_dd_bb(material)
    return "*".join(fresnel_expr(f"s{j + 1}", dd, bb) for j in range(nb))


def ampl_template(nb: int, material, precision: int):
    """Cached Solver for prod_j r(s_j; E): variables s1..s_nb and E."""
    return solver(_product(nb, material), precision)


def ray_field_template(nb: int, material, precision: int):
    """Cached Solver for U(E) = exp(i*k(E)*L) * prod_j r: variables s1..s_nb, L, E."""
    phase = f"exp(i*2*pi*E/(({HC_KEV_ANGSTROM})*1e-10)*L)"
    expr = phase if nb == 0 else f"{phase}*" + _product(nb, material)
    return solver(expr, precision)


def ray_expression(sins, material, opl=None) -> str:
    """One ray baked to a standalone expression of E: angles (and L) as literals."""
    dd, bb = material_dd_bb(material)
    terms = [fresnel_expr(str(s), dd, bb) for s in sins]
    if opl is not None:
        terms.insert(0, f"exp(i*2*pi*E/(({HC_KEV_ANGSTROM})*1e-10)*({opl}))")
    return "*".join(terms) if terms else "1"


class LineAmplitudes:
    """Per-spectral-line amplitude products prod_j r(s_j; E_m) for one ray."""

    def __init__(self, material, lines, precision: int):
        self.material = material
        self.energies = [str(line.e_kev) for line in lines]
        self.precision = precision
        self._one = const("1", precision)

    def __call__(self, sins):
        """sins: sin(theta_j) per bounce (Number or full-precision str)."""
        if not sins:
            return [self._one] * len(self.energies)
        template = ampl_template(len(sins), self.material, self.precision)
        values = {f"s{j + 1}": str(s) for j, s in enumerate(sins)}
        out = []
        for energy in self.energies:
            values["E"] = energy
            out.append(template.number(values))
        return out

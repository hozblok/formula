"""Bridge to the compiled mp_real/mp_complex backend classes.

The single Python place that encodes the mp_real_<P> / mp_complex_<P> naming.
"""

# pylint: disable=no-name-in-module, import-error
from typing import Union

from . import _formula

_REAL_PREFIX = "mp_real_"
_COMPLEX_PREFIX = "mp_complex_"


# Backend classes of every precision, real then complex, for isinstance() checks.
MP_TYPES = tuple(getattr(_formula, n) for prefix in (_REAL_PREFIX, _COMPLEX_PREFIX)
                 for n in dir(_formula) if n.startswith(prefix))
Scalar = Union[*MP_TYPES]

# Engine precision ceiling (csconstants::max_precision), independent of
# which mp_real_<P> wrappers are bound.
MAX_PRECISION = _formula.MAX_PRECISION

# The engine's rounding rule (single source of truth): smallest supported
# precision >= the request, i.e. what a value is stored at (0 if over the max).
round_up_precision = _formula.round_up_precision

# A literal at this storage precision: mp_real_<P>, or mp_complex_<P> for "(re,im)".
mp_number = _formula.mp_number

# Decimal text of a Python int/float read by value, exactly as the backend's
# arithmetic reads an operand; a subclass's __str__ never takes part.
scalar_text = _formula.scalar_text

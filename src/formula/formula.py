"""Solver wrapper to simplify calculations with Formula."""

import operator
import re
from functools import cached_property
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Optional, Union

# pylint: disable=no-name-in-module, import-error
from ._formula import FmtFlags, Formula
from .constants import DEFAULT_CASE_INSENSITIVE, DEFAULT_IMAGINARY_UNIT, DEFAULT_PRECISION
from .backend import COMPLEX_TYPES, MAX_PRECISION, MP_TYPES, mp_class, mp_precision, round_up_precision

class Solver(Formula):
    """Solver for calculating string formulas.

    Example:
        solver = Solver("A*acos(x)", precision=24)
        print("pi:", solver({"A": 2, "x": 0}))
    """

    def __init__(
        self,
        expression: str,
        precision: int = DEFAULT_PRECISION,
        imaginary_unit: str = DEFAULT_IMAGINARY_UNIT,
        case_insensitive: bool = DEFAULT_CASE_INSENSITIVE,
    ):
        if not 0 <= precision <= MAX_PRECISION:
            raise ValueError(
                f"precision must be in [0, {MAX_PRECISION}] (got {precision})"
            )
        super().__init__(expression, precision, imaginary_unit, case_insensitive)

    def _coerce_variables_to_values(
        self, values: Optional[Union[Dict[str, Any], Any]]
    ) -> Dict[str, str]:
        if not isinstance(values, Mapping):
            variables = self.variables()
            if not variables:
                if values is not None:
                    raise ValueError(
                        f"The formula has no variables, but 'values' was "
                        f"given: {values!r}"
                    )
                values = {}
            elif values is None:
                raise ValueError(f"Missing values for variables: {variables}")
            elif len(variables) == 1:
                (only_var,) = variables
                values = {only_var: values}
            else:
                raise ValueError(
                    f"Expected a Mapping for 'values' (got "
                    f"{type(values).__name__}); variables to provide: {variables}"
                )
        return {k: v if isinstance(v, str) else str(v) for k, v in values.items()}

    def __call__(
        self,
        values: Optional[Union[Dict[str, Any], Any]] = None,
        derivative: Optional[Union[str, Iterable[str]]] = None,
        format_digits: Optional[int] = None,
        format_flags=FmtFlags.default,
    ) -> Union[List[str], str]:
        """Calculate the value of the parsed formula string.

        Args:
            values: dict contains names of variables as keys and their
                values as items respectively. Or any value that can be
                cast to a string if the formula contains only one
                variable. If there is the only one variable in the expression, it is
                possible to put only the value to `values` argument.
            derivative: string variable name by which it is necessary
                to calculate the partial derivative. Or iterable of
                string values. If omitted, the derivative will not
                be calculated.
            format_digits: Result value returns formatted with at least precision
                format_digits. The original value will be rounded if required.
                Set to 0 if you want to get all available digits from a value in memory
                (all the numbers involved in the calculations will be displayed,
                including odd numbers outside the precision limit).
                The default is the same as the precision limit.
            format_flags: Flags to format the return value.
                If omitted output is generated in fixed-point notation and
                decimal base.
        Returns:
            Calculated value: string or the list of strings if the
                'derivative' parameter is passed.
        """

        if format_digits is None:
            format_digits = self.precision

        variables_to_values = self._coerce_variables_to_values(values)

        if derivative is None:
            return self.get(variables_to_values, format_digits, format_flags)
        if isinstance(derivative, str):
            return self.get_derivative(
                derivative, variables_to_values, format_digits, format_flags
            )
        try:
            names = list(derivative)
        except TypeError as ex:
            raise ValueError(
                "The value of the 'derivative' is not"
                f" a string or iterable! Its type is {type(derivative)}."
            ) from ex
        return [
            self.get_derivative(name, variables_to_values, format_digits, format_flags)
            for name in names
        ]

    def pair(
        self,
        values: Optional[Union[Dict[str, Any], Any]] = None,
        format_digits: Optional[int] = None,
        format_flags=FmtFlags.default,
    ) -> tuple[str, str]:
        """Calculate the value and return it as a (real_str, imag_str) pair.

        Same `values`/`format_digits`/`format_flags` semantics as __call__.
        Both parts are formatted with the same digits and format so a real
        expression and a complex expression whose imaginary part is exactly
        zero produce byte-equal pairs — that property is what makes
        Number.__eq__ work consistently across real/complex syntactic forms.
        """
        if format_digits is None:
            format_digits = self.precision
        variables_to_values = self._coerce_variables_to_values(values)
        return self.get_pair(variables_to_values, format_digits, format_flags)

    def number(
        self, values: Optional[Union[Dict[str, Any], Any]] = None
    ) -> "Number":
        """Evaluate to a Number at this solver's precision (no string round-trip).

        Same `values` semantics as __call__.
        """
        return Number(self.evaluate(self._coerce_variables_to_values(values)))


class Number:
    """Arbitrary-precision real/complex value backed by mp_real/mp_complex.

    The constructor parses and evaluates any Formula expression once (so
    "sin(pi/8)", "3+4*i", "1/3" all work), wrapping the C++ mp value directly;
    arithmetic, comparison and equality then run on it with no re-parsing or
    per-step rounding. The returned mp type carries the real/complex kind.
    """

    DEFAULT_PRECISION = DEFAULT_PRECISION

    _FROZEN_ATTRS = frozenset({"_value", "_precision", "_is_complex"})

    # Plain real literals need no Formula parse; the mp constructor takes them directly.
    _PLAIN_REAL = re.compile(r"^-?[0-9]+(\.[0-9]*)?([eE][+-]?[0-9]+)?$")

    def __setattr__(self, name: str, value) -> None:
        if name in self._FROZEN_ATTRS and hasattr(self, name):
            raise AttributeError(f"{name} is read-only after construction")
        super().__setattr__(name, value)

    def __init__(
        self,
        expression: Union["Number", str, int, float, *MP_TYPES],
        precision: Optional[int] = None,
    ):
        if precision is not None and not 0 < precision <= MAX_PRECISION:
            raise ValueError(f"precision must be in [1, {MAX_PRECISION}] (got {precision})")

        rounded_precision = round_up_precision(precision or self.DEFAULT_PRECISION)

        if isinstance(expression, bool):
            raise TypeError("bool is not a valid Number expression")
        if isinstance(expression, Number):
            self._value = expression._value
            self._precision = expression._precision
            self._is_complex = expression._is_complex
            if precision and self._precision != rounded_precision:
                raise ValueError(
                    f"precision mismatch: {self._precision} != {rounded_precision}"
                )
            return
        if isinstance(expression, MP_TYPES):
            self._value = expression
            self._precision = mp_precision(expression)
            if precision and self._precision != rounded_precision:
                raise ValueError(
                    f"precision mismatch: {self._precision} != {rounded_precision}"
                )
        elif isinstance(expression, (str, int, float)):
            text = str(expression)
            if self._PLAIN_REAL.match(text):
                self._value = mp_class(rounded_precision)(text)
            else:
                self._value = Solver(text, precision=rounded_precision).evaluate()
            self._precision = rounded_precision
        else:
            raise TypeError(
                f"Number expression must be Number, str, int, float, or mp_*; "
                f"got {type(expression).__name__}"
            )
        self._is_complex = isinstance(self._value, COMPLEX_TYPES)

    @classmethod
    def _wrap(cls, value, precision: int, is_complex: bool) -> "Number":
        """Fast constructor for a raw mp value of known precision/kind (hot path)."""
        n = cls.__new__(cls)
        n._value = value
        n._precision = precision
        n._is_complex = is_complex
        return n

    @property
    def is_complex(self) -> bool:
        return self._is_complex

    @property
    def precision(self) -> int:
        return self._precision

    @cached_property
    def parts(self) -> tuple[str, str]:
        """(real, imaginary) as formatted strings at this precision."""
        fmt = FmtFlags.default
        p = self._precision
        if self._is_complex:
            return self._value.real(p, fmt), self._value.imag(p, fmt)
        return self._value.str(p, fmt), "0"

    @property
    def real(self) -> "Number":
        """Real part as a real Number at this precision."""
        if self.is_complex:
            return Number(self._value.real(0, FmtFlags.default), self.precision)
        return Number(self._value)

    @property
    def imag(self) -> "Number":
        """Imaginary part as a real Number at this precision."""
        if self.is_complex:
            return Number(self._value.imag(0, FmtFlags.default), self.precision)
        return Number(0, self.precision)

    def _coerce(self, value) -> "Number":
        """Rhs at self.precision; raises ValueError on storage-precision mismatch."""
        if isinstance(value, Number) and value.precision == self.precision:
            return value
        return Number(value, self.precision)

    def _binop(self, __value, op) -> "Number":
        b = self._coerce(__value)
        a, bb = self._value, b._value
        if self._is_complex != b._is_complex:
            cls = mp_class(self._precision, is_complex=True)
            fmt = FmtFlags.default
            if not self._is_complex:
                a = cls(a.str(0, fmt))
            if not b._is_complex:
                bb = cls(bb.str(0, fmt))
        return Number._wrap(op(a, bb), self._precision,
                            self._is_complex or b._is_complex)

    @cached_property
    def cmp_key(self):
        """Real mp value at display precision — the ordering key for reals."""
        return mp_class(self._precision)(self.parts[0])

    def _cmp(self, __value: object, op):
        b = self._coerce(__value)
        if self._is_complex or b._is_complex:
            raise TypeError("complex numbers are not orderable")
        # Compare at display precision so ordering agrees with ==: guard digits
        # past parts() must not make equal-when-formatted values differ.
        return op(self.cmp_key, b.cmp_key)

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, (Number, *MP_TYPES)):
            # Like Number-vs-Number, a raw mp value keeps its own precision.
            other = __value if isinstance(__value, Number) else Number(__value)
            return self.parts == other.parts
        if not isinstance(__value, (str, int, float)):
            return NotImplemented
        return self.parts == self._coerce(__value).parts

    def __hash__(self) -> int:
        return hash(self.parts)

    def __bool__(self) -> bool:
        return self.parts != ("0", "0")

    def __str__(self) -> str:
        r, i = self.parts
        if i == "0":
            return r
        sign = "-" if i.startswith("-") else "+"
        mag = i.lstrip("-")
        if r == "0":
            return f"-{mag}*i" if sign == "-" else f"{mag}*i"
        return f"{r}{sign}{mag}*i"

    def __repr__(self) -> str:
        return f"Number({str(self)!r}, precision={self.precision})"

    def __float__(self) -> float:
        r, i = self.parts
        if i != "0":
            raise TypeError("cannot convert complex Number to float")
        return float(r)

    def __complex__(self) -> complex:
        r, i = self.parts
        return complex(float(r), float(i))

    def sqrt(self) -> "Number":
        """Square root: native mp sqrt for reals; complex falls back to ^0.5."""
        if self._is_complex:
            return self ** Number("0.5", self._precision)
        return Number._wrap(self._value.sqrt(), self._precision, False)

    def __abs__(self) -> "Number":
        if self._is_complex:
            return Number(abs(self._value))     # kind of |z| comes from the backend
        return Number._wrap(abs(self._value), self._precision, False)

    def __neg__(self) -> "Number":
        return Number._wrap(-self._value, self._precision, self._is_complex)

    def __add__(self, __value) -> "Number":
        return self._binop(__value, operator.add)

    def __sub__(self, __value) -> "Number":
        return self._binop(__value, operator.sub)

    def __mul__(self, __value) -> "Number":
        return self._binop(__value, operator.mul)

    def __truediv__(self, __value) -> "Number":
        return self._binop(__value, operator.truediv)

    def __pow__(self, __value) -> "Number":
        return self._binop(__value, operator.pow)

    def __radd__(self, __value): return self._coerce(__value) + self
    def __rsub__(self, __value): return self._coerce(__value) - self
    def __rmul__(self, __value): return self._coerce(__value) * self
    def __rtruediv__(self, __value): return self._coerce(__value) / self
    def __rpow__(self, __value): return self._coerce(__value) ** self

    def __ge__(self, __value): return self._cmp(__value, operator.ge)
    def __gt__(self, __value): return self._cmp(__value, operator.gt)
    def __le__(self, __value): return self._cmp(__value, operator.le)
    def __lt__(self, __value): return self._cmp(__value, operator.lt)

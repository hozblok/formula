"""Solver wrapper to simplify calculations with Formula."""

import operator
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Optional, Union

# pylint: disable=no-name-in-module, import-error
from ._formula import FmtFlags, Formula
from .constants import DEFAULT_CASE_INSENSITIVE, DEFAULT_IMAGINARY_UNIT
from .backend import COMPLEX_TYPES, MAX_PRECISION, mp_class


class Solver(Formula):
    """Solver for calculating string formulas.

    Example:
        solver = Solver("A*acos(x)", precision=24)
        print("pi:", solver({"A": 2, "x": 0}))
    """

    def __init__(
        self,
        expression: str,
        precision: int = 24,
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
        if values is None:
            variables_to_values: Dict[str, str] = {}
        elif isinstance(values, Mapping):
            variables_to_values = dict(values)
        else:
            variables_to_values = values
        if not isinstance(values, Mapping):
            variables = self.variables()
            if not variables:
                if values is not None:
                    raise ValueError(
                        f"The formula has no variables, but 'values' was "
                        f"given: {values!r}"
                    )
            elif values is not None and len(variables) == 1:
                (only_var,) = variables
                variables_to_values = {only_var: str(values)}
            elif values is None:
                raise ValueError(
                    f"Missing values for variables: {variables}"
                )
            else:
                raise ValueError(
                    f"Expected a Mapping for 'values' (got "
                    f"{type(values).__name__}); variables to provide: {variables}"
                )

        for key in variables_to_values:
            val = variables_to_values[key]
            if not isinstance(val, str):
                variables_to_values[key] = str(val)

        return variables_to_values

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

        result = None
        if derivative is not None:
            if isinstance(derivative, str):
                result = self.get_derivative(
                    derivative, variables_to_values, format_digits, format_flags
                )
            else:
                try:
                    iter(derivative)
                except TypeError as ex:
                    raise ValueError(
                        f"'derivative' must be a str or iterable of str "
                        f"(got {type(derivative).__name__})"
                    ) from ex
                result = [
                    self.get_derivative(
                        der, variables_to_values, format_digits, format_flags
                    )
                    for der in derivative
                ]
        else:
            result = self.get(variables_to_values, format_digits, format_flags)

        return result

    def pair(
        self,
        values: Optional[Union[Dict[str, Any], Any]] = None,
        format_digits: Optional[int] = None,
        format_flags=FmtFlags.default,
    ) -> tuple:
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


class Number:
    """Arbitrary-precision real/complex value backed by mp_real/mp_complex.

    The constructor parses and evaluates any Formula expression once (so
    "sin(pi/8)", "3+4*i", "1/3" all work), wrapping the C++ mp value directly;
    arithmetic, comparison and equality then run on it with no re-parsing or
    per-step rounding. The returned mp type carries the real/complex kind.
    """

    def __init__(
        self,
        expression: Union["Number", str, int, float],
        precision: int = 24,
    ):
        if isinstance(expression, bool) or not isinstance(
            expression, (Number, str, int, float)
        ):
            raise TypeError(
                f"Number expression must be Number, str, int, or float; "
                f"got {type(expression).__name__}"
            )
        solver = Solver(str(expression), precision=precision)
        self._value = solver.evaluate()
        self._precision = solver.precision  # rounded up to a supported precision

    @classmethod
    def _wrap(cls, value, precision: int) -> "Number":
        obj = cls.__new__(cls)
        obj._value = value
        obj._precision = precision
        return obj

    @classmethod
    def wrap(cls, value, precision: int) -> "Number":
        """Wrap a backend mp value (e.g. from Solver.evaluate) without re-parsing."""
        return cls._wrap(value, precision)

    # Real/complex kind, derived from the wrapped mp value (its sole source).
    @property
    def _is_complex(self) -> bool:
        return isinstance(self._value, COMPLEX_TYPES)

    @property
    def precision(self) -> int:
        """Rounded precision (decimal digits) the value is stored at."""
        return self._precision

    @property
    def is_complex(self) -> bool:
        return self._is_complex

    def parts(self) -> tuple:
        """(real, imaginary) as formatted strings at this precision."""
        return self._pair()

    # Coerce a foreign value to Number at this precision; the validation
    # boundary that rejects bool/None/list/etc. with a clear TypeError.
    def _as_number(self, value: object) -> "Number":
        return (
            value
            if isinstance(value, Number)
            else Number(value, precision=self._precision)
        )

    # Bring self and other to a common mp type/precision for a binary op: the
    # higher precision and complex win; a differing value is rebuilt from strings.
    def _align(self, other: "Number"):
        precision = max(self._precision, other._precision)
        is_complex = self._is_complex or other._is_complex
        cls = mp_class(precision, is_complex)

        def to_common(n: "Number"):
            if n._precision == precision and n._is_complex == is_complex:
                return n._value
            if n._is_complex:
                real, imag = n._value.real(0, FmtFlags.default), n._value.imag(0, FmtFlags.default)
            else:
                real, imag = n._value.str(0, FmtFlags.default), "0"
            return cls(real, imag) if is_complex else cls(real)

        return to_common(self), to_common(other), precision, is_complex

    def _pair(self) -> tuple[str, str]:
        fmt = FmtFlags.default
        p = self._precision
        if self._is_complex:
            return self._value.real(p, fmt), self._value.imag(p, fmt)
        return self._value.str(p, fmt), "0"

    def _binop(self, __value: object, op) -> "Number":
        a, b, precision, _ = self._align(self._as_number(__value))
        return Number._wrap(op(a, b), precision)

    def _cmp(self, __value: object, op):
        if not isinstance(__value, (Number, str, int, float)):
            return NotImplemented
        a, b, _, is_complex = self._align(self._as_number(__value))
        if is_complex:
            raise TypeError("complex numbers are not orderable")
        return op(a, b)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, (Number, str, int, float)):
            return NotImplemented
        return self._pair() == self._as_number(__value)._pair()

    def __hash__(self) -> int:
        return hash(self._pair())

    def __str__(self) -> str:
        r, i = self._pair()
        if i == "0":
            return r
        sign = "-" if i.startswith("-") else "+"
        mag = i.lstrip("-")
        if r == "0":
            return f"-{mag}*i" if sign == "-" else f"{mag}*i"
        return f"{r}{sign}{mag}*i"

    def __repr__(self) -> str:
        return f"Number({str(self)!r}, precision={self._precision})"

    def __abs__(self) -> "Number":
        return Number._wrap(abs(self._value), self._precision)

    def __neg__(self) -> "Number":
        return Number._wrap(-self._value, self._precision)

    def __add__(self, __value: Union[str, int, float, "Number"]) -> "Number":
        return self._binop(__value, operator.add)

    def __sub__(self, __value: Union[str, int, float, "Number"]) -> "Number":
        return self._binop(__value, operator.sub)

    def __mul__(self, __value: Union[str, int, float, "Number"]) -> "Number":
        return self._binop(__value, operator.mul)

    def __truediv__(self, __value: Union[str, int, float, "Number"]) -> "Number":
        return self._binop(__value, operator.truediv)

    def __pow__(self, __value: Union[str, int, float, "Number"]) -> "Number":
        return self._binop(__value, operator.pow)

    def __radd__(self, __value): return self._as_number(__value) + self
    def __rsub__(self, __value): return self._as_number(__value) - self
    def __rmul__(self, __value): return self._as_number(__value) * self
    def __rtruediv__(self, __value): return self._as_number(__value) / self
    def __rpow__(self, __value): return self._as_number(__value) ** self

    def __ge__(self, __value): return self._cmp(__value, operator.ge)
    def __gt__(self, __value): return self._cmp(__value, operator.gt)
    def __le__(self, __value): return self._cmp(__value, operator.le)
    def __lt__(self, __value): return self._cmp(__value, operator.lt)

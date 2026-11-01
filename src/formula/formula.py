"""Solver wrapper to simplify calculations with Formula."""

from dataclasses import dataclass
from functools import cached_property, lru_cache
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Optional, Union

# pylint: disable=no-name-in-module, import-error
from ._formula import FmtFlags, Formula
from .constants import DEFAULT_CASE_INSENSITIVE, DEFAULT_IMAGINARY_UNIT, DEFAULT_PRECISION
from .backend import MAX_PRECISION, MP_TYPES, mp_number, round_up_precision

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
        result = {}
        for name, value in values.items():
            # A typed value keeps its storage precision; text is the explicit conversion.
            if isinstance(value, MP_TYPES) and value.precision != self.precision:
                raise ValueError(f"precision mismatch: {name} is {value.precision}, "
                                 f"the solver {self.precision}")
            if isinstance(value, MP_TYPES) and not value.is_complex:
                result[name] = value.str(0, FmtFlags.default)
            else:
                result[name] = value if isinstance(value, str) else str(value)
        return result

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
        zero produce byte-equal pairs.
        """
        if format_digits is None:
            format_digits = self.precision
        variables_to_values = self._coerce_variables_to_values(values)
        return self.get_pair(variables_to_values, format_digits, format_flags)

    def number(self, values: Optional[Union[Dict[str, Any], Any]] = None):
        """Evaluate to an mp_real/mp_complex value at this solver's precision.

        Same `values` semantics as __call__; no string round-trip.
        """
        return self.evaluate(self._coerce_variables_to_values(values))


@dataclass(frozen=True)
class NumberContext:
    """mp_real/mp_complex factory with a fixed storage precision and no global state.

    N = NumberContext(precision=64)
    x = N("1e-30")
    z = 2 * x + N.i

    Literals only: a real one or the backend's own "(re,im)" text; an
    expression belongs to Solver. Existing backend values must have this
    storage precision; they are not rounded.
    """

    precision: int = DEFAULT_PRECISION

    def __post_init__(self):
        if isinstance(self.precision, bool) or not isinstance(self.precision, int):
            raise TypeError("precision must be an int")
        if not 0 < self.precision <= MAX_PRECISION:
            raise ValueError(
                f"precision must be in [1, {MAX_PRECISION}] (got {self.precision})"
            )
        object.__setattr__(self, "precision", round_up_precision(self.precision))

    def __call__(self, value: Union[str, int, float, *MP_TYPES]):
        if isinstance(value, MP_TYPES):
            if value.precision != self.precision:
                raise ValueError(
                    f"precision mismatch: {value.precision} != {self.precision}"
                )
            return value
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise TypeError(
                f"value must be str, int, float, or mp_*; got {type(value).__name__}"
            )
        return mp_number(self.precision, str(value))

    @staticmethod
    @lru_cache(maxsize=512)
    def _solver(expression: str, precision: int) -> Solver:
        return Solver(expression, precision)

    def eval(self, expression: str, **values):
        """Expression evaluated at this precision; variables by keyword (text or backend value)."""
        return self._solver(expression, self.precision).number(values or None)

    @cached_property
    def i(self):
        return self.eval("i")

    @cached_property
    def pi(self):
        return self.eval("pi")

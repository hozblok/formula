"""Solver wrapper to simplify calculations with Formula."""

from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Optional, Union

# pylint: disable=no-name-in-module, import-error
from ._formula import FmtFlags, Formula

MAX_PRECISION = 8192


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
        imaginary_unit: str = "i",
        case_insensitive: bool = False,
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
                pass
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
                    result = [
                        self.get_derivative(
                            der, variables_to_values, format_digits, format_flags
                        )
                        for der in derivative
                    ]
                except TypeError as ex:
                    raise ValueError(
                        "The value of the 'derivative' is not"
                        " a string or iterable! Its type is %s." % type(derivative)
                    ) from ex
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
    def __init__(
        self,
        expression: Union["Number", str, int, float],
        precision: int = 24,
    ):
        if isinstance(expression, Number):
            self._expression = expression.expression
        elif isinstance(expression, bool):
            raise TypeError(
                "Number expression must be Number, str, int, or float; got bool"
            )
        elif isinstance(expression, (str, int, float)):
            self._expression = str(expression)
        else:
            raise TypeError(
                f"Number expression must be Number, str, int, or float; "
                f"got {type(expression).__name__}"
            )
        self._precision = precision
        self._imaginary_unit = 'i'
        self._case_insensitive = False

    @property
    def expression(self):
        return self._expression

    @property
    def params(self) -> dict[str, Any]:
        return {
            "precision": self._precision,
            "imaginary_unit": self._imaginary_unit,
            "case_insensitive": self._case_insensitive,
        }

    @property
    def fixed(self) -> str:
        return Solver(self._expression, **self.params)(format_flags=FmtFlags.fixed)

    @property
    def pair_fixed(self) -> tuple[str, str]:
        return Solver(self._expression, **self.params).pair(
            format_flags=FmtFlags.fixed
        )

    def __make_operation(self, __value: object, operator: str) -> "Number":
        # Wrapping non-Number inputs in Number() is the validation boundary for
        # arithmetic: it rejects bool/None/list/dict/etc. with a clear TypeError
        # naming the offending type, instead of letting str(__value) silently
        # produce a malformed Solver expression that fails deeper in parsing.
        other = (
            __value
            if isinstance(__value, Number)
            else Number(__value, precision=self._precision)
        )
        solver = Solver(
            f"(({self.expression}) {operator} ({other.expression}))", **self.params
        )
        return Number(solver(), precision=self._precision)

    def __prepare_comparison(self, __value: object) -> List[str]:
        other = (
            __value
            if isinstance(__value, Number)
            else Number(__value, precision=self._precision)
        )
        return [self.fixed, other.fixed]

    def __make_comparison(self, left: str, right: str, operator: str) -> bool:
        solver = Solver(f"(({left}) {operator} ({right}))", **self.params)
        return solver(format_digits=1) == "1"

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, (Number, str, int, float)):
            return NotImplemented
        other = (
            __value
            if isinstance(__value, Number)
            else Number(__value, precision=self._precision)
        )
        return self.pair_fixed == other.pair_fixed

    def __hash__(self) -> int:
        return hash(self.pair_fixed)

    def __str__(self) -> str:
        return self.expression

    def __abs__(self) -> "Number":
        solver = Solver(f"abs({self.expression})", **self.params)
        return Number(solver(), precision=self._precision)

    def __add__(self, __value: Union[str, int, float, "Number"]) -> "Number":
        return self.__make_operation(__value, "+")

    def __sub__(self, __value: Union[str, int, float, "Number"]) -> "Number":
        return self.__make_operation(__value, "-")

    def __mul__(self, __value: Union[str, int, float, "Number"]) -> "Number":
        return self.__make_operation(__value, "*")

    def __truediv__(self, __value: Union[str, int, float, "Number"]) -> "Number":
        return self.__make_operation(__value, "/")

    def __pow__(self, __value: Union[str, int, float, "Number"]) -> "Number":
        return self.__make_operation(__value, "^")

    def __radd__(self, __value: Union[str, int, float]) -> "Number":
        return Number(__value, precision=self._precision).__add__(self)

    def __rsub__(self, __value: Union[str, int, float]) -> "Number":
        return Number(__value, precision=self._precision).__sub__(self)

    def __rmul__(self, __value: Union[str, int, float]) -> "Number":
        return Number(__value, precision=self._precision).__mul__(self)

    def __rtruediv__(self, __value: Union[str, int, float]) -> "Number":
        return Number(__value, precision=self._precision).__truediv__(self)

    def __rpow__(self, __value: Union[str, int, float]) -> "Number":
        return Number(__value, precision=self._precision).__pow__(self)

    def __ge__(self, __value: Union[str, int, float, "Number"]) -> bool:
        if not isinstance(__value, (Number, str, int, float)):
            return NotImplemented
        left, right = self.__prepare_comparison(__value)
        if left == right:
            return True
        return self.__make_comparison(left, right, ">")

    def __gt__(self, __value: Union[str, int, float, "Number"]) -> bool:
        if not isinstance(__value, (Number, str, int, float)):
            return NotImplemented
        left, right = self.__prepare_comparison(__value)
        if left == right:
            return False
        return self.__make_comparison(left, right, ">")

    def __le__(self, __value: Union[str, int, float, "Number"]) -> bool:
        if not isinstance(__value, (Number, str, int, float)):
            return NotImplemented
        left, right = self.__prepare_comparison(__value)
        if left == right:
            return True
        return self.__make_comparison(left, right, "<")

    def __lt__(self, __value: Union[str, int, float, "Number"]) -> bool:
        if not isinstance(__value, (Number, str, int, float)):
            return NotImplemented
        left, right = self.__prepare_comparison(__value)
        if left == right:
            return False
        return self.__make_comparison(left, right, "<")

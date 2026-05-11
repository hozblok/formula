"""Solver wrapper to simplify calculations with Formula."""

from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Optional, Union

# pylint: disable=no-name-in-module, import-error
from ._formula import FmtFlags, Formula


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
        # pylint: disable=useless-super-delegation
        super().__init__(expression, precision, imaginary_unit, case_insensitive)

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

        variables_to_values: Dict[str, str] = dict() if values is None else values
        if not isinstance(values, Mapping):
            variables = self.variables()
            if not variables:
                pass
            elif values is not None and len(variables) == 1:
                variables_to_values = {variables.pop(): str(values)}
            else:
                raise ValueError(
                    "The value of the 'values' parameter is not a dict!"
                    " Its type is %s A dictionary is expected with values for"
                    " the following variables: %s" % (type(values), str(variables))
                )

        for key in variables_to_values:
            val = variables_to_values[key]
            if not isinstance(val, str):
                variables_to_values[key] = str(val)

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


class Number:
    def __init__(
        self,
        expression: Union["Number", str, int, float],
        precision: int = 24,
    ):
        if isinstance(expression, Number):
            self._expression = expression.expression
        elif isinstance(expression, bool):
            # bool is an int subclass; reject explicitly so Number(True) doesn't
            # silently become Number("True").
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

    @property
    def expression(self):
        return self._expression

    @property
    def params(self) -> dict:
        return {"precision": self._precision}

    def __make_operation(self, __value: object, operator: str) -> "Number":
        other = (
            __value
            if isinstance(__value, Number)
            else Number(__value, precision=self._precision)
        )
        solver = Solver(
            f"(({self.expression}) {operator} ({other.expression}))", **self.params
        )
        return Number(solver(), **self.params)

    def __prepare_comparison(self, __value: object) -> List[str]:
        left = Solver(self.expression, **self.params)(format_flags=FmtFlags.fixed)
        other = (
            __value
            if isinstance(__value, Number)
            else Number(__value, precision=self._precision)
        )
        right = Solver(other.expression, **self.params)(format_flags=FmtFlags.fixed)
        return [left, right]

    def __make_comparison(self, left: str, right: str, operator: str) -> bool:
        solver = Solver(f"(({left}) {operator} ({right}))")
        return solver(format_digits=1) == "1"

    def __eq__(self, __value: object) -> bool:
        left, right = self.__prepare_comparison(__value)
        return left == right

    def __str__(self) -> str:
        return self.expression

    def __abs__(self) -> "Number":
        solver = Solver(f"abs({self.expression})", **self.params)
        return Number(solver(), **self.params)

    def __add__(self, __value: Union["Number", str, int, float]) -> "Number":
        return self.__make_operation(__value, "+")

    def __sub__(self, __value: Union["Number", str, int, float]) -> "Number":
        return self.__make_operation(__value, "-")

    def __mul__(self, __value: Union["Number", str, int, float]) -> "Number":
        return self.__make_operation(__value, "*")

    def __truediv__(self, __value: Union["Number", str, int, float]) -> "Number":
        return self.__make_operation(__value, "/")

    def __pow__(self, __value: Union["Number", str, int, float]) -> "Number":
        return self.__make_operation(__value, "^")

    def __radd__(self, __value: Union["Number", str, int, float]) -> "Number":
        return Number(__value, precision=self._precision).__add__(self)

    def __rsub__(self, __value: Union["Number", str, int, float]) -> "Number":
        return Number(__value, precision=self._precision).__sub__(self)

    def __rmul__(self, __value: Union["Number", str, int, float]) -> "Number":
        return Number(__value, precision=self._precision).__mul__(self)

    def __rtruediv__(self, __value: Union["Number", str, int, float]) -> "Number":
        return Number(__value, precision=self._precision).__truediv__(self)

    def __rpow__(self, __value: Union["Number", str, int, float]) -> "Number":
        return Number(__value, precision=self._precision).__pow__(self)

    def __ge__(self, __value: Union["Number", str, int, float]) -> bool:
        left, right = self.__prepare_comparison(__value)
        if left == right:
            return True
        return self.__make_comparison(left, right, ">")

    def __gt__(self, __value: Union["Number", str, int, float]) -> bool:
        left, right = self.__prepare_comparison(__value)
        if left == right:
            return False
        return self.__make_comparison(left, right, ">")

    def __le__(self, __value: Union["Number", str, int, float]) -> bool:
        left, right = self.__prepare_comparison(__value)
        if left == right:
            return True
        return self.__make_comparison(left, right, "<")

    def __lt__(self, __value: Union["Number", str, int, float]) -> bool:
        left, right = self.__prepare_comparison(__value)
        if left == right:
            return False
        return self.__make_comparison(left, right, "<")

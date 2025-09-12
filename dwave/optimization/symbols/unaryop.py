# Copyright 2025 D-Wave
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dwave.optimization.symbols._unaryop import _UnaryOpSymbol, _UnaryOpNodeType


class Absolute(_UnaryOpSymbol, node_type=_UnaryOpNodeType.Absolute):
    """Absolute value element-wise on a symbol.

    See Also:
        :func:`~dwave.optimization.mathematical.absolute`: equivalent function.
    """
    pass


class Cos(_UnaryOpSymbol, node_type=_UnaryOpNodeType.Cos):
    """Cosine element-wise on a symbol.

    See Also:
        :func:`~dwave.optimization.mathematical.cos`: equivalent function.

    .. versionadded:: 0.6.5
    """
    pass


class Exp(_UnaryOpSymbol, node_type=_UnaryOpNodeType.Exp):
    """Takes the values of a symbol and returns the corresponding base-e exponential.

    See Also:
        :func:`~dwave.optimization.mathematical.exp`: equivalent function.

    .. versionadded:: 0.6.2
    """
    pass


class Expit(_UnaryOpSymbol, node_type=_UnaryOpNodeType.Expit):
    """Takes the values of a symbol and returns the corresponding logistic sigmoid (expit).

    See Also:
        :func:`~dwave.optimization.mathematical.expit`: equivalent function.

    .. versionadded:: 0.5.2
    """
    pass


class Log(_UnaryOpSymbol, node_type=_UnaryOpNodeType.Log):
    """Takes the values of a symbol and returns the corresponding natural logarithm (log).

    See Also:
        :func:`~dwave.optimization.mathematical.log`: equivalent function.

    .. versionadded:: 0.5.2
    """
    pass


class Logical(_UnaryOpSymbol, node_type=_UnaryOpNodeType.Logical):
    """Logical truth value element-wise on a symbol.

    See Also:
        :func:`~dwave.optimization.mathematical.logical`: equivalent function.
    """
    pass


class Negative(_UnaryOpSymbol, node_type=_UnaryOpNodeType.Negative):
    """Numerical negative element-wise on a symbol.

    Examples:
        This example add the negative of an integer array.

        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> i = model.integer(100, upper_bound=50)
        >>> i_minus = -i
        >>> type(i_minus)
        <class 'dwave.optimization.symbols.Negative'>
    """
    pass


class Not(_UnaryOpSymbol, node_type=_UnaryOpNodeType.Not):
    """Logical negation element-wise on a symbol.

    See Also:
        :func:`~dwave.optimization.mathematical.logical_not`: equivalent function.
    """
    pass


class Rint(_UnaryOpSymbol, node_type=_UnaryOpNodeType.Rint):
    """Takes the values of a symbol and rounds them to the nearest integer.

    Examples:
        This example adds the round-int of a decision
        variable to a model.

        >>> from dwave.optimization.model import Model
        >>> from dwave.optimization.mathematical import rint
        >>> model = Model()
        >>> i = model.constant(10.4)
        >>> ii = rint(i)
        >>> type(ii)
        <class 'dwave.optimization.symbols.Rint'>
    """
    pass


class Sin(_UnaryOpSymbol, node_type=_UnaryOpNodeType.Sin):
    """Sine element-wise on a symbol.

    See Also:
        :func:`~dwave.optimization.mathematical.sin`: equivalent function.

    .. versionadded:: 0.6.5
    """
    pass


class Square(_UnaryOpSymbol, node_type=_UnaryOpNodeType.Square):
    """Squares element-wise of a symbol.

    Examples:
        This example adds the squares of an integer decision
        variable to a model.

        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> i = model.integer(10, lower_bound=-5, upper_bound=5)
        >>> ii = i**2
        >>> type(ii)
        <class 'dwave.optimization.symbols.Square'>
    """
    pass


class SquareRoot(_UnaryOpSymbol, node_type=_UnaryOpNodeType.SquareRoot):
    """Square root of a symbol.

    Examples:
        This example adds the square root of an integer decision variable to a
        model.

        >>> from dwave.optimization.model import Model
        >>> from dwave.optimization.mathematical import sqrt
        >>> model = Model()
        >>> i = model.constant(10)
        >>> ii = sqrt(i)
        >>> type(ii)
        <class 'dwave.optimization.symbols.SquareRoot'>
    """
    pass

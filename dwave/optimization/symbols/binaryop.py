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

from dwave.optimization.symbols._binaryop import _BinaryOpSymbol, _BinaryOpNodeType


class Add(_BinaryOpSymbol, node_type=_BinaryOpNodeType.Add):
    """Addition element-wise of two symbols.

    Examples:
        This example adds two integer symbols.

        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> i = model.integer(10, lower_bound=-50, upper_bound=50)
        >>> j = model.integer(10, lower_bound=0, upper_bound=10)
        >>> k = i + j
        >>> type(k)
        <class 'dwave.optimization.symbols.Add'>
    """
    pass


class And(_BinaryOpSymbol, node_type=_BinaryOpNodeType.And):
    """Boolean AND element-wise between two symbols.

    See Also:
        :func:`~dwave.optimization.mathematical.logical_and`: equivalent function.
    """
    pass


class Divide(_BinaryOpSymbol, node_type=_BinaryOpNodeType.Divide):
    """Division element-wise between two symbols.

    Examples:
        This example divides two integer symbols.

        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> i = model.integer(10, lower_bound=-50, upper_bound=-1)
        >>> j = model.integer(10, lower_bound=1, upper_bound=10)
        >>> k = i/j
        >>> type(k)
        <class 'dwave.optimization.symbols.Divide'>
    """
    pass


class Equal(_BinaryOpSymbol, node_type=_BinaryOpNodeType.Equal):
    """Equality comparison element-wise between two symbols.

    Examples:
        This example creates an equality operation between integer symbols.

        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> i = model.integer(25, upper_bound=100)
        >>> j = model.integer(25, lower_bound=-100)
        >>> k = i == j
        >>> type(k)
        <class 'dwave.optimization.symbols.Equal'>
    """
    pass


class LessEqual(_BinaryOpSymbol, node_type=_BinaryOpNodeType.LessEqual):
    """Smaller-or-equal comparison element-wise between two symbols.

    Examples:
        This example creates an inequality operation between integer symbols.

        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> i = model.integer(25, upper_bound=100)
        >>> j = model.integer(25, lower_bound=-100)
        >>> k = i <= j
        >>> type(k)
        <class 'dwave.optimization.symbols.LessEqual'>
    """
    pass


class Maximum(_BinaryOpSymbol, node_type=_BinaryOpNodeType.Maximum):
    """Maximum values in an element-wise comparison of two symbols.

    Examples:
        This example sets a symbol's values to the maximum values of two
        integer decision variables.

        >>> from dwave.optimization.model import Model
        >>> from dwave.optimization.mathematical import maximum
        ...
        >>> model = Model()
        >>> i = model.integer(100, lower_bound=-50, upper_bound=50)
        >>> j = model.integer(100, lower_bound=-20, upper_bound=150)
        >>> k = maximum(i, j)
        >>> type(k)
        <class 'dwave.optimization.symbols.Maximum'>
    """
    pass


class Minimum(_BinaryOpSymbol, node_type=_BinaryOpNodeType.Minimum):
    """Minimum values in an element-wise comparison of two symbols.

    Examples:
        This example sets a symbol's values to the minimum values of two
        integer decision variables.

        >>> from dwave.optimization.model import Model
        >>> from dwave.optimization.mathematical import minimum
        ...
        >>> model = Model()
        >>> i = model.integer(100, lower_bound=-50, upper_bound=50)
        >>> j = model.integer(100, lower_bound=-20, upper_bound=150)
        >>> k = minimum(i, j)
        >>> type(k)
        <class 'dwave.optimization.symbols.Minimum'>
    """
    pass


class Modulus(_BinaryOpSymbol, node_type=_BinaryOpNodeType.Modulus):
    """Modulus element-wise between two symbols.

    Examples:
        This example calculates the modulus of two integer symbols.

        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> i = model.integer(10, lower_bound=-50, upper_bound=50)
        >>> j = model.integer(10, lower_bound=-20, upper_bound=150)
        >>> k = i % j
        >>> type(k)
        <class 'dwave.optimization.symbols.Modulus'>
    """
    pass


class Multiply(_BinaryOpSymbol, node_type=_BinaryOpNodeType.Multiply):
    """Multiplication element-wise between two symbols.

    Examples:
        This example multiplies two integer symbols.

        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> i = model.integer(10, lower_bound=-50, upper_bound=50)
        >>> j = model.integer(10, lower_bound=0, upper_bound=10)
        >>> k = i*j
        >>> type(k)
        <class 'dwave.optimization.symbols.Multiply'>
    """
    pass


class Or(_BinaryOpSymbol, node_type=_BinaryOpNodeType.Or):
    """Boolean OR element-wise between two symbols.

    See Also:
        :func:`~dwave.optimization.mathematical.logical_or`: equivalent function.
    """
    pass


class SafeDivide(_BinaryOpSymbol, node_type=_BinaryOpNodeType.SafeDivide):
    """Safe division element-wise between two symbols.

    See also:
        :func:`~dwave.optimization.mathematical.safe_divide`: equivalent function.

    .. versionadded:: 0.6.2
    """
    pass


class Subtract(_BinaryOpSymbol, node_type=_BinaryOpNodeType.Subtract):
    """Subtraction element-wise of two symbols.

    Examples:
        This example subtracts two integer symbols.

        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> i = model.integer(10, lower_bound=-50, upper_bound=50)
        >>> j = model.integer(10, lower_bound=0, upper_bound=10)
        >>> k = i - j
        >>> type(k)
        <class 'dwave.optimization.symbols.Subtract'>
    """
    pass


class Xor(_BinaryOpSymbol, node_type=_BinaryOpNodeType.Xor):
    """Boolean XOR element-wise between two symbols.

    See Also:
        :func:`~dwave.optimization.mathematical.logical_xor`: equivalent function.

    .. versionadded:: 0.4.1
    """
    pass

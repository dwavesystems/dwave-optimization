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
        :func:`~dwave.optimization.mathematical.absolute`: Instantiation and
        usage of this symbol.
    """
    pass


class Cos(_UnaryOpSymbol, node_type=_UnaryOpNodeType.Cos):
    """Cosine element-wise on a symbol.

    See Also:
        :func:`~dwave.optimization.mathematical.cos`: Instantiation and usage of
        this symbol.

        :class:`.Sin`, :class:`.Tanh`

    .. versionadded:: 0.6.5
    """
    pass


class Exp(_UnaryOpSymbol, node_type=_UnaryOpNodeType.Exp):
    """Natural (base-e) exponential element-wise on a symbol.

    See Also:
        :func:`~dwave.optimization.mathematical.exp`: Instantiation and usage of
        this symbol.

        :class:`.Expit`, :class:`.Log`,
        :class:`~dwave.optimization.symbols.SoftMax`

    .. versionadded:: 0.6.2
    """
    pass


class Expit(_UnaryOpSymbol, node_type=_UnaryOpNodeType.Expit):
    """Logistic sigmoid (expit) element-wise on a symbol.

    See Also:
        :func:`~dwave.optimization.mathematical.expit`: Instantiation and usage
        of this symbol.

        :class:`.Exp`, :class:`.Log`,
        :class:`~dwave.optimization.symbols.SoftMax`

    .. versionadded:: 0.5.2
    """
    pass


class Log(_UnaryOpSymbol, node_type=_UnaryOpNodeType.Log):
    """Natural logarithm (log) element-wise on a symbol.

    See Also:
        :func:`~dwave.optimization.mathematical.log`: Instantiation and usage of
        this symbol.

        :class:`.Exp`, :class:`.Expit`,
        :class:`~dwave.optimization.symbols.SoftMax`

    .. versionadded:: 0.5.2
    """
    pass


class Logical(_UnaryOpSymbol, node_type=_UnaryOpNodeType.Logical):
    """Logical truth value element-wise on a symbol.

    See Also:
        :func:`~dwave.optimization.mathematical.logical`: Instantiation and
        usage of this symbol.

        :class:`~dwave.optimization.symbols.And`, :class:`.Not`,
        :class:`~dwave.optimization.symbols.Or`,
        :class:`~dwave.optimization.symbols.Xor`
    """
    pass


class Negative(_UnaryOpSymbol, node_type=_UnaryOpNodeType.Negative):
    """Numerical negative element-wise on a symbol.

    This symbol is instantiated by the minus operation, such as in the following
    example.

    Examples:
        >>> from dwave.optimization import Model
        >>> i = model.integer(3)
        >>> j = - i
        >>> type(j)
        <class 'dwave.optimization.symbols.unaryop.Negative'>
    """
    pass


class Not(_UnaryOpSymbol, node_type=_UnaryOpNodeType.Not):
    """Logical negation element-wise on a symbol.

    See Also:
        :func:`~dwave.optimization.mathematical.logical_not`: Instantiation and
        usage of this symbol.

        :class:`~dwave.optimization.symbols.And`, :class:`.Logical`,
        :class:`~dwave.optimization.symbols.Or`,
        :class:`~dwave.optimization.symbols.Xor`
    """
    pass


class Rint(_UnaryOpSymbol, node_type=_UnaryOpNodeType.Rint):
    """Rounds the values of a symbol to the nearest integer.

    See also:
        *   :func:`~dwave.optimization.mathematical.rint`: Instantiation and
            usage of this symbol.
    """
    pass


class Sin(_UnaryOpSymbol, node_type=_UnaryOpNodeType.Sin):
    """Sine element-wise on a symbol.

    See Also:
        :func:`~dwave.optimization.mathematical.sin`: Instantiation and usage of
        this symbol.

        :class:`.Cos`, :class:`.Tanh`

    .. versionadded:: 0.6.5
    """
    pass


class Square(_UnaryOpSymbol, node_type=_UnaryOpNodeType.Square):
    """Squares element-wise of a symbol.

    This symbol is instantiated by a power-of-two operation, such as in the
    following example.

    Examples:
        >>> from dwave.optimization import Model
        >>> model = Model()
        >>> i = model.integer(30)
        >>> j = i**2
        >>> print(type(j))
        <class 'dwave.optimization.symbols.unaryop.Square'>

    See Also:
        :class:`.SquareRoot`
    """
    pass


class SquareRoot(_UnaryOpSymbol, node_type=_UnaryOpNodeType.SquareRoot):
    """Square root of a symbol.

    See Also:
        :func:`~dwave.optimization.mathematical.sqrt`: Instantiation and usage
        of this symbol.

        :class:`Square`
    """
    pass


class Tanh(_UnaryOpSymbol, node_type=_UnaryOpNodeType.Tanh):
    """Tanh element-wise on a symbol.

    See Also:
        :func:`~dwave.optimization.mathematical.tanh`: Instantiation and usage
        of this symbol.

        :class:`.Cos`, :class:`.Sin`

    .. versionadded:: 0.6.11
    """
    pass

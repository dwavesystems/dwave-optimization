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

    See Also:
        *   :func:`~dwave.optimization.mathematical.add`: Instantiation and
            usage of this symbol.
        *   :class:`~dwave.optimization.symbols.NaryAdd`
        *   :class:`.Divide`, :class:`.Modulus`, :class:`.Multiply`,
            :class:`.SafeDivide`, :class:`.Subtract`
    """
    pass


class And(_BinaryOpSymbol, node_type=_BinaryOpNodeType.And):
    """Boolean AND element-wise between two symbols.

    See Also:
        *   :func:`~dwave.optimization.mathematical.logical_and`: Instantiation
            and usage of this symbol.
        *   :class:`~dwave.optimization.symbols.Logical`,
            :class:`~dwave.optimization.symbols.Not`,
            :class:`.Or`, :class:`.Xor`
    """
    pass


class Divide(_BinaryOpSymbol, node_type=_BinaryOpNodeType.Divide):
    """Division element-wise between two symbols.

    See Also:
        *   :func:`~dwave.optimization.mathematical.divide`: Instantiation and
            usage of this symbol.
        *   :class:`.Add`, :class:`.Modulus`, :class:`.Multiply`,
            :class:`.SafeDivide`, :class:`.Subtract`
    """
    pass


class Equal(_BinaryOpSymbol, node_type=_BinaryOpNodeType.Equal):
    """Equality comparison element-wise between two symbols.

    See Also:
        *   :func:`~dwave.optimization.mathematical.equal`: Instantiation and
            usage of this symbol.
        *   :class:`.LessEqual`
    """
    pass


class LessEqual(_BinaryOpSymbol, node_type=_BinaryOpNodeType.LessEqual):
    """Smaller-or-equal comparison element-wise between two symbols.

    See Also:
        *   :func:`~dwave.optimization.mathematical.less_equal`: Instantiation
            and usage of this symbol.
        *   :class:`.Equal`
    """
    pass


class Maximum(_BinaryOpSymbol, node_type=_BinaryOpNodeType.Maximum):
    """Maximum values in an element-wise comparison of two symbols.

    See Also:
        *   :func:`~dwave.optimization.mathematical.maximum`: Instantiation and
            usage of this symbol.
        *   :class:`~dwave.optimization.symbols.Max`,
            :class:`~dwave.optimization.symbols.NaryMaximum`
    """
    pass


class Minimum(_BinaryOpSymbol, node_type=_BinaryOpNodeType.Minimum):
    """Minimum values in an element-wise comparison of two symbols.

    See Also:
        *   :func:`~dwave.optimization.mathematical.minimum`: Instantiation and
            usage of this symbol.
        *   :class:`~dwave.optimization.symbols.Min`,
            :class:`~dwave.optimization.symbols.NaryMinimum`
    """
    pass


class Modulus(_BinaryOpSymbol, node_type=_BinaryOpNodeType.Modulus):
    """Modulus element-wise between two symbols.

    See Also:
        *   :func:`~dwave.optimization.mathematical.mod`: Instantiation and
            usage of this symbol.
        *   :class:`.Add`, :class:`.Divide`, :class:`.Multiply`,
            :class:`.SafeDivide`, :class:`.Subtract`
    """
    pass


class Multiply(_BinaryOpSymbol, node_type=_BinaryOpNodeType.Multiply):
    """Multiplication element-wise between two symbols.

    See Also:
        *   :func:`~dwave.optimization.mathematical.multiply`: Instantiation and
            usage of this symbol.
        *   :class:`~dwave.optimization.symbols.MatrixMultiply`,
            :class:`~dwave.optimization.symbols.Prod`
        *   :class:`.Add`, :class:`.Divide`, :class:`.Modulus`,
            :class:`.SafeDivide`, :class:`.Subtract`
        *   :class:`~dwave.optimization.symbols.NaryMultiply`
    """
    pass


class Or(_BinaryOpSymbol, node_type=_BinaryOpNodeType.Or):
    """Boolean OR element-wise between two symbols.

    See Also:
        *   :func:`~dwave.optimization.mathematical.logical_or`: Instantiation
            and usage of this symbol.
        *   :class:`.And`, :class:`~dwave.optimization.symbols.Logical`,
            :class:`~dwave.optimization.symbols.Not`, :class:`.Xor`
    """
    pass


class SafeDivide(_BinaryOpSymbol, node_type=_BinaryOpNodeType.SafeDivide):
    """Safe division element-wise between two symbols.

    See also:
        *   :func:`~dwave.optimization.mathematical.safe_divide`: Instantiation
            and usage of this symbol.
        *   :class:`.Add`, :class:`.Divide`, :class:`.Modulus`,
            :class:`.Multiply`, :class:`.Subtract`

    .. versionadded:: 0.6.2
    """
    pass


class Subtract(_BinaryOpSymbol, node_type=_BinaryOpNodeType.Subtract):
    """Subtraction element-wise of two symbols.

    See Also:
        *   :func:`~dwave.optimization.mathematical.subtract`: Instantiation and
            usage of this symbol.
        *   :class:`.Add`, :class:`.Divide`, :class:`.Modulus`,
            :class:`.Multiply`, :class:`.SafeDivide`,
    """
    pass


class Xor(_BinaryOpSymbol, node_type=_BinaryOpNodeType.Xor):
    """Boolean XOR element-wise between two symbols.

    See Also:
        *   :func:`~dwave.optimization.mathematical.logical_xor`: Instantiation
            and usage of this symbol.
        *   :class:`.And`, :class:`~dwave.optimization.symbols.Logical`,
            :class:`~dwave.optimization.symbols.Not`, :class:`.Or`

    .. versionadded:: 0.4.1
    """
    pass

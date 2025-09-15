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
    """Addition element-wise of two symbols."""
    pass


class And(_BinaryOpSymbol, node_type=_BinaryOpNodeType.And):
    """Boolean AND element-wise between two symbols.

    See Also:
        :func:`~dwave.optimization.mathematical.logical_and`: equivalent function.
    """
    pass


class Divide(_BinaryOpSymbol, node_type=_BinaryOpNodeType.Divide):
    """Division element-wise between two symbols."""
    pass


class Equal(_BinaryOpSymbol, node_type=_BinaryOpNodeType.Equal):
    """Equality comparison element-wise between two symbols."""
    pass


class LessEqual(_BinaryOpSymbol, node_type=_BinaryOpNodeType.LessEqual):
    """Smaller-or-equal comparison element-wise between two symbols."""
    pass


class Maximum(_BinaryOpSymbol, node_type=_BinaryOpNodeType.Maximum):
    """Maximum values in an element-wise comparison of two symbols."""
    pass


class Minimum(_BinaryOpSymbol, node_type=_BinaryOpNodeType.Minimum):
    """Minimum values in an element-wise comparison of two symbols."""
    pass


class Modulus(_BinaryOpSymbol, node_type=_BinaryOpNodeType.Modulus):
    """Modulus element-wise between two symbols."""
    pass


class Multiply(_BinaryOpSymbol, node_type=_BinaryOpNodeType.Multiply):
    """Multiplication element-wise between two symbols."""
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
    """Subtraction element-wise of two symbols."""
    pass


class Xor(_BinaryOpSymbol, node_type=_BinaryOpNodeType.Xor):
    """Boolean XOR element-wise between two symbols.

    See Also:
        :func:`~dwave.optimization.mathematical.logical_xor`: equivalent function.

    .. versionadded:: 0.4.1
    """
    pass

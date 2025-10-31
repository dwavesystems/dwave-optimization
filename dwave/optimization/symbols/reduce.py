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

from dwave.optimization.symbols._reduce import _ReduceSymbol, _ReduceNodeType


class All(_ReduceSymbol, node_type=_ReduceNodeType.All, default_initial=True):
    """Tests whether all elements evaluate to True.

    See Also:
        :meth:`~dwave.optimization.model.ArraySymbol.all()` equivalent method.
    """
    pass


class Any(_ReduceSymbol, node_type=_ReduceNodeType.Any, default_initial=False):
    """Tests whether any elements evaluate to True.

    See Also:
        :meth:`~dwave.optimization.model.ArraySymbol.any()` equivalent method.
    """
    pass


class Max(_ReduceSymbol, node_type=_ReduceNodeType.Max):
    """Maximum value in the elements of a symbol.

    See also:
        :meth:`~dwave.optimization.model.ArraySymbol.max()` equivalent method.
    """
    pass


class Min(_ReduceSymbol, node_type=_ReduceNodeType.Min):
    """Minimum value in the elements of a symbol.

    See also:
        :meth:`~dwave.optimization.model.ArraySymbol.min()` equivalent method.
    """
    pass


class Prod(_ReduceSymbol, node_type=_ReduceNodeType.Prod, default_initial=1):
    """Product of the elements of a symbol.

    See also:
        :meth:`~dwave.optimization.model.ArraySymbol.prod()` equivalent method.
    """
    pass


class Sum(_ReduceSymbol, node_type=_ReduceNodeType.Sum, default_initial=0):
    """Sum of the elements of a symbol.

    See Also:
        :meth:`~dwave.optimization.model.ArraySymbol.sum()` equivalent method.
    """
    pass


# Two deprecated aliases
PartialProd = Prod
PartialSum = Sum

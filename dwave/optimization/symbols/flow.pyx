# cython: auto_pickle=False

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

from cython.operator cimport typeid

from dwave.optimization._model cimport _Graph, _register, ArraySymbol, Symbol
from dwave.optimization.libcpp.nodes.flow cimport (
    ExtractNode,
    WhereNode,
)


cdef class Extract(ArraySymbol):
    """Return elements chosen from x or y depending on condition.

    See Also:
        :func:`~dwave.optimization.mathematical.where`: equivalent function.
    """
    def __init__(self, ArraySymbol condition, ArraySymbol arr):
        cdef _Graph model = condition.model

        if condition.model is not arr.model:
            raise ValueError("condition and arr do not share the same underlying model")

        cdef ExtractNode* ptr = model._graph.emplace_node[ExtractNode](
            condition.array_ptr, arr.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(Extract, typeid(ExtractNode))


cdef class Where(ArraySymbol):
    """Return elements chosen from x or y depending on condition.

    See Also:
        :func:`~dwave.optimization.mathematical.where`: equivalent function.
    """
    def __init__(self, ArraySymbol condition, ArraySymbol x, ArraySymbol y):
        cdef _Graph model = condition.model

        if condition.model is not x.model:
            raise ValueError("condition and x do not share the same underlying model")
        if condition.model is not y.model:
            raise ValueError("condition and y do not share the same underlying model")

        cdef WhereNode* ptr = model._graph.emplace_node[WhereNode](
            condition.array_ptr, x.array_ptr, y.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(Where, typeid(WhereNode))

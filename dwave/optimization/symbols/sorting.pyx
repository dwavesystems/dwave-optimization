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

from dwave.optimization._model cimport _Graph, _register, ArraySymbol
from dwave.optimization.libcpp.nodes.sorting cimport ArgSortNode


cdef class ArgSort(ArraySymbol):
    """Return an ordering of the indices that would sort (flattened) values
    of the given symbol. Note that while it will return an array with
    identical shape to the given symbol, the returned indices will always be
    indices on flattened array, similar to ``numpy.argsort(a, axis=None)``.

    Always performs a index-wise stable sort such that the relative order of
    values is maintained in the returned order.

    See Also:
        :func:`~dwave.optimization.mathematical.argsort`: equivalent function.

    .. versionadded:: 0.6.4
    """
    def __init__(self, ArraySymbol arr):
        cdef _Graph model = arr.model

        cdef ArgSortNode* ptr = model._graph.emplace_node[ArgSortNode](arr.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(ArgSort, typeid(ArgSortNode))

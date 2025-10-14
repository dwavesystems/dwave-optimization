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
from dwave.optimization.libcpp.nodes.set_routines cimport IsInNode


cdef class IsIn(ArraySymbol):
    """Determine element-wise containment between two symbols. Given two
    symbols: element and test_elements, returns an output array of the same
    shape as element such that output[index] = True if element[index] is in
    test_elements and False otherwise.

    See Also:
        :meth:`~dwave.optimization.mathematical.isin`: equivalent method.

    .. versionadded:: 0.6.8
    """
    def __init__(self, ArraySymbol element, ArraySymbol test_elements):
        cdef _Graph model = element.model

        if element.model is not test_elements.model:
            raise ValueError("element and test_elements do not share the same underlying model")

        cdef IsInNode* ptr = model._graph.emplace_node[IsInNode](element.array_ptr, test_elements.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(IsIn, typeid(IsInNode))

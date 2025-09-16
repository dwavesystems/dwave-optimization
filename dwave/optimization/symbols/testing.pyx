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
from dwave.optimization.libcpp.nodes.testing cimport ArrayValidationNode


cdef class _ArrayValidation(Symbol):
    def __init__(self, ArraySymbol array_node):
        cdef _Graph model = array_node.model

        cdef ArrayValidationNode* ptr = model._graph.emplace_node[ArrayValidationNode](
            array_node.array_ptr,
        )
        self.initialize_node(model, ptr)

_register(_ArrayValidation, typeid(ArrayValidationNode))

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
from libcpp.vector cimport vector

from dwave.optimization._model cimport _Graph, _register, ArraySymbol, Symbol
from dwave.optimization._model import _as_array_symbol
from dwave.optimization.libcpp cimport dynamic_cast_ptr
from dwave.optimization.libcpp.graph cimport ArrayNode
from dwave.optimization.libcpp.nodes.naryop cimport (
    NaryAddNode,
    NaryMaximumNode,
    NaryMinimumNode,
    NaryMultiplyNode,
)


cdef class NaryAdd(ArraySymbol):
    """Addition element-wise of `N` symbols."""
    def __init__(self, *inputs):
        if len(inputs) == 0:
            raise TypeError("must have at least one predecessor node")

        cdef _Graph model = inputs[0].model
        cdef vector[ArrayNode*] cppinputs

        cdef ArraySymbol array
        for node in inputs:
            if node.model != model:
                raise ValueError("all predecessors must be from the same model")
            array = <ArraySymbol?>node
            cppinputs.push_back(array.array_ptr)

        self.ptr = model._graph.emplace_node[NaryAddNode](cppinputs)
        self.initialize_arraynode(model, self.ptr)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef NaryAddNode* ptr = dynamic_cast_ptr[NaryAddNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")
        cdef NaryAdd x = NaryAdd.__new__(NaryAdd)
        x.ptr = ptr
        x.initialize_arraynode(symbol.model, ptr)
        return x

    def __iadd__(self, rhs):
        if not self.node_ptr.successors().empty():
            return super().__iadd__(rhs)

        try:
            rhs = _as_array_symbol(self.model, rhs)
            self.ptr.add_node((<ArraySymbol>rhs).array_ptr)
            return self
        except TypeError:
            # this should be handled by the call to ArraySymbol.__iadd__ below
            pass

        return super().__iadd__(rhs)

    cdef NaryAddNode* ptr

_register(NaryAdd, typeid(NaryAddNode))


cdef class NaryMaximum(ArraySymbol):
    """Maximum values in an element-wise comparison of `N` symbols."""
    def __init__(self, *inputs):
        if len(inputs) == 0:
            raise TypeError("must have at least one predecessor node")

        cdef _Graph model = inputs[0].model
        cdef vector[ArrayNode*] cppinputs

        cdef ArraySymbol array
        for node in inputs:
            if node.model != model:
                raise ValueError("all predecessors must be from the same model")
            array = <ArraySymbol?>node
            cppinputs.push_back(array.array_ptr)

        cdef NaryMaximumNode* ptr = model._graph.emplace_node[NaryMaximumNode](cppinputs)
        self.initialize_arraynode(model, ptr)

_register(NaryMaximum, typeid(NaryMaximumNode))


cdef class NaryMinimum(ArraySymbol):
    """Minimum values in an element-wise comparison of `N` symbols."""
    def __init__(self, *inputs):
        if len(inputs) == 0:
            raise TypeError("must have at least one predecessor node")

        cdef _Graph model = inputs[0].model
        cdef vector[ArrayNode*] cppinputs

        cdef ArraySymbol array
        for node in inputs:
            if node.model != model:
                raise ValueError("all predecessors must be from the same model")
            array = <ArraySymbol?>node
            cppinputs.push_back(array.array_ptr)

        cdef NaryMinimumNode* ptr = model._graph.emplace_node[NaryMinimumNode](cppinputs)
        self.initialize_arraynode(model, ptr)

_register(NaryMinimum, typeid(NaryMinimumNode))


cdef class NaryMultiply(ArraySymbol):
    """Multiplication element-wise between `N` symbols."""
    def __init__(self, *inputs):
        if len(inputs) == 0:
            raise TypeError("must have at least one predecessor node")

        cdef _Graph model = inputs[0].model
        cdef vector[ArrayNode*] cppinputs

        cdef ArraySymbol array
        for node in inputs:
            if node.model != model:
                raise ValueError("all predecessors must be from the same model")
            array = <ArraySymbol?>node
            cppinputs.push_back(array.array_ptr)

        self.ptr = model._graph.emplace_node[NaryMultiplyNode](cppinputs)
        self.initialize_arraynode(model, self.ptr)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef NaryMultiplyNode* ptr = dynamic_cast_ptr[NaryMultiplyNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")
        cdef NaryMultiply x = NaryMultiply.__new__(NaryMultiply)
        x.ptr = ptr
        x.initialize_arraynode(symbol.model, ptr)
        return x

    def __imul__(self, rhs):
        if not self.node_ptr.successors().empty():
            return super().__imul__(rhs)

        try:
            rhs = _as_array_symbol(self.model, rhs)
            self.ptr.add_node((<ArraySymbol>rhs).array_ptr)
            return self
        except TypeError:
            # this should be handled by the call to ArraySymbol.__imul__ below
            pass

        return super().__imul__(rhs)

    cdef NaryMultiplyNode* ptr

_register(NaryMultiply, typeid(NaryMultiplyNode))

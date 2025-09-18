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
from dwave.optimization.libcpp.graph cimport ArrayNode
from dwave.optimization.libcpp.nodes.binaryop cimport (
    AddNode,
    AndNode,
    DivideNode,
    EqualNode,
    LessEqualNode,
    MaximumNode,
    MinimumNode,
    ModulusNode,
    MultiplyNode,
    OrNode,
    SafeDivideNode,
    SubtractNode,
    XorNode,
)

# We need to be able to expose the available node types to the Python level,
# so unfortunately we need this enum and giant if-branches
cpdef enum _BinaryOpNodeType:
    Add
    And
    Divide
    Equal
    LessEqual
    Maximum
    Minimum
    Modulus
    Multiply
    Or
    SafeDivide
    Subtract
    Xor


class _BinaryOpSymbol(ArraySymbol):
    def __init_subclass__(cls, /, _BinaryOpNodeType node_type):
        if node_type is _BinaryOpNodeType.Add:
            _register(cls, typeid(AddNode))
        elif node_type is _BinaryOpNodeType.And:
            _register(cls, typeid(AndNode))
        elif node_type is _BinaryOpNodeType.Divide:
            _register(cls, typeid(DivideNode))
        elif node_type is _BinaryOpNodeType.Equal:
            _register(cls, typeid(EqualNode))
        elif node_type is _BinaryOpNodeType.LessEqual:
            _register(cls, typeid(LessEqualNode))
        elif node_type is _BinaryOpNodeType.Maximum:
            _register(cls, typeid(MaximumNode))
        elif node_type is _BinaryOpNodeType.Minimum:
            _register(cls, typeid(MinimumNode))
        elif node_type is _BinaryOpNodeType.Modulus:
            _register(cls, typeid(ModulusNode))
        elif node_type is _BinaryOpNodeType.Multiply:
            _register(cls, typeid(MultiplyNode))
        elif node_type is _BinaryOpNodeType.Or:
            _register(cls, typeid(OrNode))
        elif node_type is _BinaryOpNodeType.SafeDivide:
            _register(cls, typeid(SafeDivideNode))
        elif node_type is _BinaryOpNodeType.Subtract:
            _register(cls, typeid(SubtractNode))
        elif node_type is _BinaryOpNodeType.Xor:
            _register(cls, typeid(XorNode))
        else:
            raise RuntimeError(f"unexpected _BinaryOpNodeType: {<object>node_type!r}")

        cls._node_type = node_type

    def __init__(self, ArraySymbol lhs, ArraySymbol rhs):
        if lhs.model is not rhs.model:
            raise ValueError("lhs and rhs do not share the same underlying model")

        cdef _Graph model = lhs.model
        cdef _BinaryOpNodeType node_type = self._node_type

        cdef ArrayNode* ptr

        if node_type is _BinaryOpNodeType.Add:
            ptr = model._graph.emplace_node[AddNode](lhs.array_ptr, rhs.array_ptr)
        elif node_type is _BinaryOpNodeType.And:
            ptr = model._graph.emplace_node[AndNode](lhs.array_ptr, rhs.array_ptr)
        elif node_type is _BinaryOpNodeType.Divide:
            ptr = model._graph.emplace_node[DivideNode](lhs.array_ptr, rhs.array_ptr)
        elif node_type is _BinaryOpNodeType.Equal:
            ptr = model._graph.emplace_node[EqualNode](lhs.array_ptr, rhs.array_ptr)
        elif node_type is _BinaryOpNodeType.LessEqual:
            ptr = model._graph.emplace_node[LessEqualNode](lhs.array_ptr, rhs.array_ptr)
        elif node_type is _BinaryOpNodeType.Maximum:
            ptr = model._graph.emplace_node[MaximumNode](lhs.array_ptr, rhs.array_ptr)
        elif node_type is _BinaryOpNodeType.Minimum:
            ptr = model._graph.emplace_node[MinimumNode](lhs.array_ptr, rhs.array_ptr)
        elif node_type is _BinaryOpNodeType.Modulus:
            ptr = model._graph.emplace_node[ModulusNode](lhs.array_ptr, rhs.array_ptr)
        elif node_type is _BinaryOpNodeType.Multiply:
            ptr = model._graph.emplace_node[MultiplyNode](lhs.array_ptr, rhs.array_ptr)
        elif node_type is _BinaryOpNodeType.Or:
            ptr = model._graph.emplace_node[OrNode](lhs.array_ptr, rhs.array_ptr)
        elif node_type is _BinaryOpNodeType.SafeDivide:
            ptr = model._graph.emplace_node[SafeDivideNode](lhs.array_ptr, rhs.array_ptr)
        elif node_type is _BinaryOpNodeType.Subtract:
            ptr = model._graph.emplace_node[SubtractNode](lhs.array_ptr, rhs.array_ptr)
        elif node_type is _BinaryOpNodeType.Xor:
            ptr = model._graph.emplace_node[XorNode](lhs.array_ptr, rhs.array_ptr)
        else:
            raise RuntimeError(f"unexpected _BinaryOpNodeType: {<object>node_type!r}")

        (<ArraySymbol>self).initialize_arraynode(model, ptr)

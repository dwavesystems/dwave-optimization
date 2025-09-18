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
from dwave.optimization.libcpp.nodes.unaryop cimport (
    AbsoluteNode,
    CosNode,
    ExpNode,
    ExpitNode,
    LogNode,
    LogicalNode,
    NegativeNode,
    NotNode,
    RintNode,
    SinNode,
    SquareNode,
    SquareRootNode,
)

# We need to be able to expose the available node types to the Python level,
# so unfortunately we need this enum and giant if-branches
cpdef enum _UnaryOpNodeType:
    Absolute
    Cos
    Exp
    Expit
    Log
    Logical
    Negative
    Not
    Rint
    Sin
    Square
    SquareRoot


class _UnaryOpSymbol(ArraySymbol):
    def __init_subclass__(cls, /, _UnaryOpNodeType node_type):
        if node_type is _UnaryOpNodeType.Absolute:
            _register(cls, typeid(AbsoluteNode))
        elif node_type is _UnaryOpNodeType.Cos:
            _register(cls, typeid(CosNode))
        elif node_type is _UnaryOpNodeType.Exp:
            _register(cls, typeid(ExpNode))
        elif node_type is _UnaryOpNodeType.Expit:
            _register(cls, typeid(ExpitNode))
        elif node_type is _UnaryOpNodeType.Log:
            _register(cls, typeid(LogNode))
        elif node_type is _UnaryOpNodeType.Logical:
            _register(cls, typeid(LogicalNode))
        elif node_type is _UnaryOpNodeType.Negative:
            _register(cls, typeid(NegativeNode))
        elif node_type is _UnaryOpNodeType.Not:
            _register(cls, typeid(NotNode))
        elif node_type is _UnaryOpNodeType.Rint:
            _register(cls, typeid(RintNode))
        elif node_type is _UnaryOpNodeType.Sin:
            _register(cls, typeid(SinNode))
        elif node_type is _UnaryOpNodeType.Square:
            _register(cls, typeid(SquareNode))
        elif node_type is _UnaryOpNodeType.SquareRoot:
            _register(cls, typeid(SquareRootNode))
        else:
            raise RuntimeError(f"unexpected _UnaryOpNodeType: {<object>node_type!r}")

        cls._node_type = node_type

    def __init__(self, ArraySymbol x):
        cdef _Graph model = x.model
        cdef _UnaryOpNodeType node_type = self._node_type

        cdef ArrayNode* ptr

        if node_type is _UnaryOpNodeType.Absolute:
            ptr = model._graph.emplace_node[AbsoluteNode](x.array_ptr)
        elif node_type is _UnaryOpNodeType.Cos:
            ptr = model._graph.emplace_node[CosNode](x.array_ptr)
        elif node_type is _UnaryOpNodeType.Exp:
            ptr = model._graph.emplace_node[ExpNode](x.array_ptr)
        elif node_type is _UnaryOpNodeType.Expit:
            ptr = model._graph.emplace_node[ExpitNode](x.array_ptr)
        elif node_type is _UnaryOpNodeType.Log:
            ptr = model._graph.emplace_node[LogNode](x.array_ptr)
        elif node_type is _UnaryOpNodeType.Logical:
            ptr = model._graph.emplace_node[LogicalNode](x.array_ptr)
        elif node_type is _UnaryOpNodeType.Negative:
            ptr = model._graph.emplace_node[NegativeNode](x.array_ptr)
        elif node_type is _UnaryOpNodeType.Not:
            ptr = model._graph.emplace_node[NotNode](x.array_ptr)
        elif node_type is _UnaryOpNodeType.Rint:
            ptr = model._graph.emplace_node[RintNode](x.array_ptr)
        elif node_type is _UnaryOpNodeType.Sin:
            ptr = model._graph.emplace_node[SinNode](x.array_ptr)
        elif node_type is _UnaryOpNodeType.Square:
            ptr = model._graph.emplace_node[SquareNode](x.array_ptr)
        elif node_type is _UnaryOpNodeType.SquareRoot:
            ptr = model._graph.emplace_node[SquareRootNode](x.array_ptr)
        else:
            raise RuntimeError(f"unexpected _UnaryOpNodeType: {<object>node_type!r}")

        (<ArraySymbol>self).initialize_arraynode(model, ptr)

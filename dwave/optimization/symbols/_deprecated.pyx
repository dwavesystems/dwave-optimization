# cython: auto_pickle=False

# Copyright 2024 D-Wave
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

# Organizational note: symbols are ordered alphabetically

import collections.abc
import json
import numbers

cimport cpython.buffer
cimport cpython.object
import cython
cimport cython
import numpy as np

from cpython.ref cimport PyObject
from cython.operator cimport dereference as deref, typeid
from libc.math cimport modf
from libc.stdint cimport uintptr_t
from libcpp cimport bool
from libcpp.cast cimport dynamic_cast, reinterpret_cast
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.optional cimport nullopt, optional
from libcpp.span cimport span
from libcpp.typeindex cimport type_index
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport move
from libcpp.vector cimport vector

from dwave.optimization.expression import Expression
from dwave.optimization.libcpp cimport dynamic_cast_ptr, get, holds_alternative
from dwave.optimization.libcpp.array cimport (
    Array as cppArray,
    SizeInfo as cppSizeInfo,
    Slice as cppSlice,
    )
from dwave.optimization.libcpp.graph cimport (
    ArrayNode as cppArrayNode,
    ArrayNodePtr as cppArrayNodePtr,
    Node as cppNode,
    NodePtr as cppNodePtr,
    Graph as cppGraph,
    )
from dwave.optimization.libcpp.nodes cimport (
    AbsoluteNode as cppAbsoluteNode,
    AccumulateZipNode as cppAccumulateZipNode,
    AddNode as cppAddNode,
    AllNode as cppAllNode,
    AndNode as cppAndNode,
    AnyNode as cppAnyNode,
    AdvancedIndexingNode as cppAdvancedIndexingNode,
    ARangeNode as cppARangeNode,
    ArgSortNode as cppArgSortNode,
    ArrayValidationNode as cppArrayValidationNode,
    BasicIndexingNode as cppBasicIndexingNode,
    BinaryNode as cppBinaryNode,
    BroadcastToNode as cppBroadcastToNode,
    BSplineNode as cppBSplineNode,
    ConcatenateNode as cppConcatenateNode,
    ConstantNode as cppConstantNode,
    CopyNode as cppCopyNode,
    CosNode as cppCosNode,
    DisjointBitSetNode as cppDisjointBitSetNode,
    DisjointBitSetsNode as cppDisjointBitSetsNode,
    DisjointListNode as cppDisjointListNode,
    DisjointListsNode as cppDisjointListsNode,
    DivideNode as cppDivideNode,
    EqualNode as cppEqualNode,
    ExpitNode as cppExpitNode,
    ExpNode as cppExpNode,
    ExtractNode as cppExtractNode,
    InputNode as cppInputNode,
    IntegerNode as cppIntegerNode,
    LessEqualNode as cppLessEqualNode,
    LinearProgramFeasibleNode as cppLinearProgramFeasibleNode,
    LinearProgramNode as cppLinearProgramNode,
    LinearProgramNodeBase as cppLinearProgramNodeBase,
    LinearProgramObjectiveValueNode as cppLinearProgramObjectiveValueNode,
    LinearProgramSolutionNode as cppLinearProgramSolutionNode,
    ListNode as cppListNode,
    LogNode as cppLogNode,
    LogicalNode as cppLogicalNode,
    MaxNode as cppMaxNode,
    MaximumNode as cppMaximumNode,
    MeanNode as cppMeanNode,
    MinNode as cppMinNode,
    MinimumNode as cppMinimumNode,
    ModulusNode as cppModulusNode,
    MultiplyNode as cppMultiplyNode,
    NaryAddNode as cppNaryAddNode,
    NaryMaximumNode as cppNaryMaximumNode,
    NaryMinimumNode as cppNaryMinimumNode,
    NaryMultiplyNode as cppNaryMultiplyNode,
    NegativeNode as cppNegativeNode,
    NotNode as cppNotNode,
    OrNode as cppOrNode,
    PartialProdNode as cppPartialProdNode,
    PartialSumNode as cppPartialSumNode,
    PermutationNode as cppPermutationNode,
    ProdNode as cppProdNode,
    PutNode as cppPutNode,
    QuadraticModel as cppQuadraticModel,
    QuadraticModelNode as cppQuadraticModelNode,
    ReshapeNode as cppReshapeNode,
    ResizeNode as cppResizeNode,
    SafeDivideNode as cppSafeDivideNode,
    SetNode as cppSetNode,
    SinNode as cppSinNode,
    SizeNode as cppSizeNode,
    SoftMaxNode as cppSoftMaxNode,
    SubtractNode as cppSubtractNode,
    RintNode as cppRintNode,
    SquareNode as cppSquareNode,
    SquareRootNode as cppSquareRootNode,
    SumNode as cppSumNode,
    WhereNode as cppWhereNode,
    XorNode as cppXorNode,
)
from dwave.optimization._model import _as_array_symbol
from dwave.optimization._model cimport (
    ArraySymbol,
    _Graph,
    _register,
    Symbol,
    symbol_from_ptr,
)
from dwave.optimization.states cimport States
from dwave.optimization.symbols.collections import ListVariable
from dwave.optimization._utilities cimport as_cppshape, as_span


__all__ = [
    "BSpline",
    ]


cdef class BSpline(ArraySymbol):
    """Bspline node that takes in an array pointer, an integer degree and two vectors for knots and coefficients.

    See Also:
        :func:`~dwave.optimization.mathematical.bspline()` equivalent function.
    """
    def __init__(self, ArraySymbol x, k, t, c):

        if not isinstance(k, int):
            raise TypeError("expected an int for k")

        cdef _Graph model = x.model

        val_k = <Py_ssize_t> k

        cdef vector[double] vec_t
        for value_t in t:
            vec_t.push_back(<double> value_t)

        cdef vector[double] vec_c
        for value_c in c:
            vec_c.push_back(<double> value_c)

        self.ptr = model._graph.emplace_node[cppBSplineNode](x.array_ptr, val_k, vec_t, vec_c)
        self.initialize_arraynode(model, self.ptr)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef cppBSplineNode * ptr = dynamic_cast_ptr[cppBSplineNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")
        cdef BSpline m = BSpline.__new__(BSpline)
        m.ptr = ptr
        m.initialize_arraynode(symbol.model, ptr)
        return m

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        """Construct a BSpline from a zipfile."""
        if len(predecessors) != 1:
            raise ValueError("BSpline must have exactly one predecessor")

        # get the constant values
        with zf.open(directory + "k.json", mode="r") as f:
            kvalue = json.load(f)

        with zf.open(directory + "t.npy", mode="r") as f:
            tvalues = np.load(f, allow_pickle=False)

        with zf.open(directory + "c.npy", mode="r") as f:
            cvalues = np.load(f, allow_pickle=False)

        # pass to the constructor
        return cls(predecessors[0], kvalue, tvalues, cvalues)

    def _into_zipfile(self, zf, directory):
        """Save the BSpline constants into a zipfile"""
        cdef vector[double] tvalues = self.ptr.t()
        cdef vector[double] cvalues = self.ptr.c()

        t_array = np.array([tvalues[i] for i in range(tvalues.size())], dtype=np.double)
        c_array = np.array([cvalues[i] for i in range(cvalues.size())], dtype=np.double)

        encoder = json.JSONEncoder(separators=(',', ':'))
        zf.writestr(directory + "k.json", encoder.encode(self.ptr.k()))

        with zf.open(directory + "t.npy", mode="w", force_zip64=True) as f:
            np.save(f, t_array, allow_pickle=False)

        with zf.open(directory + "c.npy", mode="w", force_zip64=True) as f:
            np.save(f, c_array, allow_pickle=False)

    cdef cppBSplineNode * ptr

_register(BSpline, typeid(cppBSplineNode))

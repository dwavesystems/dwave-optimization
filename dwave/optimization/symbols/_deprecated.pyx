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
    "Input",
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


cdef class Input(ArraySymbol):
    """An input symbol. Functions as a "placeholder" in a model."""

    def __init__(
        self,
        model,
        shape=None,
        lower_bound=None,
        upper_bound=None,
        integral=None,
    ):
        cdef vector[Py_ssize_t] vshape = as_cppshape(tuple() if shape is None else shape)

        cdef _Graph cygraph = model

        # Get an observing pointer to the C++ InputNode
        self.ptr = cygraph._graph.emplace_node[cppInputNode](
            vshape,
            optional[double](nullopt) if lower_bound is None else optional[double](<double>lower_bound),
            optional[double](nullopt) if upper_bound is None else optional[double](<double>upper_bound),
            optional[bool](nullopt) if integral is None else optional[bool](<bool>integral),
        )

        self.initialize_arraynode(model, self.ptr)

    def integral(self):
        """Whether the input symbol will always output integers."""
        return self.ptr.integral()

    def lower_bound(self):
        """Lowest value allowed to the input."""
        return self.ptr.min()

    def set_state(self, Py_ssize_t index, state):
        """Set the state of the input symbol.

        The given state must be the same shape as the input symbol's shape.
        """

        # can't use ascontiguousarray yet because it will turn scalars into 1d arrays
        np_arr = np.asarray(state, dtype=np.double)
        if np_arr.shape != self.shape():
            raise ValueError(
                f"provided state's shape ({np_arr.shape}) does not match the Input's shape ({self.shape()})"
            )
        cdef double[::1] arr = np.ascontiguousarray(np_arr).flatten()

        # Reset our state, and check whether that's possible
        self.reset_state(index)

        self.ptr.initialize_state(
            (<States>self.model.states)._states[index],
            <span[const double]>as_span(arr)
        )

    def upper_bound(self):
        """Largest value allowed to the input."""
        return self.ptr.max()

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef cppInputNode* ptr = dynamic_cast_ptr[cppInputNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")

        cdef Input inp = Input.__new__(Input)
        inp.ptr = ptr
        inp.initialize_arraynode(symbol.model, ptr)
        return inp

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        if predecessors:
            raise ValueError(f"{cls.__name__} cannot have predecessors")

        with zf.open(directory + "properties.json", "r") as f:
            properties = json.load(f)

        return Input(model,
            shape=properties["shape"],
            lower_bound=properties["min"],
            upper_bound=properties["max"],
            integral=properties["integral"],
        )

    def _into_zipfile(self, zf, directory):
        properties = dict(
            shape=self.shape(),
            min=self.ptr.min(),
            max=self.ptr.max(),
            integral=self.ptr.integral(),
        )

        encoder = json.JSONEncoder(separators=(',', ':'))

        zf.writestr(directory + "properties.json", encoder.encode(properties))

    cdef cppInputNode* ptr

_register(Input, typeid(cppInputNode))

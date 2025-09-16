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
    "ArgSort",
    "BSpline",
    "Input",
    "Mean",
    "QuadraticModel",
    "SoftMax",
    ]


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

        cdef cppArgSortNode* ptr = model._graph.emplace_node[cppArgSortNode](arr.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(ArgSort, typeid(cppArgSortNode))


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


cdef class Mean(ArraySymbol):
    """Mean value of the elements of a symbol. If symbol is empty, 
        mean defaults to 0.0.

    See Also:
        :meth:`~dwave.optimization.mathematical.mean`: equivalent method.

    .. versionadded:: 0.6.4
    """
    def __init__(self, ArraySymbol arr):
        cdef _Graph model = arr.model

        self.ptr = model._graph.emplace_node[cppMeanNode](arr.array_ptr)
        self.initialize_arraynode(model, self.ptr)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef cppMeanNode* ptr = dynamic_cast_ptr[cppMeanNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")
        cdef Mean x = Mean.__new__(Mean)
        x.ptr = ptr
        x.initialize_arraynode(symbol.model, ptr)
        return x

    cdef cppMeanNode* ptr

_register(Mean, typeid(cppMeanNode))


cdef class QuadraticModel(ArraySymbol):
    """Quadratic model."""
    def __init__(self, ArraySymbol x, quadratic, linear=None):
        # Some checking on x
        if x.array_ptr.dynamic():
            raise ValueError("x cannot be dynamic")
        if x.ndim() != 1:
            raise ValueError("x must be a 1d array")
        if x.size() < 1:
            raise ValueError("x must have at least one element")

        if isinstance(quadratic, dict):
            self._init_from_qubo(x, quadratic, linear)
        elif isinstance(quadratic, tuple):
            self._init_from_coords(x, quadratic, linear)
        else:
            # todo: support other formats, following scipy.sparse.coo_array
            raise TypeError("quadratic must be a dict or a tuple of (data, coords)")

        if self.ptr == NULL or self.node_ptr == NULL or self.array_ptr == NULL:
            raise RuntimeError("QuadraticModel is malformed")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _init_from_coords(self, ArraySymbol x, quadratic, linear):
        # x type etc is checked by __init__
        cdef Py_ssize_t num_variables = x.size()
        cdef bint binary = x.array_ptr.logical()

        # Parse linear
        if linear is None:
            linear = []
        # Let NumPy handle type checking
        cdef double[::1] ldata = np.ascontiguousarray(linear, dtype=np.double)

        if ldata.shape[0] > num_variables:
            raise ValueError("linear index out of range of x")

        # Parse quadratic
        if not isinstance(quadratic, tuple):
            # This should have already been checked before dispatch, but just in case
            raise TypeError("quadratic must be a tuple")
        if len(quadratic) != 2:
            raise ValueError("expected a length 2 tuple")

        cdef double[::1] data = np.ascontiguousarray(quadratic[0], dtype=np.double)
        cdef int[:, ::1] coords = np.ascontiguousarray(np.atleast_2d(quadratic[1]), dtype=np.intc)

        if data.shape[0] != coords.shape[1]:
            # SciPy's error message
            raise ValueError("row, column, and data array must all be the same length")
        if data.shape[0] < 1:
            raise ValueError("quadratic must contain at least one interaction")
        if np.asarray(coords).min() < 0:
            # SciPy's error message
            raise ValueError("negative index found")
        if np.asarray(coords).max() >= num_variables:
            raise ValueError("index greater than or equal to x.size() found")

        # Construct a QuadraticModel temporarily, and then hand ownership over to the node
        cdef cppQuadraticModel* qm
        try:
            qm = new cppQuadraticModel(num_variables)

            # linear
            for i in range(ldata.shape[0]):
                qm.add_linear(i, ldata[i])

            # quadratic
            for i in range(data.shape[0]):
                if binary and coords[0, i] == coords[1, i]:
                    # linear term
                    qm.add_linear(coords[0, i], data[i])
                else:
                    # quadratic term
                    qm.add_quadratic(coords[0, i], coords[1, i], data[i])

            self.ptr = x.model._graph.emplace_node[cppQuadraticModelNode](x.array_ptr, move(deref(qm)))
        finally:
            # even if an exception is thrown, we don't leak memory
            del qm

        self.initialize_arraynode(x.model, self.ptr)

    @cython.wraparound(False)
    def _init_from_qubo(self, ArraySymbol x, quadratic, linear):
        """Construct from a QUBO in D-Wave style. I.e. ``{(u, v): bias, ...}``"""
        # x type etc is checked by __init__
        cdef Py_ssize_t num_variables = x.size()
        # cdef bint binary = x.array_ptr.logical()

        # We parse linear first, because some linear values might also appear in
        # quadratic
        cdef double[::1] ldata = np.zeros(num_variables)
        cdef Py_ssize_t v
        cdef double bias

        if linear is None:
            pass
        elif isinstance(linear, dict):
            # Cython will raise erros for bad types and for out of bounds
            # errors
            for v, bias in linear.items():
                ldata[v] += bias
        else:
            raise TypeError("if quadratic is a dict, linear must be too")

        # Now parse the quadratic
        # names are chosen to be consistent with scipy.sparse.coo_array
        cdef double[::1] data = np.empty(len(quadratic), dtype=np.double)
        cdef int[:,::1] coords = np.empty((2, len(quadratic)), dtype=np.intc)

        cdef Py_ssize_t i = 0
        cdef Py_ssize_t u
        with cython.boundscheck(False):
            # Cython will handle the type checks
            # We'll defer checking that u, v are in range to _init_from_coords
            for (u, v), bias in quadratic.items():
                coords[0, i] = u
                coords[1, i] = v
                data[i] = bias
                i += 1

        self._init_from_coords(x, (data, coords), ldata)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef cppQuadraticModelNode* ptr = dynamic_cast_ptr[cppQuadraticModelNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")

        cdef QuadraticModel qm = QuadraticModel.__new__(QuadraticModel)
        qm.ptr = ptr
        qm.initialize_arraynode(symbol.model, ptr)
        return qm

    def get_linear(self, Py_ssize_t v):
        """Get the linear bias of v"""
        if not 0 <= v < self.num_variables():
            raise ValueError(f"v out of range for a model with {self.num_variables()} variables")
        return self.ptr.get_quadratic_model().get_linear(v)

    def get_quadratic(self, Py_ssize_t u, Py_ssize_t v):
        """Get the quadratic bias of u and v. Returns 0 if not present."""
        if not 0 <= u < self.num_variables():
            raise ValueError(f"u out of range for a model with {self.num_variables()} variables")
        if not 0 <= v < self.num_variables():
            raise ValueError(f"v out of range for a model with {self.num_variables()} variables")
        return self.ptr.get_quadratic_model().get_quadratic(u, v)

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        """Construct a QuadraticModel from a zipfile."""
        if len(predecessors) != 1:
            raise ValueError("Reshape must have exactly one predecessor")

        # get the arrays
        with zf.open(directory + "linear.npy", mode="r") as f:
            ldata = np.load(f, allow_pickle=False)

        with zf.open(directory + "quadratic.npy", mode="r") as f:
            qdata = np.load(f, allow_pickle=False)

        with zf.open(directory + "coords.npy", mode="r") as f:
            coords = np.load(f, allow_pickle=False)

        # pass to the constructor
        return cls(predecessors[0], (qdata, coords), ldata)

    def _into_zipfile(self, zf, directory):
        """Save the QuadraticModel into a zipfile"""

        # Save it in a format that can be reconstructed using _init_coords
        # The square terms will be stored as quadratic, which is why we add
        # num_variables to the number of interactions
        cdef Py_ssize_t num_variables = self.num_variables()
        cdef Py_ssize_t num_terms = self.num_interactions() + num_variables

        cdef double[::1] ldata = np.empty(num_variables, dtype=np.double)
        cdef double[::1] qdata = np.empty(num_terms, dtype=np.double)
        cdef int[:,::1] coords = np.empty((2, num_terms), dtype=np.intc)

        # observing pointer
        cdef cppQuadraticModel* qm = self.ptr.get_quadratic_model()

        cdef Py_ssize_t i
        for i in range(num_variables):
            ldata[i] = qm.get_linear(i)

            qdata[i] = qm.get_quadratic(i, i)
            coords[0, i] = i
            coords[1, i] = i

        # this works because each row of coords is contiguous!
        qm.get_quadratic(&coords[0, num_variables], &coords[1, num_variables], &qdata[num_variables])

        with zf.open(directory + "linear.npy", mode="w", force_zip64=True) as f:
            np.save(f, ldata, allow_pickle=False)

        with zf.open(directory + "quadratic.npy", mode="w", force_zip64=True) as f:
            np.save(f, qdata, allow_pickle=False)

        with zf.open(directory + "coords.npy", mode="w", force_zip64=True) as f:
            np.save(f, coords, allow_pickle=False)

    cpdef Py_ssize_t num_interactions(self) noexcept:
        """The number of quadratic interactions in the quadratic model"""
        return self.ptr.get_quadratic_model().num_interactions()

    cpdef Py_ssize_t num_variables(self) noexcept:
        """The number of variables in the quadratic model."""
        return self.ptr.get_quadratic_model().num_variables()

    cdef cppQuadraticModelNode* ptr

_register(QuadraticModel, typeid(cppQuadraticModelNode))


cdef class SoftMax(ArraySymbol):
    """Softmax of a symbol.

    See Also:
        :meth:`~dwave.optimization.mathematical.softmax`: equivalent method.

    .. versionadded:: 0.6.5
    """
    def __init__(self, ArraySymbol arr):
        cdef _Graph model = arr.model

        cdef cppSoftMaxNode* ptr = model._graph.emplace_node[cppSoftMaxNode](arr.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(SoftMax, typeid(cppSoftMaxNode))

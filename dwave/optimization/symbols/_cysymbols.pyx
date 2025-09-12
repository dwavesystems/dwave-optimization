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
from dwave.optimization._utilities cimport as_cppshape, as_span


__all__ = [
    "Absolute",
    "AccumulateZip",
    "Add",
    "All",
    "And",
    "Any",
    "AdvancedIndexing",
    "ARange",
    "ArgSort",
    "BasicIndexing",
    "BinaryVariable",
    "BroadcastTo",
    "BSpline",
    "Concatenate",
    "Constant",
    "Copy",
    "Cos",
    "DisjointBitSets",
    "DisjointBitSet",
    "DisjointLists",
    "DisjointList",
    "Divide",
    "Equal",
    "Exp",
    "Expit",
    "Extract",
    "Input",
    "IntegerVariable",
    "LessEqual",
    "LinearProgram",
    "LinearProgramFeasible",
    "LinearProgramObjectiveValue",
    "LinearProgramSolution",
    "ListVariable",
    "Log",
    "Logical",
    "Max",
    "Maximum",
    "Mean",
    "Min",
    "Minimum",
    "Modulus",
    "Multiply",
    "NaryAdd",
    "NaryMaximum",
    "NaryMinimum",
    "NaryMultiply",
    "Negative",
    "Not",
    "Or",
    "PartialProd",
    "PartialSum",
    "Permutation",
    "Prod",
    "Put",
    "QuadraticModel",
    "Reshape",
    "Resize",
    "Sin",
    "Subtract",
    "SetVariable",
    "Size",
    "Rint",
    "SafeDivide",
    "SoftMax",
    "Square",
    "SquareRoot",
    "Sum",
    "Where",
    "Xor",
    ]


cdef class Absolute(ArraySymbol):
    """Absolute value element-wise on a symbol.

    See Also:
        :func:`~dwave.optimization.mathematical.absolute`: equivalent function.
    """
    def __init__(self, ArraySymbol x):
        cdef _Graph model = x.model

        cdef cppAbsoluteNode* ptr = model._graph.emplace_node[cppAbsoluteNode](x.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(Absolute, typeid(cppAbsoluteNode))


cdef class AccumulateZip(ArraySymbol):
    """Using a supplied :class:`~dwave.optimization.model.Expression`, perform
    an element-wise accumulate operation along one or more array operands. The
    accumulate operation (represented by the ``Expression``) takes as input one
    value from each of the operand arrays, as well as the result of the
    previously computed operation, and computes a new value at the next output
    index.

    This takes inspiration from
    `numpy.ufunc.accumulate <https://numpy.org/doc/2.1/reference/generated/numpy.ufunc.accumulate.html>`_ 
    but is different in that the accumulate operation can take an arbitrary
    number of arguments, instead of always two. These arguments come from
    "zipping" the supplied predecessor arrays together.

    Thus if we are given an expression `expr`, predecessor arrays `A`, `B`,
    `C`, and an initial value `init`, this node is equivalent to the
    pseudocode:

    .. code-block::

        r = [0] * len(A)
        t = init
        for args in zip(A, B, C):
            t = expr(t, *args)
            r[i] = t
        return r

    Args:
        expression:
            An :class:`~dwave.optimization.model.Expression` representing the
            accumulate operation. The first input on the expression will be given
            the previous output of the operation at each iteration over the
            values of the operands.
        operands:
            A list of the 1d-array symbols that will be the operands to the
            accumulate. There should be one fewer operands than inputs on the
            expression.
        initial (optional):
            A float representing the value used to start the accumulate. This
            will be used to set the last input of the expression on the very
            first iteration.

    .. versionadded:: 0.6.4
    """

    def __init__(self, expression, operands, object initial = 0):
        if not isinstance(expression, Expression):
            raise TypeError("expression must be of type Expression")

        # Get the underlying Graph representing our expression, and make sure it matches our
        # expectations.
        # Much of this is re-checked at the C++ level, but we can raise nicer errors here
        cdef _Graph expr = expression._model
        if expr is None:
            raise TypeError("expression has not been initialized")  # user bypassed __init__()
        for symbol in expr.iter_decisions():
            raise ValueError(
                "expression must only have inputs and constants as roots, "
                f"recieved an expression with {type(symbol).__name__} as a root"
            )

        if len(operands) == 0:
            raise ValueError("operands must contain at least one array symbol")
        if expr.num_inputs() != len(operands) + 1:
            raise ValueError("expression must have exactly one more input than number of operands")

        # Convert the operands into something we can pass to C++, and check that they all share a
        # model while we're at it
        cdef _Graph model = operands[0].model
        cdef vector[cppArrayNode*] cppoperands
        for symbol in operands:
            if symbol.model != model:
                raise ValueError("all predecessors must be from the same model")
            cppoperands.push_back((<ArraySymbol?>symbol).array_ptr)

        # Convert initial into something we can handle
        if isinstance(initial, ArraySymbol):
            self.ptr = model._graph.emplace_node[cppAccumulateZipNode](
                expr._owning_ptr, cppoperands, (<ArraySymbol>initial).array_ptr,
            )
        else:
            try:
                <double?>(initial)
            except TypeError:
                raise TypeError(
                    "expected type of `initial` to be either an int, float or an ArraySymbol, got "
                    f"{type(initial)}"
                )

            self.ptr = model._graph.emplace_node[cppAccumulateZipNode](
                expr._owning_ptr, cppoperands, <double>initial,
            )

        self.initialize_arraynode(model, self.ptr)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef cppAccumulateZipNode* ptr = dynamic_cast_ptr[cppAccumulateZipNode](
            symbol.node_ptr
        )
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")
        cdef AccumulateZip x = AccumulateZip.__new__(AccumulateZip)
        x.ptr = ptr
        x.initialize_arraynode(symbol.model, ptr)
        return x

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        from dwave.optimization import Model  # avoid circular import

        expression = Expression.__new__(Expression)  # bypass the __init__
        with zf.open(directory + "expression.nl", "r") as f:
            expression._model = Model.from_file(f)
            expression._model.lock()
        with zf.open(directory + "initial.json", "r") as f:
            initial_info = json.load(f)

        if initial_info["type"] == "double":
            initial = initial_info["value"]
            operands = predecessors
        elif initial_info["type"] == "node":
            initial = predecessors[0]
            operands = predecessors[1:]
        else:
            raise ValueError("unexpected type for initial value")

        return AccumulateZip(expression, operands, initial=initial)

    def initial(self):
        if holds_alternative[cppArrayNodePtr](self.ptr.initial):
            return symbol_from_ptr(self.model, get[cppArrayNodePtr](self.ptr.initial))
        else:
            return get[double](self.ptr.initial)

    def _into_zipfile(self, zf, directory):
        """Store a AccumulateZip symbol as a compressed file.

        Args:
            zf:
                File pointer to a compressed file to store the
                disjoint-list symbol. Strings are interpreted as a
                file name.
            directory:
                Directory where the file is located.
        Returns:
            A compressed file.

        See also:
            :meth:`._from_zipfile`
        """
        # Create a model from the shared_ptr held by the node. We need to be careful
        # not to modify it! That leads to undefined behavior.
        # There is also an asymmetry here where we serialize as a _Graph and then
        # load as Model. Right now that's OK, but we need to formalize that process.
        cdef _Graph expr = _Graph.from_shared_ptr(self.ptr.expression_ptr())
        assert expr._graph.topologically_sorted()
        assert expr._lock_count > 0  
        with zf.open(directory + "expression.nl", mode="w") as f:
            expr.into_file(f)

        # Now save information about the initial state
        encoder = json.JSONEncoder(separators=(',', ':'))
        if holds_alternative[cppArrayNodePtr](self.ptr.initial):
            initial_info = dict(
                type="node",
                value=get[cppArrayNodePtr](self.ptr.initial).topological_index()
            )
        else:
            initial_info = dict(type="double", value=get[double](self.ptr.initial))
        zf.writestr(directory + "initial.json", encoder.encode(initial_info))

    cdef cppAccumulateZipNode* ptr

_register(AccumulateZip, typeid(cppAccumulateZipNode))


cdef class Add(ArraySymbol):
    """Addition element-wise of two symbols.

    Examples:
        This example adds two integer symbols.

        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> i = model.integer(10, lower_bound=-50, upper_bound=50)
        >>> j = model.integer(10, lower_bound=0, upper_bound=10)
        >>> k = i + j
        >>> type(k)
        <class 'dwave.optimization.symbols.Add'>
    """
    def __init__(self, ArraySymbol lhs, ArraySymbol rhs):
        if lhs.model is not rhs.model:
            raise ValueError("lhs and rhs do not share the same underlying model")

        cdef _Graph model = lhs.model

        cdef cppAddNode* ptr = model._graph.emplace_node[cppAddNode](lhs.array_ptr, rhs.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(Add, typeid(cppAddNode))


cdef class All(ArraySymbol):
    """Tests whether all elements evaluate to True.

    Examples:
        This example checks all elements of a binary array.

        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> x = model.binary((20, 30))
        >>> all_x = x.all()
        >>> type(all_x)
        <class 'dwave.optimization.symbols.All'>
    """
    def __init__(self, ArraySymbol array):
        cdef _Graph model = array.model
        cdef cppAllNode* ptr = model._graph.emplace_node[cppAllNode](array.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(All, typeid(cppAllNode))


cdef class And(ArraySymbol):
    """Boolean AND element-wise between two symbols.

    See Also:
        :func:`~dwave.optimization.mathematical.logical_and`: equivalent function.
    """
    def __init__(self, ArraySymbol lhs, ArraySymbol rhs):
        if lhs.model is not rhs.model:
            raise ValueError("lhs and rhs do not share the same underlying model")

        cdef _Graph model = lhs.model

        cdef cppAndNode* ptr = model._graph.emplace_node[cppAndNode](lhs.array_ptr, rhs.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(And, typeid(cppAndNode))


cdef class Any(ArraySymbol):
    """Tests whether any elements evaluate to True.

    Examples:
        This example checks the elements of a binary array.

        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> model.states.resize(1)
        >>> x = model.constant([True, False, False])
        >>> a = x.any()
        >>> with model.lock():
        ...     assert a.state()

        >>> y = model.constant([False, False, False])
        >>> b = y.any()
        >>> with model.lock():
        ...     assert not b.state()

    .. versionadded:: 0.4.1
    """
    def __init__(self, ArraySymbol array):
        cdef _Graph model = array.model
        cdef cppAnyNode* ptr = model._graph.emplace_node[cppAnyNode](array.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(Any, typeid(cppAnyNode))


cdef class _ArrayValidation(Symbol):
    def __init__(self, ArraySymbol array_node):
        cdef _Graph model = array_node.model

        cdef cppArrayValidationNode* ptr = model._graph.emplace_node[cppArrayValidationNode](array_node.array_ptr)
        self.initialize_node(model, ptr)

_register(_ArrayValidation, typeid(cppArrayValidationNode))


ctypedef fused _start_type:
    ArraySymbol
    Py_ssize_t

ctypedef fused _stop_type:
    ArraySymbol
    Py_ssize_t

ctypedef fused _step_type:
    ArraySymbol
    Py_ssize_t


cdef class ARange(ArraySymbol):
    """Return evenly spaced integer values within a given interval.

    See Also:
        :func:`~dwave.optimization.mathematical.arange`: equivalent function.

    .. versionadded:: 0.5.2
    """
    def __init__(self, start, stop, step):
        ARange._init(self, start, stop, step)

    # Cython does not like fused types in the __init__, so we make a redundant one.
    # See https://github.com/cython/cython/issues/3758
    @staticmethod
    def _init(ARange self, _start_type start, _stop_type stop, _step_type step):
        # There are eight possible combinations of inputs, and unfortunately we
        # need to check them all
        if _start_type is Py_ssize_t and _stop_type is Py_ssize_t and _step_type is Py_ssize_t:
            raise ValueError(
                "ARange requires at least one symbol as an input. "
                f"Use model.constant(range({start}, {stop}, {step})) instead.")
        elif _start_type is Py_ssize_t and _stop_type is Py_ssize_t and _step_type is ArraySymbol:
            self.ptr = step.model._graph.emplace_node[cppARangeNode](start, stop, step.array_ptr)
            self.initialize_arraynode(step.model, self.ptr)
        elif _start_type is Py_ssize_t and _stop_type is ArraySymbol and _step_type is Py_ssize_t:
            self.ptr = stop.model._graph.emplace_node[cppARangeNode](start, stop.array_ptr, step)
            self.initialize_arraynode(stop.model, self.ptr)
        elif _start_type is Py_ssize_t and _stop_type is ArraySymbol and _step_type is ArraySymbol:
            if stop.model is not step.model:
                raise ValueError("stop and step do not share the same underlying model")
            self.ptr = stop.model._graph.emplace_node[cppARangeNode](start, stop.array_ptr, step.array_ptr)
            self.initialize_arraynode(stop.model, self.ptr)
        elif _start_type is ArraySymbol and _stop_type is Py_ssize_t and _step_type is Py_ssize_t:
            self.ptr = start.model._graph.emplace_node[cppARangeNode](start.array_ptr, stop, step)
            self.initialize_arraynode(start.model, self.ptr)
        elif _start_type is ArraySymbol and _stop_type is Py_ssize_t and _step_type is ArraySymbol:
            if start.model is not step.model:
                raise ValueError("start and step do not share the same underlying model")
            self.ptr = start.model._graph.emplace_node[cppARangeNode](start.array_ptr, stop, step.array_ptr)
            self.initialize_arraynode(start.model, self.ptr)
        elif _start_type is ArraySymbol and _stop_type is ArraySymbol and _step_type is Py_ssize_t:
            if start.model is not stop.model:
                raise ValueError("start and stop do not share the same underlying model")
            self.ptr = start.model._graph.emplace_node[cppARangeNode](start.array_ptr, stop.array_ptr, step)
            self.initialize_arraynode(start.model, self.ptr)
        elif _start_type is ArraySymbol and _stop_type is ArraySymbol and _step_type is ArraySymbol:
            if start.model is not stop.model or start.model is not step.model:
                raise ValueError("start, stop, and step do not share the same underlying model")
            self.ptr = start.model._graph.emplace_node[cppARangeNode](start.array_ptr, stop.array_ptr, step.array_ptr)
            self.initialize_arraynode(start.model, self.ptr)   
        else:
            raise RuntimeError  # shouldn't be possible

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef cppARangeNode* ptr = dynamic_cast_ptr[cppARangeNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")
        cdef ARange sym = cls.__new__(cls)
        sym.ptr = ptr
        sym.initialize_arraynode(symbol.model, ptr)
        return sym

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        with zf.open(directory + "args.json", "r") as f:
            args = json.load(f)

        if len(predecessors) + len(args) != 3:
            raise RuntimeError("unexpected number of arguments")

        predecessors = list(predecessors)  # just in case it's not pop-able

        if "step" in args:
            step = args["step"]
        else:
            step = predecessors.pop()
        if "stop" in args:
            stop = args["stop"]
        else:
            stop = predecessors.pop()
        if "start" in args:
            start = args["start"]
        else:
            start = predecessors.pop()

        return cls(start, stop, step)


    def _into_zipfile(self, zf, directory):
        super()._into_zipfile(zf, directory)

        encoder = json.JSONEncoder(separators=(',', ':'))

        # get the non-array args
        args = dict()

        start = self.ptr.start()
        if holds_alternative[Py_ssize_t](start):
            args.update(start=int(get[Py_ssize_t](start)))

        stop = self.ptr.stop()
        if holds_alternative[Py_ssize_t](stop):
            args.update(stop=int(get[Py_ssize_t](stop)))

        step = self.ptr.step()
        if holds_alternative[Py_ssize_t](step):
            args.update(step=int(get[Py_ssize_t](step)))

        zf.writestr(directory + "args.json", encoder.encode(args))

    cdef cppARangeNode* ptr

_register(ARange, typeid(cppARangeNode))


cdef class ArgSort(ArraySymbol):
    """Return an ordering of the indices that would sort (flattened) values
    of the given symbol. Note that while it will return an array with
    identical shape to the given symbol, the returned indices will always be
    indices on flattened array, similar to ``numpy.argsort(a, axis=None)``.

    Always performs a index-wise stable sort such that the relative order of
    values is maintained in the returned order.

    See Also:
        :meth:`~dwave.optimization.mathematical.argsort`: equivalent method.

    .. versionadded:: 0.6.4
    """
    def __init__(self, ArraySymbol arr):
        cdef _Graph model = arr.model

        cdef cppArgSortNode* ptr = model._graph.emplace_node[cppArgSortNode](arr.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(ArgSort, typeid(cppArgSortNode))


cdef bool _empty_slice(object slice_) noexcept:
    return slice_.start is None and slice_.stop is None and slice_.step is None


cdef class AdvancedIndexing(ArraySymbol):
    """Advanced indexing.

    Examples:
        This example uses advanced indexing to set a symbol's values.

        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> prices = model.constant([i for i in range(20)])
        >>> items = model.set(20)
        >>> values = prices[items]
        >>> type(values)
        <class 'dwave.optimization.symbols.AdvancedIndexing'>
    """
    def __init__(self, ArraySymbol array, *indices):
        cdef _Graph model = array.model

        cdef vector[cppAdvancedIndexingNode.array_or_slice] cppindices

        cdef ArraySymbol array_index
        for index in indices:
            if isinstance(index, slice):
                if index != slice(None):
                    raise ValueError("AdvancedIndexing can only parse empty slices")

                cppindices.emplace_back(cppSlice())
            else:
                array_index = index
                if array_index.model is not model:
                    raise ValueError("mismatched parent models")

                cppindices.emplace_back(array_index.array_ptr)

        self.ptr = model._graph.emplace_node[cppAdvancedIndexingNode](array.array_ptr, cppindices)

        self.initialize_arraynode(model, self.ptr)

    def __getitem__(self, index):
        # There is a very specific case we want to handle, when we are [x, :] or [:, x]
        # and we're doing the inverse indexing operation, and where the main array is
        # constant square matrix

        array = next(self.iter_predecessors())

        if (
            isinstance(array, Constant)
            and array.ndim() == 2
            and array.shape()[0] == array.shape()[1]  # square matrix
            and self.ptr.indices().size() == 2
            and isinstance(index, tuple)
            and len(index) == 2
        ):
            i0, i1 = index

            # check the [x, :][:, x] case
            if (isinstance(i0, slice) and _empty_slice(i0) and
                    isinstance(i1, ArraySymbol) and
                    holds_alternative[cppArrayNodePtr](self.ptr.indices()[0]) and
                    get[cppArrayNodePtr](self.ptr.indices()[0]) == (<ArraySymbol>i1).array_ptr and
                    holds_alternative[cppSlice](self.ptr.indices()[1])):

                return Permutation(array, i1)

            # check the [:, x][x, :] case
            if (isinstance(i1, slice) and _empty_slice(i1) and
                    isinstance(i0, ArraySymbol) and
                    holds_alternative[cppArrayNodePtr](self.ptr.indices()[1]) and
                    get[cppArrayNodePtr](self.ptr.indices()[1]) == (<ArraySymbol>i0).array_ptr and
                    holds_alternative[cppSlice](self.ptr.indices()[0])):

                return Permutation(array, i0)

        return super().__getitem__(index)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef cppAdvancedIndexingNode* ptr = dynamic_cast_ptr[cppAdvancedIndexingNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")

        cdef AdvancedIndexing sym = AdvancedIndexing.__new__(AdvancedIndexing)
        sym.ptr = ptr
        sym.initialize_arraynode(symbol.model, ptr)
        return sym

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        cdef cppNode* ptr

        indices = []
        with zf.open(directory + "indices.json", "r") as f:
            for index in json.load(f):
                if isinstance(index, numbers.Integral):
                    # lower topological index, so must exist
                    ptr = model._graph.nodes()[<Py_ssize_t>(index)].get()
                    indices.append(symbol_from_ptr(model, ptr))
                elif isinstance(index, list):
                    indices.append(slice(None))
                else:
                    raise RuntimeError("unexpected index")

        return cls(predecessors[0], *indices)

    def _into_zipfile(self, zf, directory):
        super()._into_zipfile(zf, directory)

        encoder = json.JSONEncoder(separators=(',', ':'))

        # traverse the indices. Storing arrays by their topological index and
        # slices as a triplet of (0, 0, 0) to be consistent with basic indexing
        indices = []

        cdef cppArrayNode* ptr
        for variant in self.ptr.indices():
            if holds_alternative[cppArrayNodePtr](variant):
                ptr = get[cppArrayNodePtr](variant)
                indices.append(symbol_from_ptr(self.model, ptr).topological_index())
            elif holds_alternative[cppSlice](variant):
                indices.append((0, 0, 0))
            else:
                raise RuntimeError

        zf.writestr(directory + "indices.json", encoder.encode(indices))

    cdef cppAdvancedIndexingNode* ptr

_register(AdvancedIndexing, typeid(cppAdvancedIndexingNode))


cdef class BasicIndexing(ArraySymbol):
    """Basic indexing.

    Examples:
        This example uses basic indexing to set a symbol's values.

        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> prices = model.constant([i for i in range(20)])
        >>> low_prices = prices[:10]
        >>> type(low_prices)
        <class 'dwave.optimization.symbols.BasicIndexing'>
    """
    def __init__(self, ArraySymbol array, *indices):

        cdef _Graph model = array.model

        cdef vector[cppBasicIndexingNode.slice_or_int] cppindices
        for index in indices:
            if isinstance(index, slice):
                cppindices.emplace_back(BasicIndexing.cppslice(index))
            else:
                cppindices.emplace_back(<Py_ssize_t>(index))

        self.ptr = model._graph.emplace_node[cppBasicIndexingNode](array.array_ptr, cppindices)

        self.initialize_arraynode(model, self.ptr)

    @staticmethod
    cdef cppSlice cppslice(object index):
        """Create a cppSlice from a Python slice object."""
        cdef optional[Py_ssize_t] start
        cdef optional[Py_ssize_t] stop
        cdef optional[Py_ssize_t] step

        if index.start is not None:
            start = <Py_ssize_t>(index.start)
        if index.stop is not None:
            stop = <Py_ssize_t>(index.stop)
        if index.step is not None:
            step = <Py_ssize_t>(index.step)

        return cppSlice(start, stop, step)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef cppBasicIndexingNode* ptr = dynamic_cast_ptr[cppBasicIndexingNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")

        cdef BasicIndexing sym = BasicIndexing.__new__(BasicIndexing)
        sym.ptr = ptr
        sym.initialize_arraynode(symbol.model, ptr)
        return sym

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        if len(predecessors) != 1:
            raise ValueError(f"`BasicIndexing` should have exactly one predecessor")

        with zf.open(directory + "indices.json", "r") as f:
            indices = json.load(f)

        # recover the slices
        indices = [idx if isinstance(idx, int) else slice(*idx) for idx in indices]

        return cls(predecessors[0], *indices)

    def _infer_indices(self):
        """Get the indices that induced the view"""

        indices = []  # will contain the returned indices

        # help cython out with type inference
        cdef cppSlice cppslice
        cdef Py_ssize_t index

        # ok, lets iterate
        for variant in self.ptr.infer_indices():
            if holds_alternative[cppSlice](variant):
                cppslice = get[cppSlice](variant)
                indices.append(slice(cppslice.start, cppslice.stop, cppslice.step))
            else:
                index = get[Py_ssize_t](variant)
                indices.append(index)

        return tuple(indices)

    def _into_zipfile(self, zf, directory):
        super()._into_zipfile(zf, directory)

        encoder = json.JSONEncoder(separators=(',', ':'))

        indices = [(idx.start, idx.stop, idx.step) if isinstance(idx, slice) else idx
                   for idx in self._infer_indices()]

        zf.writestr(directory + "indices.json", encoder.encode(indices))

    cdef cppBasicIndexingNode* ptr

_register(BasicIndexing, typeid(cppBasicIndexingNode))


cdef class BinaryVariable(ArraySymbol):
    """Binary decision-variable symbol.

    Examples:
        This example adds a binary variable to a model.

        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> x = model.binary((20, 30))
        >>> type(x)
        <class 'dwave.optimization.symbols.BinaryVariable'>
    """
    def __init__(self, _Graph model, shape=None):
        # Get an observing pointer to the node
        cdef vector[Py_ssize_t] vshape = as_cppshape(tuple() if shape is None else shape)

        self.ptr = model._graph.emplace_node[cppBinaryNode](vshape)

        self.initialize_arraynode(model, self.ptr)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef cppBinaryNode* ptr = dynamic_cast_ptr[cppBinaryNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")

        cdef BinaryVariable x = BinaryVariable.__new__(BinaryVariable)
        x.ptr = ptr
        x.initialize_arraynode(symbol.model, ptr)
        return x

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        """Construct a binary symbol from a compressed file.

        Args:
            zf:
                File pointer to a compressed file encoding
                a binary symbol. Strings are interpreted as a file name.
            directory:
                Directory where the file is located.
            model:
                The relevant :class:`~dwave.optimization.model.Model`.
            predecessors:
                Not currently supported.
        Returns:
            A binary symbol.

        See also:
            :meth:`._into_zipfile`
        """
        if predecessors:
            raise ValueError(f"{cls.__name__} cannot have predecessors")

        with zf.open(directory + "shape.json", "r") as f:
            shape_info = json.load(f)

        return BinaryVariable(model, shape_info["shape"])

    def _into_zipfile(self, zf, directory):
        """Store a binary symbol as a compressed file.

        Args:
            zf:
                File pointer to a compressed file to store the
                binary symbol. Strings are interpreted as a file name.
            directory:
                Directory where the file is located.
        Returns:
            A compressed file.

        See also:
            :meth:`._from_zipfile`
        """
        # the additional data we want to encode

        shape_info = dict(
            shape=self.shape()
            )

        encoder = json.JSONEncoder(separators=(',', ':'))

        zf.writestr(directory + "shape.json", encoder.encode(shape_info))

    def set_state(self, Py_ssize_t index, state):
        r"""Set the state of the binary symbol.

        The given state must be binary array with the same shape
        as the symbol.

        Args:
            index:
                Index of the state to set
            state:
                Assignment of values for the state.

        Examples:
            This example sets two states for a :math:`2 \times 3`-sized
            binary symbol.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> x = model.binary((2, 3))
            >>> model.states.resize(2)
            >>> x.set_state(0, [[True, True, False], [False, True, False]])
            >>> x.set_state(1, [[False, True, False], [False, True, False]])
        """
        # Convert the state into something we can handle in C++.
        # This also does some type checking etc
        # We go ahead and do the translation to integer now
        cdef Py_ssize_t[:] arr = np.asarray(state, dtype=np.intp).flatten()

        # Reset our state, and check whether that's possible
        self.reset_state(index)

        # Convert to a vector. We could skip this copy if it ever becomes
        # a performance bottleneck
        cdef vector[double] items
        items.reserve(arr.size)
        cdef Py_ssize_t i
        for i in range(arr.size):
            items.push_back(arr[i])

        # The validity of the state is checked in C++
        self.ptr.initialize_state((<States>self.model.states)._states[index], move(items))

    # An observing pointer to the C++ BinaryNode
    cdef cppBinaryNode* ptr

_register(BinaryVariable, typeid(cppBinaryNode))


cdef class BroadcastTo(ArraySymbol):
    """BroadcastTo symbol.

    See Also:
        :func:`~dwave.optimization.mathematical.broadcast_to`: equivalent function.

    .. versionadded:: 0.6.5
    """
    def __init__(self, ArraySymbol node, shape):
        cdef _Graph model = node.model

        cdef cppBroadcastToNode* ptr = model._graph.emplace_node[cppBroadcastToNode](
            node.array_ptr,
            as_cppshape(shape, nonnegative=False),
        )

        self.initialize_arraynode(model, ptr)

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        if len(predecessors) != 1:
            raise ValueError(f"{cls.__name__} must have exactly one predecessor")

        with zf.open(directory + "shape.json", "r") as f:
            return BroadcastTo(*predecessors, json.load(f))

    def _into_zipfile(self, zf, directory):
        encoder = json.JSONEncoder(separators=(',', ':'))
        zf.writestr(directory + "shape.json", encoder.encode(self.shape()))

    def state_size(self):
        """Broadcasting symbols are stateless"""
        return 0

_register(BroadcastTo, typeid(cppBroadcastToNode))


cdef class BSpline(ArraySymbol):
    """Bspline node that takes in an array pointer, an integer degree and two vectors for knots and coefficients.

    Examples:
        This example creates a BSpline symbol.

        >>> from dwave.optimization.model import Model
        >>> from dwave.optimization.mathematical import bspline
        >>> model = Model()
        >>> x = model.integer(lower_bound=3, upper_bound=4)
        >>> k = 2
        >>> t = [0, 1, 2, 3, 4, 5, 6]
        >>> c = [-1, 2, 0, -1]
        >>> bspline_node = bspline(x, k, t, c)
        >>> type(bspline_node)
        <class 'dwave.optimization.symbols.BSpline'>
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


cdef class Concatenate(ArraySymbol):
    """Concatenate symbol.

    See Also:
        :func:`~dwave.optimization.mathematical.concatenate()` equivalent function.

    .. versionadded:: 0.4.3
    """
    def __init__(self, object inputs, Py_ssize_t axis = 0):
        if (not isinstance(inputs, collections.abc.Sequence) or
                not all(isinstance(arr, ArraySymbol) for arr in inputs)):
            raise TypeError("concatenate takes a sequence of array symbols")

        if len(inputs) < 1:
            raise ValueError("need at least one array symbol to concatenate")

        cdef _Graph model = inputs[0].model
        cdef vector[cppArrayNode*] cppinputs

        for symbol in inputs:
            if symbol.model is not model:
                raise ValueError("all predecessors must be from the same model")
            cppinputs.push_back((<ArraySymbol?>symbol).array_ptr)

        self.ptr = model._graph.emplace_node[cppConcatenateNode](cppinputs, axis)
        self.initialize_arraynode(model, self.ptr)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef cppConcatenateNode* ptr = dynamic_cast_ptr[cppConcatenateNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")

        cdef Concatenate m = Concatenate.__new__(Concatenate)
        m.ptr = ptr
        m.initialize_arraynode(symbol.model, ptr)
        return m

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        if len(predecessors) < 1:
            raise ValueError("Concatenate must have at least one predecessor")

        with zf.open(directory + "axis.json", "r") as f:
            return Concatenate(tuple(predecessors), axis=json.load(f))

    def _into_zipfile(self, zf, directory):
        encoder = json.JSONEncoder(separators=(',', ':'))
        zf.writestr(directory + "axis.json", encoder.encode(self.axis()))

    def axis(self):
        return self.ptr.axis()

    cdef cppConcatenateNode* ptr

_register(Concatenate, typeid(cppConcatenateNode))


cdef extern from *:
    """
    #include "Python.h"

    struct PyDataSource : dwave::optimization::ConstantNode::DataSource {
        PyDataSource(PyObject* ptr) : ptr_(ptr) {
            Py_INCREF(ptr_);
        }
        ~PyDataSource() {
            Py_DECREF(ptr_);
        }

        PyObject* ptr_;
    };
    """
    cppclass PyDataSource:
        PyDataSource(PyObject*)


cdef class Constant(ArraySymbol):
    """Constant symbol.

    See also:
        :meth:`~dwave.optimization.model.Model.constant`: equivalent method.
    """
    def __init__(self, _Graph model, array_like):
        # In the future we won't need to be contiguous, but we do need to be right now
        array = np.asarray_chkfinite(array_like, dtype=np.double, order="C")

        # Get the shape and strides
        cdef vector[Py_ssize_t] shape = array.shape
        cdef vector[Py_ssize_t] strides = array.strides  # not used because contiguous for now

        # Get a pointer to the first element
        cdef const double[:] flat = array.ravel()
        cdef const double* start = NULL
        if flat.size:
            start = &flat[0]

        # Make a PyDataSource that will essentially take ownership of the numpy array,
        # preventing garbage collection from deallocating it before the C++ node is
        # destructed
        cdef unique_ptr[PyDataSource] data_source = make_unique[PyDataSource](<PyObject*>(array))
        # Get an observing pointer to the C++ ConstantNode
        self.ptr = model._graph.emplace_node[cppConstantNode](move(data_source), start, shape)

        self.initialize_arraynode(model, self.ptr)

    def __bool__(self):
        if not self._is_scalar():
            raise ValueError("the truth value of a constant with more than one element is ambiguous")

        return <bool>deref(self.ptr.buff())

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        # We never export a writeable array
        if flags & cpython.buffer.PyBUF_WRITABLE == cpython.buffer.PyBUF_WRITABLE:
            raise BufferError(f"{type(self).__name__} cannot export a writeable buffer")

        # The remaining flags are accurate to the information we export, but over-zealous.
        # We could, for instance, check whether we're contiguous and in that case not raise
        # an error.
        # But for now, we *always* expose strides, format, and we never assume that we're
        # contiguous.
        # Luckily, NumPy and memoryview always ask for everything so it doesn't really matter.
        # If there is a compelling use case we can add more information.
        if flags & cpython.buffer.PyBUF_STRIDES != cpython.buffer.PyBUF_STRIDES:
            raise BufferError(f"{type(self).__name__} always returns stride information")
        if flags & cpython.buffer.PyBUF_FORMAT != cpython.buffer.PyBUF_FORMAT:
            raise BufferError(f"{type(self).__name__} always sets the format field")
        if (flags & cpython.buffer.PyBUF_ANY_CONTIGUOUS == cpython.buffer.PyBUF_ANY_CONTIGUOUS or
                flags & cpython.buffer.PyBUF_C_CONTIGUOUS == cpython.buffer.PyBUF_C_CONTIGUOUS or
                flags & cpython.buffer.PyBUF_F_CONTIGUOUS == cpython.buffer.PyBUF_F_CONTIGUOUS):
            raise BufferError(f"{type(self).__name__} is not necessarily contiguous")

        buffer.buf = <void*>(self.ptr.buff())
        buffer.format = <char*>(self.ptr.format().c_str())
        buffer.internal = NULL
        buffer.itemsize = self.ptr.itemsize()
        buffer.len = self.ptr.len()
        buffer.ndim = self.ptr.ndim()
        buffer.obj = self
        buffer.readonly = 1
        buffer.shape = <Py_ssize_t*>(self.ptr.shape().data())
        buffer.strides = <Py_ssize_t*>(self.ptr.strides().data())
        buffer.suboffsets = NULL

    def __index__(self):
        if not self._is_integer():
            # Follow NumPy's error message
            # https://github.com/numpy/numpy/blob/66e1e3/numpy/_core/src/multiarray/number.c#L833
            raise TypeError("only integer scalar constants can be converted to a scalar index")

        return <Py_ssize_t>deref(self.ptr.buff())

    def __richcmp__(self, rhs, int op):
        # __richcmp__ is a special Cython method

        # If rhs is another Symbol, defer to ArraySymbol to handle the
        # operation. Which may or may not actually be implemented.
        # Otherwise, defer to NumPy.
        # We could also check if rhs is another Constant and handle that differently,
        # but that might lead to confusing behavior so we treat other Constants the
        # same as any other symbol.
        lhs = super() if isinstance(rhs, ArraySymbol) else np.asarray(self)

        if op == cpython.object.Py_EQ:
            return lhs.__eq__(rhs)
        elif op == cpython.object.Py_GE:
            return lhs.__ge__(rhs)
        elif op == cpython.object.Py_GT:
            return lhs.__gt__(rhs)
        elif op == cpython.object.Py_LE:
            return lhs.__le__(rhs)
        elif op == cpython.object.Py_LT:
            return lhs.__lt__(rhs)
        elif op == cpython.object.Py_NE:
            return lhs.__ne__(rhs)
        else:
            return NotImplemented  # this should never happen, but just in case

    cdef bool _is_integer(self) noexcept:
        """Return True if the constant encodes a single integer."""
        if not self._is_scalar():
            return False

        # https://stackoverflow.com/q/1521607 for the integer test
        cdef double dummy
        return modf(deref(self.ptr.buff()), &dummy) == <double>0.0

    cdef bool _is_scalar(self) noexcept:
        """Return True if the constant encodes a single value."""
        # The size check is redundant, but worth checking in order to avoid segfaults
        return self.ptr.size() == 1 and self.ptr.ndim() == 0

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef cppConstantNode* ptr = dynamic_cast_ptr[cppConstantNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")

        cdef Constant constant = Constant.__new__(Constant)
        constant.ptr = ptr
        constant.initialize_arraynode(symbol.model, ptr)
        return constant

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        """Construct a constant symbol from a compressed file.

        Args:
            zf:
                File pointer to a compressed file encoding
                a constant symbol. Strings are interpreted as a file name.
            directory:
                Directory where the file is located.
            model:
                The relevant :class:`~dwave.optimization.model.Model`.
            predecessors:
                Not currently supported.
        Returns:
            A constant symbol.

        See also:
            :meth:`._into_zipfile`
        """
        if predecessors:
            raise ValueError(f"{cls.__name__} cannot have predecessors")

        with zf.open(directory + "array.npy", mode="r") as f:
            array = np.load(f, allow_pickle=False)

        return cls(model, array)

    def _into_zipfile(self, zf, directory):
        """Store a constant symbol as a compressed file.

        Args:
            zf:
                File pointer to a compressed file to store the
                constant symbol. Strings are interpreted as a file name.
            directory:
                Directory where the file is located.
        Returns:
            A compressed file.

        See also:
            :meth:`._from_zipfile`
        """
        super()._into_zipfile(zf, directory)
        with zf.open(directory + "array.npy", mode="w", force_zip64=True) as f:
            # dev note: I benchmarked using some lower-level functions
            # like np.lib.format.write_array() etc and it didn't have
            # any noticeable impact on performance (numpy==1.26.3).
            np.save(f, self, allow_pickle=False)

    def maybe_equals(self, other):
        cdef Py_ssize_t maybe = super().maybe_equals(other)
        cdef Py_ssize_t NOT = 0
        cdef Py_ssize_t MAYBE = 1
        cdef Py_ssize_t DEFINITELY = 2
        if maybe != MAYBE:
            return DEFINITELY if maybe else NOT

        # avoid NumPy deprecation warning by casting to bool. But also
        # `bool` in this namespace is a C++ class so we do an explicit if else
        equal = (np.asarray(self) == np.asarray(other)).all()
        return DEFINITELY if equal else NOT

    def state(self, Py_ssize_t index=0, *, bool copy = True):
        """Return the state of the constant symbol.

        Args:
            index:
                Index of the state.
            copy:
                Copy the state. Currently only ``True`` is supported.
        Returns:
            A copy of the state.
        """
        if not copy:
            raise NotImplementedError("copy=False is not (yet) supported")

        return np.array(self, copy=copy)

    # An observing pointer to the C++ ConstantNode
    cdef cppConstantNode* ptr

_register(Constant, typeid(cppConstantNode))


cdef class Copy(ArraySymbol):
    """An array symbol that is a copy of another array symbol.

    See Also:
        :meth:`ArraySymbol.copy` Equivalent method.

    .. versionadded:: 0.5.1
    """
    def __init__(self, ArraySymbol node):
        cdef _Graph model = node.model

        cdef cppCopyNode* ptr = model._graph.emplace_node[cppCopyNode](node.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(Copy, typeid(cppCopyNode))


cdef class Cos(ArraySymbol):
    """Cosine element-wise on a symbol.

    See Also:
        :func:`~dwave.optimization.mathematical.cos`: equivalent function.

    .. versionadded:: 0.6.5
    """
    def __init__(self, ArraySymbol x):
        cdef _Graph model = x.model

        cdef cppCosNode* ptr = model._graph.emplace_node[cppCosNode](x.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(Cos, typeid(cppCosNode))


cdef class DisjointBitSets(Symbol):
    """Disjoint-sets decision-variable symbol.

    Examples:
        This example adds a disjoint-sets symbol to a model.

        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> s = model.disjoint_bit_sets(primary_set_size=100, num_disjoint_sets=5)
        >>> type(s[0])
        <class 'dwave.optimization.symbols.DisjointBitSets'>
    """
    def __init__(
        self, _Graph model, Py_ssize_t primary_set_size, Py_ssize_t num_disjoint_sets
    ):
        # Get an observing pointer to the node
        self.ptr = model._graph.emplace_node[cppDisjointBitSetsNode](
            primary_set_size, num_disjoint_sets
        )

        self.initialize_node(model, self.ptr)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef cppDisjointBitSetsNode* ptr = dynamic_cast_ptr[cppDisjointBitSetsNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")

        cdef DisjointBitSets x = DisjointBitSets.__new__(DisjointBitSets)
        x.ptr = ptr
        x.initialize_node(symbol.model, ptr)
        return x

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        """Construct a disjoint-sets symbol from a compressed file.

        Args:
            zf:
                File pointer to a compressed file encoding a
                disjoint-sets symbol. Strings are interpreted as a
                file name.
            directory:
                Directory where the file is located.
            model:
                The relevant :class:`~dwave.optimization.model.Model`.
            predecessors:
                Not currently supported.
        Returns:
            A disjoint-sets symbol.

        See also:
            :meth:`._into_zipfile`
        """
        if predecessors:
            raise ValueError(f"{cls.__name__} cannot have predecessors")

        with zf.open(directory + "shape.json", "r") as f:
            shape_info = json.load(f)

        return DisjointBitSets(
            model,
            primary_set_size=shape_info["primary_set_size"],
            num_disjoint_sets=shape_info["num_disjoint_sets"],
        )

    def _into_zipfile(self, zf, directory):
        """Store a disjoint-sets symbol as a compressed file.

        Args:
            zf:
                File pointer to a compressed file to store the
                disjoint-sets symbol. Strings are interpreted as a
                file name.
            directory:
                Directory where the file is located.
        Returns:
            A compressed file.

        See also:
            :meth:`._from_zipfile`
        """
        # the additional data we want to encode
        shape_info = dict(
            primary_set_size=int(self.ptr.primary_set_size()),  # max is inclusive
            num_disjoint_sets=self.ptr.num_disjoint_sets(),
        )

        encoder = json.JSONEncoder(separators=(',', ':'))

        zf.writestr(directory + "shape.json", encoder.encode(shape_info))

    def set_state(self, Py_ssize_t index, state):
        r"""Set the state of the disjoint-sets symbol.

        The given state must be a partition of ``range(primary_set_size)``
        into :meth:`.num_disjoint_sets` partitions, encoded as a 2D
        :code:`num_disjoint_sets` :math:`\times` :code:`primary_set_size` 
        Boolean array.

        Args:
            index:
                Index of the state to set
            state:
                Assignment of values for the state.
        """
        # Reset our state, and check whether that's possible
        self.reset_state(index)

        # Convert the state into something we can handle in C++.
        # This also does some type checking etc
        # We go ahead and do the translation to integer now
        cdef bool[:, :] arr = np.asarray(state, dtype=np.bool_)

        # Convert to a vector. We could skip this copy if it ever becomes
        # a performance bottleneck
        cdef vector[vector[double]] sets
        sets.resize(arr.shape[0])
        cdef Py_ssize_t i
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                sets[i].push_back(arr[i, j])

        # The validity of the state is checked in C++
        self.ptr.initialize_state((<States>self.model.states)._states[index], move(sets))

    def _state_from_zipfile(self, zf, directory, Py_ssize_t state_index):
        arrays = []
        for i in range(self.num_disjoint_sets()):
            with zf.open(directory+f"set{i}", mode="r") as f:
                arrays.append(np.load(f, allow_pickle=False))

        self.set_state(state_index, arrays)

    def _states_from_zipfile(self, zf, *, num_states, version):
        directory = f"nodes/{self.topological_index()}/"
        for i in range(num_states):
            self._state_from_zipfile(zf, f"{directory}states/{i}/", i)

    def _state_into_zipfile(self, zf, directory, Py_ssize_t state_index):
        # do this first to get any potential error messages out of the way
        # todo: use a view not a copy
        arrays = [np.asarray(s.state(state_index), dtype=np.int8) for s in self.iter_successors()]

        for i, arr in enumerate(arrays):
            with zf.open(directory+f"set{i}", mode="w", force_zip64=True) as f:
                np.save(f, arr, allow_pickle=False)

    def _states_into_zipfile(self, zf, *, num_states, version):
        directory = f"nodes/{self.topological_index()}/"
        for i in filter(self.has_state, range(num_states)):
            self._state_into_zipfile(
                zf,
                directory=f"{directory}states/{i}/",
                state_index=i,
                )

    def num_disjoint_sets(self):
        """Return the number of disjoint sets in the symbol."""
        return self.ptr.num_disjoint_sets()

    # An observing pointer to the C++ DisjointBitSetsNode
    cdef cppDisjointBitSetsNode* ptr

_register(DisjointBitSets, typeid(cppDisjointBitSetsNode))


cdef class DisjointBitSet(ArraySymbol):
    """Disjoint-sets successor symbol.

    Examples:
        This example adds a disjoint-sets symbol to a model.

        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> s = model.disjoint_bit_sets(primary_set_size=100, num_disjoint_sets=5)
        >>> type(s[1][0])
        <class 'dwave.optimization.symbols.DisjointBitSet'>
    """
    def __init__(self, DisjointBitSets parent, Py_ssize_t set_index):
        if set_index < 0 or set_index >= parent.num_disjoint_sets():
            raise ValueError(
                "`set_index` must be less than the number of disjoint sets of the parent"
            )

        if set_index > <Py_ssize_t>(parent.ptr.successors().size()):
            raise ValueError("`DisjointBitSet`s must be created successively")

        cdef _Graph model = parent.model
        if set_index == <Py_ssize_t>(parent.ptr.successors().size()):
            # The DisjointBitSet has not been added to the model yet, so add it
            self.ptr = model._graph.emplace_node[cppDisjointBitSetNode](parent.ptr)
        else:
            # Already been added to the model, so grab the pointer from the parent's
            # successors
            self.ptr = dynamic_cast_ptr[cppDisjointBitSetNode](
                parent.ptr.successors()[set_index].ptr
            )

        self.initialize_arraynode(model, self.ptr)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef cppDisjointBitSetNode* ptr = dynamic_cast_ptr[cppDisjointBitSetNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")
        cdef DisjointBitSet x = DisjointBitSet.__new__(DisjointBitSet)
        x.ptr = ptr
        x.initialize_arraynode(symbol.model, ptr)
        return x

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        """Construct a disjoint-set symbol from a compressed file.

        Args:
            zf:
                File pointer to a compressed file encoding a
                disjoint-set symbol. Strings are interpreted as a
                file name.
            directory:
                Directory where the file is located.
            model:
                The relevant :class:`~dwave.optimization.model.Model`.
            predecessors:
                Not currently supported.
        Returns:
            A disjoint-set symbol.

        See also:
            :meth:`._into_zipfile`
        """
        if len(predecessors) != 1:
            raise ValueError(f"`DisjointBitSet` should have exactly one predecessor")

        with zf.open(directory + "index.json", "r") as f:
            index = json.load(f)

        return DisjointBitSet(predecessors[0], index)

    def _into_zipfile(self, zf, directory):
        """Store a disjoint-set symbol as a compressed file.

        Args:
            zf:
                File pointer to a compressed file to store the
                disjoint-set symbol. Strings are interpreted as a
                file name.
            directory:
                Directory where the file is located.
        Returns:
            A compressed file.

        See also:
            :meth:`._from_zipfile`
        """
        # the additional data we want to encode
        encoder = json.JSONEncoder(separators=(',', ':'))

        zf.writestr(directory + "index.json", encoder.encode(self.set_index()))

    def set_index(self):
        """Return the index for the set."""
        return self.ptr.set_index()

    # An observing pointer to the C++ DisjointBitSetNode
    cdef cppDisjointBitSetNode* ptr

_register(DisjointBitSet, typeid(cppDisjointBitSetNode))


cdef class DisjointLists(Symbol):
    """Disjoint-lists decision-variable symbol.

    Examples:
        This example adds a disjoint-lists symbol to a model.

        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> l = model.disjoint_lists(primary_set_size=10, num_disjoint_lists=2)
        >>> type(l[0])
        <class 'dwave.optimization.symbols.DisjointLists'>
    """
    def __init__(
        self, _Graph model, Py_ssize_t primary_set_size, Py_ssize_t num_disjoint_lists
    ):
        # Get an observing pointer to the node
        self.ptr = model._graph.emplace_node[cppDisjointListsNode](
            primary_set_size, num_disjoint_lists
        )

        self.initialize_node(model, self.ptr)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef cppDisjointListsNode* ptr = dynamic_cast_ptr[cppDisjointListsNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")
        cdef DisjointLists x = DisjointLists.__new__(DisjointLists)
        x.ptr = ptr
        x.initialize_node(symbol.model, ptr)
        return x

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        """Construct a disjoint-lists symbol from a compressed file.

        Args:
            zf:
                File pointer to a compressed file encoding a
                disjoint-lists symbol. Strings are interpreted as a
                file name.
            directory:
                Directory where the file is located.
            model:
                The relevant :class:`~dwave.optimization.model.Model`.
            predecessors:
                Not currently supported.
        Returns:
            A disjoint-lists symbol.

        See also:
            :meth:`._into_zipfile`
        """
        if predecessors:
            raise ValueError(f"{cls.__name__} cannot have predecessors")

        with zf.open(directory + "shape.json", "r") as f:
            shape_info = json.load(f)

        return DisjointLists(
            model,
            primary_set_size=shape_info["primary_set_size"],
            num_disjoint_lists=shape_info["num_disjoint_lists"],
        )

    def _into_zipfile(self, zf, directory):
        """Store a disjoint-lists symbol as a compressed file.

        Args:
            zf:
                File pointer to a compressed file to store the
                disjoint-lists symbol. Strings are interpreted as a
                file name.
            directory:
                Directory where the file is located.
        Returns:
            A compressed file.

        See also:
            :meth:`._from_zipfile`
        """
        # the additional data we want to encode
        shape_info = dict(
            primary_set_size=self.ptr.primary_set_size(),
            num_disjoint_lists=self.ptr.num_disjoint_lists(),
        )

        encoder = json.JSONEncoder(separators=(',', ':'))

        zf.writestr(directory + "shape.json", encoder.encode(shape_info))

    def set_state(self, Py_ssize_t index, state):
        r"""Set the state of the disjoint-lists symbol.

        The given state must be a partition of ``range(primary_set_size)``
        into :meth:`.num_disjoint_lists` partitions as a list of lists.

        Args:
            index:
                Index of the state to set
            state:
                Assignment of values for the state.
        """
        # Reset our state, and check whether that's possible
        self.reset_state(index)

        # Convert to a vector. We could skip this copy if it ever becomes
        # a performance bottleneck
        cdef vector[vector[double]] items
        items.resize(len(state))
        cdef Py_ssize_t i, j
        cdef Py_ssize_t[:] arr
        for i in range(len(state)):
            items[i].reserve(len(state[i]))
            # Convert to a numpy array for type checking, coercion, etc.
            arr = np.asarray(state[i], dtype=np.intp)
            for j in range(len(state[i])):
                items[i].push_back(arr[j])

        # The validity of the state is checked in C++
        self.ptr.initialize_state((<States>self.model.states)._states[index], move(items))

    def _state_from_zipfile(self, zf, directory, Py_ssize_t state_index):
        arrays = []
        for i in range(self.num_disjoint_lists()):
            with zf.open(directory+f"list{i}", mode="r") as f:
                arrays.append(np.load(f, allow_pickle=False))

        self.set_state(state_index, arrays)

    def _states_from_zipfile(self, zf, num_states, version):
        directory = f"nodes/{self.topological_index()}/"
        for i in range(num_states):
            self._state_from_zipfile(zf, f"{directory}states/{i}/", i)

    def _state_into_zipfile(self, zf, directory, Py_ssize_t state_index):
        # do this first to get any potential error messages out of the way
        # todo: use a view not a copy
        arrays = [li.state(state_index) for li in self.iter_successors()]

        for i, arr in enumerate(arrays):
            with zf.open(directory+f"list{i}", mode="w", force_zip64=True) as f:
                np.save(f, arr, allow_pickle=False)

    def _states_into_zipfile(self, zf, num_states, version):
        directory = f"nodes/{self.topological_index()}/"
        for i in filter(self.has_state, range(num_states)):
            self._state_into_zipfile(
                zf,
                directory=f"{directory}states/{i}/",
                state_index=i,
                )

    def num_disjoint_lists(self):
        """Return the number of disjoint lists in the symbol."""
        return self.ptr.num_disjoint_lists()

    # An observing pointer to the C++ DisjointListsNode
    cdef cppDisjointListsNode* ptr

_register(DisjointLists, typeid(cppDisjointListsNode))


cdef class DisjointList(ArraySymbol):
    """Disjoint-lists successor symbol.

    Examples:
        This example adds a disjoint-lists symbol to a model.

        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> l = model.disjoint_lists(primary_set_size=10, num_disjoint_lists=2)
        >>> type(l[1][0])
        <class 'dwave.optimization.symbols.DisjointList'>
    """
    def __init__(self, DisjointLists parent, Py_ssize_t list_index):
        if list_index < 0 or list_index >= parent.num_disjoint_lists():
            raise ValueError(
                "`list_index` must be less than the number of disjoint sets of the parent"
            )

        if list_index > <Py_ssize_t>(parent.ptr.successors().size()):
            raise ValueError("`DisjointList`s must be created successively")

        cdef _Graph model = parent.model
        if list_index == <Py_ssize_t>(parent.ptr.successors().size()):
            # The DisjointListNode has not been added to the model yet, so add it
            self.ptr = model._graph.emplace_node[cppDisjointListNode](parent.ptr)
        else:
            # Already been added to the model, so grab the pointer from the parent's
            # successors
            self.ptr = dynamic_cast_ptr[cppDisjointListNode](
                parent.ptr.successors()[list_index].ptr
            )

        self.initialize_arraynode(model, self.ptr)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef cppDisjointListNode* ptr = dynamic_cast_ptr[cppDisjointListNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")
        cdef DisjointList x = DisjointList.__new__(DisjointList)
        x.ptr = ptr
        x.initialize_arraynode(symbol.model, ptr)
        return x

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        """Construct a disjoint-list symbol from a compressed file.

        Args:
            zf:
                File pointer to a compressed file encoding a
                disjoint-list symbol. Strings are interpreted as a
                file name.
            directory:
                Directory where the file is located.
            model:
                The relevant :class:`~dwave.optimization.model.Model`.
            predecessors:
                Not currently supported.
        Returns:
            A disjoint-list symbol.

        See also:
            :meth:`._into_zipfile`
        """
        if len(predecessors) != 1:
            raise ValueError(f"`DisjointList` should have exactly one predecessor")

        with zf.open(directory + "index.json", "r") as f:
            index = json.load(f)

        return DisjointList(predecessors[0], index)

    def _into_zipfile(self, zf, directory):
        """Store a disjoint-list symbol as a compressed file.

        Args:
            zf:
                File pointer to a compressed file to store the
                disjoint-list symbol. Strings are interpreted as a
                file name.
            directory:
                Directory where the file is located.
        Returns:
            A compressed file.

        See also:
            :meth:`._from_zipfile`
        """
        # the additional data we want to encode
        encoder = json.JSONEncoder(separators=(',', ':'))

        zf.writestr(directory + "index.json", encoder.encode(self.list_index()))

    def list_index(self):
        """Return the index for the list."""
        return self.ptr.list_index()

    # An observing pointer to the C++ DisjointListNode
    cdef cppDisjointListNode* ptr

_register(DisjointList, typeid(cppDisjointListNode))


cdef class Divide(ArraySymbol):
    """Division element-wise between two symbols.

    Examples:
        This example divides two integer symbols.

        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> i = model.integer(10, lower_bound=-50, upper_bound=-1)
        >>> j = model.integer(10, lower_bound=1, upper_bound=10)
        >>> k = i/j
        >>> type(k)
        <class 'dwave.optimization.symbols.Divide'>
    """
    def __init__(self, ArraySymbol lhs, ArraySymbol rhs):
        if lhs.model is not rhs.model:
            raise ValueError("lhs and rhs do not share the same underlying model")

        cdef _Graph model = lhs.model

        self.ptr = model._graph.emplace_node[cppDivideNode](lhs.array_ptr, rhs.array_ptr)
        self.initialize_arraynode(model, self.ptr)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef cppDivideNode* ptr = dynamic_cast_ptr[cppDivideNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")
        cdef Divide x = Divide.__new__(Divide)
        x.ptr = ptr
        x.initialize_arraynode(symbol.model, ptr)
        return x

    cdef cppDivideNode* ptr

_register(Divide, typeid(cppDivideNode))


cdef class Equal(ArraySymbol):
    """Equality comparison element-wise between two symbols.

    Examples:
        This example creates an equality operation between integer symbols.

        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> i = model.integer(25, upper_bound=100)
        >>> j = model.integer(25, lower_bound=-100)
        >>> k = i == j
        >>> type(k)
        <class 'dwave.optimization.symbols.Equal'>
    """
    def __init__(self, ArraySymbol lhs, ArraySymbol rhs):
        if lhs.model is not rhs.model:
            raise ValueError("lhs and rhs do not share the same underlying model")

        cdef _Graph model = lhs.model

        cdef cppEqualNode* ptr = model._graph.emplace_node[cppEqualNode](lhs.array_ptr, rhs.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(Equal, typeid(cppEqualNode))


cdef class Exp(ArraySymbol):
    """Takes the values of a symbol and returns the corresponding base-e exponential.

    See Also:
        :func:`~dwave.optimization.mathematical.exp`: equivalent function.

    .. versionadded:: 0.6.2
    """
    def __init__(self, ArraySymbol x):
        cdef _Graph model = x.model

        cdef cppExpNode* ptr = model._graph.emplace_node[cppExpNode](x.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(Exp, typeid(cppExpNode))


cdef class Expit(ArraySymbol):
    """Takes the values of a symbol and returns the corresponding logistic sigmoid (expit).

    See Also:
        :func:`~dwave.optimization.mathematical.expit`: equivalent function.

    .. versionadded:: 0.5.2
    """
    def __init__(self, ArraySymbol x):
        cdef _Graph model = x.model

        cdef cppExpitNode* ptr = model._graph.emplace_node[cppExpitNode](x.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(Expit, typeid(cppExpitNode))


cdef class Extract(ArraySymbol):
    """Return elements chosen from x or y depending on condition.

    See Also:
        :func:`~dwave.optimization.mathematical.where`: equivalent function.
    """
    def __init__(self, ArraySymbol condition, ArraySymbol arr):
        cdef _Graph model = condition.model

        if condition.model is not arr.model:
            raise ValueError("condition and arr do not share the same underlying model")

        cdef cppExtractNode* ptr = model._graph.emplace_node[cppExtractNode](
            condition.array_ptr, arr.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(Extract, typeid(cppExtractNode))


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


cdef class IntegerVariable(ArraySymbol):
    """Integer decision-variable symbol.

    Examples:
        This example adds an integer symbol to a model.

        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> i = model.integer(25, upper_bound=100)
        >>> type(i)
        <class 'dwave.optimization.symbols.IntegerVariable'>
    """
    def __init__(self, _Graph model, shape=None, lower_bound=None, upper_bound=None):
        cdef vector[Py_ssize_t] vshape = as_cppshape(tuple() if shape is None else shape)

        if lower_bound is None and upper_bound is None:
            self.ptr = model._graph.emplace_node[cppIntegerNode](vshape)
        elif lower_bound is None:
            self.ptr = model._graph.emplace_node[cppIntegerNode](vshape, nullopt, <double>upper_bound)
        elif upper_bound is None:
            self.ptr = model._graph.emplace_node[cppIntegerNode](vshape, <double>lower_bound)
        else:
            self.ptr = model._graph.emplace_node[cppIntegerNode](vshape, <double>lower_bound, <double>upper_bound)

        self.initialize_arraynode(model, self.ptr)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef cppIntegerNode* ptr = dynamic_cast_ptr[cppIntegerNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")

        cdef IntegerVariable x = IntegerVariable.__new__(IntegerVariable)
        x.ptr = ptr
        x.initialize_arraynode(symbol.model, ptr)
        return x

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        if predecessors:
            raise ValueError(f"{cls.__name__} cannot have predecessors")

        with zf.open(directory + "shape.json", "r") as f:
            shape_info = json.load(f)

        return IntegerVariable(model,
                               shape=shape_info["shape"],
                               lower_bound=shape_info["lb"],
                               upper_bound=shape_info["ub"],
                               )

    def _into_zipfile(self, zf, directory):
        # the additional data we want to encode

        shape_info = dict(
            shape=self.shape(),
            lb=self.lower_bound(),
            ub=self.upper_bound(),
            )

        encoder = json.JSONEncoder(separators=(',', ':'))

        zf.writestr(directory + "shape.json", encoder.encode(shape_info))

    def lower_bound(self):
        """The lowest value allowed for the integer symbol."""
        return int(self.ptr.lower_bound())

    def set_state(self, Py_ssize_t index, state):
        """Set the state of the integer node.

        The given state must be integer array of the integer node shape.
        """
        # Convert the state into something we can handle in C++.
        # This also does some type checking etc
        # We go ahead and do the translation to integer now
        cdef Py_ssize_t[:] arr = np.asarray(state, dtype=np.intp).flatten()

        # Reset our state, and check whether that's possible
        self.reset_state(index)

        # Convert to a vector. We could skip this copy if it ever becomes
        # a performance bottleneck
        cdef vector[double] items
        items.reserve(arr.size)
        cdef Py_ssize_t i
        for i in range(arr.size):
            items.push_back(arr[i])

        # The validity of the state is checked in C++
        self.ptr.initialize_state((<States>self.model.states)._states[index], move(items))

    def upper_bound(self):
        """The highest value allowed for the integer symbol."""
        return int(self.ptr.upper_bound())

    # An observing pointer to the C++ IntegerNode
    cdef cppIntegerNode* ptr

_register(IntegerVariable, typeid(cppIntegerNode))


cdef class LessEqual(ArraySymbol):
    """Smaller-or-equal comparison element-wise between two symbols.

    Examples:
        This example creates an inequality operation between integer symbols.

        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> i = model.integer(25, upper_bound=100)
        >>> j = model.integer(25, lower_bound=-100)
        >>> k = i <= j
        >>> type(k)
        <class 'dwave.optimization.symbols.LessEqual'>
    """
    def __init__(self, ArraySymbol lhs, ArraySymbol rhs):
        if lhs.model is not rhs.model:
            raise ValueError("lhs and rhs do not share the same underlying model")

        cdef _Graph model = lhs.model

        cdef cppLessEqualNode* ptr = model._graph.emplace_node[cppLessEqualNode](lhs.array_ptr, rhs.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(LessEqual, typeid(cppLessEqualNode))


cdef class LinearProgram(Symbol):
    """Find a solution to the linear program (LP) defined by the predecessors.

    See Also:
        :func:`~dwave.optimization.mathematical.linprog`

    .. versionadded:: 0.6.0
    """

    def __init__(self, ArraySymbol c,
                 ArraySymbol b_lb = None,
                 ArraySymbol A = None,
                 ArraySymbol b_ub = None,
                 ArraySymbol A_eq = None,
                 ArraySymbol b_eq = None,
                 ArraySymbol lb = None,
                 ArraySymbol ub = None):
        cdef _Graph model = c.model

        cdef cppArrayNode* c_ptr = c.array_ptr

        cdef cppArrayNode* b_lb_ptr = LinearProgram.as_arraynodeptr(model, b_lb)
        cdef cppArrayNode* A_ptr = LinearProgram.as_arraynodeptr(model, A)
        cdef cppArrayNode* b_ub_ptr = LinearProgram.as_arraynodeptr(model, b_ub)

        cdef cppArrayNode* A_eq_ptr = LinearProgram.as_arraynodeptr(model, A_eq)
        cdef cppArrayNode* b_eq_ptr = LinearProgram.as_arraynodeptr(model, b_eq)

        cdef cppArrayNode* lb_ptr = LinearProgram.as_arraynodeptr(model, lb)
        cdef cppArrayNode* ub_ptr = LinearProgram.as_arraynodeptr(model, ub)

        self.ptr = model._graph.emplace_node[cppLinearProgramNode](
            c_ptr, b_lb_ptr, A_ptr, b_ub_ptr, A_eq_ptr, b_eq_ptr, lb_ptr, ub_ptr)
        self.initialize_node(model, self.ptr)

    @staticmethod
    cdef cppArrayNode* as_arraynodeptr(_Graph model, ArraySymbol x) except? NULL:
        # alias for nullptr if x is None else x.array_ptr, but Cython gets confused
        # about that
        # also checks that the model is correct
        if x is None:
            return NULL
        if x.model is not model:
            raise ValueError("all symbols must share the same underlying model")
        return x.array_ptr

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef cppLinearProgramNode* ptr = dynamic_cast_ptr[cppLinearProgramNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")
        cdef LinearProgram x = LinearProgram.__new__(LinearProgram)
        x.ptr = ptr
        x.initialize_node(symbol.model, ptr)
        return x

    def state(self, Py_ssize_t index = 0):
        """Return the current solution to the LP.

        If the LP is not feasible, the solution is not meaningful.
        """

        # While LP is not an ArraySymbol, we nonetheless can access the state

        cdef Py_ssize_t num_states = self.model.states.size()
        if not -num_states <= index < num_states:
            raise ValueError(f"index out of range: {index}")
        elif index < 0:  # allow negative indexing
            index += num_states

        if not self.model.is_locked() and self.node_ptr.topological_index() < 0:
            raise TypeError("the state of an intermediate variable cannot be accessed without "
                            "locking the model first. See model.lock().")

        # Rather than using a StateView, let's just do an explicit copy here

        cdef States states = self.model.states  # for Cython access
        states.resolve()
        self.model._graph.recursive_initialize(states._states.at(index), self.node_ptr)

        solution = self.ptr.solution(states._states.at(index))

        cdef double[::1] state = np.empty(self._num_columns(), dtype=np.double)

        if <Py_ssize_t>solution.size() != state.shape[0]:
            raise RuntimeError  # should never happen, but avoid the segfault just in case

        for i in range(state.shape[0]):
            state[i] = solution[i]

        return np.asarray(state)

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        with zf.open(directory + "arguments.json", "r") as f:
            args = json.load(f)
            return LinearProgram(**{arg: predecessors[index] for arg, index in args.items()})

    def _into_zipfile(self, zf, directory):
        encoder = json.JSONEncoder(separators=(',', ':'))
        zf.writestr(
            directory + "arguments.json",
            encoder.encode({arg.decode(): val for arg, val in self.ptr.get_arguments()})
        )

    cdef Py_ssize_t _num_columns(self) except -1:
        """The number of columns in the LP."""
        if self.ptr.variables_shape().size() != 1:
            raise RuntimeError  # should never happen, but avoid the segfault just in case
        return self.ptr.variables_shape()[0]

    def _set_state(self, Py_ssize_t index, state):
        """Set the output of the LP."""
        if not self.model.is_locked() and self.node_ptr.topological_index() < 0:
            raise TypeError("the state of an intermediate variable cannot be set without "
                            "locking the model first. See model.lock().")

        # Convert the state into something we can handle in C++.
        # This also does some type checking etc
        cdef double[::1] arr = np.ascontiguousarray(state, dtype=np.double)

        # Reset our state, and check whether that's possible
        self.reset_state(index)

        # Also make sure our predecessors all have states
        cdef States states = self.model.states  # for Cython access
        for pred in self.iter_predecessors():
            self.model._graph.recursive_initialize(states._states.at(index), (<Symbol>pred).node_ptr)

        # The validity of the state is checked in C++
        self.ptr.initialize_state(states._states.at(index), as_span(arr))

    def _states_from_zipfile(self, zf, *, num_states, version):
        if version < (1, 0):
            raise ValueError("LinearProgram symbol serialization requires serialization version 1.0 or newer")

        # Test whether we have any states saved.
        try:
            states_info = zf.getinfo(f"nodes/{self.topological_index()}/states.npy")
        except KeyError:
            # No states, so nothing to load
            return

        # If we have states to load, then go ahead and do so.
        with zf.open(states_info, mode="r") as f:
            states = np.load(f, allow_pickle=False)

        for state_index, state in enumerate(states):
            if np.isnan(state).any():  # we saved missing states with nan
                continue
            self._set_state(state_index, state)

    def _states_into_zipfile(self, zf, *, num_states, version):
        if version < (1, 0):
            raise ValueError("LinearProgram symbol serialization requires serialization version 1.0 or newer")

        # check if there is anything to save, if no then just go ahead and return
        if not any(self.has_state(i) for i in range(num_states)):
            return

        # We'll save our states into a dense array. And use NaN to signal when no state
        # is present.
        states = np.empty((num_states, self._num_columns()), dtype=np.double)
        for state_index in range(num_states):
            if self.has_state(state_index):
                # we save the state regardless of whether it is feasible or not
                # In the future we could choose to ignore infeasible states.
                states[state_index, :] = self.state(state_index)
            else:
                states[state_index, :] = np.nan

        # Ok, we have the states, now we just save them into our directory as a NumPy array
        fname = f"nodes/{self.topological_index()}/states.npy"
        with zf.open(fname, mode="w", force_zip64=True) as f:
            np.save(f, states, allow_pickle=False)

    cdef cppLinearProgramNode* ptr

_register(LinearProgram, typeid(cppLinearProgramNode))


cdef class LinearProgramFeasible(ArraySymbol):
    """Return whether the parent LP symbol's current solution is feasible.

    See Also:
        :func:`~dwave.optimization.mathematical.linprog`

    .. versionadded:: 0.6.0
    """

    def __init__(self, Symbol lp):
        cdef _Graph model = lp.model

        cdef cppLinearProgramNodeBase* base_ptr = dynamic_cast_ptr[cppLinearProgramNodeBase](lp.node_ptr)
        if not base_ptr:
            raise TypeError("Provided symbol must be derived from the LP base class")

        cdef cppLinearProgramFeasibleNode* ptr = model._graph.emplace_node[cppLinearProgramFeasibleNode](base_ptr)
        self.initialize_arraynode(model, ptr)

_register(LinearProgramFeasible, typeid(cppLinearProgramFeasibleNode))


cdef class LinearProgramObjectiveValue(ArraySymbol):
    """Return the objective value of the parent LP symbol's current solution.

    See Also:
        :func:`~dwave.optimization.mathematical.linprog`

    .. versionadded:: 0.6.0
    """

    def __init__(self, Symbol lp):
        cdef _Graph model = lp.model

        cdef cppLinearProgramNodeBase* base_ptr = dynamic_cast_ptr[cppLinearProgramNodeBase](lp.node_ptr)
        if not base_ptr:
            raise TypeError("Provided symbol must be derived from the LP base class")

        cdef cppLinearProgramObjectiveValueNode* ptr = model._graph.emplace_node[cppLinearProgramObjectiveValueNode](base_ptr)
        self.initialize_arraynode(model, ptr)

_register(LinearProgramObjectiveValue, typeid(cppLinearProgramObjectiveValueNode))


cdef class LinearProgramSolution(ArraySymbol):
    """Return the current solution of the parent LP symbol as an array.

    See Also:
        :func:`~dwave.optimization.mathematical.linprog`

    .. versionadded:: 0.6.0
    """

    def __init__(self, Symbol lp):
        cdef _Graph model = lp.model

        cdef cppLinearProgramNodeBase* base_ptr = dynamic_cast_ptr[cppLinearProgramNodeBase](lp.node_ptr)
        if not base_ptr:
            raise TypeError("Provided symbol must be derived from the LP base class")

        cdef cppLinearProgramSolutionNode* ptr = model._graph.emplace_node[cppLinearProgramSolutionNode](base_ptr)
        self.initialize_arraynode(model, ptr)

_register(LinearProgramSolution, typeid(cppLinearProgramSolutionNode))


cdef class ListVariable(ArraySymbol):
    """List decision-variable symbol.

    Examples:
        This example adds a list symbol to a model.

        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> l = model.list(10)
        >>> type(l)
        <class 'dwave.optimization.symbols.ListVariable'>
    """
    def __init__(self, _Graph model, Py_ssize_t n):
        # Get an observing pointer to the node
        self.ptr = model._graph.emplace_node[cppListNode](n)

        self.initialize_arraynode(model, self.ptr)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef cppListNode* ptr = dynamic_cast_ptr[cppListNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")

        cdef ListVariable x = ListVariable.__new__(ListVariable)
        x.ptr = ptr
        x.initialize_arraynode(symbol.model, ptr)
        return x

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        if predecessors:
            raise ValueError(f"{cls.__name__} cannot have predecessors")

        with zf.open(directory + "shape.json", "r") as f:
            shape_info = json.load(f)

        return ListVariable(model, n=shape_info["max_value"])

    def _into_zipfile(self, zf, directory):
        # the additional data we want to encode
        cdef cppSizeInfo sizeinfo = self.ptr.sizeinfo()

        shape_info = dict(
            max_value = int(self.ptr.max()) + 1,  # max is inclusive
            min_size = sizeinfo.min.value(),
            max_size = sizeinfo.max.value(),
            )

        encoder = json.JSONEncoder(separators=(',', ':'))

        zf.writestr(directory + "shape.json", encoder.encode(shape_info))

    def set_state(self, Py_ssize_t index, values):
        """Set the state of the list node.

        The given values must be a sub-permuation of ``range(n)`` where ``n`` is
        the size of the list.
        """
        # Convert the values into something we can handle in C++.
        # This also does some type checking etc
        # We go ahead and do the translation to integer now
        cdef Py_ssize_t[:] arr = np.asarray(values, dtype=np.intp)

        # Reset our state, and check whether that's possible
        self.reset_state(index)

        # Convert to a vector. We could skip this copy if it ever becomes
        # a performance bottleneck
        cdef vector[double] items
        items.reserve(arr.size)
        cdef Py_ssize_t i
        for i in range(arr.size):
            items.push_back(arr[i])

        # The validity of the state is checked in C++
        self.ptr.initialize_state((<States>self.model.states)._states[index], move(items))

    # An observing pointer to the C++ ListNode
    cdef cppListNode* ptr

_register(ListVariable, typeid(cppListNode))


cdef class Log(ArraySymbol):
    """Takes the values of a symbol and returns the corresponding natural logarithm (log).

    See Also:
        :func:`~dwave.optimization.mathematical.log`: equivalent function.

    .. versionadded:: 0.5.2
    """
    def __init__(self, ArraySymbol x):
        cdef _Graph model = x.model

        cdef cppLogNode* ptr = model._graph.emplace_node[cppLogNode](x.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(Log, typeid(cppLogNode))


cdef class Logical(ArraySymbol):
    """Logical truth value element-wise on a symbol.

    See Also:
        :func:`~dwave.optimization.mathematical.logical`: equivalent function.
    """
    def __init__(self, ArraySymbol x):
        cdef _Graph model = x.model

        cdef cppLogicalNode* ptr = model._graph.emplace_node[cppLogicalNode](x.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(Logical, typeid(cppLogicalNode))


cdef class Max(ArraySymbol):
    """Maximum value in the elements of a symbol.

    See also:
        :meth:`~dwave.optimization.model.ArraySymbol.max()` equivalent method.
    """
    def __init__(self, ArraySymbol node, *, initial=None):
        cdef _Graph model = node.model

        if initial is None:
            self.ptr = model._graph.emplace_node[cppMaxNode](node.array_ptr)
        else:
            self.ptr = model._graph.emplace_node[cppMaxNode](node.array_ptr, <double?>initial)

        self.initialize_arraynode(model, self.ptr)

    @property
    def initial(self):
        """The initial value to the operation. Returns ``None`` if not provided.

        .. versionadded:: 0.6.4
        """
        return self.ptr.init.value() if self.ptr.init.has_value() else None

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef cppMaxNode* ptr = dynamic_cast_ptr[cppMaxNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")
        cdef Max sym = Max.__new__(Max)
        sym.ptr = ptr
        sym.initialize_arraynode(symbol.model, ptr)
        return sym

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        # Test whether we have any states saved.
        try:
            info = zf.getinfo(directory + "initial.npy")
        except KeyError:
            # No states, so nothing to load
            initial = None
        else:
            with zf.open(info, "r") as f:
                initial = np.load(f)

        return cls(*predecessors, initial=initial)

    def _into_zipfile(self, zf, directory):
        if (init := self.initial) is not None:
            # NumPy serialization is overkill but it's type-safe
            with zf.open(directory + "initial.npy", mode="w", force_zip64=True) as f:
                np.save(f, init, allow_pickle=False)

    def maybe_equals(self, other):
        maybe = super().maybe_equals(other)

        # mismatched initial values can turn uncertainty into a definite no
        if maybe == 1 and self.initial != other.initial:
            return 0

        return maybe

    cdef cppMaxNode* ptr

_register(Max, typeid(cppMaxNode))


cdef class Maximum(ArraySymbol):
    """Maximum values in an element-wise comparison of two symbols.

    Examples:
        This example sets a symbol's values to the maximum values of two
        integer decision variables.

        >>> from dwave.optimization.model import Model
        >>> from dwave.optimization.mathematical import maximum
        ...
        >>> model = Model()
        >>> i = model.integer(100, lower_bound=-50, upper_bound=50)
        >>> j = model.integer(100, lower_bound=-20, upper_bound=150)
        >>> k = maximum(i, j)
        >>> type(k)
        <class 'dwave.optimization.symbols.Maximum'>
    """
    def __init__(self, ArraySymbol lhs, ArraySymbol rhs):
        if lhs.model is not rhs.model:
            raise ValueError("lhs and rhs do not share the same underlying model")

        cdef _Graph model = lhs.model

        cdef cppMaximumNode* ptr = model._graph.emplace_node[cppMaximumNode](lhs.array_ptr, rhs.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(Maximum, typeid(cppMaximumNode))


cdef class Mean(ArraySymbol):
    """Mean value of the elements of a symbol. If symbol is empty, 
        mean defaults to 0.0.

    Examples:
        This example takes the mean of one symbol.

        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import mean
        ...
        >>> model = Model()
        >>> i = model.integer(4)
        >>> m = mean(i)
        >>> type(m)
        <class 'dwave.optimization.symbols.Mean'>
    
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


cdef class Min(ArraySymbol):
    """Minimum value in the elements of a symbol.

    See also:
        :meth:`~dwave.optimization.model.ArraySymbol.min()` equivalent method.
    """
    def __init__(self, ArraySymbol node, *, initial=None):
        cdef _Graph model = node.model

        if initial is None:
            self.ptr = model._graph.emplace_node[cppMinNode](node.array_ptr)
        else:
            self.ptr = model._graph.emplace_node[cppMinNode](node.array_ptr, <double?>initial)

        self.initialize_arraynode(model, self.ptr)

    @property
    def initial(self):
        """The initial value to the operation. Returns ``None`` if not provided.

        .. versionadded:: 0.6.4
        """
        return self.ptr.init.value() if self.ptr.init.has_value() else None

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef cppMinNode* ptr = dynamic_cast_ptr[cppMinNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")
        cdef Min sym = Min.__new__(Min)
        sym.ptr = ptr
        sym.initialize_arraynode(symbol.model, ptr)
        return sym

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        # Test whether we have any states saved.
        try:
            info = zf.getinfo(directory + "initial.npy")
        except KeyError:
            # Nothing to load
            initial = None
        else:
            with zf.open(info, "r") as f:
                initial = np.load(f)

        return cls(*predecessors, initial=initial)

    def _into_zipfile(self, zf, directory):
        if (init := self.initial) is not None:
            # NumPy serialization is overkill but it's type-safe
            with zf.open(directory + "initial.npy", mode="w", force_zip64=True) as f:
                np.save(f, init, allow_pickle=False)

    def maybe_equals(self, other):
        maybe = super().maybe_equals(other)

        # mismatched initial values can turn uncertainty into a definite no
        if maybe == 1 and self.initial != other.initial:
            return 0

        return maybe

    cdef cppMinNode* ptr

_register(Min, typeid(cppMinNode))


cdef class Minimum(ArraySymbol):
    """Minimum values in an element-wise comparison of two symbols.

    Examples:
        This example sets a symbol's values to the minimum values of two
        integer decision variables.

        >>> from dwave.optimization.model import Model
        >>> from dwave.optimization.mathematical import minimum
        ...
        >>> model = Model()
        >>> i = model.integer(100, lower_bound=-50, upper_bound=50)
        >>> j = model.integer(100, lower_bound=-20, upper_bound=150)
        >>> k = minimum(i, j)
        >>> type(k)
        <class 'dwave.optimization.symbols.Minimum'>
    """
    def __init__(self, ArraySymbol lhs, ArraySymbol rhs):
        if lhs.model is not rhs.model:
            raise ValueError("lhs and rhs do not share the same underlying model")

        cdef _Graph model = lhs.model

        cdef cppMinimumNode* ptr = model._graph.emplace_node[cppMinimumNode](lhs.array_ptr, rhs.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(Minimum, typeid(cppMinimumNode))


cdef class Modulus(ArraySymbol):
    """Modulus element-wise between two symbols.

    Examples:
        This example calculates the modulus of two integer symbols.

        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> i = model.integer(10, lower_bound=-50, upper_bound=50)
        >>> j = model.integer(10, lower_bound=-20, upper_bound=150)
        >>> k = i % j
        >>> type(k)
        <class 'dwave.optimization.symbols.Modulus'>
    """
    def __init__(self, ArraySymbol lhs, ArraySymbol rhs):
        if lhs.model is not rhs.model:
            raise ValueError("lhs and rhs do not share the same underlying model")

        cdef _Graph model = lhs.model

        cdef cppModulusNode* ptr = model._graph.emplace_node[cppModulusNode](lhs.array_ptr, rhs.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(Modulus, typeid(cppModulusNode))


cdef class Multiply(ArraySymbol):
    """Multiplication element-wise between two symbols.

    Examples:
        This example multiplies two integer symbols.

        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> i = model.integer(10, lower_bound=-50, upper_bound=50)
        >>> j = model.integer(10, lower_bound=0, upper_bound=10)
        >>> k = i*j
        >>> type(k)
        <class 'dwave.optimization.symbols.Multiply'>
    """
    def __init__(self, ArraySymbol lhs, ArraySymbol rhs):
        if lhs.model is not rhs.model:
            raise ValueError("lhs and rhs do not share the same underlying model")

        cdef _Graph model = lhs.model

        cdef cppMultiplyNode* ptr = model._graph.emplace_node[cppMultiplyNode](lhs.array_ptr, rhs.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(Multiply, typeid(cppMultiplyNode))


cdef class NaryAdd(ArraySymbol):
    """Addition element-wise of `N` symbols.

    Examples:
        This example add three integer symbols.

        >>> from dwave.optimization.model import Model
        >>> from dwave.optimization.mathematical import add
        ...
        >>> model = Model()
        >>> i = model.integer((10, 10), lower_bound=-50, upper_bound=50)
        >>> j = model.integer((10, 10), lower_bound=-20, upper_bound=150)
        >>> k = model.integer((10, 10), lower_bound=0, upper_bound=100)
        >>> l = add(i, j, k)
        >>> type(l)
        <class 'dwave.optimization.symbols.NaryAdd'>
    """
    def __init__(self, *inputs):
        if len(inputs) == 0:
            raise TypeError("must have at least one predecessor node")

        cdef _Graph model = inputs[0].model
        cdef vector[cppArrayNode*] cppinputs

        cdef ArraySymbol array
        for node in inputs:
            if node.model != model:
                raise ValueError("all predecessors must be from the same model")
            array = <ArraySymbol?>node
            cppinputs.push_back(array.array_ptr)

        self.ptr = model._graph.emplace_node[cppNaryAddNode](cppinputs)
        self.initialize_arraynode(model, self.ptr)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef cppNaryAddNode* ptr = dynamic_cast_ptr[cppNaryAddNode](symbol.node_ptr)
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

    cdef cppNaryAddNode* ptr

_register(NaryAdd, typeid(cppNaryAddNode))


cdef class NaryMaximum(ArraySymbol):
    """Maximum values in an element-wise comparison of `N` symbols.

    Examples:
        This example sets a symbol's values to the maximum values of
        three integer decision variables.

        >>> from dwave.optimization.model import Model
        >>> from dwave.optimization.mathematical import maximum
        ...
        >>> model = Model()
        >>> i = model.integer((10, 10), lower_bound=-50, upper_bound=50)
        >>> j = model.integer((10, 10), lower_bound=-20, upper_bound=150)
        >>> k = model.integer((10, 10), lower_bound=0, upper_bound=100)
        >>> l = maximum(i, j, k)
        >>> type(l)
        <class 'dwave.optimization.symbols.NaryMaximum'>
    """
    def __init__(self, *inputs):
        if len(inputs) == 0:
            raise TypeError("must have at least one predecessor node")

        cdef _Graph model = inputs[0].model
        cdef vector[cppArrayNode*] cppinputs

        cdef ArraySymbol array
        for node in inputs:
            if node.model != model:
                raise ValueError("all predecessors must be from the same model")
            array = <ArraySymbol?>node
            cppinputs.push_back(array.array_ptr)

        cdef cppNaryMaximumNode* ptr = model._graph.emplace_node[cppNaryMaximumNode](cppinputs)
        self.initialize_arraynode(model, ptr)

_register(NaryMaximum, typeid(cppNaryMaximumNode))


cdef class NaryMinimum(ArraySymbol):
    """Minimum values in an element-wise comparison of `N` symbols.

    Examples:
        This example sets a symbol's values to the minimum values of
        three integer decision variables.

        >>> from dwave.optimization.model import Model
        >>> from dwave.optimization.mathematical import minimum
        ...
        >>> model = Model()
        >>> i = model.integer((10, 10), lower_bound=-50, upper_bound=50)
        >>> j = model.integer((10, 10), lower_bound=-20, upper_bound=150)
        >>> k = model.integer((10, 10), lower_bound=0, upper_bound=100)
        >>> l = minimum(i, j, k)
        >>> type(l)
        <class 'dwave.optimization.symbols.NaryMinimum'>
    """
    def __init__(self, *inputs):
        if len(inputs) == 0:
            raise TypeError("must have at least one predecessor node")

        cdef _Graph model = inputs[0].model
        cdef vector[cppArrayNode*] cppinputs

        cdef ArraySymbol array
        for node in inputs:
            if node.model != model:
                raise ValueError("all predecessors must be from the same model")
            array = <ArraySymbol?>node
            cppinputs.push_back(array.array_ptr)

        cdef cppNaryMinimumNode* ptr = model._graph.emplace_node[cppNaryMinimumNode](cppinputs)
        self.initialize_arraynode(model, ptr)

_register(NaryMinimum, typeid(cppNaryMinimumNode))


cdef class NaryMultiply(ArraySymbol):
    """Multiplication element-wise between `N` symbols.

    Examples:
        This example multiplies three integer decision variables.

        >>> from dwave.optimization.model import Model
        >>> from dwave.optimization.mathematical import multiply
        ...
        >>> model = Model()
        >>> i = model.integer((10, 10), lower_bound=-50, upper_bound=50)
        >>> j = model.integer((10, 10), lower_bound=-20, upper_bound=150)
        >>> k = model.integer((10, 10), lower_bound=0, upper_bound=100)
        >>> l = multiply(i, j, k)
        >>> type(l)
        <class 'dwave.optimization.symbols.NaryMultiply'>
    """
    def __init__(self, *inputs):
        if len(inputs) == 0:
            raise TypeError("must have at least one predecessor node")

        cdef _Graph model = inputs[0].model
        cdef vector[cppArrayNode*] cppinputs

        cdef ArraySymbol array
        for node in inputs:
            if node.model != model:
                raise ValueError("all predecessors must be from the same model")
            array = <ArraySymbol?>node
            cppinputs.push_back(array.array_ptr)

        self.ptr = model._graph.emplace_node[cppNaryMultiplyNode](cppinputs)
        self.initialize_arraynode(model, self.ptr)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef cppNaryMultiplyNode* ptr = dynamic_cast_ptr[cppNaryMultiplyNode](symbol.node_ptr)
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

    cdef cppNaryMultiplyNode* ptr

_register(NaryMultiply, typeid(cppNaryMultiplyNode))


cdef class Negative(ArraySymbol):
    """Numerical negative element-wise on a symbol.

    Examples:
        This example add the negative of an integer array.

        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> i = model.integer(100, upper_bound=50)
        >>> i_minus = -i
        >>> type(i_minus)
        <class 'dwave.optimization.symbols.Negative'>
    """
    def __init__(self, ArraySymbol x):
        cdef _Graph model = x.model

        cdef cppNegativeNode* ptr = model._graph.emplace_node[cppNegativeNode](x.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(Negative, typeid(cppNegativeNode))


cdef class Not(ArraySymbol):
    """Logical negation element-wise on a symbol.

    See Also:
        :func:`~dwave.optimization.mathematical.logical_not`: equivalent function.
    """
    def __init__(self, ArraySymbol x):
        cdef _Graph model = x.model

        cdef cppNotNode* ptr = model._graph.emplace_node[cppNotNode](x.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(Not, typeid(cppNotNode))


cdef class Or(ArraySymbol):
    """Boolean OR element-wise between two symbols.

    See Also:
        :func:`~dwave.optimization.mathematical.logical_or`: equivalent function.
    """
    def __init__(self, ArraySymbol lhs, ArraySymbol rhs):
        if lhs.model is not rhs.model:
            raise ValueError("lhs and rhs do not share the same underlying model")

        cdef _Graph model = lhs.model

        cdef cppOrNode* ptr = model._graph.emplace_node[cppOrNode](lhs.array_ptr, rhs.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(Or, typeid(cppOrNode))


cdef class PartialProd(ArraySymbol):
    """Multiply of the elements of a symbol along an axis.

    See also:
        :meth:`~dwave.optimization.model.ArraySymbol.prod()` equivalent method.

    .. versionadded:: 0.5.1
    """
    def __init__(self, ArraySymbol array, int axis, *, initial=None):
        cdef _Graph model = array.model

        if not isinstance(axis, numbers.Integral):
            raise TypeError("axis should be an int")

        if not (0 <= axis < array.ndim()):
            raise ValueError("axis should be 0 <= axis < ndim()")

        if initial is None:
            self.ptr = model._graph.emplace_node[cppPartialProdNode](
                array.array_ptr,
                <int?>axis,
            )
        else:
            self.ptr = model._graph.emplace_node[cppPartialProdNode](
                array.array_ptr,
                <int?>axis,
                <double?>initial,
            )
        self.initialize_arraynode(model, self.ptr)

    def axes(self):
        axes = self.ptr.axes()
        return tuple(axes[i] for i in range(axes.size()))

    @property
    def initial(self):
        """The initial value to the operation. Returns ``None`` if not provided.

        .. versionadded:: 0.6.4
        """
        return self.ptr.init.value() if self.ptr.init.has_value() else None

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef cppPartialProdNode* ptr = dynamic_cast_ptr[cppPartialProdNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")
        cdef PartialProd ps = PartialProd.__new__(PartialProd)
        ps.ptr = ptr
        ps.initialize_arraynode(symbol.model, ptr)
        return ps

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        if len(predecessors) != 1:
            raise ValueError("PartialProd must have exactly one predecessor")

        with zf.open(directory + "axes.json", "r") as f:
            axis = json.load(f)[0]

        try:
            info = zf.getinfo(directory + "initial.npy")
        except KeyError:
            # Nothing to load
            initial = None
        else:
            with zf.open(info, "r") as f:
                initial = np.load(f)

        return cls(*predecessors, axis=axis, initial=initial)

    def _into_zipfile(self, zf, directory):
        encoder = json.JSONEncoder(separators=(',', ':'))

        # Save information about the axes (always present)
        zf.writestr(directory + "axes.json", encoder.encode(self.axes()))

        # If we have an initial value, save that too
        if (init := self.initial) is not None:
            # NumPy serialization is overkill but it's type-safe
            with zf.open(directory + "initial.npy", mode="w", force_zip64=True) as f:
                np.save(f, init, allow_pickle=False)

    def maybe_equals(self, other):
        maybe = super().maybe_equals(other)

        # mismatched initial values can turn uncertainty into a definite no
        if maybe == 1 and self.initial != other.initial:
            return 0

        return maybe

    cdef cppPartialProdNode* ptr

_register(PartialProd, typeid(cppPartialProdNode))


cdef class PartialSum(ArraySymbol):
    """Sum of the elements of a symbol along an axis.

    See also:
        :meth:`~dwave.optimization.model.ArraySymbol.sum()` equivalent method.

    .. versionadded:: 0.4.1
    """
    def __init__(self, ArraySymbol array, int axis, *, initial=None):
        cdef _Graph model = array.model

        if not isinstance(axis, numbers.Integral):
            raise TypeError("axis should be an int")

        if not (0 <= axis < array.ndim()):
            raise ValueError("axis should be 0 <= axis < ndim()")

        if initial is None:
            self.ptr = model._graph.emplace_node[cppPartialSumNode](
                array.array_ptr,
                <int?>axis,
            )
        else:
            self.ptr = model._graph.emplace_node[cppPartialSumNode](
                array.array_ptr,
                <int?>axis,
                <double?>initial,
            )
        self.initialize_arraynode(model, self.ptr)

    def axes(self):
        axes = self.ptr.axes()
        return tuple(axes[i] for i in range(axes.size()))

    @property
    def initial(self):
        """The initial value to the operation. Returns ``None`` if not provided.

        .. versionadded:: 0.6.4
        """
        return self.ptr.init.value() if self.ptr.init.has_value() else None

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef cppPartialSumNode* ptr = dynamic_cast_ptr[cppPartialSumNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")
        cdef PartialSum ps = PartialSum.__new__(PartialSum)
        ps.ptr = ptr
        ps.initialize_arraynode(symbol.model, ptr)
        return ps

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        if len(predecessors) != 1:
            raise ValueError("PartialSum must have exactly one predecessor")

        with zf.open(directory + "axes.json", "r") as f:
            axis = json.load(f)[0]

        try:
            info = zf.getinfo(directory + "initial.npy")
        except KeyError:
            # Nothing to load
            initial = None
        else:
            with zf.open(info, "r") as f:
                initial = np.load(f)

        return cls(*predecessors, axis=axis, initial=initial)

    def _into_zipfile(self, zf, directory):
        encoder = json.JSONEncoder(separators=(',', ':'))

        # Save information about the axes (always present)
        zf.writestr(directory + "axes.json", encoder.encode(self.axes()))

        # If we have an initial value, save that too
        if (init := self.initial) is not None:
            # NumPy serialization is overkill but it's type-safe
            with zf.open(directory + "initial.npy", mode="w", force_zip64=True) as f:
                np.save(f, init, allow_pickle=False)

    def maybe_equals(self, other):
        maybe = super().maybe_equals(other)

        # mismatched initial values can turn uncertainty into a definite no
        if maybe == 1 and self.initial != other.initial:
            return 0

        return maybe

    cdef cppPartialSumNode* ptr

_register(PartialSum, typeid(cppPartialSumNode))


cdef class Permutation(ArraySymbol):
    """Permutation of the elements of a symbol.

    Examples:
        This example creates a permutation of a constant symbol.

        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> C = model.constant([[1, 2, 3], [2, 3, 1], [0, 1, 0]])
        >>> l = model.list(3)
        >>> p = C[l, :][:, l]
        >>> type(p)
        <class 'dwave.optimization.symbols.Permutation'>
    """
    def __init__(self, Constant array, ListVariable x):
        # todo: Loosen the types accepted. But this Cython code doesn't yet have
        # the type heirarchy needed so for how we specify explicitly

        if array.model is not x.model:
            raise ValueError("array and x do not share the same underlying model")

        cdef cppPermutationNode* ptr = array.model._graph.emplace_node[cppPermutationNode](
            array.array_ptr, x.array_ptr)
        self.initialize_arraynode(array.model, ptr)

_register(Permutation, typeid(cppPermutationNode))


cdef class Prod(ArraySymbol):
    """Product of the elements of a symbol.

    See also:
        :meth:`~dwave.optimization.model.ArraySymbol.prod()` equivalent method.
    """
    def __init__(self, ArraySymbol node, *, initial=None):
        cdef _Graph model = node.model

        if initial is None:
            self.ptr = model._graph.emplace_node[cppProdNode](node.array_ptr)
        else:
            self.ptr = model._graph.emplace_node[cppProdNode](node.array_ptr, <double?>initial)

        self.initialize_arraynode(model, self.ptr)

    @property
    def initial(self):
        """The initial value to the operation. Returns ``None`` if not provided.

        .. versionadded:: 0.6.4
        """
        return self.ptr.init.value() if self.ptr.init.has_value() else None

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef cppProdNode* ptr = dynamic_cast_ptr[cppProdNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")
        cdef Prod sym = Prod.__new__(Prod)
        sym.ptr = ptr
        sym.initialize_arraynode(symbol.model, ptr)
        return sym

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        # Test whether we have any states saved.
        try:
            info = zf.getinfo(directory + "initial.npy")
        except KeyError:
            # Nothing to load
            initial = None
        else:
            with zf.open(info, "r") as f:
                initial = np.load(f)

        return cls(*predecessors, initial=initial)

    def _into_zipfile(self, zf, directory):
        if (init := self.initial) is not None:
            # NumPy serialization is overkill but it's type-safe
            with zf.open(directory + "initial.npy", mode="w", force_zip64=True) as f:
                np.save(f, init, allow_pickle=False)

    def maybe_equals(self, other):
        maybe = super().maybe_equals(other)

        # mismatched initial values can turn uncertainty into a definite no
        if maybe == 1 and self.initial != other.initial:
            return 0

        return maybe

    cdef cppProdNode* ptr

_register(Prod, typeid(cppProdNode))


cdef class Put(ArraySymbol):
    """A symbol that replaces the specified elements in an array with given values.

    See Also:
        :func:`~dwave.optimization.mathematical.put`: equivalent function.

    .. versionadded:: 0.4.4
    """
    def __init__(self, ArraySymbol array, ArraySymbol indices, ArraySymbol values):
        cdef _Graph model = array.model

        if indices.model is not model or values.model is not model:
            raise ValueError(
                "array, indices, and values do not all share the same underlying model"
            )

        cdef cppPutNode* ptr = model._graph.emplace_node[cppPutNode](
            array.array_ptr, indices.array_ptr, values.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(Put, typeid(cppPutNode))


cdef class QuadraticModel(ArraySymbol):
    """Quadratic model.

    Examples:
        This example adds a quadratic model.

        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> x = model.binary(3)
        >>> Q = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 1): 1, (1, 2): 3, (2, 2): 2}
        >>> qm = model.quadratic_model(x, Q)
        >>> type(qm)
        <class 'dwave.optimization.symbols.QuadraticModel'>
    """
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


cdef class Reshape(ArraySymbol):
    """Reshaped symbol.

    See Also:
        :meth:`ArraySymbol.reshape() <dwave.optimization.model.ArraySymbol.reshape>`: equivalent method.

    .. versionadded:: 0.5.1
    """
    def __init__(self, ArraySymbol node, shape):
        cdef _Graph model = node.model

        self.ptr = model._graph.emplace_node[cppReshapeNode](
            node.array_ptr,
            as_cppshape(shape, nonnegative=False),
            )

        self.initialize_arraynode(model, self.ptr)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef cppReshapeNode* ptr = dynamic_cast_ptr[cppReshapeNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")

        cdef Reshape m = Reshape.__new__(Reshape)
        m.ptr = ptr
        m.initialize_arraynode(symbol.model, ptr)
        return m

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        if len(predecessors) != 1:
            raise ValueError("Reshape must have exactly one predecessor")

        with zf.open(directory + "shape.json", "r") as f:
            return Reshape(*predecessors, json.load(f))

    def _into_zipfile(self, zf, directory):
        encoder = json.JSONEncoder(separators=(',', ':'))
        zf.writestr(directory + "shape.json", encoder.encode(self.shape()))

    cdef cppReshapeNode* ptr

_register(Reshape, typeid(cppReshapeNode))


cdef class Resize(ArraySymbol):
    """Resize symbol.

    See also:
        :func:`~dwave.optimization.mathematical.resize`: equivalent function.

        :meth:`ArraySymbol.resize() <dwave.optimization.model.ArraySymbol.resize>`: equivalent method.

    .. versionadded:: 0.6.4
    """
    def __init__(self, ArraySymbol symbol, shape, fill_value=None):
        cdef _Graph model = symbol.model

        if fill_value is None:
            self.ptr = model._graph.emplace_node[cppResizeNode](
                symbol.array_ptr,
                as_cppshape(shape, nonnegative=True),
            )
        else:
            self.ptr = model._graph.emplace_node[cppResizeNode](
                symbol.array_ptr,
                as_cppshape(shape, nonnegative=True),
                <double>fill_value,
            )

        self.initialize_arraynode(model, self.ptr)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef cppResizeNode* ptr = dynamic_cast_ptr[cppResizeNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")

        cdef Resize m = Resize.__new__(Resize)
        m.ptr = ptr
        m.initialize_arraynode(symbol.model, ptr)
        return m

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        if len(predecessors) != 1:
            raise ValueError("Resize must have exactly one predecessor")

        with zf.open(directory + "shape.json", "r") as f:
            shape = json.load(f)
        with zf.open(directory + "fill_value.npy", "r") as f:
            fill_value = np.load(f)

        return Resize(*predecessors, shape, fill_value)

    def _into_zipfile(self, zf, directory):
        encoder = json.JSONEncoder(separators=(',', ':'))
        zf.writestr(directory + "shape.json", encoder.encode(self.shape()))
        with zf.open(directory + "fill_value.npy", mode="w", force_zip64=True) as f:
            np.save(f, self.ptr.fill_value(), allow_pickle=False)

    cdef cppResizeNode* ptr

_register(Resize, typeid(cppResizeNode))


cdef class SetVariable(ArraySymbol):
    """Set decision-variable symbol.

    A set variable's possible states are the subsets of ``range(n)``.

    Args:
        model: The model.
        n: The possible states of the set variable are the subsets of ``range(n)``.
        min_size: The minimum set size.
        max_size: The maximum set size.

    Examples:
        This example adds a set symbol to a model.

        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> s = model.set(10)
        >>> type(s)
        <class 'dwave.optimization.symbols.SetVariable'>
    """
    def __init__(self, _Graph model, Py_ssize_t n, Py_ssize_t min_size, Py_ssize_t max_size):
        self.ptr = model._graph.emplace_node[cppSetNode](n, min_size, max_size)
        self.initialize_arraynode(model, self.ptr)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef cppSetNode* ptr = dynamic_cast_ptr[cppSetNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")

        cdef SetVariable x = SetVariable.__new__(SetVariable)
        x.ptr = ptr
        x.initialize_arraynode(symbol.model, ptr)
        return x

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        if predecessors:
            raise ValueError(f"{cls.__name__} cannot have predecessors")

        with zf.open(directory + "shape.json", "r") as f:
            shape_info = json.load(f)

        return SetVariable(model,
                           n=shape_info.get("max_value"),
                           min_size=shape_info.get("min_size"),
                           max_size=shape_info.get("max_size"),
                           )

    def _into_zipfile(self, zf, directory):
        # the additional data we want to encode

        cdef cppSizeInfo sizeinfo = self.ptr.sizeinfo()

        shape_info = dict(
            max_value = int(self.ptr.max()) + 1,  # max is inclusive
            min_size = sizeinfo.min.value(),
            max_size = sizeinfo.max.value(),
            )

        encoder = json.JSONEncoder(separators=(',', ':'))

        zf.writestr(directory + "shape.json", encoder.encode(shape_info))

    def set_state(self, Py_ssize_t index, values):
        """Set the state of the set node.

        The given state must be a subset of ``range(n)`` where ``n`` is the size
        of the set.
        """
        if isinstance(values, collections.abc.Set):
            values = sorted(values)

        # Convert the state into something we can handle in C++.
        # This also does some type checking etc
        # We go ahead and do the translation to integer now
        cdef Py_ssize_t[:] arr = np.asarray(values, dtype=np.intp)

        # Reset our state, and check whether that's possible
        self.reset_state(index)

        # Convert to a vector. We could skip this copy if it ever becomes
        # a performance bottleneck
        cdef vector[double] items
        items.reserve(arr.size)
        cdef Py_ssize_t i
        for i in range(arr.size):
            items.push_back(arr[i])

        # The validity of the state is checked in C++
        self.ptr.initialize_state((<States>self.model.states)._states[index], move(items))

    # Observing pointer to the node
    cdef cppSetNode* ptr

_register(SetVariable, typeid(cppSetNode))


cdef class Size(ArraySymbol):
    def __init__(self, ArraySymbol array):
        cdef _Graph model = array.model

        cdef cppSizeNode* ptr = model._graph.emplace_node[cppSizeNode](array.array_ptr)
        self.initialize_arraynode(array.model, ptr)

_register(Size, typeid(cppSizeNode))


cdef class Rint(ArraySymbol):
    """Takes the values of a symbol and rounds them to the nearest integer.

    Examples:
        This example adds the round-int of a decision
        variable to a model.

        >>> from dwave.optimization.model import Model
        >>> from dwave.optimization.mathematical import rint
        >>> model = Model()
        >>> i = model.constant(10.4)
        >>> ii = rint(i)
        >>> type(ii)
        <class 'dwave.optimization.symbols.Rint'>
    """
    def __init__(self, ArraySymbol x):
        cdef _Graph model = x.model

        cdef cppRintNode* ptr = model._graph.emplace_node[cppRintNode](x.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(Rint, typeid(cppRintNode))


cdef class SafeDivide(ArraySymbol):
    """Safe division element-wise between two symbols.

    See also:
        :func:`~dwave.optimization.mathematical.safe_divide`: equivalent function.

    .. versionadded:: 0.6.2
    """
    def __init__(self, ArraySymbol lhs, ArraySymbol rhs):
        if lhs.model is not rhs.model:
            raise ValueError("lhs and rhs do not share the same underlying model")

        cdef _Graph model = lhs.model

        cdef cppSafeDivideNode* ptr = model._graph.emplace_node[cppSafeDivideNode](
            lhs.array_ptr, rhs.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(SafeDivide, typeid(cppSafeDivideNode))


cdef class Sin(ArraySymbol):
    """Sine element-wise on a symbol.

    See Also:
        :func:`~dwave.optimization.mathematical.sin`: equivalent function.

    .. versionadded:: 0.6.5
    """
    def __init__(self, ArraySymbol x):
        cdef _Graph model = x.model

        cdef cppSinNode* ptr = model._graph.emplace_node[cppSinNode](x.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(Sin, typeid(cppSinNode))


cdef class SoftMax(ArraySymbol):
    """Softmax of a symbol.

    Example:
        These examples compute the softmax of one symbol.

        >>> from dwave.optimization.model import Model
        >>> from dwave.optimization.symbols import SoftMax
        >>> import numpy as np
        ...
        >>> model = Model()
        >>> model.states.resize(1)
        >>> c = model.constant([1, 2, 3])
        >>> sm = SoftMax(c)
        >>> expected = np.array([0.09003057, 0.24472847, 0.66524096])
        >>> with model.lock():
        ...     print(np.isclose(sm.state(0), expected).all())
        True

        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import softmax
        ...
        >>> model = Model()
        >>> i = model.integer(3)
        >>> sm = softmax(i)
        >>> type(sm)
        <class 'dwave.optimization.symbols.SoftMax'>

    See Also:
        :meth:`~dwave.optimization.mathematical.softmax`: equivalent method.

    .. versionadded:: 0.6.5
    """
    def __init__(self, ArraySymbol arr):
        cdef _Graph model = arr.model

        cdef cppSoftMaxNode* ptr = model._graph.emplace_node[cppSoftMaxNode](arr.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(SoftMax, typeid(cppSoftMaxNode))


cdef class Square(ArraySymbol):
    """Squares element-wise of a symbol.

    Examples:
        This example adds the squares of an integer decision
        variable to a model.

        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> i = model.integer(10, lower_bound=-5, upper_bound=5)
        >>> ii = i**2
        >>> type(ii)
        <class 'dwave.optimization.symbols.Square'>
    """
    def __init__(self, ArraySymbol x):
        cdef _Graph model = x.model

        cdef cppSquareNode* ptr = model._graph.emplace_node[cppSquareNode](x.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(Square, typeid(cppSquareNode))


cdef class SquareRoot(ArraySymbol):
    """Square root of a symbol.

    Examples:
        This example adds the square root of an integer decision variable to a
        model.

        >>> from dwave.optimization.model import Model
        >>> from dwave.optimization.mathematical import sqrt
        >>> model = Model()
        >>> i = model.constant(10)
        >>> ii = sqrt(i)
        >>> type(ii)
        <class 'dwave.optimization.symbols.SquareRoot'>
    """
    def __init__(self, ArraySymbol x):
        cdef _Graph model = x.model

        cdef cppSquareRootNode* ptr = model._graph.emplace_node[cppSquareRootNode](x.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(SquareRoot, typeid(cppSquareRootNode))


cdef class Subtract(ArraySymbol):
    """Subtraction element-wise of two symbols.

    Examples:
        This example subtracts two integer symbols.

        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> i = model.integer(10, lower_bound=-50, upper_bound=50)
        >>> j = model.integer(10, lower_bound=0, upper_bound=10)
        >>> k = i - j
        >>> type(k)
        <class 'dwave.optimization.symbols.Subtract'>
    """
    def __init__(self, ArraySymbol lhs, ArraySymbol rhs):
        if lhs.model is not rhs.model:
            raise ValueError("lhs and rhs do not share the same underlying model")

        cdef _Graph model = lhs.model

        cdef cppSubtractNode* ptr = model._graph.emplace_node[cppSubtractNode](lhs.array_ptr, rhs.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(Subtract, typeid(cppSubtractNode))


cdef class Sum(ArraySymbol):
    """Sum of the elements of a symbol.

    See Also:
        :meth:`~dwave.optimization.model.ArraySymbol.sum()` equivalent method.
    """
    def __init__(self, ArraySymbol node, *, initial=None):
        cdef _Graph model = node.model

        if initial is None:
            self.ptr = model._graph.emplace_node[cppSumNode](node.array_ptr)
        else:
            self.ptr = model._graph.emplace_node[cppSumNode](node.array_ptr, <double?>initial)

        self.initialize_arraynode(model, self.ptr)

    @property
    def initial(self):
        """The initial value to the operation. Returns ``None`` if not provided.

        .. versionadded:: 0.6.4
        """
        return self.ptr.init.value() if self.ptr.init.has_value() else None

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef cppSumNode* ptr = dynamic_cast_ptr[cppSumNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")
        cdef Sum sym = Sum.__new__(Sum)
        sym.ptr = ptr
        sym.initialize_arraynode(symbol.model, ptr)
        return sym

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        # Test whether we have any states saved.
        try:
            info = zf.getinfo(directory + "initial.npy")
        except KeyError:
            # No states, so nothing to load
            initial = None
        else:
            with zf.open(info, "r") as f:
                initial = np.load(f)

        return cls(*predecessors, initial=initial)

    def _into_zipfile(self, zf, directory):
        if (init := self.initial) is not None:
            # NumPy serialization is overkill but it's type-safe
            with zf.open(directory + "initial.npy", mode="w", force_zip64=True) as f:
                np.save(f, init, allow_pickle=False)

    def maybe_equals(self, other):
        maybe = super().maybe_equals(other)

        # mismatched initial values can turn uncertainty into a definite no
        if maybe == 1 and self.initial != other.initial:
            return 0

        return maybe

    cdef cppSumNode* ptr

_register(Sum, typeid(cppSumNode))


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

        cdef cppWhereNode* ptr = model._graph.emplace_node[cppWhereNode](
            condition.array_ptr, x.array_ptr, y.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(Where, typeid(cppWhereNode))


cdef class Xor(ArraySymbol):
    """Boolean XOR element-wise between two symbols.

    See Also:
        :func:`~dwave.optimization.mathematical.logical_xor`: equivalent function.

        .. versionadded:: 0.4.1
    """
    def __init__(self, ArraySymbol lhs, ArraySymbol rhs):
        if lhs.model is not rhs.model:
            raise ValueError("lhs and rhs do not share the same underlying model")

        cdef _Graph model = lhs.model

        cdef cppXorNode* ptr = model._graph.emplace_node[cppXorNode](lhs.array_ptr, rhs.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(Xor, typeid(cppXorNode))

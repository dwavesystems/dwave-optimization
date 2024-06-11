# distutils: language = c++

# Copyright 2024 D-Wave Systems Inc.
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

import cython
import numpy as np

from libcpp cimport bool
from libcpp.cast cimport dynamic_cast
from libcpp.optional cimport nullopt, optional
from libcpp.utility cimport move
from libcpp.vector cimport vector

from dwave.optimization.libcpp cimport get, holds_alternative, span
from dwave.optimization.libcpp.array cimport (
    Array as cppArray,
    ArrayPtr as cppArrayPtr,
    SizeInfo as cppSizeInfo,
    Slice as cppSlice,
    )
from dwave.optimization.libcpp.graph cimport Node as cppNode
from dwave.optimization.libcpp.nodes cimport (
    AbsoluteNode as cppAbsoluteNode,
    AddNode as cppAddNode,
    AllNode as cppAllNode,
    AndNode as cppAndNode,
    AdvancedIndexingNode as cppAdvancedIndexingNode,
    ArrayValidationNode as cppArrayValidationNode,
    BasicIndexingNode as cppBasicIndexingNode,
    BinaryNode as cppBinaryNode,
    ConstantNode as cppConstantNode,
    DisjointBitSetNode as cppDisjointBitSetNode,
    DisjointBitSetsNode as cppDisjointBitSetsNode,
    DisjointListNode as cppDisjointListNode,
    DisjointListsNode as cppDisjointListsNode,
    EqualNode as cppEqualNode,
    IntegerNode as cppIntegerNode,
    LessEqualNode as cppLessEqualNode,
    ListNode as cppListNode,
    MaxNode as cppMaxNode,
    MaximumNode as cppMaximumNode,
    MinNode as cppMinNode,
    MinimumNode as cppMinimumNode,
    MultiplyNode as cppMultiplyNode,
    NaryAddNode as cppNaryAddNode,
    NaryMaximumNode as cppNaryMaximumNode,
    NaryMinimumNode as cppNaryMinimumNode,
    NaryMultiplyNode as cppNaryMultiplyNode,
    NegativeNode as cppNegativeNode,
    OrNode as cppOrNode,
    PermutationNode as cppPermutationNode,
    ProdNode as cppProdNode,
    QuadraticModel as cppQuadraticModel,
    QuadraticModelNode as cppQuadraticModelNode,
    ReshapeNode as cppReshapeNode,
    SetNode as cppSetNode,
    SubtractNode as cppSubtractNode,
    SquareNode as cppSquareNode,
    SumNode as cppSumNode,
    )
from dwave.optimization.model cimport ArrayObserver, Model, NodeObserver
from cython.operator cimport dereference as deref

__all__ = [
    "Absolute",
    "Add",
    "All",
    "And",
    "AdvancedIndexing",
    "BasicIndexing",
    "BinaryVariable",
    "_CombinedIndexing",
    "Constant",
    "DisjointBitSets",
    "DisjointBitSet",
    "DisjointLists",
    "DisjointList",
    "Equal",
    "IntegerVariable",
    "LessEqual",
    "ListVariable",
    "Max",
    "Maximum",
    "Min",
    "Minimum",
    "Multiply",
    "NaryAdd",
    "NaryMaximum",
    "NaryMinimum",
    "NaryMultiply",
    "Negative",
    "Or",
    "Permutation",
    "Prod",
    "QuadraticModel",
    "Reshape",
    "Subtract",
    "SetVariable",
    "Square",
    "Sum",
    ]

# We would like to be able to do constructions like dynamic_cast[cppConstantNode*](...)
# but Cython does not allow pointers as template types
# see https://github.com/cython/cython/issues/2143
# So instead we create our own wrapper to handle this case. Crucially, the
# template type is the class, but it dynamically casts on the pointer
cdef extern from *:
    """
    template<class T, class F>
    T* dynamic_cast_ptr(F* ptr) {
        return dynamic_cast<T*>(ptr);
    }
    """
    cdef T* dynamic_cast_ptr[T](...) noexcept


# This is the most robust way I can think of to do this that doesn't rely on compiler-specific
# info like type_info::name.  Note that while the cases in this function are listed alphabetically
# it could lead to miscasting if one node-type extends from another (eg, binary extends from 
# integer) but the /parent/ class occurs first alphabetically.
cdef object symbol_from_ptr(Model model, cppArrayOrNode* ptr):
    """Create a Python/Cython symbol from a C++ Node*."""

    if dynamic_cast_ptr[cppAbsoluteNode](ptr):
        return Absolute.from_ptr(model, dynamic_cast_ptr[cppAbsoluteNode](ptr))

    elif dynamic_cast_ptr[cppAddNode](ptr):
        return Add.from_ptr(model, dynamic_cast_ptr[cppAddNode](ptr))

    elif dynamic_cast_ptr[cppAllNode](ptr):
        return All.from_ptr(model, dynamic_cast_ptr[cppAllNode](ptr))

    elif dynamic_cast_ptr[cppAndNode](ptr):
        return And.from_ptr(model, dynamic_cast_ptr[cppAndNode](ptr))

    elif dynamic_cast_ptr[cppAdvancedIndexingNode](ptr):
        return AdvancedIndexing.from_ptr(model, dynamic_cast_ptr[cppAdvancedIndexingNode](ptr))

    elif dynamic_cast_ptr[cppArrayValidationNode](ptr):
        return _ArrayValidation.from_ptr(model, dynamic_cast_ptr[cppArrayValidationNode](ptr))

    elif dynamic_cast_ptr[cppBasicIndexingNode](ptr):
        return BasicIndexing.from_ptr(model, dynamic_cast_ptr[cppBasicIndexingNode](ptr))

    elif dynamic_cast_ptr[cppBinaryNode](ptr):
        return BinaryVariable.from_ptr(model, dynamic_cast_ptr[cppBinaryNode](ptr))

    elif dynamic_cast_ptr[cppConstantNode](ptr):
        return Constant.from_ptr(model, dynamic_cast_ptr[cppConstantNode](ptr))

    elif dynamic_cast_ptr[cppDisjointBitSetsNode](ptr):
        return DisjointBitSets.from_ptr(model, dynamic_cast_ptr[cppDisjointBitSetsNode](ptr))

    elif dynamic_cast_ptr[cppDisjointBitSetNode](ptr):
        return DisjointBitSet.from_ptr(model, dynamic_cast_ptr[cppDisjointBitSetNode](ptr))

    elif dynamic_cast_ptr[cppDisjointListsNode](ptr):
        return DisjointLists.from_ptr(model, dynamic_cast_ptr[cppDisjointListsNode](ptr))

    elif dynamic_cast_ptr[cppDisjointListNode](ptr):
        return DisjointList.from_ptr(model, dynamic_cast_ptr[cppDisjointListNode](ptr))

    elif dynamic_cast_ptr[cppEqualNode](ptr):
        return Equal.from_ptr(model, dynamic_cast_ptr[cppEqualNode](ptr))

    elif dynamic_cast_ptr[cppIntegerNode](ptr):
        return IntegerVariable.from_ptr(model, dynamic_cast_ptr[cppIntegerNode](ptr))

    elif dynamic_cast_ptr[cppLessEqualNode](ptr):
        return LessEqual.from_ptr(model, dynamic_cast_ptr[cppLessEqualNode](ptr))

    elif dynamic_cast_ptr[cppListNode](ptr):
        return ListVariable.from_ptr(model, dynamic_cast_ptr[cppListNode](ptr))

    elif dynamic_cast_ptr[cppMaximumNode](ptr):
        return Maximum.from_ptr(model, dynamic_cast_ptr[cppMaximumNode](ptr))

    elif dynamic_cast_ptr[cppMaxNode](ptr):
        return Max.from_ptr(model, dynamic_cast_ptr[cppMaxNode](ptr))

    elif dynamic_cast_ptr[cppMinimumNode](ptr):
        return Minimum.from_ptr(model, dynamic_cast_ptr[cppMinimumNode](ptr))

    elif dynamic_cast_ptr[cppMinNode](ptr):
        return Min.from_ptr(model, dynamic_cast_ptr[cppMinNode](ptr))

    elif dynamic_cast_ptr[cppMultiplyNode](ptr):
        return Multiply.from_ptr(model, dynamic_cast_ptr[cppMultiplyNode](ptr))

    elif dynamic_cast_ptr[cppNaryAddNode](ptr):
        return NaryAdd.from_ptr(model, dynamic_cast_ptr[cppNaryAddNode](ptr))

    elif dynamic_cast_ptr[cppNaryMaximumNode](ptr):
        return NaryMaximum.from_ptr(model, dynamic_cast_ptr[cppNaryMaximumNode](ptr))

    elif dynamic_cast_ptr[cppNaryMinimumNode](ptr):
        return NaryMinimum.from_ptr(model, dynamic_cast_ptr[cppNaryMinimumNode](ptr))

    elif dynamic_cast_ptr[cppNaryMultiplyNode](ptr):
        return NaryMultiply.from_ptr(model, dynamic_cast_ptr[cppNaryMultiplyNode](ptr))

    elif dynamic_cast_ptr[cppNegativeNode](ptr):
        return Negative.from_ptr(model, dynamic_cast_ptr[cppNegativeNode](ptr))

    elif dynamic_cast_ptr[cppOrNode](ptr):
        return Or.from_ptr(model, dynamic_cast_ptr[cppOrNode](ptr))

    elif dynamic_cast_ptr[cppPermutationNode](ptr):
        return Permutation.from_ptr(model, dynamic_cast_ptr[cppPermutationNode](ptr))

    elif dynamic_cast_ptr[cppProdNode](ptr):
        return Prod.from_ptr(model, dynamic_cast_ptr[cppProdNode](ptr))

    elif dynamic_cast_ptr[cppQuadraticModelNode](ptr):
        return QuadraticModel.from_ptr(model, dynamic_cast_ptr[cppQuadraticModelNode](ptr))

    elif dynamic_cast_ptr[cppReshapeNode](ptr):
        return Reshape.from_ptr(model, dynamic_cast_ptr[cppReshapeNode](ptr))

    elif dynamic_cast_ptr[cppSetNode](ptr):
        return SetVariable.from_ptr(model, dynamic_cast_ptr[cppSetNode](ptr))

    elif dynamic_cast_ptr[cppSquareNode](ptr):
        return Square.from_ptr(model, dynamic_cast_ptr[cppSquareNode](ptr))

    elif dynamic_cast_ptr[cppSubtractNode](ptr):
        return Subtract.from_ptr(model, dynamic_cast_ptr[cppSubtractNode](ptr))

    elif dynamic_cast_ptr[cppSumNode](ptr):
        return Sum.from_ptr(model, dynamic_cast_ptr[cppSumNode](ptr))

    raise RuntimeError("given pointer cannot be cast to a known node type")


cdef vector[Py_ssize_t] _as_cppshape(object shape):
    """Convert a shape specified as a python object to a C++ vector."""

    # Use the same error messages as NumPy

    if isinstance(shape, numbers.Integral):
        return _as_cppshape((shape,))

    if not isinstance(shape, collections.abc.Sequence):
        raise TypeError(f"expected a sequence of integers or a single integer, got '{repr(shape)}'")

    shape = tuple(shape)  # cast from list or whatever

    if not all(isinstance(x, numbers.Integral) for x in shape):
        raise ValueError(f"expected a sequence of integers or a single integer, got '{repr(shape)}'")

    if any(x < 0 for x in shape):
        raise ValueError("negative dimensions are not allowed")

    return shape


cdef bool _empty_slice(object slice_) noexcept:
    return slice_.start is None and slice_.stop is None and slice_.step is None


cdef class Absolute(ArrayObserver):
    """Absolute value element-wise on a symbol.
    
    Examples:
        This example adds the absolute value of an integer decision 
        variable to a model.
        
        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> i = model.integer(1, lower_bound=-50, upper_bound=50) 
        >>> i_abs = abs(i)
        >>> type(i_abs)
        dwave.optimization.symbols.Absolute
    """
    def __init__(self, ArrayObserver x):
        cdef Model model = x.model

        self.ptr = model._graph.emplace_node[cppAbsoluteNode](x.node_ptr)
        self.initialize_node(model, self.ptr)
        self.initialize_array(self.ptr)

    @staticmethod
    cdef Absolute from_ptr(Model model, cppAbsoluteNode* ptr):
        cdef Absolute x = Absolute.__new__(Absolute)
        x.ptr = ptr
        x.initialize_node(model, ptr)
        x.initialize_array(ptr)
        return x

    cdef cppAbsoluteNode* ptr


cdef class Add(ArrayObserver):
    """Addition element-wise of two symbols.
    
    Examples:
        This example adds two integer symbols.
        
        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> i = model.integer(10, lower_bound=-50, upper_bound=50) 
        >>> j = model.integer(10, lower_bound=0, upper_bound=10)
        >>> k = i + j
        >>> type(k)
        dwave.optimization.symbols.Add
    """
    def __init__(self, ArrayObserver lhs, ArrayObserver rhs):
        if lhs.model is not rhs.model:
            raise ValueError("lhs and rhs do not share the same underlying model")

        cdef Model model = lhs.model

        self.ptr = model._graph.emplace_node[cppAddNode](lhs.node_ptr, rhs.node_ptr)
        self.initialize_node(model, self.ptr)
        self.initialize_array(self.ptr)

    @staticmethod
    cdef Add from_ptr(Model model, cppAddNode* ptr):
        cdef Add x = Add.__new__(Add)
        x.ptr = ptr
        x.initialize_node(model, ptr)
        x.initialize_array(ptr)
        return x

    cdef cppAddNode* ptr


cdef class All(ArrayObserver):
    """Tests whether all elements evaluate to True.
    
    Examples:
        This example checks all elements of a binary array.
        
        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> x = model.binary((20, 30)) 
        >>> all_x = x.all()
        >>> type(all_x)
        dwave.optimization.symbols.All
    """
    def __init__(self, ArrayObserver array):
        cdef Model model = array.model
        self.ptr = model._graph.emplace_node[cppAllNode](array.node_ptr)
        self.initialize_node(model, self.ptr)
        self.initialize_array(self.ptr)

    @staticmethod
    cdef All from_ptr(Model model, cppAllNode* ptr):
        cdef All s = All.__new__(All)
        s.ptr = ptr
        s.initialize_node(model, ptr)
        s.initialize_array(ptr)
        return s

    cdef cppAllNode* ptr


cdef class And(ArrayObserver):
    """Boolean AND element-wise between two symbols.
    
    Examples:
        This example creates an AND operation between binary arrays.
        
        >>> from dwave.optimization.model import Model
        >>> from dwave.optimization.mathematical import logical_and
        ...
        >>> model = Model()
        >>> x = model.binary(200)
        >>> y = model.binary(200)
        >>> z = logical_and(x, y)
        >>> type(z)
        dwave.optimization.symbols.And
    """ 
    def __init__(self, ArrayObserver lhs, ArrayObserver rhs):
        if lhs.model is not rhs.model:
            raise ValueError("lhs and rhs do not share the same underlying model")

        cdef Model model = lhs.model

        self.ptr = model._graph.emplace_node[cppAndNode](lhs.node_ptr, rhs.node_ptr)
        self.initialize_node(model, self.ptr)
        self.initialize_array(self.ptr)

    @staticmethod
    cdef And from_ptr(Model model, cppAndNode* ptr):
        cdef And x = And.__new__(And)
        x.ptr = ptr
        x.initialize_node(model, ptr)
        x.initialize_array(ptr)
        return x

    cdef cppAndNode* ptr


cdef class _ArrayValidation(NodeObserver):
    def __init__(self, ArrayObserver array_node):
        cdef Model model = array_node.model

        self.ptr = model._graph.emplace_node[cppArrayValidationNode](array_node.node_ptr)

        self.initialize_node(model, self.ptr)

    @staticmethod
    cdef _ArrayValidation from_ptr(Model model, cppArrayValidationNode* ptr):
        cdef _ArrayValidation m = _ArrayValidation.__new__(_ArrayValidation)
        m.ptr = ptr
        m.initialize_node(model, ptr)
        return m

    cdef cppArrayValidationNode* ptr


cdef class AdvancedIndexing(ArrayObserver):
    """Advanced indexing.
    
    Examples:
        This example uses advanced indexing to set a symbol's values.
        
        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> prices = model.constant([i for i in range(20)])
        >>> items = model.set(20)
        >>> values = prices[items]
        >>> type(values)
        dwave.optimization.symbols.AdvancedIndexing
    """
    def __init__(self, ArrayObserver array, *indices):
        cdef Model model = array.model

        cdef vector[cppAdvancedIndexingNode.array_or_slice] cppindices

        cdef ArrayObserver array_index
        for index in indices:
            if isinstance(index, slice):
                if index != slice(None):
                    raise ValueError("AdvancedIndexing can only parse empty slices")

                cppindices.emplace_back(cppSlice())
            else:
                array_index = index
                if array_index.model != model:
                    raise ValueError("mismatched parent models")

                cppindices.emplace_back(array_index.array_ptr)

        self.ptr = model._graph.emplace_node[cppAdvancedIndexingNode](array.node_ptr, cppindices)

        self.initialize_node(model, self.ptr)
        self.initialize_array(self.ptr)

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
                    isinstance(i1, ArrayObserver) and
                    holds_alternative[cppArrayPtr](self.ptr.indices()[0]) and
                    get[cppArrayPtr](self.ptr.indices()[0]) == (<ArrayObserver>i1).array_ptr and
                    holds_alternative[cppSlice](self.ptr.indices()[1])):

                return Permutation(array, i1)

            # check the [:, x][x, :] case
            if (isinstance(i1, slice) and _empty_slice(i1) and
                    isinstance(i0, ArrayObserver) and
                    holds_alternative[cppArrayPtr](self.ptr.indices()[1]) and
                    get[cppArrayPtr](self.ptr.indices()[1]) == (<ArrayObserver>i0).array_ptr and
                    holds_alternative[cppSlice](self.ptr.indices()[0])):

                return Permutation(array, i0)

        return super().__getitem__(index)

    @staticmethod
    cdef AdvancedIndexing from_ptr(Model model, cppAdvancedIndexingNode* ptr):
        cdef AdvancedIndexing sym = AdvancedIndexing.__new__(AdvancedIndexing)
        sym.ptr = ptr
        sym.initialize_node(model, ptr)
        sym.initialize_array(ptr)
        return sym

    @classmethod
    def _from_zipfile(cls, zf, directory, Model model, predecessors):
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

        cdef cppArray* ptr
        for variant in self.ptr.indices():
            if holds_alternative[cppArrayPtr](variant):
                ptr = get[cppArrayPtr](variant)
                indices.append(symbol_from_ptr(self.model, ptr).topological_index())
            elif holds_alternative[cppSlice](variant):
                indices.append((0, 0, 0))
            else:
                raise RuntimeError

        zf.writestr(directory + "indices.json", encoder.encode(indices))

    cdef cppAdvancedIndexingNode* ptr


cdef class BasicIndexing(ArrayObserver):
    """Basic indexing.
    
    Examples:
        This example uses basic indexing to set a symbol's values.
        
        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> prices = model.constant([i for i in range(20)])
        >>> low_prices = prices[:10]
        >>> type(low_prices)
        dwave.optimization.symbols.BasicIndexing
    """
    def __init__(self, ArrayObserver array, *indices):

        cdef Model model = array.model

        cdef vector[cppBasicIndexingNode.slice_or_int] cppindices
        for index in indices:
            if isinstance(index, slice):
                cppindices.emplace_back(BasicIndexing.cppslice(index))
            else:
                cppindices.emplace_back(<Py_ssize_t>(index))

        self.ptr = model._graph.emplace_node[cppBasicIndexingNode](array.node_ptr, cppindices)

        self.initialize_node(model, self.ptr)
        self.initialize_array(self.ptr)

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

    @staticmethod
    cdef BasicIndexing from_ptr(Model model, cppBasicIndexingNode* ptr):
        cdef BasicIndexing sym = BasicIndexing.__new__(BasicIndexing)
        sym.ptr = ptr
        sym.initialize_node(model, ptr)
        sym.initialize_array(ptr)
        return sym

    @classmethod
    def _from_zipfile(cls, zf, directory, Model model, predecessors):
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


cdef class BinaryVariable(ArrayObserver):
    """Binary decision-variable symbol.
    
    Examples:
        This example adds a binary variable to a model.
        
        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> x = model.binary((20, 30))
        >>> type(x)
        dwave.optimization.symbols.BinaryVariable
    """
    def __init__(self, Model model, shape=None):
        # Get an observing pointer to the node
        cdef vector[Py_ssize_t] vshape = _as_cppshape(tuple() if shape is None else shape)

        self.ptr = model._graph.emplace_node[cppBinaryNode](vshape)

        self.initialize_node(model, self.ptr)
        self.initialize_array(self.ptr)

    @staticmethod
    cdef BinaryVariable from_ptr(Model model, cppBinaryNode* ptr):
        cdef BinaryVariable x = BinaryVariable.__new__(BinaryVariable)
        x.ptr = ptr
        x.initialize_node(model, ptr)
        x.initialize_array(ptr)
        return x

    @classmethod
    def _from_zipfile(cls, zf, directory, Model model, predecessors):
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
        self.ptr.initialize_state(self.model.states._states[index], move(items))

    # An observing pointer to the C++ BinaryNode
    cdef cppBinaryNode* ptr


cdef class Constant(ArrayObserver):
    """Constant symbol.
    
    Examples:
        This example adds a constant symbol to a model.
        
        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> a = model.constant(20)
        >>> type(a)
        dwave.optimization.symbols.Constant
    """
    def __init__(self, Model model, array_like):
        # In the future we won't need to be contiguous, but we do need to be right now
        array = np.asarray(array_like, dtype=np.double, order="C")

        # Get the shape and strides
        cdef vector[Py_ssize_t] shape = array.shape
        cdef vector[Py_ssize_t] strides = array.strides  # not used because contiguous for now

        # Get a pointer to the first element
        cdef double[:] flat = array.ravel()
        cdef double* start = NULL
        if flat.size:
            start = &flat[0]

        # Get an observing pointer to the C++ ConstantNode
        self.ptr = model._graph.emplace_node[cppConstantNode](start, shape)

        self.initialize_node(model, self.ptr)
        self.initialize_array(self.ptr)

        # Have the parent model hold a reference to the array, so it's kept alive
        model._data_sources.append(array)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        buffer.buf = <void*>(self.ptr.buff())
        buffer.format = <char*>(self.ptr.format().c_str())
        buffer.internal = NULL
        buffer.itemsize = self.ptr.itemsize()
        buffer.len = self.ptr.len()
        buffer.ndim = self.ptr.ndim()
        buffer.obj = self
        buffer.readonly = 1  # todo: consider loosening this requirement
        buffer.shape = <Py_ssize_t*>(self.ptr.shape().data())
        buffer.strides = <Py_ssize_t*>(self.ptr.strides().data())
        buffer.suboffsets = NULL

    @staticmethod
    cdef Constant from_ptr(Model model, cppConstantNode* ptr):
        cdef Constant constant = Constant.__new__(Constant)
        constant.ptr = ptr
        constant.initialize_node(model, ptr)
        constant.initialize_array(ptr)
        return constant

    @classmethod
    def _from_zipfile(cls, zf, directory, Model model, predecessors):
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
        if maybe != 1:
            return True if maybe else False

        # avoid NumPy deprecation warning by casting to bool. But also
        # `bool` in this namespace is a C++ class so we do an explicit if else
        equal = (np.asarray(self) == np.asarray(other)).all()
        return True if equal else False

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


cdef class DisjointBitSets(NodeObserver):
    """Disjoint-sets decision-variable symbol.
    
    Examples:
        This example adds a disjoint-sets symbol to a model.
        
        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> s = model.disjoint_bit_sets(primary_set_size=100, num_disjoint_sets=5)
        >>> type(s[0])
        dwave.optimization.symbols.DisjointBitSets
    """
    def __init__(
        self, Model model, Py_ssize_t primary_set_size, Py_ssize_t num_disjoint_sets
    ):
        # Get an observing pointer to the node
        self.ptr = model._graph.emplace_node[cppDisjointBitSetsNode](
            primary_set_size, num_disjoint_sets
        )

        self.initialize_node(model, self.ptr)

    @staticmethod
    cdef DisjointBitSets from_ptr(Model model, cppDisjointBitSetsNode* ptr):
        cdef DisjointBitSets x = DisjointBitSets.__new__(DisjointBitSets)
        x.ptr = ptr
        x.initialize_node(model, ptr)
        return x

    @classmethod
    def _from_zipfile(cls, zf, directory, Model model, predecessors):
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
        :math:`num_disjoint_sets \times primary_set_size` Boolean array.
               
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
        self.ptr.initialize_state(self.model.states._states[index], move(sets))

    def _state_from_zipfile(self, zf, directory, Py_ssize_t state_index):
        arrays = []
        for i in range(self.num_disjoint_sets()):
            with zf.open(directory+f"set{i}", mode="r") as f:
                arrays.append(np.load(f, allow_pickle=False))

        self.set_state(state_index, arrays)

    def _state_into_zipfile(self, zf, directory, Py_ssize_t state_index):
        # do this first to get any potential error messages out of the way
        # todo: use a view not a copy
        arrays = [np.asarray(s.state(state_index), dtype=np.int8) for s in self.iter_successors()]

        for i, arr in enumerate(arrays):
            with zf.open(directory+f"set{i}", mode="w", force_zip64=True) as f:
                np.save(f, arr, allow_pickle=False)

    def num_disjoint_sets(self):
        """Return the number of disjoint sets in the symbol."""
        return self.ptr.num_disjoint_sets()

    # An observing pointer to the C++ DisjointBitSetsNode
    cdef cppDisjointBitSetsNode* ptr


cdef class DisjointBitSet(ArrayObserver):
    """Disjoint-sets successor symbol.
    
    Examples:
        This example adds a disjoint-sets symbol to a model.
        
        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> s = model.disjoint_bit_sets(primary_set_size=100, num_disjoint_sets=5)
        >>> type(s[1][0])
        dwave.optimization.symbols.DisjointBitSet
    """
    def __init__(self, DisjointBitSets parent, Py_ssize_t set_index):
        if set_index < 0 or set_index >= parent.num_disjoint_sets():
            raise ValueError(
                "`set_index` must be less than the number of disjoint sets of the parent"
            )

        if set_index > <Py_ssize_t>(parent.ptr.successors().size()):
            raise ValueError("`DisjointBitSet`s must be created successively")

        cdef Model model = parent.model
        if set_index == <Py_ssize_t>(parent.ptr.successors().size()):
            # The DisjointBitSet has not been added to the model yet, so add it
            self.ptr = model._graph.emplace_node[cppDisjointBitSetNode](parent.ptr)
        else:
            # Already been added to the model, so grab the pointer from the parent's
            # successors
            self.ptr = dynamic_cast_ptr[cppDisjointBitSetNode](
                parent.ptr.successors()[set_index].ptr
            )

        self.initialize_node(model, self.ptr)
        self.initialize_array(self.ptr)

    @staticmethod
    cdef DisjointBitSet from_ptr(Model model, cppDisjointBitSetNode* ptr):
        cdef DisjointBitSet x = DisjointBitSet.__new__(DisjointBitSet)
        x.ptr = ptr
        x.initialize_node(model, ptr)
        x.initialize_array(ptr)
        return x

    @classmethod
    def _from_zipfile(cls, zf, directory, Model model, predecessors):
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


cdef class DisjointLists(NodeObserver):
    """Disjoint-lists decision-variable symbol.
    
    Examples:
        This example adds a disjoint-lists symbol to a model.
        
        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> l = model.disjoint_lists(primary_set_size=10, num_disjoint_lists=2)
        >>> type(l[0])
        dwave.optimization.symbols.DisjointLists
    """
    def __init__(
        self, Model model, Py_ssize_t primary_set_size, Py_ssize_t num_disjoint_lists
    ):
        # Get an observing pointer to the node
        self.ptr = model._graph.emplace_node[cppDisjointListsNode](
            primary_set_size, num_disjoint_lists
        )

        self.initialize_node(model, self.ptr)

    @staticmethod
    cdef DisjointLists from_ptr(Model model, cppDisjointListsNode* ptr):
        cdef DisjointLists x = DisjointLists.__new__(DisjointLists)
        x.ptr = ptr
        x.initialize_node(model, ptr)
        return x

    @classmethod
    def _from_zipfile(cls, zf, directory, Model model, predecessors):
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
        self.ptr.initialize_state(self.model.states._states[index], move(items))

    def _state_from_zipfile(self, zf, directory, Py_ssize_t state_index):
        arrays = []
        for i in range(self.num_disjoint_lists()):
            with zf.open(directory+f"list{i}", mode="r") as f:
                arrays.append(np.load(f, allow_pickle=False))

        self.set_state(state_index, arrays)

    def _state_into_zipfile(self, zf, directory, Py_ssize_t state_index):
        # do this first to get any potential error messages out of the way
        # todo: use a view not a copy
        arrays = [li.state(state_index) for li in self.iter_successors()]

        for i, arr in enumerate(arrays):
            with zf.open(directory+f"list{i}", mode="w", force_zip64=True) as f:
                np.save(f, arr, allow_pickle=False)

    def num_disjoint_lists(self):
        """Return the number of disjoint lists in the symbol."""
        return self.ptr.num_disjoint_lists()

    # An observing pointer to the C++ DisjointListsNode
    cdef cppDisjointListsNode* ptr


cdef class DisjointList(ArrayObserver):
    """Disjoint-lists successor symbol.
    
    Examples:
        This example adds a disjoint-lists symbol to a model.
        
        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> l = model.disjoint_lists(primary_set_size=10, num_disjoint_lists=2)
        >>> type(l[1][0])
        dwave.optimization.symbols.DisjointList
    """
    def __init__(self, DisjointLists parent, Py_ssize_t list_index):
        if list_index < 0 or list_index >= parent.num_disjoint_lists():
            raise ValueError(
                "`list_index` must be less than the number of disjoint sets of the parent"
            )

        if list_index > <Py_ssize_t>(parent.ptr.successors().size()):
            raise ValueError("`DisjointList`s must be created successively")

        cdef Model model = parent.model
        if list_index == <Py_ssize_t>(parent.ptr.successors().size()):
            # The DisjointListNode has not been added to the model yet, so add it
            self.ptr = model._graph.emplace_node[cppDisjointListNode](parent.ptr)
        else:
            # Already been added to the model, so grab the pointer from the parent's
            # successors
            self.ptr = dynamic_cast_ptr[cppDisjointListNode](
                parent.ptr.successors()[list_index].ptr
            )

        self.initialize_node(model, self.ptr)
        self.initialize_array(self.ptr)

    @staticmethod
    cdef DisjointList from_ptr(Model model, cppDisjointListNode* ptr):
        cdef DisjointList x = DisjointList.__new__(DisjointList)
        x.ptr = ptr
        x.initialize_node(model, ptr)
        x.initialize_array(ptr)
        return x

    @classmethod
    def _from_zipfile(cls, zf, directory, Model model, predecessors):
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


cdef class Equal(ArrayObserver):
    """Equality comparison element-wise between two symbols.
    
    Examples:
        This example creates an equality operation between integer symbols.
        
        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> i = model.integer(25, upper_bound=100)
        >>> j = model.integer(25, lower_bound=-100)
        >>> k = i == j
        >>> type(k)
        dwave.optimization.symbols.Equal
    """
    def __init__(self, ArrayObserver lhs, ArrayObserver rhs):
        if lhs.model is not rhs.model:
            raise ValueError("lhs and rhs do not share the same underlying model")

        cdef Model model = lhs.model

        self.ptr = model._graph.emplace_node[cppEqualNode](lhs.node_ptr, rhs.node_ptr)
        self.initialize_node(model, self.ptr)
        self.initialize_array(self.ptr)

    @staticmethod
    cdef Equal from_ptr(Model model, cppEqualNode* ptr):
        cdef Equal x = Equal.__new__(Equal)
        x.ptr = ptr
        x.initialize_node(model, ptr)
        x.initialize_array(ptr)
        return x

    cdef cppEqualNode* ptr


cdef class LessEqual(ArrayObserver):
    """Smaller-or-equal comparison element-wise between two symbols.
    
    Examples:
        This example creates an inequality operation between integer symbols.
        
        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> i = model.integer(25, upper_bound=100)
        >>> j = model.integer(25, lower_bound=-100)
        >>> k = i <= j
        >>> type(k)
        dwave.optimization.symbols.LessEqual
    """
    def __init__(self, ArrayObserver lhs, ArrayObserver rhs):
        if lhs.model is not rhs.model:
            raise ValueError("lhs and rhs do not share the same underlying model")

        cdef Model model = lhs.model

        self.ptr = model._graph.emplace_node[cppLessEqualNode](lhs.node_ptr, rhs.node_ptr)
        self.initialize_node(model, self.ptr)
        self.initialize_array(self.ptr)

    @staticmethod
    cdef LessEqual from_ptr(Model model, cppLessEqualNode* ptr):
        cdef LessEqual x = LessEqual.__new__(LessEqual)
        x.ptr = ptr
        x.initialize_node(model, ptr)
        x.initialize_array(ptr)
        return x

    cdef cppLessEqualNode* ptr


cdef class ListVariable(ArrayObserver):
    """List decision-variable symbol.
    
    Examples:
        This example adds a list symbol to a model.
        
        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> l = model.list(10)
        >>> type(l)
        dwave.optimization.symbols.ListVariable
    """
    def __init__(self, Model model, Py_ssize_t n):
        # Get an observing pointer to the node
        self.ptr = model._graph.emplace_node[cppListNode](n)

        self.initialize_node(model, self.ptr)
        self.initialize_array(self.ptr)

    @staticmethod
    cdef ListVariable from_ptr(Model model, cppListNode* ptr):
        cdef ListVariable x = ListVariable.__new__(ListVariable)
        x.ptr = ptr
        x.initialize_node(model, ptr)
        x.initialize_array(ptr)
        return x

    @classmethod
    def _from_zipfile(cls, zf, directory, Model model, predecessors):
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

    def set_state(self, Py_ssize_t index, state):
        """Set the state of the list node.

        The given state must be a permuation of ``range(len(state))``.
        """
        # Convert the state into something we can handle in C++.
        # This also does some type checking etc
        # We go ahead and do the translation to integer now
        cdef Py_ssize_t[:] arr = np.asarray(state, dtype=np.intp)

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
        self.ptr.initialize_state(self.model.states._states[index], move(items))

    # An observing pointer to the C++ ListNode
    cdef cppListNode* ptr


cdef class IntegerVariable(ArrayObserver):
    """Integer decision-variable symbol.
    
    Examples:
        This example adds an integer symbol to a model.
        
        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> i = model.integer(25, upper_bound=100)
        >>> type(i)
        dwave.optimization.symbols.IntegerVariable
    """
    def __init__(self, Model model, shape=None, lower_bound=None, upper_bound=None):
        cdef vector[Py_ssize_t] vshape = _as_cppshape(tuple() if shape is None else shape )

        if lower_bound is None and upper_bound is None:
            self.ptr = model._graph.emplace_node[cppIntegerNode](vshape)
        elif lower_bound is None:
            self.ptr = model._graph.emplace_node[cppIntegerNode](vshape, nullopt, <double>upper_bound)
        elif upper_bound is None:
            self.ptr = model._graph.emplace_node[cppIntegerNode](vshape, <double>lower_bound)
        else:
            self.ptr = model._graph.emplace_node[cppIntegerNode](vshape, <double>lower_bound, <double>upper_bound)

        self.initialize_node(model, self.ptr)
        self.initialize_array(self.ptr)

    @staticmethod
    cdef IntegerVariable from_ptr(Model model, cppIntegerNode* ptr):
        cdef IntegerVariable x = IntegerVariable.__new__(IntegerVariable)
        x.ptr = ptr
        x.initialize_node(model, ptr)
        x.initialize_array(ptr)
        return x

    @classmethod
    def _from_zipfile(cls, zf, directory, Model model, predecessors):
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
        """The lowest value the integer(s) can take (inclusive)."""
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
        self.ptr.initialize_state(self.model.states._states[index], move(items))

    def upper_bound(self):
        """The highest value the integer(s) can take (inclusive)."""
        return int(self.ptr.upper_bound())

    # An observing pointer to the C++ IntegerNode
    cdef cppIntegerNode* ptr


cdef class Max(ArrayObserver):
    """Maximum value in the elements of a symbol.
    
    Examples:
        This example adds the maximum value of an integer decision 
        variable to a model.
        
        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> i = model.integer(100, lower_bound=-50, upper_bound=50) 
        >>> i_max = i.max()
        >>> type(i_max)
        dwave.optimization.symbols.Max
    """
    def __init__(self, ArrayObserver node):
        cdef Model model = node.model

        self.ptr = model._graph.emplace_node[cppMaxNode](node.node_ptr)

        self.initialize_node(model, self.ptr)
        self.initialize_array(self.ptr)

    @staticmethod
    cdef Max from_ptr(Model model, cppMaxNode* ptr):
        cdef Max m = Max.__new__(Max)
        m.ptr = ptr
        m.initialize_node(model, ptr)
        m.initialize_array(ptr)
        return m

    cdef cppMaxNode* ptr


cdef class Maximum(ArrayObserver):
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
        dwave.optimization.symbols.Maximum
    """
    def __init__(self, ArrayObserver lhs, ArrayObserver rhs):
        if lhs.model is not rhs.model:
            raise ValueError("lhs and rhs do not share the same underlying model")

        cdef Model model = lhs.model

        self.ptr = model._graph.emplace_node[cppMaximumNode](lhs.node_ptr, rhs.node_ptr)

        self.initialize_node(model, self.ptr)
        self.initialize_array(self.ptr)

    @staticmethod
    cdef Maximum from_ptr(Model model, cppMaximumNode* ptr):
        cdef Maximum m = Maximum.__new__(Maximum)
        m.ptr = ptr
        m.initialize_node(model, ptr)
        m.initialize_array(ptr)
        return m

    cdef cppMaximumNode* ptr


cdef class Min(ArrayObserver):
    """Minimum value in the elements of a symbol.
    
    Examples:
        This example adds the minimum value of an integer decision 
        variable to a model.
        
        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> i = model.integer(100, lower_bound=-50, upper_bound=50) 
        >>> i_min = i.min()
        >>> type(i_min)
        dwave.optimization.symbols.Min
    """
    def __init__(self, ArrayObserver node):
        cdef Model model = node.model

        self.ptr = model._graph.emplace_node[cppMinNode](node.node_ptr)

        self.initialize_node(model, self.ptr)
        self.initialize_array(self.ptr)

    @staticmethod
    cdef Min from_ptr(Model model, cppMinNode* ptr):
        cdef Min m = Min.__new__(Min)
        m.ptr = ptr
        m.initialize_node(model, ptr)
        m.initialize_array(ptr)
        return m

    cdef cppMinNode* ptr


cdef class Minimum(ArrayObserver):
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
        dwave.optimization.symbols.Minimum
    """
    def __init__(self, ArrayObserver lhs, ArrayObserver rhs):
        if lhs.model is not rhs.model:
            raise ValueError("lhs and rhs do not share the same underlying model")

        cdef Model model = lhs.model

        self.ptr = model._graph.emplace_node[cppMinimumNode](lhs.node_ptr, rhs.node_ptr)

        self.initialize_node(model, self.ptr)
        self.initialize_array(self.ptr)

    @staticmethod
    cdef Minimum from_ptr(Model model, cppMinimumNode* ptr):
        cdef Minimum m = Minimum.__new__(Minimum)
        m.ptr = ptr
        m.initialize_node(model, ptr)
        m.initialize_array(ptr)
        return m

    cdef cppMinimumNode* ptr


cdef class Multiply(ArrayObserver):
    """Multiplication element-wise between two symbols.
    
    Examples:
        This example multiplies two integer symbols.
        
        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> i = model.integer(10, lower_bound=-50, upper_bound=50) 
        >>> j = model.integer(10, lower_bound=0, upper_bound=10)
        >>> k = i*j
        >>> type(k)
        dwave.optimization.symbols.Multiply
    """
    def __init__(self, ArrayObserver lhs, ArrayObserver rhs):
        if lhs.model is not rhs.model:
            raise ValueError("lhs and rhs do not share the same underlying model")

        cdef Model model = lhs.model

        self.ptr = model._graph.emplace_node[cppMultiplyNode](lhs.node_ptr, rhs.node_ptr)
        self.initialize_node(model, self.ptr)
        self.initialize_array(self.ptr)

    @staticmethod
    cdef Multiply from_ptr(Model model, cppMultiplyNode* ptr):
        cdef Multiply x = Multiply.__new__(Multiply)
        x.ptr = ptr
        x.initialize_node(model, ptr)
        x.initialize_array(ptr)
        return x

    cdef cppMultiplyNode* ptr


cdef class NaryAdd(ArrayObserver):
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
        >>> l = add([i, j, k])
        >>> type(l)
        dwave.optimization.symbols.NaryAdd
    """
    def __init__(self, *inputs):
        if len(inputs) == 0:
            raise TypeError("must have at least one predecessor node")

        cdef Model model = inputs[0].model
        cdef vector[cppNode*] cppinputs

        cdef ArrayObserver array
        for node in inputs:
            if node.model != model:
                raise ValueError("all predecessors must be from the same model")
            array = <ArrayObserver?>node
            cppinputs.push_back(array.node_ptr)

        self.ptr = model._graph.emplace_node[cppNaryAddNode](cppinputs)
        self.initialize_node(model, self.ptr)
        self.initialize_array(self.ptr)

    @staticmethod
    cdef NaryAdd from_ptr(Model model, cppNaryAddNode* ptr):
        cdef NaryAdd x = NaryAdd.__new__(NaryAdd)
        x.ptr = ptr
        x.initialize_node(model, ptr)
        x.initialize_array(ptr)
        return x

    cdef cppNaryAddNode* ptr


cdef class NaryMaximum(ArrayObserver):
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
        >>> l = maximum([i, j, k])
        >>> type(l)
        dwave.optimization.symbols.NaryMaximum
    """
    def __init__(self, *inputs):
        if len(inputs) == 0:
            raise TypeError("must have at least one predecessor node")

        cdef Model model = inputs[0].model
        cdef vector[cppNode*] cppinputs

        cdef ArrayObserver array
        for node in inputs:
            if node.model != model:
                raise ValueError("all predecessors must be from the same model")
            array = <ArrayObserver?>node
            cppinputs.push_back(array.node_ptr)

        self.ptr = model._graph.emplace_node[cppNaryMaximumNode](cppinputs)
        self.initialize_node(model, self.ptr)
        self.initialize_array(self.ptr)

    @staticmethod
    cdef NaryMaximum from_ptr(Model model, cppNaryMaximumNode* ptr):
        cdef NaryMaximum x = NaryMaximum.__new__(NaryMaximum)
        x.ptr = ptr
        x.initialize_node(model, ptr)
        x.initialize_array(ptr)
        return x

    cdef cppNaryMaximumNode* ptr


cdef class NaryMinimum(ArrayObserver):
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
        >>> l = minimum([i, j, k])
        >>> type(l)
        dwave.optimization.symbols.NaryMinimum
    """
    def __init__(self, *inputs):
        if len(inputs) == 0:
            raise TypeError("must have at least one predecessor node")

        cdef Model model = inputs[0].model
        cdef vector[cppNode*] cppinputs

        cdef ArrayObserver array
        for node in inputs:
            if node.model != model:
                raise ValueError("all predecessors must be from the same model")
            array = <ArrayObserver?>node
            cppinputs.push_back(array.node_ptr)

        self.ptr = model._graph.emplace_node[cppNaryMinimumNode](cppinputs)
        self.initialize_node(model, self.ptr)
        self.initialize_array(self.ptr)

    @staticmethod
    cdef NaryMinimum from_ptr(Model model, cppNaryMinimumNode* ptr):
        cdef NaryMinimum x = NaryMinimum.__new__(NaryMinimum)
        x.ptr = ptr
        x.initialize_node(model, ptr)
        x.initialize_array(ptr)
        return x

    cdef cppNaryMinimumNode* ptr


cdef class NaryMultiply(ArrayObserver):
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
        >>> l = multiply([i, j, k])
        >>> type(l)
        dwave.optimization.symbols.NaryMultiply
    """
    def __init__(self, *inputs):
        if len(inputs) == 0:
            raise TypeError("must have at least one predecessor node")

        cdef Model model = inputs[0].model
        cdef vector[cppNode*] cppinputs

        cdef ArrayObserver array
        for node in inputs:
            if node.model != model:
                raise ValueError("all predecessors must be from the same model")
            array = <ArrayObserver?>node
            cppinputs.push_back(array.node_ptr)

        self.ptr = model._graph.emplace_node[cppNaryMultiplyNode](cppinputs)
        self.initialize_node(model, self.ptr)
        self.initialize_array(self.ptr)

    @staticmethod
    cdef NaryMultiply from_ptr(Model model, cppNaryMultiplyNode* ptr):
        cdef NaryMultiply x = NaryMultiply.__new__(NaryMultiply)
        x.ptr = ptr
        x.initialize_node(model, ptr)
        x.initialize_array(ptr)
        return x

    cdef cppNaryMultiplyNode* ptr


cdef class Negative(ArrayObserver):
    """Numerical negative element-wise on a symbol.
    
    Examples:
        This example add the negative of an integer array.
        
        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> i = model.integer(100, upper_bound=50)
        >>> i_minus = -i
        >>> type(i_minus)
        dwave.optimization.symbols.Negative
    """
    def __init__(self, ArrayObserver x):
        cdef Model model = x.model

        self.ptr = model._graph.emplace_node[cppNegativeNode](x.node_ptr)
        self.initialize_node(model, self.ptr)
        self.initialize_array(self.ptr)

    @staticmethod
    cdef Negative from_ptr(Model model, cppNegativeNode* ptr):
        cdef Negative x = Negative.__new__(Negative)
        x.ptr = ptr
        x.initialize_node(model, ptr)
        x.initialize_array(ptr)
        return x

    cdef cppNegativeNode* ptr


cdef class Or(ArrayObserver):
    """Boolean OR element-wise between two symbols.
    
    Examples:
        This example creates an OR operation between binary arrays.
        
        >>> from dwave.optimization.model import Model
        >>> from dwave.optimization.mathematical import logical_or
        ...
        >>> model = Model()
        >>> x = model.binary(200)
        >>> y = model.binary(200)
        >>> z = logical_or(x, y)
        >>> type(z)
        dwave.optimization.symbols.Or
    """ 
    def __init__(self, ArrayObserver lhs, ArrayObserver rhs):
        if lhs.model is not rhs.model:
            raise ValueError("lhs and rhs do not share the same underlying model")

        cdef Model model = lhs.model

        self.ptr = model._graph.emplace_node[cppOrNode](lhs.node_ptr, rhs.node_ptr)
        self.initialize_node(model, self.ptr)
        self.initialize_array(self.ptr)

    @staticmethod
    cdef Or from_ptr(Model model, cppOrNode* ptr):
        cdef Or x = Or.__new__(Or)
        x.ptr = ptr
        x.initialize_node(model, ptr)
        x.initialize_array(ptr)
        return x

    cdef cppOrNode* ptr


cdef class Permutation(ArrayObserver):
    """Permutation of the elements of a symbol.
    
    Examples:
        This example creates a permutation of a constant symbol.
        
        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> C = model.constant([[1, 2, 3], [2, 3, 1], [0, 1, 0]])
        >>> l = model.list(3)
        >>> p = C[l, :][:, l]
        >>> type(p)
        dwave.optimization.symbols.Permutation
    """
    def __init__(self, Constant array, ListVariable x):
        # todo: Loosen the types accepted. But this Cython code doesn't yet have
        # the type heirarchy needed so for how we specify explicitly

        if array.model is not x.model:
            raise ValueError("array and x do not share the same underlying model")

        self.ptr = array.model._graph.emplace_node[cppPermutationNode](array.node_ptr, x.node_ptr)

        self.initialize_node(array.model, self.ptr)
        self.initialize_array(self.ptr)

    @staticmethod
    cdef Permutation from_ptr(Model model, cppPermutationNode* ptr):
        cdef Permutation p = Permutation.__new__(Permutation)
        p.ptr = ptr
        p.initialize_node(model, ptr)
        p.initialize_array(ptr)
        return p

    cdef cppPermutationNode* ptr


cdef class Prod(ArrayObserver):
    """Product of the elements of a symbol.
    
    Examples:
        This example adds the product of an integer symbol's 
        elements to a model.
        
        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> i = model.integer(100, lower_bound=-50, upper_bound=50) 
        >>> i_prod = i.prod()
        >>> type(i_prod)
        dwave.optimization.symbols.Prod
    """
    def __init__(self, ArrayObserver node):
        cdef Model model = node.model

        self.ptr = model._graph.emplace_node[cppProdNode](node.node_ptr)

        self.initialize_node(model, self.ptr)
        self.initialize_array(self.ptr)

    @staticmethod
    cdef Prod from_ptr(Model model, cppProdNode* ptr):
        cdef Prod m = Prod.__new__(Prod)
        m.ptr = ptr
        m.initialize_node(model, ptr)
        m.initialize_array(ptr)
        return m

    cdef cppProdNode* ptr


cdef class QuadraticModel(ArrayObserver):
    """Quadratic model.
    
    Examples:
        This example adds a quadratic model.
        
        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> x = model.binary(3) 
        >>> Q = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 1): 1, (1, 2): 3, (2, 2): 2}
        >>> qm = model.quadratic_model(x, Q)
        >>> type(qm)
        dwave.optimization.symbols.QuadraticModel
    """
    def __init__(self, ArrayObserver x, quadratic, linear=None):
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
    def _init_from_coords(self, ArrayObserver x, quadratic, linear):
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

            self.ptr = x.model._graph.emplace_node[cppQuadraticModelNode](x.node_ptr, move(deref(qm)))
        finally:
            # even if an exception is thrown, we don't leak memory
            del qm

        self.initialize_node(x.model, self.ptr)
        self.initialize_array(self.ptr)

    @cython.wraparound(False)
    def _init_from_qubo(self, ArrayObserver x, quadratic, linear):
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

    @staticmethod
    cdef QuadraticModel from_ptr(Model model, cppQuadraticModelNode* ptr):
        cdef QuadraticModel qm = QuadraticModel.__new__(QuadraticModel)
        qm.ptr = ptr
        qm.initialize_node(model, ptr)
        qm.initialize_array(ptr)
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
    def _from_zipfile(cls, zf, directory, Model model, predecessors):
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


cdef class Reshape(ArrayObserver):
    """Reshaped symbol.
    
    Examples:
        This example adds a reshaped binary symbol.
        
        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> x = model.binary((2, 3)) 
        >>> x_t = x.reshape((3, 2))
        >>> type(x_t)
        dwave.optimization.symbols.Reshape
    """
    def __init__(self, ArrayObserver node, shape):
        cdef Model model = node.model

        self.ptr = model._graph.emplace_node[cppReshapeNode](
            node.node_ptr,
            _as_cppshape(shape),
            )

        self.initialize_node(model, self.ptr)
        self.initialize_array(self.ptr)

    @staticmethod
    cdef Reshape from_ptr(Model model, cppReshapeNode* ptr):
        cdef Reshape m = Reshape.__new__(Reshape)
        m.ptr = ptr
        m.initialize_node(model, ptr)
        m.initialize_array(ptr)
        return m

    @classmethod
    def _from_zipfile(cls, zf, directory, Model model, predecessors):
        if len(predecessors) != 1:
            raise ValueError("Reshape must have exactly one predecessor")

        with zf.open(directory + "shape.json", "r") as f:
            return Reshape(*predecessors, json.load(f))

    def _into_zipfile(self, zf, directory):
        encoder = json.JSONEncoder(separators=(',', ':'))
        zf.writestr(directory + "shape.json", encoder.encode(self.shape()))

    cdef cppReshapeNode* ptr


cdef class SetVariable(ArrayObserver):
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
        dwave.optimization.symbols.SetVariable

    """
    def __init__(self, Model model, Py_ssize_t n, Py_ssize_t min_size, Py_ssize_t max_size):
        self.ptr = model._graph.emplace_node[cppSetNode](n, min_size, max_size)
        self.initialize_node(model, self.ptr)
        self.initialize_array(self.ptr)

    @staticmethod
    cdef SetVariable from_ptr(Model model, cppSetNode* ptr):
        cdef SetVariable x = SetVariable.__new__(SetVariable)
        x.ptr = ptr
        x.initialize_node(model, ptr)
        x.initialize_array(ptr)
        return x

    @classmethod
    def _from_zipfile(cls, zf, directory, Model model, predecessors):
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

    def set_state(self, Py_ssize_t index, state):
        """Set the state of the set node.

        The given state must be a permuation of ``range(len(state))``.
        """
        if isinstance(state, collections.abc.Set):
            state = sorted(state)

        # Convert the state into something we can handle in C++.
        # This also does some type checking etc
        # We go ahead and do the translation to integer now
        cdef Py_ssize_t[:] arr = np.asarray(state, dtype=np.intp)

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
        self.ptr.initialize_state(self.model.states._states[index], move(items))

    # Observing pointer to the node
    cdef cppSetNode* ptr


cdef class Square(ArrayObserver):
    """Squares element-wise of a symbol.
    
    Examples:
        This example adds the squares of an integer decision 
        variable to a model.
        
        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> i = model.integer(10, lower_bound=-5, upper_bound=5) 
        >>> ii = i**2
        >>> type(ii)
        dwave.optimization.symbols.Square
    """
    def __init__(self, ArrayObserver x):
        cdef Model model = x.model

        self.ptr = model._graph.emplace_node[cppSquareNode](x.node_ptr)
        self.initialize_node(model, self.ptr)
        self.initialize_array(self.ptr)

    @staticmethod
    cdef Square from_ptr(Model model, cppSquareNode* ptr):
        cdef Square x = Square.__new__(Square)
        x.ptr = ptr
        x.initialize_node(model, ptr)
        x.initialize_array(ptr)
        return x

    cdef cppSquareNode* ptr

cdef class Subtract(ArrayObserver):
    """Subtraction element-wise of two symbols.
    
    Examples:
        This example subtracts two integer symbols.
        
        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> i = model.integer(10, lower_bound=-50, upper_bound=50) 
        >>> j = model.integer(10, lower_bound=0, upper_bound=10)
        >>> k = i - j
        >>> type(k)
        dwave.optimization.symbols.Subtract
    """
    def __init__(self, ArrayObserver lhs, ArrayObserver rhs):
        if lhs.model is not rhs.model:
            raise ValueError("lhs and rhs do not share the same underlying model")

        cdef Model model = lhs.model

        self.ptr = model._graph.emplace_node[cppSubtractNode](lhs.node_ptr, rhs.node_ptr)
        self.initialize_node(model, self.ptr)
        self.initialize_array(self.ptr)

    @staticmethod
    cdef Subtract from_ptr(Model model, cppSubtractNode* ptr):
        cdef Subtract x = Subtract.__new__(Subtract)
        x.ptr = ptr
        x.initialize_node(model, ptr)
        x.initialize_array(ptr)
        return x

    cdef cppSubtractNode* ptr


cdef class Sum(ArrayObserver):
    """Sum of the elements of a symbol.
    
    Examples:
        This example adds the sum of an integer symbol's 
        elements to a model.
        
        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> i = model.integer(100, lower_bound=-50, upper_bound=50) 
        >>> i_sum = i.sum()
        >>> type(i_sum)
        dwave.optimization.symbols.Sum
    """
    def __init__(self, ArrayObserver array):
        cdef Model model = array.model
        self.ptr = model._graph.emplace_node[cppSumNode](array.node_ptr)
        self.initialize_node(model, self.ptr)
        self.initialize_array(self.ptr)

    @staticmethod
    cdef Sum from_ptr(Model model, cppSumNode* ptr):
        cdef Sum s = Sum.__new__(Sum)
        s.ptr = ptr
        s.initialize_node(model, ptr)
        s.initialize_array(ptr)
        return s

    cdef cppSumNode* ptr

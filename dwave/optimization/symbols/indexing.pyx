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

import json
import numbers

from cython.operator cimport typeid
from libcpp cimport bool
from libcpp.optional cimport optional
from libcpp.vector cimport vector

from dwave.optimization._model cimport _Graph, _register, ArraySymbol, Symbol, symbol_from_ptr
from dwave.optimization.libcpp cimport dynamic_cast_ptr, get, holds_alternative
from dwave.optimization.libcpp.array cimport Slice
from dwave.optimization.libcpp.graph cimport ArrayNode, Node
from dwave.optimization.libcpp.nodes.indexing cimport (
    AdvancedIndexingNode,
    BasicIndexingNode,
    PermutationNode,
)
from dwave.optimization.symbols.collections import ListVariable
from dwave.optimization.symbols.constants import Constant


cdef bool _empty_slice(object slice_) noexcept:
    return slice_.start is None and slice_.stop is None and slice_.step is None


cdef class AdvancedIndexing(ArraySymbol):
    """Advanced indexing."""
    def __init__(self, ArraySymbol array, *indices):
        cdef _Graph model = array.model

        cdef vector[AdvancedIndexingNode.array_or_slice] cppindices

        cdef ArraySymbol array_index
        for index in indices:
            if isinstance(index, slice):
                if index != slice(None):
                    raise ValueError("AdvancedIndexing can only parse empty slices")

                cppindices.emplace_back(Slice())
            else:
                array_index = index
                if array_index.model is not model:
                    raise ValueError("mismatched parent models")

                cppindices.emplace_back(array_index.array_ptr)

        self.ptr = model._graph.emplace_node[AdvancedIndexingNode](array.array_ptr, cppindices)

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
                    holds_alternative["ArrayNode*"](self.ptr.indices()[0]) and
                    get["ArrayNode*"](self.ptr.indices()[0]) == (<ArraySymbol>i1).array_ptr and
                    holds_alternative[Slice](self.ptr.indices()[1])):

                return Permutation(array, i1)

            # check the [:, x][x, :] case
            if (isinstance(i1, slice) and _empty_slice(i1) and
                    isinstance(i0, ArraySymbol) and
                    holds_alternative["ArrayNode*"](self.ptr.indices()[1]) and
                    get["ArrayNode*"](self.ptr.indices()[1]) == (<ArraySymbol>i0).array_ptr and
                    holds_alternative[Slice](self.ptr.indices()[0])):

                return Permutation(array, i0)

        return super().__getitem__(index)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef AdvancedIndexingNode* ptr = dynamic_cast_ptr[AdvancedIndexingNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")

        cdef AdvancedIndexing sym = AdvancedIndexing.__new__(AdvancedIndexing)
        sym.ptr = ptr
        sym.initialize_arraynode(symbol.model, ptr)
        return sym

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        cdef Node* ptr

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

        cdef ArrayNode* ptr
        for variant in self.ptr.indices():
            if holds_alternative["ArrayNode*"](variant):
                ptr = get["ArrayNode*"](variant)
                indices.append(symbol_from_ptr(self.model, ptr).topological_index())
            elif holds_alternative[Slice](variant):
                indices.append((0, 0, 0))
            else:
                raise RuntimeError

        zf.writestr(directory + "indices.json", encoder.encode(indices))

    cdef AdvancedIndexingNode* ptr

_register(AdvancedIndexing, typeid(AdvancedIndexingNode))


cdef class BasicIndexing(ArraySymbol):
    """Basic indexing."""
    def __init__(self, ArraySymbol array, *indices):

        cdef _Graph model = array.model

        cdef vector[BasicIndexingNode.slice_or_int] cppindices
        for index in indices:
            if isinstance(index, slice):
                cppindices.emplace_back(BasicIndexing.cppslice(index))
            else:
                cppindices.emplace_back(<Py_ssize_t>(index))

        self.ptr = model._graph.emplace_node[BasicIndexingNode](array.array_ptr, cppindices)

        self.initialize_arraynode(model, self.ptr)

    @staticmethod
    cdef Slice cppslice(object index):
        """Create a Slice from a Python slice object."""
        cdef optional[Py_ssize_t] start
        cdef optional[Py_ssize_t] stop
        cdef optional[Py_ssize_t] step

        if index.start is not None:
            start = <Py_ssize_t>(index.start)
        if index.stop is not None:
            stop = <Py_ssize_t>(index.stop)
        if index.step is not None:
            step = <Py_ssize_t>(index.step)

        return Slice(start, stop, step)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef BasicIndexingNode* ptr = dynamic_cast_ptr[BasicIndexingNode](symbol.node_ptr)
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
        cdef Slice cppslice
        cdef Py_ssize_t index

        # ok, lets iterate
        for variant in self.ptr.infer_indices():
            if holds_alternative[Slice](variant):
                cppslice = get[Slice](variant)
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

    cdef BasicIndexingNode* ptr

_register(BasicIndexing, typeid(BasicIndexingNode))


cdef class Permutation(ArraySymbol):
    """Permutation of the elements of a symbol."""
    def __init__(self, ArraySymbol array, ArraySymbol x):
        # todo: Loosen the types accepted. But this Cython code doesn't yet have
        # the type heirarchy needed so for how we specify explicitly
        if not isinstance(array, Constant):
            raise TypeError("array must be a Constant")
        if not isinstance(x, ListVariable):
            raise TypeError("x must be a ListVariable")

        if array.model is not x.model:
            raise ValueError("array and x do not share the same underlying model")

        cdef PermutationNode* ptr = array.model._graph.emplace_node[PermutationNode](
            array.array_ptr, x.array_ptr)
        self.initialize_arraynode(array.model, ptr)

_register(Permutation, typeid(PermutationNode))

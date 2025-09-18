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

import collections.abc
import json

import numpy as np

from cython.operator cimport typeid
from libcpp cimport bool
from libcpp.utility cimport move
from libcpp.vector cimport vector

from dwave.optimization._model cimport _Graph, _register, ArraySymbol, Symbol
from dwave.optimization.libcpp cimport dynamic_cast_ptr
from dwave.optimization.libcpp.array cimport SizeInfo
from dwave.optimization.libcpp.nodes.collections cimport (
    DisjointBitSetNode,
    DisjointBitSetsNode,
    DisjointListNode,
    DisjointListsNode,
    ListNode,
    SetNode,
)
from dwave.optimization.states cimport States


cdef class DisjointBitSet(ArraySymbol):
    """Disjoint-sets successor symbol.

    See Also:
        :meth:`~dwave.optimization.model.Model.disjoint_bit_sets`: equivalent method.
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
            self.ptr = model._graph.emplace_node[DisjointBitSetNode](parent.ptr)
        else:
            # Already been added to the model, so grab the pointer from the parent's
            # successors
            self.ptr = dynamic_cast_ptr[DisjointBitSetNode](
                parent.ptr.successors()[set_index].ptr
            )

        self.initialize_arraynode(model, self.ptr)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef DisjointBitSetNode* ptr = dynamic_cast_ptr[DisjointBitSetNode](symbol.node_ptr)
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
    cdef DisjointBitSetNode* ptr

_register(DisjointBitSet, typeid(DisjointBitSetNode))


cdef class DisjointBitSets(Symbol):
    """Disjoint-sets decision-variable symbol.

    See Also:
        :meth:`~dwave.optimization.model.Model.disjoint_bit_sets`: equivalent method.
    """
    def __init__(
        self, _Graph model, Py_ssize_t primary_set_size, Py_ssize_t num_disjoint_sets
    ):
        # Get an observing pointer to the node
        self.ptr = model._graph.emplace_node[DisjointBitSetsNode](
            primary_set_size, num_disjoint_sets
        )

        self.initialize_node(model, self.ptr)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef DisjointBitSetsNode* ptr = dynamic_cast_ptr[DisjointBitSetsNode](symbol.node_ptr)
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
    cdef DisjointBitSetsNode* ptr

_register(DisjointBitSets, typeid(DisjointBitSetsNode))


cdef class DisjointList(ArraySymbol):
    """Disjoint-lists successor symbol.

    See Also:
        :meth:`~dwave.optimization.model.Model.disjoint_lists`: associated method.
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
            self.ptr = model._graph.emplace_node[DisjointListNode](parent.ptr)
        else:
            # Already been added to the model, so grab the pointer from the parent's
            # successors
            self.ptr = dynamic_cast_ptr[DisjointListNode](
                parent.ptr.successors()[list_index].ptr
            )

        self.initialize_arraynode(model, self.ptr)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef DisjointListNode* ptr = dynamic_cast_ptr[DisjointListNode](symbol.node_ptr)
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
    cdef DisjointListNode* ptr

_register(DisjointList, typeid(DisjointListNode))


cdef class DisjointLists(Symbol):
    """Disjoint-lists decision-variable symbol.

    See Also:
        :meth:`~dwave.optimization.model.Model.disjoint_lists`: equivalent method.
    """
    def __init__(
        self, _Graph model, Py_ssize_t primary_set_size, Py_ssize_t num_disjoint_lists
    ):
        # Get an observing pointer to the node
        self.ptr = model._graph.emplace_node[DisjointListsNode](
            primary_set_size, num_disjoint_lists
        )

        self.initialize_node(model, self.ptr)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef DisjointListsNode* ptr = dynamic_cast_ptr[DisjointListsNode](symbol.node_ptr)
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
        
        Examples:
            This example sets the state of a disjoint-lists symbol. You can
            inspect the state of each list individually.
            
            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> lists_symbol, lists_array = model.disjoint_lists(
            ...     primary_set_size=5,
            ...     num_disjoint_lists=3
            ... )
            >>> with model.lock():
            ...     model.states.resize(1)
            ...     lists_symbol.set_state(0, [[0, 1, 2, 3], [4], []])
            ...     for index, disjoint_list in enumerate(lists_array):
            ...         print(f"DisjointList {index}:")
            ...         print(disjoint_list.state(0))
            DisjointList 0:
            [0. 1. 2. 3.]
            DisjointList 1:
            [4.]
            DisjointList 2:
            []
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
    cdef DisjointListsNode* ptr

_register(DisjointLists, typeid(DisjointListsNode))


cdef class ListVariable(ArraySymbol):
    """List decision-variable symbol.

    See Also:
        :meth:`~dwave.optimization.model.Model.list`: equivalent method.
    """
    def __init__(self, _Graph model, Py_ssize_t n):
        # Get an observing pointer to the node
        self.ptr = model._graph.emplace_node[ListNode](n)

        self.initialize_arraynode(model, self.ptr)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef ListNode* ptr = dynamic_cast_ptr[ListNode](symbol.node_ptr)
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
        cdef SizeInfo sizeinfo = self.ptr.sizeinfo()

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
    cdef ListNode* ptr

_register(ListVariable, typeid(ListNode))


cdef class SetVariable(ArraySymbol):
    """Set decision-variable symbol.

    A set variable's possible states are the subsets of ``range(n)``.

    See Also:
        :meth:`~dwave.optimization.model.Model.set`: equivalent method.
    """
    def __init__(self, _Graph model, Py_ssize_t n, Py_ssize_t min_size, Py_ssize_t max_size):
        self.ptr = model._graph.emplace_node[SetNode](n, min_size, max_size)
        self.initialize_arraynode(model, self.ptr)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef SetNode* ptr = dynamic_cast_ptr[SetNode](symbol.node_ptr)
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

        cdef SizeInfo sizeinfo = self.ptr.sizeinfo()

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
    cdef SetNode* ptr

_register(SetVariable, typeid(SetNode))

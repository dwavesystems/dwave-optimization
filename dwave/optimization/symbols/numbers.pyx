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

import numpy as np

from cython.operator cimport typeid
from libcpp.optional cimport nullopt
from libcpp.utility cimport move
from libcpp.vector cimport vector

from dwave.optimization._model cimport _Graph, _register, ArraySymbol, Symbol
from dwave.optimization._utilities cimport as_cppshape
from dwave.optimization.libcpp cimport dynamic_cast_ptr
from dwave.optimization.libcpp.nodes.numbers cimport (
    BinaryNode,
    IntegerNode,
)
from dwave.optimization.states cimport States


cdef class BinaryVariable(ArraySymbol):
    """Binary decision-variable symbol.

    See also:
        :meth:`~dwave.optimization.model.Model.binary`: equivalent method.
    """
    def __init__(self, _Graph model, shape=None):
        # Get an observing pointer to the node
        cdef vector[Py_ssize_t] vshape = as_cppshape(tuple() if shape is None else shape)

        self.ptr = model._graph.emplace_node[BinaryNode](vshape)

        self.initialize_arraynode(model, self.ptr)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef BinaryNode* ptr = dynamic_cast_ptr[BinaryNode](symbol.node_ptr)
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
    cdef BinaryNode* ptr

_register(BinaryVariable, typeid(BinaryNode))


cdef class IntegerVariable(ArraySymbol):
    """Integer decision-variable symbol.

    See Also:
        :meth:`~dwave.optimization.model.Model.integer`: equivalent method.
    """
    def __init__(self, _Graph model, shape=None, lower_bound=None, upper_bound=None):
        cdef vector[Py_ssize_t] vshape = as_cppshape(tuple() if shape is None else shape)

        if lower_bound is None and upper_bound is None:
            self.ptr = model._graph.emplace_node[IntegerNode](vshape)
        elif lower_bound is None:
            self.ptr = model._graph.emplace_node[IntegerNode](vshape, nullopt, <double>upper_bound)
        elif upper_bound is None:
            self.ptr = model._graph.emplace_node[IntegerNode](vshape, <double>lower_bound)
        else:
            self.ptr = model._graph.emplace_node[IntegerNode](vshape, <double>lower_bound, <double>upper_bound)

        self.initialize_arraynode(model, self.ptr)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef IntegerNode* ptr = dynamic_cast_ptr[IntegerNode](symbol.node_ptr)
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
    cdef IntegerNode* ptr

_register(IntegerVariable, typeid(IntegerNode))

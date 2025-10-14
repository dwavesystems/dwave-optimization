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
from libcpp.optional cimport nullopt, optional
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
    def __init__(self, _Graph model, shape=None, lower_bound=None, upper_bound=None):
        cdef vector[Py_ssize_t] cppshape = as_cppshape(
            tuple() if shape is None else shape
        )

        cdef optional[vector[double]] cpplower_bound = nullopt
        cdef optional[vector[double]] cppupper_bound = nullopt
        cdef const double[:] mem

        if lower_bound is not None:
            # Allow users to input doubles
            lower_bound_arr = np.asarray_chkfinite(lower_bound,
                                                   dtype=np.double, order="C")
            # For lower bounds, round up
            lower_bound_arr = np.ceil(lower_bound_arr).astype(np.double)
            if (lower_bound_arr.ndim == 0) or (lower_bound_arr.shape == cppshape):
                if lower_bound_arr.size > 0:
                    mem = lower_bound_arr.ravel()
                    cpplower_bound.emplace(&mem[0], (&mem[-1]) + 1)
            else:
                raise ValueError("lower_bound should be None, scalar, or the same shape")

        if upper_bound is not None:
            # Allow users to input doubles
            upper_bound_arr = np.asarray_chkfinite(upper_bound,
                                                   dtype=np.double, order="C")
            # For upper bounds, round down
            upper_bound_arr = np.floor(upper_bound_arr).astype(np.double)
            if (upper_bound_arr.ndim == 0) or (upper_bound_arr.shape == cppshape):
                if upper_bound_arr.size > 0:
                    mem = upper_bound_arr.ravel()
                    cppupper_bound.emplace(&mem[0], (&mem[-1]) + 1)
            else:
                raise ValueError("upper bound should be None, scalar, or the same shape")

        self.ptr = model._graph.emplace_node[BinaryNode](
            cppshape, cpplower_bound, cppupper_bound
        )
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
        if predecessors:
            raise ValueError(f"{cls.__name__} cannot have predecessors")

        with zf.open(directory + "shape.json", "r") as f:
            shape_info = json.load(f)

        # needs to be compatible with older versions
        try:
            info = zf.getinfo(directory + "lower_bound.npy")
        except KeyError:
            lower_bound = None
        else:
            with zf.open(info, "r") as f:
                lower_bound = np.load(f, allow_pickle=False)

        # needs to be compatible with older versions
        try:
            info = zf.getinfo(directory + "upper_bound.npy")
        except KeyError:
            upper_bound = None
        else:
            with zf.open(info, "r") as f:
                upper_bound = np.load(f, allow_pickle=False)

        return BinaryVariable(model,
                              shape=shape_info["shape"],
                              lower_bound=lower_bound,
                              upper_bound=upper_bound,
                              )

    def _into_zipfile(self, zf, directory):
        shape_info = dict(shape=self.shape())
        encoder = json.JSONEncoder(separators=(',', ':'))
        zf.writestr(directory + "shape.json", encoder.encode(shape_info))

        lower_bound = self.lower_bound()
        # if all values in the array are the same, simply save a scalar
        if lower_bound.size and (lower_bound == lower_bound.flat[0]).all():
            lower_bound = lower_bound.flat[0]
        # NumPy serialization is overkill but it's type-safe
        with zf.open(directory + "lower_bound.npy", mode="w", force_zip64=True) as f:
            np.save(f, lower_bound, allow_pickle=False)

        upper_bound = self.upper_bound()
        # if all values in the array are the same, simply save a scalar
        if upper_bound.size and (upper_bound == upper_bound.flat[0]).all():
            upper_bound = upper_bound.flat[0]
        # NumPy serialization is overkill but it's type-safe
        with zf.open(directory + "upper_bound.npy", mode="w", force_zip64=True) as f:
            np.save(f, upper_bound, allow_pickle=False)

    def lower_bound(self):
        """Lower bound(s) of Binary symbol."""
        try:
            return np.asarray(self.ptr.lower_bound())
        except IndexError:
            pass
        return np.asarray([self.ptr.lower_bound(i) for i in range(self.size())]).reshape(self.shape())

    def set_state(self, Py_ssize_t index, state):
        r"""Set the state of the binary symbol.

        The given state must be binary array with the same shape as the symbol.

        Examples:
            This example sets two states for a :math:`2 \times 3`-sized
            binary symbol.

            >>> from dwave.optimization.model import Model
            >>> import numpy as np
            ...
            >>> model = Model()
            >>> x = model.binary((2, 3))
            >>> model.states.resize(2)
            >>> x.set_state(0, [[True, True, False], [False, True, False]])
            >>> print(np.equal(x.state(0), [[True, True, False], [False, True, False]]).all())
            True
            >>> x.set_state(1, [[False, True, False], [False, True, False]])
            >>> print(np.equal(x.state(1), [[False, True, False], [False, True, False]]).all())
            True
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
        """Upper bound(s) of Binary symbol."""
        try:
            return np.asarray(self.ptr.upper_bound())
        except IndexError:
            pass
        return np.asarray([self.ptr.upper_bound(i) for i in range(self.size())]).reshape(self.shape())

    # An observing pointer to the C++ BinaryNode
    cdef BinaryNode* ptr

_register(BinaryVariable, typeid(BinaryNode))


cdef class IntegerVariable(ArraySymbol):
    """Integer decision-variable symbol.

    See Also:
        :meth:`~dwave.optimization.model.Model.integer`: equivalent method.
    """
    def __init__(self, _Graph model, shape=None, lower_bound=None, upper_bound=None):
        cdef vector[Py_ssize_t] cppshape = as_cppshape(
            tuple() if shape is None else shape
        )

        cdef optional[vector[double]] cpplower_bound = nullopt
        cdef optional[vector[double]] cppupper_bound = nullopt
        cdef const double[:] mem

        if lower_bound is not None:
            # Allow users to input doubles
            lower_bound_arr = np.asarray_chkfinite(lower_bound,
                                                   dtype=np.double, order="C")
            # For lower bounds, round up
            lower_bound_arr = np.ceil(lower_bound_arr).astype(np.double)
            if (lower_bound_arr.ndim == 0) or (lower_bound_arr.shape == cppshape):
                if lower_bound_arr.size > 0:
                    mem = lower_bound_arr.ravel()
                    cpplower_bound.emplace(&mem[0], (&mem[-1]) + 1)
            else:
                raise ValueError("lower_bound should be None, scalar, or the same shape")

        if upper_bound is not None:
            # Allow users to input doubles
            upper_bound_arr = np.asarray_chkfinite(upper_bound,
                                                   dtype=np.double, order="C")
            # For upper bounds, round down
            upper_bound_arr = np.floor(upper_bound_arr).astype(np.double)
            if (upper_bound_arr.ndim == 0) or (upper_bound_arr.shape == cppshape):
                if upper_bound_arr.size > 0:
                    mem = upper_bound_arr.ravel()
                    cppupper_bound.emplace(&mem[0], (&mem[-1]) + 1)
            else:
                raise ValueError("upper bound should be None, scalar, or the same shape")

        self.ptr = model._graph.emplace_node[IntegerNode](
            cppshape, cpplower_bound, cppupper_bound
        )
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

        # needs to be compatible with older versions
        try:
            info = zf.getinfo(directory + "lower_bound.npy")
        except KeyError:
            lower_bound = shape_info["lb"]
        else:
            with zf.open(info, "r") as f:
                lower_bound = np.load(f, allow_pickle=False)

        # needs to be compatible with older versions
        try:
            info = zf.getinfo(directory + "upper_bound.npy")
        except KeyError:
            upper_bound = shape_info["ub"]
        else:
            with zf.open(info, "r") as f:
                upper_bound = np.load(f, allow_pickle=False)

        return IntegerVariable(model,
                               shape=shape_info["shape"],
                               lower_bound=lower_bound,
                               upper_bound=upper_bound,
                               )

    def _into_zipfile(self, zf, directory):
        shape_info = dict(shape=self.shape())
        lower_bound = self.lower_bound()
        upper_bound = self.upper_bound()

        # This is for backward compatiblity and should be ignored
        if lower_bound.size == 1 and upper_bound.size == 1:
            shape_info["lb"] = lower_bound.item()
            shape_info["ub"] = upper_bound.item() 

        encoder = json.JSONEncoder(separators=(',', ':'))
        zf.writestr(directory + "shape.json", encoder.encode(shape_info))

        # if all values in the array are the same, simply save a scalar
        if lower_bound.size and (lower_bound == lower_bound.flat[0]).all():
            lower_bound = lower_bound.flat[0]
        # NumPy serialization is overkill but it's type-safe
        with zf.open(directory + "lower_bound.npy", mode="w", force_zip64=True) as f:
            np.save(f, lower_bound, allow_pickle=False)

        # if all values in the array are the same, simply save a scalar
        if upper_bound.size and (upper_bound == upper_bound.flat[0]).all():
            upper_bound = upper_bound.flat[0]
        # NumPy serialization is overkill but it's type-safe
        with zf.open(directory + "upper_bound.npy", mode="w", force_zip64=True) as f:
            np.save(f, upper_bound, allow_pickle=False)

    def lower_bound(self):
        """Lower bound(s) of Integer symbol."""
        try:
            return np.asarray(self.ptr.lower_bound())
        except IndexError:
            pass
        return np.asarray([self.ptr.lower_bound(i) for i in range(self.size())]).reshape(self.shape())

    def set_state(self, Py_ssize_t index, state):
        r"""Set the state of the integer symbol.

        The given state must be an integer array with the same shape as the
        symbol.

        Examples:
            This example successfully sets one state for a :math:`2 \times
            2`-sized integer symbol.

            >>> from dwave.optimization.model import Model
            >>> import numpy as np
            ...
            >>> model = Model()
            >>> x = model.integer((2, 2), lower_bound=2, upper_bound=[[3,4], [2, 5]])
            >>> model.states.resize(1)
            >>> x.set_state(0, [[3, 4], [2, 3]])
            >>> print(np.equal(x.state(0), [[3, 4], [2, 3]]).all())
            True
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
        """Upper bound(s) of Integer symbol."""
        try:
            return np.asarray(self.ptr.upper_bound())
        except IndexError:
            pass
        return np.asarray([self.ptr.upper_bound(i) for i in range(self.size())]).reshape(self.shape())

    # An observing pointer to the C++ IntegerNode
    cdef IntegerNode* ptr

_register(IntegerVariable, typeid(IntegerNode))

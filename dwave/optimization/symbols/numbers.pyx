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

    Args:
        model: The model.
        shape (optional): Shape of the binary array to create.
        lower_bound (optional): Lower bound(s) for the symbol. Can be
            scalar (one bound for all variables) or an array (one bound for
            each variable). Non-boolean values are rounded up to the domain
            [0,1]. If None, the default value of 0 is used.
        upper_bound (optional): Upper bound(s) for the symbol. Can be
            scalar (one bound for all variables) or an array (one bound for
            each variable). Non-boolean values are rounded down to the domain
            [0,1]. If None, the default value of 1 is used.

    Returns:
        A binary symbol.

    Examples:
        This example adds a :math:`20 \time 30`-sized binary variable to a model.

        >>> from dwave.optimization.model import Model
        >>> from dwave.optimization.symbols import BinaryVariable
        >>> model = Model()
        >>> x = BinaryVariable(model, (20, 30))
        >>> type(x)
        <class 'dwave.optimization.symbols.numbers.BinaryVariable'>

        This example adds a :math:`5`-sized binary symbol and index-wise
        bounds to a model.

        >>> from dwave.optimization.model import Model
        >>> from dwave.optimization.symbols import BinaryVariable
        >>> model = Model()
        >>> b = BinaryVariable(model, 5, lower_bound=[-1, 0, 1, 0, 1], 
        ...                    upper_bound=[1, 0, 1, 1, 1])
        >>> [0, 0, 1, 0, 1] == [b.lower_bound(i) for i in range(b.size())]
        True
        >>> [1, 0, 1, 1, 1] == [b.upper_bound(i) for i in range(b.size())]
        True

        This example adds a :math:`2`-sized binary symbol with a scalar lower
        bound and index-wise upper bounds to a model.

        >>> from dwave.optimization.model import Model
        >>> from dwave.optimization.symbols import BinaryVariable
        >>> model = Model()
        >>> b = BinaryVariable(model, 2, lower_bound=-1.1, upper_bound=[1.1, 0.9])
        >>> [0, 0] == [b.lower_bound(i) for i in range(b.size())]
        True
        >>> [1, 0] == [b.upper_bound(i) for i in range(b.size())]
        True

    See also:
        :meth:`~dwave.optimization.model.Model.binary`: equivalent method.

    .. versionchanged:: 0.6.7
        Beginning in version 0.6.7, user-defined bounds and index-wise bounds
        are supported.
    """
    def __init__(self, _Graph model, shape=None, lower_bound=None, upper_bound=None):
        cdef vector[Py_ssize_t] vshape = as_cppshape(
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
            if (lower_bound_arr.ndim == 0) or (lower_bound_arr.shape == vshape):
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
            if (upper_bound_arr.ndim == 0) or (upper_bound_arr.shape == vshape):
                mem = upper_bound_arr.ravel()
                cppupper_bound.emplace(&mem[0], (&mem[-1]) + 1)
            else:
                raise ValueError("upper bound should be None, scalar, or the same shape")

        self.ptr = model._graph.emplace_node[BinaryNode](
            vshape, cpplower_bound, cppupper_bound
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
                if (lower_bound.size == 1):
                    lower_bound = lower_bound[0]

        # needs to be compatible with older versions
        try:
            info = zf.getinfo(directory + "upper_bound.npy")
        except KeyError:
            upper_bound = None
        else:
            with zf.open(info, "r") as f:
                upper_bound = np.load(f, allow_pickle=False)
                if (upper_bound.size == 1):
                    upper_bound = upper_bound[0]

        return BinaryVariable(model,
                              shape=shape_info["shape"],
                              lower_bound=lower_bound,
                              upper_bound=upper_bound,
                              )

    def _into_zipfile(self, zf, directory):
        shape_info = dict(shape=self.shape())
        encoder = json.JSONEncoder(separators=(',', ':'))
        zf.writestr(directory + "shape.json", encoder.encode(shape_info))

        lower_bound = np.array([self.lower_bound(i) for i in range(self.size())], dtype=np.double)
        # if all values in the array are the same, simply save a scalar
        if (np.all(lower_bound == lower_bound[0])):
            lower_bound = lower_bound[:1]
        else:
            lower_bound = lower_bound.reshape(self.shape())
        # NumPy serialization is overkill but it's type-safe
        with zf.open(directory + "lower_bound.npy", mode="w", force_zip64=True) as f:
            np.save(f, lower_bound, allow_pickle=False)

        upper_bound = np.array([self.upper_bound(i) for i in range(self.size())], dtype=np.double)
        # if all values in the array are the same, simply save a scalar
        if (np.all(upper_bound == upper_bound[0])):
            upper_bound = upper_bound[:1]
        else:
            upper_bound = upper_bound.reshape(self.shape())
        # NumPy serialization is overkill but it's type-safe
        with zf.open(directory + "upper_bound.npy", mode="w", force_zip64=True) as f:
            np.save(f, upper_bound, allow_pickle=False)

    def lower_bound(self, Py_ssize_t index):
        """The lowest value allowed for the binary symbol at the given index."""
        return int(self.ptr.lower_bound(index))

    def set_state(self, Py_ssize_t index, state):
        r"""Set the state of the binary symbol.

        The given state must be binary array with the same shape as the symbol.

        Examples:
            This example sets two states for a :math:`2 \times 3`-sized
            binary symbol.

            >>> from dwave.optimization.model import Model
            >>> from dwave.optimization.symbols import BinaryVariable
            >>> import numpy as np
            ...
            >>> model = Model()
            >>> x = BinaryVariable(model, (2, 3))
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

    def upper_bound(self, Py_ssize_t index):
        """The highest value allowed for the binary symbol at the given index."""
        return int(self.ptr.upper_bound(index))

    # An observing pointer to the C++ BinaryNode
    cdef BinaryNode* ptr

_register(BinaryVariable, typeid(BinaryNode))


cdef class IntegerVariable(ArraySymbol):
    """Integer decision-variable symbol.

    Args:
        model: The model.
        shape (optional): Shape of the integer array to create.
        lower_bound (optional): Lower bound(s) for the symbol. Can be
            scalar (one bound for all variables) or an array (one bound for
            each variable). Non-integer values are rounded up. If None, the
            default value is used.
        upper_bound (optional): Upper bound(s) for the symbol. Can be
            scalar (one bound for all variables) or an array (one bound for
            each variable). Non-integer values are down up. If None, the
            default value is used.

    Returns:
        An integer symbol.

    Examples:
        This example adds a :math:`25`-sized integer symbol with a scalar lower
        bound and index-wise upper bounds to a model.

        >>> from dwave.optimization.model import Model
        >>> from dwave.optimization.symbols import IntegerVariable
        >>> model = Model()
        >>> i = IntegerVariable(model, 25, upper_bound=100)
        >>> type(i)
        <class 'dwave.optimization.symbols.numbers.IntegerVariable'>

        This example adds a :math:`5`-sized integer symbol with index-wise
        bounds to a model.

        >>> from dwave.optimization.model import Model
        >>> from dwave.optimization.symbols import IntegerVariable
        >>> model = Model()
        >>> i = IntegerVariable(model, 5, lower_bound=[-1, 0, 3, 0, 2], 
        ...                     upper_bound=[1, 2, 3, 4, 5])
        >>> [-1, 0, 3, 0, 2] == [i.lower_bound(j) for j in range(i.size())]
        True
        >>> [1, 2, 3, 4, 5] == [i.upper_bound(j) for j in range(i.size())]
        True

        This example adds a :math:`2`-sized integer symbol with a scalar lower
        bound and index-wise upper bounds to a model.

        >>> from dwave.optimization.model import Model
        >>> from dwave.optimization.symbols import IntegerVariable
        >>> model = Model()
        >>> i = IntegerVariable(model, 2, lower_bound=-1.1, upper_bound=[1.1, 2.9])
        >>> [-1, -1] == [i.lower_bound(j) for j in range(i.size())]
        True
        >>> [1, 2] == [i.upper_bound(j) for j in range(i.size())]
        True

    See Also:
        :meth:`~dwave.optimization.model.Model.integer`: equivalent method.

    .. versionchanged:: 0.6.7
        Beginning in version 0.6.7, user-defined index-wise bounds are
        supported.
    """
    def __init__(self, _Graph model, shape=None, lower_bound=None, upper_bound=None):
        cdef vector[Py_ssize_t] vshape = as_cppshape(
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
            if (lower_bound_arr.ndim == 0) or (lower_bound_arr.shape == vshape):
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
            if (upper_bound_arr.ndim == 0) or (upper_bound_arr.shape == vshape):
                mem = upper_bound_arr.ravel()
                cppupper_bound.emplace(&mem[0], (&mem[-1]) + 1)
            else:
                raise ValueError("upper bound should be None, scalar, or the same shape")

        self.ptr = model._graph.emplace_node[IntegerNode](
            vshape, cpplower_bound, cppupper_bound
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
                if (lower_bound.size == 1):
                    lower_bound = lower_bound[0]

        # needs to be compatible with older versions
        try:
            info = zf.getinfo(directory + "upper_bound.npy")
        except KeyError:
            upper_bound = shape_info["ub"]
        else:
            with zf.open(info, "r") as f:
                upper_bound = np.load(f, allow_pickle=False)
                if (upper_bound.size == 1):
                    upper_bound = upper_bound[0]

        return IntegerVariable(model,
                               shape=shape_info["shape"],
                               lower_bound=lower_bound,
                               upper_bound=upper_bound,
                               )

    def _into_zipfile(self, zf, directory):
        shape_info = dict(shape=self.shape())
        encoder = json.JSONEncoder(separators=(',', ':'))
        zf.writestr(directory + "shape.json", encoder.encode(shape_info))

        lower_bound = np.array([self.lower_bound(i) for i in range(self.size())], dtype=np.double)
        # if all values in the array are the same, simply save a scalar
        if (np.all(lower_bound == lower_bound[0])):
            lower_bound = lower_bound[:1]
        else:
            lower_bound = lower_bound.reshape(self.shape())
        # NumPy serialization is overkill but it's type-safe
        with zf.open(directory + "lower_bound.npy", mode="w", force_zip64=True) as f:
            np.save(f, lower_bound, allow_pickle=False)

        upper_bound = np.array([self.upper_bound(i) for i in range(self.size())], dtype=np.double)
        # if all values in the array are the same, simply save a scalar
        if (np.all(upper_bound == upper_bound[0])):
            upper_bound = upper_bound[:1]
        else:
            upper_bound = upper_bound.reshape(self.shape())
        # NumPy serialization is overkill but it's type-safe
        with zf.open(directory + "upper_bound.npy", mode="w", force_zip64=True) as f:
            np.save(f, upper_bound, allow_pickle=False)

    def lower_bound(self, Py_ssize_t index):
        """The lowest value allowed for the integer symbol at the given index."""
        return int(self.ptr.lower_bound(index))

    def set_state(self, Py_ssize_t index, state):
        r"""Set the state of the integer symbol.

        The given state must be an integer array with the same shape as the
        symbol.

        Examples:
            This example successfully sets one state for a :math:`2 \times
            2`-sized integer symbol.

            >>> from dwave.optimization.model import Model
            >>> from dwave.optimization.symbols import IntegerVariable
            >>> import numpy as np
            ...
            >>> model = Model()
            >>> x = IntegerVariable(model, (2, 2), lower_bound=2, upper_bound=[[3,4], [2, 5]])
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

    def upper_bound(self, Py_ssize_t index):
        """The highest value allowed for the integer symbol at the given index."""
        return int(self.ptr.upper_bound(index))

    # An observing pointer to the C++ IntegerNode
    cdef IntegerNode* ptr

_register(IntegerVariable, typeid(IntegerNode))

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
from libcpp cimport bool
from libcpp.optional cimport nullopt, optional
from libcpp.span cimport span
from libcpp.vector cimport vector

from dwave.optimization._model cimport _Graph, _register, ArraySymbol, Symbol
from dwave.optimization._utilities cimport as_cppshape, as_span
from dwave.optimization.libcpp cimport dynamic_cast_ptr
from dwave.optimization.libcpp.nodes.inputs cimport InputNode
from dwave.optimization.states cimport States



cdef class Input(ArraySymbol):
    """An input symbol that acts as a placeholder in a model.

    See the :func:`~dwave.optimization.model.Model.input` function for
    instantiation and additional information.
    """

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
        self.ptr = cygraph._graph.emplace_node[InputNode](
            vshape,
            optional[double](nullopt) if lower_bound is None else optional[double](<double>lower_bound),
            optional[double](nullopt) if upper_bound is None else optional[double](<double>upper_bound),
            optional[bool](nullopt) if integral is None else optional[bool](<bool>integral),
        )

        self.initialize_arraynode(model, self.ptr)

    def integral(self):
        """Returns True if the symbol always returns integers."""
        return self.ptr.integral()

    def lower_bound(self):
        """Lowest value allowed in the symbol."""
        return self.ptr.min()

    def set_state(self, Py_ssize_t index, state):
        """Set the state of the symbol.

        Args:
            index:
                Index of the state to set.
            state:
                Assignment of values for the state. The specified state must
                have the same shape as the symbol.

        Examples:
            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> y = model.input(shape=(2, 3), upper_bound=10, integral=False)
            >>> z = y.max()
            >>> with model.lock():
            ...     model.states.resize(1)
            ...     y.set_state(0, [[1.5, 2, 9], [-12, 4, 10]])
            ...     print(z.state(0))
            10.0
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
        """Largest value allowed in the symbol."""
        return self.ptr.max()

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef InputNode* ptr = dynamic_cast_ptr[InputNode](symbol.node_ptr)
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

    cdef InputNode* ptr

_register(Input, typeid(InputNode))

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

from dwave.optimization._model cimport _Graph, _register, ArraySymbol, Symbol
from dwave.optimization._utilities cimport as_span
from dwave.optimization.libcpp cimport dynamic_cast_ptr
from dwave.optimization.libcpp.graph cimport ArrayNode
from dwave.optimization.libcpp.nodes.lp cimport (
    LinearProgramFeasibleNode,
    LinearProgramNode,
    LinearProgramNodeBase,
    LinearProgramObjectiveValueNode,
    LinearProgramSolutionNode,
)
from dwave.optimization.states cimport States


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

        cdef ArrayNode* c_ptr = c.array_ptr

        cdef ArrayNode* b_lb_ptr = LinearProgram.as_arraynodeptr(model, b_lb)
        cdef ArrayNode* A_ptr = LinearProgram.as_arraynodeptr(model, A)
        cdef ArrayNode* b_ub_ptr = LinearProgram.as_arraynodeptr(model, b_ub)

        cdef ArrayNode* A_eq_ptr = LinearProgram.as_arraynodeptr(model, A_eq)
        cdef ArrayNode* b_eq_ptr = LinearProgram.as_arraynodeptr(model, b_eq)

        cdef ArrayNode* lb_ptr = LinearProgram.as_arraynodeptr(model, lb)
        cdef ArrayNode* ub_ptr = LinearProgram.as_arraynodeptr(model, ub)

        self.ptr = model._graph.emplace_node[LinearProgramNode](
            c_ptr, b_lb_ptr, A_ptr, b_ub_ptr, A_eq_ptr, b_eq_ptr, lb_ptr, ub_ptr)
        self.initialize_node(model, self.ptr)

    @staticmethod
    cdef ArrayNode* as_arraynodeptr(_Graph model, ArraySymbol x) except? NULL:
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
        cdef LinearProgramNode* ptr = dynamic_cast_ptr[LinearProgramNode](symbol.node_ptr)
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

    cdef LinearProgramNode* ptr

_register(LinearProgram, typeid(LinearProgramNode))


cdef class LinearProgramFeasible(ArraySymbol):
    """Return whether the parent LP symbol's current solution is feasible.

    See Also:
        :func:`~dwave.optimization.mathematical.linprog`

    .. versionadded:: 0.6.0
    """

    def __init__(self, Symbol lp):
        cdef _Graph model = lp.model

        cdef LinearProgramNodeBase* base_ptr = dynamic_cast_ptr[LinearProgramNodeBase](lp.node_ptr)
        if not base_ptr:
            raise TypeError("Provided symbol must be derived from the LP base class")

        cdef LinearProgramFeasibleNode* ptr = model._graph.emplace_node[LinearProgramFeasibleNode](base_ptr)
        self.initialize_arraynode(model, ptr)

_register(LinearProgramFeasible, typeid(LinearProgramFeasibleNode))


cdef class LinearProgramObjectiveValue(ArraySymbol):
    """Return the objective value of the parent LP symbol's current solution.

    See Also:
        :func:`~dwave.optimization.mathematical.linprog`

    .. versionadded:: 0.6.0
    """

    def __init__(self, Symbol lp):
        cdef _Graph model = lp.model

        cdef LinearProgramNodeBase* base_ptr = dynamic_cast_ptr[LinearProgramNodeBase](lp.node_ptr)
        if not base_ptr:
            raise TypeError("Provided symbol must be derived from the LP base class")

        cdef LinearProgramObjectiveValueNode* ptr = model._graph.emplace_node[LinearProgramObjectiveValueNode](base_ptr)
        self.initialize_arraynode(model, ptr)

_register(LinearProgramObjectiveValue, typeid(LinearProgramObjectiveValueNode))


cdef class LinearProgramSolution(ArraySymbol):
    """Return the current solution of the parent LP symbol as an array.

    See Also:
        :func:`~dwave.optimization.mathematical.linprog`

    .. versionadded:: 0.6.0
    """

    def __init__(self, Symbol lp):
        cdef _Graph model = lp.model

        cdef LinearProgramNodeBase* base_ptr = dynamic_cast_ptr[LinearProgramNodeBase](lp.node_ptr)
        if not base_ptr:
            raise TypeError("Provided symbol must be derived from the LP base class")

        cdef LinearProgramSolutionNode* ptr = model._graph.emplace_node[LinearProgramSolutionNode](base_ptr)
        self.initialize_arraynode(model, ptr)

_register(LinearProgramSolution, typeid(LinearProgramSolutionNode))

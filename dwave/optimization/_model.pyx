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

import fractions
import json
import tempfile
import weakref
import zipfile

import numpy as np

from cpython.memoryview cimport PyMemoryView_FromBuffer
from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_New
from cpython.ref cimport PyObject
from cython.operator cimport dereference as deref, preincrement as inc
from cython.operator cimport typeid
from libc.string cimport memset
from libcpp.memory cimport make_shared
from libcpp.span cimport span
from libcpp.typeindex cimport type_index
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport move
from libcpp.vector cimport vector

from dwave.optimization.libcpp cimport dynamic_cast_ptr
from dwave.optimization.libcpp.array cimport broadcast_shapes
from dwave.optimization._mixins import (
    _ArraySymbolMixin,
    _StatesMixin,
    _SymbolMixin,
    ArrayInfo,
    ArraySizeInfo,
)
from dwave.optimization.utilities import _file_object_arg

__all__ = []


# Store a mapping from the type_index of each C++ Node type to the relevant
# Cython class. We don't refcount the PyObject*s pointing to each class because
# the lifespace of this map is identical to that of the type objects.
cdef unordered_map[type_index, PyObject*] _cpp_type_to_python

# Register a mapping between the given Cython class and C++ class.
cdef void _register(object cls, const type_info& typeinfo):
    """Register a Python/Cython symbol to allow it to be created from a pointer
    via `symbol_from_ptr`.
    """
    _cpp_type_to_python[type_index(typeinfo)] = <PyObject*>(cls)


cdef object symbol_from_ptr(_Graph model, Node* node_ptr):
    """Create a Python/Cython symbol from a C++ Node*."""

    # If it's null, either after the cast of just as given, then we can't get a symbol from it
    if not node_ptr:
        raise ValueError("cannot construct a Symbol from the given pointer")

    try:
        cls = <object>_cpp_type_to_python.at(type_index(typeid(deref(node_ptr))))
    except IndexError:
        # IndexError would be returned by .at()
        raise RuntimeError("given pointer cannot be cast to a known node type") from None

    # We'll use a PyCapsule to pass a node pointer through the Python layer so
    # that we can let `cls` determine the type.

    # Even though PyCapsule_New returns a PyObject*, Cython automatically makes
    # it an object (with appropriate refcounting) so that's nice.
    cap = PyCapsule_New(node_ptr, 'Node*', NULL)

    return cls._from_ptr(model, cap)


cdef class _Empty:
    pass


cdef class _Graph:
    """A ``_Graph`` is a class that manages a C++ ``dwave::optimization::Graph``.

    It is not intended for a user to use ``_Graph`` directly. Rather, classes
    may inherit from ``_Graph``.
    """
    def __cinit__(self):
        self._lock_count = 0

        self._owning_ptr = make_shared[Graph]()
        self._graph = self._owning_ptr.get()

    @staticmethod
    cdef _Graph from_shared_ptr(cls, shared_ptr[Graph] ptr):
        """Return a _Graph from a shared_ptr. We would like this to be a classmethod
        but unfortunately Cython does not allow cdef classmethods. So we make cls
        an explicit argument instead.
        """
        cdef _Graph model = cls.__new__(cls)

        model._owning_ptr = ptr
        model._graph = model._owning_ptr.get()

        if model._graph.topologically_sorted():
            model._lock_count += 1

        return model

    def __init__(self, *args, **kwargs):
        # disallow direct construction of _Graphs, they should be constructed
        # via their subclasses.
        raise ValueError("_Graphs cannot be constructed directly")

    def add_constraint(self, ArraySymbol value):
        """Add a constraint to the model.

        For a solution to the model to be :term:`feasible <feasible state>`, the
        :term:`constraint` must be satisfied.

        Args:
            value: Value that must evaluate to True for the state
                of the model to be feasible.

        Returns:
            The symbol associated with the constraint.

        Examples:
            This example adds a single constraint to a model.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> i = model.integer()
            >>> c = model.constant(5)
            >>> constraint_sym = model.add_constraint(i <= c)

            The returned constraint symbol (a
            :class:`~dwave.optimization.symbols.LessEqual` for this example) can
            be assigned, and evaluated for, a model state:

            >>> with model.lock():
            ...     model.states.resize(1)
            ...     i.set_state(0, 1) # Feasible state
            ...     print(constraint_sym.state(0))
            1.0
            >>> with model.lock():
            ...     i.set_state(0, 6) # Infeasible state
            ...     print(constraint_sym.state(0))
            0.0

        See Also:
            :meth:`.feasible`, :meth:`.iter_constraints`,
            :meth:`.num_constraints`, :meth:`.objective`,
        """
        if value is None:
            raise ValueError("value cannot be None")
        # TODO: shall we accept array valued constraints?
        self._graph.add_constraint(value.array_ptr)
        return value

    def decision_state_size(self):
        r"""Estimate the size, in bytes, of a model's decision states.

        For more details, see the :meth:`.state_size()` method. This method
        differs by counting the states of only the decision variables.

        Returns:
            int: Estimated state size of the model's decision variables.

        Examples:
            This example estimates the size of a model's decision state.
            In this example a single value is added to a :math:`5\times4` array.
            The output of the addition is also a :math:`5\times4` array.
            Each element of each array requires :math:`8` bytes to represent
            in memory.
            The total state size is :math:`(5*4 + 1 + 5*4) * 8 = 328` bytes,
            but the decision state size is only :math:`5*4*8 = 160`.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> i = model.integer((5, 4))    # 5x4 array of integers
            >>> c = model.constant(1)        # one scalar value, not a decision
            >>> y = i + c                    # 5x4 array of values, not a decision
            >>> model.state_size()           # (5*4 + 1 + 5*4) * 8 bytes
            328
            >>> model.decision_state_size()  # 5*4*8 bytes
            160

        See also:
            :meth:`Symbol.state_size()` Estimates the size of a symbol’s state

            :meth:`ArraySymbol.state_size()` Estimates the size of an array
            symbol's state

            :meth:`Model.state_size()` Estimates the size of all of a model's
            states
        """
        return sum(sym.state_size() for sym in self.iter_decisions())

    cpdef bool is_locked(self) noexcept:
        """Lock status of the model.

        No new symbols can be added to a locked model.

        Returns:
            bool: True if the model is locked.

        Examples:
            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> model.is_locked()
            False

        See also:
            :meth:`.lock`, :meth:`.unlock`
        """
        return self._lock_count > 0

    def iter_constraints(self):
        """Iterate over all constraints in the model.

        Yields:
            Symbols associated with the constraints.

        Examples:
            This example adds a single constraint to a model and iterates over it.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> i = model.integer()
            >>> c = model.constant(5)
            >>> _ = model.add_constraint(i <= c)
            >>> constraint = next(model.iter_constraints())
            >>> print(type(constraint))
            <class 'dwave.optimization.symbols.binaryop.LessEqual'>

        See Also:
            :meth:`iter_decisions`, :meth:`.iter_symbols`,
            :meth:`.num_constraints`
        """
        for ptr in self._graph.constraints():
            yield symbol_from_ptr(self, ptr)

    def iter_decisions(self):
        """Iterate over all decision variables in the model.

        Yields:
            Decision variables; for example, an
            :class:`~dwave.optimization.symbols.IntegerVariable` symbol.

        Examples:
            This example adds a single decision symbol to a model and iterates over it.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> i = model.integer()
            >>> c = model.constant(5)
            >>> _ = model.add_constraint(i <= c)
            >>> decisions = next(model.iter_decisions())

        See Also:
            :meth:`iter_constraints`, :meth:`iter_symbols`,
            :meth:`.num_decisions`
        """
        for ptr in self._graph.decisions():
            yield symbol_from_ptr(self, ptr)

    def iter_inputs(self):
        """Iterate over all inputs in the model.

        Yields:
            :class:`~dwave.optimization.symbols.Input` symbols.

        Examples:
            This example iterates over a model's inputs.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> i0, i1 = model.input(), model.input()
            >>> c = model.constant(7)
            >>> inputs = list(model.iter_inputs())
            >>> len(inputs)
            2

        .. versionadded:: 0.6.2

        See Also:
            :meth:`iter_decisions`, :meth:`iter_symbols`, :meth:`.num_inputs`
        """
        for ptr in self._graph.inputs():
            yield symbol_from_ptr(self, ptr)

    def iter_symbols(self):
        """Iterate over all symbols in the model.

        Yields:
            :class:`~dwave.optimization.model.Symbol`: Symbols of the model.

        Examples:
            This example iterates over a model's symbols.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> i = model.integer(1, lower_bound=10)
            >>> c = model.constant([[2, 3], [5, 6]])
            >>> symbol_1, symbol_2 = model.iter_symbols()

        See Also:
            :meth:`.iter_constraints`, :meth:`.iter_decisions`,
            :meth:`.num_symbols`
        """
        # Because nodes() is a span of unique_ptr, we can't just iterate over
        # it cythonically. Cython would try to do a copy/move assignment.
        nodes = self._graph.nodes()
        for i in range(nodes.size()):
            yield symbol_from_ptr(self, nodes[i].get())

    def _ith_node(self, Py_ssize_t i):
        if i < 0 or self.num_nodes() <= i:
            raise IndexError(f"{i} is out of bounds for a model with {self.num_nodes()} nodes")
        return symbol_from_ptr(self, self._graph.nodes()[i].get())

    def lock(self):
        """Lock the model.

        No new symbols can be added to a locked model.
        """
        self._graph.topological_sort()  # does nothing if already sorted, so safe to call always
        self._lock_count += 1

        # note that we do not initialize the nodes or resize the states!
        # We do it lazily for performance

    def minimize(self, ArraySymbol value):
        """Set the objective value to minimize.

        Optimization problems have an :term:`objective` and/or
        :term:`constraints <constraint>`. The objective expresses one or more
        aspects of the problem that should be minimized (equivalent to
        maximization when multiplied by a minus sign). For example, an optimized
        itinerary might minimize the value of distance traveled, cost of
        transportation, or travel time.

        Args:
            value: Value of the cost function to minimize.

        Examples:
            This example minimizes a simple polynomial, :math:`y = i^2 - 4i`,
            within bounds.

            >>> from dwave.optimization import Model
            >>> model = Model()
            >>> i = model.integer(lower_bound=-5, upper_bound=5)
            >>> c = model.constant(4)
            >>> y = i * i - c * i
            >>> model.minimize(y)

        See Also:
            :meth:`.feasible`, :meth:`.objective`
        """
        if value is None:
            raise ValueError("value cannot be None")
        if value.size() < 1:
            raise ValueError("the value of an empty array is ambiguous")
        if value.size() > 1:
            raise ValueError("the value of an array with more than one element is ambiguous")
        self._graph.set_objective(value.array_ptr)

    cpdef Py_ssize_t num_constraints(self) noexcept:
        """Return the number of constraints in the model.

        Returns:
            int: Number of constraints.

        Examples:
            This example checks the number of constraints in the model after
            adding a couple of constraints.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> i = model.integer()
            >>> c = model.constant([5, -14])
            >>> _ = model.add_constraint(i <= c[0])
            >>> _ = model.add_constraint(c[1] <= i)
            >>> model.num_constraints()
            2

        See Also:
            :meth:`.feasible`, :meth:`.iter_constraints`,
            :meth:`.num_decisions`, :meth:`.num_symbols`
        """
        return self._graph.num_constraints()

    cpdef Py_ssize_t num_decisions(self) noexcept:
        """Return the number of decision variables in the model.

        An array-of-integers symbol, for example, counts as a single
        decision symbol.

        Returns:
            int: Number of independent decision symbols.

        Examples:
            This example checks the number of decisions in a model after
            adding a single (size 20) decision symbol.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> c = model.constant([1, 5, 8.4])
            >>> i = model.integer(20, upper_bound=100)
            >>> model.num_decisions()
            1

        See Also:
            :meth:`iter_decisions`, :meth:`num_constraints`, :meth:`num_symbols`
        """
        return self._graph.num_decisions()

    def num_edges(self):
        """Return the number of edges in the model's graph.

        Returns:
            int: Number of edges in the :term:`directed acyclic graph` for the
            model.

        Examples:
            This example minimizes the sum of a single constant symbol and
            a single decision symbol, then checks the number of edges in
            the model.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> c = model.constant(5)
            >>> i = model.integer()
            >>> model.minimize(c + i)
            >>> model.num_edges()
            2

        See Also:
            :meth:`.num_nodes`, :meth:`.num_symbols`
        """
        cdef Py_ssize_t num_edges = 0
        for i in range(self._graph.num_nodes()):
            num_edges += self._graph.nodes()[i].get().successors().size()
        return num_edges

    cpdef Py_ssize_t num_inputs(self) noexcept:
        """Return the number of input symbols in the model.

        Returns:
            int: Number of :class:`~dwave.optimization.symbols.Input` symbols in
            the model.

        Examples:
            This example adds two inputs and a constant to a model and
            checks the number of inputs.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> i0, i1 = model.input(), model.input()
            >>> c = model.constant(7)
            >>> model.num_inputs()
            2

        See also:
            :class:`~dwave.optimization.symbols.Input`

            :meth:`iter_inputs`, :meth:`.num_symbols`

        .. versionadded:: 0.6.2
        """
        return self._graph.num_inputs()

    cpdef Py_ssize_t num_nodes(self) noexcept:
        """Return the number of nodes in the model's graph.

        Returns:
            int: Number of nodes in the :term:`directed acyclic graph` for the
            model.

        Examples:
            This example adds a single (size 20) decision symbol and
            a single (size 3) constant symbol, and checks the number of
            nodes in the model.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> c = model.constant([1, 5, 8.4])
            >>> i = model.integer(20, upper_bound=100)
            >>> model.num_nodes()
            2

        See also:
            :meth:`num_edges`, :meth:`.num_symbols`
        """
        return self._graph.num_nodes()

    cpdef Py_ssize_t num_symbols(self) noexcept:
        """Return the number of symbols in the model.

        Equivalent to the number of nodes in the :term:`directed acyclic graph`
        for the model.

        Returns:
            int: Number of symbols in the model.

        Examples:
            This example adds a single (size 20) decision symbol and
            a single (size 3) constant symbol, and checks the number of
            symbols in the model.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> c = model.constant([1, 5, 8.4])
            >>> i = model.integer(20, upper_bound=100)
            >>> model.num_symbols()
            2

        See also:
            :meth:`.iter_symbols`, :meth:`.num_nodes`
        """
        return self.num_nodes()

    @property
    def objective(self):
        """Objective to be minimized.

        Created when you use the :meth:`.minimize` method and associated with
        the :class:`~dwave.optimization.model.ArraySymbol` being minimized; as
        such, supports such methods as
        :meth:`~dwave.optimization.model.ArraySymbol.state`,
        :meth:`~dwave.optimization.model.ArraySymbol.reshape`,
        :meth:`~dwave.optimization.model.ArraySymbol.max`, etc.

        Examples:
            This example prints the value of the :term:`objective` of a model
            representing the simple polynomial, :math:`y = i^2 - 4i`, for a
            state with value :math:`i=2.0`.

            >>> from dwave.optimization import Model
            >>>
            >>> model = Model()
            >>> i = model.integer(lower_bound=-5, upper_bound=5)
            >>> c = model.constant(4)
            >>> y = i**2 - c * i
            >>> model.minimize(y)
            >>> with model.lock():
            ...     model.states.resize(1)
            ...     i.set_state(0, 2.0)
            ...     print(f"Objective = {model.objective.state(0)}")
            Objective = -4.0

        See Also:
            :class:`~dwave.optimization.model.ArraySymbol`: Symbol created by
            the :meth:`.minimize` method

            :meth:`.minimize`,
            :meth:`~dwave.optimization.model.ArraySymbol.state`
        """
        cdef ArrayNode* objective_ptr = self._graph.objective()

        if not objective_ptr:
            return None
        
        return symbol_from_ptr(self, objective_ptr)

    @objective.setter
    def objective(self, ArraySymbol value):
        self.minimize(value)

    def remove_redundant_symbols(self, time_limit_s=None):
        """Remove redundant symbols from the model.

        Symbols are redundant if they are the same type, they share the
        same predecessors, and they encode the same operation.

        A symbol will not be removed if it has :class:`ArraySymbol` object(s)
        referring to it.

        Args:
            time_limit_s (float): The maximum amount of time to spend trying
                to remove redundant symbols. If not provided, the method will
                run until there are no redundant symbols to remove.

        Returns:
            int: Number of symbols removed.

        Examples:

            >>> from dwave.optimization import Model
            >>> model = Model()
            >>> x = model.integer(5)
            >>> -(x + 1).sum()  # doctest: +ELLIPSIS
            <dwave.optimization.symbols.unaryop.Negative at ...>
            >>> -(x + 1).sum()  # doctest: +ELLIPSIS
            <dwave.optimization.symbols.unaryop.Negative at ...>
            >>> model.num_symbols()
            8
            >>> model.remove_redundant_symbols()
            3
            >>> model.num_symbols()
            5
        """
        if self.is_locked():
            raise ValueError("cannot remove symbols from a locked model")
        if time_limit_s is None:
            return self._graph.remove_redundant_nodes(False)
        else:
            return self._graph.remove_redundant_nodes(False, time_limit_s)

    def remove_unused_symbols(self):
        """Remove unused symbols from the model.

        A symbol is considered unused if all of the following are true :

        *   It is not a decision.
        *   It is not an ancestor of the objective.
        *   It is not an ancestor of a constraint.
        *   It has no :class:`ArraySymbol` object(s) referring to it.

        See examples below.

        Returns:
            int: Number of symbols removed.

        Examples:
            This example creates a mix of unused and used symbols, and then
            removes the unused symbols.

            >>> from dwave.optimization import Model
            >>> model = Model()
            >>> x = model.binary(5)
            >>> x.sum()  # create a symbol that will never be used # doctest: +ELLIPSIS
            <dwave.optimization...Sum at ...>
            >>> model.minimize(x.prod())
            >>> model.num_symbols()
            3
            >>> model.remove_unused_symbols()
            1
            >>> model.num_symbols()
            2

            This example creates a mix of unused and used symbols; however,
            unlike in the previous example, the unused symbol is assigned to a
            name in the namespace, preventing from being removed.

            >>> from dwave.optimization import Model
            >>> model = Model()
            >>> x = model.binary(5)
            >>> y = x.sum()  # create a symbol and assign it a name
            >>> model.minimize(x.prod())
            >>> model.num_symbols()
            3
            >>> model.remove_unused_symbols()
            0
            >>> model.num_symbols()
            3

        See Also:
            :meth:`iter_symbols`, :meth:`.num_symbols`
        """
        if self.is_locked():
            raise ValueError("cannot remove symbols from a locked model")
        return self._graph.remove_unused_nodes(False)

    def state_size(self):
        r"""Return an estimate of the size, in bytes, of a model's state.

        For a model encoding several array operations, the state of each array
        must be held in memory. This method returns an estimate of the total
        memory needed to hold a state for every symbol in the model.

        The number of bytes returned by this method is only an estimate. Some
        symbols hold additional information that is not accounted for.

        Returns:
            int: Size of the state for the model.

        Examples:
            This example estimates the size of a model's state.
            In this example a single value is added to a :math:`5\times4` array.
            The output of the addition is also a :math:`5\times4` array.
            Each element of each array requires :math:`8` bytes to represent
            in memory.
            Therefore the total state size is :math:`(5*4 + 1 + 5*4) * 8 = 328`
            bytes.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> i = model.integer((5, 4))  # 5x4 array of integers
            >>> c = model.constant(1)      # one scalar value
            >>> y = i + c                  # 5x4 array of values
            >>> model.state_size()         # (5*4 + 1 + 5*4) * 8 bytes
            328

        See also:
            :meth:`Symbol.state_size()` Estimates the size of a symbol's state

            :meth:`ArraySymbol.state_size()` Estimates the size of an array
            symbol's state

            :meth:`Model.decision_state_size()` Estimates the size of a
            model's decision states

            :ref:`opt_solver_nl_properties` Properties of the
            `Leap <https://cloud.dwavesys.com/leap/>`_ service's
            quantum-classical hybrid nonlinear solver, including limits on
            the maximum state size of a model.
        """
        return sum(sym.state_size() for sym in self.iter_symbols())

    def unlock(self):
        """Release a lock and decrement the lock count."""
        if self._lock_count < 1:
            return  # already unlocked, nothing to do

        self._lock_count -= 1

        # if we're now unlocked, then reset the topological sort and the
        # non-decision states
        if self._lock_count < 1:
            self._graph.reset_topological_sort()


cdef class Symbol(_Empty, _SymbolMixin):
    """Base class for symbols.

    :ref:`Symbols <opt_model_construction_nl_symbols>` are a model's decision
    variables, intermediate variables, constants, and mathematical operations.
    Each symbol corresponds to a node in the :term:`directed acyclic graph`
    representing the problem.
    """
    def __init__(self, *args, **kwargs):
        # disallow direct construction of symbols, they should be constructed
        # via their subclasses.
        raise ValueError("Symbols cannot be constructed directly")

    cdef void initialize_node(self, _Graph model, Node* node_ptr) noexcept:
        self.model = model

        self.node_ptr = node_ptr
        self.expired_ptr = node_ptr.expired_ptr()

    cpdef bool expired(self) noexcept:
        return deref(self.expired_ptr)

    @staticmethod
    cdef Symbol from_ptr(_Graph model, Node* ptr):
        """Construct a Symbol from a C++ Node pointer.

        There are times when a Node* needs to be passed through the Python layer
        and this method provides a mechanism to do so.
        """
        if not ptr:
            raise ValueError("cannot construct a Symbol from a nullptr")
        if model is None:
            raise ValueError("model cannot be None")

        cdef Symbol obj = Symbol.__new__(Symbol)
        obj.initialize_node(model, ptr)
        return obj

    @classmethod
    def _from_ptr(cls, model, capsule):
        """Create a Symbol from a Python capsule containing a Node pointer."""
        cdef Symbol sym = cls.__new__(cls)
        sym.initialize_node(model, <Node*>(PyCapsule_GetPointer(capsule, 'Node*')))
        return sym

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        """Construct a symbol from a compressed file.

        Args:
            zf:
                File pointer to a compressed file encoding
                a symbol. Strings are interpreted as a file name.
            directory:
                Directory where the file is located.
            model:
                The relevant :class:`~dwave.optimization.model.Model`.
            predecessors:
                Not currently supported.
        Returns:
            A symbol.

        See also:
            :meth:`._into_zipfile`
        """
        # Many symbols are constructed using this pattern, so we do it as default.
        return cls(*predecessors)

    def has_state(self, Py_ssize_t index = 0):
        """Return the initialization status of the indexed state.

        Args:
            index (int, optional, default=0): Index of the queried state.

        Returns:
            bool: True if the state is initialized.

        Examples:
            >>> from dwave.optimization import Model
            >>> model = Model()
            >>> i = model.integer(3, lower_bound=0, upper_bound=20)
            >>> with model.lock():
            ...     model.states.resize(5)
            ...     i.set_state(1, [2, 4, 15])
            ...     print(i.has_state(0), i.has_state(1))
            False True

        See Also:
            :meth:`reset_state`
        """
        if not self.model.is_locked() and self.node_ptr.topological_index() < 0:
            raise TypeError("the state of an intermediate variable cannot be accessed without "
                            "locking the model first. See model.lock().")

        if not hasattr(self.model, "states"):
            return False

        cdef States states = self.model.states  # for Cython access

        states.resolve()

        cdef Py_ssize_t num_states = states.size()

        if not -num_states <= index < num_states:
            raise ValueError(f"state index {index} out of bounds. "
                              "Initialize the state with model.states.resize(...).")
        if index < 0:  # allow negative indexing
            index += num_states

        # States are extended lazily, so if the state isn't yet long enough then this
        # node's state has not been initialized
        if <Py_ssize_t>(states._states[index].size()) <= self.node_ptr.topological_index():
            return False

        # Check that the state pointer is not null
        # We need to explicitly cast to evoke unique_ptr's operator bool
        return <bool>(states._states[index][self.node_ptr.topological_index()])

    cpdef uintptr_t id(self) noexcept:
        """Return the "identity" of the underlying node.

        This identity is unique to the underlying node, rather than the identity
        of the Python object representing it.
        Therefore, ``symbol.id()`` is not the same as ``id(symbol)``!

        Returns:
            int: Identity of the underlying node.

        Examples:
            >>> from dwave.optimization import Model
            >>> model = Model()
            >>> a = model.binary()
            >>> aa, = model.iter_symbols()
            >>> assert a.id() == aa.id()
            >>> assert id(a) != id(aa)

            While symbols are not hashable, the ``.id()`` is.

            >>> model = Model()
            >>> x = model.integer()
            >>> seen = {x.id()}

        See Also:
            *   :meth:`.shares_memory`: ``a.shares_memory(b)`` is equivalent to
                ``a.id() == b.id()``.

        """
        # We refer to the node_ptr, which is not necessarily the address of the
        # C++ node, as it subclasses Node.
        # But this is unique to each node, and functions as an id rather than
        # as a pointer, so that's OK.
        return <uintptr_t>self.node_ptr

    def _into_zipfile(self, zf, directory):
        """Store symbol-specific information to a compressed file.

        Args:
            zf:
                File pointer to a compressed file to store the
                symbol. Strings are interpreted as a file name.
            directory:
                Directory where the file is located.
        Returns:
            A compressed file.

        See also:
            :meth:`._from_zipfile`
        """
        # By default we don't save anything beyond what is saved by the model
        pass

    def iter_predecessors(self):
        """Iterate over a symbol's predecessors in the model.

        Yields:
            The symbol's predecessors.

        Examples:
            This example constructs a :math:`b = \sum a` model, where :math:`a`
            is a multiplication of two symbols, and iterates over the
            predecessors of :math:`b` (which is just :math:`a`).

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> i = model.integer((2, 2), upper_bound=20)
            >>> c = model.constant([[21, 11], [10, 4]])
            >>> a = c * i
            >>> b = a.sum()
            >>> a.id() == next(b.iter_predecessors()).id()
            True

            .. figure:: /_images/optimization/iter_predecessors.svg
               :width: 500 px
               :name: dwave-optimization-iter-predecessors-example
               :alt: Image of the model constructed in this example

               Visualization of the model as a :term:`directed acyclic graph`.
               See the :func:`~dwave.optimization.model.Model.to_networkx`
               function for information on visualizing models.

        See Also:
            :meth:`.iter_successors`
        """
        cdef vector[Node*].const_iterator it = self.node_ptr.predecessors().begin()
        cdef vector[Node*].const_iterator end = self.node_ptr.predecessors().end()
        while it != end:
            yield symbol_from_ptr(self.model, deref(it))
            inc(it)

    def iter_successors(self):
        """Iterate over a symbol's successors in the model.

        Yields:
            The symbol's successors.

        Examples:
            This example constructs a :math:`y = x + 5` model and iterates over
            the successors of :math:`x` (which is just :math:`y`).

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> x = model.binary()
            >>> y = x + 5
            >>> y.id() == next(x.iter_successors()).id()
            True

            .. figure:: /_images/optimization/iter_successors.svg
               :width: 500 px
               :name: dwave-optimization-iter-successors-example
               :alt: Image of the model constructed in this example

               Visualization of the model as a :term:`directed acyclic graph`.
               See the :func:`~dwave.optimization.model.Model.to_networkx`
               function for information on visualizing models.

        See Also:
            :meth:`.iter_predecessors`
        """
        cdef vector[Node.SuccessorView].const_iterator it = self.node_ptr.successors().begin()
        cdef vector[Node.SuccessorView].const_iterator end = self.node_ptr.successors().end()
        while it != end:
            yield symbol_from_ptr(self.model, deref(it).ptr)
            inc(it)

    def reset_state(self, Py_ssize_t index):
        """Reset the state of a symbol and any successor symbols.

        Args:
            index (int): Index of the state to reset.

        Examples:
            This example sets two states on a symbol with two successor symbols
            and resets just one state.

            >>> from dwave.optimization import Model
            >>> model = Model()
            >>> lsymbol = model.disjoint_lists_symbol(primary_set_size=5, num_disjoint_lists=2)
            >>> with model.lock():
            ...     model.states.resize(2)
            ...     lsymbol.set_state(0, [[0, 4], [1, 2, 3]])
            ...     lsymbol.set_state(1, [[3, 4], [0, 1, 2]])
            ...     print(f"state 0: {lsymbol[0].state(0)} and {lsymbol[1].state(0)}")
            ...     print(f"state 1: {lsymbol[0].state(1)} and {lsymbol[1].state(1)}")
            ...     lsymbol.reset_state(0)
            ...     print("After reset:")
            ...     print(f"state 0: {lsymbol[0].state(0)} and {lsymbol[1].state(0)}")
            ...     print(f"state 1: {lsymbol[0].state(1)} and {lsymbol[1].state(1)}")
            state 0: [0. 4.] and [1. 2. 3.]
            state 1: [3. 4.] and [0. 1. 2.]
            After reset:
            state 0: [0. 1. 2. 3. 4.] and []
            state 1: [3. 4.] and [0. 1. 2.]

        See Also:
            :meth:`has_state`
        """
        cdef States states = self.model.states

        states.resolve()

        if not 0 <= index < states.size():
            raise ValueError(f"index out of range: {index}")

        if self.node_ptr.topological_index() < 0:
            # unsorted nodes don't have a state to reset
            return

        # make sure the state vector at least contains self
        if <Py_ssize_t>(states._states[index].size()) <= self.node_ptr.topological_index():
            states._states[index].resize(self.node_ptr.topological_index() + 1)

        self.model._graph.recursive_reset(states._states[index], self.node_ptr)

    def shares_memory(self, other):
        """Determine if two symbols share memory.

        Args:
            other (:class:`.Symbol`): Another symbol.

        Returns:
            bool: True if the two symbols share memory.

        See Also:
            :meth:`id`
        """
        cdef Symbol other_
        try:
            other_ = other
        except TypeError:
            return False
        return not self.expired() and self.id() == other_.id()

    def _states_from_zipfile(self, *_, **__):
        # unlike node serialization, by default we raise an error because if
        # this is being called, it must have a state
        raise NotImplementedError(f"{type(self).__name__} has not implemented state deserialization")

    def _states_into_zipfile(self, *_, **__):
        # unlike node serialization, by default we raise an error because if
        # this is being called, it must have a state
        raise NotImplementedError(f"{type(self).__name__} has not implemented state serialization")

    def state_size(self):
        """Return an estimated size, in bytes, of a symbol's state.

        The number of bytes returned by this method is only an estimate. Some
        symbols hold additional information that is not accounted for.

        For most symbols, which are arrays, this method is
        subclassed by the :class:`~dwave.optimization.model.ArraySymbol`
        class's :meth:`~dwave.optimization.model.ArraySymbol.state_size`
        method.

        Returns:
            int: Estimated size in bytes.

        See also:
            :meth:`ArraySymbol.state_size()` An estimate of the size of an array
            symbol's state.

            :meth:`Model.state_size()` An estimate of the size of a model's
            state.
        """
        return 0

    def topological_index(self):
        """Return the topological index of the symbol.

        Returns:
            int or None: Value of the symbol's topological index if the
            model is topologically sorted; otherwise ``None``.

        Examples:
            >>> from dwave.optimization import Model
            >>> model = Model()
            >>> c1 = model.constant([5])
            >>> c1.topological_index() is None
            True
            >>> with model.lock():
            ...     print(c1.topological_index() is None)
            False
        """
        index = self.node_ptr.topological_index()
        return index if index >= 0 else None

def _as_array_symbol(model, symbol):
    if isinstance(symbol, ArraySymbol):
        return symbol
    try:
        return model.constant(symbol)
    except TypeError as err:
        raise TypeError(f"{symbol!r} cannot be interpreted as an array symbol") from err


# Ideally this wouldn't subclass Symbol, but Cython only allows a single
# extension base class, so to support that we assume all ArraySymbols are
# also Symbols (probably a fair assumption)
cdef class ArraySymbol(Symbol, _ArraySymbolMixin):
    """Base class for symbols that can be interpreted as an array."""

    def __init__(self, *args, **kwargs):
        # disallow direct construction of array symbols, they should be constructed
        # via their subclasses.
        raise ValueError("ArraySymbols cannot be constructed directly")

    cdef void initialize_arraynode(self, _Graph model, ArrayNode* array_ptr) noexcept:
        self.array_ptr = array_ptr
        self.initialize_node(model, array_ptr)

    @classmethod
    def _from_ptr(cls, model, capsule):
        cdef ArraySymbol sym = super()._from_ptr(model, capsule)

        sym.array_ptr = dynamic_cast_ptr[ArrayNode](sym.node_ptr)
        if not sym.array_ptr:
            raise TypeError(f"given pointer cannot construct an ArrayNode")

        return sym

    def __eq__(self, rhs):
        return _ArraySymbolMixin.__eq__(self, rhs)

    def __ge__(self, rhs):
        return _ArraySymbolMixin.__ge__(self, rhs)

    def __getitem__(self, index):
        return _ArraySymbolMixin.__getitem__(self, index)

    def __le__(self, rhs):
        return _ArraySymbolMixin.__le__(self, rhs)

    # For some reason, if we remove this method then the __i*__ variants break.
    # Needs more looking into.
    def __neg__(self):
        return _ArraySymbolMixin.__neg__(self)

    def info(self):
        r"""Return information about the values and size of the symbol.

        Symbols might need information about their predecessor(s) to determine
        whether they define a valid operation. For example,
        a :class:`~dwave.optimization.symbols.Divide` symbol does not permit a
        denominator with zero in its possible values.

        Returns:
            :func:`~dataclasses.dataclass`: A :func:`~dataclasses.dataclass`
            object with the following fields:

            *   ``min``: Lower bound (inclusive) on the array values.
            *   ``max``: Upper bound (inclusive) on the array values.
            *   ``integral``: Whether or not the array values are always integral.
            *   ``size``: Size of the array, as one of the following values:

                -   Integer if the array has a fixed size.
                -   :func:`~dataclasses.dataclass` object if the array has a
                    dynamic size. This object has the following fields:

                    *   ``multiplier``, ``symbol``, and ``offset``: Size of the
                        array is
                        :math:`multiplier \times \text{(size of } symbol \text{)} + offset`.
                        If the size of the array symbol is not a linear function
                        of another symbol, the ``symbol`` field is ``self``.
                    *   ``min``: Lower bound (inclusive) on the array size or
                        ``None`` if the bound is uknown.
                    *   ``max``: Upper bound (inclusive) on the array size or
                        ``None`` if the bound is unknown.

            *   ``c_contiguous``: Whether the array is in C memory order.

        .. versionadded:: 0.6.8

        .. versionadded:: 0.8.1
            The ``c_contiguous`` field was added in version 0.8.1.

        Examples:
            For a :class:`~dwave.optimization.symbols.Constant` symbol, ``min``,
            ``max``, ``integral``, and ``size`` are determined by the array.

            >>> import numpy as np
            >>> from dwave.optimization import Model
            >>>
            >>> model = Model()
            >>> c = model.constant(np.linspace(0, 4.5, num=10))
            >>> c.info()
            ArrayInfo(min=0.0, max=4.5, integral=False, size=10, c_contiguous=True)

            Because a :class:`~dwave.optimization.symbols.SetVariable` symbol is
            dynamic, its size is not expressed as an integer. And because its
            size is not derived from another symbol, its size is derived from
            itself.

            >>> s = model.set(10)
            >>> s.info()  # doctest: +ELLIPSIS
            ArrayInfo(min=0.0, max=9.0, integral=True, size=...)
            >>> sizeinfo = s.info().size  # doctest: +ELLIPSIS
            >>> sizeinfo.multiplier
            Fraction(1, 1)
            >>> sizeinfo.symbol.id() == s.id()
            True
            >>> sizeinfo.offset
            Fraction(0, 1)
            >>> sizeinfo.min
            0
            >>> sizeinfo.max
            10

            If a constant array is indexed from a set, that creates another
            dynamic array with size derived from the set.

            >>> b = c[s]
            >>> b.info()  # doctest: +ELLIPSIS
            ArrayInfo(min=0.0, max=4.5, integral=False, size=...)
            >>> sizeinfo = b.info().size  # doctest: +ELLIPSIS
            >>> sizeinfo.multiplier
            Fraction(1, 1)
            >>> sizeinfo.symbol.id() == s.id()  # size is defined by the set
            True
            >>> sizeinfo.offset
            Fraction(0, 1)
            >>> sizeinfo.min
            0
            >>> sizeinfo.max
            10

        .. versionadded:: 0.6.8
        """
        # First get the ArraySizeInfo
        cdef object size  # will be either an int or an ArraySizeInfo

        if self.array_ptr.dynamic():
            sizeinfo = self.array_ptr.sizeinfo()  # C++ SizeInfo type

            if sizeinfo.array_ptr:
                symbol = symbol_from_ptr(self.model, <ArrayNode*>sizeinfo.array_ptr)
            else:
                symbol = None

            size = ArraySizeInfo(
                multiplier=fractions.Fraction(
                    sizeinfo.multiplier.numerator(),
                    sizeinfo.multiplier.denominator(),
                ),
                symbol=symbol,
                offset=fractions.Fraction(
                    sizeinfo.offset.numerator(),
                    sizeinfo.offset.denominator(),
                ),
                min=sizeinfo.min.value() if sizeinfo.min else None,
                max=sizeinfo.max.value() if sizeinfo.max else None,
            )
        else:
            size = self.array_ptr.size()

        # Now construct the ArrayInfo
        return ArrayInfo(
            min=self.array_ptr.min(),
            max=self.array_ptr.max(),
            integral=self.array_ptr.integral(),
            size=size,
            c_contiguous=self.array_ptr.contiguous(),
        )

    def ndim(self):
        r"""Return the number of dimensions for a symbol.

        Returns:
            int: Number of dimensions.

        Examples:
            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> i = model.integer((3, 2), lower_bound=-10, upper_bound=10)
            >>> x = model.binary(2)
            >>> print(f"i: {i.ndim()}\nx: {x.ndim()}")
            i: 2
            x: 1

        See also:
            *   :func:`~dwave.optimization.model.ArraySymbol.shape`
            *   :func:`~numpy.ndim`: :doc:`NumPy <numpy:index>` function
        """
        return self.array_ptr.ndim()

    def shape(self):
        """Return the shape of the symbol.

        Returns:
            tuple: Dimensions of the array. A
            :ref:`dynamic <optimization_philosophy_tensor_programming_dynamic>`
            array symbol returns a :math:`-1` in the first dimension.

        Examples:
            This example returns the shape of a newly instantiated symbol.

            >>> from dwave.optimization import Model
            >>> model = Model()
            >>> x = model.binary(20)
            >>> x.shape()
            (20,)
            >>> s = model.set(20)
            >>> s.shape()
            (-1,)

        See also:
            *   :func:`~dwave.optimization.model.ArraySymbol.ndim`
            *   :func:`~numpy.shape`: :doc:`NumPy <numpy:index>` function
        """
        # We could do the whole buffer format thing and return a numpy array
        # but I think it's better to follow NumPy and return a tuple
        shape = self.array_ptr.shape()
        return tuple(shape[i] for i in range(shape.size()))

    def size(self):
        r"""Return the number of elements in the symbol.

        Returns:
            int or :class:`~dwave.optimization.symbols.Size`: Number of
            elements in the array. For
            :ref:`dynamic <optimization_philosophy_tensor_programming_dynamic>`
            symbols, returns a :class:`~dwave.optimization.symbols.Size` symbol.

        Examples:
            This example checks the size of a :math:`2 \times 3`
            binary symbol.

            >>> from dwave.optimization import Model
            >>> model = Model()
            >>> x = model.binary((2, 3))
            >>> x.size()
            6

        See also:
            *   :class:`~dwave.optimization.symbols.Size`: Generated symbol
            *   :func:`~numpy.size`: :doc:`NumPy <numpy:index>` function
        """
        if self.array_ptr.dynamic():
            from dwave.optimization.symbols import Size
            return Size(self)

        return self.array_ptr.size()

    def state(self, Py_ssize_t index = 0, *, bool copy = True):
        """Return the state of the symbol.

        Args:
            index (int, optional, default=0): Index of the state.

            copy (bool, optional): Currently only True is supported.

        Returns:
            :class:`numpy.ndarray`: State of the symbol.

        Examples:
            This example prints a symbol's two states: initialized
            and uninitialized.

            >>> from dwave.optimization import Model
            >>> model = Model()
            >>> x = model.binary((2, 3))
            >>> z = x.sum()
            >>> with model.lock():
            ...     model.states.resize(2)
            ...     x.set_state(0, [[0, 0, 1], [1, 0, 1]])
            ...     print(z.state(0))
            ...     print(z.state(1))
            3.0
            0.0

        See also:
            :meth:`has_state`, :meth:`reset_state`

        """
        if not copy:
            # todo: document once implemented
            raise NotImplementedError("copy=False is not (yet) supported")

        cdef Py_ssize_t num_states = self.model.states.size()

        if not -num_states <= index < num_states:
            raise ValueError(f"index out of range: {index}")
        elif index < 0:  # allow negative indexing
            index += num_states

        if not self.model.is_locked() and self.node_ptr.topological_index() < 0:
            raise TypeError("the state of an intermediate variable cannot be accessed without "
                            "locking the model first. See model.lock().")

        # Make sure there is a state to access
        cdef States states = self.model.states
        states.resolve()
        self.model._graph.recursive_initialize(states._states.at(index), self.node_ptr)

        # Create a Py_buffer that we can use to construct our memoryview
        cdef Py_buffer view

        # Cython makes it hard to set view.obj to NULL, so we do it this way.
        memset(&view, 0, sizeof(view))

        view.buf = <void*>(self.array_ptr.buff(states._states.at(index)))
        view.format = <char*>(self.array_ptr.format().c_str())
        view.internal = NULL
        view.itemsize = self.array_ptr.itemsize()
        view.len = self.array_ptr.len(states._states.at(index))
        view.ndim = self.array_ptr.ndim()
        # view.obj = NULL  # See above
        view.readonly = 1
        view.shape = <Py_ssize_t*>(self.array_ptr.shape(states._states.at(index)).data())
        view.strides = <Py_ssize_t*>(self.array_ptr.strides().data())
        view.suboffsets = NULL

        # Sometimes empty arrays have their buffer set to NULL which
        # PyMemoryView_FromBuffer() does not like. So in that case we just
        # make an empty array directly.
        # Also, be aware the the format here is NumPy type string which is not
        # exactly the same thing as a struct format string. But luckily is the
        # same for all of the dtypes we care about.
        if view.buf == NULL:
            if view.len:
                raise RuntimeError("Array had an unexpected shape")
            shape = self.array_ptr.shape(states._states.at(index))
            return np.empty(tuple(shape[i] for i in range(shape.size())), dtype=view.format)

        # Now create a memoryview object. This, naturally, holds a view to memory
        # that may be deallocated at any time, so we make a copy before returning.
        cdef object mv = PyMemoryView_FromBuffer(&view)

        return np.asarray(mv, copy=True)

    # Because of how Cython inheritance works, we need Symbol to be the first
    # inherited class. But ideally we'd like _ArraySymbolMixin to be first
    # for the mro to work. So we need to explicitly call _ArraySymbolMixin's
    # methods in some cases.
    def _states_from_zipfile(self, *args, **kwargs):
        return _ArraySymbolMixin._states_from_zipfile(self, *args, **kwargs)

    def _states_into_zipfile(self, *args, **kwargs):
        return _ArraySymbolMixin._states_into_zipfile(self, *args, **kwargs)

    def state_size(self):
        r"""Return an estimate of the size, in bytes, of the symbol's state.

        Returns:
            int: Number of bytes needed to encode the array.

        Examples:
            This example returns the size of an
            :class:`~dwave.optimization.symbols.IntegerVariable` symbol that
            encodes a :math:`5 \times 4` array of integers, each represented by
            an :math:`8`-byte float. Therefore the estimated state size is
            :math:`5*4*8 = 160` bytes.

            >>> from dwave.optimization import Model
            >>> model = Model()
            >>> i = model.integer((5, 4))  # 5x4 array of integers
            >>> i.state_size()             # 5*4*8 bytes
            160

        See also:
            :meth:`Symbol.state_size()` An estimate of the size of a symbol's
            state.

            :meth:`Model.state_size()` An estimate of the size of a model's
            state.
        """
        if not self.array_ptr.dynamic():
            # For fixed-length arrays, the state size is simply the size of the
            # array times the size of each element in the array.
            return self.array_ptr.size() * self.array_ptr.itemsize()

        sizeinfo = self.array_ptr.sizeinfo()

        # If it gets its size from elsewhere, do the calculation. We could be
        # more efficient about this, but for now let's do the simple thing
        if sizeinfo.array_ptr != self.array_ptr:
            sizeinfo = sizeinfo.substitute(self.model.num_nodes())

        # This shouldn't happen, but just in case...
        if not sizeinfo.max.has_value():
            raise RuntimeError(f"size of {self!r} is unbounded")

        return sizeinfo.max.value() * self.array_ptr.itemsize()

    def strides(self):
        """Return the stride length, in bytes, for traversing a symbol.

        Returns:
            tuple: Number of bytes to step in each dimension when
            traversing a symbol.

        Examples:
            This example returns the stride length of an
            :class:`~dwave.optimization.symbols.IntegerVariable` symbol.

            >>> from dwave.optimization import Model
            >>> model = Model()
            >>> i = model.integer((2, 3), upper_bound=20)
            >>> i.strides()
            (24, 8)

        See Also:
            :attr:`~numpy.ndarray.strides`: :doc:`NumPy <numpy:index>` attribute
        """
        strides = self.array_ptr.strides()
        return tuple(strides[i] for i in range(strides.size()))


def _broadcast_shapes(vector[Py_ssize_t] lhs, vector[Py_ssize_t] rhs):
    """Broadcast the input shapes into a single shape or throw an error if they are incompatible."""
    return tuple(broadcast_shapes(span[Py_ssize_t](lhs), span[Py_ssize_t](rhs)))


cdef class States(_Empty, _StatesMixin):
    r"""States of a model.

    :ref:`States <opt_model_construction_nl_states>` represent assignments of
    values to the symbols of a model. For example, an
    :class:`~dwave.optimization.symbols.IntegerVariable` symbol of size
    :math:`1 \times 5` might have state ``[3, 8, 0, 12, 8]``, representing one
    assignment of values to the symbol. When the symbol is a decision variable
    in the model, the state might be (part of) a solution to the modeled
    problem.

    You can set the states of a model's symbols for the purpose of testing your
    model or providing an initial state for a :term:`solver`; a model submitted
    to a solver can have its states updated by returned solutions.

    Examples:
        This example creates a model that includes the polynomial
        :math:`k = i^2 + j^2 + 1` on integer variables and manipulates its
        states to test that it behaves as expected.

        >>> from dwave.optimization import Model
        >>>
        >>> model = Model()
        >>> i = model.integer(5, lower_bound=-10, upper_bound=10) # Array of 5 int elements
        >>> j = model.integer(lower_bound=-4, upper_bound=6)      # scalar integer
        >>> k = i**2 + j**2 + 1

        At this point the size of the size of the newly created model's state is
        zero: to set states on its symbols, resize it to the number of wanted
        states using the :meth:`.resize` method.

        >>> model.states.size()
        0
        >>> model.states.resize(2)

        Lock the model to enable access to the states of successor symbols
        (non-decision variables) such as :math:`k`.

        >>> with model.lock():
        ...     i.set_state(0, [-10, -5, 0, 5, 10])
        ...     j.set_state(0, 0)
        ...     print(k.state(0))
        ...     i.set_state(1, [-10, -5, 0, 5, 10])
        ...     j.set_state(1, 2)
        ...     print(k.state(1))
        [101.  26.   1.  26. 101.]
        [105.  30.   5.  30. 105.]

        You can clear the states you set.

        >>> model.states.clear()
        >>> model.states.size()
        0

    See Also:
        :attr:`~dwave.optimization.model.Model.states`
    """
    def __init__(self, model):
        self._model_ref = weakref.ref(model)

    def __len__(self):
        """The number of model states."""
        return self.size()

    cdef void attach_states(self, vector[State] states) noexcept:
        """Attach the given states.

        Note:
            Currently replaces any current states with the given states.

            This method does not check whether the states are locked
            or that the states are valid.

        Args:
            states: States to be attached.
        """
        self._future = None
        self._result_hook = None
        self._states.swap(states)

    def clear(self):
        """Clear any saved states.

        Clears any memory allocated to the states.

        Examples:
            This example clears a state set on an integer decision symbol.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> i = model.integer(2)
            >>> model.states.resize(3)
            >>> i.set_state(0, [3, 5])
            >>> print(i.state(0))
            [3. 5.]
            >>> model.states.clear()
            >>> model.states.size()
            0

        See Also:
            :meth:`.resize`, :meth:`.size`
        """
        self.detach_states()

    cdef vector[State] detach_states(self):
        """Move the current C++ states into a returned vector.

        Leaves the model's states empty.

        Note:
            This method does not check whether the states are locked.

        Returns:
            States of the model prior to execution.
        """
        self.resolve()
        # move should impliclty leave the states in a valid state, but
        # just to be super explicit we swap with an empty vector first
        cdef vector[State] states
        self._states.swap(states)
        return move(states)

    def initialize(self):
        """Initialize any uninitialized states.

        This method is intended for use by developers of the Ocean SDK.

        See Also:
            :meth:`.clear`, :meth:`.resize`
        """
        self.resolve()

        cdef _Graph model = self._model()

        if not model.is_locked():
            raise ValueError("Cannot initialize states of an unlocked model")
        for i in range(self._states.size()):
            self._states[i].resize(model.num_nodes())
            model._graph.initialize_state(self._states[i])

    def _model(self):
        """Get a ref-counted Model object."""
        cdef _Graph m = self._model_ref()
        if m is None:
            raise ReferenceError("accessing the states of a garbage collected model")
        return m

    def _reset_intermediate_states(self):
        """Reset any non-decision states."""
        cdef Py_ssize_t num_decisions = self._model().num_decisions()
        for i in range(self.size()):
            self._states[i].resize(num_decisions)

    def resize(self, Py_ssize_t n):
        """Resize the number of states.

        To set the states of a model's symbols, for the purpose of testing your
        model or providing an initial state for a :term:`solver`, you must first
        set the size of the model's states. A newly created model has a state
        size of zero.

        Resizing to 0 is not  guaranteed to clear the memory allocated to
        states.

        Args:
            n (int): Required number of states. If smaller than the current
                :meth:`.size()`, states are reduced to the first ``n`` states by
                removing those beyond. If greater than the current
                :meth:`.size()`, new uninitialized states are added as needed to
                reach a size of ``n``.

        Examples:
            This example adds three uninitialized states to a model.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> i = model.integer(2)
            >>> model.states.resize(3)

        See Also:
            :meth:`.clear`, :meth:`.size`
        """
        self.resolve()

        if n < 0:
            raise ValueError("n must be a non-negative integer")

        self._states.resize(n)

    cpdef Py_ssize_t size(self) except -1:
        """Number of model states.

        Examples:
            This example adds three uninitialized states to a model and
            verifies the number of model states.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> model.states.resize(3)
            >>> model.states.size()
            3

        See Also:
            :meth:`.resize`
        """
        self.resolve()
        return self._states.size()

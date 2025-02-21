# Copyright 2024 D-Wave Inc.
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
import functools
import itertools
import json
import numbers
import operator
import struct
import zipfile

import numpy as np

from cpython cimport Py_buffer
from cython.operator cimport dereference as deref, preincrement as inc
from cython.operator cimport typeid
from libcpp cimport bool
from libcpp.utility cimport move
from libcpp.vector cimport vector

from dwave.optimization.libcpp.array cimport Array as cppArray
from dwave.optimization.libcpp.graph cimport DecisionNode as cppDecisionNode
from dwave.optimization.states cimport States
from dwave.optimization.states import StateView
from dwave.optimization.symbols cimport symbol_from_ptr

__all__ = []


DEFAULT_SERIALIZATION_VERSION = (0, 1)
"""A 2-tuple encoding the default serialization format used for serializing models."""

KNOWN_SERIALIZATION_VERSIONS = (
    (0, 1),
)
"""A tuple of 2-tuples listing all serialization versions supported."""


cdef class _Graph:
    """A ``_Graph`` is a class that manages a C++ ``dwave::optimization::Graph``.

    It is not intended for a user to use ``_Graph`` directly. Rather, classes
    may inherit from ``_Graph``.
    """
    def __cinit__(self):
        self._lock_count = 0
        self._data_sources = []

    def __init__(self, *args, **kwargs):
        # disallow direct construction of _Graphs, they should be constructed
        # via their subclasses.
        raise ValueError("_Graphs cannot be constructed directly")

    def add_constraint(self, ArraySymbol value):
        """Add a constraint to the model.

        Args:
            value: Value that must evaluate to True for the state
                of the model to be feasible.

        Returns:
            The constraint symbol.

        Examples:
            This example adds a single constraint to a model.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> i = model.integer()
            >>> c = model.constant(5)
            >>> constraint_sym = model.add_constraint(i <= c)

            The returned constraint symbol can be assigned and evaluated
            for a model state:

            >>> with model.lock():
            ...     model.states.resize(1)
            ...     i.set_state(0, 1) # Feasible state
            ...     print(constraint_sym.state(0))
            1.0
            >>> with model.lock():
            ...     i.set_state(0, 6) # Infeasible state
            ...     print(constraint_sym.state(0))
            0.0
        """
        if value is None:
            raise ValueError("value cannot be None")
        # TODO: shall we accept array valued constraints?
        self._graph.add_constraint(value.array_ptr)
        return value

    def decision_state_size(self):
        r"""Return an estimate of the size, in bytes, of a model's decision states.

        For more details, see :meth:`.state_size()`.
        This method differs by counting the state of only the decision variables.

        Examples:
            This example estimates the size of a model state.
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
            :meth:`Symbol.state_size()` An estimate of the size of a symbol's
            state.

            :meth:`ArraySymbol.state_size()` An estimate of the size of an array
            symbol's state.

            :meth:`Model.state_size()` An estimate of the size of a model's
            decision states.
        """
        return sum(sym.state_size() for sym in self.iter_decisions())

    @classmethod
    def from_file(cls, file, *,
                  check_header = True,
                  ):
        """Construct a model from the given file.

        Args:
            file:
                File pointer to a readable, seekable file-like object encoding
                a model. Strings are interpreted as a file name.

        Returns:
            A model.

        See also:
            :meth:`.into_file`, :meth:`.to_file`
        """
        import dwave.optimization.symbols as symbols

        if isinstance(file, str):
            with open(file, "rb") as f:
                return cls.from_file(f)

        prefix = b"DWNL"

        read_prefix = file.read(len(prefix))
        if read_prefix != prefix:
            raise ValueError("Unknown file type. Expected magic string "
                             f"{prefix!r} but received {read_prefix!r} "
                             "instead")

        version = tuple(file.read(2))

        if version not in KNOWN_SERIALIZATION_VERSIONS:
            raise ValueError("Unknown serialization format. Expected one of "
                             f"{KNOWN_SERIALIZATION_VERSIONS} but received {version} "
                             "instead. Upgrading your dwave-optimization version may help.")

        if check_header:
            # we'll need the header values to check later
            header_len = struct.unpack('<I', file.read(4))[0]
            header_data = json.loads(file.read(header_len).decode('ascii'))

        cdef _Graph model = cls()

        with zipfile.ZipFile(file, mode="r") as zf:
            model_info = json.loads(zf.read("info.json"))

            num_nodes = model_info.get("num_nodes")
            if not isinstance(num_nodes, int) or num_nodes < 0:
                raise ValueError("expected num_nodes to be a positive integer")

            with zf.open("nodetypes.txt", "r") as fcls, zf.open("adj.adjlist", "r") as fadj:
                for lineno, (classname, adjlist) in enumerate(zip(fcls, fadj)):
                    # get the predecessors
                    node_id, *predecessor_ids = map(int, adjlist.split(b" "))
                    if node_id != lineno:  # sanity check
                        raise ValueError("unexpected adj.adjlist format")

                    predecessors = []
                    for pid in predecessor_ids:
                        if not 0 <= pid < node_id:
                            raise ValueError("unexpected predecessor id")
                        predecessors.append(symbol_from_ptr(model, model._graph.nodes()[pid].get()))

                    # now make the node
                    directory = f"nodes/{node_id}/"
                    classname = classname.decode("UTF-8").rstrip("\n")

                    # take advanctage of the symbols all being in the same namespace
                    # and the fact that we (currently) encode them all by name
                    cls = getattr(symbols, classname, None)

                    if not issubclass(cls, Symbol):
                        raise ValueError("encoded model has an unsupported node type")

                    cls._from_zipfile(zf, directory, model, predecessors=predecessors)

            objective_buff = zf.read("objective.json")
            if objective_buff:
                objective_id = json.loads(objective_buff)
                if not isinstance(objective_id, int) or objective_id >= model.num_nodes():
                    raise ValueError("objective must be an integer and a valid node id")
                model.minimize(symbol_from_ptr(model, model._graph.nodes()[objective_id].get()))

            for cid in json.loads(zf.read("constraints.json")):
                model.add_constraint(symbol_from_ptr(model, model._graph.nodes()[cid].get()))

            # Read any states that have been encoded
            num_states = model_info.get("num_states")
            if not isinstance(num_states, int) or num_states < 0:
                raise ValueError("expected num_states to be a positive integer")

            if num_states > 0:
                model.states.resize(num_states)

                # now read the states of the decision variables
                num_decisions = model.num_decisions()  # use the model not the serialization
                for node in itertools.islice(model.iter_symbols(), 0, num_decisions):
                    for i in range(num_states):
                        node._state_from_zipfile(zf, f"nodes/{node.topological_index()}/states/{i}/", i)

        if check_header:
            expected = model._header_data(only_decision=False)

            if not expected.items() <= header_data.items():
                raise ValueError(
                    "header data does not match the deserialized CQM. "
                    f"Expected {expected!r} to be a subset of {header_data!r}"
                    )

        return model

    def _header_data(self, *, only_decision, max_num_states=float('inf')):
        """The header data associated with the model (but not the states)."""
        num_nodes = self.num_decisions() if only_decision else self.num_nodes()
        try:
            num_states = max(0, min(self.states.size(), max_num_states))
        except AttributeError:
            num_states = 0

        decision_state_size = self.decision_state_size()
        state_size = decision_state_size if only_decision else self.state_size()

        return dict(
            decision_state_size=decision_state_size,
            num_nodes=num_nodes,
            state_size=state_size,
            num_states=num_states,
        )

    def into_file(self, file, *,
                  Py_ssize_t max_num_states = 0,
                  bool only_decision = False,
                  object version = None,
                  ):
        """Serialize the model into an existing file.

        Args:
            file:
                File pointer to an existing writeable, seekable
                file-like object encoding a model. Strings are
                interpreted as a file name.
            max_num_states:
                Maximum number of states to serialize along with the model.
                The number of states serialized is
                ``min(model.states.size(), max_num_states)``.
            only_decision:
                If ``True``, only decision variables are serialized.
                If ``False``, all symbols are serialized.
            version:
                A 2-tuple indicating which serialization version to use.

        See also:
            :meth:`.from_file`, :meth:`.to_file`

        TODO: describe the format
        """
        if not self.is_locked():
            # lock for the duration of the method
            with self.lock():
                return self.into_file(
                    file,
                    max_num_states=max_num_states,
                    only_decision=only_decision,
                    version=version,
                    )

        if isinstance(file, str):
            with open(file, "wb") as f:
                return self.into_file(
                    f,
                    max_num_states=max_num_states,
                    only_decision=only_decision,
                    version=version,
                    )

        if version is None:
            version = DEFAULT_SERIALIZATION_VERSION
        elif version not in KNOWN_SERIALIZATION_VERSIONS:
            raise ValueError("Unknown serialization format. Expected one of "
                             f"{KNOWN_SERIALIZATION_VERSIONS} but received {version} "
                             "instead. Upgrading your dwave-optimization version may help.")

        model_info = self._header_data(
            max_num_states=max_num_states,
            only_decision=only_decision,
        )
        num_states = model_info["num_states"]

        encoder = json.JSONEncoder(separators=(',', ':'))

        # First prepend the header

        # The first 4 bytes are DWNL
        file.write(b"DWNL")

        # The next 1 byte is an unsigned byte encoding the major version of the
        # file format
        # The next 1 byte is an unsigned byte encoding the minor version of the
        # file format
        file.write(bytes(version))

        # The next 4 bytes form a little-endian unsigned int, the length of
        # the header data `HEADER_LEN`.

        # The next `HEADER_LEN` bytes form the header data. This will be `data`
        # json-serialized and encoded with 'ascii'.
        header_data = encoder.encode(model_info).encode("ascii")

        # Now pad to make the entire header divisible by 64
        padding = b' '*(64 - (len(header_data) + 4 + 2 + 4)  % 64)

        file.write(struct.pack('<I', len(header_data) + len(padding)))  # header length
        file.write(header_data)
        file.write(padding)

        # The rest of it is a zipfile
        with zipfile.ZipFile(file, mode="w") as zf:
            zf.writestr("info.json", encoder.encode(model_info))
            zf.writestr("version.txt", ".".join(map(str, version)))

            # Do three passes over the nodes

            # If we're only encoding the decision variables then we want to stop early.
            # We know that we're topologically sorted so the first num_decisions are
            # exactly the decision variables.
            stop = self.num_decisions() if only_decision else self.num_nodes()

            # On the first pass we made a nodetypes.txt file that has the node names
            with zf.open("nodetypes.txt", "w", force_zip64=True) as f:
                for node in itertools.islice(self.iter_symbols(), 0, stop):
                    f.write(type(node).__name__.encode("UTF-8"))
                    f.write(b"\n")

            # On the second pass we encode the adjacency
            with zf.open("adj.adjlist", "w", force_zip64=True) as f:
                # We don't actually need to make the Python symbols here, but it's convenient
                # Also, if we're only_decision then there will never be predecessors, but
                # let's reuse the code for now.
                stop = self.num_decisions() if only_decision else self.num_nodes()
                for node in itertools.islice(self.iter_symbols(), 0, stop):
                    f.write(f"{node.topological_index()}".encode("UTF-8"))
                    for pred in node.iter_predecessors():
                        f.write(f" {pred.topological_index()}".encode("UTF-8"))
                    f.write(b"\n")

            # On the third pass, we allow nodes to save whatever info they want
            # to in a nested node/<topological_index> directory
            for node in itertools.islice(self.iter_symbols(), 0, stop):
                directory = f"nodes/{node.topological_index()}/"
                node._into_zipfile(zf, directory)

            # Encode the objective and the constraints
            if self.objective is not None and self.objective.topological_index() < stop:
                zf.writestr("objective.json", encoder.encode(self.objective.topological_index()))
            else:
                zf.writestr("objective.json", b"")

            constraints = []  # todo: not yet available at the python level
            for c in self.iter_constraints():
                if c is not None and c.topological_index() < stop:
                    constraints.append(c.topological_index())
            zf.writestr("constraints.json", encoder.encode(constraints))

            # Encode the states if requested
            if num_states > 0:  # redundant, but good short circuit
                for node in itertools.islice(self.iter_symbols(), self.num_decisions()):
                    # only save states that have been initialized
                    for i in filter(node.has_state, range(num_states)):
                        directory = f"nodes/{node.topological_index()}/states/{i}/"
                        node._state_into_zipfile(zf, directory, i)

    cpdef bool is_locked(self) noexcept:
        """Lock status of the model.

        No new symbols can be added to a locked model.

        See also:
            :meth:`.lock`, :meth:`.unlock`
        """
        return self._lock_count > 0

    def iter_constraints(self):
        """Iterate over all constraints in the model.

        Examples:
            This example adds a single constraint to a model and iterates over it.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> i = model.integer()
            >>> c = model.constant(5)
            >>> model.add_constraint(i <= c) # doctest: +ELLIPSIS
            <dwave.optimization.symbols.LessEqual at ...>
            >>> constraints = next(model.iter_constraints())
        """
        for ptr in self._graph.constraints():
            yield symbol_from_ptr(self, ptr)

    def iter_decisions(self):
        """Iterate over all decision variables in the model.

        Examples:
            This example adds a single decision symbol to a model and iterates over it.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> i = model.integer()
            >>> c = model.constant(5)
            >>> model.add_constraint(i <= c) # doctest: +ELLIPSIS
            <dwave.optimization.symbols.LessEqual at ...>
            >>> decisions = next(model.iter_decisions())
        """
        for ptr in self._graph.decisions():
            yield symbol_from_ptr(self, ptr)

    def iter_symbols(self):
        """Iterate over all symbols in the model.

        Examples:
            This example iterates over a model's symbols.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> i = model.integer(1, lower_bound=10)
            >>> c = model.constant([[2, 3], [5, 6]])
            >>> symbol_1, symbol_2 = model.iter_symbols()
        """
        # Because nodes() is a span of unique_ptr, we can't just iterate over
        # it cythonically. Cython would try to do a copy/move assignment.
        nodes = self._graph.nodes()
        for i in range(nodes.size()):
            yield symbol_from_ptr(self, nodes[i].get())

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

        Optimization problems have an objective and/or constraints. The objective
        expresses one or more aspects of the problem that should be minimized
        (equivalent to maximization when multiplied by a minus sign). For example,
        an optimized itinerary might minimize the value of distance traveled or
        cost of transportation or travel time.

        Args:
            value: Value for which to minimize the cost function.

        Examples:
            This example minimizes a simple polynomial, :math:`y = i^2 - 4i`,
            within bounds.

            >>> from dwave.optimization import Model
            >>> model = Model()
            >>> i = model.integer(lower_bound=-5, upper_bound=5)
            >>> c = model.constant(4)
            >>> y = i*i - c*i
            >>> model.minimize(y)
        """
        if value is None:
            raise ValueError("value cannot be None")
        if value.size() < 1:
            raise ValueError("the value of an empty array is ambiguous")
        if value.size() > 1:
            raise ValueError("the value of an array with more than one element is ambiguous")
        self._graph.set_objective(value.array_ptr)

    cpdef Py_ssize_t num_constraints(self) noexcept:
        """Number of constraints in the model.

        Examples:
            This example checks the number of constraints in the model after
            adding a couple of constraints.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> i = model.integer()
            >>> c = model.constant([5, -14])
            >>> model.add_constraint(i <= c[0]) # doctest: +ELLIPSIS
            <dwave.optimization.symbols.LessEqual at ...>
            >>> model.add_constraint(c[1] <= i) # doctest: +ELLIPSIS
            <dwave.optimization.symbols.LessEqual at ...>
            >>> model.num_constraints()
            2
        """
        return self._graph.num_constraints()

    cpdef Py_ssize_t num_decisions(self) noexcept:
        """Number of independent decision nodes in the model.

        An array-of-integers symbol, for example, counts as a single
        decision node.

        Examples:
            This example checks the number of decisions in a model after
            adding a single (size 20) decision symbol.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> c = model.constant([1, 5, 8.4])
            >>> i = model.integer(20, upper_bound=100)
            >>> model.num_decisions()
            1
        """
        return self._graph.num_decisions()

    def num_edges(self):
        """Number of edges in the directed acyclic graph for the model.

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
        """
        cdef Py_ssize_t num_edges = 0
        for i in range(self._graph.num_nodes()):
            num_edges += self._graph.nodes()[i].get().successors().size()
        return num_edges

    cpdef Py_ssize_t num_nodes(self) noexcept:
        """Number of nodes in the directed acyclic graph for the model.

        See also:
            :meth:`.num_symbols`

        Examples:
            This example add a single (size 20) decision symbol and
            a single (size 3) constant symbol checks the number of
            nodes in the model.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> c = model.constant([1, 5, 8.4])
            >>> i = model.integer(20, upper_bound=100)
            >>> model.num_nodes()
            2
        """
        return self._graph.num_nodes()

    cpdef Py_ssize_t num_symbols(self) noexcept:
        """Number of symbols tracked by the model.

        Equivalent to the number of nodes in the directed acyclic
        graph for the model.

        See also:
            :meth:`.num_nodes`

        Examples:
            This example add a single (size 20) decision symbol and
            a single (size 3) constant symbol checks the number of
            symbols in the model.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> c = model.constant([1, 5, 8.4])
            >>> i = model.integer(20, upper_bound=100)
            >>> model.num_symbols()
            2
        """
        return self.num_nodes()

    def remove_unused_symbols(self):
        """Remove unused symbols from the model.

        A symbol is considered unused if all of the following are true :

        * It is not a decision.
        * It is not an ancestor of the objective.
        * It is not an ancestor of a constraint.
        * It has no :class:`ArraySymbol` object(s) referring to it. See examples
          below.

        Returns:
            The number of symbols removed.

        Examples:
            In this example we create a mix of unused and used symbols. Then
            the unused symbols are removed with ``remove_unused_symbols()``.

            >>> from dwave.optimization import Model
            >>> model = Model()
            >>> x = model.binary(5)
            >>> x.sum()  # create a symbol that will never be used # doctest: +ELLIPSIS
            <dwave.optimization.symbols.Sum at ...>
            >>> model.minimize(x.prod())
            >>> model.num_symbols()
            3
            >>> model.remove_unused_symbols()
            1
            >>> model.num_symbols()
            2

            In this example we create a mix of unused and used symbols.
            However, unlike the previous example, we assign the unused symbol
            to a name in the namespace. This prevents the symbol from being
            removed.

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

        """
        if self.is_locked():
            raise ValueError("cannot remove symbols from a locked model")
        return self._graph.remove_unused_nodes()

    def state_size(self):
        r"""Return an estimate of the size, in bytes, of a model state.

        For a model encoding several array operations, the state of each array
        must be held in memory. This method returns an estimate of the total
        memory needed to hold a state for every symbol in the model.

        The number of bytes returned by this method is only an estimate. Some
        symbols hold additional information that is not accounted for.

        Examples:
            This example estimates the size of a model state.
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
            :meth:`Symbol.state_size()` An estimate of the size of a symbol's
            state.

            :meth:`ArraySymbol.state_size()` An estimate of the size of an array
            symbol's state.

            :meth:`Model.decision_state_size()` An estimate of the size of a
            model's decision states.

            :ref:`properties_solver_properties` The properties of the
            `Leap <https://cloud.dwavesys.com/leap/>`_ service's
            quantum-classical hybrid nonlinear solver. Including limits on
            the maximum state size of a model.
        """
        return sum(sym.state_size() for sym in self.iter_symbols())

    def unlock(self):
        """Release a lock, decrementing the lock count.

        Symbols can be added to unlocked models only.

        See also:
            :meth:`.is_locked`, :meth:`.lock`
        """
        if self._lock_count < 1:
            return  # already unlocked, nothing to do

        self._lock_count -= 1

        # if we're now unlocked, then reset the topological sort and the
        # non-decision states
        if self._lock_count < 1:
            self._graph.reset_topological_sort()


cdef class Symbol:
    """Base class for symbols.

    Each symbol corresponds to a node in the directed acyclic graph representing
    the problem.
    """
    def __init__(self, *args, **kwargs):
        # disallow direct construction of symbols, they should be constructed
        # via their subclasses.
        raise ValueError("Symbols cannot be constructed directly")

    def __repr__(self):
        """Return a representation of the symbol.

        The representation refers to the identity of the underlying node, rather than
        the identity of the Python symbol.
        """
        cls = type(self)
        return f"<{cls.__module__}.{cls.__qualname__} at {self.id():#x}>"

    cdef void initialize_node(self, _Graph model, cppNode* node_ptr) noexcept:
        self.model = model

        self.node_ptr = node_ptr
        self.expired_ptr = node_ptr.expired_ptr()

    def equals(self, other):
        """Compare whether two symbols are identical.

        Args:
            other: A symbol for comparison.

        Equal symbols represent the same quantity in the model.

        Note that comparing symbols across models is expensive.

        See Also:
            :meth:`Symbol.maybe_equals`: an alternative for equality testing
            that can return false positives but is faster.
        """
        cdef Py_ssize_t maybe = self.maybe_equals(other)
        if maybe != 1:
            return True if maybe else False

        # todo: caching
        return all(p.equals(q) for p, q in zip(self.iter_predecessors(), other.iter_predecessors()))

    cpdef bool expired(self) noexcept:
        return deref(self.expired_ptr)

    @staticmethod
    cdef Symbol from_ptr(_Graph model, cppNode* ptr):
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

    @staticmethod
    def _from_symbol(Symbol symbol):
        # Symbols must overload this method
        raise ValueError("Symbols cannot be constructed directly")

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
            index: Index of the queried state.

        Returns:
            True if the state is initialized.
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
            raise ValueError(f"index out of range: {index}")
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
        Therefore, ``symdol.id()`` is not the same as ``id(symbol)``!

        Examples:
            >>> from dwave.optimization import Model
            ...
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
            :meth:`.shares_memory`: ``a.shares_memory(b)`` is equivalent to ``a.id() == b.id()``.
            
            :meth:`.equals`: ``a.equals(b)`` will return ``True`` if ``a.id() == b.id()``. Though
            the inverse is not necessarily true.

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

        Examples:
            This example constructs a :math:`b = \sum a` model, where :math:`a`
            is a multiplication of two symbols, and iterates over the
            predecessor's of :math:`b` (which is just :math:`a`).

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> i = model.integer((2, 2), upper_bound=20)
            >>> c = model.constant([[21, 11], [10, 4]])
            >>> a = c * i
            >>> b = a.sum()
            >>> a.equals(next(b.iter_predecessors()))
            True
        """
        cdef vector[cppNode*].const_iterator it = self.node_ptr.predecessors().begin()
        cdef vector[cppNode*].const_iterator end = self.node_ptr.predecessors().end()
        while it != end:
            yield symbol_from_ptr(self.model, deref(it))
            inc(it)

    def iter_successors(self):
        """Iterate over a symbol's successors in the model.

        Examples:
            This example constructs iterates over the successor symbols
            of a :class:`~dwave.optimization.symbols.DisjointLists`
            symbol.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> lsymbol, lsymbol_lists = model.disjoint_lists(
            ...     primary_set_size=5,
            ...     num_disjoint_lists=2)
            >>> lsymbol_lists[0].equals(next(lsymbol.iter_successors()))
            True
        """
        cdef vector[cppNode.SuccessorView].const_iterator it = self.node_ptr.successors().begin()
        cdef vector[cppNode.SuccessorView].const_iterator end = self.node_ptr.successors().end()
        while it != end:
            yield symbol_from_ptr(self.model, deref(it).ptr)
            inc(it)

    def maybe_equals(self, other):
        """Compare to another symbol.
        
        This method exists because a complete equality test can be expensive.

        Args:
            other: Another symbol in the model's directed acyclic graph.

        Returns: integer
            Supported return values are:

            *   ``0``---Not equal (with certainty)
            *   ``1``---Might be equal (no guarantees); a complete equality test is necessary
            *   ``2``---Are equal (with certainty)

        Examples:
            This example compares
            :class:`~dwave.optimization.symbols.IntegerVariable` symbols
            of different sizes.

            >>> from dwave.optimization import Model
            >>> model = Model()
            >>> i = model.integer(3, lower_bound=0, upper_bound=20)
            >>> j = model.integer(3, lower_bound=-10, upper_bound=10)
            >>> k = model.integer(5, upper_bound=55)
            >>> i.maybe_equals(j)
            1
            >>> i.maybe_equals(k)
            0

        See Also:
            :meth:`.equals`: a more expensive form of equality testing.
        """
        cdef Py_ssize_t NOT = 0
        cdef Py_ssize_t MAYBE = 1
        cdef Py_ssize_t DEFINITELY = 2

        # If we're the same object, then we're equal
        if self is other:
            return DEFINITELY

        if not isinstance(other, Symbol):
            return NOT

        # Should we require identical types?
        if not isinstance(self, type(other)) and not isinstance(other, type(self)):
            return NOT

        cdef Symbol rhs = other

        if self.shares_memory(rhs):
            return DEFINITELY

        # Check is that we have the right number of predecessors
        if self.node_ptr.predecessors().size() != rhs.node_ptr.predecessors().size():
            return NOT

        # Finally, out prdecessors should have the same types in the same order
        for p, q in zip(self.iter_predecessors(), rhs.iter_predecessors()):
            # Should we require identical types?
            if not isinstance(p, type(q)) and not isinstance(q, type(p)):
                return NOT

        return MAYBE

    def reset_state(self, Py_ssize_t index):
        """Reset the state of a symbol and any successor symbols.

        Args:
            index: Index of the state to reset.

        Examples:
            This example sets two states on a symbol with two successor symbols
            and resets just one state.

            >>> from dwave.optimization import Model
            >>> model = Model()
            >>> lsymbol, lsymbol_lists = model.disjoint_lists(primary_set_size=5, num_disjoint_lists=2)
            >>> with model.lock():
            ...     model.states.resize(2)
            ...     lsymbol.set_state(0, [[0, 4], [1, 2, 3]])
            ...     lsymbol.set_state(1, [[3, 4], [0, 1, 2]])
            ...     print(f"state 0: {lsymbol_lists[0].state(0)} and {lsymbol_lists[1].state(0)}")
            ...     print(f"state 1: {lsymbol_lists[0].state(1)} and {lsymbol_lists[1].state(1)}")
            ...     lsymbol.reset_state(0)
            ...     print("After reset:")
            ...     print(f"state 0: {lsymbol_lists[0].state(0)} and {lsymbol_lists[1].state(0)}")
            ...     print(f"state 1: {lsymbol_lists[0].state(1)} and {lsymbol_lists[1].state(1)}")
            state 0: [0. 4.] and [1. 2. 3.]
            state 1: [3. 4.] and [0. 1. 2.]
            After reset:
            state 0: [0. 1. 2. 3. 4.] and []
            state 1: [3. 4.] and [0. 1. 2.]
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
            other: Another symbol.

        Returns:
            True if the two symbols share memory.
        """
        cdef Symbol other_
        try:
            other_ = other
        except TypeError:
            return False
        return not self.expired() and self.id() == other_.id()

    def _state_from_zipfile(self, zf, directory, Py_ssize_t state_index):
        # unlike node serialization, by default we raise an error because if
        # this is being called, it must have a state
        raise NotImplementedError(f"{type(self).__name__} has not implemented state deserialization")

    def _state_into_zipfile(self, zf, directory, Py_ssize_t state_index):
        # unlike node serialization, by default we raise an error because if
        # this is being called, it must have a state
        raise NotImplementedError(f"{type(self).__name__} has not implemented state serialization")

    def state_size(self):
        """Return an estimated size, in bytes, of a symbol's state.

        The number of bytes returned by this method is only an estimate. Some
        symbols hold additional information that is not accounted for.

        For most symbols, which are arrays, this method is
        subclassed by the :class:`~dwave.optimization.model.ArraySymbol
        class's :meth:`~dwave.optimization.model.ArraySymbol.state_size`
        method.

        See also:
            :meth:`ArraySymbol.state_size()` An estimate of the size of an array
            symbol's state.

            :meth:`Model.state_size()` An estimate of the size of a model's
            state.

            
        """
        return 0

    def topological_index(self):
        """Topological index of the symbol.

        Return ``None`` if the model is not topologically sorted.

        Examples:
            This example prints the indices of a two-symbol model.

            >>> from dwave.optimization import Model
            >>> model = Model()
            >>> i = model.integer(100, lower_bound=20)
            >>> sum_i = i.sum()
            >>> with model.lock():
            ...     for symbol in model.iter_symbols():
            ...         print(f"Symbol {type(symbol)} is node {symbol.topological_index()}")
            Symbol <class 'dwave.optimization.symbols.IntegerVariable'> is node 0
            Symbol <class 'dwave.optimization.symbols.Sum'> is node 1
        """
        index = self.node_ptr.topological_index()
        return index if index >= 0 else None

def _split_indices(indices):
    """Given a set of indices, made up of slices, integers, and array symbols,
    create two consecutive indexing operations that can be passed to
    BasicIndexing and AdvancedIndexing respectively.
    """
    # this is pure-Python and could be moved out of this .pyx file at some point

    basic_indices = []
    advanced_indices = []

    for index in indices:
        if isinstance(index, numbers.Integral):
            # Only basic handles numeric indices and it removes the axis so
            # only one gets the index.
            basic_indices.append(index)
        elif isinstance(index, slice):
            if index.start is None and index.stop is None and index.step is None:
                # empty slice, both handle it
                basic_indices.append(index)
                advanced_indices.append(index)
            else:
                # Advanced can only handle empty slices, so we do basic first
                basic_indices.append(index)
                advanced_indices.append(slice(None))
        elif isinstance(index, (ArraySymbol, np.ndarray)):
            # Only advanced handles arrays, it preserves the axis so basic gets
            # an empty slice.
            # We allow np.ndarray here for testing purposes. They are not (yet)
            # natively handled by AdvancedIndexingNode.
            basic_indices.append(slice(None))
            advanced_indices.append(index)

        else:
            # this should be checked by the calling function, but just in case
            raise RuntimeError("unexpected index type")

    return tuple(basic_indices), tuple(advanced_indices)


# Ideally this wouldn't subclass Symbol, but Cython only allows a single
# extension base class, so to support that we assume all ArraySymbols are
# also Symbols (probably a fair assumption)
cdef class ArraySymbol(Symbol):
    """Base class for symbols that can be interpreted as an array."""

    def __init__(self, *args, **kwargs):
        # disallow direct construction of array symbols, they should be constructed
        # via their subclasses.
        raise ValueError("ArraySymbols cannot be constructed directly")

    cdef void initialize_arraynode(self, _Graph model, cppArrayNode* array_ptr) noexcept:
        self.array_ptr = array_ptr
        self.initialize_node(model, array_ptr)

    def __abs__(self):
        from dwave.optimization.symbols import Absolute  # avoid circular import
        return Absolute(self)

    def __add__(self, rhs):
        if isinstance(rhs, ArraySymbol):
            from dwave.optimization.symbols import Add  # avoid circular import
            return Add(self, rhs)

        return NotImplemented

    def __bool__(self):
        # In the future we might want to return a Bool symbol, but __bool__ is so
        # fundamental that I am hesitant to do even that.
        raise ValueError("the truth value of an array symbol is ambiguous")

    def __eq__(self, rhs):
        if isinstance(rhs, ArraySymbol):
            # We could consider returning a Constant(True) is the case that self is rhs

            from dwave.optimization.symbols import Equal # avoid circular import
            return Equal(self, rhs)

        return NotImplemented

    def __getitem__(self, index):
        import dwave.optimization.symbols  # avoid circular import
        if isinstance(index, tuple):
            index = list(index)

            # for all indexing styles, empty slices are padded to fill out the
            # number of dimension
            while len(index) < self.ndim():
                index.append(slice(None))

            if all(isinstance(idx, (slice, numbers.Integral)) for idx in index):
                # Basic indexing
                # https://numpy.org/doc/stable/user/basics.indexing.html#basic-indexing
                return dwave.optimization.symbols.BasicIndexing(self, *index)

            elif all(isinstance(idx, ArraySymbol) or
                     (isinstance(idx, slice) and idx.start is None and idx.stop is None and idx.step is None)
                     for idx in index):
                # Advanced indexing
                # https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing

                return dwave.optimization.symbols.AdvancedIndexing(self, *index)

            elif all(isinstance(idx, (ArraySymbol, slice, numbers.Integral)) for idx in index):
                # Combined indexing
                # https://numpy.org/doc/stable/user/basics.indexing.html#combining-advanced-and-basic-indexing

                # We handle this by doing basic and then advanced indexing. In principal the other
                # order may be more efficient in some cases, but for now let's do the simple thing

                basic_indices, advanced_indices = _split_indices(index)
                basic = dwave.optimization.symbols.BasicIndexing(self, *basic_indices)
                return dwave.optimization.symbols.AdvancedIndexing(basic, *advanced_indices)

            else:
                # todo: consider supporting NumPy arrays directly

                # this error message is chosen to be similar to NumPy's
                raise IndexError("only integers, slices (`:`), and array symbols are valid indices")

        else:
            return self[(index,)]

    def __iadd__(self, rhs):
        # If the user is doing +=, we make the assumption that they will want to
        # do it again, so we jump to NaryAdd
        if isinstance(rhs, ArraySymbol):
            from dwave.optimization.symbols import NaryAdd # avoid circular import
            return NaryAdd(self, rhs)

        return NotImplemented

    def __imul__(self, rhs):
        # If the user is doing *=, we make the assumption that they will want to
        # do it again, so we jump to NaryMultiply
        if isinstance(rhs, ArraySymbol):
            from dwave.optimization.symbols import NaryMultiply # avoid circular import
            return NaryMultiply(self, rhs)

        return NotImplemented

    def __le__(self, rhs):
        if isinstance(rhs, ArraySymbol):
            from dwave.optimization.symbols import LessEqual # avoid circular import
            return LessEqual(self, rhs)

        return NotImplemented

    def __mod__(self, rhs):
        if isinstance(rhs, ArraySymbol):
            from dwave.optimization.symbols import Modulus # avoid circular import
            return Modulus(self, rhs)

        return NotImplemented

    def __mul__(self, rhs):
        if isinstance(rhs, ArraySymbol):
            from dwave.optimization.symbols import Multiply  # avoid circular import
            return Multiply(self, rhs)

        return NotImplemented

    def __neg__(self):
        from dwave.optimization.symbols import Negative  # avoid circular import
        return Negative(self)

    def __pow__(self, rhs):
        cdef Py_ssize_t exponent
        try:
            exponent = rhs
        except TypeError:
            return NotImplemented

        if exponent == 2:
            from dwave.optimization.symbols import Square  # avoid circular import
            return Square(self)
        # check if exponent is an integer greater than 0
        elif isinstance(exponent, numbers.Real) and exponent > 0 and int(exponent) == exponent:
            expanded = itertools.repeat(self, int(exponent))
            out = next(expanded)  # get the first one
            # multiply self by itself exponent times
            for symbol in expanded:
                out *= symbol
            return out
        raise ValueError("only integer exponents of 1 or greater are supported")

    def __sub__(self, rhs):
        if isinstance(rhs, ArraySymbol):
            from dwave.optimization.symbols import Subtract  # avoid circular import
            return Subtract(self, rhs)

        return NotImplemented

    def __truediv__(self, rhs):
        if isinstance(rhs, ArraySymbol):
            from dwave.optimization.symbols import Divide  # avoid circular import
            return Divide(self, rhs)

        return NotImplemented

    def all(self):
        """Create an :class:`~dwave.optimization.symbols.All` symbol.

        The new symbol returns True when all elements evaluate to True.
        """
        from dwave.optimization.symbols import All  # avoid circular import
        return All(self)

    def any(self):
        """Create an :class:`~dwave.optimization.symbols.Any` symbol.

        The new symbol returns True when any elements evaluate to True.

        .. versionadded:: 0.4.1
        """
        from dwave.optimization.symbols import Any  # avoid circular import
        return Any(self)

    def copy(self):
        """Return an array symbol that is a copy of the array.

        See Also:
            :class:`~dwave.optimization.symbols.Copy` Equivalent class.

        .. versionadded:: 0.5.1
        """
        from dwave.optimization.symbols import Copy  # avoid circular import
        return Copy(self)

    def flatten(self):
        """Return an array symbol collapsed into one dimension.

        Equivalent to ``symbol.reshape(-1)``.
        """
        return self.reshape(-1)

    def max(self):
        """Create a :class:`~dwave.optimization.symbols.Max` symbol.

        The new symbol returns the maximum value in its elements.
        """
        from dwave.optimization.symbols import Max  # avoid circular import
        return Max(self)

    def maybe_equals(self, other):
        # note: docstring inherited from Symbol.maybe_equal()
        cdef Py_ssize_t maybe = super().maybe_equals(other)
        cdef Py_ssize_t NOT = 0
        cdef Py_ssize_t MAYBE = 1
        cdef Py_ssize_t DEFINITELY = 2

        if maybe != 1:
            return DEFINITELY if maybe else NOT

        if not isinstance(other, ArraySymbol):
            return NOT

        if self.shape() != other.shape():
            return NOT

        # I guess we don't care about strides

        return MAYBE

    def min(self):
        """Create a :class:`~dwave.optimization.symbols.Min` symbol.

        The new symbol returns the minimum value in its elements.
        """
        from dwave.optimization.symbols import Min  # avoid circular import
        return Min(self)

    def ndim(self):
        """Return the number of dimensions for a symbol."""
        return self.array_ptr.ndim()

    def prod(self, axis=None):
        """Create a :class:`~dwave.optimization.symbols.Prod` symbol.

        The new symbol returns the product of its elements.

        .. versionadded:: 0.5.1
            The ``axis`` keyword argument was added in version 0.5.1.
        """
        import dwave.optimization.symbols

        if axis is not None:
            if not isinstance(axis, numbers.Integral):
                raise TypeError("axis of the prod should be an int")

            if not (0 <= axis < self.ndim()):
                raise ValueError("axis should be 0 <= axis < self.ndim()")

            return dwave.optimization.symbols.PartialProd(self, axis)

        return dwave.optimization.symbols.Prod(self)

    def reshape(self, *shape):
        """Create a :class:`~dwave.optimization.symbols.Reshape` symbol.

        Args:
            shape: Shape of the created symbol.

        The new symbol reshapes without changing the antecedent symbol's
        data.

        Examples:
            This example reshapes a column vector into a row vector.

            >>> from dwave.optimization import Model
            >>> model = Model()
            >>> j = model.integer(3, lower_bound=-10, upper_bound=10)
            >>> j.shape()
            (3,)
            >>> k = j.reshape((1, 3))
            >>> k.shape()
            (1, 3)
        """
        from dwave.optimization.symbols import Reshape  # avoid circular import
        if len(shape) <= 1:
            shape = shape[0]

        if not self.array_ptr.contiguous():
            return Reshape(self.copy(), shape)

        return Reshape(self, shape)
    
    def shape(self):
        """Return the shape of the symbol.

        Examples:
            This example returns the shape of a newly instantiated symbol.

            >>> from dwave.optimization import Model
            >>> model = Model()
            >>> x = model.binary(20)
            >>> x.shape()
            (20,)
        """

        # We could do the whole buffer format thing and return a numpy array
        # but I think it's better to follow NumPy and return a tuple
        shape = self.array_ptr.shape()
        return tuple(shape[i] for i in range(shape.size()))

    def size(self):
        r"""Return the number of elements in the symbol.

        If the symbol has a fixed size, returns that size as an integer.
        Otherwise, returns a :class:`~dwave.optimization.symbols.Size` symbol.

        Examples:
            This example checks the size of a :math:`2 \times 3`
            binary symbol.

            >>> from dwave.optimization import Model
            >>> model = Model()
            >>> x = model.binary((2, 3))
            >>> x.size()
            6

        """
        if self.array_ptr.dynamic():
            from dwave.optimization.symbols import Size
            return Size(self)

        return self.array_ptr.size()

    def state(self, Py_ssize_t index = 0, *, bool copy = True):
        """Return the state of the symbol.

        Args:
            index: Index of the state.

            copy: Currently only True is supported.

        Returns:
            State as a :class:`numpy.ndarray`.

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

        return np.array(StateView(self, index), copy=copy)

    def _state_from_zipfile(self, zf, directory, Py_ssize_t state_index):
        fname = directory + "array.npy"

        # check if there is any state data saved (it can be sparse)
        # todo: test for performance, there may be better ways to check
        # for a file's existence
        try:
            zipinfo = zf.getinfo(fname)
        except KeyError:
            # no state data encoded
            return

        with zf.open(zipinfo, mode="r") as f:
            # todo: consider memmap here if possible
            state = np.load(f, allow_pickle=False)

        # only decisions actually have this method. In the future we should
        # do better error checking etc to handle it
        self.set_state(state_index, state)

    def _state_into_zipfile(self, zf, directory, Py_ssize_t state_index):
        # do this first to get any potential error messages out of the way
        # todo: use a view not a copy
        array = self.state(state_index)

        # then save into the state directory
        with zf.open(directory + "array.npy", mode="w", force_zip64=True) as f:
            np.save(f, array, allow_pickle=False)

    def state_size(self):
        r"""Return an estimate of the size, in bytes, of an array symbol's state.

        For an array symbol, the estimate of the state size is exactly the
        number of bytes needed to encode the array.

        Examples:
            This example returns the size of an integer symbol.
            In this example, the symbol encodes a :math:`5\times4` of integers,
            each represented by a :math:`8` byte float.
            Therefore the estimated state size is :math:`5*4*8 = 160` bytes.

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
            raise RuntimeError("size is unbounded")

        return sizeinfo.max.value() * self.array_ptr.itemsize()

    def strides(self):
        """Return the stride length, in bytes, for traversing a symbol.

        Returns:
            Tuple of the number of bytes to step in each dimension when
            traversing a symbol.

        Examples:
            This example returns the size of an integer symbol.

            >>> from dwave.optimization import Model
            >>> model = Model()
            >>> i = model.integer((2, 3), upper_bound=20)
            >>> i.strides()
            (24, 8)
        """
        strides = self.array_ptr.strides()
        return tuple(strides[i] for i in range(strides.size()))

    def sum(self, axis=None):
        """Create a :class:`~dwave.optimization.symbols.Sum` symbol.

        The new symbol returns the sum of its elements.
        """
        import dwave.optimization.symbols

        if axis is not None:
            if not isinstance(axis, numbers.Integral):
                raise TypeError("axis of the sum should be an int")

            if not (0 <= axis < self.ndim()):
                raise ValueError("axis should be 0 <= axis < self.ndim()")

            return dwave.optimization.symbols.PartialSum(self, axis)

        return dwave.optimization.symbols.Sum(self)

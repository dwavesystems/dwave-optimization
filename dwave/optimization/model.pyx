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

"""Nonlinear models are especially suited for use with decision variables that 
represent a common logic, such as subsets of choices or permutations of ordering. 
For example, in a 
`traveling salesperson problem <https://en.wikipedia.org/wiki/Travelling_salesman_problem>`_ 
permutations of the variables representing cities can signify the order of the 
route being optimized and in a 
`knapsack problem <https://en.wikipedia.org/wiki/Knapsack_problem>`_ the 
variables representing items can be divided into subsets of packed and not 
packed.
"""

import contextlib
import collections.abc
import functools
import itertools
import json
import numbers
import operator
import struct
import tempfile
import weakref
import zipfile

import numpy as np

from cpython cimport Py_buffer
from cython.operator cimport dereference as deref, preincrement as inc
from cython.operator cimport typeid
from libcpp cimport bool
from libcpp.utility cimport move
from libcpp.vector cimport vector

from dwave.optimization.symbols cimport symbol_from_ptr


__all__ = ["Model"]


@contextlib.contextmanager
def locked(model):
    """Context manager that hold a locked model and unlocks it when the context is exited."""
    try:
        yield
    finally:
        model.unlock()


cdef class Model:
    """Nonlinear model. 
    
    The nonlinear model represents a general optimization problem with an 
    :term:`objective function` and/or constraints over variables of various 
    types.
    
    The :class:`.Model` class can contain this model and its methods provide 
    convenient utilities for working with representations of a problem.
    
    Examples:
        This example creates a model for a 
        :class:`flow-shop-scheduling <dwave.optimization.generators.flow_shop_scheduling>`
        problem with two jobs on three machines. 
    
        >>> from dwave.optimization.generators import flow_shop_scheduling
        ...
        >>> processing_times = [[10, 5, 7], [20, 10, 15]]
        >>> model = flow_shop_scheduling(processing_times=processing_times)
    """
    def __init__(self):
        self.states = States(self)

        self._data_sources = []

    def constant(self, array_like):
        r"""Create a constant symbol.

        Args:
            array_like: An |array-like|_ representing a constant. Can be a scalar
                or a NumPy array. If the array's ``dtype`` is ``np.double``, the 
                array is not copied.
                
        Returns:
            A constant symbol. 
            
        Examples:
            This example creates a :math:`1 \times 4`-sized constant symbol
            with the specified values. 
            
            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> time_limits = model.constant([10, 15, 5, 8.5])
        """
        from dwave.optimization.symbols import Constant  # avoid circular import
        return Constant(self, array_like)

    def decision_state_size(self):
        r"""An estimated size, in bytes, of the model's decision states.
        
        Examples:
            This example checks the size of a model with one 
            :math:`10 \times 10`-sized integer symbol.
            
            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> visit_site = model.integer((10, 10))     
            >>> model.decision_state_size()         
            800
        """
        return sum(sym.state_size() for sym in self.iter_decisions())

    def disjoint_bit_sets(self, int primary_set_size, int num_disjoint_sets):
        """Create a disjoint-sets symbol as a decision variable. 
        
        Divides a set of the elements of ``range(primary_set_size)`` into 
        ``num_disjoint_sets`` ordered partitions, stored as bit sets (arrays 
        of length ``primary_set_size``, with ones at the indices of elements
        currently in the set, and zeros elsewhere). The ordering of a set is 
        not semantically meaningful.

        Also creates from the symbol ``num_disjoint_sets`` extra successors 
        that output the disjoint sets as arrays.

        Args:
            primary_set_size: Number of elements in the primary set that are
                partitioned into disjoint sets.
            num_disjoint_sets: Number of disjoint sets.

        Returns:
            A tuple where the first element is the disjoint-sets symbol and 
            the second is a set of its newly added successors.
            
        Examples:
            This example creates a symbol of 10 elements that is divided 
            into 4 sets. 
            
            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> parts_set, parts_subsets = model.disjoint_bit_sets(10, 4)
        """

        from dwave.optimization.symbols import DisjointBitSets, DisjointBitSet  # avoid circular import
        main = DisjointBitSets(self, primary_set_size, num_disjoint_sets)
        sets = [DisjointBitSet(main, i) for i in range(num_disjoint_sets)]
        return main, sets

    def disjoint_lists(self, int primary_set_size, int num_disjoint_lists):
        """Create a disjoint-lists symbol as a decision variable. 
        
        Divides a set of the elements of ``range(primary_set_size)`` into 
        ``num_disjoint_lists`` ordered partitions.

        Also creates ``num_disjoint_lists`` extra successors from the
        symbol that output the disjoint lists as arrays.

        Args:
            primary_set_size: Number of elements in the primary set to 
                be partitioned into disjoint lists.
            num_disjoint_lists: Number of disjoint lists.

        Returns:
            A tuple where the first element is the disjoint-lists symbol 
            and the second is a list of its newly added successor nodes.
            
        Examples:
            This example creates a symbol of 10 elements that is divided 
            into 4 lists. 
            
            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> destinations, routes = model.disjoint_lists(10, 4)
        """

        from dwave.optimization.symbols import DisjointLists, DisjointList  # avoid circular import
        main = DisjointLists(self, primary_set_size, num_disjoint_lists)
        lists = [DisjointList(main, i) for i in range(num_disjoint_lists)]
        return main, lists

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
        if isinstance(file, str):
            with open(file, "rb") as f:
                return cls.from_file(f)

        prefix = b"DWNL"

        read_prefix = file.read(len(prefix))
        if read_prefix != prefix:
            raise ValueError("unknown file type, expected magic string "
                             f"{prefix!r} but got {read_prefix!r} "
                             "instead")

        version = tuple(file.read(2))

        if check_header:
            # we'll need the header values to check later
            header_len = struct.unpack('<I', file.read(4))[0]
            header_data = json.loads(file.read(header_len).decode('ascii'))

        cdef Model model = cls()

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
                    try:
                        _node_subclasses[classname]._from_zipfile(zf, directory, model, predecessors=predecessors)
                    except KeyError:
                        raise ValueError("encoded model has an unsupported node type")

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
        num_states = max(0, min(self.states.size(), max_num_states))

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
                
        See also:
            :meth:`.from_file`, :meth:`.to_file`

        TODO: describe the format
        """
        if not self.is_locked():
            # lock for the duration of the method
            with self.lock():
                return self.into_file(file, max_num_states=max_num_states, only_decision=only_decision)

        if isinstance(file, str):
            with open(file, "wb") as f:
                return self.into_file(
                    f,
                    max_num_states=max_num_states,
                    only_decision=only_decision,
                    )

        version = (0, 1)

        model_info = self._header_data(max_num_states=max_num_states, only_decision=only_decision)
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
            >>> model.add_constraint(i <= c)
            >>> constraints = next(model.iter_constraints())
        """
        for i in range(self._graph.num_constraints()):
            yield symbol_from_ptr(self, self._graph.constraints()[i])

    def iter_decisions(self):
        """Iterate over all decision variables in the model.
        
        Examples:
            This example adds a single decision symbol to a model and iterates over it.
            
            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> i = model.integer()
            >>> c = model.constant(5)
            >>> model.add_constraint(i <= c)
            >>> decisions = next(model.iter_decisions())
        """
        cdef Py_ssize_t num_decisions = self.num_decisions()
        cdef Py_ssize_t seen_decisions = 0

        cdef NodeObserver symbol
        for symbol in self.iter_symbols():
            if 0 <= symbol.node_ptr.topological_index() < num_decisions:
                # we found a decision!
                yield symbol
                seen_decisions += 1

                if seen_decisions >= num_decisions:
                    # we found them all
                    return

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
        for i in range(self._graph.num_nodes()):
            yield symbol_from_ptr(self, self._graph.nodes()[i].get())

    def integer(self, shape=None, lower_bound=None, upper_bound=None):
        r"""Create an integer symbol as a decision variable.
        
        Args:
            shape: Shape of the integer array to create.
    
            lower_bound: Lower bound for the symbol, which is the 
                smallest allowed integer value. If None, the default 
                value is used.
            upper_bound: Upper bound for the symbol, which is the 
                largest allowed integer value. If None, the default 
                value is used.
                
        Returns:
            An integer symbol. 
            
        Examples:
            This example creates a :math:`20 \times 20`-sized integer symbol.
            
            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> i = model.integer((20,20), lower_bound=-100, upper_bound=100)
        """
        from dwave.optimization.symbols import IntegerVariable #avoid circular import
        return IntegerVariable(self, shape, lower_bound, upper_bound)

    def binary(self, shape=None):
        r"""Create a binary symbol as a decision variable.
        
        Args:
            shape: Shape of the binary array to create.
            
        Returns:
            A binary symbol.
            
        Examples:
            This example creates a :math:`1 \times 20`-sized binary symbol.
            
            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> x = model.binary((1,20))
        """
        from dwave.optimization.symbols import BinaryVariable #avoid circular import
        return BinaryVariable(self, shape)

    def list(self, n : int):
        """Create a list symbol as a decision variable.

        Args:
            n: Values in the list are permutations of ``range(n)``.
            
        Returns:
            A list symbol. 
            
        Examples:
            This example creates a list symbol of 200 elements.
            
            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> routes = model.list(200)
        """
        from dwave.optimization.symbols import ListVariable  # avoid circular import
        return ListVariable(self, n)

    def lock(self):
        """Lock the model. 
        
        No new symbols can be added to a locked model.

        Returns:
            A context manager. If the context is subsequently exited then the
            :meth:`.unlock` will be called.
        
        See also:
            :meth:`.is_locked`, :meth:`.unlock`
            
        Examples:
            This example checks the status of a model after locking it and
            subsequently unlocking it.
            
            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> i = model.integer(20, upper_bound=100)
            >>> model.lock()
            >>> model.is_locked()
            True
            >>> model.unlock()
            >>> model.is_locked()
            False

            This example locks a model temporarily with a context manager.

            >>> model = Model()
            >>> with model.lock():
            ...     # no nodes can be added within the context
            ...     print(model.is_locked())
            True
            >>> model.is_locked()
            False
        """
        self._graph.topological_sort()  # does nothing if already sorted, so safe to call always
        self._lock_count += 1

        # note that we do not initialize the nodes or resize the states!
        # We do it lazily for performance

        return locked(self)

    def minimize(self, ArrayObserver value):
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
        self.objective = value

    def add_constraint(self, ArrayObserver value):
        """Add a constraint to the model.
        
        Args:
            value: Value that must evaluate to True for the state 
                of the model to be feasible. 
                
        Examples:
            This example adds a single constraint to a model.
            
            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> i = model.integer()
            >>> c = model.constant(5)
            >>> model.add_constraint(i <= c)
        """
        if value is None:
            raise ValueError("value cannot be None")
        # TODO: shall we accept array valued constraints?
        self._graph.add_constraint(value.array_ptr)

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

    cpdef Py_ssize_t num_constraints(self) noexcept:
        """Number of constraints in the model.
        
        Examples:
            This example checks the number of constraints in the model after 
            adding a couple of constraints.
            
            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> i = model.integer(1)
            >>> c = model.constant([5, -14])
            >>> model.add_constraint(i <= c[0])
            >>> model.add_constraint(c[1] <= i)
            >>> model.num_constraints()
            2
        """
        return self._graph.num_constraints()

    def num_symbols(self):
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

    def quadratic_model(self, ArrayObserver x, quadratic, linear=None):
        """Create a quadratic model from an array and a quadratic model.
        
        Args:
            x: An array.
            
            quadratic: Quadratic values for the quadratic model.
            
            linear: Linear values for the quadratic model.
        
        Returns:
            A quadratic model. 
            
        Examples:
            This example creates a quadratic model.
        
            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> x = model.binary(3)
            >>> Q = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 1): 1, (1, 2): 3, (2, 2): 2}
            >>> qm = model.quadratic_model(x, Q)
            
        """
        from dwave.optimization.symbols import QuadraticModel
        return QuadraticModel(x, quadratic, linear)


    def set(self, Py_ssize_t n, Py_ssize_t min_size = 0, max_size = None):
        """Create a set symbol as a decision variable.

        Args:
            n: Values in the set are subsets of ``range(n)``.
            min_size: Minimum set size. Defaults to ``0``.
            max_size: Maximum set size. Defaults to ``n``.

        Returns:
            A set symbol.
            
        Examples:
            This example creates a set symbol of up to 4 elements
            with values between 0 to 99.
            
            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> destinations = model.set(100, max_size=4)
        """
        from dwave.optimization.symbols import SetVariable  # avoid circular import
        return SetVariable(self, n, min_size, n if max_size is None else max_size)

    def state_size(self):
        """An estimate of the size, in bytes, of all states in the model.

        Iterates over the model's states and totals the sizes of all. 
        
        Examples:
            This example estimates the size of a model's states. 
            
            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> c = model.constant([1, 5, 8.4])
            >>> i = model.integer(20, upper_bound=100)
            >>> model.state_size()     
            184
        """
        return sum(sym.state_size() for sym in self.iter_symbols())

    def to_file(self, **kwargs):
        """Serialize the model to a new file-like object.
        
        See also:
            :meth:`.into_file`, :meth:`.from_file`
        """
        file = tempfile.TemporaryFile(mode="w+b")
        self.into_file(file, **kwargs)
        file.seek(0)
        return file

    def to_networkx(self):
        """Convert the model to a NetworkX graph.
        
        Note:
            Currently requires the installation of a GNU compiler.  
        
        Returns:
            A :obj:`NetworkX <networkx:networkx.Graph>` graph.
            
        Examples:
            This example converts a model to a graph. 
            
            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> c = model.constant(8)
            >>> i = model.integer((20, 30))
            >>> g = model.to_networkx()   # doctest: +SKIP
        """
        # Todo: adapt to use iter_symbols()
        # This whole method will need a re-write, it currently only works with gcc
        # but it is useful for development

        import re
        import networkx

        G = networkx.DiGraph()

        cdef cppNode* ptr
        for i in range(self._graph.num_nodes()):
            ptr = self._graph.nodes()[i].get()

            # this regex is compiler specific! Don't do this for the general case
            match = re.search("\d+([a-zA-z]+Node)", str(typeid(deref(ptr)).name()))
            if not match:
                raise ValueError

            u = (match[1], <long>(ptr))

            G.add_node(u)

            for j in range(ptr.predecessors().size()):
                pptr = ptr.predecessors()[j]

                match = re.search("\d+([a-zA-z]+Node)", str(typeid(deref(pptr)).name()))
                if not match:
                    raise ValueError

                v = (match[1], <long>(pptr))

                G.add_edge(v, u)

        return G

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
            for i in range(self.states.size()):
                # this might actually increase the size of the states in some
                # cases, but that's fine
                self.states._states[i].resize(self.num_decisions())


cdef class States:
    r"""States of a symbol in a model.
    
    States represent assignments of values to a symbol's elements. For 
    example, an :meth:`~Model.integer` symbol of size :math:`1 \times 5` 
    might have state ``[3, 8, 0, 12, 8]``, representing one assignment 
    of values to the symbol.
    
    Examples:
        This example creates a :class:`~dwave.optimization.generators.knapsack` 
        model and manipulates its states to test that it behaves as expected. 

        First, create a model. 
    
        >>> from dwave.optimization import Model
        ...
        >>> model = Model()
        >>> # Add constants
        >>> weights = model.constant([10, 20, 5, 15])
        >>> values = model.constant([-5, -7, -2, -9])
        >>> capacity = model.constant(30)
        >>> # Add the decision variable
        >>> items = model.set(4)
        >>> # add the capacity constraint
        >>> model.add_constraint(weights[items].sum() <= capacity)
        >>> # Set the objective 
        >>> model.minimize(values[items].sum())

        Lock the model to prevent changes to directed acyclic graph. At any 
        time, you can verify the locked state, which is demonstrated here. 
        
        >>> with model.lock():
        ...     model.is_locked()
        True 

        Set a couple of states on the decision variable and verify that the 
        model generates the expected values for the objective. 

        >>> model.states.resize(2)
        >>> items.set_state(0, [0, 1])
        >>> items.set_state(1, [0, 2, 3])
        >>> with model.states.lock():
        ...     print(model.objective.state(0) > model.objective.state(1))
        True

        You can clear the states you set.

        >>> model.states.clear()
        >>> model.states.size()
        0
    """
    def __init__(self, Model model):
        self._model_ref = weakref.ref(model)

    def __len__(self):
        """The number of model states."""
        return self.size()

    cdef void attach_states(self, vector[cppState] states) noexcept:
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
        """
        self.detach_states()

    cdef vector[cppState] detach_states(self):
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
        cdef vector[cppState] states
        self._states.swap(states)
        return move(states)

    def from_file(self, file, *, bool replace = True, check_header = True):
        """Construct states from the given file.

        Args:
            file:
                File pointer to a readable, seekable file-like object encoding
                the states. Strings are interpreted as a file name.
            replace:
                If ``True``, any held states are replaced with those from the file.
                If ``False``, the states are appended.
            check_header:
                Set to ``False`` to skip file-header check.
                
        Returns:
            A model.
        """
        self.resolve()

        if not replace:
            raise NotImplementedError("appending states is not (yet) implemented")

        # todo: we don't need to actually construct a model, but this is nice and
        # abstract. We should performance test and then potentially re-implement
        cdef Model model = Model.from_file(file, check_header=check_header)

        # Check that the model is compatible
        for n0, n1 in zip(model.iter_symbols(), self._model().iter_symbols()):
            # todo: replace with proper node quality testing once we have it
            if not isinstance(n0, type(n1)):
                raise ValueError("cannot load states into a model with mismatched decisions")

        self.attach_states(move(model.states.detach_states()))

    def from_future(self, future, result_hook):
        """Populate the states from the result of a future computation.
        
        A :doc:`Future <oceandocs:docs_cloud/reference/computation>` object is 
        returned by the solver to which your problem model is submitted. This 
        enables asynchronous problem submission. 
        
        Args:
            future: ``Future`` object.
            
            result_hook: Method executed to retrieve the Future.
        """
        self.resize(0)  # always clears self first

        self._future = future
        self._result_hook = result_hook

    def initialize(self):
        """Initialize any uninitialized states."""
        self.resolve()

        cdef Model model = self._model()

        if not model.is_locked():
            raise ValueError("Cannot initialize states of an unlocked model")
        for i in range(self._states.size()):
            self._states[i].resize(model.num_nodes())
            model._graph.initialize_state(self._states[i])

    def into_file(self, file):
        """Serialize the states into an existing  file.

        Args:
            file:
                File pointer to an existing writeable, seekable file-like 
                object encoding a model. Strings are interpreted as a file 
                name.

        TODO: describe the format
        """
        self.resolve()
        return self._model().into_file(file, only_decision=True, max_num_states=self.size())


    cdef Model _model(self):
        """Get a ref-counted Model object."""
        cdef Model m = self._model_ref()
        if m is None:
            raise ReferenceError("accessing the states of a garbage collected model")
        return m

    def resize(self, Py_ssize_t n):
        """Resize the number of states.

        If ``n`` is smaller than the current :meth:`.size()`, 
        states are reduced to the first ``n`` states by removing 
        those beyond. If ``n`` is greater than the current 
        :meth:`.size()`, new uninitialized states are added 
        as needed to reach a size of ``n``.

        Resizing to 0 is not  guaranteed to clear the memory allocated to
        states.
        
        Args:
            n: Required number of states. 
            
        Examples:
            This example adds three uninitialized states to a model.
            
            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> i = model.integer(2)
            >>> model.states.resize(3)
        """
        self.resolve()

        if n < 0:
            raise ValueError("n must be a non-negative integer")

        self._states.resize(n)

    cpdef resolve(self):
        """Block until states are retrieved from any pending future computations.
        
        A :doc:`Future <oceandocs:docs_cloud/reference/computation>` object is 
        returned by the solver to which your problem model is submitted. This 
        enables asynchronous problem submission. 
        """
        if self._future is not None:
            # The existance of _future means that anything we do to the 
            # state will block. So we remove it before calling the hook.
            future = self._future
            self._future = None
            result_hook = self._result_hook
            self._result_hook = None

            result_hook(self._model(), future)

    cpdef Py_ssize_t size(self) except -1:
        """Number of model states.
        
        Examples:
            This example adds three uninitialized states to a model and
            verifies the number of model states.
            
            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> model.states.resize(3)
            >>> model.size()
            3
        """
        self.resolve()
        return self._states.size()

    def to_file(self):
        """Serialize the states to a new file-like object."""
        self.resolve()
        return self._model().to_file(only_decision=True, max_num_states=self.size())


cdef class NodeObserver:
    """Class for nodes in the directed acyclic graph of the model.""" 
    cdef void initialize_node(self, Model model, cppNode* node_ptr) noexcept:
        self.model = model

        self.node_ptr = node_ptr
        self.expired_ptr = node_ptr.expired_ptr()

    def equals(self, other):
        """Compare whether two nodes are identical. 
        
        Args:
            other: A node for comparison. 
        
        Equal nodes represent the same quantity in the model.

        Note that comparing nodes across models is expensive.
        """
        cdef Py_ssize_t maybe = self.maybe_equals(other)
        if maybe != 1:
            return True if maybe else False

        # todo: caching
        return all(p.equals(q) for p, q in zip(self.iter_predecessors(), other.iter_predecessors()))

    cpdef bool expired(self) noexcept:
        return deref(self.expired_ptr)

    @classmethod


    def _from_zipfile(cls, zf, directory, Model model, predecessors):
        """Construct a node from a compressed file.

        Args:
            zf:
                File pointer to a compressed file encoding
                a node. Strings are interpreted as a file name.
            directory:
                Directory where the file is located.
            model:
                The relevant :class:`~dwave.optimization.model.Model`.
            predecessors:
                Not currently supported.
        Returns:
            A node.

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

        cdef Py_ssize_t num_states = self.model.states.size()

        if not -num_states <= index < num_states:
            raise ValueError(f"index out of range: {index}")
        if index < 0:  # allow negative indexing
            index += num_states

        self.model.states.resolve()

        # States are extended lazily, so if the state isn't yet long enough then this
        # node's state has not been initialized
        if <Py_ssize_t>(self.model.states._states[index].size()) <= self.node_ptr.topological_index():
            return False

        # Check that the state pointer is not null
        # We need to explicitly cast to evoke unique_ptr's operator bool
        return <bool>(self.model.states._states[index][self.node_ptr.topological_index()])

    def _into_zipfile(self, zf, directory):
        """Store node-specific information to a compressed file.

        Args:
            zf:
                File pointer to a compressed file to store the
                node. Strings are interpreted as a file name.
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
        """Iterate over a node's predecessors in the model.
        
        Examples:
            This example constructs a :math:`b = \sum a` model, where :math:`a` 
            is a multiplication of two symbols, and iterates over the 
            predecessor's of :math:`b` (which is just :math:`a`).
            
            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> i = model.integer((2, 3), upper_bound=20)
            >>> c = model.constant(4)
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
        """Iterate over a node's successors in the model.
        
        Examples:
            This example constructs iterates over the successor nodes
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
        """Compare to another node.
        
        Args:
            other: Another node in the model's directed acyclic graph.
            
        Returns: integer
            Supported return values are:
            
            *   ``0``---Not equal.
            *   ``1``---Might be equal.
            *   ``2``---Are equal.
        """
        cdef Py_ssize_t NOT = 0
        cdef Py_ssize_t MAYBE = 1
        cdef Py_ssize_t DEFINITELY = 2

        # If we're the same object, then we're equal
        if self is other:
            return DEFINITELY

        if not isinstance(other, NodeObserver):
            return NOT

        # Should we require identical types?
        if not isinstance(self, type(other)) and not isinstance(other, type(self)):
            return NOT

        cdef NodeObserver rhs = other

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
        """Reset the state of a node and any successor symbols.
        
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
        """
        if not 0 <= index < self.model.states.size():
            raise ValueError(f"index out of range: {index}")

        if self.node_ptr.topological_index() < 0:
            # unsorted nodes don't have a state to reset
            return

        self.model.states.resolve()

        # make sure the state vector at least contains self
        if <Py_ssize_t>(self.model.states._states[index].size()) <= self.node_ptr.topological_index():
            self.model.states._states[index].resize(self.node_ptr.topological_index() + 1)

        self.model._graph.recursive_reset(self.model.states._states[index], self.node_ptr)

    def shares_memory(self, other):
        """Determine if two symbols share memory.
        
        Args:
            other: Another symbol.
            
        Returns:
            True if the two symbols share memory.
        """
        if not isinstance(other, NodeObserver):
            return False
        cdef NodeObserver rhs = other
        return (
            <bool>(self.node_ptr)                    # Not pointing to a nullptr
            and self.node_ptr == rhs.node_ptr        # Shares an underlying node
            and not <bool>(deref(self.expired_ptr))  # The node is not expired
            )

    def _state_from_zipfile(self, zf, directory, Py_ssize_t state_index):
        # unlike node serialization, by default we raise an error because if
        # this is being called, it must have a state
        raise NotImplementedError(f"{type(self).__name__} has not implemented state deserialization")

    def _state_into_zipfile(self, zf, directory, Py_ssize_t state_index):
        # unlike node serialization, by default we raise an error because if
        # this is being called, it must have a state
        raise NotImplementedError(f"{type(self).__name__} has not implemented state serialization")

    def state_size(self):
        """Return an estimated size, in bytes, of the node's state.
        
        .. note::

            For most symbols, which are arrays, this method is 
            subclassed by the :class:`~dwave.optimization.model.ArrayObserver
            class's :meth:`~dwave.optimization.model.ArrayObserver.state_size`
            method.
        
        Returns:
            Always returns zero (nodes do not have a state).
        """
        # Nodes by default have no state.
        return 0

    def topological_index(self):
        """Topological index of the node.

        Return ``None`` if the model is not topologically sorted.
        
        Examples:
            This example prints the indices of a two-symbol model.
            
            >>> from dwave.optimization import Model
            >>> model = Model()
            >>> i = model.integer(100, lower_bound=20)
            >>> sum_i = i.sum()
            >>> with model.lock():
            >>>     for symbol in model.iter_symbols():
            ...         print(f"Symbol {type(symbol)} is node {symbol.topological_index()}")
            Symbol <class 'dwave.optimization.symbols.IntegerVariable'> is node 0
            Symbol <class 'dwave.optimization.symbols.Sum'> is node 1
        """
        index = self.node_ptr.topological_index()
        return index if index >= 0 else None

# We would really prefer to use NodeObserver.__init_subclass__ to register
# new subclasses for de-serialization. But unfortunately __init_subclass__ does
# not work with Cython cdef classes. So instead we have this function that we
# call once everything has been imported and traverse the subclass DAG.
# For now we raise a RuntimeError for name conflicts. We could handle that in
# various ways if it ever becomes a problem.
_node_subclasses = dict()
def _register_node_subclasses():
    def register(cls):
        if cls.__name__ in _node_subclasses:
            if _node_subclasses[cls.__name__] != cls:
                raise RuntimeError
            return

        _node_subclasses[cls.__name__] = cls

        for subclass in cls.__subclasses__():
            register(subclass)

    for cls in NodeObserver.__subclasses__():
        register(cls)


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
        elif isinstance(index, (ArrayObserver, np.ndarray)):
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


# Ideally this wouldn't subclass NodeObserver, but Cython only allows a single
# extension base class, so to support that we assume all ArrayObservers are
# also NodeObservers (probably a fair assumption)
cdef class ArrayObserver(NodeObserver):
    """Class for nodes of the model that handle arrays."""
    cdef void initialize_array(self, cppArray* array_ptr) noexcept:
        self.array_ptr = array_ptr

    def __abs__(self):
        from dwave.optimization.symbols import Absolute  # avoid circular import
        return Absolute(self)

    def __add__(self, ArrayObserver rhs):
        from dwave.optimization.symbols import Add  # avoid circular import
        return Add(self, rhs)

    def __eq__(self, ArrayObserver rhs):
        from dwave.optimization.symbols import Equal # avoid circular import
        return Equal(self, rhs)

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

            elif all(isinstance(idx, ArrayObserver)
                     or idx.start is None and idx.stop is None and idx.step is None
                     for idx in index):
                # Advanced indexing
                # https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing

                return dwave.optimization.symbols.AdvancedIndexing(self, *index)

            elif all(isinstance(idx, (ArrayObserver, slice, numbers.Integral)) for idx in index):
                # Combined indexing
                # https://numpy.org/doc/stable/user/basics.indexing.html#combining-advanced-and-basic-indexing

                # We handle this by doing basic and then advanced indexing. In principal the other
                # order may be more efficient in some cases, but for now let's do the simple thing

                basic_indices, advanced_indices = _split_indices(index)
                basic = dwave.optimization.symbols(self, *basic_indices)
                return dwave.optimization.symbols(basic, *advanced_indices)

            else:
                # todo: consider supporting NumPy arrays directly

                # this error message is chosen to be similar to NumPy's
                raise IndexError("only integers, slices (`:`), and array symbols are valid indices")

        else:
            return self[(index,)]

    def __le__(self, ArrayObserver rhs):
        from dwave.optimization.symbols import LessEqual # avoid circular import
        return LessEqual(self, rhs)

    def __mul__(self, ArrayObserver rhs):
        from dwave.optimization.symbols import Multiply  # avoid circular import
        return Multiply(self, rhs)

    def __neg__(self):
        from dwave.optimization.symbols import Negative  # avoid circular import
        return Negative(self)

    def __pow__(self, Py_ssize_t exponent):
        if exponent == 2:
            from dwave.optimization.symbols import Square  # avoid circular import
            return Square(self)
        raise NotImplementedError("only squaring is currently supported")

    def __sub__(self, ArrayObserver rhs):
        from dwave.optimization.symbols import Subtract  # avoid circular import
        return Subtract(self, rhs)

    def all(self):
        """Create an :class:`~dwave.optimization.symbols.All` symbol.
        
        The new symbol returns True when all elements evaluate to True.
        """
        from dwave.optimization.symbols import All  # avoid circular import
        return All(self)

    def max(self):
        """Create a :class:`~dwave.optimization.symbols.Max` symbol.
        
        The new symbol returns the maximum value in its elements.
        """
        from dwave.optimization.symbols import Max  # avoid circular import
        return Max(self)

    def maybe_equals(self, other):
        """Compare to another symbol.
        
        Args:
            other: Another symbol in the model.
            
        Returns:
            True if the two symbols might be equal.
            
        Examples:
            This example compares 
            :class:`~dwave.optimization.symbols.IntegerVariable` symbols
            of different sizes.
            
            >>> from dwave.optimization import Model
            >>> i = model.integer(3, lower_bound=0, upper_bound=20)
            >>> j = model.integer(3, lower_bound=-10, upper_bound=10)
            >>> k = model.integer(5, upper_bound=55)
            >>> i.maybe_equals(j)
            1
            >>> i.maybe_equals(k)
            0
        """
        cdef Py_ssize_t maybe = super().maybe_equals(other)
        if maybe != 1:
            return True if maybe else False

        cdef Py_ssize_t NOT = 0
        cdef Py_ssize_t MAYBE = 1

        if not isinstance(other, ArrayObserver):
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

    def prod(self):
        """Create a :class:`~dwave.optimization.symbols.Prod` symbol.
        
        The new symbol returns the product of its elements.
        """
        from dwave.optimization.symbols import Prod  # avoid circular import
        return Prod(self)

    def reshape(self, *shape):
        """Create a :class:`~dwave.optimization.symbols.Reshape` symbol.
        
        Args:
            shape: Shape of the created symbol.
        
        The new symbol reshapes without changing the antecedent symbol's 
        data.
        
        Examples:
            This example reshapes a row vector into a column vector.
            
            >>> from dwave.optimization import Model
            >>> model = Model()
            >>> j = model.integer(3, lower_bound=-10, upper_bound=10)
            >>> j.shape()
            (3,)
            >>> k = j.reshape((3,1))
            >>> k.shape()
            (1, 3)
        """
        from dwave.optimization.symbols import Reshape  # avoid circular import
        if len(shape) > 1:
            return Reshape(self, shape)
        else:
            return Reshape(self, shape[0])

    def sum(self):
        """Create a :class:`~dwave.optimization.symbols.Sum` symbol.
        
        The new symbol returns the sum of its elements.
        """
        from dwave.optimization.symbols import Sum  # avoid circular import
        return Sum(self)

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

        ``-1`` indicates a variable number of elements.
        
        Examples:
            This example checks the size of a :math:`2 \times 3`
            binary symbol.
            
            >>> from dwave.optimization import Model
            >>> model = Model()
            >>> x = model.binary((2, 3))
            >>> x.size()
            6
        """
        return self.array_ptr.size()

    def state(self, Py_ssize_t index = 0, *, bool copy = True):
        """Return the state of the node.

        Args:
            index: Index of the state.
            
            copy: Currently only True is supported.

        Returns:
            State as a :class:`numpy.ndarray`.

        Examples:
            This example prints a node two states: initialized 
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
        """Return an estimated byte-size of the state.
        
        Examples:
            This example returns the size of an integer symbol.
            
            >>> from dwave.optimization import Model
            >>> model = Model()
            >>> i = model.integer(2, lower_bound=0, upper_bound=20)
            >>> i.state_size()
            16
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

    def ndim(self):
        """Return the number of dimensions for a symbol."""
        return self.array_ptr.ndim()


cdef class StateView:
    def __init__(self, ArrayObserver symbol, Py_ssize_t index):
        self.symbol = symbol
        self.index = index

        # we're assuming this object is being created because we want to access
        # the state, so let's go ahead and create the state if it's not already
        # there
        symbol.model.states.resolve()
        symbol.model._graph.recursive_initialize(symbol.model.states._states[index], symbol.node_ptr)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        # todo: inspect/respect/test flags
        self.symbol.model.states.resolve()

        cdef cppArray* ptr = self.symbol.array_ptr

        buffer.buf = <void*>(ptr.buff(self.symbol.model.states._states.at(self.index)))
        buffer.format = <char*>(ptr.format().c_str())
        buffer.internal = NULL
        buffer.itemsize = ptr.itemsize()
        buffer.len = ptr.len(self.symbol.model.states._states.at(self.index))
        buffer.ndim = ptr.ndim()
        buffer.obj = self
        buffer.readonly = 1  # todo: consider loosening this requirement
        buffer.shape = <Py_ssize_t*>(ptr.shape(self.symbol.model.states._states.at(self.index)).data())
        buffer.strides = <Py_ssize_t*>(ptr.strides().data())
        buffer.suboffsets = NULL

        self.symbol.model.states._view_count += 1

    def __releasebuffer__(self, Py_buffer *buffer):
        self.symbol.model.states._view_count -= 1

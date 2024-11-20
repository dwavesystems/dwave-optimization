from typing import Optional
from dwave.optimization.graph_manager import (
    _GraphManager,
    ArraySymbol,
)

__all__ = ["Model"]


class Model(_GraphManager):
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
        super().__init__()

    def add_constraint(self, value: ArraySymbol):
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
        return self._add_constraint(value)

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
        from dwave.optimization.symbols import BinaryVariable  # avoid circular import
        return BinaryVariable(self, shape)

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
        return self._constant(array_like)

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
        return self._decision_state_size()

    def disjoint_bit_sets(self, primary_set_size: int, num_disjoint_sets: int):
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
                partitioned into disjoint sets. Must be non-negative.
            num_disjoint_sets: Number of disjoint sets. Must be positive.

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

        # avoid circular import
        from dwave.optimization.symbols import DisjointBitSets, DisjointBitSet

        main: DisjointBitSets = DisjointBitSets(self, primary_set_size, num_disjoint_sets)
        sets = tuple(DisjointBitSet(main, i) for i in range(num_disjoint_sets))
        return main, sets

    def disjoint_lists(self, primary_set_size: int, num_disjoint_lists: int):
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
        main: DisjointLists = DisjointLists(self, primary_set_size, num_disjoint_lists)
        lists = [DisjointList(main, i) for i in range(num_disjoint_lists)]
        return main, lists

    def feasible(self, index: int = 0):
        """Check the feasibility of the state at the input index.

        Args:
            index: index of the state to check for feasibility.

        Returns:
            Feasibility of the state.

        Examples:
            This example demonstrates checking the feasibility of a simple model with
            feasible and infeasible states.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> b = model.binary()
            >>> model.add_constraint(b) # doctest: +ELLIPSIS
            <dwave.optimization.BinaryVariable at ...>
            >>> model.states.resize(2)
            >>> b.set_state(0, 1) # Feasible
            >>> b.set_state(1, 0) # Infeasible
            >>> with model.lock():
            ...     model.feasible(0)
            True
            >>> with model.lock():
            ...     model.feasible(1)
            False
        """
        return all(sym.state(index) for sym in self.iter_constraints())

    @classmethod
    def from_file(cls, file, *, check_header=True):
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
        return cls._from_file(file, check_header=check_header)

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
        from dwave.optimization.symbols import IntegerVariable  # avoid circular import
        return IntegerVariable(self, shape, lower_bound, upper_bound)

    def into_file(self, file, *, max_num_states: int = 0, only_decision: bool = False):
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
        return self._into_file(file, max_num_states=max_num_states, only_decision=only_decision)

    def is_locked(self):
        """Lock status of the model.

        No new symbols can be added to a locked model.

        See also:
            :meth:`.lock`, :meth:`.unlock`
        """
        return self._is_locked()

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
        return self._iter_constraints()

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
        return self._iter_decisions()

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
        return self._iter_symbols()

    def list(self, n: int):
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
            >>> cntx = model.lock()
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

        return self._lock()

    def minimize(self, value: ArraySymbol):
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
        self._minimize(value)

    def num_constraints(self):
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
        return self._num_constraints()

    def num_decisions(self):
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
        return self._num_decisions()

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
        return self._num_edges()

    def num_nodes(self):
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
        return self._num_nodes()

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
        return self._num_symbols()

    @property
    def objective(self):
        """Objective to be minimized.

        Examples:
            This example prints the value of the objective of a model representing
            the simple polynomial, :math:`y = i^2 - 4i`, for a state with value
            :math:`i=2.0`.

            >>> from dwave.optimization import Model
            ...
            >>> model = Model()
            >>> i = model.integer(lower_bound=-5, upper_bound=5)
            >>> c = model.constant(4)
            >>> y = i**2 - c*i
            >>> model.minimize(y)
            >>> with model.lock():
            ...     model.states.resize(1)
            ...     i.set_state(0, 2.0)
            ...     print(f"Objective = {model.objective.state(0)}")
            Objective = -4.0
        """
        return self._objective

    def quadratic_model(self, x: ArraySymbol, quadratic, linear=None):
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
        return self._remove_unused_symbols()

    def set(self, n: int, min_size: int = 0, max_size: Optional[int] = None):
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

    @property
    def states(self):
        """States of the model.

        :ref:`States <intro_optimization_states>` represent assignments of values
        to a symbol.

        See also:
            :ref:`States methods <optimization_models>` such as
            :meth:`~dwave.optimization.model.States.size` and
            :meth:`~dwave.optimization.model.States.resize`.
        """
        return self._states

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
        return self._state_size()

    def to_file(self, **kwargs):
        """Serialize the model to a new file-like object.

        See also:
            :meth:`.into_file`, :meth:`.from_file`
        """
        return self._to_file(**kwargs)

    def to_networkx(self):
        """Convert the model to a NetworkX graph.

        Returns:
            A :obj:`NetworkX <networkx:networkx.MultiDiGraph>` graph.

        Examples:
            This example converts a model to a graph.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> one = model.constant(1)
            >>> two = model.constant(2)
            >>> i = model.integer()
            >>> model.minimize(two * i - one)
            >>> G = model.to_networkx()  # doctest: +SKIP

            One advantage of converting to NetworkX is the wide availability
            of drawing tools. See NetworkX's
            `drawing <https://networkx.org/documentation/stable/reference/drawing.html>`_
            documentation.

            This example uses `DAGVIZ <https://wimyedema.github.io/dagviz/>`_ to
            draw the NetworkX graph created in the example above.

            >>> import dagviz                      # doctest: +SKIP
            >>> r = dagviz.render_svg(G)           # doctest: +SKIP
            >>> with open("model.svg", "w") as f:  # doctest: +SKIP
            ...     f.write(r)

            This creates the following image:

            .. figure:: /_images/to_networkx_example.svg
               :width: 500 px
               :name: dwave-optimization-to-networkx-example
               :alt: Image of NetworkX Directed Graph

        """
        return self._to_networkx()

    def unlock(self):
        """Release a lock, decrementing the lock count.

        Symbols can be added to unlocked models only.

        See also:
            :meth:`.is_locked`, :meth:`.lock`
        """
        return self._unlock()

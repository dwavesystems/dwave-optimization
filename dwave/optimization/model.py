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

from __future__ import annotations

import collections
import contextlib
import functools
import hashlib
import numpy as np
import tempfile
import typing

from dwave.optimization._model import ArraySymbol, _Graph, Symbol
from dwave.optimization.states import States

if typing.TYPE_CHECKING:
    import numpy.typing

    from dwave.optimization.symbols import *

    _ShapeLike: typing.TypeAlias = typing.Union[int, collections.abc.Sequence[int]]

__all__ = ["Model"]


@contextlib.contextmanager
def locked(model: _Graph):
    """Context manager that hold a locked model and unlocks it when the context is exited."""
    try:
        yield
    finally:
        model.unlock()


class _ConstantCache:
    def __init__(self, model):
        self.model = model
        self.constant_cache: dict[bytes, Constant] = dict()

    def __call__(self, array_like: numpy.typing.ArrayLike):
        r"""Create a constant symbol.

        See :meth:`~Model.constant`.
        """
        from dwave.optimization.symbols import Constant  # avoid circular import

        array = np.asarray_chkfinite(array_like, dtype=np.double, order="C")

        # Hash the incoming array to a unique key for use in the constant cache.
        # Using BLAKE2b as it is the fastest hash available in hashlib.
        # Note that we don't care about cryptographic guarantees BLAKE2b or the other
        # hashing algorithms that hashlib provides.
        h = hashlib.blake2b()
        h.update(np.atleast_1d(array).view(dtype=np.byte))
        h.update(b"|" + str(array.shape).encode())
        hash_val = h.digest()

        if hash_val in self.constant_cache:
            assert np.all(array == self.constant_cache[hash_val].state())
            return self.constant_cache[hash_val]

        constant = Constant(self.model, array_like)
        self.constant_cache[hash_val] = constant
        return constant

    def clear_cache(self):
        """Clear the constant cache. Subsequent calls to .constant() will create
        new symbols.
        """
        self.constant_cache.clear()

    # If additional methods are added, don't forget to make them available before
    # the cache has been instantiated! See comment below .constant(...) method
    # definition.


class Model(_Graph):
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

    states: States
    """States of the model.

    :ref:`States <opt_model_construction_nl_states>` represent assignments of
    values to a symbol.

    See also:
        :ref:`States methods <optimization_models>` such as
        :meth:`~dwave.optimization.model.States.size` and
        :meth:`~dwave.optimization.model.States.resize`.
    """

    def __init__(self):
        self._objective = None
        self.states = States(self)

    @property
    def objective(self) -> typing.Optional[ArraySymbol]:
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

    @objective.setter
    def objective(self, value: ArraySymbol):
        self.minimize(value)

    def binary(self, shape: typing.Optional[_ShapeLike] = None,
               lower_bound: typing.Optional[np.typing.ArrayLike] = None,
               upper_bound: typing.Optional[np.typing.ArrayLike] = None) -> BinaryVariable:
        r"""Create a binary symbol as a decision variable.

        Args:
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
            >>> model = Model()
            >>> x = model.binary((20, 30))
            >>> type(x)
            <class 'dwave.optimization.symbols.numbers.BinaryVariable'>

            This example adds a :math:`5`-sized binary symbol and index-wise
            bounds to a model.

            >>> from dwave.optimization.model import Model
            >>> import numpy as np
            >>> model = Model()
            >>> b = model.binary(5, lower_bound=[-1, 0, 1, 0, 1], upper_bound=[1, 0, 1, 1, 1])
            >>> np.all([0, 0, 1, 0, 1] == b.lower_bound())
            np.True_
            >>> np.all([1, 0, 1, 1, 1] == b.upper_bound())
            np.True_

            This example adds a :math:`2`-sized binary symbol with a scalar lower
            bound and index-wise upper bounds to a model.

            >>> from dwave.optimization.model import Model
            >>> import numpy as np
            >>> model = Model()
            >>> b = model.binary(2, lower_bound=-1.1, upper_bound=[1.1, 0.9])
            >>> np.all([0, 0] == b.lower_bound())
            np.True_
            >>> np.all([1, 0] == b.upper_bound())
            np.True_

        See Also:
            :class:`~dwave.optimization.symbols.numbers.BinaryVariable`: equivalent symbol.

        .. versionchanged:: 0.6.7
            Beginning in version 0.6.7, user-defined bounds and index-wise
            bounds are supported.
        """
        from dwave.optimization.symbols import BinaryVariable  # avoid circular import
        return BinaryVariable(self, shape, lower_bound, upper_bound)

    def constant(self, array_like: numpy.typing.ArrayLike) -> Constant:
        r"""Create a constant symbol.

        To avoid redundancy, ``Constant``\s are cached. Repeated calls to
        ``.constant(array)`` with the same array will result in the same
        ``Constant``.
        The cache can be cleared by calling ``model.constant.clear_cache()``.

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

        See Also:
            :class:`~dwave.optimization.symbols.Constant`: equivalent symbol.

        .. versionchanged:: 0.6.4
            Beginning in version 0.6.4, constants are cached. Also known as
            `memoization <https://en.wikipedia.org/wiki/Memoization>`_.
        """
        self.__dict__["constant"] = cache = _ConstantCache(self)
        return cache(array_like)

    # Make sure the clear_cache method is available even if `.constant()` has never
    # been called. An edge case for sure, but an easy one to support.
    constant.clear_cache = functools.update_wrapper(lambda: None, _ConstantCache.clear_cache)

    def disjoint_bit_sets(
            self,
            primary_set_size: int,
            num_disjoint_sets: int,
            ) -> tuple[DisjointBitSets, tuple[DisjointBitSet, ...]]:
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

        main = DisjointBitSets(self, primary_set_size, num_disjoint_sets)
        sets = tuple(DisjointBitSet(main, i) for i in range(num_disjoint_sets))
        return main, sets

    def disjoint_lists(
            self,
            primary_set_size: int,
            num_disjoint_lists: int,
            ) -> tuple[DisjointLists, tuple[DisjointList, ...]]:
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

    def feasible(self, index: int = 0) -> bool:
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
            >>> _ = model.add_constraint(b)
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

    def input(
        self,
        shape: tuple[int, ...] = (),
        lower_bound: typing.Optional[float] = -float("inf"),
        upper_bound: typing.Optional[float] = float("inf"),
        integral: typing.Optional[bool] = None,
    ) -> Input:
        """Create an "input" symbol.

        An input symbol functions similarly to a decision variable,
        in that it takes no predecessors, but its state will always be set manually
        (not by any solver). Used as a placeholder for input to a model.

        The shape of the output array is fixed at initialization and cannot be changed.

        Provided bounds and integrality are used to supply information for
        min/max/integral/logical properties of the resulting node, and will be used to
        validate the state when set manually.

        Args:
            shape: the shape of the output array.
            lower_bound: lower bound on any possible output of the node.
            upper_bound: upper bound on any possible output of the node.
            integral: whether the output of the node should always be integral.

        Returns:
            An input symbol.

        Examples:
            This example creates two input symbols and an integer decision symbol
            and then sets the objective to the sum of all of them.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> x = model.input()
            >>> y = model.input(lower_bound=8, upper_bound=10, integral=True)
            >>> z = model.integer()
            >>> model.minimize(x + y + z)

        .. versionadded:: 0.6.2
        """
        # avoid circular import
        from dwave.optimization.symbols import Input

        return Input(
            self,
            shape=shape,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            integral=integral
        )

    def integer(
            self,
            shape: typing.Optional[_ShapeLike] = None,
            lower_bound: typing.Optional[numpy.typing.ArrayLike] = None,
            upper_bound: typing.Optional[numpy.typing.ArrayLike] = None,
            ) -> IntegerVariable:
        r"""Create an integer symbol as a decision variable.

        Args:
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
            >>> model = Model()
            >>> i = model.integer(25, upper_bound=100)
            >>> type(i)
            <class 'dwave.optimization.symbols.numbers.IntegerVariable'>

            This example adds a :math:`5`-sized integer symbol with index-wise
            bounds to a model.

            >>> from dwave.optimization.model import Model
            >>> import numpy as np
            >>> model = Model()
            >>> i = model.integer(5, lower_bound=[-1, 0, 3, 0, 2], upper_bound=[1, 2, 3, 4, 5])
            >>> np.all([-1, 0, 3, 0, 2] == i.lower_bound())
            np.True_
            >>> np.all([1, 2, 3, 4, 5] == i.upper_bound())
            np.True_

            This example adds a :math:`2`-sized integer symbol with a scalar lower
            bound and index-wise upper bounds to a model.

            >>> from dwave.optimization.model import Model
            >>> import numpy as np
            >>> model = Model()
            >>> i = model.integer(2, lower_bound=-1.1, upper_bound=[1.1, 2.9])
            >>> np.all([-1, -1] == i.lower_bound())
            np.True_
            >>> np.all([1, 2] == i.upper_bound())
            np.True_

        See Also:
            :class:`~dwave.optimization.symbols.numbers.IntegerVariable`: equivalent symbol.

        .. versionchanged:: 0.6.7
            Beginning in version 0.6.7, user-defined index-wise bounds are
            supported.
        """
        from dwave.optimization.symbols import IntegerVariable  # avoid circular import
        return IntegerVariable(self, shape, lower_bound, upper_bound)

    def list(self, n: int) -> ListVariable:
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

    def lock(self) -> contextlib.AbstractContextManager:
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
        super().lock()
        return locked(self)

    def minimize(self, value: ArraySymbol):
        # inherit the docstring from _Graph
        super().minimize(value)
        self._objective = value

    # dev note: the typing is underspecified, but it would be quite complex to fully
    # specify the linear/quadratic so let's leave it alone for now.
    def quadratic_model(self, x: ArraySymbol, quadratic, linear=None) -> QuadraticModel:
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

    def set(self,
            n: int,
            min_size: int = 0,
            max_size: typing.Optional[int] = None,
            ) -> SetVariable:
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

    def to_file(self, **kwargs) -> typing.BinaryIO:
        """Serialize the model to a new file-like object.

        See also:
            :meth:`.into_file`, :meth:`.from_file`
        """
        file = tempfile.TemporaryFile(mode="w+b")

        # into_file can raise an exception, in which case we close off the
        # tempfile before returning
        try:
            self.into_file(file, **kwargs)
        except Exception:
            file.close()
            raise

        file.seek(0)
        return file

    # NetworkX might not be installed so we just say we return an object
    def to_networkx(self) -> object:
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
        import networkx

        G = networkx.MultiDiGraph()

        # Add the symbols, in order if we happen to be topologically sorted
        G.add_nodes_from(repr(symbol) for symbol in self.iter_symbols())

        # Sanity check. If several nodes map to the same symbol repr we'll see
        # too few nodes in the graph
        if len(G) != self.num_symbols():
            raise RuntimeError("symbol repr() is not unique to the underlying node")

        # Now add the edges
        for symbol in self.iter_symbols():
            for successor in symbol.iter_successors():
                G.add_edge(repr(symbol), repr(successor))

        # Sanity check again. If the repr of symbols isn't unique to the underlying
        # node then we'll see too many nodes in the graph here
        if len(G) != self.num_symbols():
            raise RuntimeError("symbol repr() is not unique to the underlying node")

        # Add the objective if it's present. We call it "minimize" to be
        # consistent with the minimize() function
        if self.objective is not None:
            G.add_edge(repr(self.objective), "minimize")

        # Likewise if we have constraints, add a special node for them
        for symbol in self.iter_constraints():
            G.add_edge(repr(symbol), "constraint(s)")

        return G

    def unlock(self):
        # inherit the docstring from _Graph
        if not self.is_locked():
            return

        super().unlock()

        if not self.is_locked():
            self.states._reset_intermediate_states()

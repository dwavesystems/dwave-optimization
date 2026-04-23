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
import warnings

from dwave.optimization._model import ArraySymbol, _Graph, Symbol
from dwave.optimization.states import States

if typing.TYPE_CHECKING:
    import numpy.typing

    from dwave.optimization.symbols import *

    _ShapeLike: typing.TypeAlias = typing.Union[int, collections.abc.Sequence[int]]

__all__ = ["Model"]


@contextlib.contextmanager
def locked(model: _Graph):
    """Context manager that holds a locked model and unlocks it upon exiting.

    Instantiated through the :meth:`~dwave.optimization.model.Model.lock`
    method.

    Examples:
        This example creates a model, locks it, sets a state for a decision
        variable, and prints a successor symbol.

        >>> from dwave.optimization import Model
            ...
        >>> model = Model()
        >>> i = model.integer(lower_bound=-5, upper_bound=5)
        >>> j = i**2
        >>> with model.lock():
        ...     model.states.resize(1)
        ...     i.set_state(0, 2)
        ...     print(j.state(0))
        4.0
    """
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
        """Clear the cache for constant symbols.

        To prevent redundancy, constants are cached: Repeated calls to the
        :meth:`~dwave.optimization.model.Model.constant` method with the same
        argument, return the first :class:`~dwave.optimization.symbols.Constant`
        instance. After clearing the cache, subsequent such calls create new
        symbols.

        Examples:
            >>> from dwave.optimization import Model
            ...
            >>> model = Model()
            >>> a = model.constant(4)
            >>> b = model.constant(4)
            >>> model.constant.clear_cache()
            >>> c = model.constant(4)
            >>> b is a
            True
            >>> c is a
            False

        See Also:
            :meth:`~dwave.optimization.model.Model.constant`
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

    Examples:
        This example resizes the :class:`~dwave.optimization.states.States`
        class of a simple model to enable the setting of a binary variable. It
        then clears the set state and the allocated memory.

        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> x = model.binary((2, 2))
        >>> model.states.resize(1)
        >>> with model.lock():
        ...     x.set_state(0, [[0, 0], [1, 0]])
        >>> model.states.clear()
        >>> model.states.size()
        0

    See also:
        :class:`~dwave.optimization.states.States` methods to read, save, and
        manipulate the states of a model.
    """

    def __init__(self):
        self._objective = None
        self.states = States(self)

    @property
    def objective(self) -> None | ArraySymbol:
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

        See Also:
            :class:`~dwave.optimization.model.ArraySymbol`: Symbol created by
            the :meth:`.minimize` method

            :meth:`.minimize`,
            :meth:`~dwave.optimization.model.ArraySymbol.state`
        """
        return self._objective

    @objective.setter
    def objective(self, value: ArraySymbol):
        self.minimize(value)

    def binary(self, shape: None | _ShapeLike = None,
               lower_bound: None | np.typing.ArrayLike = None,
               upper_bound: None | np.typing.ArrayLike = None,
               subject_to: None | list[tuple[str, float]] = None,
               axes_subject_to: None | list[tuple[int, str | list[str], float | list[float]]] = None
               ) -> BinaryVariable:
        r"""Create a binary symbol as a decision variable.

        Args:
            shape (optional): Shape of the binary array to create, formatted as
                an integer or a tuple of integers. If None, creates a
                zero-dimensional (scalar) binary variable.
            lower_bound (optional): Lower bound(s) for the symbol. Can be
                scalar (one bound for all variables) or an array (one bound for
                each variable). Non-Boolean values are rounded up to the domain
                [0,1]. If None, the default value of 0 is used.
            upper_bound (optional): Upper bound(s) for the symbol. Can be
                scalar (one bound for all variables) or an array (one bound for
                each variable). Non-Boolean values are rounded down to the domain
                [0,1]. If None, the default value of 1 is used.
            subject_to (optional): Constraint on the sum of the values in the
                array. Must be an array of tuples where each tuple has the form:
                (operator, bound).
                - operator (str): The constraint operator ("<=", "==", or ">=").
                - bound (float): The constraint bound.
                If provided, the sum of values within the array must satisfy
                the corresponding operator–bound pair.
                Note 1: At most one sum constraint may be provided.
                Note 2: If provided, axes_subject_to must None.
            axes_subject_to (optional): Constraint on the sum of the values in
                each slice along a fixed axis in the array. Must be an array of
                tuples where each tuple has the form: (axis, operator(s), bound(s)).
                - axis (int): The axis that the constraint is applied to.
                - operator(s) (str | array[str]): The constraint operator(s)
                ("<=", "==", or ">="). A single operator applies to all slice
                along the axis; an array specifies one operator per slice.
                - bound(s) (float | array[float]): The constraint bound. A
                single value applies to all slices; an array specifies one
                bound per slice.
                If provided, the sum of values within each slice along the
                specified axis must satisfy the corresponding operator–bound pair.
                Note 1: At most one sum constraint may be provided.
                Note 2: If provided, subject_to must None.

        Returns:
            A binary symbol at the root of the :term:`directed acyclic graph`
            for the model.

        Examples:
            This example adds a :math:`20 \times 30`-sized binary variable to a
            model.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> x = model.binary((20, 30))
            >>> x.shape()
            (20, 30)

            This example adds a :math:`2`-sized binary symbol with a scalar
            lower bound and index-wise upper bounds to a model.

            >>> from dwave.optimization.model import Model
            >>> import numpy as np
            >>> model = Model()
            >>> b = model.binary(2, lower_bound=-1.1, upper_bound=[1.1, 0.9])
            >>> b.upper_bound()
            array([1., 0.])

            This example adds a :math:`(2x3)`-sized binary symbol with
            index-wise lower bounds and a sum constraint along axis 1. Let
            x_i (int i : 0 <= i <= 2) denote the sum of the values within
            slice i along axis 1. For each state defined for this symbol:
            (x_0 <= 0), (x_1 == 2), and (x_2 >= 1).

            >>> from dwave.optimization.model import Model
            >>> import numpy as np
            >>> model = Model()
            >>> b = model.binary([2, 3], lower_bound=[[0, 1, 1], [0, 1, 0]],
            ... axes_subject_to=[(1, ["<=", "==", ">="], [0, 2, 1])])
            >>> np.all(b.sum_constraints() == [(1, ["<=", "==", ">="], [0, 2, 1])])
            True

            This example adds a :math:`6`-sized binary symbol such that
            the sum of the values within the array is equal to 2.

            >>> from dwave.optimization.model import Model
            >>> import numpy as np
            >>> model = Model()
            >>> b = model.binary(6, subject_to=[("==", 2)])
            >>> np.all(b.sum_constraints() == [(["=="], [2])])
            True

        See Also:
            :class:`~dwave.optimization.symbols.BinaryVariable`: Generated
            symbol

            :meth:`.constant`, :meth:`.input`, :meth:`.integer`

            :meth:`.iter_decisions`

        .. versionchanged:: 0.6.7
            Beginning in version 0.6.7, user-defined index-wise bounds are
            supported.

        .. versionchanged:: 0.6.13
            Beginning in version 0.6.13, user-defined sum constraints are
            supported.
        """
        from dwave.optimization.symbols import BinaryVariable  # avoid circular import
        return BinaryVariable(self, shape, lower_bound, upper_bound, subject_to, axes_subject_to)

    def constant(self, array_like: numpy.typing.ArrayLike) -> Constant:
        r"""Add a constant to the model.

        To prevent redundancy, constants are cached. Repeated calls to
        :meth:`.constant` with the same ``array_like`` argument, returns the
        first :class:`~dwave.optimization.symbols.Constant` instance.
        The cache can be cleared by calling the
        :meth:`~dwave.optimization.model.Model.constant.clear_cache` method.

        Args:
            array_like: An |array-like|_ representing a constant. Can be a scalar
                or a NumPy array. If the :class:`numpy.dtype` of the array is
                :class:`numpy.double`, the array is not copied.

        Returns:
            A constant symbol at the root of the :term:`directed acyclic graph`
            for the model.

        Examples:
            This example creates a :math:`1 \times 4`-sized constant symbol
            with the specified values.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> time_limits = model.constant([10, 15, 5, 8.5])

        See Also:
            :class:`~dwave.optimization.symbols.Constant`: Generated symbol

            :meth:`.binary`, :meth:`.input`, :meth:`.integer`

            :meth:`.iter_symbols`

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
        """Add a disjoint-sets decision variable to the model.

        Divides a set of the elements of ``range(primary_set_size)`` into
        ``num_disjoint_sets`` ordered partitions, stored as bit sets (arrays
        of length ``primary_set_size``, with ones at the indices of elements
        currently in the set, and zeros elsewhere).

        Also creates from the symbol ``num_disjoint_sets`` successors that
        output the disjoint sets as arrays.

        Args:
            primary_set_size: Number of elements in the primary set that are
                partitioned into disjoint sets. Must be non-negative.
            num_disjoint_sets: Number of disjoint sets. Must be positive.

        Returns:
            A tuple where the first element is a disjoint-sets symbol at the
            root of the :term:`directed acyclic graph` for the model and the
            second is a set of ``num_disjoint_sets`` successors.

        Examples:
            This example creates a symbol of five elements that is divided into
            two sets, which could be part a
            :func:`~dwave.optimization.generators.bin_packing` problem of
            packing five items into two bins with an interest in the number of
            items packed in the first bin (three in the example solution shown
            here).

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> bins_set, bins_subsets = model.disjoint_bit_sets(5, 2)
            >>> in_bin0 = bins_subsets[0].sum()
            >>> with model.lock():
            ...     model.states.resize(1)
            ...     bins_set.set_state(0, [[0, 1, 1, 1, 0], [1, 0, 0, 0, 1]]) # Example solution
            ...     print(in_bin0.state(0))
            3.0

            .. figure:: /_images/disjoint_bit_sets.svg
                :width: 500 px
                :name: dwave-optimization-disjoint-bit-sets-example
                :alt: Image of the model constructed in this example

                Visualization of the model as a :term:`directed acyclic graph`.
                See the :func:`~dwave.optimization.model.Model.to_networkx`
                function for information on visualizing models.

        See Also:
            :class:`~dwave.optimization.symbols.DisjointBitSets`,
            :class:`~dwave.optimization.symbols.DisjointBitSet`: Generated
            symbols

            :meth:`.disjoint_lists_symbol`

            :meth:`.iter_decisions`, :meth:`.iter_successors`
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
        """Add a disjoint-lists decision variable to the model.

        .. deprecated:: 0.6.7

            The return behavior of this method will be changed in
            dwave.optimization 0.8.0. Use :meth:`.disjoint_lists_symbol`.

        Divides a set of the elements of ``range(primary_set_size)`` into
        ``num_disjoint_lists`` ordered partitions.

        Also creates from the symbol ``num_disjoint_lists`` successors  that
        output the disjoint lists as arrays.

        Args:
            primary_set_size: Number of elements in the primary set to
                be partitioned into disjoint lists. Must be non-negative.
            num_disjoint_lists: Number of disjoint lists. Must be positive.

        Returns:
            A tuple where the first element is the disjoint-lists symbol at the
            root of the :term:`directed acyclic graph` for the model
            and the second is a list of ``num_disjoint_lists`` successors.

        Examples:
            This example creates a symbol of 10 elements that is divided
            into 4 lists.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> destinations, routes = model.disjoint_lists(10, 4)

        See Also:
            :class:`~dwave.optimization.symbols.DisjointLists`,
            :class:`~dwave.optimization.symbols.DisjointList`: Generated
            symbols

            :meth:`.disjoint_bit_sets`, :meth:`.disjoint_lists_symbol`

            :meth:`.iter_decisions`, :meth:`.iter_successors`
        """

        warnings.warn(
            "The return behavior of Model.disjoint_lists() is deprecated "
            "since dwave.optimization 0.6.7 and will be changed to the "
            "behavior of Model.disjoint_lists_symbol() in 0.8.0. Use "
            "Model.disjoint_lists_symbol().",
            DeprecationWarning,
        )

        disjoint_lists = self.disjoint_lists_symbol(
            primary_set_size, num_disjoint_lists
        )
        return disjoint_lists, list(disjoint_lists)

    def disjoint_lists_symbol(
            self,
            primary_set_size: int,
            num_disjoint_lists: int,
            ) -> DisjointLists:
        """Create a disjoint-lists symbol as a decision variable.

        Divides a set of the elements of ``range(primary_set_size)`` into
        ``num_disjoint_lists`` ordered partitions.

        Also creates from the symbol ``num_disjoint_lists`` successors  that
        output the disjoint lists as arrays.

        Args:
            primary_set_size: Number of elements in the primary set to
                be partitioned into disjoint lists. Must be non-negative.
            num_disjoint_lists: Number of disjoint lists. Must be positive.

        Returns:
            A disjoint-lists symbol at the root of the
            :term:`directed acyclic graph` for the model.

        Examples:
            This example creates a symbol of 10 elements that is divided
            into 4 lists.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> disjoint_lists = model.disjoint_lists_symbol(10, 4)
            >>> disjoint_lists.primary_set_size()
            10
            >>> disjoint_lists.num_disjoint_lists()
            4
            >>> with model.lock():
            ...    model.states.resize(1)
            ...    disjoint_lists.set_state(0, [[0, 1, 2], [3, 5, 6], [4], [7, 8, 9]])
            ...    for i, disjoint_list in enumerate(disjoint_lists):
            ...        print(f"Element {i}: {disjoint_list.state(0)}")
            Element 0: [0. 1. 2.]
            Element 1: [3. 5. 6.]
            Element 2: [4.]
            Element 3: [7. 8. 9.]

            .. figure:: /_images/disjoint_lists_symbol.svg
                :width: 500 px
                :name: dwave-optimization-disjoint-lists-symbol-example
                :alt: Image of the model constructed in this example

                Visualization of the model as a :term:`directed acyclic graph`.
                See the :func:`~dwave.optimization.model.Model.to_networkx`
                function for information on visualizing models.

        See Also:
            :class:`~dwave.optimization.symbols.DisjointLists`,
            :class:`~dwave.optimization.symbols.DisjointList`: Generated
            symbols

            :meth:`.disjoint_bit_sets`

            :meth:`.iter_decisions`, :meth:`.iter_successors`
        """
        from dwave.optimization.symbols import DisjointLists, DisjointList  # avoid circular import
        disjoint_lists = DisjointLists(self, primary_set_size, num_disjoint_lists)

        # create the DisjointList symbols, which will create the successor nodes, even
        # though we won't use them directly here
        for i in range(num_disjoint_lists):
            DisjointList(disjoint_lists, i)

        return disjoint_lists

    def feasible(self, index: int = 0) -> bool:
        """Check the feasibility of a state.

        Args:
            index: Index of the state to check for feasibility.

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

        See Also:
            :meth:`.iter_constraints`, :meth:`.num_constraints`,
            :meth:`.objective`, :meth:`~dwave.optimization.states.States.size`
        """
        return all(sym.state(index) for sym in self.iter_constraints())

    def input(
        self,
        shape: tuple[int, ...] = (),
        lower_bound: None | float = -float("inf"),
        upper_bound: None | float = float("inf"),
        integral: None | bool = None,
    ) -> Input:
        """Add an input symbol as a placeholder for a decision variable.

        An input symbol functions similarly to a decision variable,
        in that it takes no predecessors, but its state is always set manually
        (and not updated if the model is submitted for solution to a solver).
        Used as a placeholder for input to a model.

        The shape of the output array is fixed at initialization and cannot be
        changed.

        The provided bounds and integrality are used to validate the state when
        set manually; for example, supplied values cannot violate the lower
        bound.

        Args:
            shape: Shape of the output array, formatted as an integer or a tuple
                of integers. If None, creates a zero-dimensional (scalar) input.
            lower_bound: Lower bound on any possible output of the node.
            upper_bound: Upper bound on any possible output of the node.
            integral: Whether the output of the node should always be integral.

        Returns:
            An input symbol at the root of the :term:`directed acyclic graph`
            for the model.

        Examples:
            This example creates an integer decision symbol and an input symbol
            it uses to multiply the sums of the integer symbol's rows.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> i = model.integer(shape=(2, 3), lower_bound=-5, upper_bound=5)
            >>> x = model.input(shape=(2, 1), lower_bound=-2, upper_bound=2, integral=True)
            >>> y = x*i.sum(axis=1)
            >>> with model.lock():
            ...    model.states.resize(1)
            ...    i.set_state(0, [[1, 2, 3], [1, 1, 2]])
            ...    x.set_state(0, [[1], [-1]])
            ...    print(y.state(0))
            [[ 6.  4.]
             [-6. -4.]]

        See Also:
            :class:`~dwave.optimization.symbols.Input`: Generated symbol

            :meth:`.constant`

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
            shape: None | _ShapeLike = None,
            lower_bound: None | numpy.typing.ArrayLike = None,
            upper_bound: None | numpy.typing.ArrayLike = None,
            subject_to: None | list[tuple[str, float]] = None,
            axes_subject_to: None | list[tuple[int, str | list[str], float | list[float]]] = None
               ) -> IntegerVariable:
        r"""Create an integer symbol as a decision variable.

        Args:
            shape (optional): Shape of the integer array to create, formatted as
                an integer or a tuple of integers. If None, creates a
                zero-dimensional (scalar) variable.
            lower_bound (optional): Lower bound(s) for the symbol. Can be
                scalar (one bound for all variables) or an array (one bound for
                each variable). Non-integer values are rounded up. If None, the
                default value, zero, is used.
            upper_bound (optional): Upper bound(s) for the symbol. Can be
                scalar (one bound for all variables) or an array (one bound for
                each variable). Non-integer values are down up. If None, the
                default value is used.
            subject_to (optional): Constraint on the sum of the values in the
                array. Must be an array of tuples where each tuple has the form:
                (operator, bound).
                - operator (str): The constraint operator ("<=", "==", or ">=").
                - bound (float): The constraint bound.
                If provided, the sum of values within the array must satisfy
                the corresponding operator–bound pair.
                Note 1: At most one sum constraint may be provided.
                Note 2: If provided, axes_subject_to must None.
            axes_subject_to (optional): Constraint on the sum of the values in
                each slice along a fixed axis in the array. Must be an array of
                tuples where each tuple has the form: (axis, operator(s), bound(s)).
                - axis (int): The axis that the constraint is applied to.
                - operator(s) (str | array[str]): The constraint operator(s)
                ("<=", "==", or ">="). A single operator applies to all slice
                along the axis; an array specifies one operator per slice.
                - bound(s) (float | array[float]): The constraint bound. A
                single value applies to all slices; an array specifies one
                bound per slice.
                If provided, the sum of values within each slice along the
                specified axis must satisfy the corresponding operator–bound pair.
                Note 1: At most one sum constraint may be provided.
                Note 2: If provided, subject_to must None.
        Returns:
            An integer symbol at the root of the :term:`directed acyclic graph`
            for the model.

        Examples:
            This example adds a :math:`5`-sized integer decision variable with
            scalar bounds to a model, and takes the logarithm of its elements.

            >>> from dwave.optimization.model import Model
            >>> from dwave.optimization.mathematical import log
            ...
            >>> model = Model()
            >>> i = model.integer(5, lower_bound=1, upper_bound=10)
            >>> i.shape()
            (5,)
            >>> a = log(i)
            >>> with model.lock():
            ...    model.states.resize(1)
            ...    i.set_state(0, [[1, 2, 3, 1, 2]])
            ...    print(a.state(0)[0])
            0.0

            This example adds a :math:`2`-sized integer symbol with a scalar
            lower bound and index-wise upper bounds to a model.

            >>> from dwave.optimization.model import Model
            >>> import numpy as np
            >>> model = Model()
            >>> i = model.integer(2, lower_bound=-1.1, upper_bound=[1.1, 2.9])
            >>> np.all([-1, -1] == i.lower_bound())
            True
            >>> np.all([1, 2] == i.upper_bound())
            True

            This example adds a :math:`(2x3)`-sized integer symbol with general
            lower and upper bounds and a sum constraint along axis 1. Let x_i
            (int i : 0 <= i <= 2) denote the sum of the values within
            slice i along axis 1. For each state defined for this symbol:
            (x_0 <= 2), (x_1 <= 4), and (x_2 <= 5).

            >>> from dwave.optimization.model import Model
            >>> import numpy as np
            >>> model = Model()
            >>> i = model.integer([2, 3], lower_bound=1, upper_bound=3,
            ... axes_subject_to=[(1, "<=", [2, 4, 5])])
            >>> np.all(i.sum_constraints() == [(1, ["<="], [2, 4, 5])])
            True

            This example adds a :math:`6`-sized integer symbol such that
            the sum of the values within the array is less than or equal
            to 20.

            >>> from dwave.optimization.model import Model
            >>> import numpy as np
            >>> model = Model()
            >>> i = model.integer(6, subject_to=[("<=", 20)])
            >>> np.all(i.sum_constraints() == [(["<="], [20])])
            True

        See Also:
            :class:`~dwave.optimization.symbols.numbers.IntegerVariable`:
            Generated symbol.

            :meth:`.binary`, :meth:`.constant`, :meth:`.input`

            :meth:`.iter_decisions`

        .. versionchanged:: 0.6.7
            Beginning in version 0.6.7, user-defined index-wise bounds are
            supported.

        .. versionchanged:: 0.6.13
            Beginning in version 0.6.13, user-defined sum constraints are
            supported.
        """
        from dwave.optimization.symbols import IntegerVariable  # avoid circular import
        return IntegerVariable(self, shape, lower_bound, upper_bound, subject_to, axes_subject_to)

    def list(self,
            n: int,
            min_size: None | int = None,
            max_size: None | int = None,
            ) -> ListVariable:
        """Add a list decision variable to the model.

        Permutations of the values in ``range(n)`` as a list.

        Args:
            n: Values in the list are permutations of ``range(n)``.
            min_size: Minimum list size. Defaults to ``max_size``.
            max_size: Maximum list size. Defaults to ``n``.

        Returns:
            A list symbol at the root of the :term:`directed acyclic graph`
            for the model.

        Examples:
            This example creates a list symbol of 200 elements.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> routes = model.list(200)

            This example creates a list symbol with at least 2 elements and at
            most 4 elements with values between 0 to 99. It sets two states of
            the decision variable with different lengths.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> routes = model.list(99, min_size=2, max_size=4)
            >>> with model.lock():
            ...    model.states.resize(2)
            ...    routes.set_state(0, [10, 2, 44])
            ...    routes.set_state(1, [67, 1])

        See Also:
            :class:`~dwave.optimization.symbols.ListVariable`: Generated
            symbol

            :meth:`.disjoint_bit_sets`, :meth:`.disjoint_lists_symbol`

            :meth:`.iter_decisions`

        .. versionchanged:: 0.6.12
            Beginning in version 0.6.12, sub-lists are supported.
        """
        from dwave.optimization.symbols import ListVariable  # avoid circular import

        if max_size is None:
            max_size = n

        if min_size is None:
            min_size = max_size

        return ListVariable(self, n, min_size, max_size)

    def lock(self) -> contextlib.AbstractContextManager:
        """Lock the model.

        No new symbols can be added to a locked model. Unlocked models do not
        allow access to methods such as
        :meth:`~dwave.optimization.model.ArraySymbol.state` and
        :meth:`~dwave.optimization.model.Symbol.topological_index` for
        intermediate (non-decision) variables.

        Returns:
            A context manager. If the context is subsequently exited, the
            :meth:`.unlock` is called.

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

        See also:
            :meth:`.is_locked`, :meth:`.unlock`

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
            max_size: None | int = None,
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

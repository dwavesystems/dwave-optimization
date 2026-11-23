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

"""Non-linear Model implementation.

Nonlinear models are especially suited for use with decision variables that
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

import collections.abc
import contextlib
import functools
import hashlib
import io
import itertools
import json
import os
import struct
import tempfile
import typing
import warnings
import zipfile

import numpy as np

from dwave.optimization._model import ArraySymbol, States, Symbol, _Graph
from dwave.optimization.utilities import _file_object_arg, _lock

if typing.TYPE_CHECKING:
    from dwave.optimization.symbols import *
    from dwave.optimization.typing import (
        ShapeLike,
        _AxesSumsSubjectToType,
        _SumSubjectToType,
    )

__all__ = ["Model"]


DEFAULT_SERIALIZATION_VERSION = (1, 0)
"""A 2-tuple encoding the default serialization format used for serializing models."""

KNOWN_SERIALIZATION_VERSIONS = (
    (0, 1),
    (1, 0),
)
"""A tuple of 2-tuples listing all serialization versions supported."""


@contextlib.contextmanager
def locked(model: _Graph):
    """Context manager that holds a locked model and unlocks it upon exiting.

    Instantiated through the :meth:`~dwave.optimization.model.Model.lock`
    method.

    Examples:
        This example creates a model, locks it, sets a state for a decision
        variable, and prints a successor symbol.

        >>> from dwave.optimization import Model
        >>>
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

    def __call__(self, array_like: np.typing.ArrayLike):
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
            >>>
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
    :term:`objective function` and/or :term:`constraints <constraint>` over
    variables of various types.

    The :class:`.Model` class can contain this model and its methods provide
    convenient utilities for working with representations of a problem.

    Examples:
        This example creates a model for a
        :class:`flow-shop-scheduling <dwave.optimization.generators.flow_shop_scheduling>`
        problem with two jobs on three machines.

        >>> from dwave.optimization.generators import flow_shop_scheduling
        >>>
        >>> processing_times = [[10, 5, 7], [20, 10, 15]]
        >>> model = flow_shop_scheduling(processing_times=processing_times)
    """

    states: States
    """States of the model.

    The :ref:`States <opt_model_construction_nl_states>` class represents
    assignments of values to a symbol.

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
        """Initialize a Model."""
        self.states = States(self)

    def binary(
        self,
        shape: ShapeLike | None = None,
        lower_bound: np.typing.ArrayLike | None = None,
        upper_bound: np.typing.ArrayLike | None = None,
        *,
        sum_subject_to: _SumSubjectToType | None = None,
        axes_sums_subject_to: _AxesSumsSubjectToType | None = None,
    ) -> BinaryVariable:
        r"""Add a binary decision variable to the model.

        A binary symbol is an array of :math:`0, 1` values assigned as a solution
        to the problem being modeled.

        .. tip::
            Your choice of decision variables significantly impacts the solver's
            performance.

            Where possible, prefer more-abstract variable types (e.g.,
            :meth:`.list` or :meth:`.set`) or to derive auxiliary variables
            rather than adding them as explicit decision variables.

            See :ref:`opt_model_construction_nl_guidance`.

        Args:
            shape:
                Shape of the binary array to create, formatted as an integer or
                a tuple of integers.
                If ``None``, creates a zero-dimensional (scalar) binary variable.
            lower_bound:
                Lower bound(s) for the symbol.
                Can be scalar (one bound for all variables) or an array (one
                bound for each variable).
                Non-Boolean values are rounded up to the domain `{0, 1}`.
                If ``None``, the default value of ``0`` is used.

                .. versionadded:: 0.6.7
                    Support for index-wise bounds.
            upper_bound:
                Upper bound(s) for the symbol.
                Can be scalar (one bound for all variables) or an array (one bound for
                each variable).
                Non-Boolean values are rounded down to the domain `{0, 1}`.
                If ``None``, the default value of ``1`` is used.

                .. versionadded:: 0.6.7
                    Support for index-wise bounds.
            sum_subject_to:
                Constraint on the sum of the values in the array.
                Must be a sequence of tuples where each tuple has the form:
                ``(operator, bound)``

                - **operator** (``str``) - The constraint operator
                  (``"<="``, ``"=="``, or ``">="``).
                - **bound** (``float``) - The constraint bound.

                If provided, the sum of values within the array must satisfy
                the corresponding operator–bound pair.
                E.g.,

                .. code-block:: python

                    x = model.binary(10, sum_subject_to=[("<=", 5)])

                enforces

                .. math::

                    \sum_{i=0}^9 x_i \leq 5

                **Currently, at most one sum constraint may be provided.**
                If provided, ``axes_sums_subject_to`` must be ``None``.

                .. versionadded:: 0.6.13
            axes_sums_subject_to:
                Constraint on the sum of the values in each slice along a fixed
                axis in the array.
                Must be a sequence of tuples where each tuple has the form:
                ``(axis, operator(s), bound(s))``

                - **axis** (``int``) - The axis that the constraint is applied to.
                - **operator(s)** (``str`` or ``list[str]``) - The constraint
                  operator (``"<="``, ``"=="``, or ``">="``).
                  A single operator applies to all slice along the axis; an
                  array specifies one operator per slice.
                - **bound(s)** (``float | list[float]``) - The constraint bound. A
                  single value applies to all slices; an array specifies one
                  bound per slice.

                If provided, the sum of values within each slice along the
                specified axis must satisfy the corresponding operator–bound pair.
                E.g.,

                .. code-block:: python

                    x = model.binary(
                        (2, 3),
                        axes_sums_subject_to=[(1, "<=", 2)],
                    )

                enforces

                .. math::

                    \forall i, \sum_{j=0}^2 x_{i,j} \leq 2

                **Currently, at most one sum constraint may be provided.**
                If provided, ``sum_subject_to`` must be ``None``.

                .. versionadded:: 0.6.13

        Returns:
            A binary symbol as a decision variable in the model.

        Examples:
            This example adds a :math:`(20, 30)`-shaped binary variable to a
            model.

            >>> from dwave.optimization.model import Model
            >>>
            >>> model = Model()
            >>> x = model.binary((20, 30))
            >>> x.shape()
            (20, 30)

            This example adds a :math:`(2,)`-shaped binary symbol with a scalar
            lower bound and index-wise upper bounds to a model.

            >>> import numpy as np
            >>> from dwave.optimization.model import Model
            >>>
            >>> model = Model()
            >>> b = model.binary(2, lower_bound=-1.1, upper_bound=[1.1, 0.9])
            >>> b.upper_bound()
            array([1., 0.])

            This example adds a :math:`(2, 3)`-shaped binary symbol with
            index-wise lower bounds and a sum constraint along axis
            :math:`1`. Let :math:`x_i` denote the sum of the values within
            slice :math:`i` along axis :math:`1`. For each state defined for
            this symbol: :math:`(x_0 \leq 0)`, :math:`(x_1 = 2)`, and
            :math:`(x_2 \geq 1)`.

            >>> import numpy as np
            >>> from dwave.optimization.model import Model
            >>>
            >>> model = Model()
            >>> b = model.binary(
            ...     [2, 3],
            ...     lower_bound=[[0, 1, 1], [0, 1, 0]],
            ...     axes_sums_subject_to=[(1, ["<=", "==", ">="], [0, 2, 1])],
            ... )
            >>> np.all(b.sum_constraints() == [(1, ["<=", "==", ">="], [0, 2, 1])])
            True

            This example adds a :math:`(6,)`-shaped binary symbol such that
            the sum of the values within the array is equal to 2.

            >>> import numpy as np
            >>> from dwave.optimization.model import Model
            >>>
            >>> model = Model()
            >>> b = model.binary(6, sum_subject_to=[("==", 2)])
            >>> np.all(b.sum_constraints() == [(["=="], [2])])
            True

        See Also:
            :class:`~dwave.optimization.symbols.BinaryVariable`: Generated
            symbol

            :meth:`.constant`, :meth:`.input`, :meth:`.integer`

            :meth:`.iter_decisions`
        """
        from dwave.optimization.symbols import BinaryVariable  # avoid circular import

        return BinaryVariable(
            self,
            shape,
            lower_bound,
            upper_bound,
            sum_subject_to=sum_subject_to,
            axes_sums_subject_to=axes_sums_subject_to,
        )

    def constant(self, array_like: np.typing.ArrayLike) -> Constant:
        r"""Add a constant to the model.

        A constant symbol is an array of floats used in the model's formulation.

        To prevent redundancy, constants are cached. Repeated calls to
        :meth:`.constant` with the same ``array_like`` argument return the
        first :class:`~dwave.optimization.symbols.Constant` instance. You can
        clear cache by calling the
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

        A disjoint-sets symbol divides a set of elements into ordered
        partitions of bit sets, with ones signifying elements in the set, where
        the placement of these ones is assigned as a solution to the problem
        being modeled.

        The elements are values of ``range(primary_set_size)``. Each of the
        ``num_disjoint_sets`` ordered partitions is a bit set (array) of length
        ``primary_set_size`` with ones at the indices of elements currently in
        the set, and zeros elsewhere. The set is disjoint in that for every
        index :math:`i` between zero and ``primary_set_size``, there is a single
        partition assigned a value of one at that position.

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
            ...     bins_set.set_state(0, [[0, 1, 1, 1, 0], [1, 0, 0, 0, 1]])  # Example solution
            ...     print(in_bin0.state(0))
            3.0

            .. figure:: /_images/optimization/disjoint_bit_sets.svg
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

            :meth:`.disjoint_lists_symbol`, :meth:`.list`, :meth:`.set`

            :meth:`.iter_decisions`, :meth:`.iter_successors`
        """
        # avoid circular import
        from dwave.optimization.symbols import DisjointBitSet, DisjointBitSets

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

        A disjoint-lists symbol divides a set of the elements of
        ``range(primary_set_size)`` into ``num_disjoint_lists`` ordered
        partitions, with the division being assigned as a solution to the
        problem being modeled.

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

            :meth:`.disjoint_bit_sets`, :meth:`.disjoint_lists_symbol`,
            :meth:`.list`, :meth:`.set`

            :meth:`.iter_decisions`, :meth:`.iter_successors`
        """
        warnings.warn(
            "The return behavior of Model.disjoint_lists() is deprecated "
            "since dwave.optimization 0.6.7 and will be changed to the "
            "behavior of Model.disjoint_lists_symbol() in 0.8.0. Use "
            "Model.disjoint_lists_symbol().",
            DeprecationWarning,
        )

        disjoint_lists = self.disjoint_lists_symbol(primary_set_size, num_disjoint_lists)
        return disjoint_lists, tuple(disjoint_lists)

    def disjoint_lists_symbol(
        self,
        primary_set_size: int,
        num_disjoint_lists: int,
    ) -> DisjointLists:
        """Create a disjoint-lists symbol as a decision variable.

        A disjoint-lists symbol divides a set of the elements of
        ``range(primary_set_size)`` into ``num_disjoint_lists`` ordered
        partitions, where the division is assigned as a solution to the problem
        being modeled.

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
            ...     model.states.resize(1)
            ...     disjoint_lists.set_state(0, [[0, 1, 2], [3, 5, 6], [4], [7, 8, 9]])
            ...     for i, disjoint_list in enumerate(disjoint_lists):
            ...         print(f"Element {i}: {disjoint_list.state(0)}")
            Element 0: [0. 1. 2.]
            Element 1: [3. 5. 6.]
            Element 2: [4.]
            Element 3: [7. 8. 9.]

            .. figure:: /_images/optimization/disjoint_lists_symbol.svg
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

            :meth:`.disjoint_bit_sets`, :meth:`.list`, :meth:`.set`

            :meth:`.iter_decisions`, :meth:`.iter_successors`
        """
        from dwave.optimization.symbols import DisjointList, DisjointLists  # avoid circular import

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
            >>> b.set_state(0, 1)  # Feasible
            >>> b.set_state(1, 0)  # Infeasible
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

    @classmethod
    @_file_object_arg(arg_index=1, arg_name="file", mode="rb")  # convert path-like into file-like
    def from_file(
        cls,
        file: str | os.PathLike[str] | typing.BinaryIO,
        *,
        check_header: bool = True,
        substitute: collections.abc.Mapping[str, collections.abc.Callable[..., type]] | None = None,
        lock: bool = False,
    ) -> typing.Self:
        """Construct a model from the given file.

        Args:
            file:
                File pointer to a readable, seekable file-like object encoding
                a model. Strings are interpreted as a file name. Files are not
                rewound to the beginning.
            check_header:
                Whether to check the header.
            substitute:
                A mapping of symbol substitutions to make when loading the file.
                The keys are strings giving the node class name to be substituted.
                The values are callables to create a different node.
                The callable should have the same signature as the substituted
                symbol's constructor.
            lock:
                Whether to return a locked model. Only locked models will include
                any saved intermediate states.

        Returns:
            A model.

        Examples:
            This example serializes a model to a
            `buffered I/O <https://docs.python.org/3/library/io.html#binary-i-o>`_
            object, then creates a new model from that object.

            >>> from dwave.optimization.generators import flow_shop_scheduling
            >>>
            >>> processing_times = [[10, 5, 7], [20, 10, 15]]
            >>> model = flow_shop_scheduling(processing_times=processing_times)
            >>> my_file = model.to_file()
            >>>
            >>> from dwave.optimization import Model
            >>> new_model = Model.from_file(my_file)

        See Also:
            :meth:`.into_file`, :meth:`.to_file`

            :meth:`~dwave.optimization.states.States.from_file` Restores states
            from a file

        .. versionchanged:: 0.6.0
            Add the ``substitute`` and ``lock`` keyword-only arguments.

        """
        from dwave.optimization import symbols

        if substitute is None:
            substitute = dict()
        elif not isinstance(substitute, collections.abc.Mapping):
            raise TypeError("expected substitute to be a mapping of node names to classes")

        version, header_data = cls._from_file_header(file)

        model = cls()

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
                        predecessors.append(model._ith_node(pid))

                    # now make the node
                    directory = f"nodes/{node_id}/"
                    classname = classname.decode("UTF-8").rstrip("\n")

                    # take advanctage of the symbols all being in the same namespace
                    # and the fact that we (currently) encode them all by name
                    node_cls = substitute.get(classname, getattr(symbols, classname, None))

                    if not issubclass(node_cls, Symbol):
                        raise TypeError("encoded model has an unsupported node type")

                    node_cls._from_zipfile(zf, directory, model, predecessors=predecessors)

            objective_buff = zf.read("objective.json")
            if objective_buff:
                objective_id = json.loads(objective_buff)
                if not isinstance(objective_id, int) or objective_id >= model.num_nodes():
                    raise ValueError("objective must be an integer and a valid node id")
                model.minimize(model._ith_node(objective_id))

            for cid in json.loads(zf.read("constraints.json")):
                model.add_constraint(model._ith_node(cid))

            if lock:
                model.lock()

            # Finally load any states from the file.
            model.states._from_zipfile(zf, version=version)

        if check_header:
            expected = model._header_data(only_decision=False)

            if not expected.items() <= header_data.items():
                raise ValueError(
                    "header data does not match the deserialized CQM. "
                    f"Expected {expected!r} to be a subset of {header_data!r}"
                )

        return model

    @staticmethod
    def _from_file_header(file: io.BytesIO) -> tuple[tuple[int, int], dict]:
        """Read/validate the header from a model file.

        Advances the stream position to the end of the header.

        Returns:
            The serialization version as a 2-tuple.

            The header data as a dict.
        """
        prefix = b"DWNL"

        read_prefix = file.read(len(prefix))
        if read_prefix != prefix:
            raise ValueError(
                "Unknown file type. Expected magic string "
                f"{prefix!r} but received {read_prefix!r} "
                "instead"
            )

        version: tuple[int, int] = tuple(file.read(2))

        if version not in KNOWN_SERIALIZATION_VERSIONS:
            raise ValueError(
                "Unknown serialization format. Expected one of "
                f"{KNOWN_SERIALIZATION_VERSIONS} but received {version} "
                "instead. Upgrading your dwave-optimization version may help."
            )

        header_len = struct.unpack("<I", file.read(4))[0]
        header_data: dict = json.loads(file.read(header_len).decode("ascii"))

        return version, header_data

    def _header_data(
        self, *, only_decision: bool, max_num_states: int | None = None
    ) -> dict[str, int]:
        """The header data associated with the model (but not the states)."""
        num_nodes = self.num_decisions() if only_decision else self.num_nodes()

        try:
            if max_num_states is None:
                num_states = self.states.size()
            elif max_num_states < 0:
                raise ValueError("max_num_states must be non-negative")
            else:
                num_states = min(self.states.size(), max_num_states)
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

    def input(
        self,
        shape: tuple[int, ...] = (),
        lower_bound: float | None = -float("inf"),
        upper_bound: float | None = +float("inf"),
        integral: bool | None = None,
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
            >>> y = x * i.sum(axis=1)
            >>> with model.lock():
            ...     model.states.resize(1)
            ...     i.set_state(0, [[1, 2, 3], [1, 1, 2]])
            ...     x.set_state(0, [[1], [-1]])
            ...     print(y.state(0))
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
            self, shape=shape, lower_bound=lower_bound, upper_bound=upper_bound, integral=integral
        )

    def integer(
        self,
        shape: ShapeLike | None = None,
        lower_bound: np.typing.ArrayLike | None = None,
        upper_bound: np.typing.ArrayLike | None = None,
        *,
        sum_subject_to: _SumSubjectToType | None = None,
        axes_sums_subject_to: _AxesSumsSubjectToType | None = None,
    ) -> IntegerVariable:
        r"""Add an integer decision variable to the model.

        An integer symbol is an array of integer values assigned as a solution
        to the problem being modeled.

        .. tip::
            Your choice of decision variables significantly impacts the solver's
            performance.

            Where possible, prefer more-abstract variable types (e.g.,
            :meth:`.list` or :meth:`.set`) or to derive auxiliary variables
            rather than adding them as explicit decision variables.

            See :ref:`opt_model_construction_nl_guidance`.

        Args:
            shape:
                Shape of the integer array to create, formatted as an integer or
                a tuple of integers.
                If ``None``, creates a zero-dimensional (scalar) variable.
            lower_bound:
                Lower bound(s) for the symbol.
                Can be scalar (one bound for all variables) or an array (one
                bound for each variable).
                Non-integer values are rounded up.

                .. versionadded:: 0.6.7
                    Support for index-wise bounds.
            upper_bound:
                Upper bound(s) for the symbol.
                Can be scalar (one bound for all variables) or an array (one bound for
                each variable).
                Non-integer values are rounded down.

                .. versionadded:: 0.6.7
                    Support for index-wise bounds.
            sum_subject_to:
                Constraint on the sum of the values in the array.
                Must be a sequence of tuples where each tuple has the form:
                ``(operator, bound)``

                - **operator** (``str``) - The constraint operator
                  (``"<="``, ``"=="``, or ``">="``).
                - **bound** (``float``) - The constraint bound.

                If provided, the sum of values within the array must satisfy
                the corresponding operator–bound pair.
                E.g.,

                .. code-block:: python

                    y = model.integer(10, sum_subject_to=[("<=", 5)])

                enforces

                .. math::

                    \sum_{i=0}^9 y_i \leq 5

                **Currently, at most one sum constraint may be provided.**
                If provided, ``axes_sums_subject_to`` must be ``None``.

                .. versionadded:: 0.6.13
            axes_sums_subject_to:
                Constraint on the sum of the values in each slice along a fixed
                axis in the array.
                Must be a sequence of tuples where each tuple has the form:
                ``(axis, operator(s), bound(s))``

                - **axis** (``int``) - The axis that the constraint is applied to.
                - **operator(s)** (``str`` or ``list[str]``) - The constraint
                  operator (``"<="``, ``"=="``, or ``">="``).
                  A single operator applies to all slice along the axis; an
                  array specifies one operator per slice.
                - **bound(s)** (``float | list[float]``) - The constraint bound. A
                  single value applies to all slices; an array specifies one
                  bound per slice.

                If provided, the sum of values within each slice along the
                specified axis must satisfy the corresponding operator–bound pair.
                E.g.,

                .. code-block:: python

                    y = model.integer(
                        (2, 3),
                        axes_sums_subject_to=[(1, "<=", 2)],
                    )

                enforces

                .. math::

                    \forall i, \sum_{j=0}^2 y_{i,j} \leq 2

                **Currently, at most one sum constraint may be provided.**
                If provided, ``sum_subject_to`` must be ``None``.

                .. versionadded:: 0.6.13

        Returns:
            An integer symbol as a decision variable in the model.

        Examples:
            This example adds a :math:`(5,)`-shaped integer decision variable with
            scalar bounds to a model, and takes the logarithm of its elements.

            >>> from dwave.optimization.model import Model
            >>> from dwave.optimization.mathematical import log
            >>>
            >>> model = Model()
            >>> i = model.integer(5, lower_bound=1, upper_bound=10)
            >>> i.shape()
            (5,)
            >>> a = log(i)
            >>> with model.lock():
            ...     model.states.resize(1)
            ...     i.set_state(0, [[1, 2, 3, 1, 2]])
            ...     print(a.state(0)[0])
            0.0

            This example adds a :math:`(2,)`-shaped integer symbol with a scalar
            lower bound and index-wise upper bounds to a model.

            >>> import numpy as np
            >>> from dwave.optimization.model import Model
            >>>
            >>> model = Model()
            >>> i = model.integer(2, lower_bound=-1.1, upper_bound=[1.1, 2.9])
            >>> np.all([-1, -1] == i.lower_bound())
            True
            >>> np.all([1, 2] == i.upper_bound())
            True

            This example adds a :math:`(2, 3)`-shaped integer symbol with general
            lower and upper bounds and a sum constraint along axis :math:`1`.
            Let :math:`x_i` denote the sum of the values within slice :math:`i`
            along axis :math:`1`. For each state defined for this symbol:
            :math:`(x_0 \leq 2)`, :math:`(x_1 \leq 4)`, and :math:`(x_2 \leq 5)`.

            >>> import numpy as np
            >>> from dwave.optimization.model import Model
            >>>
            >>> model = Model()
            >>> i = model.integer(
            ...     [2, 3],
            ...     lower_bound=1,
            ...     upper_bound=3,
            ...     axes_sums_subject_to=[(1, "<=", [2, 4, 5])],
            ... )
            >>> np.all(i.sum_constraints() == [(1, ["<="], [2, 4, 5])])
            True

            This example adds a :math:`(6,)`-shaped integer symbol such that
            the sum of the values within the array is less than or equal
            to :math:`20`.

            >>> import numpy as np
            >>> from dwave.optimization.model import Model
            >>>
            >>> model = Model()
            >>> i = model.integer(6, sum_subject_to=[("<=", 20)])
            >>> np.all(i.sum_constraints() == [(["<="], [20])])
            True

        See Also:
            :class:`~dwave.optimization.symbols.numbers.IntegerVariable`:
            Generated symbol.

            :meth:`.binary`, :meth:`.constant`, :meth:`.input`

            :meth:`.iter_decisions`
        """
        from dwave.optimization.symbols import IntegerVariable  # avoid circular import

        return IntegerVariable(
            self,
            shape,
            lower_bound,
            upper_bound,
            sum_subject_to=sum_subject_to,
            axes_sums_subject_to=axes_sums_subject_to,
        )

    @_file_object_arg(arg_index=1, arg_name="file", mode="wb")  # convert path-like into file-like
    @_lock
    def into_file(
        self,
        file: str | os.PathLike[str] | typing.BinaryIO,
        *,
        max_num_states: int = 0,
        only_decision: bool = False,
        version: tuple[int, int] | None = None,
    ) -> None:
        """Serialize the model into an existing file.

        Args:
            file:
                File pointer to an existing writeable, seekable
                file-like object encoding a model. Strings are
                interpreted as a file name.
            max_num_states:
                Maximum number of states to serialize along with the model.
                The number of states serialized is the minimum between
                :meth:`~dwave.optimization.states.States.size` and the specified
                ``max_num_states`` value.
            only_decision:
                If ``True``, only decision variables are serialized.
                If ``False``, all symbols are serialized.
            version:
                A 2-tuple indicating which serialization version to use.

        See Also:
            :meth:`.from_file`, :meth:`.to_file`

            :meth:`~dwave.optimization.states.States.into_file` Saves states to
            an existing file

        Format Specification (Version 1.0):

            This format is inspired by the `NPY format`_

            The first 4 bytes are a magic string: exactly "DWNL".

            The next 1 byte is an unsigned byte: the major version of the file
            format.

            The next 1 byte is an unsigned byte: the minor version of the file
            format.

            The next 4 bytes form a little-endian unsigned int, the length of
            the header data HEADER_LEN.

            The next HEADER_LEN bytes form the header data. This is a
            json-serialized dictionary. The dictionary contains the following
            key/values: ``decision_state_size``, ``num_nodes``, ``num_states``,
            and ``state_size``. It is terminated by a newline character and
            padded with spaces to make the entire length of the entire header
            divisible by 64.

            Following the header, the remaining data is encoded as a zip file.
            All arrays are saved using the NumPy serialization format, see
            :func:`numpy.save()`.

            The information in the header is also saved in a json-formatted
            file ``info.json``.

            The serialization version is saved in a file ``version.txt``.

            Each node is listed by type in a file ``nodetypes.txt``.

            The adjacency of the nodes is saved in an adjacency format file,
            ``adj.adjlist``. E.g., a graph with edges `a->b`, `a->c`, `b->d`
            would be saved as

            .. code-block::

                a b c
                b d
                c
                d

            The id of the object is stored in ``objective.json``, and the list
            of constraint symbols are stored by id in ``constraints.json``.

            Finally each symbol has symbol-specific storage in a directory.

            .. code-block::

                nodes/
                    <symbol id>/
                        <symbol-specific storage>
                    ...

            The states, if also saved, are saved according to
            :meth:`States.into_file()`.

        Format Specification (Version 0.1):

            Prior to version 1.0, states were saved differently.

        .. versionchanged:: 0.5.2
            Added the ``version`` keyword-only argument.
        .. versionchanged:: 0.6.0
            Added support for serialization format version 1.0.

        .. _NPY format: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
        """
        from dwave.optimization.symbols.reduce import Prod, Sum  # see note below

        version, model_info = self._into_file_header(
            file,
            version=version,
            max_num_states=max_num_states,
            only_decision=only_decision,
        )

        num_states = model_info["num_states"]

        encoder = json.JSONEncoder(separators=(",", ":"))

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
                    # There is a backwards compatibility break here in 0.6.8 where
                    # we unified PartialReduce and Reduce. For compatibility going
                    # backwards though, we want to use the PartialReduce alias
                    if isinstance(node, Prod) and node.axes():
                        f.write(b"PartialProd\n")
                        continue
                    if isinstance(node, Sum) and node.axes():
                        f.write(b"PartialSum\n")
                        continue

                    f.write(type(node).__name__.encode("UTF-8"))
                    f.write(b"\n")

            # On the second pass we encode the adjacency
            with zf.open("adj.adjlist", "w", force_zip64=True) as f:
                # We don't actually need to make the Python symbols here, but it's convenient
                # Also, if we're only_decision then there will never be predecessors, but
                # let's reuse the code for now.
                stop = self.num_decisions() if only_decision else self.num_nodes()
                for node in itertools.islice(self.iter_symbols(), 0, stop):
                    f.write(f"{node.topological_index()}".encode())
                    for pred in node.iter_predecessors():
                        f.write(f" {pred.topological_index()}".encode())
                    f.write(b"\n")

            # On the third pass, we allow nodes to save whatever info they want
            # to in a nested node/<topological_index> directory
            for node in itertools.islice(self.iter_symbols(), 0, stop):
                directory = f"nodes/{node.topological_index()}/"
                node._into_zipfile(zf, directory)

            # If we're saving states, do a fourth pass saving the states.
            if num_states > 0:
                self.states._into_zipfile(zf, num_states=num_states, version=version)

            if self.objective is not None and self.objective.topological_index() < stop:
                zf.writestr("objective.json", encoder.encode(self.objective.topological_index()))
            else:
                zf.writestr("objective.json", b"")

            constraints = []  # todo: not yet available at the python level
            for c in self.iter_constraints():
                if c is not None and c.topological_index() < stop:
                    constraints.append(c.topological_index())
            zf.writestr("constraints.json", encoder.encode(constraints))

    def _into_file_header(
        self,
        file: io.BytesIO,
        *,
        version: tuple[int, int] | None,
        max_num_states: int,
        only_decision: bool,
    ) -> tuple[tuple[int, int], dict]:
        """Write the header that precedes the zipfile part of the serialization.

        Also handles some input checking.
        """
        if version is None:
            version = DEFAULT_SERIALIZATION_VERSION
        elif version not in KNOWN_SERIALIZATION_VERSIONS:
            raise ValueError(
                "Unknown serialization format. Expected one of "
                f"{KNOWN_SERIALIZATION_VERSIONS} but received {version} "
                "instead. Upgrading your dwave-optimization version may help."
            )

        model_info = self._header_data(
            max_num_states=max_num_states,
            only_decision=only_decision,
        )

        encoder = json.JSONEncoder(separators=(",", ":"))

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
        padding = b" " * (64 - (len(header_data) + 4 + 2 + 4) % 64)

        file.write(struct.pack("<I", len(header_data) + len(padding)))  # header length
        file.write(header_data)
        file.write(padding)

        return version, model_info

    def list(
        self,
        n: int,
        min_size: int | None = None,
        max_size: int | None = None,
    ) -> ListVariable:
        """Add a list decision variable to the model.

        A list symbol is a list containing a permutation of the values in
        :math:`[0, n-1]`, where the permutation is assigned as a solution to the
        problem being modeled.

        Args:
            n: Range of values in the permutations list (zero to :math:`n - 1`).
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
            ...     model.states.resize(2)
            ...     routes.set_state(0, [10, 2, 44])
            ...     routes.set_state(1, [67, 1])

        See Also:
            :class:`~dwave.optimization.symbols.ListVariable`: Generated
            symbol

            :meth:`.disjoint_bit_sets`, :meth:`.disjoint_lists_symbol`,
            :meth:`.set`

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
        successor (non-decision) variables.

        Returns:
            A context manager. If the context is subsequently exited, the
            :meth:`.unlock` method is called.

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

        See Also:
            :meth:`.is_locked`, :meth:`.unlock`

        """
        super().lock()
        return locked(self)

    # dev note: the typing is underspecified, but it would be quite complex to fully
    # specify the linear/quadratic so let's leave it alone for now.
    def quadratic_model(self, x: ArraySymbol, quadratic, linear=None) -> QuadraticModel:
        """Add a quadratic model to the model.

        Creates a :term:`quadratic model` from a predecessor
        :class:`~dwave.optimization.model.ArraySymbol` and a quadratic model,
        such as a :term:`QUBO`.

        Args:
            x: Predecessor array symbol.

            quadratic: Quadratic values for the quadratic model. Can also
                include linear values as self loops (e.g., ``(3, 3): 2.5``).

            linear: Linear values for the quadratic model.

        Returns:
            A successor symbol that outputs the values of a quadratic model on
            its predecessor symbol's state.

        Examples:
            This example creates a :term:`binary quadratic model` (BQM).

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> x = model.binary(3)
            >>> Q = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 1): 1, (1, 2): 3, (2, 2): 2}
            >>> qm = model.quadratic_model(x, Q)

            This example creates a quadratic example and prints a state.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> i = model.integer(3, lower_bound=-5, upper_bound=10)
            >>> quad = {(0, 1): 1, (0, 2): 2, (1, 2): 3}
            >>> lin = {0: 0.5, 1: 1.2, 2: 2.0}
            >>> qm = model.quadratic_model(i, quadratic=quad, linear=lin)
            >>>
            >>> with model.lock():
            ...     model.states.resize(1)
            ...     i.set_state(0, [1, 3, 0])
            ...     print(qm.state(0).round())
            7.0

        See Also:
            :class:`~dwave.optimization.symbols.QuadraticModel`: Generated
            symbol
        """
        from dwave.optimization.symbols import QuadraticModel

        return QuadraticModel(x, quadratic, linear)

    def set(
        self,
        n: int,
        min_size: int = 0,
        max_size: int | None = None,
    ) -> SetVariable:
        """Add a set decision variable to the model.

        A set symbol is an unordered collection of values in :math:`[0, n-1]`,
        with the values assigned as a solution to the problem being modeled.

        Args:
            n: Range of values (zero to :math:`n-1)` for the set.
            min_size: Minimum set size. Defaults to ``0``.
            max_size: Maximum set size. Defaults to ``n``.

        Returns:
            A set symbol  at the root of the :term:`directed acyclic graph`
            for the model.

        Examples:
            This example creates a set symbol of up to four elements with values
            between 0 to 99, and sets a set of three elements as a state of the
            decision variable.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> destinations = model.set(100, max_size=4)
            >>>
            >>> with model.lock():
            ...     model.states.resize(1)
            ...     destinations.set_state(0, [0, 22, 58])

        See Also:
            :class:`~dwave.optimization.symbols.SetVariable`: Generated
            symbol

            :meth:`.disjoint_bit_sets`, :meth:`.disjoint_lists_symbol`,
            :meth:`.list`

            :meth:`.iter_decisions`
        """
        from dwave.optimization.symbols import SetVariable  # avoid circular import

        return SetVariable(self, n, min_size, n if max_size is None else max_size)

    def to_file(self, **kwargs) -> typing.BinaryIO:
        """Serialize the model to a new file-like object.

        Examples:
            This example serializes a model to a
            `buffered I/O <https://docs.python.org/3/library/io.html#binary-i-o>`_
            object.

            >>> from dwave.optimization.generators import flow_shop_scheduling
            >>>
            >>> processing_times = [[10, 5, 7], [20, 10, 15]]
            >>> model = flow_shop_scheduling(processing_times=processing_times)
            >>>
            >>> my_file = model.to_file()

            The temporary file created above can be saved to disk, for example,
            using Python's :mod:`shutil` module.

            .. testcode::
                :skipif: True

                import shutil

                my_file.seek(0)  # Move cursor to start
                with open("my_file.bin", "wb") as f:
                    shutil.copyfileobj(my_file, f)

        See Also:
            :meth:`.into_file`, :meth:`.from_file`

            :meth:`~dwave.optimization.states.States.to_file` Saves states to a
            file
        """
        file = tempfile.TemporaryFile(mode="w+b")  # noqa: SIM115

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

            >>> import dagviz  # doctest: +SKIP
            >>> r = dagviz.render_svg(G)  # doctest: +SKIP
            >>> with open("model.svg", "w") as f:  # doctest: +SKIP
            ...     f.write(r)

            This creates the following image:

            .. figure:: /_images/optimization/to_networkx_example.svg
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
        """Release a lock and decrement the lock count.

        Symbols can be added to unlocked models only. Unlocked models do not
        allow access to methods such as
        :meth:`~dwave.optimization.model.ArraySymbol.state` and
        :meth:`~dwave.optimization.model.Symbol.topological_index` for
        successor (non-decision) variables.

        Examples:
            This example checks the status of a model.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> i = model.integer(20, upper_bound=100)
            >>> model.is_locked()
            False

        See Also:
            :meth:`.is_locked`, :meth:`.lock`
        """
        if not self.is_locked():
            return

        super().unlock()

        if not self.is_locked():
            self.states._reset_intermediate_states()

# Copyright 2026 D-Wave
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

from __future__ import annotations

import abc
import dataclasses
import fractions
import numbers
import typing

import numpy as np

import dwave.optimization.mathematical
from dwave.optimization.utilities import _NoValue, _split_indices, _TypeError_to_NotImplemented

if typing.TYPE_CHECKING:
    import collections.abc
    import types

    from dwave.optimization import symbols
    from dwave.optimization.model import ArraySymbol, Model, Symbol
    from dwave.optimization.typing import ArraySymbolLike, ShapeLike
    from dwave.optimization.utilities import _NoValueType

__all__ = []


@dataclasses.dataclass(frozen=True)
class ArraySizeInfo:
    """Information about the size of an :class:`ArraySymbol`."""

    multiplier: fractions.Fraction
    symbol: Symbol
    offset: fractions.Fraction
    min: None | float = None
    max: None | float = None


@dataclasses.dataclass(frozen=True)
class ArrayInfo:
    """Information about an :class:`ArraySymbol`."""

    min: float
    max: float
    integral: bool
    size: int | ArraySizeInfo
    c_contiguous: bool


# We don't inherit from abc.ABC because Cython ends up ignoring it anyway.
# We *do* still mark the methods and then we can check them in the tests.
class _SymbolMixin:
    __slots__ = ()

    def __repr__(self):
        """Return a representation of the symbol.

        The representation refers to the identity of the underlying node, rather than
        the identity of the Python symbol.
        """
        cls = type(self)
        return f"<{cls.__module__}.{cls.__qualname__} at {self.id():#x}>"

    @abc.abstractmethod
    def id(self) -> int: ...


class _ArraySymbolMixin(_SymbolMixin):
    __slots__ = ()

    # Opt ArraySymbol out of default interoperability with NumPy ufuncs. We then
    # add explicit support with our various __<op>__() and __r<op>__ methods.
    # This prevents NumPy from interpreting our ArraySymbols as object arrays.
    __array_ufunc__ = None

    # Likewise, we want to opt out of default interoperability with NumPy
    # functions.
    # See https://numpy.org/neps/nep-0018-array-function-protocol.html#nep18
    def __array_function__(self, *_, **__) -> types.NotImplementedType:
        return NotImplemented

    def __abs__(self) -> symbols.Absolute:
        # Dev note: we could check whether self is non-negative and return self
        # in that case.
        from dwave.optimization.symbols import Absolute  # avoid circular import

        return Absolute(self)

    @_TypeError_to_NotImplemented
    def __add__(self: ArraySymbol, rhs: ArraySymbolLike) -> symbols.Add:
        return dwave.optimization.mathematical.add(self, rhs)

    @_TypeError_to_NotImplemented
    def __iadd__(self, rhs: ArraySymbolLike) -> symbols.NaryAdd:
        # If the user is doing +=, we make the assumption that they will want to
        # do it again, so we use NaryAdd
        # Like NumPy, we only support this if the rhs can be broadcast to our
        # shape.
        rhs = dwave.optimization.mathematical.broadcast_to(
            *dwave.optimization.mathematical.as_array_symbols(rhs, model=self.model),
            self.shape(),
        )
        return dwave.optimization.mathematical.NaryAdd(self, rhs)

    @_TypeError_to_NotImplemented
    def __radd__(self, lhs: ArraySymbolLike) -> symbols.Add:
        return dwave.optimization.mathematical.add(lhs, self)

    def __bool__(self) -> typing.NoReturn:
        # Follow NumPy as disallow conversion to boolean
        raise ValueError("the truth value of an array symbol is ambiguous")

    @_TypeError_to_NotImplemented
    def __eq__(self, rhs):
        return dwave.optimization.mathematical.equal(self, rhs)

    @_TypeError_to_NotImplemented
    def __ge__(self, rhs):
        return dwave.optimization.mathematical.less_equal(rhs, self)

    def __getitem__(
        self,
        index: int
        | slice
        | types.EllipsisType
        | ArraySymbol
        | None
        | tuple[int | slice | types.EllipsisType | ArraySymbol | None, ...],
    ) -> ArraySymbol:
        if not isinstance(index, tuple):
            return self[(index,)]

        import dwave.optimization.symbols  # avoid circular import

        newaxes, basic, advanced = _split_indices(self.shape(), index)

        out = self

        if newaxes:
            raise NotImplementedError

        if not all(isinstance(index, slice) and index == slice(None) for index in basic):
            out = dwave.optimization.symbols.BasicIndexing(out, *basic)

        if not all(isinstance(index, slice) and index == slice(None) for index in advanced):
            # Do the type promotion to ArraySymbol
            try:
                adv = [
                    index
                    if isinstance(index, slice)
                    else dwave.optimization.mathematical.as_array_symbols(index, model=self.model)[
                        0
                    ]
                    for index in advanced
                ]
            except ValueError as err:
                raise IndexError(err)

            out = dwave.optimization.symbols.AdvancedIndexing(out, *adv)

        return out

    def __iter__(self) -> collections.abc.Iterator[ArraySymbol]:
        if self.ndim() <= 0:
            # NumPy's error is TypeError("iteration over a 0-d array"), so match
            # that.
            raise TypeError("iteration over a 0-d array symbol")

        if self.shape()[0] < 0:
            # NumPy doesn't have a notion of dynamic size, but let's keep our
            # error message consistent
            raise TypeError("iteration over a dynamically sized array symbol")

        yield from (self[i] for i in range(self.shape()[0]))

    @_TypeError_to_NotImplemented
    def __le__(self, rhs):
        return dwave.optimization.mathematical.less_equal(self, rhs)

    @_TypeError_to_NotImplemented
    def __matmul__(self, rhs) -> symbols.MatrixMultiply:
        return dwave.optimization.mathematical.matmul(self, rhs)

    @_TypeError_to_NotImplemented
    def __rmatmul__(self, lhs: ArraySymbolLike) -> symbols.MatrixMultiply:
        return dwave.optimization.mathematical.matmul(lhs, self)

    @_TypeError_to_NotImplemented
    def __mod__(self, rhs: ArraySymbolLike) -> symbols.Modulus:
        return dwave.optimization.mathematical.mod(self, rhs)

    @_TypeError_to_NotImplemented
    def __rmod__(self, lhs: ArraySymbolLike) -> symbols.Modulus:
        return dwave.optimization.mathematical.mod(lhs, self)

    @_TypeError_to_NotImplemented
    def __mul__(self, rhs) -> symbols.Multiply:
        return dwave.optimization.mathematical.multiply(self, rhs)

    @_TypeError_to_NotImplemented
    def __rmul__(self, lhs: ArraySymbolLike) -> symbols.Multiply:
        return dwave.optimization.mathematical.multiply(lhs, self)

    @_TypeError_to_NotImplemented
    def __imul__(self, rhs: ArraySymbolLike) -> symbols.NaryMultiply:
        # If the user is doing *=, we make the assumption that they will want to
        # do it again, so we use NaryMultiply
        # Like NumPy, we only support this if the rhs can be broadcast to our
        # shape.
        from dwave.optimization.symbols import NaryMultiply  # avoid circular import

        rhs = dwave.optimization.mathematical.broadcast_to(
            *dwave.optimization.mathematical.as_array_symbols(rhs, model=self.model),
            self.shape(),
        )
        return NaryMultiply(self, rhs)

    def __neg__(self) -> symbols.Negative:
        from dwave.optimization.symbols import Negative  # avoid circular import

        return Negative(self)

    def __pow__(
        self, rhs: int
    ) -> symbols.Constant | typing.Self | symbols.Square | symbols.NaryMultiply:
        if not isinstance(rhs, numbers.Integral):
            return NotImplemented

        try:
            exponent = int(rhs)
        except TypeError:
            return NotImplemented

        if exponent < 0:
            raise ValueError("only non-negative exponents are supported")
        elif exponent == 0:
            return self.model.constant(1)
        elif exponent == 1:
            return self
        elif exponent == 2:
            from dwave.optimization.symbols import Square  # avoid circular import

            return Square(self)
        else:
            return dwave.optimization.mathematical.multiply(*([self] * exponent))

    @_TypeError_to_NotImplemented
    def __sub__(self, rhs: ArraySymbolLike) -> symbols.Subtract:
        return dwave.optimization.mathematical.subtract(self, rhs)

    @_TypeError_to_NotImplemented
    def __rsub__(self, lhs: ArraySymbolLike) -> symbols.Subtract:
        return dwave.optimization.mathematical.subtract(lhs, self)

    @_TypeError_to_NotImplemented
    def __truediv__(self, rhs: ArraySymbolLike) -> symbols.Divide:
        return dwave.optimization.mathematical.divide(self, rhs)

    @_TypeError_to_NotImplemented
    def __rtruediv__(self, lhs: ArraySymbolLike) -> symbols.Divide:
        return dwave.optimization.mathematical.divide(lhs, self)

    def all(
        self,
        *,
        axis: int | tuple[int, ...] | None = None,
        initial: _NoValueType | float | None = _NoValue,
    ) -> symbols.All:
        r"""Create a symbol that tests whether all array elements evaluate to True.

        Args:
            axis:
                Axis or axes along which the operation is performed.
                If unspecified, reduction is performed over all dimensions.
                If ``tuple[int, ...]``, reduction is performed along the
                specified axes.

                .. versionadded:: 0.6.8

            initial:
                This argument is ignored.

                .. versionadded:: 0.6.8

        Returns:
            :class:`~dwave.optimization.symbols.All`: A successor symbol that
            returns True if all elements of the predecessor symbol are True.

        Examples:
            This example tests the columns of a :math:`2 \times 3` array.

            >>> from dwave.optimization import Model
            >>> model = Model()
            >>> x = model.binary((2, 3))
            >>> all_x = x.all(axis=0)
            >>> with model.lock():
            ...     model.states.resize(1)
            ...     x.set_state(0, [[True, False, True], [True, True, False]])
            ...     print(all_x.state())
            [1. 0. 0.]

        See Also:
            *   :class:`~dwave.optimization.symbols.All`: Generated symbol
            *   :meth:`.any`
            *   :func:`~numpy.all`: :doc:`NumPy <numpy:index>` function

        """
        from dwave.optimization.symbols import All  # avoid circular import

        return All(self, axis=axis, initial=_NoValue)

    def any(
        self,
        *,
        axis: int | tuple[int, ...] | None = None,
        initial: _NoValueType | float | None = _NoValue,
    ):
        r"""Create a symbol that tests whether any array element evaluates to True.

        Args:
            axis:
                Axis or axes along which the operation is performed.
                If unspecified, reduction is performed over all dimensions.
                If ``tuple[int, ...]``, reduction is performed along the
                specified axes.

                .. versionadded:: 0.6.8

            initial:
                This argument is ignored.

                .. versionadded:: 0.6.8

        Returns:
            :class:`~dwave.optimization.symbols.Any`: A successor symbol that
            returns True when any element of the predecessor symbol is True.

        .. versionadded:: 0.4.1

        Examples:
            This example tests the rows of a :math:`2 \times 3` array.

            >>> from dwave.optimization import Model
            >>> model = Model()
            >>> x = model.binary((2, 3))
            >>> any_x = x.any(axis=1)
            >>> with model.lock():
            ...     model.states.resize(1)
            ...     x.set_state(0, [[False, False, False], [True, False, False]])
            ...     print(any_x.state())
            [0. 1.]

        See Also:
            *   :class:`~dwave.optimization.symbols.Any`: Generated symbol
            *   :meth:`.all`
            *   :func:`~numpy.any`: :doc:`NumPy <numpy:index>` function
        """
        from dwave.optimization.symbols import Any  # avoid circular import

        return Any(self, axis=axis, initial=_NoValue)

    def copy(self) -> symbols.Copy:
        """Create a duplicating symbol.

        Returns:
            :class:`~dwave.optimization.symbols.Copy`: A successor symbol that
            maintains an identical copy of the values of the predecessor symbol,
            using contiguous memory for performance.

        .. versionadded:: 0.5.1

        Examples:
            >>> from dwave.optimization import Model
            >>> model = Model()
            >>> x = model.binary((2, 2))
            >>> y = x.copy()
            >>> with model.lock():
            ...     model.states.resize(1)
            ...     x.set_state(0, [[False, False], [True, False]])
            ...     print((y.state() == x.state()).all())
            True

        See Also:
            *   :class:`~dwave.optimization.symbols.Copy`: Generated symbol
            *   :func:`~numpy.copy`: :doc:`NumPy <numpy:index>` function
        """
        from dwave.optimization.symbols import Copy  # avoid circular import

        return Copy(self)

    def flatten(self) -> typing.Self | symbols.Reshape:
        """Create a symbol with the array reshaped to one dimension.

        Returns:
            :class:`~dwave.optimization.symbols.Reshape`: A successor symbol
            that is equivalent to ``symbol.reshape(-1)``, with the predecessor
            symbol's array collapsed into one dimension.

        .. versionadded:: 0.5.1

        Examples:
            >>> from dwave.optimization import Model
            >>> model = Model()
            >>> x = model.binary((2, 2))
            >>> y = x.flatten()
            >>> with model.lock():
            ...     model.states.resize(1)
            ...     x.set_state(0, [[False, False], [True, False]])
            ...     print(x.state()[1, 0] == y.state()[2] == True)
            True

        See Also:
            *   :class:`~dwave.optimization.symbols.Reshape`: generated symbol
            *   :func:`~dwave.optimization.model.ArraySymbol.reshape`
            *   :meth:`~numpy.ndarray.flatten`: :doc:`NumPy <numpy:index>`
                method
        """
        return self.reshape(-1)

    @abc.abstractmethod
    def info(self) -> ArrayInfo: ...

    def max(
        self,
        *,
        axis: int | tuple[int, ...] | None = None,
        initial: _NoValueType | float | None = _NoValue,
    ) -> symbols.Max:
        r"""Create a symbol that returns the maximum value of the array.

        Args:
            axis (int or tuple[int, ...], optional, default=None):
                Axis or axes along which the operation is performed.
                If unspecified, reduction is performed over all dimensions.
                If ``tuple[int, ...]``, reduction is performed along the
                specified axes.

                .. versionadded:: 0.6.8

            initial (float or None, optional):
                Starting value for the reduction. If unspecified, the first
                element of the reduction is used.

                .. versionadded:: 0.6.4

        Returns:
            :class:`~dwave.optimization.symbols.Max`: A successor symbol that
            returns the maximum value among elements of the predecessor symbol.

        Examples:
            This example creates a symbol that returns the maximum values in
            columns of a :math:`2 \times 3` array, with the ``initial``
            argument set to zero to prevent the return of negative values.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> i = model.integer((2, 3), lower_bound=-10, upper_bound=10)
            >>> i_max = i.max(axis=0, initial=0)
            >>> with model.lock():
            ...     model.states.resize(1)
            ...     i.set_state(0, [[2, -4, 5], [8, -2, 7]])
            ...     print(i_max.state())
            [8. 0. 7.]

        See Also:
            *   :class:`~dwave.optimization.symbols.Max`: Generated symbol
            *   :func:`~dwave.optimization.model.ArraySymbol.min`
            *   :func:`~numpy.max`: :doc:`NumPy <numpy:index>` function
        """
        from dwave.optimization.symbols import Max  # avoid circular import

        return Max(self, axis=axis, initial=initial)

    def min(
        self,
        *,
        axis: int | tuple[int, ...] | None = None,
        initial: _NoValueType | float | None = _NoValue,
    ) -> symbols.Min:
        r"""Create a symbol that returns the minimum value of the array.

        Args:
            axis (int or tuple[int, ...], optional, default=None):
                Axis or axes along which the operation is performed.
                If unspecified, reduction is performed over all dimensions.
                If ``tuple[int, ...]``, reduction is performed along the
                specified axes.

                .. versionadded:: 0.6.8

            initial (float or None, optional):
                Starting value for the reduction. If unspecified, the first
                element of the reduction is used.

                .. versionadded:: 0.6.4

        Returns:
            :class:`~dwave.optimization.symbols.Min`: A successor symbol that
            returns the minimum value among elements of the predecessor symbol.

        Examples:
            This example creates a symbol that returns the minimum values in
            rows of a :math:`3 \times 2` array, with the ``initial``
            argument set to zero to prevent the return of positive values.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> i = model.integer((3, 2), lower_bound=-10, upper_bound=10)
            >>> i_min = i.min(axis=1, initial=0)
            >>> with model.lock():
            ...     model.states.resize(1)
            ...     i.set_state(0, [[2, -4], [5, 8], [-2, -7]])
            ...     print(i_min.state())
            [-4.  0. -7.]

        See Also:
            *   :class:`~dwave.optimization.symbols.Min`: Generated symbol
            *   :func:`~dwave.optimization.model.ArraySymbol.max`
            *   :func:`~numpy.min`: :doc:`NumPy <numpy:index>` function
        """
        from dwave.optimization.symbols import Min  # avoid circular import

        return Min(self, axis=axis, initial=initial)

    @property
    @abc.abstractmethod
    def model(self) -> Model: ...

    @abc.abstractmethod
    def ndim(self) -> int: ...

    def prod(self, *, axis=None, initial=_NoValue):
        r"""Create a symbol that returns the product of the array elements.

        Args:
            axis (int or tuple[int, ...], optional, default=None):
                Axis or axes along which the operation is performed.
                If unspecified, reduction is performed over all dimensions.
                If ``tuple[int, ...]``, reduction is performed along the
                specified axes.

                .. versionadded:: 0.5.1

            initial (float or None, optional):
                Starting value for the reduction. If unspecified, the first
                element of the reduction is used.

                .. versionadded:: 0.6.4

        Returns:
            :class:`~dwave.optimization.symbols.Prod`: A successor symbol that
            returns the product of the elements of the predecessor symbol.

        Examples:
            This example creates a symbol that returns the product of values in
            columns of a :math:`2 \times 3` array, with the ``initial``
            argument set to two to double the returned values.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> i = model.integer((2, 3), lower_bound=-10, upper_bound=10)
            >>> prod_i = i.prod(axis=0, initial=2)
            >>> with model.lock():
            ...     model.states.resize(1)
            ...     i.set_state(0, [[2, -4, 5], [8, -2, 7]])
            ...     print(prod_i.state())
            [32. 16. 70.]

        See Also:
            *   :class:`~dwave.optimization.symbols.Prod`: Generated symbol
            *   :func:`~numpy.prod`: :doc:`NumPy <numpy:index>` function
        """
        from dwave.optimization.symbols import Prod

        return Prod(self, axis=axis, initial=initial)

    def reshape(self, *shape: ShapeLike) -> typing.Self | symbols.Reshape:
        r"""Create a symbol with the array reshaped.

        Args:
            shape (int or tuple[int, ...]):
                Shape of the created symbol, specified as an integer argument
                per dimension or a single argument formatted as a tuple or list;
                e.g., :code:`a.reshape((1, 2))` is equivalent to
                :code:`a.reshape(1, 2)`.
                One dimension can be :math:`-1`, in which case its size is
                inferred from the other dimensions.
                For dynamically sized array symbols, the first dimension must
                be specified as :math:`-1` and the remaining dimensions must be
                even divisions or multiplications of the dimensions of the
                predecessor array symbol, as shown in the examples below.

        Returns:
            :class:`~dwave.optimization.symbols.Reshape` or ``self``: A
            successor symbol that reshapes the predecessor symbol without
            changing its values, except when the provided shape exactly matches
            the shape of the symbol, in which case the predecessor symbol is
            returned.

        Examples:
            This example reshapes a 1D vector into a :math:`1 \times 3` matrix.

            >>> from dwave.optimization import Model
            >>> model = Model()
            >>> j = model.integer(3, lower_bound=-10, upper_bound=10)
            >>> j.shape()
            (3,)
            >>> k = j.reshape((1, 3))
            >>> k.shape()
            (1, 3)

            This example reshapes a dynamic 2D array symbol.

            >>> import numpy as np
            >>>
            >>> model = Model()
            >>> a = model.constant(np.ones((10, 4)))[model.set(10), :]
            >>> a.shape()
            (-1, 4)
            >>> b = a.reshape(-1, 2, 2)  # OK because 2*2 = 4
            >>> b.shape()
            (-1, 2, 2)
            >>> c = a.reshape(-1, 4, 1, 1)  # OK because 4*1*1 = 4
            >>> c.shape()
            (-1, 4, 1, 1)
            >>> d = a.reshape(-1, 2)  # OK because 2 evenly divides 4
            >>> d.shape()
            (-1, 2)
            >>> a.reshape(
            ...     -1, 8
            ... )  # Fails because 8 does not evenly divide 4 # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            ValueError: cannot reshape array of shape (-1, 4) into shape (-1, 8)

        .. versionadded:: 0.5.1
        .. versionadded:: 0.6.5
            Add support for reshaping dynamic array symbols.

        See Also:
            *   :class:`~dwave.optimization.symbols.Reshape`: Generated symbol
            *   :func:`~dwave.optimization.mathematical.resize`,
                :func:`~dwave.optimization.model.ArraySymbol.flatten`
            *   :func:`~numpy.reshape`: :doc:`NumPy <numpy:index>` function

        """
        from dwave.optimization.symbols import Reshape  # avoid circular import

        # Handle .reshape(0, 1) vs .reshape((0, 1))
        if len(shape) <= 1:
            shape = shape[0]

        # If the given shape exactly matches self.shape() then we don't need
        # any sort of reshape. This doesn't handle the case where we're inferring
        # -1s, but I think that's enough of an edge case that it's not worth
        # duplicating the logic here.
        if isinstance(shape, int):
            shape = (shape,)
        if shape == self.shape():
            return self

        if not self.info().c_contiguous:
            return Reshape(self.copy(), shape)

        return Reshape(self, shape)

    def resize(self, shape, fill_value=None):
        """Create a symbol with the specified shape.

        Args:
            shape (tuple[int, ...]): Shape of the successor array. Dimension
                values must be non-negative.
            fill_value (int, optional, default=None): Value to use if the
                successor array is larger than the predecessor array. Defaults
                to 0.

        Returns:
            :class:`~dwave.optimization.symbols.Resize`: A successor symbol with
            the specified shape.

        Examples:
            >>> from dwave.optimization import Model
            >>>
            >>> model = Model()
            >>> s = model.set(10)  # subsets of range(10)
            >>> s_2x2 = s.resize((2, 2), fill_value=-1)
            >>>
            >>> model.states.resize(1)
            >>> with model.lock():
            ...     s.set_state(0, [0, 1, 2])
            ...     print(s_2x2.state(0))
            [[ 0.  1.]
             [ 2. -1.]]

        See Also:
            *   :class:`~dwave.optimization.symbols.Resize`: Generated
                symbol
            *   :func:`~dwave.optimization.model.ArraySymbol.reshape`
            *   :func:`~numpy.resize`: :doc:`NumPy <numpy:index>` function

        .. versionadded:: 0.6.4
        """
        from dwave.optimization.mathematical import resize  # avoid circular import

        return resize(self, shape, fill_value)

    @abc.abstractmethod
    def shape(self) -> tuple[int, ...]: ...

    def _states_from_zipfile(self, zf, *, num_states, version):
        directory = f"nodes/{self.topological_index()}/"

        # Test whether we saved all of the state together. We can detect it
        # based on the existance of an arrays.npy file.
        try:
            states_info = zf.getinfo(f"{directory}states.npy")
        except KeyError:
            states_info = None

        if states_info is None:
            for i in range(num_states):
                try:
                    state_info = zf.getinfo(f"{directory}states/{i}/array.npy")
                except KeyError:
                    # states can be missing, in which case just skip this
                    # index
                    continue

                with zf.open(state_info, mode="r") as f:
                    state = np.load(f, allow_pickle=False)

                self.set_state(i, state)
        else:
            with zf.open(states_info, mode="r") as f:
                states = np.load(f, allow_pickle=False)

            # iterate over axis 0, each "row" is then its own state
            for i, state in enumerate(states):
                self.set_state(i, state)

    def _states_into_zipfile(self, zf, *, num_states, version):
        states = [self.state(i) if self.has_state(i) else None for i in range(num_states)]

        # If we have any missing states, or if the states don't all have the
        # same shape then we save each state in a separate directory.
        # Also, before serialization version 1.0 we always saved in a ragged format.
        save_ragged = (
            not states  # implicit to other conditions, but no harm being explicit
            or version < (1, 0)
            or any(state is None for state in states)
            or len(set(state.shape for state in states)) != 1
        )

        directory = f"nodes/{self.topological_index()}/"

        if save_ragged:
            # each state gets its own sub-directory
            for i, state in enumerate(states):
                if state is None:
                    continue
                with zf.open(f"{directory}states/{i}/array.npy", mode="w", force_zip64=True) as f:
                    np.save(f, state, allow_pickle=False)

        else:
            # the states are all saved in one file
            states = np.asarray(states)  # stacks them along axis 0
            with zf.open(f"{directory}states.npy", mode="w", force_zip64=True) as f:
                np.save(f, states, allow_pickle=False)

    def sum(self, *, axis=None, initial=_NoValue):
        r"""Create a symbol that returns the sum of the array elements.

        Args:
            axis (int or tuple[int, ...], optional):
                Axis or axes along which the operation is performed.
                If unspecified, reduction is performed over all dimensions.
                If ``tuple[int, ...]``, reduction is performed along the
                specified axes.

                .. versionadded:: 0.4.1

            initial (float or None, optional):
                Starting value for the reduction. If unspecified, the first
                element of the reduction is used.

                .. versionadded:: 0.6.4

        Returns:
            :class:`~dwave.optimization.symbols.Sum`: A successor symbol that
            returns the sum of the elements of the predecessor symbol.

        Examples:
            This example creates a symbol that returns the sum of values in
            columns of a :math:`2 \times 3` array, with the ``initial``
            argument set to :math:`-3` to offset the returned values.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> i = model.integer((2, 3), lower_bound=-10, upper_bound=10)
            >>> sum_i = i.sum(axis=0, initial=-3)
            >>> with model.lock():
            ...     model.states.resize(1)
            ...     i.set_state(0, [[2, -4, 5], [8, -2, 7]])
            ...     print(sum_i.state())
            [ 7. -9.  9.]

        See Also:
            *   :class:`~dwave.optimization.symbols.Sum`: Genertaed symbol
            *   :func:`~numpy.sum`: :doc:`NumPy <numpy:index>` function
        """
        from dwave.optimization.symbols import Sum

        return Sum(self, axis=axis, initial=initial)

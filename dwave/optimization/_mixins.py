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
import contextlib
import dataclasses
import fractions
import json
import numbers
import tempfile
import typing
import zipfile

import numpy as np

import dwave.optimization.mathematical
from dwave.optimization.utilities import (
    _file_object_arg,
    _NoValue,
    _split_indices,
    _TypeError_to_NotImplemented,
)

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

    @classmethod
    def _from_zipfile(cls, zf, directory, model, predecessors):
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

        See Also:
            :meth:`._into_zipfile`
        """
        # Many symbols are constructed using this pattern, so we do it as default.
        return cls(*predecessors)

    @abc.abstractmethod
    def id(self) -> int: ...

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

        See Also:
            :meth:`._from_zipfile`
        """
        # By default we don't save anything beyond what is saved by the model

    def _states_from_zipfile(self, *_, **__) -> typing.NoReturn:
        # unlike node serialization, by default we raise an error because if
        # this is being called, it must have a state
        raise NotImplementedError(
            f"{type(self).__name__} has not implemented state deserialization"
        )

    def _states_into_zipfile(self, *_, **__) -> typing.NoReturn:
        # unlike node serialization, by default we raise an error because if
        # this is being called, it must have a state
        raise NotImplementedError(f"{type(self).__name__} has not implemented state serialization")

    def state_size(self) -> int:
        """Return an estimated size, in bytes, of a symbol's state.

        The number of bytes returned by this method is only an estimate. Some
        symbols hold additional information that is not accounted for.

        For most symbols, which are arrays, this method is
        subclassed by the :class:`~dwave.optimization.model.ArraySymbol`
        class's :meth:`~dwave.optimization.model.ArraySymbol.state_size`
        method.

        Returns:
            int: Estimated size in bytes.

        See Also:
            :meth:`ArraySymbol.state_size()` An estimate of the size of an array
            symbol's state.

            :meth:`Model.state_size()` An estimate of the size of a model's
            state.
        """
        return 0


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


# We don't inherit from abc.ABC because Cython ends up ignoring it anyway.
# We *do* still mark the methods and then we can check them in the tests.
class _StatesMixin:
    __slots__ = ()

    @abc.abstractmethod
    def _model(self) -> Model: ...

    @_file_object_arg(arg_index=1, arg_name="file", mode="rb")  # convert path-like into file-like
    def from_file(
        self,
        file,
        *,
        replace=True,  # undoc until supported
        check_header=True,  # undocumented until fixed, see
        # https://github.com/dwavesystems/dwave-optimization/issues/22
    ):
        """Construct states from the given file.

        Args:
            file:
                File pointer to a readable, seekable file-like object encoding
                the states. Strings are interpreted as a file name.
            replace (bool):
                Currently only the default is supported.
            check_header (bool):
                Currently unsupported.

        Returns:
            States as assigned in the file.

        Examples:
            This example creates a simple model, sets two states, and then saves
            those states to a
            `buffered I/O <https://docs.python.org/3/library/io.html#binary-i-o>`_
            object. It clears and then restores the states from file.

            >>> import io
            >>>
            >>> from dwave.optimization import Model
            >>>
            >>> model = Model()
            >>> i = model.integer(5, lower_bound=-10, upper_bound=10)
            >>> j = i**2 + 1
            >>>
            >>> model.states.resize(2)
            >>> i.set_state(0, [-10, -5, 0, 5, 10])
            >>> i.set_state(1, [-8, -3, 1, 4, 9])
            >>>
            >>> with model.lock(), model.states.to_file() as f:
            ...     buffer = f.read()  # Save the states to a temporary buffer
            >>>
            >>> model.states.clear()
            >>>
            >>> with model.lock(), io.BytesIO(buffer) as f:
            ...     model.states.from_file(f)  # Restore the states
            >>>
            >>> with model.lock():
            ...     print(j.state(0))
            ...     print(j.state(1))
            [101.  26.   1.  26. 101.]
            [65. 10.  2. 17. 82.]

        See Also:
            :meth:`.into_file`, :meth:`.to_file`

            :meth:`~dwave.optimization.model.Model.from_file` Restores a model
            from a file
        """
        if not replace:
            raise NotImplementedError("appending states is not (yet) implemented")

        # Validate the header, and get the serialization version.
        version, _ = self._model()._from_file_header(file)

        with zipfile.ZipFile(file, mode="r") as zf:
            self._from_zipfile(zf, version=version)

    def from_future(self, future, result_hook):
        """Populate the states from the result of a future computation.

        A :class:`~dwave.cloud.computation.Future` object is returned by the
        :term:`solver` to which you submit your problem model. This enables
        asynchronous problem submission.

        This method is intended for use by developers of the Ocean SDK.

        Args:
            future: A :ref:`future <asyncio-futures>` object such as a
                :class:`~dwave.cloud.computation.Future` object.

            result_hook: Method executed to retrieve the Future.

        See Also:
            :meth:`.from_file`, :meth:`.resolve`

            :meth:`~dwave.optimization.model.Model.from_file` Restores a model
            from a file
        """
        self.resize(0)  # always clears self first

        self._future = future
        self._result_hook = result_hook

    def _from_zipfile(self, zf, *, version):
        """Extract any states from the given zipfile.

        ``zf`` must be a :class:`zipfile.ZipFile`.
        """
        self.resolve()

        model_info = json.loads(zf.read("info.json"))

        # Read any states that have been encoded
        num_states = model_info.get("num_states")
        if not isinstance(num_states, int) or num_states < 0:
            raise ValueError("expected num_states to be a positive integer")
        elif not num_states:
            return  # no states to load, so just return early

        # Get a refcounted model for the duration of this function.
        model = self._model()

        # We'll be overwriting the states, so clear everything that's in there
        model.states.resize(0)
        model.states.resize(num_states)

        for symbol in model.iter_decisions():
            symbol._states_from_zipfile(
                zf,
                num_states=num_states,
                version=version,
            )

        for symbol in model.iter_inputs():
            symbol._states_from_zipfile(
                zf,
                num_states=num_states,
                version=version,
            )

    @property
    @abc.abstractmethod
    def _future(self) -> object: ...

    @_future.setter
    @abc.abstractmethod
    def _future(self, future): ...

    @_file_object_arg(arg_index=1, arg_name="file", mode="wb")  # convert path-like into file-like
    def into_file(
        self,
        file,
        *,
        version=None,
    ):
        """Serialize the states into an existing  file.

        Args:
            file:
                File pointer to an existing writeable, seekable file-like
                object encoding a model. Strings are interpreted as a file
                name.
            version:
                A 2-tuple indicating which serialization version to use; for
                example, ``(1, 0)`` represents version 1.0. By default, uses the
                latest version.

        Examples:
            This example creates a simple model, sets two states, and then saves
            those states to a
            `buffered I/O <https://docs.python.org/3/library/io.html#binary-i-o>`_
            object. It then changes a state and saves into the previously
            created file.

            >>> from dwave.optimization import Model
            >>>
            >>> model = Model()
            >>> i = model.integer(5, lower_bound=-10, upper_bound=10)
            >>> j = i**2 + 1
            >>>
            >>> model.states.resize(2)
            >>> i.set_state(0, [-10, -5, 0, 5, 10])
            >>> i.set_state(1, [-8, -3, 1, 4, 9])
            >>>
            >>> with model.lock():
            ...     model.states.into_file("model.nl")  # doctest: +SKIP

        See Also:
            :meth:`.from_file`, :meth:`.to_file`

            :meth:`~dwave.optimization.model.Model.into_file` Saves a model
            into an existing file

        Format Specification (Version 1.0):

            The first section of the file is the header, as described in
            :meth:`Model.into_file()`.

            Following the header, the remaining data is encoded as a zip file.
            All arrays are saved using the NumPy serialization format, see
            :func:`numpy.save()`.

            The information in the header is also saved in a json-formatted
            file ``info.json``.

            The serialization version is saved in a file ``version.txt``.

            The states have the following structure.

            Symbols with a state that's uniquely determined by their predecessor's
            states do not have their states serialized.

            .. _dwopt-nondeterministic-state-ref:

            .. note::

                Beginning with ``dwave-optimization`` version 0.8.0, all
                non-decisions are treated as having a deterministic state.
                This means any intermediate states in the serialized file are
                ignored as an implementation detail.

            For symbols with a fixed shape and which have all states initialized,
            the states are stored as a ``(num_states, *symbol.shape())`` array.

            .. code-block::

                nodes/
                    <symbol id>/
                        states.npy
                    ...

            For symbols without a fixed shape, or for which not all states are
            initialized, the states are each saved in a separate array.

            .. code-block::

                nodes/
                    <node id>/
                        states/
                            <state index>/
                                array.npy
                            ...
                    ...

            This format allows the states and the model to be saved in the same
            file, sharing the header.

        Format Specification (Version 0.1):

            Saved as a :class:`Model` encoding only the decision symbols.

        .. versionchanged:: 0.5.2
            Added the ``version`` keyword-only argument.
        .. versionchanged:: 0.6.0
            Added support for serialization format version 1.0.
        .. versionchanged:: 0.8.0
            Non-decisions no longer serialize their states.
            See :ref:`Note <dwopt-nondeterministic-state-ref>` above.
        """
        self.resolve()

        if isinstance(version, tuple) and version < (1, 0):
            # In serialization version 0.1, we saved a model with only
            # decisions encoding the states.
            return self._model().into_file(
                file,
                only_decision=True,
                max_num_states=self.size(),
                version=version,
            )

        model = self._model()  # get a ref-counted model

        version, model_info = model._into_file_header(
            file,
            version=version,
            max_num_states=self.size(),
            only_decision=False,
        )

        num_states = model_info["num_states"]

        encoder = json.JSONEncoder(separators=(",", ":"))

        with zipfile.ZipFile(file, mode="w") as zf:
            zf.writestr("info.json", encoder.encode(model_info))
            zf.writestr("version.txt", ".".join(map(str, version)))

            self._into_zipfile(zf, num_states=num_states, version=version)

    def _into_zipfile(self, zf, *, num_states, version):
        """Save any states into the given zipfile.

        ``zf`` must be a :class:`zipfile.ZipFile`.
        """
        if num_states <= 0:
            # nothing so save, so shortcut
            return

        model = self._model()  # get a ref-counted model

        for symbol in model.iter_decisions():
            symbol._states_into_zipfile(
                zf,
                num_states=num_states,
                version=version,
            )

        # If the model is not locked then the Inputs definitely won't have
        # states.
        if model.is_locked():
            for symbol in model.iter_inputs():
                symbol._states_into_zipfile(
                    zf,
                    num_states=num_states,
                    version=version,
                )

    @abc.abstractmethod
    def resize(self, n: int) -> None: ...

    def resolve(self):
        """Block until states are retrieved from any pending future computations.

        A :class:`~dwave.cloud.computation.Future` object is returned by the
        :term:`solver` to which you submit your problem model. This enables
        asynchronous problem submission.

        This method is intended for use by developers of the Ocean SDK.

        See Also:
            :meth:`.from_future`
        """
        if self._future is not None:
            # The existance of _future means that anything we do to the
            # state will block. So we remove it before calling the hook.
            future = self._future
            self._future = None
            result_hook = self._result_hook
            self._result_hook = None

            result_hook(self._model(), future)

    @property
    @abc.abstractmethod
    def _result_hook(self) -> object: ...

    @_future.setter
    @abc.abstractmethod
    def _result_hook(self, future): ...

    @contextlib.contextmanager
    def to_file(self, **kwargs) -> typing.Generator[typing.BinaryIO]:
        """Serialize the states to a new file-like object.

        Examples:
            This example creates a simple model, sets two states, and then creates a
            `buffered I/O <https://docs.python.org/3/library/io.html#binary-i-o>`_
            object.

            .. testcode::

                from dwave.optimization import Model

                model = Model()

                i = model.integer(5, lower_bound=-10, upper_bound=10)
                j = i**2 + 1

                model.states.resize(2)

                with model.lock(), model.states.to_file() as f:
                    pass

            The temporary file created above can be saved to disk, for example,
            using Python's :mod:`shutil` module.

        See Also:
            :meth:`.from_file`, :meth:`.into_file`

            :meth:`~dwave.optimization.model.Model.to_file` Saves a model
            into a file
        """
        with tempfile.TemporaryFile(mode="w+b") as file:
            self.into_file(file, **kwargs)
            file.seek(0)
            yield file

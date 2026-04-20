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

from __future__ import annotations

import collections
import functools
import numbers
import typing

import numpy as np
import numpy.typing

from dwave.optimization._model import _broadcast_shapes, ArraySymbol, Symbol
from dwave.optimization.model import Model
from dwave.optimization.symbols import (
    Add,
    And,
    ARange,
    ArgSort,
    BroadcastTo,
    BSpline,
    Concatenate,
    Cos,
    Divide,
    Equal,
    Exp,
    Expit,
    Extract,
    IsIn,
    LessEqual,
    LinearProgram,
    LinearProgramFeasible,
    LinearProgramObjectiveValue,
    LinearProgramSolution,
    Log,
    Logical,
    MatrixMultiply,
    Maximum,
    Mean,
    Minimum,
    Modulus,
    Multiply,
    NaryAdd,
    NaryMaximum,
    NaryMinimum,
    NaryMultiply,
    Not,
    Or,
    Put,
    Resize,
    Rint,
    Roll,
    SafeDivide,
    Sin,
    SoftMax,
    SquareRoot,
    Subtract,
    Tanh,
    Transpose,
    Where,
    Xor,
)
from dwave.optimization.typing import ArraySymbolLike


__all__ = [
    "absolute",
    "add",
    "arange",
    "argsort",
    "as_array_symbols",
    "atleast_1d",
    "atleast_2d",
    "broadcast_shapes",
    "broadcast_symbols",
    "broadcast_to",
    "bspline",
    "concatenate",
    "cos",
    "divide",
    "equal",
    "exp",
    "expit",
    "extract",
    "hstack",
    "isin",
    "less_equal",
    "linprog",
    "log",
    "logical",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "matmul",
    "maximum",
    "mean",
    "minimum",
    "mod",
    "multiply",
    "put",
    "resize",
    "rint",
    "roll",
    "safe_divide",
    "sin",
    "softmax",
    "sqrt",
    "stack",
    "subtract",
    "tanh",
    "transpose",
    "vstack",
    "where",
]


def _binaryop(
    binaryop: type,
    naryop: type | None = None,
):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # First let's make sure we have the correct number of arguments for our function
            if len(args) < 2:
                raise TypeError(f"{f.__name__}() missing 2 required positional arguments")
            if len(args) > 2 and naryop is None:
                # If we don't have a nary variation then we don't support more than two arguments
                raise TypeError(
                    f"{f.__name__}() takes 2 positional arguments but {len(args)} were given"
                )

            # For simplicity, let's convert all of the array-likes into np.ndarray.
            # We don't yet go all of the way to ArraySymbol because we'd like to avoid
            # this function having side effects (i.e., adding a new symbol to the model)
            # We also force float64 because that's the underlying type we use, but if we
            # eventually support multiple dtypes we'll want to be more general.
            arrays: tuple[np.typing.NDArray | ArraySymbol] = tuple(
                arg if isinstance(arg, ArraySymbol) else np.asarray(arg, dtype=np.float64) for arg in args
            )

            # Now get the shape we'd be broadcasting to
            shape = broadcast_shapes(
                *(arr.shape() if isinstance(arr, ArraySymbol) else arr.shape for arr in arrays),
            )

            # Ok, no errors so far so our shapes are compatible. Now we need to
            # actually convert them into array symbols
            array_symbols = as_array_symbols(*arrays)

            # And finally reshape them for broadcasting. BinaryOp specifically
            # supports broadcasting length 1 arrays, so we can avoid some
            # reshapes in that case
            if len(args) == 2:
                reshaped = tuple(
                    broadcast_to(sym, shape) if np.prod(sym.shape()) != 1 else sym
                    for sym in array_symbols
                )
                return binaryop(*reshaped, **kwargs)

            # For NaryOp though we need to do full broadcasting.
            # If naryop is None we shouldn't have been able to get here.
            return naryop(*broadcast_symbols(*array_symbols), **kwargs)

        return wrapper
    return decorator


absolute = abs
"""Return the absolute value element-wise on a symbol.

An alias for the Python :func:`abs` function.

Args:
    x (:class:`~dwave.optimization.model.ArraySymbol`): Array symbol.
Returns:
    :class:`~dwave.optimization.symbols.Absolute`: Successor symbol that
    returns the absolute value element-wise on the predecessor symbol.

Examples:
    This example adds the minimum absolute value of an integer decision variable
    to a model.

    >>> from dwave.optimization.model import Model
    >>> from dwave.optimization.mathematical import absolute
    ...
    >>> model = Model()
    >>> i = model.integer((3, 3), lower_bound=-10)
    >>> j = absolute(i).min()
    >>> model.states.resize(1)
    >>> with model.lock():
    ...     i.set_state(0, [[4, -3, 1], [-2, 1, 2], [-4, 7, 22]])
    ...     print(j.state(0))
    1.0

See Also:
    :class:`~dwave.optimization.symbols.Absolute`: Generated symbol

    :func:`abs` (Python function), :data:`~numpy.absolute`
    (:doc:`NumPy <numpy:index>` function)

.. versionadded:: 0.6.2
"""


@_binaryop(binaryop=Add, naryop=NaryAdd)
def add(x1: ArraySymbolLike, x2: ArraySymbolLike, *xi: ArraySymbolLike) -> Add | NaryAdd:
    r"""Sum element-wise two or more symbols and arrays.

    Equivalently, you can use the ``+`` operator (e.g., :code:`i + j`).

    Args:
        x1, x2: Operand array symbol or |array-like|_ to be added.
        *xi: Additional array symbols or |array-like|_ to be added.

    Returns:
        Successor symbol that sums the predecessor symbols (and/or arrays)
        element-wise. For two symbols, generates an
        :class:`~dwave.optimization.symbols.Add` symbol; for three or more
        symbols, generates a :class:`~dwave.optimization.symbols.NaryAdd` symbol.

    Examples:
        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import add

        Add two symbols.

        >>> model = Model()
        >>> i = model.integer(2)
        >>> j = model.integer(2)
        >>> i_plus_j = add(i, j)    # alternatively: i_plus_j = i + j
        >>> with model.lock():
        ...     model.states.resize(1)
        ...     i.set_state(0, [3, 5])
        ...     j.set_state(0, [7, 5])
        ...     print(i_plus_j.state(0))
        [10. 10.]

        Add a symbol and array.

        >>> model = Model()
        >>> i = model.integer(2)
        >>> i_plus_array = i + [5, -3]     # equivalently: i_plus_array = add(i, [5, -3])

    See Also:
        :class:`~dwave.optimization.symbols.Add`,
        :class:`~dwave.optimization.symbols.NaryAdd`: Generated symbol

        :data:`~numpy.add`: :doc:`NumPy <numpy:index>` function

        :func:`.divide`, :func:`.mod`, :func:`.multiply`,
        :func:`.safe_divide`, :func:`.subtract`
    """
    raise RuntimeError("implemented by the _binaryop() decorator")


def arange(start: typing.Union[int, ArraySymbol, None] = None,
           stop: typing.Union[int, ArraySymbol, None] = None,
           step: typing.Union[int, ArraySymbol, None] = None,
           ) -> ArraySymbol:
    """Evenly space values within an interval.

    Args:
        start: Start of the interval; if only one argument is provided,
            interpreted as the ``stop`` argument. Default is 0.
        stop: End of the interval.
        step: Spacing between values. Default is 1

    To add the generated :class:`~dwave.optimization.symbols.ARange` symbol to
    the model's underlying :term:`directed acyclic graph`, at least one argument
    must be a symbol in the model, thus providing one or more predecessors.

    Returns:
        Successor symbol with evenly spaced integer values.

    Examples:
        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import arange
        ...
        >>> i = model.integer(1, lower_bound=-5)
        >>> points = arange(i, 4, 2)
        >>> with model.lock():
        ...     model.states.resize(2)
        ...     i.set_state(0, -4)
        ...     i.set_state(1, 0)
        ...     print(points.state(0))
        ...     print(points.state(1))
        [-4. -2.  0.  2.]
        [0. 2.]

    See Also:
        :class:`~dwave.optimization.symbols.ARange`: Generated symbol

        :func:`~numpy.arange`: :doc:`NumPy <numpy:index>` function

    .. versionadded:: 0.5.2
    """

    if stop is None and step is None:
        # then start is actually stop
        start, stop = None, start

    if start is None:
        start = 0
    if stop is None:
        raise ValueError("stop cannot be None")
    if step is None:
        step = 1

    return ARange(start, stop, step)


def argsort(array: ArraySymbol) -> ArgSort:
    """Order a symbol's indices to sort the flattened array's values.

    Creates a symbol that, for the given symbol, returns an ordering of indices
    that represents a
    `stable sort <https://en.wikipedia.org/wiki/Category:Stable_sorts>`_ of
    the values. Although the indices are for a flattened array, similar
    to NumPy's :func:`~numpy.argsort` function with ``axis=None``, this symbol
    is shaped identically to the predecessor symbol.

    Args:
        array: Array symbol to sort.

    Returns:
        Successor symbol with an ordering of the predecessor symbol's indices
        that sorts its flattened array's values.

    Examples:
        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import argsort
        ...
        >>> model = Model()
        >>> a = model.constant([5, 2, 7, 4, 9, 1])
        >>> indices = argsort(a)
        >>> indices.shape()
        (6,)
        >>> with model.lock():
        ...    model.states.resize(1)
        ...    print(indices.state())
        [5. 1. 3. 0. 2. 4.]
        >>> a = model.constant([[5, 2, 7], [4, 9, 1]])
        >>> indices = argsort(a)
        >>> indices.shape()
        (2, 3)
        >>> with model.lock():
        ...    model.states.resize(1)
        ...    print(indices.state())
        [[5. 1. 3.]
         [0. 2. 4.]]

    See Also:
        :class:`~dwave.optimization.symbols.ArgSort`: Generated symbol

        :func:`~numpy.argsort`: :doc:`NumPy <numpy:index>` function

    .. versionadded:: 0.6.4
    """
    return ArgSort(array)


def as_array_symbols(
    *arrays: ArraySymbolLike,
    model: Model | None = None,
) -> tuple[ArraySymbol, ...]:
    """Convert arrays to array symbols.

    Args:
        *arrays: Arrays to convert.
        model: Model to which to add the generated
            :class:`~dwave.optimization.model.ArraySymbol` symbols. If not
            specified, inferred from the given inputs.

    Returns:
        Tuple of symbols representing the converted arrays.

    Examples:
        >>> import numpy as np
        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import as_array_symbols
        ...
        >>> model = Model()
        >>> c = as_array_symbols(np.random.randint(
        ...     low=-3,
        ...     high=5,
        ...     size=(15, 25)),
        ...     model=model)
        >>> min_c = c[0].min()
        >>> with model.lock():
        ...    model.states.resize(1)
        ...    print(min_c.state(0) >= -3)
        True

    See Also:
        :class:`~dwave.optimization.model.ArraySymbol`: Generated symbol

        :func:`.atleast_1d`, :func:`.atleast_2d`,
        :meth:`~dwave.optimization.model.Model.constant`,
        :func:`~dwave.optimization.model.Model.input`

    .. versionadded:: 0.6.11
    """
    # Find the model, if one wasn't provided
    if model is None:
        for array in arrays:
            if isinstance(array, Symbol):
                model = array.model
                break
        else:
            raise TypeError("given arrays must contain at least one ArraySymbol")

    # Try to convert anything that isn't already an array symbol into a constant
    out: list[ArraySymbol] = []
    for array in arrays:
        if isinstance(array, ArraySymbol):
            if array.model is not model:
                raise ValueError("given array symbols do not all share the same underlying model")
            out.append(array)
        else:
            try:
                out.append(model.constant(array))
            except TypeError as err:
                raise TypeError(f"{array!r} cannot be interpreted as an array symbol") from err

    return tuple(out)


@typing.overload
def atleast_1d(array: ArraySymbol) -> ArraySymbol: ...
@typing.overload
def atleast_1d(*arrays: ArraySymbol) -> tuple[ArraySymbol, ...]: ...


def atleast_1d(*arrays):
    """Ensure array symbols are at least one dimensional.

    Args:
        arrays: One or more array symbols.
    Returns:
        Successor array symbol, or a tuple of successor array symbols, with at
        least one dimension.

    Examples:
        >>> from dwave.optimization import atleast_1d, Model
        ...
        >>> model = Model()
        >>> a = model.constant(1)
        >>> b = model.constant([2])
        >>> c, d = atleast_1d(a, b)
        >>> c.shape()  # a is converted into a 1d array
        (1,)
        >>> d.shape()  # b is not changed
        (1,)
        >>> d is b
        True
        >>> e = model.constant([[0, 1, 2], [3, 4, 5]])
        >>> f = atleast_1d(e)  # e is not changed
        >>> f.shape()
        (2, 3)
        >>> f is e
        True

    See Also:
        :func:`~numpy.atleast_2d`: :doc:`NumPy <numpy:index>` function

        :func:`.as_array_symbols`, :func:`.atleast_2d()`

    .. versionadded:: 0.6.0
    """
    result: list[ArraySymbol] = []
    for array in arrays:
        if not isinstance(array, ArraySymbol):
            raise TypeError(f"{array!r} is not an array symbol")
        if array.ndim() == 0:
            result.append(array.reshape(1))
        else:
            result.append(array)
    if len(result) == 1:
        return result[0]
    return tuple(result)


@typing.overload
def atleast_2d(array: ArraySymbol) -> ArraySymbol: ...
@typing.overload
def atleast_2d(*arrays: ArraySymbol) -> tuple[ArraySymbol, ...]: ...


def atleast_2d(*arrays):
    """Ensure array symbols are at least two dimensional.

    Args:
        arrays: One or more array symbols.
    Returns:
        Successor array symbol, or a tuple of successor array symbols, with at
        least two dimensions.

    Examples:
        >>> from dwave.optimization import atleast_2d, Model
        ...
        >>> model = Model()
        >>> a = model.constant(1)
        >>> b = model.constant([[2]])
        >>> c, d = atleast_2d(a, b)
        >>> c.shape()  # a is converted into a 2d array
        (1, 1)
        >>> d.shape()  # c is not changed
        (1, 1)
        >>> d is b
        True
        >>> e = model.constant([[0, 1, 2], [3, 4, 5]])
        >>> f = atleast_2d(e)  # e is not changed
        >>> f.shape()
        (2, 3)
        >>> f is e
        True

    See Also:
        :func:`~numpy.atleast_2d`: :doc:`NumPy <numpy:index>` function

        :func:`.as_array_symbols`, :func:`atleast_1d()`

    .. versionadded:: 0.6.0
    """
    result: list[ArraySymbol] = []
    for array in arrays:
        if not isinstance(array, ArraySymbol):
            raise TypeError(f"{array!r} is not an array symbol")
        shape = array.shape()
        if len(shape) == 0:
            result.append(array.reshape(1, 1))
        elif len(shape) == 1:
            result.append(array.reshape(1, *shape))
        else:
            result.append(array)
    if len(result) == 1:
        return result[0]
    return tuple(result)


def broadcast_shapes(*shapes: int | tuple[int, ...]) -> tuple[int, ...]:
    """Calculate the broadcast shape.

    You can set a value of :math:`-1` for a dimension to calculate the broadcast
    shape for
    :ref:`dynamically sized <optimization_philosophy_tensor_programming_dynamic>`
    array symbols.

    See NumPy's :ref:`broadcasting <basics.broadcasting>` documentation for
    an introduction to broadcasting.

    Args:
        *shapes: Shapes to be broadcast.

    Returns:
        Calculated broadcasted shape.

    Examples:
        >>> from dwave.optimization import broadcast_shapes
        >>> broadcast_shapes((1, 3), (2, 1))
        (2, 3)
        >>> broadcast_shapes((-1, 5), (1,))
        (-1, 5)

    See Also:
        :func:`~numpy.broadcast_shapes`: :doc:`NumPy <numpy:index>` function

        :func:`.broadcast_symbols`, :func:`.broadcast_to`

        :meth:`~dwave.optimization.model.ArraySymbol.copy`,
        :meth:`~dwave.optimization.model.ArraySymbol.reshape`
        :meth:`~dwave.optimization.model.ArraySymbol.resize`
        :func:`.roll`
        :func:`.transpose`

    .. versionadded:: 0.6.11
    """
    shape: tuple[int, ...] = tuple()
    for s in shapes:
        if isinstance(s, numbers.Integral):
            shape = _broadcast_shapes(shape, (int(s),))
        else:
            shape = _broadcast_shapes(shape, s)
    return shape


def broadcast_symbols(*args: ArraySymbol) -> tuple[ArraySymbol, ...]:
    """Broadcast multiple array symbols to a single shape.

    For
    :ref:`dynamically sized <optimization_philosophy_tensor_programming_dynamic>`
    array symbols, you can make use of the :math:`-1` value in the dynamic
    dimension, as shown in the examples.

    See NumPy's :ref:`broadcasting <basics.broadcasting>` documentation for
    an introduction to broadcasting.

    Args:
        *args: Array symbols to broadcast.

    Returns:
        Tuple of successor array symbols, all with the same shape. For each
        predecessor symbol, a
        new :class:`~dwave.optimization.symbols.BroadcastTo` symbol is returned
        unless the predecessor already has the broadcast shape, in which case
        the predecessor itself returned.

    Examples:
        >>> from dwave.optimization import broadcast_symbols, broadcast_to, Model
        ...
        >>> model = Model()
        >>> x = model.integer(3)  # shape = (3,)
        >>> y = model.constant([[0], [1]])  # shape = (2, 1)
        >>> xb, yb = broadcast_symbols(x, y)
        >>> xb.shape()
        (2, 3)
        >>> yb.shape()
        (2, 3)
        >>> # Dynamically sized array:
        >>> s1 = model.set(4).reshape(-1, 1)
        >>> s1.shape()
        (-1, 1)
        >>> s2 = broadcast_to(s1, (-1, 2))
        >>> s2.shape()
        (-1, 2)
        >>> s3, s4 = broadcast_symbols(s1, s2)
        >>> s3.shape()
        (-1, 2)
        >>> s4 is s2
        True

    See Also:
        :class:`~dwave.optimization.symbols.BroadcastTo`: Generated symbol

        :class:`~numpy.broadcast`: :doc:`NumPy <numpy:index>` function

        :func:`.broadcast_shapes`, :func:`.broadcast_to`

        :meth:`~dwave.optimization.model.ArraySymbol.copy`,
        :meth:`~dwave.optimization.model.ArraySymbol.reshape`
        :meth:`~dwave.optimization.model.ArraySymbol.resize`
        :func:`.roll`
        :func:`.transpose`

    .. versionadded:: 0.6.5
    """
    shape = broadcast_shapes(*(x.shape() for x in args))
    return tuple(broadcast_to(x, shape) for x in args)


def broadcast_to(x: ArraySymbol, shape: tuple[int, ...] | int) -> ArraySymbol:
    """Broadcast an array symbol to a new shape.

    For
    :ref:`dynamically sized <optimization_philosophy_tensor_programming_dynamic>`
    array symbols, you can make use of the :math:`-1` value in the dynamic
    dimension, as shown in the examples.

    See NumPy's :ref:`broadcasting <basics.broadcasting>` documentation for
    an introduction to broadcasting.

    Args:
        x: Array symbol to broadcast to a new shape.
        shape: Shape for the broadcasted symbol. A single integer ``i`` is
            interpreted as ``(i,)``.

    Returns:
        Successor :class:`~dwave.optimization.symbols.BroadcastTo` symbol unless
        its predecessor symbol has the requested shape, in which case the
        predecessor itself is returned.

    Examples:
        >>> from dwave.optimization import broadcast_to, Model
        ...
        >>> model = Model()
        >>> x = model.constant([0, 1, 2, 3, 4])
        >>> x.shape()
        (5,)
        >>> y = broadcast_to(x, (2, 5))
        >>> y.shape()
        (2, 5)
        >>> model.states.resize(1)
        >>> with model.lock():
        ...     y.state()
        array([[0., 1., 2., 3., 4.],
               [0., 1., 2., 3., 4.]])
        >>> # Dynamically sized array:
        >>> s = model.set(4)
        >>> s.shape()
        (-1,)
        >>> s4 = broadcast_to(s.reshape(-1, 1), (-1, 4))
        >>> with model.lock():
        ...     s.set_state(0, {2, 3})
        ...     print(s4.state(0))
        [[2. 2. 2. 2.]
         [3. 3. 3. 3.]]

    See Also:
        :class:`~dwave.optimization.symbols.BroadcastTo`: Generated symbol

        :func:`~numpy.broadcast_to`: :doc:`NumPy <numpy:index>` function

        :func:`.broadcast_shapes`, :func:`.broadcast_symbols`

        :meth:`~dwave.optimization.model.ArraySymbol.copy`,
        :meth:`~dwave.optimization.model.ArraySymbol.reshape`,
        :meth:`~dwave.optimization.model.ArraySymbol.resize`,
        :func:`.roll`,
        :func:`.transpose`

    .. versionadded:: 0.6.5
    """
    # We only create a new symbol if necessary
    if x.shape() == shape or (isinstance(shape, int) and x.shape() == (shape,)):
        return x

    return BroadcastTo(x, shape)


def bspline(x: ArraySymbol, k: int, t: list, c: list) -> ArraySymbol:
    """Return B-spline values for a symbol.

    Args:
        x: Array symbol for which to create B-spline values.
        k: B-spline degree.
        t: Knots.
        c: Spline coefficients.

    Returns:
        Successor symbol with B-spline values for the predecessor symbol.

    Examples:
        >>> from dwave.optimization import bspline, Model
        ...
        >>> model = Model()
        >>> x = model.integer(3, lower_bound=2, upper_bound=4)
        >>> bs = bspline(x, k=2, t=[0, 1, 2, 3, 4, 5, 6], c=[-1, 2, 0, -1])
        >>> with model.lock():
        ...     model.states.resize(1)
        ...     x.set_state(0, [2, 3, 4])
        ...     print(bs.state(0))
        [ 0.5  1.  -0.5]

    See Also:
        :class:`~dwave.optimization.symbols.BSpline`: Generated symbol

        :class:`~scipy.interpolate.BSpline`: :doc:`SciPy <scipy:index>` class

    .. versionadded:: 0.5.4
    """
    return BSpline(x, k, t, c)


def concatenate(arrays: typing.Sequence[ArraySymbol],
                axis: int = 0,
                ) -> ArraySymbol:
    """Concatenate one or more symbols across an axis.

    Args:
        arrays: Array symbols to concatenate.
        axis: The concatenation axis. Default is axis 0 (vertical stacking in
            the case of 2D arrays).

    Returns:
        Successor symbol that concatenates the predecessor symbols along the
        specified axis.

    Examples:
        This example concatenates two constant symbols along the first axis.

        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import concatenate
        ...
        >>> model = Model()
        >>> a = model.constant([[0,1], [2,3]])
        >>> b = model.constant([[4,5]])
        >>> a_b = concatenate((a,b), axis=0)
        >>> a_b.shape()
        (3, 2)
        >>> with model.lock():
        ...     model.states.resize(1)
        ...     print(a_b.state(0))
        [[0. 1.]
         [2. 3.]
         [4. 5.]]

    See Also:
        :class:`~dwave.optimization.symbols.Concatenate`: Generated symbol

        :func:`~numpy.concatenate`: :doc:`NumPy <numpy:index>` function

        :func:`~dwave.optimization.mathematical.hstack`,
        :func:`~dwave.optimization.mathematical.stack`,
        :func:`~dwave.optimization.mathematical.vstack`

    .. versionadded:: 0.4.3
    """
    return Concatenate(arrays, axis=axis)


def cos(x: ArraySymbol) -> Cos:
    """Calculate element-wise the trigonometric cosine of a symbol.

    Args:
        x: Array giving the angles, in radians.

    Returns:
        Successor symbol that is the trigonometric cosine of the values in its
        predecessor symbol.

    Examples:
        >>> import numpy as np
        >>> from dwave.optimization import Model, cos
        ...
        >>> model = Model()
        >>> x = model.constant([0, np.pi / 2])
        >>> y = cos(x)
        ...
        >>> model.states.resize(1)
        >>> with model.lock():
        ...     print(y.state())
        [1.000000e+00 6.123234e-17]

    See Also:
        :class:`~dwave.optimization.symbols.Cos`: Generated symbol

        :data:`~numpy.cos`: :doc:`NumPy <numpy:index>` function

        :func:`.sin`, :func:`.tanh`

    .. versionadded:: 0.6.5
    """
    return Cos(x)


@_binaryop(Divide)
def divide(x1: ArraySymbolLike, x2: ArraySymbolLike) -> Divide:
    r"""Divide element-wise two symbols (or arrays).

    Equivalently, you can use the ``/`` operator (e.g., :code:`i / j`).

    Args:
        x1: Numerator array symbol or |array-like|_.
        x2: Denominator array symbol or |array-like|_.

    Returns:
        Successor symbol that divides the predecessor symbols (or arrays)
        element-wise.

    Raises:
        ValueError: If a denominator element is zero. See also the
            :func:`.safe_divide` function.

    Examples:
        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import divide

        Divide two symbols.

        >>> model = Model()
        >>> i = model.integer(2, lower_bound=1)
        >>> j = model.integer(2, lower_bound=1)
        >>> k = divide(i, j)   # alternatively: k = i / j
        >>> with model.lock():
        ...     model.states.resize(1)
        ...     i.set_state(0, [21, 10])
        ...     j.set_state(0, [7, 2])
        ...     print(k.state(0))
        [3. 5.]

        Divide a symbol and array.

        >>> model = Model()
        >>> i = model.integer(2, lower_bound=1)
        >>> l = i / [2, 0.5] # equivalently: l = divide(i, [2, 0.5])


    See Also:
        :class:`~dwave.optimization.symbols.Divide`: Generated symbol

        :data:`~numpy.divide`: :doc:`NumPy <numpy:index>` function

        :func:`.add`, :func:`.mod`, :func:`.multiply`,
        :func:`.safe_divide`, :func:`.subtract`
    """
    raise RuntimeError("implemented by the _binaryop() decorator")


@_binaryop(Equal)
def equal(x1: ArraySymbolLike, x2: ArraySymbolLike) -> Equal:
    """Return element-wise equality between two symbols (or arrays).

    Equivalently, you can use the ``==`` operator (e.g., :code:`i == j`).

    Args:
        x1, x2: Array symbol or |array-like|_.

    Returns:
        Successor symbol that is the element-wise equality of its predecessor
        symbols (or arrays).

    Examples:
        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import equal

        Equality between symbols.

        >>> model = Model()
        >>> i = model.integer(3, lower_bound=1)
        >>> c = model.constant([[1, 3, 5], [2, 4, 6]])
        >>> i_c = equal(i, c[0, :])     # Alternatively, i_c = i == c[0, :]
        >>> with model.lock():
        ...     model.states.resize(1)
        ...     i.set_state(0, [1, 3, 7])
        ...     print(i_c.state(0))
        [1. 1. 0.]

        Equality between a symbol and array.

        >>> model = Model()
        >>> i = model.integer(3, lower_bound=1)
        >>> i_array = i == [2, 6, 8] # equivalently: i_array = equal(i, [2, 6, 8])

    See Also:
        :class:`~dwave.optimization.symbols.Equal`: Generated symbol

        :data:`~numpy.equal`: :doc:`NumPy <numpy:index>` function

        :func:`.less_equal`, :func:`.isin`
    """
    raise RuntimeError("implemented by the _binaryop() decorator")


def exp(x: ArraySymbol) -> Exp:
    """Calculate element-wise the natural (base-e) exponential of a symbol.

    Args:
        x: Array symbol.

    Returns:
        Successor symbol that is the base-e exponential values of its
        predecessor symbol's array elements.

    Examples:
        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import exp
        ...
        >>> model = Model()
        >>> x = model.constant(1.0)
        >>> exp_x = exp(x)
        >>> model.states.resize(1)
        >>> with model.lock():
        ...     print(exp_x.state())
        2.718281828459045

    See Also:
        :class:`~dwave.optimization.symbols.Exp`: Generated symbol

        :data:`~numpy.exp`: :doc:`NumPy <numpy:index>` function

        :func:`.expit`, :func:`.log`, :func:`.softmax`

    .. versionadded:: 0.6.2
    """
    return Exp(x)

def expit(x: ArraySymbol) -> Expit:
    """Calculate element-wise the logistic sigmoid of a symbol.

    Args:
        x: Array symbol.

    Returns:
        Successor symbol that is the logistic sigmoid values of its predecessor
        symbol's array elements.

    Examples:
        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import expit
        ...
        >>> model = Model()
        >>> x = model.constant(1.0)
        >>> expit_x = expit(x)
        >>> model.states.resize(1)
        >>> with model.lock():
        ...     print(expit_x.state())
        0.7310585786300049

    See Also:
        :class:`~dwave.optimization.symbols.Expit`: Generated symbol

        :data:`~scipy.special.expit`: :doc:`SciPy <scipy:index>` function

        :func:`.exp`, :func:`.log`, :func:`.softmax`

    .. versionadded:: 0.5.2
    """
    return Expit(x)


def extract(condition: ArraySymbol, arr: ArraySymbol) -> Extract:
    """Extract the elements of an array symbol where a condition is true.

    Args:
        condition:
            Condition under which to return elements of ``arr`` when it
            evaluates as ``True``.
        arr:
            Array symbol.

    Returns:
        Successor symbol that contains the elements of its predecessor symbol
        that meet the specified condition.

    Examples:
        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import extract
        ...
        >>> model = Model()
        >>> condition = model.binary(3)
        >>> arr = model.constant([4, 5, 6])
        >>> extracted = extract(condition, arr)
        >>> model.states.resize(1)
        >>> with model.lock():
        ...     condition.set_state(0, [True, False, True])
        ...     print(extracted.state())
        [4. 6.]

    See Also:
        :class:`~dwave.optimization.symbols.Extract`: Generated symbol

        :func:`~numpy.extract`: :doc:`NumPy <numpy:index>` function

        :func:`.isin`, :func:`.put`, :func:`.where`

    .. versionadded:: 0.6.3
    """
    return Extract(condition, arr)


def hstack(arrays: collections.abc.Sequence[ArraySymbol]) -> ArraySymbol:
    """Horizontally stack a sequence of array symbols.

    Equivalent to concatenation along the second axis, except for
    one-dimensional array symbols, which it concatenates along the first axis.

    Args:
        arrays: Array symbols to be concatenated. Arrays must have the same
            shape along all but the second axis unless they are 1D arrays.

    Returns:
        Successor symbol that concatenates its predecessor array symbols.

    Examples:

        >>> from dwave.optimization import hstack, Model

        >>> model = Model()
        >>> model.states.resize(1)
        ...
        >>> a = model.constant([0])
        >>> b = model.constant([1, 2, 3])
        >>> h = hstack((a, b))
        >>> with model.lock():
        ...     h.state()
        array([0., 1., 2., 3.])

        >>> model = Model()
        >>> model.states.resize(1)
        ...
        >>> a = model.constant([[0], [3]])
        >>> b = model.constant([[1, 2], [4, 5]])
        >>> h = hstack((a, b))
        >>> with model.lock():
        ...     h.state()
        array([[0., 1., 2.],
               [3., 4., 5.]])

    See Also:
        :class:`~dwave.optimization.symbols.Concatenate`: Generated symbol

        :func:`~numpy.hstack`: :doc:`NumPy <numpy:index>` function

        :func:`~dwave.optimization.mathematical.concatenate`,
        :func:`~dwave.optimization.mathematical.stack`,
        :func:`~dwave.optimization.mathematical.vstack`

    .. versionadded:: 0.6.0
    """
    arrays = atleast_1d(*arrays)
    if not isinstance(arrays, tuple):
        arrays = (arrays,)
    if arrays and arrays[0].ndim() == 1:
        return concatenate(arrays, 0)
    else:
        return concatenate(arrays, 1)


def isin(element: ArraySymbol, test_elements: ArraySymbol) -> IsIn:
    """Return which values of one array symbol are in another.

    Determines element-wise containment between two symbols.

    Args:
        element: Array symbol.
        test_elements: Array symbol containing the values against which to test
            each value of the ``element`` symbol.

    Returns:
        Successor array symbol, with the same shape as ``element``, with values
        being the element-wise containment between its two predecessor symbols;
        i.e., true if the corresponding element in the first symbol is contained
        in the second and false otherwise.

    Examples:
        >>> from dwave.optimization.model import Model
        >>> from dwave.optimization.mathematical import isin
        ...
        >>> model = Model()
        >>> model.states.resize(1)
        >>> element = model.constant([1.3, -1.0, 3.0])
        >>> test_elements = model.constant([3.0, -1.1, -2.0, 4.1, -1.0])
        >>> contains = isin(element, test_elements)
        >>> with model.lock():
        ...     print(contains.state(0))
        [0. 1. 1.]

    See Also:
        :class:`~dwave.optimization.symbols.IsIn`: Generated symbol

        :func:`~numpy.isin`: :doc:`NumPy <numpy:index>` function

        :func:`.extract`, :func:`.put`, :func:`.where`

    .. versionadded:: 0.6.8
    """
    return IsIn(element, test_elements)


class LPResult:
    """Outputs of a linear program.

    See the :func:`.linprog` function for information and usage.
    """

    lp: LinearProgram
    """Linear-program symbol.

    See Also:
        :class:`~dwave.optimization.symbols.LinearProgram`
    """

    def __init__(self, lp: LinearProgram):
        self.lp = lp

    @functools.cached_property
    def fun(self) -> LinearProgramObjectiveValue:
        """Value of the objective.

        See Also:
            :class:`~dwave.optimization.symbols.LinearProgramObjectiveValue`
        """
        return LinearProgramObjectiveValue(self.lp)

    @functools.cached_property
    def success(self) -> LinearProgramFeasible:
        """True if the linear program's indexed state is a feasible solution.

        See Also:
            :class:`~dwave.optimization.symbols.LinearProgramFeasible`
        """
        return LinearProgramFeasible(self.lp)

    @functools.cached_property
    def x(self) -> LinearProgramSolution:
        """Assignments of the decision variables.

        See Also:
            :class:`~dwave.optimization.symbols.LinearProgramSolution`
        """
        return LinearProgramSolution(self.lp)


@_binaryop(LessEqual)
def less_equal(x1: ArraySymbolLike, x2: ArraySymbolLike) -> LessEqual:
    """Return element-wise less than or equal between two symbols (or arrays).

    Args:
        x1, x2: Operand array symbol or |array-like|_.

    Returns:
        Successor symbol that is the element-wise less than or equal
        (:math:`x_1 \le x_2`) for its predecessor symbols (or arrays).

    Examples:
        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import less_equal
        ...
        >>> model = Model()
        >>> i = model.integer(3, lower_bound=1)
        >>> c = model.constant([[1, 3, 5], [2, 4, 6]])
        >>> i_c = less_equal(i, c[0, :])     # Alternatively, i_c = i <= c[0, :]
        >>> with model.lock():
        ...     model.states.resize(1)
        ...     i.set_state(0, [1, 2, 7])
        ...     print(i_c.state(0))
        [1. 1. 0.]
        >>> # Testing equality between a symbol and an array
        >>> i_array = i <= [2, 1, 8] # equivalently: i_array = less_equal(i, [2, 1, 8])

    See Also:
        :class:`~dwave.optimization.symbols.LessEqual`: Generated symbol

        :data:`~numpy.less_equal`: :doc:`NumPy <numpy:index>` function

        :func:`.equal`, :func:`.isin`
    """
    raise RuntimeError("implemented by the _binaryop() decorator")


def linprog(
        c: ArraySymbol,
        A_ub: None | ArraySymbol = None,  # alias for A
        b_ub: None | ArraySymbol = None,
        A_eq: None | ArraySymbol = None,
        b_eq: None | ArraySymbol = None,
        *,  # the args up until here match SciPy's linprog() which accepts them positionally
        b_lb: None | ArraySymbol = None,
        A: None | ArraySymbol = None,
        lb: None | ArraySymbol = None,
        ub: None | ArraySymbol = None,
        ) -> LPResult:
    r"""Solve a :term:`linear program`.

    Linear programs solve problems of the form:

    .. math::
        \text{minimize:} \\
        & c^T x \\
        \text{subject to:} \\
        &b_{lb} \leq A x \leq b_{ub} \\
        &A_{eq} x = b_{eq} \\
        &lb \leq x \leq ub

    Or, equivalently:

    .. math::
        \text{minimize:} \\
        & c^T x \\
        \text{subject to:} \\
        &A_{ub} x \leq b_{ub} \\
        &A_{eq} x = b_{eq} \\
        &lb \leq x \leq ub

    Args:
        c: Coefficients of the linear objective as a 1D array symbol.
        A_ub: Alias of ``A``. At most one of
            ``A_ub`` and ``A`` may be specified.
        b_ub: Linear inequality upper bounds as a 1D array symbol.
        A_eq: Linear equality matrix as a 2D array symbol.
        b_eq: Linear equality bounds as a 1D array symbol.
        b_lb: Linear inequality lower bounds as a 1D array symbol.
        A: Linear inequality matrix as a 2D array symbol. At most one of
            ``A_ub`` and ``A`` may be specified.
        lb: Lower bounds on ``x`` as a 1D array symbol.
        ub: Upper bounds on ``x`` as a 1D array symbol.

    Returns:
        :class:`.LPResult` class containing the results of the linear program.

    Examples:
        This example adds the following linear program to a model.

        .. math::
            \text{minimize: } & -x_0 - 2x_1 \\
            \text{subject to: } & x_0 + x_1 <= 1

        >>> from dwave.optimization import linprog, Model
        ...
        >>> model = Model()
        >>> c = model.constant([-1, -2])
        >>> A_ub = model.constant([[1, 1]])
        >>> b_ub = model.constant([1])
        >>> res = linprog(c, A_ub=A_ub, b_ub=b_ub)
        >>> _ = model.add_constraint(res.success)  # tell the model you want feasible solutions
        ...
        >>> model.states.resize(1)
        >>> x = res.x
        >>> with model.lock():
        ...     x.state()
        array([0., 1.])

    See Also:
        :class:`~dwave.optimization.symbols.LinearProgram`,
        :class:`~dwave.optimization.symbols.LinearProgramFeasible`,
        :class:`~dwave.optimization.symbols.LinearProgramObjectiveValue`,
        :class:`~dwave.optimization.symbols.LinearProgramSolution`: generated
        symbols.

        :func:`~scipy.optimize.linprog`: :doc:`SciPy <scipy:index>` function

    .. versionadded:: 0.6.0
    """
    if A is not None and A_ub is not None:
        raise ValueError("can provide A or A_ub, but not both")
    elif A_ub is not None:
        A = A_ub

    return LPResult(LinearProgram(c, b_lb, A, b_ub, A_eq, b_eq, lb, ub))


def log(x: ArraySymbol) -> Log:
    """Calculate element-wise the natural logarithm of a symbol.

    Args:
        x: Array symbol.

    Returns:
        Successor symbol that is the natural logarithm values of of its
        predecessor symbol's array elements.

    Examples:
        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import log
        ...
        >>> model = Model()
        >>> x = model.constant(1.0)
        >>> log_x = log(x)
        >>> model.states.resize(1)
        >>> with model.lock():
        ...     print(log_x.state())
        0.0

    See Also:
        :class:`~dwave.optimization.symbols.Log`: Generated symbol

        :data:`~numpy.log`: :doc:`NumPy <numpy:index>` function

        :func:`.exp`, :func:`.expit`, :func:`.softmax`

    .. versionadded:: 0.5.2
    """
    return Log(x)


def logical(x: ArraySymbol) -> Logical:
    r"""Return the element-wise truth value of a symbol.

    Args:
        x: Array symbol.

    Returns:
        Successor symbol that is the element-wise truth value of its predecessor
        symbol's array elements.

    Examples:
        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import logical
        ...
        >>> model = Model()
        >>> x = model.constant([0, 1, -2, .5])
        >>> logical_x = logical(x)
        >>> model.states.resize(1)
        >>> with model.lock():
        ...     print(logical_x.state())
        [0. 1. 1. 1.]

    See Also:
        :class:`~dwave.optimization.symbols.Logical`: Generated symbol

        :func:`.logical_and`, :func:`.logical_not`, :func:`.logical_or`,
        :func:`.logical_xor`
    """
    return Logical(x)


@_binaryop(And)
def logical_and(x1: ArraySymbolLike, x2: ArraySymbolLike) -> And:
    r"""Calculate element-wise the logical AND of two symbols (or arrays).

    Args:
        x1, x2: Operand array symbol or |array-like|_.

    Returns:
        Successor symbol that is the element-wise AND of the array elements of
        its predecessor symbols (or arrays).

    Examples:
        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import logical_and
        ...
        >>> model = Model()
        >>> x = model.binary(3)
        >>> y = model.binary(3)
        >>> z = logical_and(x, y)
        >>> with model.lock():
        ...     model.states.resize(1)
        ...     x.set_state(0, [True, True, False])
        ...     y.set_state(0, [False, True, False])
        ...     print(z.state(0))
        [0. 1. 0.]
        >>> # AND between a symbol and an array
        >>> z_array = logical_and(x, [True, False, True])

    See Also:
        :class:`~dwave.optimization.symbols.And`: Generated symbol

        :data:`~numpy.logical_and`: :doc:`NumPy <numpy:index>` function

        :func:`.logical`, :func:`.logical_not`, :func:`.logical_or`,
        :func:`.logical_xor`
    """
    raise RuntimeError("implemented by the _binaryop() decorator")


def logical_not(x: ArraySymbol) -> Not:
    r"""Calculate element-wise the logical NOT of a symbol.

    Args:
        x: Array symbol.

    Returns:
        Successor symbol that is the element-wise NOT of the array elements of
        its predecessor symbol.

    Examples:
        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import logical_not
        ...
        >>> model = Model()
        >>> x = model.constant([0, 1, -2, .5])
        >>> not_x = logical_not(x)
        >>> model.states.resize(1)
        >>> with model.lock():
        ...     print(not_x.state())
        [1. 0. 0. 0.]

    See Also:
        :class:`~dwave.optimization.symbols.Not`: Generated symbol

        :data:`~numpy.logical_not`: :doc:`NumPy <numpy:index>` function

        :func:`.logical`, :func:`.logical_and`, :func:`.logical_or`,
        :func:`.logical_xor`
    """
    return Not(x)


@_binaryop(Or)
def logical_or(x1: ArraySymbolLike, x2: ArraySymbolLike) -> Or:
    r"""Calculate element-wise the logical OR of two symbols (or arrays).

    Args:
        x1, x2: Operand array symbol or |array-like|_.

    Returns:
        Successor symbol that is the element-wise OR of the array elements of
        its predecessor symbols (or arrays).

    Examples:
        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import logical_or
        ...
        >>> model = Model()
        >>> x = model.binary(3)
        >>> y = model.binary(3)
        >>> z = logical_or(x, y)
        >>> with model.lock():
        ...     model.states.resize(1)
        ...     x.set_state(0, [True, True, False])
        ...     y.set_state(0, [False, True, False])
        ...     print(z.state(0))
        [1. 1. 0.]
        >>> # OR between a symbol and an array
        >>> z_array = logical_or(x, [True, False, True])

    See Also:
        :class:`~dwave.optimization.symbols.Or`: equivalent symbol.

        :data:`~numpy.logical_or`: :doc:`NumPy <numpy:index>` function

        :func:`.logical`, :func:`.logical_and`, :func:`.logical_not`,
        :func:`.logical_xor`
    """
    raise RuntimeError("implemented by the _binaryop() decorator")


@_binaryop(Xor)
def logical_xor(x1: ArraySymbolLike, x2: ArraySymbolLike) -> Xor:
    r"""Calculate element-wise logical XOR of two symbols (or arrays).

    Args:
        x1, x2: Operand array symbol or |array-like|_.

    Returns:
        Successor symbol that is the element-wise XOR of the array elements of
        its predecessor symbols (or arrays).

    Examples:
        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import logical_xor
        ...
        >>> model = Model()
        >>> x = model.binary(3)
        >>> y = model.binary(3)
        >>> z = logical_xor(x, y)
        >>> with model.lock():
        ...     model.states.resize(1)
        ...     x.set_state(0, [True, True, False])
        ...     y.set_state(0, [False, True, False])
        ...     print(z.state(0))
        [1. 0. 0.]
        >>> # XOR between a symbol and an array
        >>> z_array = logical_xor(x, [True, False, True])

    See Also:
        :class:`~dwave.optimization.symbols.Xor`: equivalent symbol.

        :data:`~numpy.logical_xor`: :doc:`NumPy <numpy:index>` function

        :func:`.logical`, :func:`.logical_and`, :func:`.logical_not`,
        :func:`.logical_or`

    .. versionadded:: 0.4.1
    """
    raise RuntimeError("implemented by the _binaryop() decorator")


def matmul(x: ArraySymbolLike, y: ArraySymbolLike) -> MatrixMultiply:
    r"""Compute the matrix product of two array symbols (or arrays).

    Args:
        x, y: Operand array symbol or |array-like|_.

    The size of the last axis of ``x`` must be equal to the size of the
    second-to-last axis of ``y``. If either ``x`` or ``y`` is one-dimensional,
    they are treated as a row or column vector respectively. If both are 1D,
    produces a scalar (and the operation is equivalent to the dot product of two
    vectors).

    Otherwise, if the shapes can be broadcast together, either by broadcasting
    missing leading axes (e.g.
    :math:`(2, 5, 3, 4, 2), (2, 6) \rightarrow (2, 5, 3, 4, 6)`) or by
    broadcasting axes for which one of the operands has size 1 (e.g.
    :math:`(3, 1, 4, 2), (1, 7, 2, 3) \rightarrow (3, 7, 4, 3)`), returns the
    result after broadcasting.

    Returns:
        Successor symbol representing the matrix product. For ``x`` and ``y``
        with shapes :math:`(..., n, k)` and :math:`(..., k, m)`, the generated
        :class:`~dwave.optimization.symbols.MatrixMultiply` has shape
        :math:`(..., n, m)`.

    Examples:
        This example computes the dot product of two integer arrays and also
        multiplies the first symbol by an array with a different shape.

        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import matmul
        ...
        >>> model = Model()
        >>> i = model.integer(3)
        >>> j = model.integer(3)
        >>> m = matmul(i, j)
        >>> m_array = matmul(i, [[1, 2], [3, 4], [5, 6]])
        >>> with model.lock():
        ...     model.states.resize(1)
        ...     i.set_state(0, [1, 2, 3])
        ...     j.set_state(0, [4, 5, 6])
        ...     print(m.state(0))
        ...     print(m_array.state(0))
        32.0
        [22. 28.]

    See Also:
        :class:`~dwave.optimization.symbols.MatrixMultiply`: Generated symbol

        :data:`~numpy.matmul`: :doc:`NumPy <numpy:index>` function

        :func:`.multiply` function and
        :meth:`~dwave.optimization.model.ArraySymbol.prod()` method

    .. versionadded:: 0.6.10
    """
    x, y = as_array_symbols(x, y)

    if not (x.ndim() == 1 or y.ndim() == 1) and x.shape()[:-2] != y.shape()[:-2]:
        # The shapes don't match, but it may be possible to do a broadcast. The
        # vector broadcast case is handled by MatrixMultiplyNode, so we need to
        # handle all the other cases by adding one or two BroadcastNodes.
        x_shape = [1,] * (y.ndim() - x.ndim()) + list(x.shape())
        y_shape = [1,] * (x.ndim() - y.ndim()) + list(y.shape())

        for i in range(len(x_shape) - 2):
            if x_shape[i] == 1:
                x_shape[i] = y_shape[i]
            elif y_shape[i] == 1:
                y_shape[i] = x_shape[i]
            elif x_shape[i] != y_shape[i]:
                raise ValueError("Could not broadcast operands")

        if tuple(x_shape) != x.shape():
            x = broadcast_to(x, tuple(x_shape))
        if tuple(y_shape) != y.shape():
            y = broadcast_to(y, tuple(y_shape))

        return MatrixMultiply(x, y)

    return MatrixMultiply(x, y)


@_binaryop(Maximum, NaryMaximum)
def maximum(
    x1: ArraySymbolLike,
    x2: ArraySymbolLike,
    *xi: ArraySymbolLike,
) -> Maximum | NaryMaximum:
    r"""Return an element-wise maximum of two or more symbols and arrays.

    Args:
        x1, x2: Operand array symbol or |array-like|_.
        *xi: Additional array symbols or |array-like|_.

    Returns:
        Successor symbol that is the element-wise maximum of the predecessor
        symbols (or arrays). For two symbols, generates a
        :class:`~dwave.optimization.symbols.Maximum` symbol; for three
        or more symbols, generates a
        :class:`~dwave.optimization.symbols.NaryMaximum` symbol.

    Examples:
        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import maximum

        Maximum between two symbols.

        >>> model = Model()
        >>> i = model.integer(2)
        >>> j = model.integer(2)
        >>> m = maximum(i, j)
        >>> with model.lock():
        ...     model.states.resize(1)
        ...     i.set_state(0, [3, 5])
        ...     j.set_state(0, [7, 2])
        ...     print(m.state(0))
        [7. 5.]

        Maximum between a symbol and array.

        >>> model = Model()
        >>> i = model.integer(2)
        >>> m_array = maximum(i, [2, 6])

    See Also:
        :class:`~dwave.optimization.symbols.Maximum`,
        :class:`~dwave.optimization.symbols.NaryMaximum`: Generated symbol

        :data:`~numpy.maximum`: :doc:`NumPy <numpy:index>` function

        :meth:`~dwave.optimization.model.ArraySymbol.max` method
    """
    raise RuntimeError("implemented by the _binaryop() decorator")


def mean(array: ArraySymbol) -> Mean:
    r"""Calculate the mean value of an array symbol.

    Args:
        array: Array symbol.

    Returns:
        Successor symbol that is the mean of its predecessor symbol's array
        elements. If the predecessor symbol is empty, mean defaults to 0.0.

    Examples:
        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import mean
        ...
        >>> model = Model()
        >>> i = model.integer(3)
        >>> m = mean(i)
        >>> with model.lock():
        ...     model.states.resize(1)
        ...     i.set_state(0, [8, 4, 3])
        ...     print(m.state(0))
        5.0

    See Also:
        :class:`~dwave.optimization.symbols.Mean`: Generated symbol

        :func:`~numpy.mean`: :doc:`NumPy <numpy:index>` function

    .. versionadded:: 0.6.4
    """
    return Mean(array)


@_binaryop(Minimum, NaryMinimum)
def minimum(
    x1: ArraySymbolLike,
    x2: ArraySymbolLike,
    *xi: ArraySymbolLike,
) -> Minimum | NaryMinimum:
    r"""Return an element-wise minimum of two or more symbols and arrays.

    Args:
        x1, x2: Operand array symbol or |array-like|_.
        *xi: Additional array symbols or |array-like|_.

    Returns:
        Successor symbol that is the element-wise minimum of the given symbols
        (or arrays). For two symbols, generates a
        :class:`~dwave.optimization.symbols.Minimum` symbol; for three or more
        symbols, generates a :class:`~dwave.optimization.symbols.NaryMinimum`
        symbol.

    Examples:
        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import minimum

        Minimum between two symbols.

        >>> model = Model()
        >>> i = model.integer(2)
        >>> j = model.integer(2)
        >>> m = minimum(i, j)
        >>> with model.lock():
        ...     model.states.resize(1)
        ...     i.set_state(0, [3, 5])
        ...     j.set_state(0, [7, 2])
        ...     print(m.state(0))
        [3. 2.]

        Minimum between a symbol and array.

        >>> model = Model()
        >>> i = model.integer(2)
        >>> m_array = minimum(i, [2, 6])

    See Also:
        :class:`~dwave.optimization.symbols.Minimum`,
        :class:`~dwave.optimization.symbols.NaryMinimum`: Generated symbol

        :data:`~numpy.minimum`: :doc:`NumPy <numpy:index>` function

        :meth:`~dwave.optimization.model.ArraySymbol.min` method
    """
    raise RuntimeError("implemented by the _binaryop() decorator")


@_binaryop(Modulus)
def mod(x1: ArraySymbolLike, x2: ArraySymbolLike) -> Modulus:
    r"""Calculate element-wise the modulus of two symbols (or arrays).

    Equivalently, you can use the ``%`` operator (e.g., :code:`i % j`).

    Args:
        x1: Dividend array symbol or |array-like|_.
        x2: Divisor array symbol or |array-like|_.

    Returns:
        Successor symbol that is the remainder of the predecessor symbols (or
        arrays) element-wise.

    Examples:
        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import mod

        Calculate the modulus of two integer symbols, :math:`i \mod{j}`, with
        different combinations of positive and negative values.

        >>> model = Model()
        >>> i = model.integer(4, lower_bound=-5)
        >>> j = model.integer(4, lower_bound=-3)
        >>> k = mod(i, j) # alternatively: k = i % j
        >>> with model.lock():
        ...     model.states.resize(1)
        ...     i.set_state(0, [5, -5, 5, -5])
        ...     j.set_state(0, [3, 3, -3, -3])
        ...     print(k.state(0))
        [ 2.  1. -1. -2.]

        Calculate the modulus of a scalar float value and a binary symbol.

        >>> model = Model()
        >>> j = model.binary(2)
        >>> k = 0.33 % j # equivalently: k = mod(0.33, j)
        >>> with model.lock():
        ...     model.states.resize(1)
        ...     j.set_state(0, [0, 1])
        ...     print(k.state(0))
        [0.   0.33]

    See Also:
        See Also:
        :class:`~dwave.optimization.symbols.Modulus`: Generated symbol

        :data:`~numpy.mod`: :doc:`NumPy <numpy:index>` function

        :func:`.add`, :func:`.divide`, :func:`.multiply`,
        :func:`.safe_divide`, :func:`.subtract`
    """
    raise RuntimeError("implemented by the _binaryop() decorator")


@_binaryop(Multiply, NaryMultiply)
def multiply(
    x1: ArraySymbolLike,
    x2: ArraySymbolLike,
    *xi: ArraySymbolLike,
) -> Multiply | NaryMultiply:
    r"""Multiply element-wise two or more symbols and arrays.

    Equivalently, you can use the ``*`` operator (e.g., :code:`i * j`).

    Args:
        x1, x2: Operand array symbol or |array-like|_ to be multiplied.
        *xi: Additional array symbol or |array-like|_ to be multiplied.

    Returns:
        Successor symbol that multiplies the predecessor symbols (and/or arrays)
        element-wise. For two symbols, generates a
        :class:`~dwave.optimization.symbols.Multiply` symbol; for three or more
        symbols, generates a :class:`~dwave.optimization.symbols.NaryMultiply`
        symbol.

    Examples:
        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import multiply

        Multiply two symbols.

        >>> model = Model()
        >>> i = model.integer(2)
        >>> j = model.integer(2)
        >>> k = multiply(i, j)   # alternatively: k = i * j
        >>> with model.lock():
        ...     model.states.resize(1)
        ...     i.set_state(0, [3, 5])
        ...     j.set_state(0, [7, 2])
        ...     print(k.state(0))
        [21. 10.]

        Multiply a symbol and array.

        >>> model = Model()
        >>> i = model.integer(2)
        >>> k_array = i * [8, 4] # equivalently: k_array = multiply(i, [8, 4])

    See Also:
        See Also:
        :class:`~dwave.optimization.symbols.Multiply`,
        :class:`~dwave.optimization.symbols.NaryMultiply`: Generated symbol

        :data:`~numpy.multiply`: :doc:`NumPy <numpy:index>` function

        :func:`.add`, :func:`.divide`, :func:`.mod`, :func:`.safe_divide`,
        :func:`.subtract`
    """
    raise RuntimeError("implemented by the _binaryop() decorator")


def put(array: ArraySymbol, indices: ArraySymbol, values: ArraySymbol) -> Put:
    r"""Replace the specified elements of an array symbol with specified values.

    This function is mostly equivalent to the following function defined for
    :doc:`NumPy <numpy:index>` arrays.

    .. code-block:: python

        def put(array, indices, values):
            array = array.copy()
            array.flat[indices] = values
            return array

    Args:
        array: Array symbol to modify. Must be not be a
            :ref:`dynamically sized <optimization_philosophy_tensor_programming_dynamic>`
            array symbol or a scalar.
        indices:
            Indices in ``array``, flattened, of values to be replaced.

            .. warning::
                Ensure that ``indices`` does not contain duplicate values.
                For duplicate values, which of the possible corresponding values
                of ``values`` is to be propagated is undefined, and performance
                of the model is likely to be degraded.

        values: Values to place in ``array`` for the elements specified by
            indices of ``indices``.

    Examples:
        >>> import numpy as np
        >>> from dwave.optimization import Model
        >>> from dwave.optimization import put
        ...
        >>> model = Model()
        ...
        >>> array = model.constant(np.zeros((3, 3)))
        >>> indices = model.constant([0, 1, 2])
        >>> values = model.integer(3)
        >>> array = put(array, indices, values)  # replace values in array
        ...
        >>> model.states.resize(1)
        >>> with model.lock():
        ...     values.set_state(0, [10, 20, 30])
        ...     print(array.state(0))
        [[10. 20. 30.]
         [ 0.  0.  0.]
         [ 0.  0.  0.]]

    See Also:
        :class:`~dwave.optimization.symbols.Put`: Generated symbol

        :func:`~numpy.put`: :doc:`NumPy <numpy:index>` function

        :func:`.extract`, :func:`.isin`, :func:`.where`

    .. versionadded:: 0.4.4
    """
    return Put(array, indices, values)


def resize(
        array: ArraySymbol,
        shape: typing.Union[int, collections.abc.Sequence[int]],
        fill_value: None | float = None,
) -> Resize:
    """Resize a symbol to a specified shape.

    Args:
        array: Array symbol to be resized.
        shape: Shape of the new array. All dimension sizes must be non-negative.
        fill_value: Value to be used if the successor array is larger than
            its predecessor. Defaults to 0.

    Returns:
        Successor symbol with the specified shape.

    Examples:
        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import resize
        ...
        >>> model = Model()
        >>> s = model.set(10)  # subsets of range(10)
        >>> s_2x2 = resize(s, (2, 2), fill_value=-1)
        ...
        >>> model.states.resize(1)
        >>> with model.lock():
        ...     s.set_state(0, [0, 1, 2])
        ...     print(s_2x2.state(0))
        [[ 0.  1.]
         [ 2. -1.]]

    See also:
        :class:`~dwave.optimization.symbols.Resize`: Generated symbol

        :func:`~numpy.resize`: :doc:`NumPy <numpy:index>` function

        :meth:`~dwave.optimization.model.ArraySymbol.resize` method

        :func:`.broadcast_to`,
        :meth:`~dwave.optimization.model.ArraySymbol.copy`,
        :meth:`~dwave.optimization.model.ArraySymbol.reshape`,
        :func:`.roll`,
        :func:`.transpose`

    .. versionadded:: 0.6.4
    """
    return Resize(array, shape, fill_value=fill_value)


def rint(x: ArraySymbol) -> Rint:
    """Round element-wise the values of an array symbol.

    Rounds the value of every element to its nearest integer.

    Args:
        x: Array symbol.

    Returns:
        Successor symbol with values of its predecessor symbol rounded to the
        nearest integer.

    Examples:
        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import rint
        ...
        >>> model = Model()
        >>> x = model.constant(1.3)
        >>> rint_x = rint(x)
        >>> model.states.resize(1)
        >>> with model.lock():
        ...     print(rint_x.state())
        1.0

    See Also:
        :class:`~dwave.optimization.symbols.Rint`: Generated symbol
    """
    return Rint(x)


def roll(
    array: ArraySymbol,
    shift: ArraySymbol | tuple[int, ...] | int,
    axis: None | tuple[int, ...] | int = None,
) -> Roll:
    """Roll an array symbol's elements along an axis.

    Args:
        array: Array symbol.
        shift: Number of places to shift the array elements. If ``axis`` is
            specified, ``shift`` must be a single number that applies to all
            axes or the same length as ``axis``.
        axis: Axis or axes to be shifted. If not specified, the array is treated
            as flat while shifting.

    Returns:
        Successor symbol with the values of its predecessor shifted.

    Examples:
        >>> from dwave.optimization import Model, roll
        ...
        >>> model = Model()
        >>> x = model.constant(range(10))
        >>> r0 = roll(x, 2)
        >>> r1 = roll(x, -2)
        >>> r2 = roll(x.reshape(2, 5), shift=1, axis=0)
        >>> r3 = roll(x.reshape(2, 5), shift=[1, 2], axis=[1, 0])
        ...
        >>> model.states.resize(1)
        >>> with model.lock():
        ...     print(r0.state())
        ...     print(r1.state())
        ...     print(r2.state())
        ...     print(r3.state())
        [8. 9. 0. 1. 2. 3. 4. 5. 6. 7.]
        [2. 3. 4. 5. 6. 7. 8. 9. 0. 1.]
        [[5. 6. 7. 8. 9.]
         [0. 1. 2. 3. 4.]]
        [[4. 0. 1. 2. 3.]
         [9. 5. 6. 7. 8.]]

    See Also:
        :class:`~dwave.optimization.symbols.Roll`: Generated symbol

        :func:`~numpy.roll`: :doc:`NumPy <numpy:index>` function

        :func:`.broadcast_to`,
        :meth:`~dwave.optimization.model.ArraySymbol.copy`,
        :meth:`~dwave.optimization.model.ArraySymbol.reshape`,
        :func:`.resize`,
        :func:`.transpose`

    .. versionadded:: 0.6.9
    """
    return Roll(array, shift=shift, axis=axis)


@_binaryop(SafeDivide)
def safe_divide(x1: ArraySymbolLike, x2: ArraySymbolLike) -> SafeDivide:
    r"""Divide element-wise two symbols but return 0 where the denominator
    is 0.

    This function is not strictly mathematical division. Rather it encodes
    the following function:

    .. math::
        f(a, b) = \begin{cases}
            a / b & \text{for } b \neq 0 \\
            0 & \text{for } b = 0
        \end{cases}

    Such a definition is useful [#buzzard]_ in cases where ``x2`` is non-zero by
    construction, or otherwise enforced to be non-zero, because you can create a
    model with division using a symbol that might otherwise have zero values.

    .. [#buzzard] Buzzard, Kevin (5 Jul 2020),
       `"Division by zero in type theory: a FAQ" <xena_>`_,
       Xena Project (Blog), retrieved 2025-05-20
    .. _xena: https://xenaproject.wordpress.com/2020/07/05/division-by-zero-in-type-theory-a-faq/

    Args:
        x1: Numerator array symbol or |array-like|_.
        x2: Denominator array symbol or |array-like|_.

    Returns:
        Successor symbol that divides the predecessor symbols (or arrays)
        element-wise.

    Examples:
        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import safe_divide
        ...
        >>> model = Model()
        >>> a = model.constant([-1, 0, 1, 2])
        >>> b = model.constant([2, 1, 0, -1])
        >>> x = safe_divide(a, b)
        >>> model.states.resize(1)
        >>> with model.lock():
        ...     print(x.state())
        [-0.5  0.   0.  -2. ]

    See Also:
        :class:`~dwave.optimization.symbols.SafeDivide`: Generated symbol

        :data:`~numpy.divide`: :doc:`NumPy <numpy:index>` function

        :func:`.add`, :func:`.divide`, :func:`.mod`, :func:`.multiply`,
        :func:`.subtract`

    .. versionadded:: 0.6.2
    """
    raise RuntimeError("implemented by the _binaryop() decorator")


def sin(x) -> Sin:
    """Calculate element-wise the trigonometric sine of a symbol.

    Args:
        x: Array giving the angles, in radians.

    Returns:
        Successor symbol that is the trigonometric sine of the values in its
        predecessor symbol.

    Examples:
        >>> import numpy as np
        >>> from dwave.optimization import Model, sin
        ...
        >>> model = Model()
        >>> x = model.constant([0, np.pi / 2])
        >>> y = sin(x)
        ...
        >>> model.states.resize(1)
        >>> with model.lock():
        ...     print(y.state())
        [0. 1.]

    See Also:
        :class:`~dwave.optimization.symbols.Sin`: Generated symbol

        :data:`~numpy.sin`: :doc:`NumPy <numpy:index>` function

        :func:`.cos`, :func:`.tanh`

    .. versionadded:: 0.6.5
    """
    return Sin(x)


def softmax(array: ArraySymbol) -> SoftMax:
    r"""Return softmax of a symbol.

    Given a flattened array :math:`x: [x_1, x_2, ..., x_n]`,
    :math:`\text{softmax}(x)` returns an array :math:`[y_1, y_2, ..., y_n]`
    such that
    :math:`y_i = \frac{\exp(x_i)}{(\exp(x_1) + \exp(x_2) + ... + \exp(x_n))}`.

    Args:
        array: Array symbol.

    Returns:
        Successor symbol where elements are the softmax of the predecessor
        symbol's corresponding elements.

    Example:
        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import softmax
        >>> import numpy as np
        ...
        >>> model = Model()
        >>> i = model.integer(3)
        >>> sm = softmax(i)
        >>> expected = np.array([0.0900305731703, 0.6652409557748, 0.244728471054])
        >>> with model.lock():
        ...     model.states.resize(1)
        ...     i.set_state(0, [1, 3, 2])
        ...     print(np.isclose(sm.state(), expected).all())
        True

    See Also:
        :class:`~dwave.optimization.symbols.SoftMax`: equivalent symbol.

        :func:`~scipy.special.softmax`: :doc:`SciPy <scipy:index>` function

        :func:`.exp`, :func:`.expit`, :func:`.log`

    .. versionadded:: 0.6.5
    """
    return SoftMax(array)


def sqrt(x: ArraySymbol) -> SquareRoot:
    r"""Calculate element-wise the square root of a symbol.

    Args:
        x: Array symbol.

    Returns:
        Successor symbol that is the square root of the values of its
        predecessor symbol.

    Examples:
        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import sqrt
        ...
        >>> model = Model()
        >>> x = model.constant(16)
        >>> sqrt_x = sqrt(x)
        >>> model.states.resize(1)
        >>> with model.lock():
        ...     print(sqrt_x.state())
        4.0

    See Also:
        :class:`~dwave.optimization.symbols.SquareRoot`: Generated symbol

        :class:`~dwave.optimization.symbols.Square`
    """
    return SquareRoot(x)


def stack(arrays: collections.abc.Sequence[ArraySymbol], axis: int = 0) -> ArraySymbol:
    """Stack a sequence of array symbols.

    Args:
        arrays: Sequence of array symbols to join.

    Returns:
        Successor symbol that joins its predecessor array symbols on an axis.

    Examples:
        This example stacks three scalars on axis 0.

        >>> from dwave.optimization import Model, stack
        ...
        >>> model = Model()
        >>> x = [model.constant(1), model.constant(2), model.constant(3)]
        >>> s = stack(x)
        >>> s.shape()
        (3,)
        >>> with model.lock():
        ...     model.states.resize(1)
        ...     print(s.state(0))
        [1. 2. 3.]

        This example stacks three 1D arrays on axis 0 and 1.

        >>> from dwave.optimization import Model, stack
        ...
        >>> model = Model()
        >>> a = model.constant([1,2])
        >>> b = model.constant([3,4])
        >>> c = model.constant([5,6])
        >>> s0 = stack((a,b,c), axis=0)
        >>> s0.shape()
        (3, 2)
        >>> s1 = stack((a,b,c), axis=1)
        >>> s1.shape()
        (2, 3)
        >>> with model.lock():
        ...     model.states.resize(1)
        ...     print(s0.state(0))
        ...     print(s1.state(0))
        [[1. 2.]
         [3. 4.]
         [5. 6.]]
        [[1. 3. 5.]
         [2. 4. 6.]]

    See Also:
        :class:`~dwave.optimization.symbols.Concatenate`: Generated symbol

        :func:`~numpy.stack`: :doc:`NumPy <numpy:index>` function

        :func:`~dwave.optimization.mathematical.concatenate`,
        :func:`~dwave.optimization.mathematical.hstack`,
        :func:`~dwave.optimization.mathematical.vstack`

    .. versionadded:: 0.5.0
    """
    if (not isinstance(arrays, collections.abc.Sequence) or
            not all(isinstance(arr, ArraySymbol) for arr in arrays)):
        raise TypeError("stack() takes a sequence of array symbols of the same shape")

    if len(arrays) == 0:
        raise ValueError("need at least one array symbol to stack")

    shape = arrays[0].shape()

    if not all(arr.shape() == shape for arr in arrays):
        raise ValueError("all input array symbols must have the same shape")

    if not 0 <= axis <= len(shape):
        raise ValueError(f'axis {axis} is out of bounds for array'
                         f' of dimension {len(shape) + 1}')

    new_shape = tuple(shape[:axis]) + (1,) + (shape[axis:])  # add the axis and then concatenate
    return concatenate([arr.reshape(new_shape) for arr in arrays], axis)


@_binaryop(Subtract)
def subtract(x1: ArraySymbolLike, x2: ArraySymbolLike) -> Subtract:
    """Subtract element-wise two symbols (or arrays).

    Equivalently, you can use the ``-`` operator (e.g., :code:`i - j`).

    Args:
        x1, x2: Operand array symbol or |array-like|_ to subtract.

    Returns:
        Successor symbol that subtracts the predecessor symbols (or arrays)
        element-wise.

    Examples:
        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import subtract

        Subtract two symbols.

        >>> model = Model()
        >>> i = model.integer(2, lower_bound=3)
        >>> j = model.integer(2, upper_bound=20)
        >>> k = subtract(i, j)   # alternatively: k = i - j
        >>> with model.lock():
        ...    model.states.resize(1)
        ...    i.set_state(0, [21, 10])
        ...    j.set_state(0, [7, 2])
        ...    print(k.state(0))
        [14.  8.]

        Subtract an array from a symbol.

        >>> model = Model()
        >>> i = model.integer(2, lower_bound=3)
        >>> l = i - [2, 2] # equivalently: l = subtract(i, [2, 2])

    See Also:
        :class:`~dwave.optimization.symbols.Subtract`: Generated symbol

        :data:`~numpy.subtract`: :doc:`NumPy <numpy:index>` function

        :func:`.add`, :func:`.divide`, :func:`.mod`, :func:`.multiply`,
        :func:`.safe_divide`
    """
    raise RuntimeError("implemented by the _binaryop() decorator")


def tanh(x) -> Tanh:
    """Calculate element-wise the trigonometric hyperbolic tangent of a symbol.

    Args:
        x: Array giving the angles, in radians.

    Returns:
        Successor symbol that is the trigonometric hyperbolic tangent of the
        values in its predecessor symbol.

    Examples:
        >>> import numpy as np
        >>> from dwave.optimization import Model, tanh
        ...
        >>> model = Model()
        >>> x = model.constant([0, np.pi / 2])
        >>> y = tanh(x)
        ...
        >>> model.states.resize(1)
        >>> with model.lock():
        ...     print(y.state())
        [0.         0.91715234]

    See Also:
        :class:`~dwave.optimization.symbols.Tanh`: equivalent symbol.

        :data:`~numpy.tanh`: :doc:`NumPy <numpy:index>` function

        :func:`.cos`, :func:`.sin`

    .. versionadded:: 0.6.11
    """
    return Tanh(x)


def transpose(array: ArraySymbol) -> Transpose:
    r"""Transpose an array symbol.

    Args:
        array: Array symbol to transpose. For a
            :ref:`dynamically sized <optimization_philosophy_tensor_programming_dynamic>`
            array, must have dimension at most one.

    Returns:
        Successor symbol that is the transpose of its predecessor symbol. For a
        one-dimensional array, returns an unchanged view of the predecessor
        array. For a 2D array, returns the standard matrix transpose. For an
        :math:`n`-dimensional array, the transpose simply reverses the order of
        the axes.

    Examples:
        This example transposes a 5-element vector.

        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import transpose
        ...
        >>> model = Model()
        >>> array = model.constant([0, 1, 2, 3, 4])
        >>> transpose = transpose(array)
        >>> model.states.resize(1)
        >>> with model.lock():
        ...     print(transpose.state())
        [0. 1. 2. 3. 4.]


        This example transposes a :math:`2 \times 3` matrix.

        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import transpose
        ...
        >>> model = Model()
        >>> array = model.constant([[0, 1, 2], [3, 4, 5]])
        >>> transpose = transpose(array)
        >>> model.states.resize(1)
        >>> with model.lock():
        ...     print(transpose.state())
        [[0. 3.]
         [1. 4.]
         [2. 5.]]


        This example transposes a :math:`2 \times 3 \times 2` matrix.

        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import transpose
        ...
        >>> model = Model()
        >>> array = model.constant([[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]])
        >>> transpose = transpose(array)
        >>> model.states.resize(1)
        >>> with model.lock():
        ...     print(transpose.state())
        [[[ 0.  6.]
          [ 2.  8.]
          [ 4. 10.]]
        <BLANKLINE>
         [[ 1.  7.]
          [ 3.  9.]
          [ 5. 11.]]]

    See Also:
        :class:`~dwave.optimization.symbols.Transpose`: equivalent symbol.

        :func:`~numpy.transpose`: :doc:`NumPy <numpy:index>` function

        :func:`.broadcast_to`,
        :meth:`~dwave.optimization.model.ArraySymbol.copy`,
        :meth:`~dwave.optimization.model.ArraySymbol.reshape`
        :meth:`~dwave.optimization.model.ArraySymbol.resize`
        :func:`.roll`

    .. versionadded:: 0.6.8
    """
    return Transpose(array)


def vstack(arrays: collections.abc.Sequence[ArraySymbol]) -> ArraySymbol:
    """Vertically stack a sequence of array symbols.

    Equivalent to concatenation along the first axis.

    Args:
        arrays: Array symbols to be concatenated. Arrays must have the same
            shape along all but the first axis unless they are 0D or 1D arrays.

    Returns:
        Successor array symbol that concatenates its predecessor array symbols.

    Examples:

        >>> from dwave.optimization import vstack, Model

        >>> model = Model()
        >>> model.states.resize(1)
        ...
        >>> a = model.constant([0])
        >>> b = model.constant([1])
        >>> h = stack((a, b))
        >>> with model.lock():
        ...     h.state()
        array([[0.],
               [1.]])

        >>> model = Model()
        >>> model.states.resize(1)
        ...
        >>> a = model.constant([0, 1, 2])
        >>> b = model.constant([[3, 4, 5], [6, 7, 8]])
        >>> h = vstack((a, b))
        >>> with model.lock():
        ...     h.state()
        array([[0., 1., 2.],
               [3., 4., 5.],
               [6., 7., 8.]])

    See Also:
        :class:`~dwave.optimization.symbols.Concatenate`: Generated symbol

        :func:`~numpy.vstack`: :doc:`NumPy <numpy:index>` function

        :func:`~dwave.optimization.mathematical.concatenate`,
        :func:`~dwave.optimization.mathematical.hstack`,
        :func:`~dwave.optimization.mathematical.stack`

    .. versionadded:: 0.6.0
    """
    arrays = atleast_2d(*arrays)
    if not isinstance(arrays, tuple):
        arrays = (arrays,)
    return concatenate(arrays, 0)


def where(condition: ArraySymbol, x: ArraySymbol, y: ArraySymbol) -> Where:
    """Select elements from either of two array symbols.

    Args:
        condition:
            Condition that where true selects elements from ``x`` and where
            false from ``y``. If ``x`` and ``y`` are
            :ref:`dynamically sized <optimization_philosophy_tensor_programming_dynamic>`,
            ``condition`` must be a single value.
        x, y:
            Array symbol from which to choose values.

    Returns:
        Successor symbol with element values from predecessor symbol ``x`` where
        ``condition`` is true and from predecessor symbol ``y`` where false.

    Examples:
        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import where

        This example uses a binary array to to select between two arrays.

        >>> model = Model()
        >>> condition = model.binary(3)
        >>> x = model.constant([1., 2., 3.])
        >>> y = model.constant([4., 5., 6.])
        >>> a = where(condition, x, y)
        >>> with model.lock():
        ...     model.states.resize(1)
        ...     condition.set_state(0, [True, True, False])
        ...     print(a.state())
        [1. 2. 6.]

        This example uses a single binary variable to choose between two sets.

        >>> model = Model()
        >>> condition = model.binary()
        >>> x = model.set(10)  # any subset of range(10)
        >>> y = model.set(10)  # any subset of range(10)
        >>> a = where(condition, x, y)
        >>> with model.lock():
        ...     model.states.resize(1)
        ...     condition.set_state(0, True)
        ...     x.set_state(0, [0., 2., 3.])
        ...     y.set_state(0, [1])
        ...     print(a.state())
        [0. 2. 3.]

    See Also:
        :class:`~dwave.optimization.symbols.Where`: Generated symbol

        :func:`~numpy.where`: :doc:`NumPy <numpy:index>` function

        :func:`.extract`, :func:`.isin`, :func:`.put`
    """
    return Where(condition, x, y)

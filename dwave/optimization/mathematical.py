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

import collections
import functools
import typing

from dwave.optimization._model import ArraySymbol
from dwave.optimization.symbols import (
    Add,
    And,
    ARange,
    Concatenate,
    Divide,
    Expit,
    Logical,
    Maximum,
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
    Rint,
    SquareRoot,
    Where,
    Xor,
)


__all__ = [
    "add",
    "arange",
    "concatenate",
    "divide",
    "expit",
    "logical",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "maximum",
    "minimum",
    "mod",
    "multiply",
    "put",
    "rint",
    "sqrt",
    "stack",
    "where",
]


def _op(BinaryOp: type, NaryOp: type, reduce_method: str):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(x1, *xi, **kwargs):
            if not xi:
                raise TypeError(
                        f"Cannot call {f.__name__} on a single node; did you mean to "
                        f"call `x.{reduce_method}()` ?"
                    )
            elif len(xi) == 1:
                return BinaryOp(x1, xi[0], **kwargs)
            else:
                return NaryOp(x1, *xi, **kwargs)
        return wrapper
    return decorator


@_op(Add, NaryAdd, "sum")
def add(x1: ArraySymbol, x2: ArraySymbol, *xi: ArraySymbol) -> typing.Union[Add, NaryAdd]:
    r"""Return an element-wise addition on the given symbols.

    In the underlying directed acyclic expression graph, produces an ``Add``
    node if two array nodes are provided and a ``NaryAdd`` node otherwise.

    Args:
        x1, x2: Input array symbol to be added.
        *xi: Additional input array symbols to be added.

    Returns:
        A symbol that sums the given symbols element-wise.
        Adding two symbols returns a :class:`~dwave.optimization.symbols.Add`.
        Adding three or more symbols returns a
        :class:`~dwave.optimization.symbols.NaryAdd`.

    Examples:
        This example adds two integer symbols of size :math:`1 \times 2`.
        Equivalently, you can use the ``+`` operator (e.g., :code:`i + j`).

        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import add
        ...
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
    """
    raise RuntimeError("implemented by the op() decorator")


def arange(start: typing.Union[int, ArraySymbol, None] = None,
           stop: typing.Union[int, ArraySymbol, None] = None,
           step: typing.Union[int, ArraySymbol, None] = None,
           ) -> ArraySymbol:
    """Return an array symbol with evenly spaced values within a given interval.

    Args:
        start: Start of the interval. Unless only one argument is provided in
            which it is interpreted as the ``stop``.
        stop: End of the interval.
        step: Spacing between values.

    See Also:
        :class:`~dwave.optimization.ARange`: equivalent symbol.

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


def concatenate(array_likes: typing.Union[collections.abc.Iterable, ArraySymbol],
                axis: int = 0,
                ) -> ArraySymbol:
    r"""Return the concatenation of one or more symbols on the given axis.

    Args:
        array_like: Array symbols to concatenate.
        axis: The concatenation axis.

    Returns:
        A symbol that is the concatenation of the given symbols along the specified axis.

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
        >>> type(a_b)
        <class 'dwave.optimization.symbols.Concatenate'>
        >>> with model.lock():
        ...     model.states.resize(1)
        ...     print(a_b.state(0))
        [[0. 1.]
         [2. 3.]
         [4. 5.]]
    """
    if isinstance(array_likes, ArraySymbol):
        return array_likes

    if (isinstance(array_likes, collections.abc.Iterable)
            and isinstance(array_likes, collections.abc.Sized)
            and (0 < len(array_likes))):

        if len(array_likes) == 1:
            return array_likes[0]

        return Concatenate(tuple(array_likes), axis)

    raise TypeError("concatenate takes one or more ArraySymbol as input")


def divide(x1: ArraySymbol, x2: ArraySymbol) -> Divide:
    r"""Return an element-wise division on the given symbols.

    In the underlying directed acyclic expression graph, produces a
    ``Divide`` node if two array nodes are provided.

    Args:
        x1, x2: Input array symbol.

    Returns:
        A symbol that divides the given symbols element-wise.
        Dividing two symbols returns a
        :class:`~dwave.optimization.symbols.Divide`.

    Examples:
        This example divides two integer symbols.
        Equivalently, you can use the ``/`` operator (e.g., :code:`i / j`).

        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import divide
        ...
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
    """
    return Divide(x1, x2)

def expit(x: ArraySymbol) -> Expit:
    """Return an element-wise logistic sigmoid on the given symbol.

    Args:
        x: Input symbol.

    Returns:
        A symbol that propagates the values of the logistic sigmoid of a given symbol.

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
        :class:`~dwave.optimization.symbols.Expit`: equivalent symbol.

    .. versionadded:: 0.5.2
    """
    return Expit(x)


def logical(x: ArraySymbol) -> Logical:
    r"""Return the element-wise truth value on the given symbol.

    Args:
        x: Input array symbol.

    Returns:
        A symbol that propagates the element-wise truth value of the given symbol.

    Examples:
        This example shows the truth values an array symbol.

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
        :class:`~dwave.optimization.symbols.Logical`: equivalent symbol.
    """
    return Logical(x)


def logical_and(x1: ArraySymbol, x2: ArraySymbol) -> And:
    r"""Return an element-wise logical AND on the given symbols.

    Args:
        x1, x2: Input array symbol.

    Returns:
        A symbol that is the element-wise AND of the given symbols.

    Examples:
        This example ANDs two binary symbols of size :math:`1 \times 3`.

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

    See Also:
        :class:`~dwave.optimization.symbols.And`: equivalent symbol.
    """
    return And(x1, x2)


def logical_not(x: ArraySymbol) -> Not:
    r"""Return an element-wise logical NOT on the given symbol.

    Args:
        x: Input array symbol.

    Returns:
        A symbol that propagates the element-wise NOT of the given symbol.

    Examples:
        This example negates an array symbol.

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
        :class:`~dwave.optimization.symbols.Not`: equivalent symbol.
    """
    return Not(x)


def logical_or(x1: ArraySymbol, x2: ArraySymbol) -> Or:
    r"""Return an element-wise logical OR on the given symbols.

    Args:
        x1, x2: Input array symbol.

    Returns:
        A symbol that is the element-wise OR of the given symbols.

    Examples:
        This example ORs two binary symbols of size :math:`1 \times 3`.

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

    See Also:
        :class:`~dwave.optimization.symbols.Or`: equivalent symbol.
    """
    return Or(x1, x2)


def logical_xor(x1: ArraySymbol, x2: ArraySymbol) -> Xor:
    r"""Return an element-wise logical XOR on the given symbols.

    Args:
        x1, x2: Input array symbol.

    Returns:
        A symbol that is the element-wise XOR of the given symbols.

    Examples:
        This example XORs two binary symbols of size :math:`1 \times 3`.

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

    See Also:
        :class:`~dwave.optimization.symbols.Xor`: equivalent symbol.

    .. versionadded:: 0.4.1
    """
    return Xor(x1, x2)


@_op(Maximum, NaryMaximum, "max")
def maximum(x1: ArraySymbol, x2: ArraySymbol, *xi: ArraySymbol,
            ) -> typing.Union[Maximum, NaryMaximum]:
    r"""Return an element-wise maximum of the given symbols.

    In the underlying directed acyclic expression graph, produces a
    ``Maximum`` node if two array nodes are provided and a
    ``NaryMaximum`` node otherwise.

    Args:
        x1, x2: Input array symbol.
        *xi: Additional input array symbols.

    Returns:
        A symbol that is the element-wise maximum of the given symbols.
        Taking the element-wise maximum of two symbols returns a
        :class:`~dwave.optimization.symbols.Maximum`.
        Taking the element-wise maximum of three or more symbols returns a
        :class:`~dwave.optimization.symbols.NaryMaximum`.

    Examples:
        This example maximizes two integer symbols of size :math:`1 \times 2`.

        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import maximum
        ...
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
    """
    raise RuntimeError("implemented by the op() decorator")


@_op(Minimum, NaryMinimum, "min")
def minimum(x1: ArraySymbol, x2: ArraySymbol, *xi: ArraySymbol,
            ) -> typing.Union[Minimum, NaryMinimum]:
    r"""Return an element-wise minimum of the given symbols.

    In the underlying directed acyclic expression graph, produces a
    ``Minimum`` node if two array nodes are provided and a
    ``NaryMinimum`` node otherwise.

    Args:
        x1, x2: Input array symbol.
        *xi: Additional input array symbols.

    Returns:
        A symbol that is the element-wise minimum of the given symbols.
        Taking the element-wise minimum of two symbols returns a
        :class:`~dwave.optimization.symbols.Minimum`.
        Taking the element-wise minimum of three or more symbols returns a
        :class:`~dwave.optimization.symbols.NaryMinimum`.

    Examples:
        This example minimizes two integer symbols of size :math:`1 \times 2`.

        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import minimum
        ...
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
    """
    raise RuntimeError("implemented by the op() decorator")


def mod(x1: ArraySymbol, x2: ArraySymbol) -> Modulus:
    r"""Return an element-wise modulus of the given symbols.

    Args:
        x1, x2: Input array symbol.

    Returns:
        A symbol that is the element-wise modulus of the given symbols.

    Examples:
        This example demonstrates the behavior of the modulus of two integer
        symbols :math:`i \mod{j}` with different combinations of positive and
        negative values. Equivalently, you can use the ``%`` operator
        (e.g., :code:`i % j`).

        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import mod
        ...
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

        This example demonstrates the modulus of a scalar float value and a
        binary symbol.

        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import mod
        ...
        >>> model = Model()
        >>> i = model.constant(0.33)
        >>> j = model.binary(2)
        >>> k = mod(i, j) # alternatively: k = i % j
        >>> with model.lock():
        ...     model.states.resize(1)
        ...     j.set_state(0, [0, 1])
        ...     print(k.state(0))
        [0.   0.33]
    """
    return Modulus(x1, x2)


@_op(Multiply, NaryMultiply, "multiply")
def multiply(x1: ArraySymbol, x2: ArraySymbol, *xi: ArraySymbol,
             ) -> typing.Union[Multiply, NaryMultiply]:
    r"""Return an element-wise multiplication on the given symbols.

    In the underlying directed acyclic expression graph, produces a
    ``Multiply`` node if two array nodes are provided and a
    ``NaryMultiply`` node otherwise.

    Args:
        x1, x2: Input array symbol.
        *xi: Additional input array symbols.

    Returns:
        A symbol that multiplies the given symbols element-wise.
        Multipying two symbols returns a
        :class:`~dwave.optimization.symbols.Minimum`.
        Multiplying three or more symbols returns a
        :class:`~dwave.optimization.symbols.NaryMinimum`.

    Examples:
        This example multiplies two integer symbols of size :math:`1 \times 2`.
        Equivalently, you can use the ``*`` operator (e.g., :code:`i * j`).

        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import multiply
        ...
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
    """
    raise RuntimeError("implemented by the op() decorator")


def put(array: ArraySymbol, indices: ArraySymbol, values: ArraySymbol) -> Put:
    r"""Replace the specified elements in an array with given values.

    This function is roughly equivalent to the following function defined for
    NumPy arrays.

    .. code-block:: python

        def put(array, indices, values):
            array = array.copy()
            array.flat[indices] = values
            return array

    Args:
        array: Base array. Must be not be a dynamic array nor a scalar.
        indices:
            The indices in the flattened base array to be replaced.

            .. warning::
                If ``indices`` has duplicate values, it is undefined which
                of the possible corresponding values from ``values`` will
                be propagated.
                This will likely hurt the performance of the model.
                Care should be taken to ensure that ``indices`` does not
                contain duplicates.

        values: Values to place in ``array`` at ``indices``.

    Examples:
        For some models, it is useful to overwrite some elements in an array.

        >>> import numpy as np
        >>> from dwave.optimization import Model
        >>> from dwave.optimization import put
        ...
        >>> model = Model()
        ...
        >>> array = model.constant(np.zeros((3, 3)))
        >>> indices = model.constant([0, 1, 2])
        >>> values = model.integer(3)
        >>> array = put(array, indices, values)  # replace array with one that has been overwritten
        ...
        >>> model.states.resize(1)
        >>> with model.lock():
        ...     values.set_state(0, [10, 20, 30])
        ...     print(array.state(0))
        [[10. 20. 30.]
         [ 0.  0.  0.]
         [ 0.  0.  0.]]

    See Also:
        :class:`~dwave.optimization.symbols.Put`: equivalent symbol.

        :func:`numpy.put`: The NumPy function that this function emulates for
        :class:`~dwave.optimization.model.ArraySymbol`\s.

    .. versionadded:: 0.4.4
    """
    return Put(array, indices, values)


def rint(x: ArraySymbol) -> Rint:
    """Return an element-wise round to the nearest integer on the given symbol.

    Args:
        x: Input symbol.

    Returns:
        A symbol that propagates the values of the given symbol rounded to the
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
        :class:`~dwave.optimization.symbols.Rint`: equivalent symbol.
    """
    return Rint(x)


def sqrt(x: ArraySymbol) -> SquareRoot:
    r"""Return an element-wise sqrt on the given symbol.
    Args:
        x: Input symbol.
    Returns:
        A symbol that propagates the sqrt of the given symbol.
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
        :class:`~dwave.optimization.symbols.SquareRoot`: equivalent symbol.
    """
    return SquareRoot(x)


def stack(arrays: collections.abc.Iterable[ArraySymbol], axis: int = 0) -> ArraySymbol:
    r"""Joins a sequence of ArraySymbols along a new axis.

    Args:
        arrays: sequence of ArraySymbol
    Returns:
        The joined ArraySymbols on a new axis
    Examples:
        This example stacks three scalars on the first axis.

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

        This example stacks three 1d arrays on axis 0 and 1

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
    """
    if all(isinstance(arr, ArraySymbol) for arr in arrays):
        if not 0 <= axis <= arrays[0].ndim():
            raise ValueError(f'axis {axis} is out of bounds for array'
                             f' of dimension {arrays[0].ndim() + 1}')

        make_shape = lambda x: (x.shape()[:axis]) + (1, ) + (x.shape()[axis:])
        return concatenate([arr.reshape(make_shape(arr)) for arr in arrays], axis)

    raise ValueError("stack takes an Iterable of ArraySymbols of same shape")


def where(condition: ArraySymbol, x: ArraySymbol, y: ArraySymbol) -> Where:
    """Return elements chosen from x or y depending on condition.

    Args:
        condition:
            Where ``True``, yield ``x``, otherwise yield ``y``.
        x, y:
            Values from which to choose.
            If ``x`` and ``y`` are dynamically sized then ``condition``
            must be a single value.

    Returns:
        An :class:`~dwave.optimization.model.ArraySymbol`
        with elements from ``x`` where ``condition`` is ``True``,
        and elements from ``y`` elsewhere.

    Examples:

        This example uses a single binary variable to choose between two arrays.

        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import where
        ...
        >>> model = Model()
        >>> condition = model.binary()
        >>> x = model.constant([1., 2., 3.])
        >>> y = model.constant([4., 5., 6.])
        >>> a = where(condition, x, y)
        >>> with model.lock():
        ...     model.states.resize(1)
        ...     condition.set_state(0, False)
        ...     print(a.state())
        [4. 5. 6.]

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

    """
    return Where(condition, x, y)

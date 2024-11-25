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

from dwave.optimization.model import ArraySymbol
from dwave.optimization.symbols import (
    Add,
    And,
    Concatenate,
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
    SquareRoot,
    Where,
    Xor,
)


__all__ = [
    "add",
    "concatenate",
    "logical",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "maximum",
    "minimum",
    "mod",
    "multiply",
    "sqrt",
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
    raise RuntimeError("implementated by the op() decorator")


def concatenate(array_likes : typing.Union[collections.abc.Iterable, ArraySymbol], axis : int = 0) -> ArraySymbol:
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

    if isinstance(array_likes, collections.abc.Sequence) and (0 < len(array_likes)):
        if isinstance(array_likes[0], ArraySymbol):
            if len(array_likes) == 1:
                return array_likes[0]

            return Concatenate(tuple(array_likes), axis)

    raise TypeError("concatenate takes one or more ArraySymbol as input")


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
    raise RuntimeError("implementated by the op() decorator")


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
    raise RuntimeError("implementated by the op() decorator")


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
    raise RuntimeError("implementated by the op() decorator")


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
        4
    See Also:
        :class:`~dwave.optimization.symbols.SquareRoot`: equivalent symbol.
    """
    return SquareRoot(x)


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

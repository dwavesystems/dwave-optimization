# Copyright 2025 D-Wave
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

import collections.abc
import functools
import inspect
import typing

from dwave.optimization.model import Model


__all__ = ["expression"]


class Expression:
    """An expression that can be used as an input to other symbols.

    Instantiated through the :func:`.expression` function.

    Examples:
        This example creates expression :math:`v_0 + 2*(v_1 + 1)` and uses it
        for an :class:`.AccumulateZip` symbol that calculates the cumulative
        value of sequential elements of an
        :class:`~dwave.optimization.symbols.IntegerVariable` symbol,
        :math:`v_1`, where :math:`v_0` is the result from the previous elements.

        For example, for input
        :math:`v_1 = [1, 1, 0, 1]` and an initial value, :math:`v_0[0]` of 2,
        the first element of the :class:`.AccumulateZip` symbol is
        :math:`2 + 2*(1 + 1) = 6`, the 2nd is therefore
        :math:`6 + 2*(1 + 1) = 10`, the 3rd is :math:`10 + 2*(0 + 1) = 12`, etc.

        >>> from dwave.optimization import expression, Model
        >>> from dwave.optimization.symbols import AccumulateZip
        ...
        >>> model = Model()
        >>> i1 = model.integer(4, lower_bound=0, upper_bound=10)
        ...
        >>> @expression(v1=dict(lower_bound=-10, upper_bound=20))
        ... def func(v0, v1):
        ...     return 2*(v1 + 1) + v0
        ...
        >>> j = AccumulateZip(func, (i1, ), initial=2)
        ...
        >>> with model.lock():
        ...     model.states.resize(1)
        ...     i1.set_state(0,[1, 1, 0, 1])
        ...     print(j.state(0))
        [ 6. 10. 12. 16.]

    See Also:
        :func:`.expression`

    .. versionadded:: 0.6.4
    """
    _function: collections.abc.Callable  # todo: better typing?
    _model: Model

    def __init__(self):
        raise ValueError("Expression cannot be constructed directly. "
                         "Use the expression() function/decorator instead.")

    def __str__(self) -> str:
        return f"expression({self._function!r})"


@typing.overload
def expression(function: collections.abc.Callable, **kwargs) -> Expression: ...
@typing.overload
def expression(**kwargs) -> collections.abc.Callable: ...


def expression(*args, **kwargs):
    """Transform a function into an :class:`Expression`.

    Such expressions are used as input by some symbols.

    Args:
        function: Callable function that is executed once to generate the
            :class:`Expression`.

    Examples:
        >>> from dwave.optimization import expression

        >>> @expression
        ... def func(a, b, c):
        ...     return (a + b) * c

        >>> @expression(a=dict(lower_bound=0), b=dict(upper_bound=1))
        ... def func(a, b, c):
        ...     return (a + b) * c

    See Also:
        :class:`.Expression`

    .. versionadded:: 0.6.4
    """
    if len(args) == 0:
        def _decorator(function):
            return expression(function, **kwargs)
        return _decorator
    elif len(args) == 1:
        function, = args
    else:
        raise TypeError(
            f"expression() takes 0 or 1 positional arguments but {len(args)} were given",
        )

    if not callable(function):
        raise TypeError(f'{function!r} is not a callable object')

    model = Model()

    # Create the inputs. By default we loosen the values to -inf/+inf, but we also
    # allow the user to overwrite that default.
    default_kwargs = dict(lower_bound=-float('inf'), upper_bound=float('inf'))
    inputs = []
    for parameter in inspect.signature(function).parameters:
        input_kwargs = default_kwargs | kwargs.get(parameter, dict())
        inputs.append(model.input(**input_kwargs))
    if len(inputs) < 1:
        raise ValueError("function must accept at least one argument")

    # Use function to construct the symbols and/or raise any exceptions
    model.minimize(function(*inputs))
    model.lock()

    # Finally create an Expression object to hold the model and return it to
    # the user.
    expr = Expression.__new__(Expression)
    expr._function = function
    expr._model = model
    return expr

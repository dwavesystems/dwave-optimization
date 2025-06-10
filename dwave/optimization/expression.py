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

import inspect
import typing

from dwave.optimization.model import Model


__all__ = ["expression"]


class Expression:
    _function: typing.Callable  # todo: better typing?
    _model: Model

    def __init__(self):
        raise ValueError("Expression cannot be constructed directly. "
                         "Use the expression() function/decorator instead.")

    def __str__(self) -> str:
        return f"expression({self._function!r})"

    @staticmethod
    def _from_function(function: typing.Callable) -> Expression:
        expr = Expression.__new__(Expression)
        expr._function = function
        expr._model = model = Model()

        # Use function to construct the symbols. Let the relevant methods to raise
        # any relevant exceptions.
        inputs = [model.input() for _ in inspect.signature(function).parameters]
        if len(inputs) < 1:
            raise ValueError("function must accept at least one argument")
        model.minimize(function(*inputs))

        return expr


def expression(function: typing.Callable) -> Expression:
    """Create an :class:`Expression` from a function."""
    return Expression._from_function(function)

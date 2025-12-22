# Copyright 2025 D-Wave Inc.
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

import typing

import numpy.typing

from dwave.optimization._model import ArraySymbol

__all__ = [
    "ArraySymbolLike",
]

ArraySymbolLike: typing.TypeAlias = ArraySymbol | numpy.typing.ArrayLike
"""Either a :class:`~dwave.optimization.ArraySymbol` or a NumPy
`array-like <https://numpy.org/devdocs/glossary.html#term-array_like>`_.
"""

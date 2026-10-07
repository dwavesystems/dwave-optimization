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

import typing

import numpy as np

from dwave.optimization._model import _Graph
from dwave.optimization.model import ArraySymbol as _ArraySymbol
from dwave.optimization.typing import ShapeLike

_SumConstraint: typing.TypeAlias = list[
    tuple[list[str], list[float]] | tuple[int, list[str], list[float]]
]
_AxesSubjetTo: typing.TypeAlias = list[tuple[int, str | list[str], float | list[float]]]

class BinaryVariable(_ArraySymbol):
    def __init__(
        self,
        model: _Graph,
        shape: ShapeLike | None = None,
        lower_bound: np.typing.ArrayLike | None = None,
        upper_bound: np.typing.ArrayLike | None = None,
        sum_subject_to: list[tuple[str, float]] | None = None,
        axes_sums_subject_to: _AxesSubjetTo | None = None,
    ): ...
    def lower_bound(self) -> np.typing.NDArray[np.double]: ...
    def set_state(self, index: int, state: np.typing.ArrayLike) -> None: ...
    def sum_constraints(self) -> _SumConstraint: ...
    def upper_bound(self) -> np.typing.NDArray[np.double]: ...

class IntegerVariable(_ArraySymbol):
    def __init__(
        self,
        model: _Graph,
        shape: ShapeLike | None = None,
        lower_bound: np.typing.ArrayLike | None = None,
        upper_bound: np.typing.ArrayLike | None = None,
        sum_subject_to: list[tuple[str, float]] | None = None,
        axes_sums_subject_to: _AxesSubjetTo | None = None,
    ): ...
    def lower_bound(self) -> np.typing.NDArray[np.double]: ...
    def set_state(self, index: int, state: np.typing.ArrayLike) -> None: ...
    def sum_constraints(self) -> _SumConstraint: ...
    def upper_bound(self) -> np.typing.NDArray[np.double]: ...

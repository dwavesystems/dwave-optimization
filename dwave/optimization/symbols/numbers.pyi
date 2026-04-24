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

import collections.abc
import numpy as np
import typing
from dwave.optimization.model import ArraySymbol as _ArraySymbol
from dwave.optimization._model import _Graph

_ShapeLike: typing.TypeAlias = typing.Union[int, collections.abc.Sequence[int]]
_SumConstraint: typing.TypeAlias = list[typing.Union[tuple[list[str], list[float]], 
                                        tuple[int, list[str], list[float]]]]
_AxesSubjetTo: typing.TypeAlias = list[tuple[int, str | list[str], float | list[float]]]

class BinaryVariable(_ArraySymbol):
    def __init__(self, 
                 model: _Graph,
                 shape: None | _ShapeLike = None, 
                 lower_bound: None | np.typing.ArrayLike = None,
                 upper_bound: None | np.typing.ArrayLike = None,
                 subject_to: None | list[tuple[str, float]] = None,
                 axes_subject_to: None | _AxesSubjetTo = None): ...
    def lower_bound(self) -> np.typing.NDArray[np.double]: ...
    def set_state(self, index: int, state: np.typing.ArrayLike): ...
    def sum_constraints(self) -> _SumConstraint: ...
    def upper_bound(self) -> np.typing.NDArray[np.double]: ...

class IntegerVariable(_ArraySymbol):
    def __init__(self, 
                 model: _Graph,
                 shape: None | _ShapeLike = None, 
                 lower_bound: None | np.typing.ArrayLike = None,
                 upper_bound: None | np.typing.ArrayLike = None,
                 subject_to: None | list[tuple[str, float]] = None,
                 axes_subject_to: None | _AxesSubjetTo = None): ...
    def lower_bound(self) -> np.typing.NDArray[np.double]: ...
    def set_state(self, index: int, state: np.typing.ArrayLike): ...
    def sum_constraints(self) -> _SumConstraint: ...
    def upper_bound(self) -> np.typing.NDArray[np.double]: ...


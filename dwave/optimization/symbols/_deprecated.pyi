# Copyright 2024 D-Wave Systems Inc.
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

from dwave.optimization.model import Symbol, ArraySymbol

__all__: list[str]


class AdvancedIndexing(ArraySymbol):
    ...


class BasicIndexing(ArraySymbol):
    ...


class BinaryVariable(ArraySymbol):
    def set_state(self, index: int, state: numpy.typing.ArrayLike): ...


class BSpline(ArraySymbol):
    ...


class Concatenate(ArraySymbol):
    ...


class Constant(ArraySymbol):
    def __bool__(self) -> bool: ...
    def __index__(self) -> int: ...

    # these methods don't (yet) have an ArraySymbol overload
    def __ge__(self, rhs: numpy.typing.ArrayLike) -> numpy.typing.NDArray[numpy.bool]: ...
    def __gt__(self, rhs: numpy.typing.ArrayLike) -> numpy.typing.NDArray[numpy.bool]: ...
    def __lt__(self, rhs: numpy.typing.ArrayLike) -> numpy.typing.NDArray[numpy.bool]: ...

    @typing.overload
    def __eq__(self, rhs: ArraySymbol) -> Equal: ...
    @typing.overload
    def __eq__(self, rhs: numpy.typing.ArrayLike) -> numpy.typing.NDArray[numpy.bool]: ...
    @typing.overload
    def __le__(self, rhs: ArraySymbol) -> LessEqual: ...
    @typing.overload
    def __le__(self, rhs: numpy.typing.ArrayLike) -> numpy.typing.NDArray[numpy.bool]: ...


class Copy(ArraySymbol):
    ...


class Extract(ArraySymbol):
    ...


class Input(ArraySymbol):
    def integral(self) -> bool: ...
    def lower_bound(self) -> float: ...
    def set_state(self, index: int, state: numpy.typing.ArrayLike): ...
    def upper_bound(self) -> float: ...


class IntegerVariable(ArraySymbol):
    def lower_bound(self) -> float: ...
    def set_state(self, index: int, state: numpy.typing.ArrayLike): ...
    def upper_bound(self) -> float: ...


class NaryAdd(ArraySymbol):
    ...


class NaryMaximum(ArraySymbol):
    ...


class NaryMinimum(ArraySymbol):
    ...


class NaryMultiply(ArraySymbol):
    ...


class Permutation(ArraySymbol):
    ...


class QuadraticModel(ArraySymbol):
    ...


class Reshape(ArraySymbol):
    ...


class Resize(ArraySymbol):
    ...


class Size(ArraySymbol):
    ...


class Where(ArraySymbol):
    ...

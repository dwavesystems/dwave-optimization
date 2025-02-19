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


class Absolute(ArraySymbol):
    ...


class Add(ArraySymbol):
    ...


class All(ArraySymbol):
    ...


class And(ArraySymbol):
    ...


class Any(ArraySymbol):
    ...


class AdvancedIndexing(ArraySymbol):
    ...


class BasicIndexing(ArraySymbol):
    ...


class BinaryVariable(ArraySymbol):
    def set_state(self, index: int, state: numpy.typing.ArrayLike): ...


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


class DisjointBitSets(Symbol):
    def set_state(self, index: int, state: numpy.typing.ArrayLike): ...


class DisjointBitSet(ArraySymbol):
    ...


class DisjointLists(Symbol):
    def set_state(self, index: int, state: numpy.typing.ArrayLike): ...


class DisjointList(ArraySymbol):
    ...


class Divide(ArraySymbol):
    ...


class Equal(ArraySymbol):
    ...


class Expit(ArraySymbol):
    ...


class IntegerVariable(ArraySymbol):
    def lower_bound(self) -> float: ...
    def set_state(self, index: int, state: numpy.typing.ArrayLike): ...
    def upper_bound(self) -> float: ...


class LessEqual(ArraySymbol):
    ...


class ListVariable(ArraySymbol):
    def set_state(self, index: int, state: numpy.typing.ArrayLike): ...


class Logical(ArraySymbol):
    ...


class Max(ArraySymbol):
    ...


class Maximum(ArraySymbol):
    ...


class Min(ArraySymbol):
    ...


class Minimum(ArraySymbol):
    ...


class Modulus(ArraySymbol):
    ...
    

class Multiply(ArraySymbol):
    ...


class NaryAdd(ArraySymbol):
    ...


class NaryMaximum(ArraySymbol):
    ...


class NaryMinimum(ArraySymbol):
    ...


class NaryMultiply(ArraySymbol):
    ...


class Negative(ArraySymbol):
    ...


class Not(ArraySymbol):
    ...


class Or(ArraySymbol):
    ...


class PartialSum(ArraySymbol):
    ...


class Permutation(ArraySymbol):
    ...


class Prod(ArraySymbol):
    ...


class QuadraticModel(ArraySymbol):
    ...


class Reshape(ArraySymbol):
    ...


class Rint(ArraySymbol):
    ...


class SetVariable(ArraySymbol):
    def set_state(self, index: int, state: numpy.typing.ArrayLike): ...


class Size(ArraySymbol):
    ...


class Square(ArraySymbol):
    ...


class SquareRoot(ArraySymbol):
    ...


class Subtract(ArraySymbol):
    ...


class Sum(ArraySymbol):
    ...


class Where(ArraySymbol):
    ...


class Xor(ArraySymbol):
    ...

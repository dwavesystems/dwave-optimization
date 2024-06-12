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

import numpy.typing

from dwave.optimization.model import Symbol, ArraySymbol


class Absolute(ArraySymbol):
    ...


class Add(ArraySymbol):
    ...


class All(ArraySymbol):
    ...


class And(ArraySymbol):
    ...


class AdvancedIndexing(ArraySymbol):
    ...


class BasicIndexing(ArraySymbol):
    ...


class BinaryVariable(ArraySymbol):
    def set_state(self, index: int, state: numpy.typing.ArrayLike): ...


class Constant(ArraySymbol):
    ...


class DisjointBitSets(Symbol):
    def set_state(self, index: int, state: numpy.typing.ArrayLike): ...


class DisjointBitSet(ArraySymbol):
    ...


class DisjointLists(Symbol):
    def set_state(self, index: int, state: numpy.typing.ArrayLike): ...


class DisjointList(ArraySymbol):
    ...


class Equal(ArraySymbol):
    ...


class IntegerVariable(ArraySymbol):
    def lower_bound(self) -> float: ...
    def set_state(self, index: int, state: numpy.typing.ArrayLike): ...
    def upper_bound(self) -> float: ...


class LessEqual(ArraySymbol):
    ...


class ListVariable(ArraySymbol):
    def set_state(self, index: int, state: numpy.typing.ArrayLike): ...


class Max(ArraySymbol):
    ...


class Maximum(ArraySymbol):
    ...


class Min(ArraySymbol):
    ...


class Minimum(ArraySymbol):
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


class Or(ArraySymbol):
    ...


class Permutation(ArraySymbol):
    ...


class Prod(ArraySymbol):
    ...


class QuadraticModel(ArraySymbol):
    ...


class Reshape(ArraySymbol):
    ...


class SetVariable(ArraySymbol):
    def set_state(self, index: int, state: numpy.typing.ArrayLike): ...


class Square(ArraySymbol):
    ...


class Subtract(ArraySymbol):
    ...


class Sum(ArraySymbol):
    ...

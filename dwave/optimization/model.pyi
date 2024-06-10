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


class ArrayObserver:
    state: StatesView


class Model:
    def constant(self, array_like) -> Constant: ...
    def list(self, n: int) -> ListVariable: ...
    def lock(self) -> None: ...
    def num_nodes(self) -> int: ...
    def resize_states(self, n: int): ...


class Constant(ArrayObserver):
    ...


class ListVariable(ArrayObserver):
    def set_state(self, index: int, state: numpy.typing.ArrayLike) -> None: ...


class StatesView:
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> numpy.typing.NDArray[numpy.float64]: ...

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

from dwave.optimization.model import ArraySymbol as _ArraySymbol
from dwave.optimization.utilities import _NoValueType

class _ReduceNodeType:
    All: int
    Any: int
    Max: int
    Min: int
    Prod: int
    Sum: int

class _ReduceSymbol(_ArraySymbol):
    def __init_subclass__(
        cls,
        /,
        node_type: _ReduceNodeType,
        default_initial: None | float,
    ): ...

    def __init__(
        self,
        array: _ArraySymbol,
        *,
        axis: None | int | tuple[int, ...],
        initial: None | _NoValueType | float,
    ): ...

    @property
    def initial(self) -> None | float: ...

    def axes(self) -> tuple[int, ...]: ...

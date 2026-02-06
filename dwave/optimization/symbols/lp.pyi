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

import numpy

from dwave.optimization.model import ArraySymbol, Symbol

class LinearProgram(Symbol):
    def feasible(self, index: int = 0) -> bool: ...
    def objective_value(self, index: int = 0) -> float: ...
    def state(self, index: int = 0) -> numpy.typing.NDArray: ...

class LinearProgramFeasible(ArraySymbol): ...
class LinearProgramObjectiveValue(ArraySymbol): ...
class LinearProgramSolution(ArraySymbol): ...

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

class _BinaryOpNodeType:
    Add: int
    And: int
    Divide: int
    Equal: int
    LessEqual: int
    Maximum: int
    Minimum: int
    Modulus: int
    Multiply: int
    Or: int
    SafeDivide: int
    Subtract: int
    Xor: int

class _BinaryOpSymbol(_ArraySymbol):
    def __init_subclass__(cls, /, node_type: _BinaryOpNodeType): ...

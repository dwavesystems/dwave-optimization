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

# This is the namespace we want the user to import symbols from. So let's
# pull everything into here

from dwave.optimization.symbols.accumulate_zip import AccumulateZip
from dwave.optimization.symbols.binaryop import (
    Add,
    And,
    Divide,
    Equal,
    LessEqual,
    Maximum,
    Minimum,
    Modulus,
    Multiply,
    Or,
    SafeDivide,
    Subtract,
    Xor,
)
from dwave.optimization.symbols.collections import (
    DisjointBitSet,
    DisjointBitSets,
    DisjointList,
    DisjointLists,
    ListVariable,
    SetVariable,
)
from dwave.optimization.symbols.constants import Constant
from dwave.optimization.symbols.creation import ARange
from dwave.optimization.symbols.flow import (
    Extract,
    Where,
)
from dwave.optimization.symbols.indexing import (
    AdvancedIndexing,
    BasicIndexing,
    Permutation,
)
from dwave.optimization.symbols.inputs import Input
from dwave.optimization.symbols.interpolation import BSpline
from dwave.optimization.symbols.lp import (
    LinearProgram,
    LinearProgramFeasible,
    LinearProgramObjectiveValue,
    LinearProgramSolution,
)
from dwave.optimization.symbols.manipulation import (
    BroadcastTo,
    Concatenate,
    Copy,
    Put,
    Reshape,
    Resize,
    Size,
)
from dwave.optimization.symbols.naryop import (
    NaryAdd,
    NaryMaximum,
    NaryMinimum,
    NaryMultiply,
)
from dwave.optimization.symbols.numbers import (
    BinaryVariable,
    IntegerVariable,
)
from dwave.optimization.symbols.quadratic_model import QuadraticModel
from dwave.optimization.symbols.reduce import (
    All,
    Any,
    Max,
    Min,
    PartialProd,
    PartialSum,
    Prod,
    Sum,
)
from dwave.optimization.symbols.softmax import SoftMax
from dwave.optimization.symbols.sorting import ArgSort
from dwave.optimization.symbols.statistics import Mean
from dwave.optimization.symbols.testing import _ArrayValidation
from dwave.optimization.symbols.unaryop import (
    Absolute,
    Cos,
    Exp,
    Expit,
    Log,
    Logical,
    Negative,
    Not,
    Rint,
    Sin,
    Square,
    SquareRoot,
)


__all__ = [
    "Absolute",
    "AccumulateZip",
    "Add",
    "All",
    "And",
    "Any",
    "AdvancedIndexing",
    "ARange",
    "ArgSort",
    "BasicIndexing",
    "BinaryVariable",
    "BroadcastTo",
    "BSpline",
    "Concatenate",
    "Constant",
    "Copy",
    "Cos",
    "DisjointBitSets",
    "DisjointBitSet",
    "DisjointLists",
    "DisjointList",
    "Divide",
    "Equal",
    "Exp",
    "Expit",
    "Extract",
    "Input",
    "IntegerVariable",
    "LessEqual",
    "LinearProgram",
    "LinearProgramFeasible",
    "LinearProgramObjectiveValue",
    "LinearProgramSolution",
    "ListVariable",
    "Log",
    "Logical",
    "Max",
    "Maximum",
    "Mean",
    "Min",
    "Minimum",
    "Modulus",
    "Multiply",
    "NaryAdd",
    "NaryMaximum",
    "NaryMinimum",
    "NaryMultiply",
    "Negative",
    "Not",
    "Or",
    "PartialProd",
    "PartialSum",
    "Permutation",
    "Prod",
    "Put",
    "QuadraticModel",
    "Reshape",
    "Resize",
    "Sin",
    "Subtract",
    "SetVariable",
    "Size",
    "Rint",
    "SafeDivide",
    "SoftMax",
    "Square",
    "SquareRoot",
    "Sum",
    "Where",
    "Xor",
]

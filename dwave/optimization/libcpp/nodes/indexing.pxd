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

from libcpp.span cimport span
from libcpp.vector cimport vector

from dwave.optimization.libcpp cimport  variant
from dwave.optimization.libcpp.array cimport Slice
from dwave.optimization.libcpp.graph cimport ArrayNode


cdef extern from "dwave-optimization/nodes/indexing.hpp" namespace "dwave::optimization" nogil:
    cdef cppclass AdvancedIndexingNode(ArrayNode):
        ctypedef variant["ArrayNode*", Slice] array_or_slice

        span[const array_or_slice] indices()

    cdef cppclass BasicIndexingNode(ArrayNode):
        ctypedef variant[Slice, Py_ssize_t] slice_or_int

        vector[slice_or_int] infer_indices() except +

    cdef cppclass PermutationNode(ArrayNode):
        pass

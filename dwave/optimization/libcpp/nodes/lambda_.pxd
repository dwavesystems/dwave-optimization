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

from libcpp.memory cimport shared_ptr

from dwave.optimization.libcpp cimport  variant
from dwave.optimization.libcpp.graph cimport ArrayNode, Graph


cdef extern from "dwave-optimization/nodes/lambda.hpp" namespace "dwave::optimization" nogil:
    cdef cppclass AccumulateZipNode(ArrayNode):
        ctypedef variant["ArrayNode*", double] array_or_double
        shared_ptr[Graph] expression_ptr()
        const array_or_double initial

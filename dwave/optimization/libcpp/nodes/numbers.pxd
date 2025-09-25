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

from libcpp.vector cimport vector

from dwave.optimization.libcpp.graph cimport ArrayNode
from dwave.optimization.libcpp.state cimport State


cdef extern from "dwave-optimization/nodes/numbers.hpp" namespace "dwave::optimization" nogil:
    cdef cppclass IntegerNode(ArrayNode):
        void initialize_state(State&, vector[double]) except+
        double lower_bound(Py_ssize_t index)
        double upper_bound(Py_ssize_t index)
        double lower_bound() except+
        double upper_bound() except+

    cdef cppclass BinaryNode(ArrayNode):
        void initialize_state(State&, vector[double]) except+
        double lower_bound(Py_ssize_t index)
        double upper_bound(Py_ssize_t index)
        double lower_bound() except+
        double upper_bound() except+

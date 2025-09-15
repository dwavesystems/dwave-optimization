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
from libcpp.optional cimport nullopt_t, nullopt


cdef extern from *:
    """
    #include <variant>

    template<typename T, typename U, typename V>
    using variant3 = std::variant<T, U, V>;
    """

    cdef cppclass variant3[T, U, V]:
        variant3()
        variant3(T)
        variant3(U)
        variant3(V)

        variant3& operator=(variant3&)
        T& emplace[T](...)

cdef extern from "dwave-optimization/nodes/numbers.hpp" namespace \
        "dwave::optimization::IntegerNode" nogil:
    ctypedef variant3[const vector[double], double, nullopt_t] bounds_t


cdef extern from "dwave-optimization/nodes/numbers.hpp" namespace "dwave::optimization" nogil:
    cdef cppclass IntegerNode(ArrayNode):
        void initialize_state(State&, vector[double]) except+
        double lower_bound(Py_ssize_t index)
        double upper_bound(Py_ssize_t index)

    cdef cppclass BinaryNode(ArrayNode):
        void initialize_state(State&, vector[double]) except+
        double lower_bound(Py_ssize_t index)
        double upper_bound(Py_ssize_t index)

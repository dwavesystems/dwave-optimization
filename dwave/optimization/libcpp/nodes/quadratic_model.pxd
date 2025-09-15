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

from dwave.optimization.libcpp.graph cimport ArrayNode


cdef extern from "dwave-optimization/nodes/quadratic_model.hpp" namespace "dwave::optimization" nogil:
    cdef cppclass QuadraticModel:
        QuadraticModel(int num_variables)
        Py_ssize_t num_variables() const
        Py_ssize_t num_interactions() const
        void add_linear(int v, double bias)
        double get_linear(int v) const

        void add_quadratic(int u, int v, double bias)
        double get_quadratic(int u, int v) const
        void get_quadratic(int* row, int* col, double* quad) const

    cdef cppclass QuadraticModelNode(ArrayNode):
        QuadraticModel* get_quadratic_model()

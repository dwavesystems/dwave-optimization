# distutils: language = c++
# distutils: include_dirs = dwave/optimization/include/

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

from libcpp cimport bool
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.vector cimport vector

from dwave.optimization.libcpp.array cimport Array, span
from dwave.optimization.libcpp.state cimport State

cdef extern from "dwave-optimization/graph.hpp" namespace "dwave::optimization" nogil:
    cdef cppclass Graph:
        T* emplace_node[T](...) except +ValueError
        void initialize_state(State&) except+
        span[const unique_ptr[Node]] nodes() const
        span[Array*] constraints() const
        Py_ssize_t num_nodes()
        Py_ssize_t num_decisions()
        Py_ssize_t num_constraints()
        @staticmethod
        void recursive_initialize(State&, Node*) except+
        @staticmethod
        void recursive_reset(State&, Node*) except+
        void reset_topological_sort()
        void set_objective(Array*) except+
        void add_constraint(Array*) except+
        void topological_sort()
        bool topologically_sorted() const


    cdef cppclass Node:
        struct SuccessorView:
            Node* ptr
        shared_ptr[bool] expired_ptr() const
        const vector[Node*]& predecessors() const
        const vector[SuccessorView]& successors() const
        Py_ssize_t topological_index()

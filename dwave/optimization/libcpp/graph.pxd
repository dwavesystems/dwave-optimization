# Copyright 2024 D-Wave
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

from dwave.optimization.libcpp cimport span
from dwave.optimization.libcpp.array cimport Array
from dwave.optimization.libcpp.state cimport State

cdef extern from "dwave-optimization/graph.hpp" namespace "dwave::optimization" nogil:
    cdef cppclass Node:
        struct SuccessorView:
            Node* ptr
        shared_ptr[bool] expired_ptr() const
        const vector[Node*]& predecessors() const
        const vector[SuccessorView]& successors() const
        Py_ssize_t topological_index()

    cdef cppclass ArrayNode(Node, Array):
        pass

    cdef cppclass DecisionNode(Node):
        pass

# Sometimes Cython isn't able to reason about pointers as template inputs, so
# we make a few aliases for convenience
ctypedef Node* NodePtr
ctypedef ArrayNode* ArrayNodePtr
ctypedef DecisionNode* DecisionNodePtr

cdef extern from "dwave-optimization/graph.hpp" namespace "dwave::optimization" nogil:
    cdef cppclass Graph:
        T* emplace_node[T](...) except+
        void initialize_state(State&) except+
        span[const unique_ptr[Node]] nodes() const
        span[const ArrayNodePtr] constraints()
        span[const DecisionNodePtr] decisions()
        Py_ssize_t num_nodes()
        Py_ssize_t num_decisions()
        Py_ssize_t num_constraints()
        @staticmethod
        void recursive_initialize(State&, Node*) except+
        @staticmethod
        void recursive_reset(State&, Node*) except+
        void reset_topological_sort()
        void set_objective(ArrayNode*) except+
        void add_constraint(ArrayNode*) except+
        void topological_sort()
        bool topologically_sorted() const
        Py_ssize_t remove_unused_nodes()

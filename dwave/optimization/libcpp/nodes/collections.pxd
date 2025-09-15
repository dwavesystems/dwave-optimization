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

from dwave.optimization.libcpp.graph cimport ArrayNode, Node
from dwave.optimization.libcpp.state cimport State


cdef extern from "dwave-optimization/nodes/collections.hpp" namespace "dwave::optimization" nogil:
    cdef cppclass DisjointBitSetsNode(Node):
        void initialize_state(State&, vector[vector[double]]) except+
        Py_ssize_t primary_set_size() const
        Py_ssize_t num_disjoint_sets() const

    cdef cppclass DisjointBitSetNode(ArrayNode):
        Py_ssize_t set_index()

    cdef cppclass DisjointListsNode(Node):
        void initialize_state(State&, vector[vector[double]]) except+
        Py_ssize_t num_disjoint_lists() const
        Py_ssize_t primary_set_size() const

    cdef cppclass DisjointListNode(ArrayNode):
        Py_ssize_t list_index()

    cdef cppclass ListNode(ArrayNode):
        ListNode(Py_ssize_t) except+
        void initialize_state(State&, vector[double]) except+

    cdef cppclass SetNode(ArrayNode):
        SetNode(Py_ssize_t, Py_ssize_t, Py_ssize_t) except+
        void initialize_state(State&, vector[double]) except+

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

from libcpp.vector cimport vector

from dwave.optimization.libcpp cimport span, variant
from dwave.optimization.libcpp.array cimport Array, ArrayPtr, Slice
from dwave.optimization.libcpp.graph cimport Node
from dwave.optimization.libcpp.state cimport State


cdef extern from "dwave-optimization/nodes/collections.hpp" namespace "dwave::optimization" nogil:
    cdef cppclass DisjointBitSetsNode(Node):
        void initialize_state(State&, vector[vector[double]]) except+
        Py_ssize_t primary_set_size() const
        Py_ssize_t num_disjoint_sets() const

    cdef cppclass DisjointBitSetNode(Node, Array):
        Py_ssize_t set_index()

    cdef cppclass DisjointListsNode(Node):
        void initialize_state(State&, vector[vector[double]]) except+
        Py_ssize_t num_disjoint_lists() const
        Py_ssize_t primary_set_size() const

    cdef cppclass DisjointListNode(Node, Array):
        Py_ssize_t list_index()

    cdef cppclass ListNode(Node, Array):
        ListNode(Py_ssize_t) except+
        void initialize_state(State&, vector[double]) except+

    cdef cppclass SetNode(Node, Array):
        SetNode(Py_ssize_t, Py_ssize_t, Py_ssize_t) except+
        void initialize_state(State&, vector[double]) except+


cdef extern from "dwave-optimization/nodes/constants.hpp" namespace "dwave::optimization" nogil:
    cdef cppclass ConstantNode(Array, Node):
        const double* buff() const


cdef extern from "dwave-optimization/nodes/indexing.hpp" namespace "dwave::optimization" nogil:
    cdef cppclass AdvancedIndexingNode(Node, Array):
        ctypedef variant[ArrayPtr, Slice] array_or_slice

        AdvancedIndexingNode(Node*, vector[array_or_slice]) except +

        span[const array_or_slice] indices()

    cdef cppclass BasicIndexingNode(Node, Array):
        ctypedef variant[Slice, Py_ssize_t] slice_or_int

        BasicIndexingNode(Node*, vector[slice_or_int]) except +

        vector[slice_or_int] infer_indices() except +

    cdef cppclass PermutationNode(Node, Array):
        pass

    cdef cppclass ReshapeNode(Node, Array):
        pass


cdef extern from "dwave-optimization/nodes/mathematical.hpp" namespace "dwave::optimization" nogil:
    cdef cppclass AbsoluteNode(Node, Array):
        pass

    cdef cppclass AddNode(Node, Array):
        pass

    cdef cppclass AllNode(Node, Array):
        pass

    cdef cppclass AndNode(Node, Array):
        pass

    cdef cppclass EqualNode(Node, Array):
        pass

    cdef cppclass LessEqualNode(Node, Array):
        pass

    cdef cppclass MaximumNode(Node, Array):
        pass

    cdef cppclass MaxNode(Node, Array):
        pass

    cdef cppclass MinimumNode(Node, Array):
        pass

    cdef cppclass MinNode(Node, Array):
        pass

    cdef cppclass MultiplyNode(Node, Array):
        pass

    cdef cppclass NaryAddNode(Node, Array):
        pass

    cdef cppclass NaryMaximumNode(Node, Array):
        pass

    cdef cppclass NaryMinimumNode(Node, Array):
        pass

    cdef cppclass NaryMultiplyNode(Node, Array):
        pass

    cdef cppclass NegativeNode(Node, Array):
        pass

    cdef cppclass OrNode(Node, Array):
        pass

    cdef cppclass ProdNode(Node, Array):
        pass

    cdef cppclass SquareNode(Node, Array):
        pass

    cdef cppclass SubtractNode(Node, Array):
        pass

    cdef cppclass SumNode(Node, Array):
        pass


cdef extern from "dwave-optimization/nodes/numbers.hpp" namespace "dwave::optimization" nogil:
    cdef cppclass IntegerNode(Node, Array):
        void initialize_state(State&, vector[double]) except+
        double lower_bound()
        double upper_bound()

    cdef cppclass BinaryNode(Node, Array):
        void initialize_state(State&, vector[double]) except+


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

    cdef cppclass QuadraticModelNode(Node, Array):
        QuadraticModel* get_quadratic_model()


cdef extern from "dwave-optimization/nodes/testing.hpp" namespace "dwave::optimization" nogil:
    cdef cppclass ArrayValidationNode(Node):
        pass

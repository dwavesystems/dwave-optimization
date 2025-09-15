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
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map

from dwave.optimization.libcpp.graph cimport ArrayNode, Node
from dwave.optimization.libcpp.state cimport State


cdef extern from "dwave-optimization/nodes/lp.hpp" namespace "dwave::optimization" nogil:
    cdef cppclass LinearProgramFeasibleNode(ArrayNode):
        pass

    cdef cppclass LinearProgramNode(Node):
        unordered_map[string, ssize_t] get_arguments()
        void initialize_state(State&, ...) except +  # Cython gets confused between span<double> and span<const double>
        span[const double] solution(const State&) const
        span[const Py_ssize_t] variables_shape() const

    cdef cppclass LinearProgramNodeBase(Node):
        pass

    cdef cppclass LinearProgramObjectiveValueNode(ArrayNode):
        pass

    cdef cppclass LinearProgramSolutionNode(ArrayNode):
        pass

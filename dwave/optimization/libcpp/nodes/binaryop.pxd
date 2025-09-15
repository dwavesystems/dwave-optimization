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


cdef extern from "dwave-optimization/nodes/binaryop.hpp" namespace "dwave::optimization" nogil:
    cdef cppclass AddNode(ArrayNode):
        pass

    cdef cppclass AndNode(ArrayNode):
        pass

    cdef cppclass DivideNode(ArrayNode):
        pass

    cdef cppclass EqualNode(ArrayNode):
        pass

    cdef cppclass LessEqualNode(ArrayNode):
        pass

    cdef cppclass MaximumNode(ArrayNode):
        pass

    cdef cppclass MinimumNode(ArrayNode):
        pass

    cdef cppclass ModulusNode(ArrayNode):
        pass

    cdef cppclass MultiplyNode(ArrayNode):
        pass

    cdef cppclass OrNode(ArrayNode):
        pass

    cdef cppclass SafeDivideNode(ArrayNode):
        pass

    cdef cppclass SubtractNode(ArrayNode):
        pass

    cdef cppclass XorNode(ArrayNode):
        pass

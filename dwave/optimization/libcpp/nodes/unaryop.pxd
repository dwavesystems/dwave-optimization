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


cdef extern from "dwave-optimization/nodes/unaryop.hpp" namespace "dwave::optimization" nogil:
    cdef cppclass AbsoluteNode(ArrayNode):
        pass

    cdef cppclass CosNode(ArrayNode):
        pass

    cdef cppclass ExpitNode(ArrayNode):
        pass

    cdef cppclass ExpNode(ArrayNode):
        pass

    cdef cppclass LogNode(ArrayNode):
        pass

    cdef cppclass LogicalNode(ArrayNode):
        pass

    cdef cppclass NegativeNode(ArrayNode):
        pass

    cdef cppclass NotNode(ArrayNode):
        pass

    cdef cppclass RintNode(ArrayNode):
        pass

    cdef cppclass SinNode(ArrayNode):
        pass

    cdef cppclass SquareNode(ArrayNode):
        pass
        
    cdef cppclass SquareRootNode(ArrayNode):
        pass

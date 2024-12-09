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

from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

__all__ = ["NodeStateData", "State"]

cdef extern from "dwave-optimization/state.hpp" namespace "dwave::optimization" nogil:
    cdef cppclass NodeStateData:
        pass

    ctypedef vector[unique_ptr[NodeStateData]] State

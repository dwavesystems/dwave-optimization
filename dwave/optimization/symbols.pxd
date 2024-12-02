# distutils: language = c++

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

from libcpp.typeinfo cimport type_info

from dwave.optimization.graph_manager cimport _GraphManager
from dwave.optimization.libcpp.graph cimport Array as cppArray
from dwave.optimization.libcpp.graph cimport Node as cppNode

cdef void _register(object cls, const type_info& typeinfo)

cdef object symbol_from_ptr(_GraphManager model, cppNode* ptr)

# Copyright 2024 D-Wave Inc.
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

from libc.stdint cimport uintptr_t
from libcpp cimport bool
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector

from dwave.optimization.libcpp.graph cimport ArrayNode as cppArrayNode, Node as cppNode
from dwave.optimization.libcpp.graph cimport Graph as cppGraph
from dwave.optimization.libcpp.state cimport State as cppState

__all__ = []


cdef class _Graph:
    cpdef bool is_locked(self) noexcept
    cpdef Py_ssize_t num_constraints(self) noexcept
    cpdef Py_ssize_t num_decisions(self) noexcept
    cpdef Py_ssize_t num_nodes(self) noexcept
    cpdef Py_ssize_t num_symbols(self) noexcept

    # Make the _Graph class weak referenceable
    cdef object __weakref__

    cdef cppGraph _graph

    # The number of times "lock()" has been called.
    cdef readonly Py_ssize_t _lock_count

    # Used to keep NumPy arrays that own data alive etc etc
    # We could pair each of these with an expired_ptr for the node holding
    # memory for easier cleanup later if that becomes a concern.
    cdef object _data_sources


cdef class Symbol:
    # Inheriting nodes must call this method from their __init__()
    cdef void initialize_node(self, _Graph model, cppNode* node_ptr) noexcept

    cpdef uintptr_t id(self) noexcept

    # Exactly deref(self.expired_ptr)
    cpdef bool expired(self) noexcept

    @staticmethod
    cdef Symbol from_ptr(_Graph model, cppNode* ptr)

    # Hold on to a reference to the _Graph, both for access but also, importantly,
    # to ensure that the model doesn't get garbage collected unless all of
    # the observers have also been garbage collected.
    cdef readonly _Graph model

    # Hold Node* pointer. This is redundant as most observers will also hold
    # a pointer to their observed node with the correct type. But the cost
    # of a redundant pointer is quite small for these Python objects and it
    # simplifies things quite a bit.
    cdef cppNode* node_ptr

    # The node's expired flag. If the node is destructed, the boolean value
    # pointed to by the expired_ptr will be set to True
    cdef shared_ptr[bool] expired_ptr


# Ideally this wouldn't subclass Symbol, but Cython only allows a single
# extension base class, so to support that we assume all ArraySymbols are
# also Symbols (probably a fair assumption)
cdef class ArraySymbol(Symbol):
    # Inheriting symbols must call this method from their __init__()
    cdef void initialize_arraynode(self, _Graph model, cppArrayNode* array_ptr) noexcept

    # Hold ArrayNode* pointer. Again this is redundant, because we're also holding
    # a pointer to Node* and we can theoretically dynamic cast each time.
    # But again, it's cheap and it simplifies things.
    cdef cppArrayNode* array_ptr

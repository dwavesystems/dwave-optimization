# cython: auto_pickle=False

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

from libc.stdint cimport uintptr_t
from libcpp cimport bool
from libcpp.memory cimport shared_ptr
from libcpp.typeinfo cimport type_info
from libcpp.vector cimport vector

from dwave.optimization.libcpp.graph cimport ArrayNode, Graph, Node
from dwave.optimization.libcpp.state cimport State

__all__ = []


cdef void _register(object cls, const type_info& typeinfo)

cdef object symbol_from_ptr(_Graph model, Node* ptr)


cdef class _Graph:
    @staticmethod
    cdef _Graph from_shared_ptr(object cls, shared_ptr[Graph])

    cpdef bool is_locked(self) noexcept
    cpdef Py_ssize_t num_constraints(self) noexcept
    cpdef Py_ssize_t num_decisions(self) noexcept
    cpdef Py_ssize_t num_inputs(self) noexcept
    cpdef Py_ssize_t num_nodes(self) noexcept
    cpdef Py_ssize_t num_symbols(self) noexcept

    # Make the _Graph class weak referenceable
    cdef object __weakref__

    # The lifespan of the C++ Graph is managed by a shared_ptr<Graph>, but
    # we add an additional (redundant) Graph* because Cython knows what to do
    # with that.
    cdef shared_ptr[Graph] _owning_ptr
    cdef Graph* _graph

    # The number of times "lock()" has been called.
    cdef readonly Py_ssize_t _lock_count


cdef class Symbol:
    # Inheriting nodes must call this method from their __init__()
    cdef void initialize_node(self, _Graph model, Node* node_ptr) noexcept

    cpdef uintptr_t id(self) noexcept

    # Exactly deref(self.expired_ptr)
    cpdef bool expired(self) noexcept

    @staticmethod
    cdef Symbol from_ptr(_Graph model, Node* ptr)

    # Hold on to a reference to the _Graph, both for access but also, importantly,
    # to ensure that the model doesn't get garbage collected unless all of
    # the observers have also been garbage collected.
    cdef readonly _Graph model

    # Hold Node* pointer. This is redundant as most observers will also hold
    # a pointer to their observed node with the correct type. But the cost
    # of a redundant pointer is quite small for these Python objects and it
    # simplifies things quite a bit.
    cdef Node* node_ptr

    # The node's expired flag. If the node is destructed, the boolean value
    # pointed to by the expired_ptr will be set to True
    cdef shared_ptr[const bool] expired_ptr


# Ideally this wouldn't subclass Symbol, but Cython only allows a single
# extension base class, so to support that we assume all ArraySymbols are
# also Symbols (probably a fair assumption)
cdef class ArraySymbol(Symbol):
    # Inheriting symbols must call this method from their __init__()
    cdef void initialize_arraynode(self, _Graph model, ArrayNode* array_ptr) noexcept

    # Hold ArrayNode* pointer. Again this is redundant, because we're also holding
    # a pointer to Node* and we can theoretically dynamic cast each time.
    # But again, it's cheap and it simplifies things.
    cdef ArrayNode* array_ptr


cdef class States:
    cdef void attach_states(self, vector[State]) noexcept
    cdef vector[State] detach_states(self)
    cpdef resolve(self)
    cpdef Py_ssize_t size(self) except -1

    # In order to not create a circular reference, we only hold a weakref
    # to the model from the states. This introduces some overhead, but it
    # makes sure that the Model is promptly garbage collected
    cdef readonly object _model_ref

    # The state(s) of the model kept as a ragged vector-of-vectors (each
    # State is a vector).
    # Accessors should check the length of the state when accessing!
    cdef vector[State] _states

    # Object that contains or will contain the information needed to construct states
    cdef readonly object _future
    cdef readonly object _result_hook

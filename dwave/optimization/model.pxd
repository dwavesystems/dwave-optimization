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

from libcpp cimport bool
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector

from dwave.optimization.libcpp.array cimport Array as cppArray
from dwave.optimization.libcpp.graph cimport Graph as cppGraph, Node as cppNode
from dwave.optimization.libcpp.state cimport State as cppState

__all__ = ["Model"]


cdef class Model:
    cpdef bool is_locked(self) noexcept
    cpdef Py_ssize_t num_decisions(self) noexcept
    cpdef Py_ssize_t num_nodes(self) noexcept
    cpdef Py_ssize_t num_constraints(self) noexcept
    
    # Allow dynamic attributes on the Model class
    cdef dict __dict__

    # Make the Model class weak referenceable
    cdef object __weakref__

    cdef cppGraph _graph

    cdef readonly object objective  # todo: cdef ArrayObserver?

    cdef readonly States states

    # The number of times "lock()" has been called.
    cdef readonly Py_ssize_t _lock_count

    # Used to keep NumPy arrays that own data alive etc etc
    # We could pair each of these with an expired_ptr for the node holding
    # memory for easier cleanup later if that becomes a concern.
    cdef object _data_sources


cdef class States:
    """The states/solutions of the model."""

    cdef void attach_states(self, vector[cppState]) noexcept
    cdef vector[cppState] detach_states(self)
    cpdef resolve(self)
    cpdef Py_ssize_t size(self) except -1

    cdef Model _model(self)

    # In order to not create a circular reference, we only hold a weakref
    # to the model from the states. This introduces some overhead, but it
    # makes sure that the Model is promptly garbage collected
    cdef readonly object _model_ref

    # The state(s) of the model kept as a ragged vector-of-vectors (each
    # cppState is a vector).
    # Accessors should check the length of the state when accessing!
    cdef vector[cppState] _states

    # The number of views into the states that exist. The model cannot
    # be unlocked while there are unsafe views
    cdef Py_ssize_t _view_count

    # Object that contains or will contain the information needed to construct states
    cdef readonly object _future
    cdef readonly object _result_hook


cdef class NodeObserver:
    # Inheriting nodes must call this method from their __init__()
    cdef void initialize_node(self, Model model, cppNode* node_ptr) noexcept

    # Exactly deref(self.expired_ptr)
    cpdef bool expired(self) noexcept

    # Hold on to a reference to the Model, both for access but also, importantly,
    # to ensure that the model doesn't get garbage collected unless all of
    # the observers have also been garbage collected.
    cdef readonly Model model

    # Hold Node* pointer. This is redundant as most observers will also hold
    # a pointer to their observed node with the correct type. But the cost
    # of a redundant pointer is quite small for these Python objects and it
    # simplifies things quite a bit.
    cdef cppNode* node_ptr

    # The node's expired flag. If the node is destructed, the boolean value
    # pointed to by the expired_ptr will be set to True
    cdef shared_ptr[bool] expired_ptr


# Ideally this wouldn't subclass NodeObserver, but Cython only allows a single
# extension base class, so to support that we assume all ArrayObservers are
# also NodeObservers (probably a fair assumption)
cdef class ArrayObserver(NodeObserver):
    # Inheriting nodes must call this method from their __init__()
    cdef void initialize_array(self, cppArray* array_ptr) noexcept

    # Hold Array* pointer. Again this is redundant, because we're also holding
    # a pointer to Node* and we can theoretically dynamic cast each time.
    # But again, it's cheap and it simplifies things.
    cdef cppArray* array_ptr


cdef class StateView:
    cdef readonly Py_ssize_t index  # which state we're accessing
    cdef readonly ArrayObserver symbol

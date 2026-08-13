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

from libcpp.vector cimport vector

from dwave.optimization.libcpp.state cimport State


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

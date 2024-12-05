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

from dwave.optimization.libcpp.state cimport State as cppState
from dwave.optimization.model cimport _Graph


cdef class States:
    cdef void attach_states(self, vector[cppState]) noexcept
    cdef vector[cppState] detach_states(self)
    cpdef resolve(self)
    cpdef Py_ssize_t size(self) except -1

    cdef _Graph _model(self)

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
    cdef public Py_ssize_t _view_count

    # Object that contains or will contain the information needed to construct states
    cdef readonly object _future
    cdef readonly object _result_hook

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

# As of Cython 3.0.8 these are not in Cython's libcpp

cdef extern from "<variant>" namespace "std" nogil:
    cdef cppclass variant[T, U]:
        variant()
        variant(T)
        variant(U)

    T get[T](...)
    bint holds_alternative[T](...)


# We would like to be able to do constructions like dynamic_cast[cppConstantNode*](...)
# but Cython does not allow pointers as template types
# see https://github.com/cython/cython/issues/2143
# So instead we create our own wrapper to handle this case. Crucially, the
# template type is the class, but it dynamically casts on the pointer
cdef extern from *:
    """
    template<class T, class F>
    T* dynamic_cast_ptr(F* ptr) {
        return dynamic_cast<T*>(ptr);
    }
    """
    cdef T* dynamic_cast_ptr[T](...) noexcept

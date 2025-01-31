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

from libcpp.optional cimport optional
from libcpp.string cimport string

from dwave.optimization.libcpp cimport span
from dwave.optimization.libcpp.state cimport State

__all__ = ["Array"]


cdef extern from "dwave-optimization/array.hpp" namespace "dwave::optimization" nogil:
    cdef cppclass Array:
        double* buff(State&)
        bint contiguous() const
        bint dynamic() const
        const string& format() const
        Py_ssize_t itemsize() const
        Py_ssize_t len(State&) const
        Py_ssize_t len() const
        bint logical() const
        double max() const
        double min() const
        Py_ssize_t ndim() const
        Py_ssize_t size() const
        SizeInfo sizeinfo() const
        span[const Py_ssize_t] shape(State&) const
        span[const Py_ssize_t] shape() const
        span[const Py_ssize_t] strides() const

    cdef cppclass SizeInfo:
        SizeInfo substitute() const
        SizeInfo substitute(ssize_t) const

        const Array* array_ptr
        optional[Py_ssize_t] min
        optional[Py_ssize_t] max

    cdef cppclass Slice:
        Slice()
        Slice(optional[Py_ssize_t] stop)
        Slice(optional[Py_ssize_t] start, optional[Py_ssize_t] stop)
        Slice(optional[Py_ssize_t] start, optional[Py_ssize_t] stop, optional[Py_ssize_t] step)

        Py_ssize_t start
        Py_ssize_t stop
        Py_ssize_t step

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

from libc.stdint cimport int16_t, int32_t, int64_t
from libcpp.span cimport span
from libcpp.vector cimport vector

__all__ = ["as_cppshape", "as_span"]


# cython.numeric includes complex numbers which we don't want
ctypedef fused numeric:
    signed char  # int8_t, but Cython is grumpy about that for some reason
    int16_t
    int32_t
    int64_t
    float
    double


cdef vector[Py_ssize_t] as_cppshape(object shape, bint nonnegative=?)

cdef span[numeric] as_span(numeric[::1] array)

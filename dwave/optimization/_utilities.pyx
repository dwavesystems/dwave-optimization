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

import collections
import numbers


cdef vector[Py_ssize_t] as_cppshape(object shape, bint nonnegative = True):
    """Convert a shape specified as a python object to a C++ vector."""

    # Use the same error messages as NumPy

    if isinstance(shape, numbers.Integral):
        return as_cppshape((shape,), nonnegative=nonnegative)

    if not isinstance(shape, collections.abc.Sequence):
        raise TypeError(f"expected a sequence of integers or a single integer, got '{repr(shape)}'")

    shape = tuple(shape)  # cast from list or whatever

    if not all(isinstance(x, numbers.Integral) for x in shape):
        raise ValueError(f"expected a sequence of integers or a single integer, got '{repr(shape)}'")

    if nonnegative and any(x < 0 for x in shape):
        raise ValueError("negative dimensions are not allowed")

    return shape


cdef span[numeric] as_span(numeric[::1] array):
    """Convert a Cython memoryview over contiguous memory into a C++ span"""
    if array.size:
        return span[numeric](&array[0], <size_t>array.size)
    return span[numeric]()

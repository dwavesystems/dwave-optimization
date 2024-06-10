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

cdef extern from "<span>" namespace "std" nogil:
    cdef cppclass span[T]:
        ctypedef size_t size_type
        ctypedef ptrdiff_t difference_type

        cppclass iterator:
            iterator() except +
            iterator(iterator&) except +
            T& operator*()
            iterator operator++()
            iterator operator--()
            iterator operator++(int)
            iterator operator--(int)
            iterator operator+(size_type)
            iterator operator-(size_type)
            difference_type operator-(iterator)
            difference_type operator-(const_iterator)
            bint operator==(iterator)
            bint operator==(const_iterator)
            bint operator!=(iterator)
            bint operator!=(const_iterator)
            bint operator<(iterator)
            bint operator<(const_iterator)
            bint operator>(iterator)
            bint operator>(const_iterator)
            bint operator<=(iterator)
            bint operator<=(const_iterator)
            bint operator>=(iterator)
            bint operator>=(const_iterator)

        span()
        span(T* ptr)

        T& operator[](ssize_t)

        iterator begin()
        T* data()
        iterator end()
        size_type size()


cdef extern from "<variant>" namespace "std" nogil:
    cdef cppclass variant[T, U]:
        variant()
        variant(T)
        variant(U)

    T get[T](...)
    bint holds_alternative[T](...)

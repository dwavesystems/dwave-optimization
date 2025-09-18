# cython: auto_pickle=False

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

cimport cpython.buffer
cimport cpython.object
import numpy as np

from cpython.ref cimport PyObject
from cython.operator cimport dereference as deref, typeid
from libc.math cimport modf
from libcpp cimport bool
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector

from dwave.optimization._model cimport _Graph, _register, ArraySymbol, Symbol
from dwave.optimization.libcpp cimport dynamic_cast_ptr
from dwave.optimization.libcpp.nodes.constants cimport ConstantNode

cdef extern from *:
    """
    #include "Python.h"

    struct PyDataSource : dwave::optimization::ConstantNode::DataSource {
        PyDataSource(PyObject* ptr) : ptr_(ptr) {
            Py_INCREF(ptr_);
        }
        ~PyDataSource() {
            Py_DECREF(ptr_);
        }

        PyObject* ptr_;
    };
    """
    cppclass PyDataSource:
        PyDataSource(PyObject*)


cdef class Constant(ArraySymbol):
    """Constant symbol.

    See also:
        :meth:`~dwave.optimization.model.Model.constant`: equivalent method.
    """
    def __init__(self, _Graph model, array_like):
        # In the future we won't need to be contiguous, but we do need to be right now
        array = np.asarray_chkfinite(array_like, dtype=np.double, order="C")

        # Get the shape and strides
        cdef vector[Py_ssize_t] shape = array.shape
        cdef vector[Py_ssize_t] strides = array.strides  # not used because contiguous for now

        # Get a pointer to the first element
        cdef const double[:] flat = array.ravel()
        cdef const double* start = NULL
        if flat.size:
            start = &flat[0]

        # Make a PyDataSource that will essentially take ownership of the numpy array,
        # preventing garbage collection from deallocating it before the C++ node is
        # destructed
        cdef unique_ptr[PyDataSource] data_source = make_unique[PyDataSource](<PyObject*>(array))
        # Get an observing pointer to the C++ ConstantNode
        self.ptr = model._graph.emplace_node[ConstantNode](move(data_source), start, shape)

        self.initialize_arraynode(model, self.ptr)

    def __bool__(self):
        if not self._is_scalar():
            raise ValueError("the truth value of a constant with more than one element is ambiguous")

        return <bool>deref(self.ptr.buff())

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        # We never export a writeable array
        if flags & cpython.buffer.PyBUF_WRITABLE == cpython.buffer.PyBUF_WRITABLE:
            raise BufferError(f"{type(self).__name__} cannot export a writeable buffer")

        # The remaining flags are accurate to the information we export, but over-zealous.
        # We could, for instance, check whether we're contiguous and in that case not raise
        # an error.
        # But for now, we *always* expose strides, format, and we never assume that we're
        # contiguous.
        # Luckily, NumPy and memoryview always ask for everything so it doesn't really matter.
        # If there is a compelling use case we can add more information.
        if flags & cpython.buffer.PyBUF_STRIDES != cpython.buffer.PyBUF_STRIDES:
            raise BufferError(f"{type(self).__name__} always returns stride information")
        if flags & cpython.buffer.PyBUF_FORMAT != cpython.buffer.PyBUF_FORMAT:
            raise BufferError(f"{type(self).__name__} always sets the format field")
        if (flags & cpython.buffer.PyBUF_ANY_CONTIGUOUS == cpython.buffer.PyBUF_ANY_CONTIGUOUS or
                flags & cpython.buffer.PyBUF_C_CONTIGUOUS == cpython.buffer.PyBUF_C_CONTIGUOUS or
                flags & cpython.buffer.PyBUF_F_CONTIGUOUS == cpython.buffer.PyBUF_F_CONTIGUOUS):
            raise BufferError(f"{type(self).__name__} is not necessarily contiguous")

        buffer.buf = <void*>(self.ptr.buff())
        buffer.format = <char*>(self.ptr.format().c_str())
        buffer.internal = NULL
        buffer.itemsize = self.ptr.itemsize()
        buffer.len = self.ptr.len()
        buffer.ndim = self.ptr.ndim()
        buffer.obj = self
        buffer.readonly = 1
        buffer.shape = <Py_ssize_t*>(self.ptr.shape().data())
        buffer.strides = <Py_ssize_t*>(self.ptr.strides().data())
        buffer.suboffsets = NULL

    def __index__(self):
        if not self._is_integer():
            # Follow NumPy's error message
            # https://github.com/numpy/numpy/blob/66e1e3/numpy/_core/src/multiarray/number.c#L833
            raise TypeError("only integer scalar constants can be converted to a scalar index")

        return <Py_ssize_t>deref(self.ptr.buff())

    def __richcmp__(self, rhs, int op):
        # __richcmp__ is a special Cython method

        # If rhs is another Symbol, defer to ArraySymbol to handle the
        # operation. Which may or may not actually be implemented.
        # Otherwise, defer to NumPy.
        # We could also check if rhs is another Constant and handle that differently,
        # but that might lead to confusing behavior so we treat other Constants the
        # same as any other symbol.
        lhs = super() if isinstance(rhs, ArraySymbol) else np.asarray(self)

        if op == cpython.object.Py_EQ:
            return lhs.__eq__(rhs)
        elif op == cpython.object.Py_GE:
            return lhs.__ge__(rhs)
        elif op == cpython.object.Py_GT:
            return lhs.__gt__(rhs)
        elif op == cpython.object.Py_LE:
            return lhs.__le__(rhs)
        elif op == cpython.object.Py_LT:
            return lhs.__lt__(rhs)
        elif op == cpython.object.Py_NE:
            return lhs.__ne__(rhs)
        else:
            return NotImplemented  # this should never happen, but just in case

    cdef bool _is_integer(self) noexcept:
        """Return True if the constant encodes a single integer."""
        if not self._is_scalar():
            return False

        # https://stackoverflow.com/q/1521607 for the integer test
        cdef double dummy
        return modf(deref(self.ptr.buff()), &dummy) == <double>0.0

    cdef bool _is_scalar(self) noexcept:
        """Return True if the constant encodes a single value."""
        # The size check is redundant, but worth checking in order to avoid segfaults
        return self.ptr.size() == 1 and self.ptr.ndim() == 0

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef ConstantNode* ptr = dynamic_cast_ptr[ConstantNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")

        cdef Constant constant = Constant.__new__(Constant)
        constant.ptr = ptr
        constant.initialize_arraynode(symbol.model, ptr)
        return constant

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        """Construct a constant symbol from a compressed file.

        Args:
            zf:
                File pointer to a compressed file encoding
                a constant symbol. Strings are interpreted as a file name.
            directory:
                Directory where the file is located.
            model:
                The relevant :class:`~dwave.optimization.model.Model`.
            predecessors:
                Not currently supported.
        Returns:
            A constant symbol.

        See also:
            :meth:`._into_zipfile`
        """
        if predecessors:
            raise ValueError(f"{cls.__name__} cannot have predecessors")

        with zf.open(directory + "array.npy", mode="r") as f:
            array = np.load(f, allow_pickle=False)

        return cls(model, array)

    def _into_zipfile(self, zf, directory):
        """Store a constant symbol as a compressed file.

        Args:
            zf:
                File pointer to a compressed file to store the
                constant symbol. Strings are interpreted as a file name.
            directory:
                Directory where the file is located.
        Returns:
            A compressed file.

        See also:
            :meth:`._from_zipfile`
        """
        super()._into_zipfile(zf, directory)
        with zf.open(directory + "array.npy", mode="w", force_zip64=True) as f:
            # dev note: I benchmarked using some lower-level functions
            # like np.lib.format.write_array() etc and it didn't have
            # any noticeable impact on performance (numpy==1.26.3).
            np.save(f, self, allow_pickle=False)

    def maybe_equals(self, other):
        cdef Py_ssize_t maybe = super().maybe_equals(other)
        cdef Py_ssize_t NOT = 0
        cdef Py_ssize_t MAYBE = 1
        cdef Py_ssize_t DEFINITELY = 2
        if maybe != MAYBE:
            return DEFINITELY if maybe else NOT

        # avoid NumPy deprecation warning by casting to bool. But also
        # `bool` in this namespace is a C++ class so we do an explicit if else
        equal = (np.asarray(self) == np.asarray(other)).all()
        return DEFINITELY if equal else NOT

    def state(self, Py_ssize_t index=0, *, bool copy = True):
        """Return the state of the constant symbol.

        Args:
            index:
                Index of the state.
            copy:
                Copy the state. Currently only ``True`` is supported.
        Returns:
            A copy of the state.
        """
        if not copy:
            raise NotImplementedError("copy=False is not (yet) supported")

        return np.array(self, copy=copy)

    # An observing pointer to the C++ ConstantNode
    cdef ConstantNode* ptr

_register(Constant, typeid(ConstantNode))

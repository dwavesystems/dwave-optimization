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

import collections.abc
import json

import numpy as np

from cython.operator cimport typeid
from libcpp.vector cimport vector

from dwave.optimization._model cimport _Graph, _register, ArraySymbol, Symbol
from dwave.optimization._utilities cimport as_cppshape
from dwave.optimization.libcpp cimport dynamic_cast_ptr
from dwave.optimization.libcpp.graph cimport ArrayNode
from dwave.optimization.libcpp.nodes.manipulation cimport (
    BroadcastToNode,
    ConcatenateNode,
    CopyNode,
    PutNode,
    ReshapeNode,
    ResizeNode,
    SizeNode,
)


cdef class BroadcastTo(ArraySymbol):
    """BroadcastTo symbol.

    See Also:
        :func:`~dwave.optimization.mathematical.broadcast_to`: equivalent function.

    .. versionadded:: 0.6.5
    """
    def __init__(self, ArraySymbol node, shape):
        cdef _Graph model = node.model

        cdef BroadcastToNode* ptr = model._graph.emplace_node[BroadcastToNode](
            node.array_ptr,
            as_cppshape(shape, nonnegative=False),
        )

        self.initialize_arraynode(model, ptr)

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        if len(predecessors) != 1:
            raise ValueError(f"{cls.__name__} must have exactly one predecessor")

        with zf.open(directory + "shape.json", "r") as f:
            return BroadcastTo(*predecessors, json.load(f))

    def _into_zipfile(self, zf, directory):
        encoder = json.JSONEncoder(separators=(',', ':'))
        zf.writestr(directory + "shape.json", encoder.encode(self.shape()))

    def state_size(self):
        """Broadcasting symbols are stateless"""
        return 0

_register(BroadcastTo, typeid(BroadcastToNode))


cdef class Concatenate(ArraySymbol):
    """Concatenate symbol.

    See Also:
        :func:`~dwave.optimization.mathematical.concatenate()` equivalent function.

    .. versionadded:: 0.4.3
    """
    def __init__(self, object inputs, Py_ssize_t axis = 0):
        if (not isinstance(inputs, collections.abc.Sequence) or
                not all(isinstance(arr, ArraySymbol) for arr in inputs)):
            raise TypeError("concatenate takes a sequence of array symbols")

        if len(inputs) < 1:
            raise ValueError("need at least one array symbol to concatenate")

        cdef _Graph model = inputs[0].model
        cdef vector[ArrayNode*] cppinputs

        for symbol in inputs:
            if symbol.model is not model:
                raise ValueError("all predecessors must be from the same model")
            cppinputs.push_back((<ArraySymbol?>symbol).array_ptr)

        self.ptr = model._graph.emplace_node[ConcatenateNode](cppinputs, axis)
        self.initialize_arraynode(model, self.ptr)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef ConcatenateNode* ptr = dynamic_cast_ptr[ConcatenateNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")

        cdef Concatenate m = Concatenate.__new__(Concatenate)
        m.ptr = ptr
        m.initialize_arraynode(symbol.model, ptr)
        return m

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        if len(predecessors) < 1:
            raise ValueError("Concatenate must have at least one predecessor")

        with zf.open(directory + "axis.json", "r") as f:
            return Concatenate(tuple(predecessors), axis=json.load(f))

    def _into_zipfile(self, zf, directory):
        encoder = json.JSONEncoder(separators=(',', ':'))
        zf.writestr(directory + "axis.json", encoder.encode(self.axis()))

    def axis(self):
        return self.ptr.axis()

    cdef ConcatenateNode* ptr

_register(Concatenate, typeid(ConcatenateNode))


cdef class Copy(ArraySymbol):
    """An array symbol that is a copy of another array symbol.

    See Also:
        :meth:`ArraySymbol.copy` Equivalent method.

    .. versionadded:: 0.5.1
    """
    def __init__(self, ArraySymbol node):
        cdef _Graph model = node.model

        cdef CopyNode* ptr = model._graph.emplace_node[CopyNode](node.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(Copy, typeid(CopyNode))


cdef class Put(ArraySymbol):
    """A symbol that replaces the specified elements in an array with given values.

    See Also:
        :func:`~dwave.optimization.mathematical.put`: equivalent function.

    .. versionadded:: 0.4.4
    """
    def __init__(self, ArraySymbol array, ArraySymbol indices, ArraySymbol values):
        cdef _Graph model = array.model

        if indices.model is not model or values.model is not model:
            raise ValueError(
                "array, indices, and values do not all share the same underlying model"
            )

        cdef PutNode* ptr = model._graph.emplace_node[PutNode](
            array.array_ptr, indices.array_ptr, values.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(Put, typeid(PutNode))


cdef class Reshape(ArraySymbol):
    """Reshaped symbol.

    See Also:
        :meth:`ArraySymbol.reshape() <dwave.optimization.model.ArraySymbol.reshape>`: equivalent method.

    .. versionadded:: 0.5.1
    """
    def __init__(self, ArraySymbol node, shape):
        cdef _Graph model = node.model

        self.ptr = model._graph.emplace_node[ReshapeNode](
            node.array_ptr,
            as_cppshape(shape, nonnegative=False),
            )

        self.initialize_arraynode(model, self.ptr)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef ReshapeNode* ptr = dynamic_cast_ptr[ReshapeNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")

        cdef Reshape m = Reshape.__new__(Reshape)
        m.ptr = ptr
        m.initialize_arraynode(symbol.model, ptr)
        return m

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        if len(predecessors) != 1:
            raise ValueError("Reshape must have exactly one predecessor")

        with zf.open(directory + "shape.json", "r") as f:
            return Reshape(*predecessors, json.load(f))

    def _into_zipfile(self, zf, directory):
        encoder = json.JSONEncoder(separators=(',', ':'))
        zf.writestr(directory + "shape.json", encoder.encode(self.shape()))

    cdef ReshapeNode* ptr

_register(Reshape, typeid(ReshapeNode))


cdef class Resize(ArraySymbol):
    """Resize symbol.

    See also:
        :func:`~dwave.optimization.mathematical.resize`: equivalent function.

        :meth:`ArraySymbol.resize() <dwave.optimization.model.ArraySymbol.resize>`: equivalent method.

    .. versionadded:: 0.6.4
    """
    def __init__(self, ArraySymbol symbol, shape, fill_value=None):
        cdef _Graph model = symbol.model

        if fill_value is None:
            self.ptr = model._graph.emplace_node[ResizeNode](
                symbol.array_ptr,
                as_cppshape(shape, nonnegative=True),
            )
        else:
            self.ptr = model._graph.emplace_node[ResizeNode](
                symbol.array_ptr,
                as_cppshape(shape, nonnegative=True),
                <double>fill_value,
            )

        self.initialize_arraynode(model, self.ptr)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef ResizeNode* ptr = dynamic_cast_ptr[ResizeNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")

        cdef Resize m = Resize.__new__(Resize)
        m.ptr = ptr
        m.initialize_arraynode(symbol.model, ptr)
        return m

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        if len(predecessors) != 1:
            raise ValueError("Resize must have exactly one predecessor")

        with zf.open(directory + "shape.json", "r") as f:
            shape = json.load(f)
        with zf.open(directory + "fill_value.npy", "r") as f:
            fill_value = np.load(f)

        return Resize(*predecessors, shape, fill_value)

    def _into_zipfile(self, zf, directory):
        encoder = json.JSONEncoder(separators=(',', ':'))
        zf.writestr(directory + "shape.json", encoder.encode(self.shape()))
        with zf.open(directory + "fill_value.npy", mode="w", force_zip64=True) as f:
            np.save(f, self.ptr.fill_value(), allow_pickle=False)

    cdef ResizeNode* ptr

_register(Resize, typeid(ResizeNode))


cdef class Size(ArraySymbol):
    def __init__(self, ArraySymbol array):
        cdef _Graph model = array.model

        cdef SizeNode* ptr = model._graph.emplace_node[SizeNode](array.array_ptr)
        self.initialize_arraynode(array.model, ptr)

_register(Size, typeid(SizeNode))

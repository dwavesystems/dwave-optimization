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

from cython.operator cimport typeid
from libcpp.vector cimport vector


from dwave.optimization._model cimport _Graph, _register, ArraySymbol, Symbol
from dwave.optimization.libcpp cimport dynamic_cast_ptr
from dwave.optimization.libcpp.graph cimport ArrayNode
from dwave.optimization.libcpp.nodes.set_routines cimport IsDisjointCoverNode, IsInNode


cdef class IsDisjointCover(ArraySymbol):
    """Tests whether the symbols are disjoint, set-like, and cover a set of integers

    See Also:

    .. versionadded:: 0.7.2
    """
    def __init__(self, object inputs, Py_ssize_t n):
        if (not isinstance(inputs, collections.abc.Sequence) or
                not all(isinstance(arr, ArraySymbol) for arr in inputs)):
            raise TypeError("disjoint_cover takes a sequence of array symbols")

        if len(inputs) < 1:
            raise ValueError("need at least one array symbol to to form a disjoint cover")

        cdef _Graph model = inputs[0].model
        cdef vector[ArrayNode*] cppinputs

        for symbol in inputs:
            if symbol.model is not model:
                raise ValueError("all predecessors must be from the same model")
            cppinputs.push_back((<ArraySymbol?>symbol).array_ptr)

        self.primary_set_size = n
        cdef IsDisjointCoverNode* ptr = model._graph.emplace_node[IsDisjointCoverNode](cppinputs, n)
        self.initialize_arraynode(model, ptr)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef IsDisjointCoverNode* ptr = dynamic_cast_ptr[IsDisjointCoverNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")
        cdef IsDisjointCover sym = cls.__new__(cls)
        sym.primary_set_size = ptr.primary_set_size()
        sym.initialize_arraynode(symbol.model, ptr)
        return sym

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        with zf.open(directory + "args.json", "r") as f:
            args = json.load(f)
        return cls(list(predecessors), args["primary_set_size"])

    def _into_zipfile(self, zf, directory):
        super()._into_zipfile(zf, directory)

        encoder = json.JSONEncoder(separators=(',', ':'))

        # get the non-array args
        args = dict()
        args.update(primary_set_size=int(self.primary_set_size))
        zf.writestr(directory + "args.json", encoder.encode(args))

    cdef Py_ssize_t primary_set_size

_register(IsDisjointCover, typeid(IsDisjointCoverNode))


cdef class IsIn(ArraySymbol):
    """Tests which values of one symbol are in another symbol.

    See Also:
        :func:`~dwave.optimization.mathematical.isin`: Instantiation and usage
        of this symbol.

        :class:`~dwave.optimization.symbols.Extract`,
        :class:`~dwave.optimization.symbols.Put`,
        :class:`~dwave.optimization.symbols.Where`

    .. versionadded:: 0.6.8
    """
    def __init__(self, ArraySymbol element, ArraySymbol test_elements):
        cdef _Graph model = element.model

        if element.model is not test_elements.model:
            raise ValueError("element and test_elements do not share the same underlying model")

        cdef IsInNode* ptr = model._graph.emplace_node[IsInNode](element.array_ptr, test_elements.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(IsIn, typeid(IsInNode))

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

import json

from cython.operator cimport typeid

from dwave.optimization._model cimport _Graph, _register, ArraySymbol, Symbol
from dwave.optimization.libcpp cimport dynamic_cast_ptr, get, holds_alternative
from dwave.optimization.libcpp.nodes.creation cimport ARangeNode

__all__ = ["ARange"]


ctypedef fused _start_type:
    ArraySymbol
    Py_ssize_t

ctypedef fused _stop_type:
    ArraySymbol
    Py_ssize_t

ctypedef fused _step_type:
    ArraySymbol
    Py_ssize_t


cdef class ARange(ArraySymbol):
    """Return evenly spaced integer values within a given interval.

    See Also:
        :func:`~dwave.optimization.mathematical.arange`: equivalent function.

    .. versionadded:: 0.5.2
    """
    def __init__(self, start, stop, step):
        ARange._init(self, start, stop, step)

    # Cython does not like fused types in the __init__, so we make a redundant one.
    # See https://github.com/cython/cython/issues/3758
    @staticmethod
    def _init(ARange self, _start_type start, _stop_type stop, _step_type step):
        # There are eight possible combinations of inputs, and unfortunately we
        # need to check them all
        if _start_type is Py_ssize_t and _stop_type is Py_ssize_t and _step_type is Py_ssize_t:
            raise ValueError(
                "ARange requires at least one symbol as an input. "
                f"Use model.constant(range({start}, {stop}, {step})) instead.")
        elif _start_type is Py_ssize_t and _stop_type is Py_ssize_t and _step_type is ArraySymbol:
            self.ptr = step.model._graph.emplace_node[ARangeNode](start, stop, step.array_ptr)
            self.initialize_arraynode(step.model, self.ptr)
        elif _start_type is Py_ssize_t and _stop_type is ArraySymbol and _step_type is Py_ssize_t:
            self.ptr = stop.model._graph.emplace_node[ARangeNode](start, stop.array_ptr, step)
            self.initialize_arraynode(stop.model, self.ptr)
        elif _start_type is Py_ssize_t and _stop_type is ArraySymbol and _step_type is ArraySymbol:
            if stop.model is not step.model:
                raise ValueError("stop and step do not share the same underlying model")
            self.ptr = stop.model._graph.emplace_node[ARangeNode](start, stop.array_ptr, step.array_ptr)
            self.initialize_arraynode(stop.model, self.ptr)
        elif _start_type is ArraySymbol and _stop_type is Py_ssize_t and _step_type is Py_ssize_t:
            self.ptr = start.model._graph.emplace_node[ARangeNode](start.array_ptr, stop, step)
            self.initialize_arraynode(start.model, self.ptr)
        elif _start_type is ArraySymbol and _stop_type is Py_ssize_t and _step_type is ArraySymbol:
            if start.model is not step.model:
                raise ValueError("start and step do not share the same underlying model")
            self.ptr = start.model._graph.emplace_node[ARangeNode](start.array_ptr, stop, step.array_ptr)
            self.initialize_arraynode(start.model, self.ptr)
        elif _start_type is ArraySymbol and _stop_type is ArraySymbol and _step_type is Py_ssize_t:
            if start.model is not stop.model:
                raise ValueError("start and stop do not share the same underlying model")
            self.ptr = start.model._graph.emplace_node[ARangeNode](start.array_ptr, stop.array_ptr, step)
            self.initialize_arraynode(start.model, self.ptr)
        elif _start_type is ArraySymbol and _stop_type is ArraySymbol and _step_type is ArraySymbol:
            if start.model is not stop.model or start.model is not step.model:
                raise ValueError("start, stop, and step do not share the same underlying model")
            self.ptr = start.model._graph.emplace_node[ARangeNode](start.array_ptr, stop.array_ptr, step.array_ptr)
            self.initialize_arraynode(start.model, self.ptr)   
        else:
            raise RuntimeError  # shouldn't be possible

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef ARangeNode* ptr = dynamic_cast_ptr[ARangeNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")
        cdef ARange sym = cls.__new__(cls)
        sym.ptr = ptr
        sym.initialize_arraynode(symbol.model, ptr)
        return sym

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        with zf.open(directory + "args.json", "r") as f:
            args = json.load(f)

        if len(predecessors) + len(args) != 3:
            raise RuntimeError("unexpected number of arguments")

        predecessors = list(predecessors)  # just in case it's not pop-able

        if "step" in args:
            step = args["step"]
        else:
            step = predecessors.pop()
        if "stop" in args:
            stop = args["stop"]
        else:
            stop = predecessors.pop()
        if "start" in args:
            start = args["start"]
        else:
            start = predecessors.pop()

        return cls(start, stop, step)


    def _into_zipfile(self, zf, directory):
        super()._into_zipfile(zf, directory)

        encoder = json.JSONEncoder(separators=(',', ':'))

        # get the non-array args
        args = dict()

        start = self.ptr.start()
        if holds_alternative[Py_ssize_t](start):
            args.update(start=int(get[Py_ssize_t](start)))

        stop = self.ptr.stop()
        if holds_alternative[Py_ssize_t](stop):
            args.update(stop=int(get[Py_ssize_t](stop)))

        step = self.ptr.step()
        if holds_alternative[Py_ssize_t](step):
            args.update(step=int(get[Py_ssize_t](step)))

        zf.writestr(directory + "args.json", encoder.encode(args))

    cdef ARangeNode* ptr

_register(ARange, typeid(ARangeNode))

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
import numbers

import numpy as np

from cython.operator cimport typeid
from libcpp.optional cimport optional
from libcpp.span cimport span
from libcpp.vector cimport vector

from dwave.optimization._model cimport _Graph, _register, ArraySymbol
from dwave.optimization.libcpp.graph cimport ArrayNode
from dwave.optimization.libcpp.nodes.reduce cimport (
    AllNode,
    AnyNode,
    MaxNode,
    MinNode,
    ProdNode,
    SumNode,
)
from dwave.optimization.utilities import _NoValue


# We need to be able to expose the available node types to the Python level,
# so unfortunately we need this enum and giant if-branches
cpdef enum _ReduceNodeType:
    All
    Any
    Max
    Min
    Prod
    Sum


class _ReduceSymbol(ArraySymbol):
    def __init_subclass__(cls, /, _ReduceNodeType node_type, default_initial=None):
        if node_type is _ReduceNodeType.All:
            _register(cls, typeid(AllNode))
        elif node_type is _ReduceNodeType.Any:
            _register(cls, typeid(AnyNode))
        elif node_type is _ReduceNodeType.Max:
            _register(cls, typeid(MaxNode))
        elif node_type is _ReduceNodeType.Min:
            _register(cls, typeid(MinNode))
        elif node_type is _ReduceNodeType.Prod:
            _register(cls, typeid(ProdNode))
        elif node_type is _ReduceNodeType.Sum:
            _register(cls, typeid(SumNode))
        else:
            raise RuntimeError(f"unexpected _ReduceNodeType: {<object>node_type!r}")

        cls._node_type = node_type
        cls._default_initial = default_initial

    def __init__(self, ArraySymbol array, *, axis=None, initial=_NoValue):
        cdef _Graph model = array.model

        # Convert the kwargs into something that can be understood by C++
        # The correctness will be checked at the C++ level
        cdef vector[Py_ssize_t] cppaxes
        if axis is None:
            pass  # empty vector means all axis are reduced
        elif isinstance(axis, numbers.Integral):
            cppaxes.push_back(<Py_ssize_t?>axis)
        else:
            cppaxes = list(axis)

        cdef optional[double] cppinitial
        if initial is None:
            pass  # nothing to change
        elif initial is _NoValue:
            if self._default_initial is not None:
                cppinitial = <double?>self._default_initial
        else:
            cppinitial = <double?>initial

        cdef _ReduceNodeType node_type = self._node_type
        cdef ArrayNode* ptr

        if node_type is _ReduceNodeType.All:
            ptr = model._graph.emplace_node[AllNode](array.array_ptr, cppaxes, cppinitial)
        elif node_type is _ReduceNodeType.Any:
            ptr = model._graph.emplace_node[AnyNode](array.array_ptr, cppaxes, cppinitial)
        elif node_type is _ReduceNodeType.Max:
            ptr = model._graph.emplace_node[MaxNode](array.array_ptr, cppaxes, cppinitial)
        elif node_type is _ReduceNodeType.Min:
            ptr = model._graph.emplace_node[MinNode](array.array_ptr, cppaxes, cppinitial)
        elif node_type is _ReduceNodeType.Prod:
            ptr = model._graph.emplace_node[ProdNode](array.array_ptr, cppaxes, cppinitial)
        elif node_type is _ReduceNodeType.Sum:
            ptr = model._graph.emplace_node[SumNode](array.array_ptr, cppaxes, cppinitial)
        else:
            raise RuntimeError(f"unexpected _ReduceNodeType: {<object>node_type!r}")

        (<ArraySymbol>self).initialize_arraynode(model, ptr)

    @property
    def initial(self):
        # Unfortunately we don't hold a pointer to our own "type" so we need
        # to do this giant if-branch every time
        cdef _ReduceNodeType node_type = self._node_type
        cdef ArrayNode* ptr = (<ArraySymbol>self).array_ptr

        cdef optional[double] initial

        if node_type is _ReduceNodeType.All:
            initial = (<AllNode*>ptr).initial
        elif node_type is _ReduceNodeType.Any:
            initial = (<AnyNode*>ptr).initial
        elif node_type is _ReduceNodeType.Max:
            initial = (<MaxNode*>ptr).initial
        elif node_type is _ReduceNodeType.Min:
            initial = (<MinNode*>ptr).initial
        elif node_type is _ReduceNodeType.Prod:
            initial = (<ProdNode*>ptr).initial
        elif node_type is _ReduceNodeType.Sum:
            initial = (<SumNode*>ptr).initial
        else:
            raise RuntimeError(f"unexpected _ReduceNodeType: {<object>node_type!r}")

        return initial.value() if initial.has_value() else None

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        kwargs = dict()

        # Test whether we have an initial value saved
        try:
            info = zf.getinfo(directory + "initial.npy")
        except KeyError:
            # Nothing to load
            pass
        else:
            with zf.open(info, "r") as f:
                kwargs.update(initial=np.load(f))

        # And whether we have axes
        try:
            info = zf.getinfo(directory + "axes.json")
        except KeyError:
            # Nothing to load
            pass
        else:
            with zf.open(info, "r") as f:
                kwargs.update(axis=json.load(f))

        return cls(*predecessors, **kwargs)

    def _into_zipfile(self, zf, directory):
        encoder = json.JSONEncoder(separators=(',', ':'))

        # If axes isn't empty (i.e. full reduction) save that info
        if (axes := self.axes):
            zf.writestr(directory + "axes.json", encoder.encode(self.axes()))

        # If we have an initial value, save that too
        if (init := self.initial) is not None:
            # NumPy serialization is overkill but it's type-safe
            with zf.open(directory + "initial.npy", mode="w", force_zip64=True) as f:
                np.save(f, init, allow_pickle=False)

    def axes(self):
        # Unfortunately we don't hold a pointer to our own "type" so we need
        # to do this giant if-branch every time
        cdef _ReduceNodeType node_type = self._node_type
        cdef ArrayNode* ptr = (<ArraySymbol>self).array_ptr

        cdef span[const Py_ssize_t] axes

        if node_type is _ReduceNodeType.All:
            axes = (<AllNode*>ptr).axes()
        elif node_type is _ReduceNodeType.Any:
            axes = (<AnyNode*>ptr).axes()
        elif node_type is _ReduceNodeType.Max:
            axes = (<MaxNode*>ptr).axes()
        elif node_type is _ReduceNodeType.Min:
            axes = (<MinNode*>ptr).axes()
        elif node_type is _ReduceNodeType.Prod:
            axes = (<ProdNode*>ptr).axes()
        elif node_type is _ReduceNodeType.Sum:
            axes = (<SumNode*>ptr).axes()
        else:
            raise RuntimeError(f"unexpected _ReduceNodeType: {<object>node_type!r}")

        return tuple(axes[i] for i in range(axes.size()))

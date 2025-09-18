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

from dwave.optimization._model cimport _Graph, _register, ArraySymbol, Symbol
from dwave.optimization.libcpp cimport dynamic_cast_ptr
from dwave.optimization.libcpp.graph cimport ArrayNode
from dwave.optimization.libcpp.nodes.reduce cimport *


cdef class All(ArraySymbol):
    """Tests whether all elements evaluate to True."""
    def __init__(self, ArraySymbol array):
        cdef _Graph model = array.model
        cdef AllNode* ptr = model._graph.emplace_node[AllNode](array.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(All, typeid(AllNode))


cdef class Any(ArraySymbol):
    """Tests whether any elements evaluate to True.

    Examples:
        This example checks the elements of a binary array.

        >>> from dwave.optimization.model import Model
        >>> model = Model()
        >>> model.states.resize(1)
        >>> x = model.constant([True, False, False])
        >>> a = x.any()
        >>> with model.lock():
        ...     assert a.state()

        >>> y = model.constant([False, False, False])
        >>> b = y.any()
        >>> with model.lock():
        ...     assert not b.state()

    .. versionadded:: 0.4.1
    """
    def __init__(self, ArraySymbol array):
        cdef _Graph model = array.model
        cdef AnyNode* ptr = model._graph.emplace_node[AnyNode](array.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(Any, typeid(AnyNode))


cdef class Max(ArraySymbol):
    """Maximum value in the elements of a symbol.

    See also:
        :meth:`~dwave.optimization.model.ArraySymbol.max()` equivalent method.
    """
    def __init__(self, ArraySymbol node, *, initial=None):
        cdef _Graph model = node.model

        if initial is None:
            self.ptr = model._graph.emplace_node[MaxNode](node.array_ptr)
        else:
            self.ptr = model._graph.emplace_node[MaxNode](node.array_ptr, <double?>initial)

        self.initialize_arraynode(model, self.ptr)

    @property
    def initial(self):
        """The initial value to the operation. Returns ``None`` if not provided.

        .. versionadded:: 0.6.4
        """
        return self.ptr.init.value() if self.ptr.init.has_value() else None

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef MaxNode* ptr = dynamic_cast_ptr[MaxNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")
        cdef Max sym = Max.__new__(Max)
        sym.ptr = ptr
        sym.initialize_arraynode(symbol.model, ptr)
        return sym

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        # Test whether we have any states saved.
        try:
            info = zf.getinfo(directory + "initial.npy")
        except KeyError:
            # No states, so nothing to load
            initial = None
        else:
            with zf.open(info, "r") as f:
                initial = np.load(f)

        return cls(*predecessors, initial=initial)

    def _into_zipfile(self, zf, directory):
        if (init := self.initial) is not None:
            # NumPy serialization is overkill but it's type-safe
            with zf.open(directory + "initial.npy", mode="w", force_zip64=True) as f:
                np.save(f, init, allow_pickle=False)

    def maybe_equals(self, other):
        maybe = super().maybe_equals(other)

        # mismatched initial values can turn uncertainty into a definite no
        if maybe == 1 and self.initial != other.initial:
            return 0

        return maybe

    cdef MaxNode* ptr

_register(Max, typeid(MaxNode))


cdef class Min(ArraySymbol):
    """Minimum value in the elements of a symbol.

    See also:
        :meth:`~dwave.optimization.model.ArraySymbol.min()` equivalent method.
    """
    def __init__(self, ArraySymbol node, *, initial=None):
        cdef _Graph model = node.model

        if initial is None:
            self.ptr = model._graph.emplace_node[MinNode](node.array_ptr)
        else:
            self.ptr = model._graph.emplace_node[MinNode](node.array_ptr, <double?>initial)

        self.initialize_arraynode(model, self.ptr)

    @property
    def initial(self):
        """The initial value to the operation. Returns ``None`` if not provided.

        .. versionadded:: 0.6.4
        """
        return self.ptr.init.value() if self.ptr.init.has_value() else None

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef MinNode* ptr = dynamic_cast_ptr[MinNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")
        cdef Min sym = Min.__new__(Min)
        sym.ptr = ptr
        sym.initialize_arraynode(symbol.model, ptr)
        return sym

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        # Test whether we have any states saved.
        try:
            info = zf.getinfo(directory + "initial.npy")
        except KeyError:
            # Nothing to load
            initial = None
        else:
            with zf.open(info, "r") as f:
                initial = np.load(f)

        return cls(*predecessors, initial=initial)

    def _into_zipfile(self, zf, directory):
        if (init := self.initial) is not None:
            # NumPy serialization is overkill but it's type-safe
            with zf.open(directory + "initial.npy", mode="w", force_zip64=True) as f:
                np.save(f, init, allow_pickle=False)

    def maybe_equals(self, other):
        maybe = super().maybe_equals(other)

        # mismatched initial values can turn uncertainty into a definite no
        if maybe == 1 and self.initial != other.initial:
            return 0

        return maybe

    cdef MinNode* ptr

_register(Min, typeid(MinNode))


cdef class PartialProd(ArraySymbol):
    """Multiply of the elements of a symbol along an axis.

    See also:
        :meth:`~dwave.optimization.model.ArraySymbol.prod()` equivalent method.

    .. versionadded:: 0.5.1
    """
    def __init__(self, ArraySymbol array, int axis, *, initial=None):
        cdef _Graph model = array.model

        if not isinstance(axis, numbers.Integral):
            raise TypeError("axis should be an int")

        if not (0 <= axis < array.ndim()):
            raise ValueError("axis should be 0 <= axis < ndim()")

        if initial is None:
            self.ptr = model._graph.emplace_node[PartialProdNode](
                array.array_ptr,
                <int?>axis,
            )
        else:
            self.ptr = model._graph.emplace_node[PartialProdNode](
                array.array_ptr,
                <int?>axis,
                <double?>initial,
            )
        self.initialize_arraynode(model, self.ptr)

    def axes(self):
        axes = self.ptr.axes()
        return tuple(axes[i] for i in range(axes.size()))

    @property
    def initial(self):
        """The initial value to the operation. Returns ``None`` if not provided.

        .. versionadded:: 0.6.4
        """
        return self.ptr.init.value() if self.ptr.init.has_value() else None

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef PartialProdNode* ptr = dynamic_cast_ptr[PartialProdNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")
        cdef PartialProd ps = PartialProd.__new__(PartialProd)
        ps.ptr = ptr
        ps.initialize_arraynode(symbol.model, ptr)
        return ps

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        if len(predecessors) != 1:
            raise ValueError("PartialProd must have exactly one predecessor")

        with zf.open(directory + "axes.json", "r") as f:
            axis = json.load(f)[0]

        try:
            info = zf.getinfo(directory + "initial.npy")
        except KeyError:
            # Nothing to load
            initial = None
        else:
            with zf.open(info, "r") as f:
                initial = np.load(f)

        return cls(*predecessors, axis=axis, initial=initial)

    def _into_zipfile(self, zf, directory):
        encoder = json.JSONEncoder(separators=(',', ':'))

        # Save information about the axes (always present)
        zf.writestr(directory + "axes.json", encoder.encode(self.axes()))

        # If we have an initial value, save that too
        if (init := self.initial) is not None:
            # NumPy serialization is overkill but it's type-safe
            with zf.open(directory + "initial.npy", mode="w", force_zip64=True) as f:
                np.save(f, init, allow_pickle=False)

    def maybe_equals(self, other):
        maybe = super().maybe_equals(other)

        # mismatched initial values can turn uncertainty into a definite no
        if maybe == 1 and self.initial != other.initial:
            return 0

        return maybe

    cdef PartialProdNode* ptr

_register(PartialProd, typeid(PartialProdNode))


cdef class PartialSum(ArraySymbol):
    """Sum of the elements of a symbol along an axis.

    See also:
        :meth:`~dwave.optimization.model.ArraySymbol.sum()` equivalent method.

    .. versionadded:: 0.4.1
    """
    def __init__(self, ArraySymbol array, int axis, *, initial=None):
        cdef _Graph model = array.model

        if not isinstance(axis, numbers.Integral):
            raise TypeError("axis should be an int")

        if not (0 <= axis < array.ndim()):
            raise ValueError("axis should be 0 <= axis < ndim()")

        if initial is None:
            self.ptr = model._graph.emplace_node[PartialSumNode](
                array.array_ptr,
                <int?>axis,
            )
        else:
            self.ptr = model._graph.emplace_node[PartialSumNode](
                array.array_ptr,
                <int?>axis,
                <double?>initial,
            )
        self.initialize_arraynode(model, self.ptr)

    def axes(self):
        axes = self.ptr.axes()
        return tuple(axes[i] for i in range(axes.size()))

    @property
    def initial(self):
        """The initial value to the operation. Returns ``None`` if not provided.

        .. versionadded:: 0.6.4
        """
        return self.ptr.init.value() if self.ptr.init.has_value() else None

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef PartialSumNode* ptr = dynamic_cast_ptr[PartialSumNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")
        cdef PartialSum ps = PartialSum.__new__(PartialSum)
        ps.ptr = ptr
        ps.initialize_arraynode(symbol.model, ptr)
        return ps

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        if len(predecessors) != 1:
            raise ValueError("PartialSum must have exactly one predecessor")

        with zf.open(directory + "axes.json", "r") as f:
            axis = json.load(f)[0]

        try:
            info = zf.getinfo(directory + "initial.npy")
        except KeyError:
            # Nothing to load
            initial = None
        else:
            with zf.open(info, "r") as f:
                initial = np.load(f)

        return cls(*predecessors, axis=axis, initial=initial)

    def _into_zipfile(self, zf, directory):
        encoder = json.JSONEncoder(separators=(',', ':'))

        # Save information about the axes (always present)
        zf.writestr(directory + "axes.json", encoder.encode(self.axes()))

        # If we have an initial value, save that too
        if (init := self.initial) is not None:
            # NumPy serialization is overkill but it's type-safe
            with zf.open(directory + "initial.npy", mode="w", force_zip64=True) as f:
                np.save(f, init, allow_pickle=False)

    def maybe_equals(self, other):
        maybe = super().maybe_equals(other)

        # mismatched initial values can turn uncertainty into a definite no
        if maybe == 1 and self.initial != other.initial:
            return 0

        return maybe

    cdef PartialSumNode* ptr

_register(PartialSum, typeid(PartialSumNode))


cdef class Prod(ArraySymbol):
    """Product of the elements of a symbol.

    See also:
        :meth:`~dwave.optimization.model.ArraySymbol.prod()` equivalent method.
    """
    def __init__(self, ArraySymbol node, *, initial=None):
        cdef _Graph model = node.model

        if initial is None:
            self.ptr = model._graph.emplace_node[ProdNode](node.array_ptr)
        else:
            self.ptr = model._graph.emplace_node[ProdNode](node.array_ptr, <double?>initial)

        self.initialize_arraynode(model, self.ptr)

    @property
    def initial(self):
        """The initial value to the operation. Returns ``None`` if not provided.

        .. versionadded:: 0.6.4
        """
        return self.ptr.init.value() if self.ptr.init.has_value() else None

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef ProdNode* ptr = dynamic_cast_ptr[ProdNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")
        cdef Prod sym = Prod.__new__(Prod)
        sym.ptr = ptr
        sym.initialize_arraynode(symbol.model, ptr)
        return sym

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        # Test whether we have any states saved.
        try:
            info = zf.getinfo(directory + "initial.npy")
        except KeyError:
            # Nothing to load
            initial = None
        else:
            with zf.open(info, "r") as f:
                initial = np.load(f)

        return cls(*predecessors, initial=initial)

    def _into_zipfile(self, zf, directory):
        if (init := self.initial) is not None:
            # NumPy serialization is overkill but it's type-safe
            with zf.open(directory + "initial.npy", mode="w", force_zip64=True) as f:
                np.save(f, init, allow_pickle=False)

    def maybe_equals(self, other):
        maybe = super().maybe_equals(other)

        # mismatched initial values can turn uncertainty into a definite no
        if maybe == 1 and self.initial != other.initial:
            return 0

        return maybe

    cdef ProdNode* ptr

_register(Prod, typeid(ProdNode))


cdef class Sum(ArraySymbol):
    """Sum of the elements of a symbol.

    See Also:
        :meth:`~dwave.optimization.model.ArraySymbol.sum()` equivalent method.
    """
    def __init__(self, ArraySymbol node, *, initial=None):
        cdef _Graph model = node.model

        if initial is None:
            self.ptr = model._graph.emplace_node[SumNode](node.array_ptr)
        else:
            self.ptr = model._graph.emplace_node[SumNode](node.array_ptr, <double?>initial)

        self.initialize_arraynode(model, self.ptr)

    @property
    def initial(self):
        """The initial value to the operation. Returns ``None`` if not provided.

        .. versionadded:: 0.6.4
        """
        return self.ptr.init.value() if self.ptr.init.has_value() else None

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef SumNode* ptr = dynamic_cast_ptr[SumNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")
        cdef Sum sym = Sum.__new__(Sum)
        sym.ptr = ptr
        sym.initialize_arraynode(symbol.model, ptr)
        return sym

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        # Test whether we have any states saved.
        try:
            info = zf.getinfo(directory + "initial.npy")
        except KeyError:
            # No states, so nothing to load
            initial = None
        else:
            with zf.open(info, "r") as f:
                initial = np.load(f)

        return cls(*predecessors, initial=initial)

    def _into_zipfile(self, zf, directory):
        if (init := self.initial) is not None:
            # NumPy serialization is overkill but it's type-safe
            with zf.open(directory + "initial.npy", mode="w", force_zip64=True) as f:
                np.save(f, init, allow_pickle=False)

    def maybe_equals(self, other):
        maybe = super().maybe_equals(other)

        # mismatched initial values can turn uncertainty into a definite no
        if maybe == 1 and self.initial != other.initial:
            return 0

        return maybe

    cdef SumNode* ptr

_register(Sum, typeid(SumNode))

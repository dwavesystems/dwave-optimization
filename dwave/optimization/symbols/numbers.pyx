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

import collections.abc
import numpy as np

from cython.operator cimport typeid
from libcpp.optional cimport nullopt, optional
from libcpp.utility cimport move
from libcpp.vector cimport vector

from dwave.optimization._model cimport _Graph, _register, ArraySymbol, Symbol
from dwave.optimization._utilities cimport as_cppshape
from dwave.optimization.libcpp cimport dynamic_cast_ptr
from dwave.optimization.libcpp.nodes.numbers cimport (
    NumberNode,
    BinaryNode,
    IntegerNode,
)
from dwave.optimization.states cimport States


# Convert the str operators "==", "<=", ">=" into their corresponding
# C++ objects.
cdef NumberNode.SumConstraint.Operator _parse_python_operator(str op) except *:
    if op == "==":
        return NumberNode.SumConstraint.Operator.Equal
    elif op == "<=":
        return NumberNode.SumConstraint.Operator.LessEqual
    elif op == ">=":
        return NumberNode.SumConstraint.Operator.GreaterEqual
    else:
        raise TypeError(f"Invalid sum constraint operator: {op!r}")


# Convert the user-defined sum constraints for NumberNode into the 
# corresponding C++ objects passed to NumberNode.
cdef vector[NumberNode.SumConstraint] _convert_python_sum_constraints(
         subject_to=None, axes_subject_to=None) except *:
    cdef vector[NumberNode.SumConstraint] output
    cdef optional[Py_ssize_t] cpp_axis = nullopt
    cdef vector[NumberNode.SumConstraint.Operator] cpp_ops
    cdef vector[double] cpp_bounds
    cdef double[:] mem

    if subject_to is not None:
        for constraint in subject_to:
            if not isinstance(constraint, tuple) or len(constraint) != 2:
                raise TypeError("A sum constraint on an entire number array must be"
                                " a tuple with two elements: `operator` and `bound`")

            py_ops, py_bounds = constraint
            cpp_axis = nullopt
            if not isinstance(py_ops, str):
                raise TypeError("Sum constraint operator on entire number array should be a str.")

            cpp_ops.resize(1)
            cpp_bounds.resize(1)
            cpp_ops[0] = _parse_python_operator(py_ops)
            cpp_bounds[0] = py_bounds
            output.push_back(NumberNode.SumConstraint(cpp_axis, move(cpp_ops), move(cpp_bounds)))

    if axes_subject_to is not None:
        for axis_constraint in axes_subject_to:
            if not isinstance(axis_constraint, tuple) or len(axis_constraint) != 3:
                raise TypeError("Each axis sum constraint must be a tuple with "
                                "three elements: axis, operator(s), bound(s)")

            axis, py_ops, py_bounds = axis_constraint
            if not isinstance(axis, int):
                raise TypeError("Constrained axis must be an int or None.")
            cpp_axis = <Py_ssize_t> axis

            if isinstance(py_ops, str):
                cpp_ops.resize(1)
                # One operator defined for all slices.
                cpp_ops[0] = _parse_python_operator(py_ops)
            elif isinstance(py_ops, collections.abc.Iterable):
                # Operator defined per slice.
                cpp_ops.reserve(len(py_ops))
                for op in py_ops:
                    cpp_ops.push_back(_parse_python_operator(op))
            else:
                raise TypeError("Axis sum constraint operator(s) should be str or an"
                                " iterable of str(s).")

            bound_array = np.asarray_chkfinite(py_bounds, dtype=np.double)
            if (bound_array.ndim <= 1):
                mem = bound_array.ravel()
                cpp_bounds.reserve(mem.shape[0])
                for i in range(mem.shape[0]):
                    cpp_bounds.push_back(mem[i])
            else:
                raise TypeError("Axis sum constraint bound(s) should be scalar or 1D-array.")

            output.push_back(NumberNode.SumConstraint(cpp_axis, move(cpp_ops), move(cpp_bounds)))

    return output

# Convert the C++ operators into their corresponding str
cdef str _parse_cpp_operators(NumberNode.SumConstraint.Operator op):
    if op == NumberNode.SumConstraint.Operator.Equal:
        return "=="
    elif op == NumberNode.SumConstraint.Operator.LessEqual:
        return "<="
    elif op == NumberNode.SumConstraint.Operator.GreaterEqual:
        return ">="
    else:
        raise TypeError(f"Invalid sum constraint operator: {op!r}")


cdef class BinaryVariable(ArraySymbol):
    """Binary decision-variable symbol.

    See Also:
        :meth:`~dwave.optimization.model.Model.binary`: Instantiation and
        usage of this symbol.
    """
    def __init__(self, _Graph model, shape=None, lower_bound=None, upper_bound=None,
                 subject_to=None, axes_subject_to=None):
        cdef vector[Py_ssize_t] cppshape = as_cppshape(
            tuple() if shape is None else shape
        )

        cdef optional[vector[double]] cpplower_bound = nullopt
        cdef optional[vector[double]] cppupper_bound = nullopt
        cdef vector[BinaryNode.SumConstraint] cpp_sum_constraints = _convert_python_sum_constraints(subject_to, axes_subject_to)
        cdef const double[:] mem

        if lower_bound is not None:
            # Allow users to input doubles
            lower_bound_arr = np.asarray_chkfinite(lower_bound,
                                                   dtype=np.double, order="C")
            # For lower bounds, round up
            lower_bound_arr = np.ceil(lower_bound_arr).astype(np.double)
            if (lower_bound_arr.ndim == 0) or (lower_bound_arr.shape == cppshape):
                if lower_bound_arr.size > 0:
                    mem = lower_bound_arr.ravel()
                    cpplower_bound.emplace(&mem[0], (&mem[-1]) + 1)
            else:
                raise ValueError("lower_bound should be None, scalar, or the same shape")

        if upper_bound is not None:
            # Allow users to input doubles
            upper_bound_arr = np.asarray_chkfinite(upper_bound,
                                                   dtype=np.double, order="C")
            # For upper bounds, round down
            upper_bound_arr = np.floor(upper_bound_arr).astype(np.double)
            if (upper_bound_arr.ndim == 0) or (upper_bound_arr.shape == cppshape):
                if upper_bound_arr.size > 0:
                    mem = upper_bound_arr.ravel()
                    cppupper_bound.emplace(&mem[0], (&mem[-1]) + 1)
            else:
                raise ValueError("upper bound should be None, scalar, or the same shape")

        self.ptr = model._graph.emplace_node[BinaryNode](
            cppshape, cpplower_bound, cppupper_bound, cpp_sum_constraints
        )
        self.initialize_arraynode(model, self.ptr)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef BinaryNode* ptr = dynamic_cast_ptr[BinaryNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")

        cdef BinaryVariable x = BinaryVariable.__new__(BinaryVariable)
        x.ptr = ptr
        x.initialize_arraynode(symbol.model, ptr)
        return x

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        if predecessors:
            raise ValueError(f"{cls.__name__} cannot have predecessors")

        with zf.open(directory + "shape.json", "r") as f:
            shape_info = json.load(f)

        # needs to be compatible with older versions
        try:
            info = zf.getinfo(directory + "lower_bound.npy")
        except KeyError:
            lower_bound = None
        else:
            with zf.open(info, "r") as f:
                lower_bound = np.load(f, allow_pickle=False)

        # needs to be compatible with older versions
        try:
            info = zf.getinfo(directory + "upper_bound.npy")
        except KeyError:
            upper_bound = None
        else:
            with zf.open(info, "r") as f:
                upper_bound = np.load(f, allow_pickle=False)

        # needs to be compatible with older versions
        try:
            info = zf.getinfo(directory + "sum_constraints.json")
        except KeyError:
            subject_to = None
            axes_subject_to = None
        else:
            with zf.open(info, "r") as f:
                subject_to = []
                axes_subject_to = []
                # Note that import is a list of lists, not a list of tuples.
                # Hence we convert to tuple. We could also support lists.
                for item in json.load(f):
                    if len(item) == 2:
                        # Inconvenient but `subject_to` expects scalars, not lists
                        subject_to.append((item[0][0], item[1][0]))
                    else:
                        axes_subject_to.append(tuple(item))

        return BinaryVariable(model,
                              shape=shape_info["shape"],
                              lower_bound=lower_bound,
                              upper_bound=upper_bound,
                              subject_to=subject_to,
                              axes_subject_to=axes_subject_to
                              )

    def _into_zipfile(self, zf, directory):
        shape_info = dict(shape=self.shape())
        encoder = json.JSONEncoder(separators=(',', ':'))
        zf.writestr(directory + "shape.json", encoder.encode(shape_info))

        lower_bound = self.lower_bound()
        # if all values in the array are the same, simply save a scalar
        if lower_bound.size and (lower_bound == lower_bound.flat[0]).all():
            lower_bound = lower_bound.flat[0]
        # NumPy serialization is overkill but it's type-safe
        with zf.open(directory + "lower_bound.npy", mode="w", force_zip64=True) as f:
            np.save(f, lower_bound, allow_pickle=False)

        upper_bound = self.upper_bound()
        # if all values in the array are the same, simply save a scalar
        if upper_bound.size and (upper_bound == upper_bound.flat[0]).all():
            upper_bound = upper_bound.flat[0]
        # NumPy serialization is overkill but it's type-safe
        with zf.open(directory + "upper_bound.npy", mode="w", force_zip64=True) as f:
            np.save(f, upper_bound, allow_pickle=False)

        sum_constraints = self.sum_constraints()
        if len(sum_constraints) > 0:
            # Using json here converts the tuples to lists
            zf.writestr(directory + "sum_constraints.json", encoder.encode(sum_constraints))

    def lower_bound(self):
        """Lower bound(s) of the symbol."""
        try:
            return np.asarray(self.ptr.lower_bound())
        except IndexError:
            pass
        return np.asarray([self.ptr.lower_bound(i) for i in range(self.size())]).reshape(self.shape())

    def set_state(self, Py_ssize_t index, state):
        r"""Set the state of the symbol.

        Args:
            index (int): Index of the state to set.
            state (\ |array-like|_\ ): Assignment of values for the state. The
                specified state must be binary array with the same shape as the
                symbol.

        Examples:
            This example sets two states for a :math:`2 \times 3`-sized
            binary symbol.

            >>> from dwave.optimization.model import Model
            >>> import numpy as np
            ...
            >>> model = Model()
            >>> x = model.binary((2, 3))
            >>> model.states.resize(2)
            >>> x.set_state(0, [[True, True, False], [False, True, False]])
            >>> print(np.equal(x.state(0), [[True, True, False], [False, True, False]]).all())
            True
            >>> x.set_state(1, [[False, True, False], [False, True, False]])
            >>> print(np.equal(x.state(1), [[False, True, False], [False, True, False]]).all())
            True
        """
        # Convert the state into something we can handle in C++.
        # This also does some type checking etc
        # We go ahead and do the translation to integer now
        cdef Py_ssize_t[:] arr = np.asarray(state, dtype=np.intp).flatten()

        # Reset our state, and check whether that's possible
        self.reset_state(index)

        # Convert to a vector. We could skip this copy if it ever becomes
        # a performance bottleneck
        cdef vector[double] items
        items.reserve(arr.size)
        cdef Py_ssize_t i
        for i in range(arr.size):
            items.push_back(arr[i])

        # The validity of the state is checked in C++
        self.ptr.initialize_state((<States>self.model.states)._states[index], move(items))

    def sum_constraints(self):
        """Sum constraints of Binary symbol as a list of tuples where each tuple
        is of the form: ([operator], [bound]) or (axis, [operator(s)], [bound(s)])."""
        cdef vector[NumberNode.SumConstraint] sum_constraints = self.ptr.sum_constraints()
        cdef optional[Py_ssize_t] axis

        output = []
        for i in range(sum_constraints.size()):
            constraint = &sum_constraints[i]
            axis = constraint.axis()
            py_ops = [_parse_cpp_operators(constraint.op(j)) for j in
                      range(constraint.num_operators())]
            py_bounds = [constraint.bound(j) for j in range(constraint.num_bounds())]
            # axis may be nullopt
            if axis.has_value():
                output.append((axis.value(), py_ops, py_bounds))
            else:
                output.append((py_ops, py_bounds))

        return output

    def upper_bound(self):
        """Upper bound(s) of the symbol."""
        try:
            return np.asarray(self.ptr.upper_bound())
        except IndexError:
            pass
        return np.asarray([self.ptr.upper_bound(i) for i in range(self.size())]).reshape(self.shape())

    # An observing pointer to the C++ BinaryNode
    cdef BinaryNode* ptr

_register(BinaryVariable, typeid(BinaryNode))


cdef class IntegerVariable(ArraySymbol):
    """Integer decision-variable symbol.

    See Also:
        :meth:`~dwave.optimization.model.Model.integer`: Instantiation and
        usage of this symbol.
    """
    def __init__(self, _Graph model, shape=None, lower_bound=None, upper_bound=None,
                 subject_to=None, axes_subject_to=None):
        cdef vector[Py_ssize_t] cppshape = as_cppshape(
            tuple() if shape is None else shape
        )

        cdef optional[vector[double]] cpplower_bound = nullopt
        cdef optional[vector[double]] cppupper_bound = nullopt
        cdef vector[IntegerNode.SumConstraint] cpp_sum_constraints = _convert_python_sum_constraints(subject_to, axes_subject_to)
        cdef const double[:] mem

        if lower_bound is not None:
            # Allow users to input doubles
            lower_bound_arr = np.asarray_chkfinite(lower_bound,
                                                   dtype=np.double, order="C")
            # For lower bounds, round up
            lower_bound_arr = np.ceil(lower_bound_arr).astype(np.double)
            if (lower_bound_arr.ndim == 0) or (lower_bound_arr.shape == cppshape):
                if lower_bound_arr.size > 0:
                    mem = lower_bound_arr.ravel()
                    cpplower_bound.emplace(&mem[0], (&mem[-1]) + 1)
            else:
                raise ValueError("lower_bound should be None, scalar, or the same shape")

        if upper_bound is not None:
            # Allow users to input doubles
            upper_bound_arr = np.asarray_chkfinite(upper_bound,
                                                   dtype=np.double, order="C")
            # For upper bounds, round down
            upper_bound_arr = np.floor(upper_bound_arr).astype(np.double)
            if (upper_bound_arr.ndim == 0) or (upper_bound_arr.shape == cppshape):
                if upper_bound_arr.size > 0:
                    mem = upper_bound_arr.ravel()
                    cppupper_bound.emplace(&mem[0], (&mem[-1]) + 1)
            else:
                raise ValueError("upper bound should be None, scalar, or the same shape")

        self.ptr = model._graph.emplace_node[IntegerNode](
            cppshape, cpplower_bound, cppupper_bound, cpp_sum_constraints
        )
        self.initialize_arraynode(model, self.ptr)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef IntegerNode* ptr = dynamic_cast_ptr[IntegerNode](symbol.node_ptr)
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")

        cdef IntegerVariable x = IntegerVariable.__new__(IntegerVariable)
        x.ptr = ptr
        x.initialize_arraynode(symbol.model, ptr)
        return x

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        if predecessors:
            raise ValueError(f"{cls.__name__} cannot have predecessors")

        with zf.open(directory + "shape.json", "r") as f:
            shape_info = json.load(f)

        # needs to be compatible with older versions
        try:
            info = zf.getinfo(directory + "lower_bound.npy")
        except KeyError:
            lower_bound = shape_info["lb"]
        else:
            with zf.open(info, "r") as f:
                lower_bound = np.load(f, allow_pickle=False)

        # needs to be compatible with older versions
        try:
            info = zf.getinfo(directory + "upper_bound.npy")
        except KeyError:
            upper_bound = shape_info["ub"]
        else:
            with zf.open(info, "r") as f:
                upper_bound = np.load(f, allow_pickle=False)

        # needs to be compatible with older versions
        try:
            info = zf.getinfo(directory + "sum_constraints.json")
        except KeyError:
            subject_to = None
            axes_subject_to = None
        else:
            with zf.open(info, "r") as f:
                subject_to = []
                axes_subject_to = []
                # Note that import is a list of lists, not a list of tuples.
                # Hence we convert to tuple. We could also support lists.
                for item in json.load(f):
                    if len(item) == 2:
                        # Inconvenient but `subject_to` expects scalars, not lists
                        subject_to.append((item[0][0], item[1][0]))
                    else:
                        axes_subject_to.append(tuple(item))

        return IntegerVariable(model,
                               shape=shape_info["shape"],
                               lower_bound=lower_bound,
                               upper_bound=upper_bound,
                               subject_to=subject_to,
                               axes_subject_to=axes_subject_to
                               )

    def _into_zipfile(self, zf, directory):
        shape_info = dict(shape=self.shape())
        lower_bound = self.lower_bound()
        upper_bound = self.upper_bound()

        # This is for backward compatiblity and should be ignored
        if lower_bound.size == 1 and upper_bound.size == 1:
            shape_info["lb"] = lower_bound.item()
            shape_info["ub"] = upper_bound.item()

        encoder = json.JSONEncoder(separators=(',', ':'))
        zf.writestr(directory + "shape.json", encoder.encode(shape_info))

        # if all values in the array are the same, simply save a scalar
        if lower_bound.size and (lower_bound == lower_bound.flat[0]).all():
            lower_bound = lower_bound.flat[0]
        # NumPy serialization is overkill but it's type-safe
        with zf.open(directory + "lower_bound.npy", mode="w", force_zip64=True) as f:
            np.save(f, lower_bound, allow_pickle=False)

        # if all values in the array are the same, simply save a scalar
        if upper_bound.size and (upper_bound == upper_bound.flat[0]).all():
            upper_bound = upper_bound.flat[0]
        # NumPy serialization is overkill but it's type-safe
        with zf.open(directory + "upper_bound.npy", mode="w", force_zip64=True) as f:
            np.save(f, upper_bound, allow_pickle=False)

        sum_constraints = self.sum_constraints()
        if len(sum_constraints) > 0:
            # Using json here converts the tuples to lists
            zf.writestr(directory + "sum_constraints.json", encoder.encode(sum_constraints))

    def lower_bound(self):
        """Lower bound(s) of the symbol."""
        try:
            return np.asarray(self.ptr.lower_bound())
        except IndexError:
            pass
        return np.asarray([self.ptr.lower_bound(i) for i in range(self.size())]).reshape(self.shape())

    def set_state(self, Py_ssize_t index, state):
        r"""Set the state of the integer symbol.

        Args:
            index (int):
                Index of the state to set.
            state (\ |array-like|_\ ):
                Assignment of values for the state. The specified state must
                have the same shape as the symbol.

        Examples:
            This example successfully sets one state for a :math:`2 \times
            2`-sized integer symbol.

            >>> from dwave.optimization.model import Model
            >>> import numpy as np
            ...
            >>> model = Model()
            >>> x = model.integer((2, 2), lower_bound=2, upper_bound=[[3,4], [2, 5]])
            >>> model.states.resize(1)
            >>> x.set_state(0, [[3, 4], [2, 3]])
            >>> print(np.equal(x.state(0), [[3, 4], [2, 3]]).all())
            True
        """
        # Convert the state into something we can handle in C++.
        # This also does some type checking etc
        # We go ahead and do the translation to integer now
        cdef Py_ssize_t[:] arr = np.asarray(state, dtype=np.intp).flatten()

        # Reset our state, and check whether that's possible
        self.reset_state(index)

        # Convert to a vector. We could skip this copy if it ever becomes
        # a performance bottleneck
        cdef vector[double] items
        items.reserve(arr.size)
        cdef Py_ssize_t i
        for i in range(arr.size):
            items.push_back(arr[i])

        # The validity of the state is checked in C++
        self.ptr.initialize_state((<States>self.model.states)._states[index], move(items))

    def sum_constraints(self):
        """Sum constraints of Integer symbol as a list of tuples where each tuple
        is of the form: ([operator], [bound]) or (axis, [operator(s)], [bound(s)])."""
        cdef vector[NumberNode.SumConstraint] sum_constraints = self.ptr.sum_constraints()
        cdef optional[Py_ssize_t] axis

        output = []
        for i in range(sum_constraints.size()):
            constraint = &sum_constraints[i]
            axis = constraint.axis()
            py_ops = [_parse_cpp_operators(constraint.op(j)) for j in
                      range(constraint.num_operators())]
            py_bounds = [constraint.bound(j) for j in range(constraint.num_bounds())]
            # axis may be nullopt
            if axis.has_value():
                output.append((axis.value(), py_ops, py_bounds))
            else:
                output.append((py_ops, py_bounds))

        return output

    def upper_bound(self):
        """Upper bound(s) of the symbol."""
        try:
            return np.asarray(self.ptr.upper_bound())
        except IndexError:
            pass
        return np.asarray([self.ptr.upper_bound(i) for i in range(self.size())]).reshape(self.shape())

    # An observing pointer to the C++ IntegerNode
    cdef IntegerNode* ptr

_register(IntegerVariable, typeid(IntegerNode))

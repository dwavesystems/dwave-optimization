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
from libcpp.vector cimport vector

from dwave.optimization.expression import Expression
from dwave.optimization._model cimport _Graph, _register, ArraySymbol, Symbol, symbol_from_ptr
from dwave.optimization.libcpp cimport dynamic_cast_ptr, get, holds_alternative
from dwave.optimization.libcpp.graph cimport ArrayNode
from dwave.optimization.libcpp.nodes.lambda_ cimport AccumulateZipNode


cdef class AccumulateZip(ArraySymbol):
    """Accumulates element-wise operations over operand symbols.

    .. note:: This symbol is used by direct instantiation, as shown in the
        example below.

    Performs an element-wise accumulate operation, as specified by a
    :class:`~dwave.optimization.expression.Expression`, along one or more array
    operands. For each output index, the operation inputs one value from each
    array symbol and the result of the previous operation, or, for the first
    index, the ``initial`` parameter.

    This symbol expands the :doc:`NumPy <numpy:index>`
    :meth:`~numpy.ufunc.accumulate` function by enabling the accumulate
    operation to accept an arbitrary number of array inputs and "zipping" them
    together. For a given expression ``expr``, predecessor array symbols ``A``,
    ``B``, ``C``, and an initial value, ``init``, this symbol is equivalent to
    the pseudocode:

    .. code-block::

        r = [0] * len(A)
        t = init
        for args in zip(A, B, C):
            t = expr(t, *args)
            r[i] = t
        return r

    Args:
        expression (:class:`~dwave.optimization.expression.Expression`):
            An operation to accumulate, specified as an expression. Its first
            argument takes the value of the ``initial`` parameter for index zero
            of its element-wise operation and then, for the remaining array
            elements, the output of the previous index. The expression's next
            arguments are assigned to the symbols specified by the ``operands``
            parameter.
        operands (tuple[:class:`~dwave.optimization.model.Symbol`, ...]):
            Operands to the accumulate operation as an iterable of 1D-array
            symbols. There should be one fewer operands than arguments of the
            expression.
        initial (float, optional):
            Start value of the accumulate operation. Used to set the first
            argument of the expression on the very first iteration (index zero).

    Examples:
        This example directly adds an :class:`.AccumulateZip` symbol to a model
        that calculates the cumulative value of the expression
        :math:`-v_0*(v_1 + v_2)`, where, for each iteration, :math:`v_1, v_2`
        are sequential elements of two
        :class:`~dwave.optimization.symbols.IntegerVariable` symbols and
        :math:`v_0` is the result from the previous elements or, for the first
        elements, defined by the configured ``initial`` value.

        For example, for inputs
        :math:`v_1, v_2 = [1, 1, 0, 1, 1], [0, 0, 2, 2, 2]` and an initial
        value, :math:`v_0[0]` of 2, the first element of the
        :class:`.AccumulateZip` symbol  is :math:`-2 * (1 + 0) = -2`, the 2nd is
        therefore :math:`-(-2) * (1 + 0) = 2`, the 3rd is
        :math:`-2 * (0 + 2) = -4`, etc.

        >>> from dwave.optimization import expression, Model
        >>> from dwave.optimization.symbols import AccumulateZip
        ...
        >>> model = Model()
        >>> i1 = model.integer(5, lower_bound=-10, upper_bound=15)
        >>> i2 = model.integer(5, upper_bound=10)
        >>> j = AccumulateZip(expression(lambda v0, v1, v2: -v0*(v1 + v2)), (i1, i2), initial=2)
        >>> with model.lock():
        ...     model.states.resize(1)
        ...     i1.set_state(0,[1, 1, 0, 1, 1])
        ...     i2.set_state(0, [0, 0, 2, 2, 2])
        ...     print(j.state(0))
        [ -2.   2.  -4.  12. -36.]

    See Also:
        :meth:`~numpy.ufunc.accumulate`: :doc:`NumPy <numpy:index>` function

    .. versionadded:: 0.6.4
    """

    def __init__(self, expression, operands, object initial = 0):
        if not isinstance(expression, Expression):
            raise TypeError("expression must be of type Expression")

        # Get the underlying Graph representing our expression, and make sure it matches our
        # expectations.
        # Much of this is re-checked at the C++ level, but we can raise nicer errors here
        cdef _Graph expr = expression._model
        if expr is None:
            raise TypeError("expression has not been initialized")  # user bypassed __init__()
        for symbol in expr.iter_decisions():
            raise ValueError(
                "expression must only have inputs and constants as roots, "
                f"received an expression with {type(symbol).__name__} as a root"
            )

        if len(operands) == 0:
            raise ValueError("operands must contain at least one array symbol")
        if expr.num_inputs() != len(operands) + 1:
            raise ValueError("expression must have exactly one more input than number of operands")

        # Convert the operands into something we can pass to C++, and check that they all share a
        # model while we're at it
        cdef _Graph model = operands[0].model
        cdef vector[ArrayNode*] cppoperands
        for symbol in operands:
            if symbol.model != model:
                raise ValueError("all predecessors must be from the same model")
            cppoperands.push_back((<ArraySymbol?>symbol).array_ptr)

        # Convert initial into something we can handle
        if isinstance(initial, ArraySymbol):
            self.ptr = model._graph.emplace_node[AccumulateZipNode](
                expr._owning_ptr, cppoperands, (<ArraySymbol>initial).array_ptr,
            )
        else:
            try:
                <double?>(initial)
            except TypeError:
                raise TypeError(
                    "expected type of `initial` to be either an int, float or an ArraySymbol, got "
                    f"{type(initial)}"
                )

            self.ptr = model._graph.emplace_node[AccumulateZipNode](
                expr._owning_ptr, cppoperands, <double>initial,
            )

        self.initialize_arraynode(model, self.ptr)

    @classmethod
    def _from_symbol(cls, Symbol symbol):
        cdef AccumulateZipNode* ptr = dynamic_cast_ptr[AccumulateZipNode](
            symbol.node_ptr
        )
        if not ptr:
            raise TypeError(f"given symbol cannot construct a {cls.__name__}")
        cdef AccumulateZip x = AccumulateZip.__new__(AccumulateZip)
        x.ptr = ptr
        x.initialize_arraynode(symbol.model, ptr)
        return x

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        from dwave.optimization import Model  # avoid circular import

        expression = Expression.__new__(Expression)  # bypass the __init__
        with zf.open(directory + "expression.nl", "r") as f:
            expression._model = Model.from_file(f)
            expression._model.lock()
        with zf.open(directory + "initial.json", "r") as f:
            initial_info = json.load(f)

        if initial_info["type"] == "double":
            initial = initial_info["value"]
            operands = predecessors
        elif initial_info["type"] == "node":
            initial = predecessors[0]
            operands = predecessors[1:]
        else:
            raise ValueError("unexpected type for initial value")

        return AccumulateZip(expression, operands, initial=initial)

    def initial(self):
        if holds_alternative["ArrayNode*"](self.ptr.initial):
            return symbol_from_ptr(self.model, get["ArrayNode*"](self.ptr.initial))
        else:
            return get[double](self.ptr.initial)

    def _into_zipfile(self, zf, directory):
        """Store a AccumulateZip symbol as a compressed file.

        Args:
            zf:
                File pointer to a compressed file to store the
                disjoint-list symbol. Strings are interpreted as a
                file name.
            directory:
                Directory where the file is located.
        Returns:
            A compressed file.

        See also:
            :meth:`._from_zipfile`
        """
        # Create a model from the shared_ptr held by the node. We need to be careful
        # not to modify it! That leads to undefined behavior.
        # There is also an asymmetry here where we serialize as a _Graph and then
        # load as Model. Right now that's OK, but we need to formalize that process.
        cdef _Graph expr = _Graph.from_shared_ptr(self.ptr.expression_ptr())
        assert expr._graph.topologically_sorted()
        assert expr._lock_count > 0  
        with zf.open(directory + "expression.nl", mode="w") as f:
            expr.into_file(f)

        # Now save information about the initial state
        encoder = json.JSONEncoder(separators=(',', ':'))
        if holds_alternative["ArrayNode*"](self.ptr.initial):
            initial_info = dict(
                type="node",
                value=get["ArrayNode*"](self.ptr.initial).topological_index()
            )
        else:
            initial_info = dict(type="double", value=get[double](self.ptr.initial))
        zf.writestr(directory + "initial.json", encoder.encode(initial_info))

    cdef AccumulateZipNode* ptr

_register(AccumulateZip, typeid(AccumulateZipNode))

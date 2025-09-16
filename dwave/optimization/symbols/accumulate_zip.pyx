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
    """Using a supplied :class:`~dwave.optimization.model.Expression`, perform
    an element-wise accumulate operation along one or more array operands. The
    accumulate operation (represented by the ``Expression``) takes as input one
    value from each of the operand arrays, as well as the result of the
    previously computed operation, and computes a new value at the next output
    index.

    This takes inspiration from
    `numpy.ufunc.accumulate <https://numpy.org/doc/2.1/reference/generated/numpy.ufunc.accumulate.html>`_ 
    but is different in that the accumulate operation can take an arbitrary
    number of arguments, instead of always two. These arguments come from
    "zipping" the supplied predecessor arrays together.

    Thus if we are given an expression `expr`, predecessor arrays `A`, `B`,
    `C`, and an initial value `init`, this node is equivalent to the
    pseudocode:

    .. code-block::

        r = [0] * len(A)
        t = init
        for args in zip(A, B, C):
            t = expr(t, *args)
            r[i] = t
        return r

    Args:
        expression:
            An :class:`~dwave.optimization.model.Expression` representing the
            accumulate operation. The first input on the expression will be given
            the previous output of the operation at each iteration over the
            values of the operands.
        operands:
            A list of the 1d-array symbols that will be the operands to the
            accumulate. There should be one fewer operands than inputs on the
            expression.
        initial (optional):
            A float representing the value used to start the accumulate. This
            will be used to set the last input of the expression on the very
            first iteration.

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
                f"recieved an expression with {type(symbol).__name__} as a root"
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

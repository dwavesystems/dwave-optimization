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

from cython.operator cimport typeid

from dwave.optimization._model cimport _Graph, _register, ArraySymbol
from dwave.optimization.libcpp cimport dynamic_cast_ptr
from dwave.optimization.libcpp.graph cimport ArrayNode
from dwave.optimization.libcpp.nodes.unaryop cimport (
    AbsoluteNode,
    CosNode,
    ExpNode,
    ExpitNode,
    LogNode,
    LogicalNode,
    NegativeNode,
    NotNode,
    RintNode,
    SinNode,
    SquareNode,
    SqrtNode,
    TanhNode,
    UnaryExpressionNode,
)
from dwave.optimization.expression import Expression
from dwave.optimization.model import Model

# We need to be able to expose the available node types to the Python level,
# so unfortunately we need this enum and giant if-branches
cpdef enum _UnaryOpNodeType:
    Absolute
    Cos
    Exp
    Expit
    Log
    Logical
    Negative
    Not
    Rint
    Sin
    Square
    SquareRoot
    Tanh


class _UnaryOpSymbol(ArraySymbol):
    def __init_subclass__(cls, /, _UnaryOpNodeType node_type):
        if node_type is _UnaryOpNodeType.Absolute:
            _register(cls, typeid(AbsoluteNode))
        elif node_type is _UnaryOpNodeType.Cos:
            _register(cls, typeid(CosNode))
        elif node_type is _UnaryOpNodeType.Exp:
            _register(cls, typeid(ExpNode))
        elif node_type is _UnaryOpNodeType.Expit:
            _register(cls, typeid(ExpitNode))
        elif node_type is _UnaryOpNodeType.Log:
            _register(cls, typeid(LogNode))
        elif node_type is _UnaryOpNodeType.Logical:
            _register(cls, typeid(LogicalNode))
        elif node_type is _UnaryOpNodeType.Negative:
            _register(cls, typeid(NegativeNode))
        elif node_type is _UnaryOpNodeType.Not:
            _register(cls, typeid(NotNode))
        elif node_type is _UnaryOpNodeType.Rint:
            _register(cls, typeid(RintNode))
        elif node_type is _UnaryOpNodeType.Sin:
            _register(cls, typeid(SinNode))
        elif node_type is _UnaryOpNodeType.Square:
            _register(cls, typeid(SquareNode))
        elif node_type is _UnaryOpNodeType.SquareRoot:
            _register(cls, typeid(SqrtNode))
        elif node_type is _UnaryOpNodeType.Tanh:
            _register(cls, typeid(TanhNode))
        else:
            raise RuntimeError(f"unexpected _UnaryOpNodeType: {<object>node_type!r}")

        cls._node_type = node_type

    def __init__(self, ArraySymbol x):
        cdef _Graph model = x.model
        cdef _UnaryOpNodeType node_type = self._node_type

        cdef ArrayNode* ptr

        if node_type is _UnaryOpNodeType.Absolute:
            ptr = model._graph.emplace_node[AbsoluteNode](x.array_ptr)
        elif node_type is _UnaryOpNodeType.Cos:
            ptr = model._graph.emplace_node[CosNode](x.array_ptr)
        elif node_type is _UnaryOpNodeType.Exp:
            ptr = model._graph.emplace_node[ExpNode](x.array_ptr)
        elif node_type is _UnaryOpNodeType.Expit:
            ptr = model._graph.emplace_node[ExpitNode](x.array_ptr)
        elif node_type is _UnaryOpNodeType.Log:
            ptr = model._graph.emplace_node[LogNode](x.array_ptr)
        elif node_type is _UnaryOpNodeType.Logical:
            ptr = model._graph.emplace_node[LogicalNode](x.array_ptr)
        elif node_type is _UnaryOpNodeType.Negative:
            ptr = model._graph.emplace_node[NegativeNode](x.array_ptr)
        elif node_type is _UnaryOpNodeType.Not:
            ptr = model._graph.emplace_node[NotNode](x.array_ptr)
        elif node_type is _UnaryOpNodeType.Rint:
            ptr = model._graph.emplace_node[RintNode](x.array_ptr)
        elif node_type is _UnaryOpNodeType.Sin:
            ptr = model._graph.emplace_node[SinNode](x.array_ptr)
        elif node_type is _UnaryOpNodeType.Square:
            ptr = model._graph.emplace_node[SquareNode](x.array_ptr)
        elif node_type is _UnaryOpNodeType.SquareRoot:
            ptr = model._graph.emplace_node[SqrtNode](x.array_ptr)
        elif node_type is _UnaryOpNodeType.Tanh:
            ptr = model._graph.emplace_node[TanhNode](x.array_ptr)
        else:
            raise RuntimeError(f"unexpected _UnaryOpNodeType: {<object>node_type!r}")

        (<ArraySymbol>self).initialize_arraynode(model, ptr)


cdef class UnaryExpressionOp(ArraySymbol):
    """Performs arbitrary operation element-wise over operand symbol.

    .. note:: This symbol is used by direct instantiation, as shown in the
        example below.

    Performs an element-wise operation on a single operand, as specified by a
    :class:`~dwave.optimization.expression.Expression`.

    Args:
        expression (:class:`~dwave.optimization.expression.Expression`):
            An unary (single argument) operation perform on the input array,
            specified as an expression.
        operand (:class:`~dwave.optimization.model.Symbol`):
            Operand to the element-wise operation.

    Examples:
        This example directly adds an :class:`.UnaryExpressionOp` symbol to a model
        that multiplies the operand by two and adds one.

        >>> from dwave.optimization import expression, Model
        >>> from dwave.optimization.symbols import UnaryExpressionOp
        >>>
        >>> model = Model()

        >>> @expression
        ... def expr(x):
        ...     return 2 * x + 1
        >>>
        >>> a = model.constant([1, 2, 3])
        >>> op_a = UnaryExpressionOp(expr, a)
        >>> with model.lock():
        ...     model.states.resize(1)
        ...     print(op_a.state(0))
        [3. 5. 7.]

    .. versionadded:: 0.8.1
    """

    def __init__(self, expression, operand):
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

        if expr.num_inputs() != 1:
            raise ValueError("expression must have exactly one input")

        # Convert the operand into something we can pass to C++
        cdef _Graph model = operand.model
        cdef ArrayNode* cppoperand = (<ArraySymbol?>operand).array_ptr

        self.ptr = model._graph.emplace_node[UnaryExpressionNode](expr._owning_ptr, cppoperand)

        self.initialize_arraynode(model, self.ptr)

    @classmethod
    def _from_ptr(cls, model, capsule):
        cdef UnaryExpressionOp sym = super()._from_ptr(model, capsule)

        sym.ptr = dynamic_cast_ptr[UnaryExpressionNode](sym.node_ptr)
        if not sym.ptr:
            raise TypeError(f"given pointer cannot construct a UnaryExpressionOp")

        return sym

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        from dwave.optimization import Model  # avoid circular import

        expression = Expression.__new__(Expression)  # bypass the __init__
        with zf.open(directory + "expression.nl", "r") as f:
            expression._model = Model.from_file(f)
            expression._model.lock()

        return UnaryExpressionOp(expression, predecessors[0])

    def _into_zipfile(self, zf, directory):
        """Store a UnaryExpressionOp symbol as a compressed file.

        Args:
            zf:
                File pointer to a compressed file to store the
                UnaryExpressionOp symbol. Strings are interpreted as a
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
        cdef _Graph expr = _Graph.from_shared_ptr(Model, self.ptr.expression_ptr())
        assert expr._graph.topologically_sorted()
        assert expr._lock_count > 0  
        with zf.open(directory + "expression.nl", mode="w") as f:
            expr.into_file(f)

    cdef UnaryExpressionNode* ptr

_register(UnaryExpressionOp, typeid(UnaryExpressionNode))

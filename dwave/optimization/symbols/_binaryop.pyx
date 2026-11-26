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
from dwave.optimization.libcpp.nodes.binaryop cimport (
    AddNode,
    AndNode,
    BinaryExpressionNode,
    DivideNode,
    EqualNode,
    LessEqualNode,
    MaximumNode,
    MinimumNode,
    ModulusNode,
    MultiplyNode,
    OrNode,
    SubtractNode,
    XorNode,
)
from dwave.optimization.expression import Expression
from dwave.optimization.model import Model

# We need to be able to expose the available node types to the Python level,
# so unfortunately we need this enum and giant if-branches
cpdef enum _BinaryOpNodeType:
    Add
    And
    Divide
    Equal
    LessEqual
    Maximum
    Minimum
    Modulus
    Multiply
    Or
    Subtract
    Xor


class _BinaryOpSymbol(ArraySymbol):
    def __init_subclass__(cls, /, _BinaryOpNodeType node_type):
        if node_type is _BinaryOpNodeType.Add:
            _register(cls, typeid(AddNode))
        elif node_type is _BinaryOpNodeType.And:
            _register(cls, typeid(AndNode))
        elif node_type is _BinaryOpNodeType.Divide:
            _register(cls, typeid(DivideNode))
        elif node_type is _BinaryOpNodeType.Equal:
            _register(cls, typeid(EqualNode))
        elif node_type is _BinaryOpNodeType.LessEqual:
            _register(cls, typeid(LessEqualNode))
        elif node_type is _BinaryOpNodeType.Maximum:
            _register(cls, typeid(MaximumNode))
        elif node_type is _BinaryOpNodeType.Minimum:
            _register(cls, typeid(MinimumNode))
        elif node_type is _BinaryOpNodeType.Modulus:
            _register(cls, typeid(ModulusNode))
        elif node_type is _BinaryOpNodeType.Multiply:
            _register(cls, typeid(MultiplyNode))
        elif node_type is _BinaryOpNodeType.Or:
            _register(cls, typeid(OrNode))
        elif node_type is _BinaryOpNodeType.Subtract:
            _register(cls, typeid(SubtractNode))
        elif node_type is _BinaryOpNodeType.Xor:
            _register(cls, typeid(XorNode))
        else:
            raise RuntimeError(f"unexpected _BinaryOpNodeType: {<object>node_type!r}")

        cls._node_type = node_type

    def __init__(self, ArraySymbol lhs, ArraySymbol rhs):
        if lhs.model is not rhs.model:
            raise ValueError("lhs and rhs do not share the same underlying model")

        cdef _Graph model = lhs.model
        cdef _BinaryOpNodeType node_type = self._node_type

        cdef ArrayNode* ptr

        if node_type is _BinaryOpNodeType.Add:
            ptr = model._graph.emplace_node[AddNode](lhs.array_ptr, rhs.array_ptr)
        elif node_type is _BinaryOpNodeType.And:
            ptr = model._graph.emplace_node[AndNode](lhs.array_ptr, rhs.array_ptr)
        elif node_type is _BinaryOpNodeType.Divide:
            ptr = model._graph.emplace_node[DivideNode](lhs.array_ptr, rhs.array_ptr)
        elif node_type is _BinaryOpNodeType.Equal:
            ptr = model._graph.emplace_node[EqualNode](lhs.array_ptr, rhs.array_ptr)
        elif node_type is _BinaryOpNodeType.LessEqual:
            ptr = model._graph.emplace_node[LessEqualNode](lhs.array_ptr, rhs.array_ptr)
        elif node_type is _BinaryOpNodeType.Maximum:
            ptr = model._graph.emplace_node[MaximumNode](lhs.array_ptr, rhs.array_ptr)
        elif node_type is _BinaryOpNodeType.Minimum:
            ptr = model._graph.emplace_node[MinimumNode](lhs.array_ptr, rhs.array_ptr)
        elif node_type is _BinaryOpNodeType.Modulus:
            ptr = model._graph.emplace_node[ModulusNode](lhs.array_ptr, rhs.array_ptr)
        elif node_type is _BinaryOpNodeType.Multiply:
            ptr = model._graph.emplace_node[MultiplyNode](lhs.array_ptr, rhs.array_ptr)
        elif node_type is _BinaryOpNodeType.Or:
            ptr = model._graph.emplace_node[OrNode](lhs.array_ptr, rhs.array_ptr)
        elif node_type is _BinaryOpNodeType.Subtract:
            ptr = model._graph.emplace_node[SubtractNode](lhs.array_ptr, rhs.array_ptr)
        elif node_type is _BinaryOpNodeType.Xor:
            ptr = model._graph.emplace_node[XorNode](lhs.array_ptr, rhs.array_ptr)
        else:
            raise RuntimeError(f"unexpected _BinaryOpNodeType: {<object>node_type!r}")

        (<ArraySymbol>self).initialize_arraynode(model, ptr)


cdef class BinaryExpressionOp(ArraySymbol):
    # TODO: fix docstring
    """Performs arbitrary operation element-wise over operand symbol.

    .. note:: This symbol is used by direct instantiation, as shown in the
        example below.

    Performs an element-wise operation, as specified by a
    :class:`~dwave.optimization.expression.Expression`, on two operands.

    Args:
        expression (:class:`~dwave.optimization.expression.Expression`):
            An unary (single argument) operation perform on the input array,
            specified as an expression.
        lhs (:class:`~dwave.optimization.model.Symbol`):
            The first operand to the element-wise operation.
        rhs (:class:`~dwave.optimization.model.Symbol`):
            The second operand to the element-wise operation.

    Examples:
        This example directly adds an :class:`.BinaryExpressionOp` symbol to a model
        that multiplies the operands together and adds one.

        >>> from dwave.optimization import expression, Model
        >>> from dwave.optimization.symbols import BinaryExpressionOp
        >>>
        >>> model = Model()

        >>> @expression
        ... def expr(x, y):
        ...     return x * y + 1
        >>>
        >>> a = model.constant([1, 2, 3])
        >>> b = model.constant([4, 5, 6])
        >>> bin_op = BinaryExpressionOp(expr, a, b)
        >>> with model.lock():
        ...     model.states.resize(1)
        ...     print(bin_op.state(0))
        [ 5. 11. 19.]

    .. versionadded:: 0.8.1
    """

    def __init__(self, expression, ArraySymbol lhs, ArraySymbol rhs):
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

        if expr.num_inputs() != 2:
            raise ValueError("expression must have exactly two inputs")

        # Convert the operand into something we can pass to C++
        cdef _Graph model = lhs.model
        if model != rhs.model:
            raise ValueError("both predecessors must be from the same model")

        self.ptr = model._graph.emplace_node[BinaryExpressionNode](expr._owning_ptr, lhs.array_ptr, rhs.array_ptr)

        self.initialize_arraynode(model, self.ptr)

    @classmethod
    def _from_ptr(cls, model, capsule):
        cdef BinaryExpressionOp sym = super()._from_ptr(model, capsule)

        sym.ptr = dynamic_cast_ptr[BinaryExpressionNode](sym.node_ptr)
        if not sym.ptr:
            raise TypeError(f"given pointer cannot construct a BinaryExpressionOp")

        return sym

    @classmethod
    def _from_zipfile(cls, zf, directory, _Graph model, predecessors):
        from dwave.optimization import Model  # avoid circular import

        expression = Expression.__new__(Expression)  # bypass the __init__
        with zf.open(directory + "expression.nl", "r") as f:
            expression._model = Model.from_file(f)
            expression._model.lock()

        return BinaryExpressionOp(expression, predecessors[0], predecessors[1])

    def _into_zipfile(self, zf, directory):
        """Store a BinaryExpressionOp symbol as a compressed file.

        Args:
            zf:
                File pointer to a compressed file to store the
                BinaryExpressionOp symbol. Strings are interpreted as a file
                name.
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

    cdef BinaryExpressionNode* ptr

_register(BinaryExpressionOp, typeid(BinaryExpressionNode))

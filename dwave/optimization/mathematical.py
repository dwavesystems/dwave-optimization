# Copyright 2024 D-Wave Systems Inc.
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

import collections
import functools
import itertools

from dwave.optimization.model import ArrayObserver
from dwave.optimization.symbols import (
    Add,
    And,
    Maximum,
    Minimum,
    Multiply,
    NaryAdd,
    NaryMaximum,
    NaryMinimum,
    NaryMultiply,
    Or,
)


__all__ = [
    "add",
    "logical_or",
    "logical_and",
    "maximum",
    "minimum",
    "multiply",
]


def op(BinaryOp: type, NaryOp: type, reduce_method: str):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(arg0, *args, **kwargs):
            if len(args) == 0:
                iterable = arg0
                if not isinstance(iterable, collections.abc.Iterable):
                    raise ValueError(
                        f"Cannot call {f.__name__} on a single node; did you mean to "
                        f"call `x.{reduce_method}()` ?"
                    )
                return NaryOp(*iterable)
            elif len(args) == 1:
                lhs = arg0
                rhs, = args
                return BinaryOp(lhs, rhs)
            else:
                return NaryOp(arg0, *args)
        return wrapper
    return decorator

@op(Add, NaryAdd, "sum")
def add(*args, **kwargs):
    r"""Return an element-wise addition on the given symbols. 
    
    In the underlying directed acyclic expression graph, produces an ``Add`` 
    node if two array nodes are provided and a ``NaryAdd`` node otherwise. 
    
    Returns:
        A symbol that sums the given symbols element-wise. 
        
    Examples:
        This example adds two integer symbols of size :math:`1 \times 2`.
        Equivalently, you can use the ``+`` operator (e.g., :code:`i + j`).
        
        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import add
        ...
        >>> model = Model()
        >>> i = model.integer(2)
        >>> j = model.integer(2)
        >>> i_plus_j = add(i, j)    # alternatively: i_plus_j = i + j
        >>> model.lock()
        >>> model.states.resize(1)
        >>> i.set_state(0, [3, 5])
        >>> j.set_state(0, [7, 5])
        >>> print(i_plus_j.state(0))
        [10. 10.]
    """
    ...


def logical_and(lhs: ArrayObserver, rhs: ArrayObserver):
    r"""Return an element-wise logical AND on the given symbols. 
    
    Args:
        lhs: Left-hand side symbol. 
        
        rhs: Right-hand side symbol. 
    
    Returns:
        A symbol that is the element-wise AND of the given symbols. 
        
    Examples:
        This example ANDs two binary symbols of size :math:`1 \times 3`.
        
        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import logical_and
        ...
        >>> model = Model()
        >>> x = model.binary(3)
        >>> y = model.binary(3)
        >>> z = logical_and(x, y)
        >>> model.lock()
        >>> model.states.resize(1)
        >>> x.set_state(0, [True, True, False])
        >>> y.set_state(0, [False, True, False])
        >>> print(z.state(0))
        [0. 1. 0.]
    """
    return And(lhs, rhs)


def logical_or(lhs: ArrayObserver, rhs: ArrayObserver):
    r"""Return an element-wise logical OR on the given symbols. 
    
    Args:
        lhs: Left-hand side symbol. 
        
        rhs: Right-hand side symbol. 
    
    Returns:
        A symbol that is the element-wise OR of the given symbols. 
        
    Examples:
        This example ORs two binary symbols of size :math:`1 \times 3`.
        
        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import logical_or
        ...
        >>> model = Model()
        >>> x = model.binary(3)
        >>> y = model.binary(3)
        >>> z = logical_or(x, y)
        >>> model.lock()
        >>> model.states.resize(1)
        >>> x.set_state(0, [True, True, False])
        >>> y.set_state(0, [False, True, False])
        >>> print(z.state(0))
        [1. 1. 0.]
    """
    return Or(lhs, rhs)


@op(Maximum, NaryMaximum, "max")
def maximum(*args, **kwargs):
    r"""Return an element-wise maximum of the given symbols. 
    
    In the underlying directed acyclic expression graph, produces a 
    ``Maximum`` node if two array nodes are provided and a 
    ``NaryMaximum`` node otherwise. 
    
    Returns:
        A symbol that is the element-wise maximum of the given symbols.
        
    Examples:
        This example maximizes two integer symbols of size :math:`1 \times 2`.
        
        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import maximum
        ...
        >>> model = Model()
        >>> i = model.integer(2)
        >>> j = model.integer(2)
        >>> m = maximum(i, j)
        >>> model.lock()
        >>> model.states.resize(1)
        >>> i.set_state(0, [3, 5])
        >>> j.set_state(0, [7, 2])
        >>> print(m.state(0))
        [7. 5.]
    """
    ...


@op(Minimum, NaryMinimum, "min")
def minimum(*args, **kwargs):
    r"""Return an element-wise minimum of the given symbols. 
    
    In the underlying directed acyclic expression graph, produces a 
    ``Minimum`` node if two array nodes are provided and a 
    ``NaryMinimum`` node otherwise. 
    
    Returns:
        A symbol that is the element-wise minimum of the given symbols.
        
    Examples:
        This example minimizes two integer symbols of size :math:`1 \times 2`.
        
        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import minimum
        ...
        >>> model = Model()
        >>> i = model.integer(2)
        >>> j = model.integer(2)
        >>> m = minimum(i, j)
        >>> model.lock()
        >>> model.states.resize(1)
        >>> i.set_state(0, [3, 5])
        >>> j.set_state(0, [7, 2])
        >>> print(m.state(0))
        [3. 2.]
    """
    ...

@op(Multiply, NaryMultiply, "multiply")
def multiply(*args, **kwargs):
    r"""Return an element-wise multiplication on the given symbols. 
    
    In the underlying directed acyclic expression graph, produces a 
    ``Multiply`` node if two array nodes are provided and a 
    ``NaryMultiply`` node otherwise. 
    
    Returns:
        A symbol that multiplies the given symbols element-wise.
        
    Examples:
        This example multiplies two integer symbols of size :math:`1 \times 2`.
        Equivalently, you can use the ``*`` operator (e.g., :code:`i * j`).
        
        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import multiply
        ...
        >>> model = Model()
        >>> i = model.integer(2)
        >>> j = model.integer(2)
        >>> k = multiply(i, j)   # alternatively: k = i * j
        >>> model.lock()
        >>> model.states.resize(1)
        >>> i.set_state(0, [3, 5])
        >>> j.set_state(0, [7, 2])
        >>> print(k.state(0))
        [21., 10.]
    """
    ...

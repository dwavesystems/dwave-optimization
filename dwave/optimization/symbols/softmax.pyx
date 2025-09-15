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

from dwave.optimization.libcpp.nodes.softmax cimport SoftMaxNode
from dwave.optimization._model cimport ArraySymbol, _register, _Graph


cdef class SoftMax(ArraySymbol):
    """Softmax of a symbol.

    Example:
        These examples compute the softmax of one symbol.

        >>> from dwave.optimization.model import Model
        >>> from dwave.optimization.symbols import SoftMax
        >>> import numpy as np
        ...
        >>> model = Model()
        >>> model.states.resize(1)
        >>> c = model.constant([1, 2, 3])
        >>> sm = SoftMax(c)
        >>> expected = np.array([0.09003057, 0.24472847, 0.66524096])
        >>> with model.lock():
        ...     print(np.isclose(sm.state(0), expected).all())
        True

        >>> from dwave.optimization import Model
        >>> from dwave.optimization.mathematical import softmax
        ...
        >>> model = Model()
        >>> i = model.integer(3)
        >>> sm = softmax(i)
        >>> type(sm)
        <class 'dwave.optimization.symbols.SoftMax'>

    See Also:
        :meth:`~dwave.optimization.mathematical.softmax`: equivalent method.

    .. versionadded:: 0.6.5
    """
    def __init__(self, ArraySymbol arr):
        cdef _Graph model = arr.model

        cdef SoftMaxNode* ptr = model._graph.emplace_node[SoftMaxNode](arr.array_ptr)
        self.initialize_arraynode(model, ptr)

_register(SoftMax, typeid(SoftMaxNode))

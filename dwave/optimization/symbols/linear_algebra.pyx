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
from libcpp.string cimport string

from dwave.optimization._model cimport _Graph, _register, ArraySymbol
from dwave.optimization.libcpp.nodes.linear_algebra cimport (
    MatrixMultiplyNode,
)

# Can't access static attributes from Cython without a workaround. So here it
# is.
cdef extern from * nogil:
    """
    std::string matmul_implementation() {
        return dwave::optimization::MatrixMultiplyNode::implementation;
    }
    """
    string matmul_implementation()


cdef class MatrixMultiply(ArraySymbol):
    """MatrixMultiply symbol.

    See Also:
        :func:`~dwave.optimization.mathematical.matmul`: equivalent function.

    .. versionadded:: 0.6.10
    """
    def __init__(self, ArraySymbol x, ArraySymbol y):
        cdef _Graph model = x.model
        if y.model is not model:
            raise ValueError("operands must be from the same model")

        cdef MatrixMultiplyNode* ptr = model._graph.emplace_node[MatrixMultiplyNode](
            x.array_ptr, y.array_ptr,
        )
        self.initialize_arraynode(model, ptr)

    @staticmethod
    def implementation():
        """Return the matrix multiplication implementation.

        Either `"blas"` or `"fallback"`.
        """
        return bytes(matmul_implementation()).decode()

_register(MatrixMultiply, typeid(MatrixMultiplyNode))

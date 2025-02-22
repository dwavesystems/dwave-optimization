# Copyright 2024 D-Wave
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

import weakref

from libcpp.utility cimport move

from dwave.optimization.libcpp.array cimport Array as cppArray
from dwave.optimization.model cimport ArraySymbol, _Graph
from dwave.optimization.model import Model

__all__ = ["States"]


cdef class States:
    r"""States of a symbol in a model.

    States represent assignments of values to a symbol's elements. For
    example, an :meth:`~Model.integer` symbol of size :math:`1 \times 5`
    might have state ``[3, 8, 0, 12, 8]``, representing one assignment
    of values to the symbol.

    Examples:
        This example creates a :class:`~dwave.optimization.generators.knapsack`
        model and manipulates its states to test that it behaves as expected.

        First, create a model.

        >>> from dwave.optimization import Model
        ...
        >>> model = Model()
        >>> # Add constants
        >>> weights = model.constant([10, 20, 5, 15])
        >>> values = model.constant([-5, -7, -2, -9])
        >>> capacity = model.constant(30)
        >>> # Add the decision variable
        >>> items = model.set(4)
        >>> # add the capacity constraint
        >>> model.add_constraint(weights[items].sum() <= capacity) # doctest: +ELLIPSIS
        <dwave.optimization.symbols.LessEqual at ...>
        >>> # Set the objective
        >>> model.minimize(values[items].sum())

        Lock the model to prevent changes to directed acyclic graph. At any
        time, you can verify the locked state, which is demonstrated here.

        >>> with model.lock():
        ...     model.is_locked()
        True

        Set a couple of states on the decision variable and verify that the
        model generates the expected values for the objective.

        >>> model.states.resize(2)
        >>> items.set_state(0, [0, 1])
        >>> items.set_state(1, [0, 2, 3])
        >>> with model.lock():
        ...     print(model.objective.state(0) > model.objective.state(1))
        True

        You can clear the states you set.

        >>> model.states.clear()
        >>> model.states.size()
        0
    """
    def __init__(self, model):
        if not isinstance(model, Model):
            raise TypeError("model must be an instance of Model")
        self._model_ref = weakref.ref(model)

    def __len__(self):
        """The number of model states."""
        return self.size()

    cdef void attach_states(self, vector[cppState] states) noexcept:
        """Attach the given states.

        Note:
            Currently replaces any current states with the given states.

            This method does not check whether the states are locked
            or that the states are valid.

        Args:
            states: States to be attached.
        """
        self._future = None
        self._result_hook = None
        self._states.swap(states)

    def clear(self):
        """Clear any saved states.

        Clears any memory allocated to the states.

        Examples:
            This example clears a state set on an integer decision symbol.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> i = model.integer(2)
            >>> model.states.resize(3)
            >>> i.set_state(0, [3, 5])
            >>> print(i.state(0))
            [3. 5.]
            >>> model.states.clear()
        """
        self.detach_states()

    cdef vector[cppState] detach_states(self):
        """Move the current C++ states into a returned vector.

        Leaves the model's states empty.

        Note:
            This method does not check whether the states are locked.

        Returns:
            States of the model prior to execution.
        """
        self.resolve()
        # move should impliclty leave the states in a valid state, but
        # just to be super explicit we swap with an empty vector first
        cdef vector[cppState] states
        self._states.swap(states)
        return move(states)

    def from_file(self, file, *, replace = True, check_header = True):
        """Construct states from the given file.

        Args:
            file:
                File pointer to a readable, seekable file-like object encoding
                the states. Strings are interpreted as a file name.
            replace:
                If ``True``, any held states are replaced with those from the file.
                If ``False``, the states are appended.
            check_header:
                Set to ``False`` to skip file-header check.

        Returns:
            A model.
        """
        self.resolve()

        if not replace:
            raise NotImplementedError("appending states is not (yet) implemented")

        # todo: we don't need to actually construct a model, but this is nice and
        # abstract. We should performance test and then potentially re-implement
        cdef _Graph model = Model.from_file(file, check_header=check_header)
        cdef States states = model.states

        # Check that the model is compatible
        for n0, n1 in zip(model.iter_symbols(), self._model().iter_symbols()):
            # todo: replace with proper node quality testing once we have it
            if not isinstance(n0, type(n1)):
                raise ValueError("cannot load states into a model with mismatched decisions")

        self.attach_states(move(states.detach_states()))

    def from_future(self, future, result_hook):
        """Populate the states from the result of a future computation.

        A :doc:`Future <oceandocs:docs_cloud/reference/computation>` object is
        returned by the solver to which your problem model is submitted. This
        enables asynchronous problem submission.

        Args:
            future: ``Future`` object.

            result_hook: Method executed to retrieve the Future.
        """
        self.resize(0)  # always clears self first

        self._future = future
        self._result_hook = result_hook

    def initialize(self):
        """Initialize any uninitialized states."""
        self.resolve()

        cdef _Graph model = self._model()

        if not model.is_locked():
            raise ValueError("Cannot initialize states of an unlocked model")
        for i in range(self._states.size()):
            self._states[i].resize(model.num_nodes())
            model._graph.initialize_state(self._states[i])

    def into_file(self, file, *,
                  version = None,
                  ):
        """Serialize the states into an existing  file.

        Args:
            file:
                File pointer to an existing writeable, seekable file-like
                object encoding a model. Strings are interpreted as a file
                name.
            version:
                A 2-tuple indicating which serialization version to use.

        TODO: describe the format
        """
        self.resolve()
        return self._model().into_file(
            file,
            only_decision=True,
            max_num_states=self.size(),
            version=version,
            )


    cdef _Graph _model(self):
        """Get a ref-counted Model object."""
        cdef _Graph m = self._model_ref()
        if m is None:
            raise ReferenceError("accessing the states of a garbage collected model")
        return m

    def _reset_intermediate_states(self):
        """Reset any non-decision states."""
        cdef Py_ssize_t num_decisions = self._model().num_decisions()
        for i in range(self.size()):
            self._states[i].resize(num_decisions)

    def resize(self, Py_ssize_t n):
        """Resize the number of states.

        If ``n`` is smaller than the current :meth:`.size()`,
        states are reduced to the first ``n`` states by removing
        those beyond. If ``n`` is greater than the current
        :meth:`.size()`, new uninitialized states are added
        as needed to reach a size of ``n``.

        Resizing to 0 is not  guaranteed to clear the memory allocated to
        states.

        Args:
            n: Required number of states.

        Examples:
            This example adds three uninitialized states to a model.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> i = model.integer(2)
            >>> model.states.resize(3)
        """
        self.resolve()

        if n < 0:
            raise ValueError("n must be a non-negative integer")

        self._states.resize(n)

    cpdef resolve(self):
        """Block until states are retrieved from any pending future computations.

        A :doc:`Future <oceandocs:docs_cloud/reference/computation>` object is
        returned by the solver to which your problem model is submitted. This
        enables asynchronous problem submission.
        """
        if self._future is not None:
            # The existance of _future means that anything we do to the
            # state will block. So we remove it before calling the hook.
            future = self._future
            self._future = None
            result_hook = self._result_hook
            self._result_hook = None

            result_hook(self._model(), future)

    cpdef Py_ssize_t size(self) except -1:
        """Number of model states.

        Examples:
            This example adds three uninitialized states to a model and
            verifies the number of model states.

            >>> from dwave.optimization.model import Model
            >>> model = Model()
            >>> model.states.resize(3)
            >>> model.states.size()
            3
        """
        self.resolve()
        return self._states.size()

    def to_file(self):
        """Serialize the states to a new file-like object."""
        self.resolve()
        return self._model().to_file(only_decision=True, max_num_states=self.size())


cdef class StateView:
    def __init__(self, ArraySymbol symbol, Py_ssize_t index):
        self.symbol = symbol
        self.index = index

        # we're assuming this object is being created because we want to access
        # the state, so let's go ahead and create the state if it's not already
        # there
        cdef States states = symbol.model.states  # for Cython access
        
        states.resolve()
        symbol.model._graph.recursive_initialize(states._states.at(index), symbol.node_ptr)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        # todo: inspect/respect/test flags
        self.symbol.model.states.resolve()

        cdef States states = self.symbol.model.states  # for Cython access

        cdef cppArray* ptr = self.symbol.array_ptr

        buffer.buf = <void*>(ptr.buff(states._states.at(self.index)))
        buffer.format = <char*>(ptr.format().c_str())
        buffer.internal = NULL
        buffer.itemsize = ptr.itemsize()
        buffer.len = ptr.len(states._states.at(self.index))
        buffer.ndim = ptr.ndim()
        buffer.obj = self
        buffer.readonly = 1  # todo: consider loosening this requirement
        buffer.shape = <Py_ssize_t*>(ptr.shape(states._states.at(self.index)).data())
        buffer.strides = <Py_ssize_t*>(ptr.strides().data())
        buffer.suboffsets = NULL

        states._view_count += 1

    def __releasebuffer__(self, Py_buffer *buffer):
        self.symbol.model.states._view_count -= 1

    cdef readonly Py_ssize_t index  # which state we're accessing
    cdef readonly ArraySymbol symbol

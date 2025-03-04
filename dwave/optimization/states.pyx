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

import json
import tempfile
import weakref
import zipfile

from libcpp.utility cimport move

from dwave.optimization.libcpp.array cimport Array as cppArray
from dwave.optimization.model cimport ArraySymbol, _Graph
from dwave.optimization.model import Model
from dwave.optimization.utilities import _file_object_arg

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

    @_file_object_arg("rb")  # translate str/bytes file inputs into file objects
    def from_file(self, file, *,
                  replace = True,       # undoc until supported
                  check_header = True,  # undocumented until fixed, see
                                        # https://github.com/dwavesystems/dwave-optimization/issues/22
                  ):
        """Construct states from the given file.

        Args:
            file:
                File pointer to a readable, seekable file-like object encoding
                the states. Strings are interpreted as a file name.

        Returns:
            A model.

        See Also:
            :meth:`States.from_file()`.
        """
        if not replace:
            raise NotImplementedError("appending states is not (yet) implemented")

        # Validate the header, and get the serialization version.
        version, _ = _Graph._from_file_header(file)

        with zipfile.ZipFile(file, mode="r") as zf:
            self._from_zipfile(zf, version=version)

    def _from_zipfile(self, zf, *, version):
        """Given a zipfile containing a serialized model, attempt to extract the
        state(s) associated with any node types that might save them.

        ``zf`` must be a :class:`zipfile.ZipFile`.
        """
        self.resolve()

        model_info = json.loads(zf.read("info.json"))

        # Read any states that have been encoded
        num_states = model_info.get("num_states")
        if not isinstance(num_states, int) or num_states < 0:
            raise ValueError("expected num_states to be a positive integer")
        elif not num_states:
            return  # no states to load, so just return early

        # Get a refcounted model for the duration of this function.
        cdef _Graph model = self._model()

        # We'll be overwriting the states, so clear everything that's in there
        model.states.resize(0)
        model.states.resize(num_states)

        for symbol in model.iter_symbols():
            # we don't load the state of any nodes that uniquely determine
            # their state from their predecessors
            if symbol._deterministic_state():
                continue

            symbol._states_from_zipfile(
                zf,
                num_states=num_states,
                version=version,
            )

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

    @_file_object_arg("wb")  # translate str/bytes file inputs into file objects
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

        Format Specification (Version 1.0):

            The first section of the file is the header, as described in
            :meth:`Model.into_file()`.

            Following the header, the remaining data is encoded as a zip file.
            All arrays are saved using the NumPy serialization format, see
            :func:`numpy.save()`.

            The information in the header is also saved in a json-formatted
            file ``info.json``.

            The serialization version is saved in a file ``version.txt``.

            The states have the following structure.

            Symbols with a state that's uniquely determined by their predecessor's
            states and :class:`~dwave.optimization.symbols.Constant` symbols do
            not have their states serialized.

            For symbols with a fixed shape and which have all states initialized,
            the states are stored as a ``(num_states, *symbol.shape())`` array.

            .. code-block::

                nodes/
                    <symbol id>/
                        states.npy
                    ...

            For symbols without a fixed shape, or for which not all states are
            initialized, the states are each saved in a separate array.

            .. code-block::

                nodes/
                    <node id>/
                        states/
                            <state index>/
                                array.npy
                            ...
                    ...

            This format allows the states and the model to be saved in the same
            file, sharing the header.

        Format Specification (Version 0.1):

            Saved as a :class:`Model` encoding only the decision symbols.

        See Also:
            :meth:`States.from_file()`.

        .. versionchanged:: 0.5.2
            Added the ``version`` keyword-only argument.
        .. versionchanged:: 0.6.0
            Added support for serialization format version 1.0.
        """
        self.resolve()

        if isinstance(version, tuple) and version < (1, 0):
            # In serialization version 0.1, we saved a model with only
            # decisions encoding the states.
            return self._model().into_file(
                file,
                only_decision=True,
                max_num_states=self.size(),
                version=version,
                )

        model = self._model()  # get a ref-counted model

        version, model_info = model._into_file_header(
            file,
            version=version,
            max_num_states=self.size(),
            only_decision=False,
        )

        num_states = model_info["num_states"]

        encoder = json.JSONEncoder(separators=(',', ':'))

        with zipfile.ZipFile(file, mode="w") as zf:
            zf.writestr("info.json", encoder.encode(model_info))
            zf.writestr("version.txt", ".".join(map(str, version)))

            self._into_zipfile(zf, num_states=num_states, version=version)

    def _into_zipfile(self, zf, *, num_states, version):
        """Given a zipfile containing a serialized model, attempt to save the
        state(s) associated with any node types that might save them.

        ``zf`` must be a :class:`zipfile.ZipFile`.
        """
        if num_states <= 0:
            # nothing so save, so shortcut
            return

        model = self._model()  # get a ref-counted model

        for symbol in model.iter_symbols():
            if symbol._deterministic_state():
                continue

            symbol._states_into_zipfile(
                zf,
                num_states=num_states,
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

    def to_file(self, **kwargs):
        """Serialize the states to a new file-like object."""
        file = tempfile.TemporaryFile(mode="w+b")

        # into_file can raise an exception, in which case we close off the
        # tempfile before returning
        try:
            self.into_file(file, **kwargs)
        except Exception:
            file.close()
            raise

        file.seek(0)
        return file


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

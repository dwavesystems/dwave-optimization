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

import functools
import numbers
import os
import types

__all__ = []


# Adopted from NumPy under BSD license
# https://github.com/numpy/numpy/blob/0959a54/LICENSE.txt
# https://github.com/numpy/numpy/blob/0959a54/numpy/_globals.py#L32-L63
#
# We could just use numpy._globals._NoValue directly but because that's private
# and subject to change, we choose to instead maintain our own.
class _NoValueType:
    """Special keyword value to use when no other default is appropriate.

    Specifically useful to distinguish from ``None``.
    """
    __instance = None

    def __new__(cls):
        # ensure that only one instance ever exists
        if not cls.__instance:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __repr__(self):
        return "<no value>"


_NoValue = _NoValueType()


def _file_object_arg(mode: str):
    """In several methods we want to accept a file name or a file-like object.

    The ``mode`` argument is the same as for ``open()``.

    This method assumes that the file argument is the first one. We could
    generalize if we need to.
    """
    def decorator(method):
        @functools.wraps(method)
        def _method(cls_or_self, file, *args, **kwargs):
            if isinstance(file, (str, bytes, os.PathLike)):
                with open(os.fspath(file), mode) as fobj:
                    return method(cls_or_self, fobj, *args, **kwargs)
            else:
                return method(cls_or_self, file, *args, **kwargs)
        return _method
    return decorator


def _lock(method):
    """Decorator for Model methods that lock the model for the duration."""
    @functools.wraps(method)
    def _method(obj, *args, **kwargs):
        if not obj.is_locked():
            with obj.lock():
                return method(obj, *args, **kwargs)
        else:
            return method(obj, *args, **kwargs)
    return _method


def _split_indices(
    shape: tuple[int, ...],
    index: tuple[int | slice | None | types.EllipsisType | object, ...],
):
    """Given a combined indexing operation, split into several steps.

    Args:
        shape: The shape of the indexed array.
        index: A tuple of indexers.

    Returns:
        In order that they should be applied:
            * A list of new axes that should be added, as with ``np.expand_dims()``.
            * Indexers that can be given to ``BasicIndexing``
            * Indexers that can be given to ``AdvancedIndexing``.

    """
    # developer note: there is a known issue where splitting the indexers in this
    # way cannot account for all cases.
    # See https://github.com/dwavesystems/dwave-optimization/issues/465 for more
    # information.

    indices: list = list(index)

    # Handle the ellipses if it's present
    if (count := sum(idx is ... for idx in indices)) > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    elif count == 1:
        # We have an ellipses, so we need to replace it with empty slice(s) until
        # we hit the correct length

        # First, find where the ellipses is
        for loc, index in enumerate(indices):
            if index is ...:
                break
        else:
            raise RuntimeError  # shouldn't be able to get here

        # Now that we know where the ellipses is, we remove it and then replace
        # it with empty slices until we hit our desired ndim. Though we need
        # to make sure not to count the newaxes towards that count
        indices.pop(loc)
        for _ in range(sum(idx is not None for idx in indices), len(shape)):
            indices.insert(loc, slice(None))

    # Now divide everything that's remaining between basic and advanced indexing
    newaxes: list[int] = []
    basic: list[slice | int] = []
    advanced: list[slice | object] = []
    for i, index in enumerate(indices):
        if index is None:
            # We'll insert the new axis before calling basid/advanced indexing
            newaxes.append(i)
            basic.append(slice(None))
            advanced.append(slice(None))
        elif isinstance(index, numbers.Integral):
            # Only basic handles numeric indices and it removes the axis so
            # only basic gets the index
            basic.append(index)
        elif isinstance(index, slice) and index == slice(None):
            # Empty slices are handled by both basic and advanced indexing
            basic.append(slice(None))
            advanced.append(slice(None))
        elif isinstance(index, slice):
            # Non-empty slice are only handled by basic indexing
            basic.append(index)
            advanced.append(slice(None))
        else:
            # For anything else, we defer to advanced indexing for the type
            # checking
            basic.append(slice(None))
            advanced.append(index)

    return tuple(newaxes), tuple(basic), tuple(advanced)


def _TypeError_to_NotImplemented(f):
    """Convert any TypeErrors raised by the given function into NotImplemented"""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except TypeError:
            return NotImplemented
    return wrapper

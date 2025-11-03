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

import os
import functools

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

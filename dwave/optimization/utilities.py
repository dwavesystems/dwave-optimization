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


def _file_object_arg(mode):
    """In several methods we want to accept a string giving a file name and
    interpret it as a file object.

    This handles that case by passing through non-strings, but calling open()
    on strings.

    This method assumes that the file argument is the first one. We could
    generalize if we need to.

    Before adding this method we were frequently forgetting to pass all other
    args/kwargs through which was a source of bugs.
    """
    def decorator(method):
        @functools.wraps(method)
        def _method(cls_or_self, file, *args, **kwargs):
            if isinstance(file, str):
                with open(file, mode) as fobj:
                    return method(cls_or_self, fobj, *args, **kwargs)
            else:
                return method(cls_or_self, file, *args, **kwargs)
        return _method
    return decorator

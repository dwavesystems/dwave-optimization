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

import collections.abc
import typing

from dwave.optimization.model import Model


class States:
    def __init__(self, model: Model): ...
    def __len__(self) -> int: ...
    def clear(self): ...

    def from_file(
        self,
        file: typing.Union[typing.BinaryIO, collections.abc.ByteString, str],
        *,
        replace: bool = True,
        check_header: bool = True,
        ) -> Model: ...

    def from_future(self, future: object, result_hook: collections.abc.Callable): ...
    def initialize(self): ...

    def into_file(
        self,
        file: typing.Union[typing.BinaryIO, collections.abc.ByteString, str],
        version: typing.Optional[tuple[int, int]] = None,
        ): ...

    def _reset_intermediate_states(self): ...
    def resize(self, n: int): ...
    def resolve(self): ...
    def size(self) -> int: ...
    def to_file(self) -> typing.BinaryIO: ...

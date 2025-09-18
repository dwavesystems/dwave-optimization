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

import typing

import numpy as _np

from dwave.optimization.model import ArraySymbol as _ArraySymbol

class Constant(_ArraySymbol):
    def __bool__(self) -> bool: ...
    def __index__(self) -> int: ...

    # these methods don't (yet) have an ArraySymbol overload
    def __ge__(self, rhs: _np.typing.ArrayLike) -> _np.typing.NDArray[_np.bool]: ...
    def __gt__(self, rhs: _np.typing.ArrayLike) -> _np.typing.NDArray[_np.bool]: ...
    def __lt__(self, rhs: _np.typing.ArrayLike) -> _np.typing.NDArray[_np.bool]: ...

    @typing.overload
    def __eq__(self, rhs: _ArraySymbol) -> _ArraySymbol: ...
    @typing.overload
    def __eq__(self, rhs: _np.typing.ArrayLike) -> _np.typing.NDArray[_np.bool]: ...
    @typing.overload
    def __le__(self, rhs: _ArraySymbol) -> _ArraySymbol: ...
    @typing.overload
    def __le__(self, rhs: _np.typing.ArrayLike) -> _np.typing.NDArray[_np.bool]: ...

// Copyright 2025 D-Wave
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#pragma once

#include "dwave-optimization/config.hpp"

// We want to use ssize_t (to match Python's Py_ssize_t) in a lot of places
// but it's not always available.

#if DWOPT_SYS_TYPES_HAS_SSIZE_T

// If ssize_t is available (via <sys/types.hp>) then we use it.

#include <sys/types.h>  // for ssize_t

namespace dwave::optimization {
using ::ssize_t;  // so dwave::optimization::ssize_t works everywhere
}  // namespace dwave::optimization

#else

// We try to match Windows
// https://github.com/python/cpython/blob/333071231/PC/pyconfig.h#L222-L230
// though using std::ptrdiff_t is more convenient to match Win32/Win64 and
// we assert the match in Cython.

#include <cstddef>  // for std::ptrdiff_t

namespace dwave::optimization {
using ssize_t = std::ptrdiff_t;
}  // namespace dwave::optimization

#endif

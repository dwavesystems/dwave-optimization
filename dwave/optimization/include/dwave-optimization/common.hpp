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

/// Support for different compilers and/or backports from future C++ versions

#pragma once

// We want to use ssize_t (to match Python's Py_ssize_t) in a lot of places
// but it's not always available.
// We will assert the match between ssize_t and Py_ssize_t in the Cython code.

#if defined(DWOPT__SYS_TYPES_HAS_SSIZE_T)

// Meson did a compiler probe and set the DWOPT__SYS_TYPES_HAS_SSIZE_T flag,
// or the user set it explicitly.
// We add it to the dwave::optimization namespace so that
// dwave::optimization::ssize_t works everywhere.

#include <sys/types.h>

namespace dwave::optimization {
using ::ssize_t;
}  // namespace dwave::optimization

#else

// Fallback to using std::ptrdiff_t because that supports Win32/Win64 equally well.

#include <cstddef>

namespace dwave::optimization {
using ssize_t = std::ptrdiff_t;
}  // namespace dwave::optimization

#endif

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

#if defined(__has_include)
#if __has_include(<sys/types.h>)

// Ok, we have sys/types.h which on POSIX systems has ssize_t.
// However, Windows also has sys/types.h but does not have ssize_t.
// POSIX systems are required to have SSIZE_MAX in limits.h so we use that
// to distinguish these cases.
// In theory there could be some weird compiler chain that passes all of these
// but doesn't have ssize_t, but I think we're reasonably safe.

#include <climits>

#if defined(SSIZE_MAX)

#include <sys/types.h>

// Ok, we (probably) have ssize_t!
// We do add it to the dwave::optimization namespace so that
// dwave::optimization::ssize_t works everywhere.

namespace dwave::optimization {
using ::ssize_t;
}  // namespace dwave::optimization

#define DWOPT__SYS_TYPES_HAS_SSIZE_T

#endif
#endif
#endif

#if !defined(DWOPT__SYS_TYPES_HAS_SSIZE_T)

// Fallback to using std::ptrdiff_t because that supports Win32/Win64 equally well

#include <cstddef>

namespace dwave::optimization {
using ssize_t = std::ptrdiff_t;
}  // namespace dwave::optimization

#endif

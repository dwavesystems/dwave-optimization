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

/// Support for different compilers and backports from future C++ versions

#pragma once

#if defined(_MSC_VER) && defined(_WIN64) && (_MSC_VER > 1400) || defined(__MINGW32__) || \
        defined(__MINGW64__)  // Not POSIX

#include <cstdint>  // for int64_t

// We use ssize_t everywhere to match Python's Py_ssize_t.
// However ssize_t is only defined for posix, so we define it in windows.
namespace dwave::optimization {
typedef std::int64_t ssize_t;
}  // namespace dwave::optimization

#else  // Not Windows

#include <sys/types.h>  // for ssize_t

#endif

namespace dwave::optimization {

// backport unreachable from c++23
[[noreturn]] inline void unreachable() {
#if defined(_MSC_VER) && !defined(__clang__)  // MSVC
    __assume(false);
#else  // GCC, Clang
    __builtin_unreachable();
#endif
}

}  // namespace dwave::optimization

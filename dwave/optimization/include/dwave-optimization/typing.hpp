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

#include <concepts>
#include <cstdint>

namespace dwave::optimization {

/// The `DType<T>` concept is satisfied if and only if `T` is one of the
/// supported data types. This list is a subset of the `dtype`s supported by
/// NumPy as is designed for consistency/compatibility with NumPy.
/// See https://numpy.org/doc/stable/reference/arrays.dtypes.html for more
/// information.
template <typename T>
concept DType = std::same_as<T, float> ||         // np.float32
                std::same_as<T, double> ||        // np.float64
                std::same_as<T, std::int8_t> ||   // np.int8
                std::same_as<T, std::int16_t> ||  // np.int16
                std::same_as<T, std::int32_t> ||  // np.int32
                std::same_as<T, std::int64_t>;    // np.int64

/// The `OptionalDType<T>` concept is satisfied if and only if `T` is `void`
/// or satisfies `DType<T>`.
/// `void` is used to signal that the dtype is unknown at compile-time.
template <typename T>
concept OptionalDType = DType<T> || std::same_as<T, void>;

// Dev note: We'd like to work with `DType` only, but in order to work with the
// buffer protocol (https://docs.python.org/3/c-api/buffer.html) we need to
// use format characters. And, annoyingly, we need one more format character
// than dtype to cover all of the compilers/OS that we care about.

/// Format characters are used to signal how bytes in a fixed-size block of
/// memory should be interpreted. This list is a subset of the format characters
/// supported by Python's `struct` library.
/// See https://docs.python.org/3/library/struct.html for more information.
enum class FormatCharacter : char {
    f = 'f',  // float
    d = 'd',  // double
    i = 'i',  // int
    b = 'b',  // signed char
    h = 'h',  // signed short
    l = 'l',  // signed long
    Q = 'Q',  // signed long long
};

}  // namespace dwave::optimization

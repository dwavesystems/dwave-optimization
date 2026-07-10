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
#include <type_traits>

#include "dwave-optimization/type_list.hpp"

namespace dwave::optimization {

// We want to support an enumerated list of data types, corresponding to a
// subset supported by NumPy.
// See https://numpy.org/doc/stable/reference/arrays.dtypes.html for more
// information.

/// Supported dtypes as a type_list
using DTypes = type_list<
    float,          // np.float32
    double,         // np.float64
    bool,           // np.bool_
    std::int8_t,    // np.int8
    std::int16_t,   // np.int16
    std::int32_t,   // np.int32
    std::int64_t>;  // np.int64

/// The `DType<T>` concept is satisfied if and only if `T` is one of the
/// supported dtypes.
template <typename T>
concept DType = std::same_as<T, float> or         // np.float32
                std::same_as<T, double> or        // np.float64
                std::same_as<T, bool> or          // np.bool_
                std::same_as<T, std::int8_t> or   // np.int8
                std::same_as<T, std::int16_t> or  // np.int16
                std::same_as<T, std::int32_t> or  // np.int32
                std::same_as<T, std::int64_t>;    // np.int64
// We could have instead done DTypes::contains<T>() but this way the compiler
// error message is a lot clearer

/// The `OptionalDType<T>` concept is satisfied if and only if `T` is `void`
/// or satisfies `DType<T>`.
/// `void` is used to signal that the dtype is unknown at compile-time.
template <typename T>
concept OptionalDType = DType<T> or std::same_as<T, void>;

/// The `DTypeLike<T>` concept is satisfied for supported types, as well as
/// const/volatile and references.
template <typename T>
concept DTypeLike = DType<std::remove_cvref_t<T>>;

// Dev note: We'd like to work with `DType` only, but in order to work with the
// buffer protocol (https://docs.python.org/3/c-api/buffer.html) we need to
// use format characters. And, annoyingly, we need one more format character
// than dtype to cover all of the compilers/OS that we care about.

/// Format characters are used to signal how bytes in a fixed-size block of
/// memory should be interpreted. This list is a subset of the format characters
/// supported by Python's `struct` library.
/// See https://docs.python.org/3/library/struct.html for more information.
enum class FormatCharacter : char {
    float_ = 'f',
    double_ = 'd',
    bool_ = '?',
    int_ = 'i',
    signedchar_ = 'b',
    signedshort_ = 'h',
    signedlong_ = 'l',
    signedlonglong_ = 'q',
};

/// Get the format character associated with a supported DType.
template <DTypeLike T>
constexpr FormatCharacter format_of() {
    using U = std::remove_cvref_t<T>;
    if constexpr (std::same_as<U, float>) return FormatCharacter::float_;
    if constexpr (std::same_as<U, double>) return FormatCharacter::double_;
    if constexpr (std::same_as<U, bool>) return FormatCharacter::bool_;
    if constexpr (std::same_as<U, int>) return FormatCharacter::int_;
    if constexpr (std::same_as<U, signed char>) return FormatCharacter::signedchar_;
    if constexpr (std::same_as<U, signed short>) return FormatCharacter::signedshort_;
    if constexpr (std::same_as<U, signed long>) return FormatCharacter::signedlong_;
    if constexpr (std::same_as<U, signed long long>) return FormatCharacter::signedlonglong_;
}

}  // namespace dwave::optimization

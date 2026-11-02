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
#include <type_traits>

#include "dwave-optimization/type_list.hpp"

namespace dwave::optimization {

/// We want to support an enumerated list of data types, corresponding to a
/// subset supported by NumPy.
/// See https://numpy.org/doc/stable/reference/arrays.dtypes.html for more
/// information.
/// We actually prefer that users make use of the fixed-width integer types
/// in their applications, but in order to provide compiler-agnostic support
/// we defined our list of supported DTypes using the builtins. This means
/// that almost always one will be redundant (because long is either 32 or
/// 64 bits depending on the platform).

/// Supported dtypes as a type_list
using DTypes = type_list<
    float,        // np.float32
    double,       // np.float64
    bool,         // np.bool_
    signed char,  // np.int8
    short,        // np.int16
    int,          // np.int32
    long,         // np.long
    long long     // np.longlong
>;

/// The `DType<T>` concept is satisfied if and only if `T` is one of the
/// supported dtypes.
template <typename T>
concept DType = std::same_as<T, float> or        // np.float32
                std::same_as<T, double> or       // np.float64
                std::same_as<T, bool> or         // np.bool_
                std::same_as<T, signed char> or  // np.int8
                std::same_as<T, short> or        // np.int16
                std::same_as<T, int> or          // np.int32
                std::same_as<T, long> or         // np.long
                std::same_as<T, long long>;      // np.longlong
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

/// Format characters are used to signal how bytes in a fixed-size block of
/// memory should be interpreted. This list is a subset of the format characters
/// supported by Python's `struct` library.
/// See https://docs.python.org/3/library/struct.html for more information.
enum class FormatCharacter : char {
    float_ = 'f',
    double_ = 'd',
    bool_ = '?',
    signedchar_ = 'b',
    int_ = 'i',
    short_ = 'h',
    long_ = 'l',
    longlong_ = 'q',
};

/// Get the format character associated with a supported DType.
template <DTypeLike T>
constexpr FormatCharacter format_of() {
    using U = std::remove_cvref_t<T>;
    if constexpr (std::same_as<U, float>) {
        return FormatCharacter::float_;
    } else if constexpr (std::same_as<U, double>) {
        return FormatCharacter::double_;
    } else if constexpr (std::same_as<U, bool>) {
        return FormatCharacter::bool_;
    } else if constexpr (std::same_as<U, signed char>) {
        return FormatCharacter::signedchar_;
    } else if constexpr (std::same_as<U, int>) {
        return FormatCharacter::int_;
    } else if constexpr (std::same_as<U, short>) {
        return FormatCharacter::short_;
    } else if constexpr (std::same_as<U, long>) {
        return FormatCharacter::long_;
    } else if constexpr (std::same_as<U, long long>) {
        return FormatCharacter::longlong_;
    } else {
        static_assert(false, "unexpected dtype");
    }
}

/// NumPy's type promotion doesn't match the behavior of C++'s numeric promotions
/// This class is meant to mimic the behavior of the `numpy.promote_types()`
/// function.
template <DTypeLike T, DTypeLike U>
class promote_types {
 private:
    consteval static auto promote_types_() {
        using T_ = std::remove_cvref_t<T>;
        using U_ = std::remove_cvref_t<U>;

        if constexpr (std::same_as<T_, bool>) {
            // bool doesn't promote any other types
            return U_{};
        } else if constexpr (std::same_as<U_, bool>) {
            // bool doesn't promote any other types
            return T_{};
        } else if constexpr (std::integral<T_> and std::integral<U_>) {
            // If both are integers, promote to the larger type.
            // Regardless of platform, we consider int < long < longlong
            if constexpr (std::same_as<T_, long long>) {
                // If either are long long then promote to long long
                return T_{};
            } else if constexpr (std::same_as<U_, long long>) {
                // if either are long long then promote to long long
                return U_{};
            } else if constexpr (std::same_as<T_, long>) {
                // if neither are long long and either is long, then promote to long
                return T_{};
            } else if constexpr (std::same_as<U_, long>) {
                // if neither are long long and either is long, then promote to long
                return U_{};
            } else {
                // neither are long nor long long, so the larger sized integer wins
                return std::conditional_t<sizeof(T_) >= sizeof(U_), T_, U_>{};
            }
        } else if constexpr (std::floating_point<T_> and std::floating_point<U_>) {
            // If both are floating point, promote to the larger type
            return std::conditional_t<sizeof(T_) >= sizeof(U_), T_, U_>{};
        } else if constexpr (std::integral<T_>) {
            // T_ is an integer and U_ is a floating point, so make sure we return
            // a large enough type
            return std::conditional_t<sizeof(T_) <= 2, U_, double>{};
        } else {
            // T_ is a floating point and U_ is an integer, so make sure we return
            // a large enough type
            return std::conditional_t<sizeof(U_) <= 2, T_, double>{};
        }
    }

 public:
    using type = decltype(promote_types_());
};

template <DTypeLike T, DTypeLike U>
using promote_types_t = typename promote_types<T, U>::type;

/// Test whether `From` can be safely cast to `To` according to NumPy's promotion
/// rules.
/// Note that `np.can_cast(int64, float64, "safe")` is `True` according to NumPy.
template <typename From, typename To>
concept can_cast =
    DType<From> and DType<To> and
    (std::same_as<promote_types_t<From, To>, To> or
     (std::signed_integral<From> and std::signed_integral<To> and sizeof(From) == sizeof(To))  // same size integers
);

}  // namespace dwave::optimization

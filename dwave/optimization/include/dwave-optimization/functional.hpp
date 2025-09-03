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

#include <cmath>
#include <cstdlib>

namespace dwave::optimization::functional {

template <class T>
struct abs {
    constexpr T operator()(const T& x) const { return std::abs(x); }
};

template <class T>
struct cos {
    auto operator()(const T& num) const { return std::cos(num); }
};

template <class T>
struct exp {
    constexpr auto operator()(const T& x) const { return std::exp(x); }
};

template <class T>
struct expit {
    constexpr double operator()(const T& x) const { return 1.0 / (1.0 + std::exp(-1. * x)); }
};

template <class T>
struct log {
    constexpr auto operator()(const T& x) const { return std::log(x); }
};

template <class T>
struct logical {
    constexpr bool operator()(const T& x) const { return x; }
};

template <class T>
struct logical_xor {
    constexpr bool operator()(const T& x, const T& y) const {
        return static_cast<bool>(x) != static_cast<bool>(y);
    }
};

template <class T>
struct max {
    constexpr T operator()(const T& x, const T& y) const { return std::max(x, y); }
};

template <class T>
struct min {
    constexpr T operator()(const T& x, const T& y) const { return std::min(x, y); }
};

template <class T>
struct modulus {
    constexpr T operator()(const T& x, const T& y) const {
        // Copy numpy behavior and return 0 for `x % 0`
        if (y == 0) return 0;

        T result;
        if constexpr (std::integral<T>) {
            result = std::div(x, y).rem;
        } else {
            result = std::fmod(x, y);
        }

        if ((std::signbit(x) != std::signbit(y)) && (result != 0)) {
            // Make result consistent with numpy for different-sign arguments
            result += y;
        }

        return result;
    }
};

template <class T>
struct rint {
    constexpr auto operator()(const T& x) const { return std::rint(x); }
};

template <class T>
struct safe_divides {
    constexpr T operator()(const T& lhs, const T& rhs) const {
        if (!rhs) return 0;
        return lhs / rhs;
    }
};

template <class T>
struct sin {
    auto operator()(const T& num) const { return std::sin(num); }
};

template <class T>
struct square {
    constexpr T operator()(const T& x) const { return x * x; }
};

template <class T>
struct square_root {
    constexpr auto operator()(const T& x) const { return std::sqrt(x); }
};

}  // namespace dwave::optimization::functional

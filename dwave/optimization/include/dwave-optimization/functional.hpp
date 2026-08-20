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

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdlib>

namespace dwave::optimization::functional {

template <class T>
struct abs {
    static constexpr T operator()(const T& x) { return std::abs(x); }
};

template <class T>
struct cos {
    static auto operator()(const T& num) { return std::cos(num); }
};

template <class T>
struct exp {
    static constexpr auto operator()(const T& x) { return std::exp(x); }
};

template <class T>
struct expit {
    static constexpr double operator()(const T& x) { return 1.0 / (1.0 + std::exp(-1. * x)); }
};

template <class T>
struct log {
    static constexpr auto operator()(const T& x) { return std::log(x); }
};

template <class T>
struct logical {
    static constexpr bool operator()(const T& x) { return x; }
};

template <class T>
struct logical_xor {
    static constexpr bool operator()(const T& x, const T& y) {
        return static_cast<bool>(x) != static_cast<bool>(y);
    }
};

template <class T>
struct max {
    static constexpr T operator()(const T& x, const T& y) { return std::max(x, y); }
};

template <class T>
struct min {
    static constexpr T operator()(const T& x, const T& y) { return std::min(x, y); }
};

template <class T>
struct modulus {
    static constexpr T operator()(const T& x, const T& y) {
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
    static constexpr auto operator()(const T& x) { return std::rint(x); }
};

template <class T>
struct safe_divides {
    static constexpr T operator()(const T& lhs, const T& rhs) {
        if (!rhs) return 0;
        return lhs / rhs;
    }
};

template <class T>
struct sin {
    static auto operator()(const T& num) { return std::sin(num); }
};

template <class T>
struct square {
    static constexpr T operator()(const T& x) { return x * x; }
};

template <class T>
struct square_root {
    static constexpr auto operator()(const T& x) { return std::sqrt(x); }
};

template <class T>
struct tanh {
    static auto operator()(const T& num) { return std::tanh(num); }
};

}  // namespace dwave::optimization::functional

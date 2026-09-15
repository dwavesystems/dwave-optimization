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
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstdlib>
#include <limits>
#include <utility>

#include "dwave-optimization/interval.hpp"
#include "dwave-optimization/typing.hpp"

namespace dwave::optimization::functional {

enum class Monotonicity { Decreasing = -1, None = 0, Increasing = 1 };

template <typename UnaryOp>
struct UnaryOpMixin {
    template <DType T>
    requires(UnaryOp::monotonic != Monotonicity::None)
    static auto operator()(const interval<T>& domain) {
        using return_type = interval<decltype(UnaryOp::operator()(T()))>;

        // op(empty domain) -> empty domain
        if (not static_cast<bool>(domain)) return return_type();

        assert(
            domain <= UnaryOp::template domain<T> and
            "input domain must be a subset of the func's domain"
        );

        // We don't worry about outward rounding here because this overload is meant
        // to reflect the behavior of the scalar overload, not necessarily to be
        // mathematically correct.
        // We *do* assume that UnaryOp (e.g., std::exp()) is monotonic, which
        // is not always true, but I think it's an OK assumption for our purposes.
        if constexpr (UnaryOp::monotonic == Monotonicity::Increasing) {
            return return_type(
                UnaryOp::operator()(domain.infimum), UnaryOp::operator()(domain.supremum)
            );
        } else if constexpr (UnaryOp::monotonic == Monotonicity::Decreasing) {
            return return_type(
                UnaryOp::operator()(domain.supremum), UnaryOp::operator()(domain.infimum)
            );
        } else {
            assert(false and "unexpected monotonicity");
            std::unreachable();
        }
    }

    template <DType T>
    static constexpr interval<T> domain = interval<T>::all();
};

struct absolute : UnaryOpMixin<absolute> {
    template <DType T>
    static T operator()(const T& x) {
        // Unlike NumPy/std, we define std::abs(INT_MIN) to equal INT_MAX under the reasoning
        // that it's more important to us to preserve the sign than to preseve the correct value.
        if constexpr (std::integral<T>) {
            if (x == std::numeric_limits<T>::lowest()) return std::numeric_limits<T>::max();
        }

        // std::abs() is not defined for int8 or int16 so we static_cast to avoid widening.
        return static_cast<T>(std::abs(x));
    }
    static bool operator()(const bool& x) { return x; }

    template <DType T>
    static interval<T> operator()(const interval<T>& domain) {
        if (not static_cast<bool>(domain)) return {};  // op(empty domain) -> empty domain

        assert(domain.infimum <= domain.supremum);  // implied by non-empty

        // If the domain is non-negative, then absolute is identity
        if (0 <= domain.infimum) return domain;

        // If the domain is negative, then absolute is just the inverse
        if (domain.supremum < 0) return -domain;

        // Otherwise, the domain straddles 0

        // Handle the -INT_MIN case. Again we treat abs(-INT_MIN) as INT_MAX under the reasoning
        // that [INT_MIN, ...] is probably intended to mean unbounded.
        if constexpr (std::integral<T>) {
            if (domain.infimum == std::numeric_limits<T>::lowest()) {
                return interval<T>(0, std::numeric_limits<T>::max());
            }
        }

        return interval<T>(
            0, -domain.infimum < domain.supremum ? domain.supremum : -domain.infimum
        );
    }
    static interval<bool> operator()(const interval<bool>& domain) { return domain; }

    static constexpr Monotonicity monotonic = Monotonicity::None;
};

struct cos : UnaryOpMixin<cos> {
    static auto operator()(const DType auto& x) { return std::cos(x); }

    template <DType T>
    static interval<decltype(std::cos(T()))> operator()(const interval<T>& domain) {
        if (not static_cast<bool>(domain)) return {};  // op(empty domain) -> empty domain

        // It is possible to be a lot more specific than this by checking whether
        // our domain spans a full period or not, but I think this is of dubious
        // benefit to the user so for now we just return [-1, +1]
        return {-1, +1};
    }

    static constexpr Monotonicity monotonic = Monotonicity::None;
};

struct exp : UnaryOpMixin<exp> {
    static auto operator()(const DType auto& x) { return std::exp(x); }
    using UnaryOpMixin::operator();

    static constexpr Monotonicity monotonic = Monotonicity::Increasing;
};

struct expit : UnaryOpMixin<expit> {
    template <DType T>
    static auto operator()(const T& x) {
        return 1 / (1 + std::exp(-x));
    }
    using UnaryOpMixin::operator();

    static constexpr Monotonicity monotonic = Monotonicity::Increasing;
};

struct log : UnaryOpMixin<log> {
    template <DType T>
    static auto operator()(const T& x) {
        assert(domain<T>.contains(x) and "x must be non-negative");
        return std::log(x);
    }
    using UnaryOpMixin::operator();

    template <DType T>
    static constexpr interval<T> domain = interval<T>::nonnegative();

    static constexpr Monotonicity monotonic = Monotonicity::Increasing;
};

struct logical : UnaryOpMixin<logical> {
    static bool operator()(const DType auto& x) { return x; }

    static interval<bool> operator()(const interval<bool>& domain) { return domain; }
    template <DType T>
    static interval<bool> operator()(const interval<T>& domain) {
        if (not static_cast<bool>(domain)) return {};  // op(empty domain) -> empty domain

        if (domain.infimum == 0 and domain.supremum == 0) return interval(false, false);
        if (domain.infimum <= 0 and domain.supremum >= 0) return interval(false, true);
        return interval(true, true);
    }

    static constexpr Monotonicity monotonic = Monotonicity::None;
};

struct logical_not : UnaryOpMixin<logical_not> {
    static bool operator()(const DType auto& x) { return not x; }

    static interval<bool> operator()(const interval<bool>& domain) {
        if (not static_cast<bool>(domain)) return {};  // op(empty domain) -> empty domain
        return interval(not domain.supremum, not domain.infimum);
    }
    template <DType T>
    static interval<bool> operator()(const interval<T>& domain) {
        // Call the more specific interval<bool> overload
        return operator()(logical{}(domain));
    }

    static constexpr Monotonicity monotonic = Monotonicity::None;
};

template <class T>
struct logical_xor {
    static bool operator()(const T& x, const T& y) {
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

struct negative : UnaryOpMixin<negative> {
    template <class T>
    requires(DType<T> and not std::same_as<T, bool>)  // not defined for bool
    static auto operator()(const T& x) {
        // We define -INT_MIN to equal INT_MAX under the reasoning that it's more
        // important to us to preserve the sign than to preseve the correct value.
        if constexpr (std::integral<T>) {
            if (x == std::numeric_limits<T>::lowest()) return std::numeric_limits<T>::max();
        }

        return static_cast<T>(-x);  // so it doesn't widen e.g., int8_t->int
    }
    using UnaryOpMixin::operator();

    static constexpr Monotonicity monotonic = Monotonicity::Decreasing;
};

struct rint : UnaryOpMixin<rint> {
    static auto operator()(const DType auto& x) { return std::rint(x); }
    using UnaryOpMixin::operator();

    static constexpr Monotonicity monotonic = Monotonicity::Increasing;
};

template <class T>
struct safe_divides {
    static constexpr T operator()(const T& lhs, const T& rhs) {
        if (!rhs) return 0;
        return lhs / rhs;
    }
};

struct sin : UnaryOpMixin<sin> {
    static auto operator()(const DType auto& x) { return std::sin(x); }

    template <DType T>
    static interval<decltype(std::sin(T()))> operator()(const interval<T>& domain) {
        if (not static_cast<bool>(domain)) return {};  // op(empty domain) -> empty domain

        // It is possible to be a lot more specific than this by checking whether
        // our domain spans a full period or not, but I think this is of dubious
        // benefit to the user so for now we just return [-1, +1]
        return {-1, +1};
    }

    static constexpr Monotonicity monotonic = Monotonicity::None;
};

struct square : UnaryOpMixin<square> {
    template <DType T>
    static T operator()(const T& x) {
        return x * x;
    }
    static bool operator()(const bool& x) { return x; }

    template <DType T>
    static interval<T> operator()(const interval<T>& domain) {
        if (not static_cast<bool>(domain)) return {};  // op(empty domain) -> empty domain

        assert(domain.infimum <= domain.supremum);  // implied by non-empty

        square op{};
        T inf_squared = op(domain.infimum);
        T sup_squared = op(domain.supremum);

        // Non-negative domain: square is increasing
        if (0 <= domain.infimum) return interval<T>(inf_squared, sup_squared);

        // Non-positive domain: square is decreasing
        if (domain.supremum <= 0) return interval<T>(sup_squared, inf_squared);

        // Otherwise the domain straddles 0: minimum is 0, maximum is the larger squared endpoint.

        return interval<T>(0, inf_squared < sup_squared ? sup_squared : inf_squared);
    }
    static interval<bool> operator()(const interval<bool>& domain) { return domain; }

    static constexpr Monotonicity monotonic = Monotonicity::None;
};

struct square_root : UnaryOpMixin<square_root> {
    template <DType T>
    static auto operator()(const T& x) {
        assert(domain<T>.contains(x) and "x must be non-negative");
        return std::sqrt(x);
    }
    using UnaryOpMixin::operator();

    template <DType T>
    static constexpr interval<T> domain = interval<T>::nonnegative();

    static constexpr Monotonicity monotonic = Monotonicity::Increasing;
};

struct tanh : UnaryOpMixin<tanh> {
    static auto operator()(const DType auto& num) { return std::tanh(num); }
    using UnaryOpMixin::operator();

    static constexpr Monotonicity monotonic = Monotonicity::Increasing;
};

}  // namespace dwave::optimization::functional

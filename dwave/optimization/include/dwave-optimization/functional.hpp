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

#include "dwave-optimization/interval.hpp"
#include "dwave-optimization/typing.hpp"

namespace dwave::optimization::functional {

enum class Monotonicity { Decreasing = -1, None = 0, Increasing = 1 };

namespace mixins {

template <typename UnaryOp>
struct UnaryOpMixin {
    /// For monotonic unary ops, calculate the interval extension of the scalar overload.
    template <DType T>
    requires(UnaryOp::monotonicity[0] != Monotonicity::None and requires {
        UnaryOp::operator()(T());
    }) static constexpr auto operator()(const interval<T>& x_enclosure) {
        assert(static_cast<bool>(x_enclosure) and "x's enclosure cannot be empty");
        assert(
            x_enclosure <= UnaryOp::template domain<T>[0] and
            "x's enclosure must be a subset of the unary op's domain"
        );

        using return_type = interval<decltype(UnaryOp::operator()(T()))>;

        // We don't worry about outward rounding here because this overload is meant
        // to reflect the behavior of the scalar overload, not necessarily to be
        // mathematically correct.
        // We *do* assume that UnaryOp (e.g., std::exp()) is monotonic, which
        // is not always true, but I think it's an OK assumption for our purposes.
        if constexpr (UnaryOp::monotonicity[0] == Monotonicity::Increasing) {
            return return_type(
                UnaryOp::operator()(x_enclosure.infimum), UnaryOp::operator()(x_enclosure.supremum)
            );
        } else if constexpr (UnaryOp::monotonicity[0] == Monotonicity::Decreasing) {
            return return_type(
                UnaryOp::operator()(x_enclosure.supremum), UnaryOp::operator()(x_enclosure.infimum)
            );
        } else {
            static_assert(false, "unexpected monotonicity");
        }
    }

    /// The domain of the operator. `domain[n]` is the nth factor of the domain.
    /// Unary ops are assumed to be defined for all possible inputs unless they tell us otherwise.
    template <DType T>
    static constexpr std::array<interval<T>, 1> domain{interval<T>::all()};
    // Note: because we use intervals to encode the domain, this is technically the bounding
    // box rather than the domain.

    /// The montonicity of the operator. `monotonicity[n]` is the monotonicity of the nth argument.
    /// Unary ops are assumed not to be monotonic unless they tell us otherwise.
    static constexpr std::array<Monotonicity, 1> monotonicity{Monotonicity::None};
};

}  // namespace mixins

struct absolute : mixins::UnaryOpMixin<absolute> {
    /// Calculate the absolute value of `x`.
    template <DType T>
    static constexpr T operator()(T x) noexcept {
        if constexpr (std::same_as<T, bool>) {
            return x;
        } else if constexpr (std::signed_integral<T>) {
            // NumPy defines `absolute(INT_MIN) := INT_MIN` whereas we define
            // `absolute(INT_MIN) := INT_MAX` in order to preserve the sign.
            if (x == std::numeric_limits<T>::min()) return std::numeric_limits<T>::max();
            // std::abs() will widen int8_t or int16_t, so we add a static cast
            return static_cast<T>(std::abs(x));
        } else if constexpr (std::floating_point<T>) {
            assert(not std::isnan(x) and "x cannot be nan");
            // std::abs() will not widen any of the floating point we care about
            return std::abs(x);
        } else {
            static_assert(false, "unexpected dtype");
        }
    }

    /// Calculate the interval extension of `x`'s enclosure.
    template <DType T>
    static interval<T> operator()(const interval<T>& x_enclosure) noexcept {
        assert(static_cast<bool>(x_enclosure) and "x's enclosure cannot be empty");

        if constexpr (std::same_as<T, bool>) {
            return x_enclosure;
        } else {
            // If the domain is non-negative, then absolute is identity
            if (0 <= x_enclosure.infimum) return x_enclosure;

            // If x is always negative, then absolute is just the inverse
            if (x_enclosure.supremum < 0) {
                return interval<T>(
                    operator()(x_enclosure.supremum), operator()(x_enclosure.infimum)
                );
            }

            // Otherwise, the domain straddles 0
            return interval<T>(0, operator()(x_enclosure.infimum)) |
                   interval<T>(0, operator()(x_enclosure.supremum));
        }
    }
};

struct cos : mixins::UnaryOpMixin<cos> {
    // dev note: these are marked constexpr even though std::cos() isn't
    // actually constexpr until C++26. Luckily everything works fine with
    // this approach and it's a bit more future-proof.

    /// Calculate the cosine of `x`.
    /// We want to disallow `nan`s so we define `cos(+/-inf) := +0.0`.
    template <DType T>
    static constexpr auto operator()(T x) {
        if constexpr (std::floating_point<T>) {
            assert(not std::isnan(x) and "x cannot be nan");

            if (std::isinf(x)) return T{0};
        }

        // NumPy uses the smallest floating point it can and we follow.
        if constexpr (can_cast<T, float>) {
            return std::cosf(x);
        } else if constexpr (can_cast<T, double>) {
            return std::cos(x);
        } else {
            static_assert(false, "unexpected dtype");
        }
    }

    /// Calculate the interval extension of `x`'s enclosure.
    template <DType T>
    static constexpr auto operator()(const interval<T>&) {
        // It is possible to be a lot more specific than this by checking whether
        // our domain spans a full period or not, but I think this is of dubious
        // benefit to the user so for now we just return [-1, +1]
        using return_type = decltype(operator()(T()));
        return interval<return_type>(return_type(-1), return_type(+1));
    }
};

struct exp : mixins::UnaryOpMixin<exp> {
    // dev note: these are marked constexpr even though std::exp() isn't
    // actually constexpr until C++26. Luckily everything works fine with
    // this approach and it's a bit more future-proof.

    template <DType T>
    static constexpr auto operator()(T x) {
        assert((std::integral<T> or not std::isnan(x)) and "x cannot be nan");

        // NumPy uses the smallest floating point it can and we follow.
        if constexpr (can_cast<T, float>) {
            return std::expf(x);
        } else if constexpr (can_cast<T, double>) {
            return std::exp(x);
        } else {
            static_assert(false, "unexpected dtype");
        }
    }
    using UnaryOpMixin::operator();

    static constexpr std::array<Monotonicity, 1> monotonicity{Monotonicity::Increasing};
};

struct expit : mixins::UnaryOpMixin<expit> {
    template <DType T>
    static constexpr auto operator()(T x) {
        // Inherit our promotion rules from exp to match SciPy's behavior
        if constexpr (std::same_as<T, bool>) return operator()(static_cast<signed char>(x));
        const auto y = exp{}(static_cast<T>(-x));
        return static_cast<decltype(y)>(1 / (1 + y));
    }
    using UnaryOpMixin::operator();

    static constexpr std::array<Monotonicity, 1> monotonicity{Monotonicity::Increasing};
};

struct log : mixins::UnaryOpMixin<log> {
    // dev note: these are marked constexpr even though std::log() isn't
    // actually constexpr until C++26. Luckily everything works fine with
    // this approach and it's a bit more future-proof.

    template <DType T>
    static constexpr auto operator()(T x) {
        assert((std::integral<T> or not std::isnan(x)) and "x cannot be nan");
        assert(domain<T>[0].contains(x) and "x must be non-negative");

        // NumPy uses the smallest floating point it can and we follow.
        if constexpr (can_cast<T, float>) {
            return std::logf(x);
        } else if constexpr (can_cast<T, double>) {
            return std::log(x);
        } else {
            static_assert(false, "unexpected dtype");
        }
    }
    using UnaryOpMixin::operator();

    template <DType T>
    static constexpr std::array<interval<T>, 1> domain{interval<T>::nonnegative()};

    static constexpr std::array<Monotonicity, 1> monotonicity{Monotonicity::Increasing};
};

struct logical : mixins::UnaryOpMixin<logical> {
    template <DType T>
    static constexpr bool operator()(T x) {
        assert((std::integral<decltype(x)> or not std::isnan(x)) and "x cannot be nan");
        return static_cast<bool>(x);
    }

    template <DType T>
    static constexpr interval<bool> operator()(const interval<T>& x_enclosure) {
        assert(static_cast<bool>(x_enclosure) and "x's enclosure cannot be empty");

        if constexpr (std::same_as<T, bool>) {
            return x_enclosure;
        } else {
            const auto& [inf, sup] = x_enclosure;

            // If x is pinned to 0 then we're strictly false
            if (inf == false and sup == false) return interval<bool>(false, false);

            // If x is strictly positive or strictly negative, then we're strictly true
            if (sup < 0 or 0 < inf) return interval<bool>(true, true);

            // Otherwise it's ambiguous
            return interval<bool>::all();
        }
    }
};

struct logical_not : mixins::UnaryOpMixin<logical_not> {
    template <DType T>
    static constexpr bool operator()(T x) {
        return not logical{}(x);
    }

    template <DType T>
    static constexpr interval<bool> operator()(const interval<T>& x_enclosure) {
        assert(static_cast<bool>(x_enclosure) and "x's enclosure cannot be empty");
        if constexpr (std::same_as<T, bool>) {
            // The main path. Simplify negate the interval
            return interval<bool>(not x_enclosure.supremum, not x_enclosure.infimum);
        } else {
            // Otherwise get the boolean value associate with our enclosure and then
            // go through the main path
            return operator()(logical{}(x_enclosure));
        }
    }
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

struct negative : mixins::UnaryOpMixin<negative> {
    template <DType T>
    requires(not std::same_as<T, bool>)  // not defined for bool
    static constexpr auto operator()(T x) {
        // We define -INT_MIN to equal INT_MAX under the reasoning that it's more
        // important to us to preserve the sign than to preseve the correct value.
        if constexpr (std::signed_integral<T>) {
            if (x == std::numeric_limits<T>::lowest()) return std::numeric_limits<T>::max();
        }

        return static_cast<T>(-x);  // so it doesn't widen e.g., int8_t->int
    }
    using UnaryOpMixin::operator();

    static constexpr std::array<Monotonicity, 1> monotonicity{Monotonicity::Decreasing};
};

struct rint : mixins::UnaryOpMixin<rint> {
    // dev note: these are marked constexpr even though std::rint() isn't
    // actually constexpr in any C++ std as of 2026.
    // Luckily everything works fine with this approach and it's a bit more future-proof.

    template <DType T>
    static constexpr auto operator()(T x) {
        // NumPy uses the smallest floating point it can and we follow.
        if constexpr (can_cast<T, float>) {
            return std::rintf(x);
        } else if constexpr (can_cast<T, double>) {
            return std::rint(x);
        } else {
            static_assert(false, "unexpected dtype");
        }
    }
    using UnaryOpMixin::operator();

    static constexpr std::array<Monotonicity, 1> monotonicity{Monotonicity::Increasing};
};

template <class T>
struct safe_divides {
    static constexpr T operator()(const T& lhs, const T& rhs) {
        if (!rhs) return 0;
        return lhs / rhs;
    }
};

struct sin : mixins::UnaryOpMixin<sin> {
    // dev note: these are marked constexpr even though std::sin() isn't
    // actually constexpr until C++26. Luckily everything works fine with
    // this approach and it's a bit more future-proof.

    /// Calculate the sine of `x`.
    template <DType T>
    static constexpr auto operator()(T x) {
        // NumPy uses the smallest floating point it can and we follow.
        if constexpr (can_cast<T, float>) {
            return std::sinf(x);
        } else if constexpr (can_cast<T, double>) {
            return std::sin(x);
        } else {
            static_assert(false, "unexpected dtype");
        }
    }

    /// Calculate the interval extension of `x`'s enclosure.
    template <DType T>
    static constexpr auto operator()(const interval<T>&) {
        // It is possible to be a lot more specific than this by checking whether
        // our domain spans a full period or not, but I think this is of dubious
        // benefit to the user so for now we just return [-1, +1]
        using return_type = decltype(operator()(T()));
        return interval<return_type>(return_type(-1), return_type(+1));
    }
};

struct sqrt : mixins::UnaryOpMixin<sqrt> {
    // dev note: these are marked constexpr even though std::sqrt() isn't
    // actually constexpr until C++26. Luckily everything works fine with
    // this approach and it's a bit more future-proof.

    template <DType T>
    static constexpr auto operator()(T x) {
        assert(domain<T>[0].contains(x) and "x must be non-negative");

        // NumPy uses the smallest floating point it can and we follow.
        if constexpr (can_cast<T, float>) {
            return std::sqrtf(x);
        } else if constexpr (can_cast<T, double>) {
            return std::sqrt(x);
        } else {
            static_assert(false, "unexpected dtype");
        }
    }
    using UnaryOpMixin::operator();

    template <DType T>
    static constexpr std::array<interval<T>, 1> domain{interval<T>::nonnegative()};

    static constexpr std::array<Monotonicity, 1> monotonicity{Monotonicity::Increasing};
};

struct square : mixins::UnaryOpMixin<square> {
    template <DType T>
    static constexpr T operator()(T x) {
        if constexpr (std::same_as<T, bool>) {
            return x;
        } else if constexpr (std::signed_integral<T>) {
            using limits = std::numeric_limits<T>;

#if !defined(DWOPT__FORCE_FALLBACK) && defined(__has_builtin)
#if __has_builtin(__builtin_mul_overflow)  // needs its own line
            // We really want C++26 std::saturating_mul, but while we're on C++23
            // we use the __builtin_mul_overflow (GCC and Clang) if it's available.
            if (T res; not __builtin_mul_overflow(x, x, &res)) return res;
            return limits::max();
#endif
#endif
            // Otherwise, fallback to a simple std-only implementation.
            if (x > 0 and x > limits::max() / x) return limits::max();
            if (x < 0 and x < limits::max() / x) return limits::max();
            return x * x;
        } else if constexpr (std::floating_point<T>) {
            return x * x;
        } else {
            static_assert(false, "unexpected dtype");
        }
    }

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
};

struct tanh : mixins::UnaryOpMixin<tanh> {
    // dev note: these are marked constexpr even though std::tanh() isn't
    // actually constexpr until C++26. Luckily everything works fine with
    // this approach and it's a bit more future-proof.

    template <DType T>
    static constexpr auto operator()(T x) {
        // NumPy uses the smallest floating point it can and we follow.
        if constexpr (can_cast<T, float>) {
            return std::tanhf(x);
        } else if constexpr (can_cast<T, double>) {
            return std::tanh(x);
        } else {
            static_assert(false, "unexpected dtype");
        }
    }
    using UnaryOpMixin::operator();

    static constexpr std::array<Monotonicity, 1> monotonicity{Monotonicity::Increasing};
};

}  // namespace dwave::optimization::functional

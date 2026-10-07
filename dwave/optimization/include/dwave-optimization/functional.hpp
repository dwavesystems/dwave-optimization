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

template <typename BinaryOp>
struct BinaryOpMixin {
    template <DType T>
    requires(
        BinaryOp::monotonicity[0] != Monotonicity::None and
        BinaryOp::monotonicity[1] != Monotonicity::None and
        requires { BinaryOp::operator()(T(), T()); }
    )
    static constexpr auto operator()(
        const interval<T>& lhs_enclosure,
        const interval<T>& rhs_enclosure
    ) {
        assert(static_cast<bool>(lhs_enclosure) and "lhs's enclosure cannot be empty");
        assert(static_cast<bool>(rhs_enclosure) and "rhs's enclosure cannot be empty");

        assert(
            lhs_enclosure <= BinaryOp::template domain<T>[0] and
            "lhs's enclosure must be a subset of the binary op's domain"
        );
        assert(
            rhs_enclosure <= BinaryOp::template domain<T>[1] and
            "rhs's enclosure must be a subset of the binary op's domain"
        );

        using return_type = interval<decltype(BinaryOp::operator()(T(), T()))>;

        const return_type out(
            BinaryOp::operator()(
                BinaryOp::monotonicity[0] == Monotonicity::Increasing ? lhs_enclosure.infimum
                                                                      : lhs_enclosure.supremum,
                BinaryOp::monotonicity[1] == Monotonicity::Increasing ? rhs_enclosure.infimum
                                                                      : rhs_enclosure.supremum
            ),
            BinaryOp::operator()(
                BinaryOp::monotonicity[0] == Monotonicity::Increasing ? lhs_enclosure.supremum
                                                                      : lhs_enclosure.infimum,
                BinaryOp::monotonicity[1] == Monotonicity::Increasing ? rhs_enclosure.supremum
                                                                      : rhs_enclosure.infimum
            )
        );

        // If our intervals are non-empty and we get an empty output, then either our
        // op isn't actually monotonic or we have another bug.
        assert(static_cast<bool>(out));

        return out;
    }

    /// The domain of the operator. `domain[n]` is the nth factor of the domain.
    /// Binary ops are assumed to be defined for all possible inputs unless they tell us otherwise.
    template <DType T>
    static constexpr std::array<interval<T>, 2> domain{interval<T>::all(), interval<T>::all()};
    // Note: because we use intervals to encode the domain, this is technically the bounding
    // box rather than the domain.

    /// The montonicity of the operator. `monotonicity[n]` is the monotonicity of the nth argument.
    /// Binary ops are assumed not to be monotonic unless they tell us otherwise.
    static constexpr std::array<Monotonicity, 2> monotonicity{
        Monotonicity::None,
        Monotonicity::None
    };
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

struct add : mixins::BinaryOpMixin<add> {
    template <DType T>
    static constexpr T operator()(T lhs, T rhs) {
        using limits = std::numeric_limits<T>;

        if constexpr (std::same_as<T, bool>) {
            // NumPy treats addition over booleans as logical or.
            // We use bitwise or because it is vectorizable.
            return lhs | rhs;
        } else if constexpr (std::signed_integral<T>) {
            // For integers we do saturating addition.
            // In C++26 we could use std::saturating_add() but for now we backport it

#if !defined(DWOPT__FORCE_FALLBACK) && defined(__has_builtin)
#if __has_builtin(__builtin_add_overflow)  // needs its own line
            // We really want C++26 std::__builtin_add_overflow, but while we're on C++23
            // we use the __builtin_add_overflow (GCC and Clang) if it's available.
            if (T out; not __builtin_add_overflow(lhs, rhs, &out)) return out;
            if (lhs < 0) {
                return limits::lowest();
            } else {
                return limits::max();
            }
#endif
#endif
            // This fallback is overkill but I had already written it before deciding to use
            // __builtin_add_overflow so might as well use it.
            // Implementation follows https://locklessinc.com/articles/sat_arithmetic/
            // I have tried to mirror the implementation from that link using our code style

            using UT = std::make_unsigned_t<T>;

            UT ux = static_cast<UT>(lhs);
            UT uy = static_cast<UT>(rhs);
            UT out = ux + uy;

            ux = (ux >> limits::digits) + limits::max();

            if (static_cast<T>((ux ^ uy) | ~(uy ^ out)) >= 0) {
                out = ux;
            }

            return static_cast<T>(out);
        } else if constexpr (std::floating_point<T>) {
            assert(not std::isnan(lhs) and "lhs cannot be nan");
            assert(not std::isnan(rhs) and "rhs cannot be nan");

            // -inf + inf would result in NaN so we define it to be inf
            if (std::isinf(lhs) and -lhs == rhs) return limits::infinity();

            return lhs + rhs;
        } else {
            static_assert(false, "unsupported dtype");
        }
    }
    using BinaryOpMixin::operator();

    static constexpr std::array<Monotonicity, 2> monotonicity{
        Monotonicity::Increasing,
        Monotonicity::Increasing
    };
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

struct divide : mixins::BinaryOpMixin<divide> {
    /// Calculate lhs / rhs. We implement "safe" division, so `x / 0 := 0`
    template <DType T>
    static constexpr auto operator()(T lhs, T rhs) {
        // Unlike most other ufuncs, np.divide promotes all integer types (including bool)
        // to double.
        if constexpr (std::integral<T>) {
            return operator()(static_cast<double>(lhs), static_cast<double>(rhs));
        } else {
            // We want to be safe (i.e. lhs / 0 := 0)
            if (not rhs) return std::signbit(lhs) == std::signbit(rhs) ? T{0} : -T{0};

            // Also handle the case that can result in nan
            if (std::isinf(lhs) and std::isinf(rhs)) {
                if (std::signbit(lhs) == std::signbit(rhs)) {
                    return +std::numeric_limits<T>::infinity();
                } else {
                    return -std::numeric_limits<T>::infinity();
                }
            }

            return lhs / rhs;
        }
    }

    template <DType T>
    static constexpr auto operator()(
        const interval<T>& lhs_enclosure,
        const interval<T>& rhs_enclosure
    ) {
        // If either lhs or rhs is an empty domain, then so is their division
        assert(static_cast<bool>(lhs_enclosure) and "lhs's enclosure cannot be empty");
        assert(static_cast<bool>(rhs_enclosure) and "rhs's enclosure cannot be empty");

        if constexpr (std::integral<T>) {
            // Unlike most other ufuncs, np.divide promotes all integer types (including bool)
            // to double.
            return operator()(interval<double>(lhs_enclosure), interval<double>(rhs_enclosure));
        } else {
            const auto& [lhs_infimum, lhs_supremum] = lhs_enclosure;
            const auto& [rhs_infimum, rhs_supremum] = rhs_enclosure;

            // If either lhs or rhs is pinned to zero then so is our function
            if (lhs_infimum == 0 and lhs_supremum == 0) return interval<T>(0, 0);
            if (rhs_infimum == 0 and rhs_supremum == 0) return interval<T>(0, 0);

            // If rhs straddles 0, then we can get arbitrarily close to +/- inf
            if (rhs_infimum < 0 and 0 < rhs_supremum) return interval<T>::all();

            // if rhs is non-negative or non-positive then the sign of lhs determines our range of
            // outputs
            if (rhs_infimum == 0 and 0 < rhs_supremum) {
                if (0 <= lhs_infimum) return interval<T>::nonnegative();
                if (lhs_supremum <= 0) return interval<T>::nonpositive();
                return interval<T>::all();
            }
            if (rhs_infimum < 0 and rhs_supremum == 0) {
                if (0 <= lhs_infimum) return interval<T>::nonpositive();
                if (lhs_supremum <= 0) return interval<T>::nonnegative();
                return interval<T>::all();
            }

            // Otherwise we just need to include our four corners
            interval<T> out;
            out |= divide{}(lhs_infimum, rhs_infimum);
            out |= divide{}(lhs_infimum, rhs_supremum);
            out |= divide{}(lhs_supremum, rhs_infimum);
            out |= divide{}(lhs_supremum, rhs_supremum);
            return out;
        }
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

struct less_equal : mixins::BinaryOpMixin<less_equal> {
    template <DType T>
    static constexpr bool operator()(T lhs, T rhs) {
        assert((std::integral<T> or not std::isnan(lhs)) and "lhs cannot be nan");
        assert((std::integral<T> or not std::isnan(rhs)) and "rhs cannot be nan");

        return lhs <= rhs;
    }
    using BinaryOpMixin::operator();

    static constexpr std::array<Monotonicity, 2> monotonicity{
        Monotonicity::Decreasing,
        Monotonicity::Increasing
    };
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

struct logical_and : mixins::BinaryOpMixin<logical_and> {
    template <DType T>
    static constexpr bool operator()(T lhs, T rhs) {
        assert((std::integral<T> or not std::isnan(lhs)) and "lhs cannot be nan");
        assert((std::integral<T> or not std::isnan(rhs)) and "rhs cannot be nan");

        // bitwise and is vectorizable
        return static_cast<bool>(lhs) & static_cast<bool>(rhs);
    }

    template <DType T>
    static constexpr interval<bool> operator()(
        const interval<T>& lhs_enclosure,
        const interval<T>& rhs_enclosure
    ) {
        assert(static_cast<bool>(lhs_enclosure) and "lhs's enclosure cannot be empty");
        assert(static_cast<bool>(rhs_enclosure) and "rhs's enclosure cannot be empty");

        // if both lhs and rhs only contain truthy values, then our output will
        // always be true
        if (not lhs_enclosure.contains(0) and not rhs_enclosure.contains(0)) return {true, true};

        // if lhs is always falsy, then our output will always be false
        if (lhs_enclosure.contains(0) and lhs_enclosure.infimum == lhs_enclosure.supremum)
            return {false, false};

        // likewise for rhs
        if (rhs_enclosure.contains(0) and rhs_enclosure.infimum == rhs_enclosure.supremum)
            return {false, false};

        // Otherwise we can output either true or false
        return {false, true};
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

struct logical_or : mixins::BinaryOpMixin<logical_or> {
    template <DType T>
    static constexpr bool operator()(T lhs, T rhs) {
        // bitwise or is vectorizable
        return static_cast<bool>(lhs) | static_cast<bool>(rhs);
    }

    template <DType T>
    static constexpr interval<bool> operator()(
        const interval<T>& lhs_enclosure,
        const interval<T>& rhs_enclosure
    ) {
        assert(static_cast<bool>(lhs_enclosure) and "lhs's enclosure cannot be empty");
        assert(static_cast<bool>(rhs_enclosure) and "rhs's enclosure cannot be empty");

        // if lhs only contains truthy values, then our output will always be true
        if (not lhs_enclosure.contains(0)) return {true, true};

        // likewise for rhs
        if (not rhs_enclosure.contains(0)) return {true, true};

        // Both lhs and rhs contain 0, so either being a single value makes it
        // always falsy. If both are, then our output will always be false
        if (lhs_enclosure.infimum == lhs_enclosure.supremum and
            rhs_enclosure.infimum == rhs_enclosure.supremum)
            return {false, false};

        // Otherwise we can output either true or false
        return {false, true};
    }
};

struct logical_xor : mixins::BinaryOpMixin<logical_xor> {
    template <DType T>
    static constexpr bool operator()(T lhs, T rhs) {
        // use ^ rather than xor here for symmetry with logical_and/logical_or even though
        // for xor they are the same thing.
        return static_cast<bool>(lhs) ^ static_cast<bool>(rhs);
    }

    template <DType T>
    static constexpr interval<bool> operator()(
        const interval<T>& lhs_enclosure,
        const interval<T>& rhs_enclosure
    ) {
        assert(static_cast<bool>(lhs_enclosure) and "lhs's enclosure cannot be empty");
        assert(static_cast<bool>(rhs_enclosure) and "rhs's enclosure cannot be empty");

        // A domain fixes its truthiness if it excludes 0 (always truthy) or if it holds
        // a single value (truthy or falsy, but not both)
        const bool lhs_fixed =
            not lhs_enclosure.contains(0) or lhs_enclosure.infimum == lhs_enclosure.supremum;
        const bool rhs_fixed =
            not rhs_enclosure.contains(0) or rhs_enclosure.infimum == rhs_enclosure.supremum;

        // Unlike logical_and/logical_or, xor is never determined by one operand alone,
        // so if either is ambiguous we can output either true or false
        if (not lhs_fixed or not rhs_fixed) return {false, true};

        const bool res = not lhs_enclosure.contains(0) xor not rhs_enclosure.contains(0);
        return {res, res};
    }
};

struct maximum : mixins::BinaryOpMixin<maximum> {
    template <DType T>
    static constexpr T operator()(T lhs, T rhs) {
        if constexpr (std::same_as<T, bool>) {
            // NumPy treats the maximum of two bools as logical or, and the bitwise
            // operator is vectorizable
            return lhs | rhs;
        } else {
            return lhs > rhs ? lhs : rhs;
        }
    }
    using BinaryOpMixin::operator();

    static constexpr std::array<Monotonicity, 2> monotonicity{
        Monotonicity::Increasing,
        Monotonicity::Increasing
    };
};

struct minimum : mixins::BinaryOpMixin<minimum> {
    template <DType T>
    static constexpr T operator()(T lhs, T rhs) {
        if constexpr (std::same_as<T, bool>) {
            // NumPy treats the minimum of two bools as logical and, and the bitwise
            // operator is vectorizable
            return lhs & rhs;
        } else {
            return lhs < rhs ? lhs : rhs;
        }
    }
    using BinaryOpMixin::operator();

    static constexpr std::array<Monotonicity, 2> monotonicity{
        Monotonicity::Increasing,
        Monotonicity::Increasing
    };
};

struct multiply : mixins::BinaryOpMixin<multiply> {
    template <DType T>
    static constexpr T operator()(T lhs, T rhs) {
        if constexpr (std::same_as<T, bool>) {
            // NumPy treats the product of two bools as logical and, and the bitwise
            // operator is vectorizable
            return lhs & rhs;
        } else if constexpr (std::signed_integral<T>) {
            // Unlike NumPy, we want to do saturating multiplication

            using limits = std::numeric_limits<T>;

#if !defined(DWOPT__FORCE_FALLBACK) && defined(__has_builtin)
#if __has_builtin(__builtin_mul_overflow)  // needs its own line
            // We really want C++26 std::saturating_mul, but while we're on C++23
            // we use the __builtin_mul_overflow (GCC and Clang) if it's available.
            if (T out; not __builtin_mul_overflow(lhs, rhs, &out)) return out;

            // If we overflowed, our output depends on our sign
            if ((lhs < 0) ^ (rhs < 0)) {
                return limits::lowest();
            } else {
                return limits::max();
            }
#endif
#endif
            // fallback to simple std-only implementation

            if (lhs > 0) {
                if (rhs > 0) {
                    if (lhs > limits::max() / rhs) return limits::max();
                } else {
                    if (rhs < limits::lowest() / lhs) return limits::lowest();
                }
            } else {
                if (rhs > 0) {
                    if (lhs < limits::lowest() / rhs) return limits::lowest();
                } else if (lhs != 0 and rhs < limits::max() / lhs) {
                    return limits::max();
                }
            }

            return lhs * rhs;

        } else if constexpr (std::floating_point<T>) {
            // For floating
            assert(not std::isnan(lhs) and "lhs cannot be nan");
            assert(not std::isnan(rhs) and "rhs cannot be nan");

            // inf * 0 is Nan so we define inf * 0 := 0
            if ((lhs == T{0} and std::isinf(rhs)) or (std::isinf(lhs) and rhs == T{0})) {
                if (std::signbit(lhs) xor std::signbit(rhs)) {
                    return -T{0};
                } else {
                    return +T{0};
                }
            }

            return lhs * rhs;
        } else {
            static_assert(false, "unsupported dtype");
        }
    }

    template <DType T>
    constexpr static interval<T> operator()(
        const interval<T>& lhs_enclosure,
        const interval<T>& rhs_enclosure
    ) {
        assert(static_cast<bool>(lhs_enclosure) and "lhs's enclosure cannot be empty");
        assert(static_cast<bool>(rhs_enclosure) and "rhs's enclosure cannot be empty");

        // Start with an empty interval, then make sure it includes all four corners
        interval<T> out;
        out |= multiply{}(lhs_enclosure.infimum, rhs_enclosure.infimum);
        out |= multiply{}(lhs_enclosure.infimum, rhs_enclosure.supremum);
        out |= multiply{}(lhs_enclosure.supremum, rhs_enclosure.infimum);
        out |= multiply{}(lhs_enclosure.supremum, rhs_enclosure.supremum);
        return out;
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

/// Compute the remainder complementary to floor division.
/// This follows NumPy rather than C++ (std::remainder is the complement to round(x / y)).
struct remainder : mixins::BinaryOpMixin<remainder> {
    // dev note: As of Sept 2026, no version of Apple Clang supports `constexpr std::fmod()`,
    // even though that's part of the C++23 standard.
    // Luckily we can mark them as `constexpr` anyway and just let it fail if someone tries.
    template <DType T>
    constexpr static T operator()(T lhs, T rhs) {
        // Copy NumPy behavior and return 0 for `lhs % 0`
        // For floats, NumPy actually returns nan when rhs is 0.0, but we follow
        // their int convention and return 0.
        // We also make sure to follow their sign convention even for 0 rhs.
        if (rhs == 0) return rhs;

        T result;
        if constexpr (std::floating_point<T>) {
            assert(not std::isnan(lhs) and "lhs cannot be nan");
            assert(not std::isnan(rhs) and "rhs cannot be nan");

            // Unlike NumPy, we define inf % rhs := copysign(0.0, rhs) rather than NaN
            if (std::isinf(lhs)) return std::copysign(T{0}, rhs);

            result = std::fmod(lhs, rhs);
        } else {
            result = lhs % rhs;
        }

        // Follow NumPy/Python's sign conventions.
        if (result) {
            if (((lhs > 0) != (rhs > 0))) {
                result += rhs;
            }
        } else if constexpr (std::floating_point<T>) {
            result = std::copysign(T{0}, rhs);
        }

        return result;
    }

    template <DType T>
    constexpr static interval<T> operator()(const interval<T>&, const interval<T>& rhs_enclosure) {
        assert(static_cast<bool>(rhs_enclosure) and "rhs's enclosure cannot be empty");

        // We could consider the lhs when calculating the bounds, but for now let's
        // assume rhs always spans a full "period". IMO this is more intuitive and will
        // be the norm for most models we care about.

        if constexpr (std::same_as<T, bool>) {
            // always 0
            return interval<T>(false, false);
        } else {
            // Whatever our output is, it needs to include 0.
            interval<T> bounds = rhs_enclosure | interval<T>(0, 0);

            // If we're integral, we don't want to include the endpoints (nor for floating but
            // in that case it's infintesimal so we don't try to drop epsilon or whatever).
            if constexpr (std::integral<T>) {
                if (bounds.supremum > 0) bounds.supremum -= 1;
                if (bounds.infimum < 0) bounds.infimum += 1;
            }

            return bounds;
        }
    }
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

struct sin : mixins::UnaryOpMixin<sin> {
    // dev note: these are marked constexpr even though std::sin() isn't
    // actually constexpr until C++26. Luckily everything works fine with
    // this approach and it's a bit more future-proof.

    /// Calculate the sine of `x`.
    /// We want to disallow `nan`s so we define `sin(+/-inf) := +0.0`.
    template <DType T>
    static constexpr auto operator()(T x) {
        if constexpr (std::floating_point<T>) {
            assert(not std::isnan(x) and "x cannot be nan");
            if (std::isinf(x)) return T{0};
        }

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
    static interval<T> operator()(const interval<T>& x_enclosure) {
        assert(static_cast<bool>(x_enclosure) and "x's enclosure cannot be empty");
        if constexpr (std::same_as<T, bool>) {
            // square is just identity for boolean types
            return x_enclosure;
        } else {
            constexpr square op{};
            T inf_squared = op(x_enclosure.infimum);
            T sup_squared = op(x_enclosure.supremum);

            // Non-negative domain: square is increasing
            if (0 <= x_enclosure.infimum) return interval<T>(inf_squared, sup_squared);

            // Non-positive domain: square is decreasing
            if (x_enclosure.supremum <= 0) return interval<T>(sup_squared, inf_squared);

            // Otherwise the domain straddles 0: minimum is 0, maximum is the larger squared
            // endpoint.
            return interval<T>(0, inf_squared < sup_squared ? sup_squared : inf_squared);
        }
    }
};

struct subtract : mixins::BinaryOpMixin<subtract> {
    template <DType T>
    requires(not std::same_as<T, bool>)  // Follow NumPy and disallow bool inputs
    static constexpr T operator()(T lhs, T rhs) {
        using limits = std::numeric_limits<T>;

        if constexpr (std::signed_integral<T>) {
            // For integers we do saturating subtraction
            // In C++26 we could use std::saturating_sub() but for now we backport it

#if !defined(DWOPT__FORCE_FALLBACK) && defined(__has_builtin)
#if __has_builtin(__builtin_sub_overflow)  // needs its own line
            // Use a builtin available to Clang and GCC
            if (T out; not __builtin_sub_overflow(lhs, rhs, &out)) return out;
            if (lhs < 0) {
                return limits::lowest();
            } else {
                return limits::max();
            }
#endif
#endif
            // This fallback is overkill but I had already written it before deciding to use
            // __builtin_sub_overflow so might as well use it.
            // Implementation follows https://locklessinc.com/articles/sat_arithmetic/
            // I have tried to mirror the implementation from that link using our code style

            using UT = std::make_unsigned_t<T>;

            UT ux = static_cast<UT>(lhs);
            UT uy = static_cast<UT>(rhs);
            UT res = ux - uy;

            ux = (ux >> limits::digits) + limits::max();

            if (static_cast<T>((ux ^ uy) & (ux ^ res)) < 0) {
                res = ux;
            }

            return static_cast<T>(res);

        } else {
            // For floating we don't have any asymmetries to worry about
            // so we can just fall back to add so we can inherit its handling
            // of infinities.
            return add{}(lhs, -rhs);
        }
    }
    using BinaryOpMixin::operator();

    static constexpr std::array<Monotonicity, 2> monotonicity{
        Monotonicity::Increasing,
        Monotonicity::Decreasing
    };
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

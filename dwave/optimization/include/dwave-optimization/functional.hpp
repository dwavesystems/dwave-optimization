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

#include <array>
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

namespace mixins {

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

template <typename BinaryOp>
struct BinaryOpMixin {
    template <DType T>
    requires(
        BinaryOp::monotonicity[0] != Monotonicity::None and
        BinaryOp::monotonicity[1] != Monotonicity::None and
        requires { BinaryOp::operator()(T(), T()); }
    )
    static constexpr auto operator()(const interval<T>& lhs, const interval<T>& rhs) {
        using return_type = interval<decltype(BinaryOp::operator()(T(), T()))>;

        // If either lhs or rhs is an empty domain, then so is the output
        if (not static_cast<bool>(lhs)) return return_type();
        if (not static_cast<bool>(rhs)) return return_type();

        // TODO: handle invalid input domains (e.g., 0 in division) once we have
        // a binary op that hit that

        const return_type result(
            BinaryOp::operator()(
                BinaryOp::monotonicity[0] == Monotonicity::Increasing ? lhs.infimum : lhs.supremum,
                BinaryOp::monotonicity[1] == Monotonicity::Increasing ? rhs.infimum : rhs.supremum
            ),
            BinaryOp::operator()(
                BinaryOp::monotonicity[0] == Monotonicity::Increasing ? lhs.supremum : lhs.infimum,
                BinaryOp::monotonicity[1] == Monotonicity::Increasing ? rhs.supremum : rhs.infimum
            )
        );

        // If our intervals are non-empty and we get an empty output, then either our
        // op isn't actually monotonic or we have another bug.
        assert(static_cast<bool>(result));

        return result;
    }

    // BinaryOps are assumed not to be monotonic unless they tell us otherwise.
    static constexpr std::array<Monotonicity, 2> monotonicity{
        Monotonicity::None,
        Monotonicity::None
    };
};

}  // namespace mixins

struct absolute : mixins::UnaryOpMixin<absolute> {
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

// TODO: add docstring and note that:
// -we do saturating addition for int
// - -inf + inf := inf
// - inputting nan is undefined
// - boolean addition is logical or
struct add : mixins::BinaryOpMixin<add> {
    template <DType T>
    static constexpr T operator()(T lhs, T rhs) {
        using limits = std::numeric_limits<T>;

        if constexpr (std::same_as<T, bool>) {
            // bitwise operator is vectorizable
            return lhs | rhs;
        } else if constexpr (std::signed_integral<T>) {
            // For integers we do saturating addition.
            // In C++26 we could use std::saturating_add() but for now we backport it

#if !defined(DWOPT_FORCE_FALLBACK) && (defined(__GNUC__) || defined(__clang__))
            // Use a builtin available to Clang and GCC
            if (T res; not __builtin_add_overflow(lhs, rhs, &res)) return res;
            if (lhs < 0) {
                return limits::lowest();
            } else {
                return limits::max();
            }
#else
            // This fallback is overkill but I had already written it before deciding to use
            // __builtin_add_overflow so might as well use it.
            // Implementation follows https://locklessinc.com/articles/sat_arithmetic/
            // I have tried to mirror the implementation from that link using our code style

            using UT = std::make_unsigned_t<T>;

            UT ux = static_cast<UT>(lhs);
            UT uy = static_cast<UT>(rhs);
            UT res = ux + uy;

            ux = (ux >> limits::digits) + limits::max();

            if (static_cast<T>((ux ^ uy) | ~(uy ^ res)) >= 0) {
                res = ux;
            }

            return static_cast<T>(res);
#endif
        } else if constexpr (std::floating_point<T>) {
            // -inf + inf would result in NaN
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

/// Unlike NumPy we define x / 0 := 0
struct divide : mixins::BinaryOpMixin<divide> {
    template <DType T>
    requires(std::floating_point<T>)  // Follow NumPy and only define for floating types
    static constexpr T operator()(T lhs, T rhs) {
        // If we supported integer division, we'd need to support saturating division

        // We want to be safe (i.e. lhs / 0 := 0)
        if (not rhs) return std::signbit(lhs) == std::signbit(rhs) ? T(0) : -T(0);

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

    template <DType T>
    requires(std::floating_point<T>)  // Follow NumPy and only define for floating types
    static constexpr interval<T> operator()(const interval<T>& lhs, const interval<T>& rhs) {
        // If either lhs or rhs is an empty domain, then so is their division
        if (not static_cast<bool>(lhs)) return {};
        if (not static_cast<bool>(rhs)) return {};

        // If either lhs or rhs is pinned to zero then so is our function
        if (lhs.infimum == 0 and lhs.supremum == 0) return {0, 0};
        if (rhs.infimum == 0 and rhs.supremum == 0) return {0, 0};

        // If rhs straddles 0, then we can get arbitrarily close to +/- inf
        if (rhs.infimum < 0 and 0 < rhs.supremum) return interval<T>::all();

        // if rhs is non-negative or non-positive then the sign of lhs determines our range of
        // outputs
        if (rhs.infimum == 0 and 0 < rhs.supremum) {
            if (0 <= lhs.infimum) return interval<T>::nonnegative();
            if (lhs.supremum <= 0) return interval<T>::nonpositive();
            return interval<T>::all();
        }
        if (rhs.infimum < 0 and rhs.supremum == 0) {
            if (0 <= lhs.infimum) return interval<T>::nonpositive();
            if (lhs.supremum <= 0) return interval<T>::nonnegative();
            return interval<T>::all();
        }

        // Otherwise we just need to include our four corners
        interval<T> out;
        out |= interval<T>(divide{}(lhs.infimum, rhs.infimum));
        out |= interval<T>(divide{}(lhs.infimum, rhs.supremum));
        out |= interval<T>(divide{}(lhs.supremum, rhs.infimum));
        out |= interval<T>(divide{}(lhs.supremum, rhs.supremum));
        return out;
    }
};

struct equal : mixins::BinaryOpMixin<equal> {
    template <DType T>
    static constexpr bool operator()(T lhs, T rhs) {
        return lhs == rhs;
    }

    template <DType T>
    static constexpr interval<bool> operator()(const interval<T>& lhs, const interval<T>& rhs) {
        // If either lhs or rhs is an empty domain, then so is the output
        if (not static_cast<bool>(lhs)) return {};
        if (not static_cast<bool>(rhs)) return {};

        // If the domains don't overlap then no value in lhs can equal one in rhs
        if (not static_cast<bool>(lhs & rhs)) return {false, false};

        // The domains overlap, so if each holds a single value it's the same value
        if (lhs.infimum == lhs.supremum and rhs.infimum == rhs.supremum) return {true, true};

        // Otherwise we can output either true or false
        return {false, true};
    }
};

struct exp : mixins::UnaryOpMixin<exp> {
    static auto operator()(const DType auto& x) { return std::exp(x); }
    using UnaryOpMixin::operator();

    static constexpr Monotonicity monotonic = Monotonicity::Increasing;
};

struct expit : mixins::UnaryOpMixin<expit> {
    template <DType T>
    static auto operator()(const T& x) {
        return 1 / (1 + std::exp(-x));
    }
    using UnaryOpMixin::operator();

    static constexpr Monotonicity monotonic = Monotonicity::Increasing;
};

struct less_equal : mixins::BinaryOpMixin<less_equal> {
    template <DType T>
    static constexpr bool operator()(T lhs, T rhs) {
        return lhs <= rhs;
    }
    using BinaryOpMixin::operator();

    static constexpr std::array<Monotonicity, 2> monotonicity{
        Monotonicity::Decreasing,
        Monotonicity::Increasing
    };
};

struct log : mixins::UnaryOpMixin<log> {
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

struct logical : mixins::UnaryOpMixin<logical> {
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

struct logical_and : mixins::BinaryOpMixin<logical_and> {
    template <DType T>
    static constexpr bool operator()(T lhs, T rhs) {
        // bitwise and is vectorizable
        return static_cast<bool>(lhs) & static_cast<bool>(rhs);
    }

    template <DType T>
    static constexpr interval<bool> operator()(const interval<T>& lhs, const interval<T>& rhs) {
        // If either lhs or rhs is an empty domain, then so is the logical_and
        if (not static_cast<bool>(lhs)) return {};
        if (not static_cast<bool>(rhs)) return {};

        // if both lhs and rhs only contain truthy values, then our output will
        // always be true
        if (not lhs.contains(0) and not rhs.contains(0)) return {true, true};

        // if lhs is always falsy, then our output will always be false
        if (lhs.contains(0) and lhs.infimum == lhs.supremum) return {false, false};

        // likewise for rhs
        if (rhs.contains(0) and rhs.infimum == rhs.supremum) return {false, false};

        // Otherwise we can output either true or false
        return {false, true};
    }
};

struct logical_not : mixins::UnaryOpMixin<logical_not> {
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

struct logical_or : mixins::BinaryOpMixin<logical_or> {
    template <DType T>
    static constexpr bool operator()(T lhs, T rhs) {
        // bitwise or is vectorizable
        return static_cast<bool>(lhs) | static_cast<bool>(rhs);
    }

    template <DType T>
    static constexpr interval<bool> operator()(const interval<T>& lhs, const interval<T>& rhs) {
        // If either lhs or rhs is an empty domain, then so is the logical_or
        if (not static_cast<bool>(lhs)) return {};
        if (not static_cast<bool>(rhs)) return {};

        // if lhs only contains truthy values, then our output will always be true
        if (not lhs.contains(0)) return {true, true};

        // likewise for rhs
        if (not rhs.contains(0)) return {true, true};

        // Both lhs and rhs contain 0, so either being a single value makes it
        // always falsy. If both are, then our output will always be false
        if (lhs.infimum == lhs.supremum and rhs.infimum == rhs.supremum) return {false, false};

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
    static constexpr interval<bool> operator()(const interval<T>& lhs, const interval<T>& rhs) {
        // If either lhs or rhs is an empty domain, then so is the logical_xor
        if (not static_cast<bool>(lhs)) return {};
        if (not static_cast<bool>(rhs)) return {};

        // A domain fixes its truthiness if it excludes 0 (always truthy) or if it holds
        // a single value (truthy or falsy, but not both)
        const bool lhs_fixed = not lhs.contains(0) or lhs.infimum == lhs.supremum;
        const bool rhs_fixed = not rhs.contains(0) or rhs.infimum == rhs.supremum;

        // Unlike logical_and/logical_or, xor is never determined by one operand alone,
        // so if either is ambiguous we can output either true or false
        if (not lhs_fixed or not rhs_fixed) return {false, true};

        const bool res = not lhs.contains(0) xor not rhs.contains(0);
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

#if !defined(DWOPT_FORCE_FALLBACK) && (defined(__GNUC__) || defined(__clang__))
            // Use a builtin available to Clang and GCC
            if (T res; not __builtin_mul_overflow(lhs, rhs, &res)) return res;

            // If we overflowed, our output depends on our sign
            if ((lhs < 0) ^ (rhs < 0)) {
                return limits::lowest();
            } else {
                return limits::max();
            }
#else
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
                } else if (lhs != 0 and rhs < limits::max() / lhs)
                    return limits::max();
            }

            return lhs * rhs;
#endif

        } else if constexpr (std::floating_point<T>) {
            // For floating

            // inf * 0 is Nan so we define inf * 0 := 0
            if ((lhs == T(0) and std::isinf(rhs)) or (std::isinf(lhs) and rhs == T(0))) {
                if (std::signbit(lhs) ^ std::signbit(rhs)) {
                    return -T(0);
                } else {
                    return +T(0);
                }
            }

            return lhs * rhs;
        } else {
            static_assert(false, "unsupported dtype");
        }
    }

    template <DType T>
    constexpr static interval<T> operator()(const interval<T>& lhs, const interval<T>& rhs) {
        // If either lhs or rhs is an empty domain, then so is their product
        if (not static_cast<bool>(lhs)) return {};
        if (not static_cast<bool>(rhs)) return {};

        // Start with an empty interval, then make sure it includes all four corners
        interval<T> out;
        out |= interval<T>(multiply{}(lhs.infimum, rhs.infimum));
        out |= interval<T>(multiply{}(lhs.infimum, rhs.supremum));
        out |= interval<T>(multiply{}(lhs.supremum, rhs.infimum));
        out |= interval<T>(multiply{}(lhs.supremum, rhs.supremum));
        return out;
    }
};

struct negative : mixins::UnaryOpMixin<negative> {
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
            // Unlike NumPy, we define inf % rhs := copysign(0.0, rhs) rather than NaN
            if (std::isinf(lhs)) return std::copysign(T(0), rhs);

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
            result = std::copysign(T(0), rhs);
        }

        return result;
    }

    template <DType T>
    constexpr static interval<T> operator()(const interval<T>& lhs, const interval<T>& rhs) {
        // If either lhs or rhs is an empty domain, then so is their modulus
        if (not static_cast<bool>(lhs)) return {};
        if (not static_cast<bool>(rhs)) return {};

        // We could consider the lhs when calculating the bounds, but for now let's
        // assume rhs always spans a full "period". IMO this is more intuitive and will
        // be the norm for most models we care about.

        if constexpr (std::same_as<T, bool>) {
            // always 0
            return interval<T>(false, false);
        } else {
            // Whatever our output is, is needs to include 0.
            interval<T> bounds = rhs | interval<T>(0, 0);

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

struct sin : mixins::UnaryOpMixin<sin> {
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

struct square : mixins::UnaryOpMixin<square> {
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

struct square_root : mixins::UnaryOpMixin<square_root> {
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

struct subtract : mixins::BinaryOpMixin<subtract> {
    template <DType T>
    requires(not std::same_as<T, bool>)  // Follow NumPy and disallow bool inputs
    static constexpr T operator()(T lhs, T rhs) {
        using limits = std::numeric_limits<T>;

        if constexpr (std::signed_integral<T>) {
            // For integers we do saturating subtraction
            // In C++26 we could use std::saturating_sub() but for now we backport it

#if !defined(DWOPT_FORCE_FALLBACK) && (defined(__GNUC__) || defined(__clang__))
            // Use a builtin available to Clang and GCC
            if (T res; not __builtin_sub_overflow(lhs, rhs, &res)) return res;
            if (lhs < 0) {
                return limits::lowest();
            } else {
                return limits::max();
            }
#else
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
#endif
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
    static auto operator()(const DType auto& num) { return std::tanh(num); }
    using UnaryOpMixin::operator();

    static constexpr Monotonicity monotonic = Monotonicity::Increasing;
};

}  // namespace dwave::optimization::functional

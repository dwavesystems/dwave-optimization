// Copyright 2026 D-Wave
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
#include <compare>
#include <concepts>
#include <iosfwd>
#include <limits>

#include "dwave-optimization/typing.hpp"

namespace dwave::optimization {

/// An interval encodes a range of possible values.
///
/// Note that this class does not (yet) implement outward rounding.
template <DType T>
struct interval {
    /// Construct an empty interval.
    constexpr interval() = default;

    /// Construct an interval of values between inf and sup (inclusive).
    /// When ``sup < inf`` the interval is treated as empty.
    constexpr interval(T inf, T sup) noexcept : infimum(inf), supremum(sup) {}

    /// Copy constructor.
    interval(const interval&) = default;

    /// Create an one interval from another, allowing for safe type promotions
    template <can_cast<T> U>
    constexpr interval(const interval<U>& other) noexcept :
        interval(other.infimum, other.supremum) {}

    /// Move constructor.
    interval(interval&&) = default;

    /// Copy assignment operator.
    interval& operator=(const interval&) = default;

    /// Move assignment operator.
    interval& operator=(interval&&) = default;

    /// Destructor.
    ~interval() = default;

    /// An interval evalutes to `true` if it is not empty.
    explicit constexpr operator bool() const noexcept { return infimum <= supremum; }

    /// Two intervals are treated as equal if they are the same type and have the same endpoints
    /// or if they are both null.
    constexpr bool operator==(const interval& rhs) const {
        if (not static_cast<bool>(*this) and not static_cast<bool>(rhs)) return true;  // both null
        return infimum == rhs.infimum and supremum == rhs.supremum;
    }
    // dev note: we could support other type combinations in the future

    /// Comparison operators <, <=, >=, > are used for strict subset, subset, superset, and strict
    /// superset respectively.
    constexpr std::partial_ordering operator<=>(const interval& rhs) const {
        // If we're equal then we're equivalent
        if (*this == rhs) return std::partial_ordering::equivalent;

        // If lhs != rhs then at most one can be empty
        if (not static_cast<bool>(*this)) return std::partial_ordering::less;   // empty < not-empty
        if (not static_cast<bool>(rhs)) return std::partial_ordering::greater;  // not-empty > empty

        // Ok, neither are empty

        // If lhs <= rhs and lhs != rhs then lhs < rhs
        if (rhs.infimum <= infimum and supremum <= rhs.supremum) {
            return std::partial_ordering::less;
        }

        // If lhs >= rhs and lhs != rhs then lhs > rhs
        if (infimum <= rhs.infimum and rhs.supremum <= supremum) {
            return std::partial_ordering::greater;
        }

        // Otherwise we're not comparable
        return std::partial_ordering::unordered;
    }
    // dev note: we could support other type combinations in the future

    /// Negate and swap the values in the interval.
    /// For boolean intervals, negation is treated as logical not.
    constexpr interval operator-() const {
        // For bool, we overload this to be negation
        if constexpr (std::same_as<T, bool>) return interval(not supremum, not infimum);

        // -INT_MIN is undefined. Under the assumption that if the user is using
        // INT_MIN/INT_MAX they probably are trying to say "unbounded" we do a
        // weird thing and just define -INT_MIN := INT_MAX and -INT_MAX := INT_MIN
        // even though that's wrong and leads to some slightly weird outcomes
        if constexpr (std::integral<T>) {
            using limits = std::numeric_limits<T>;
            if (infimum == limits::lowest() and supremum == limits::max()) {
                return *this;
            } else if (infimum == limits::lowest()) {
                return interval(-supremum, limits::max());
            } else if (supremum == limits::max()) {
                return interval(limits::lowest(), -infimum);
            }
        }

        return interval(-supremum, -infimum);
    }

    /// Intersection with ``rhs``.
    constexpr interval& operator&=(const interval& rhs) {
        // If lhs is an empty interval, then the intersection is just lhs
        if (not static_cast<bool>(*this)) return *this;

        // If rhs is an empty interval, then the intersection is just rhs
        if (not static_cast<bool>(rhs)) return *this = rhs;

        if (infimum < rhs.infimum) infimum = rhs.infimum;
        if (rhs.supremum < supremum) supremum = rhs.supremum;

        return *this;
    }

    /// Union with ``rhs``.
    constexpr interval& operator|=(const interval& rhs) {
        // If rhs is an empty interval, then taking the union with it does nothing
        if (not static_cast<bool>(rhs)) return *this;

        // If lhs is an empty interval, then the union is just rhs
        if (not static_cast<bool>(*this)) return *this = rhs;

        if (rhs.infimum < infimum) infimum = rhs.infimum;
        if (supremum < rhs.supremum) supremum = rhs.supremum;

        return *this;
    }

    /// Interection of two intervals.
    friend constexpr interval operator&(interval lhs, const interval& rhs) {
        lhs &= rhs;
        return lhs;
    }

    /// Union of two intervals
    friend constexpr interval operator|(interval lhs, const interval& rhs) {
        lhs |= rhs;
        return lhs;
    }

    /// The maximum expressible interval
    static consteval interval all() {
        using limits = std::numeric_limits<T>;
        if constexpr (limits::has_infinity) {
            return interval(-limits::infinity(), limits::infinity());
        } else {
            return interval(limits::lowest(), limits::max());
        }
    }

    /// Test whether `x` is a value in the interval.
    constexpr bool contains(const T& x) const { return infimum <= x and x <= supremum; }
    // dev note: we could support other type combinations in the future

    /// All expressible non-negative values.
    static consteval interval nonnegative() {
        using limits = std::numeric_limits<T>;
        if constexpr (limits::has_infinity) {
            return interval(0, limits::infinity());
        } else {
            return interval(0, limits::max());
        }
    }

    T infimum = 1;
    T supremum = 0;
};

// Intervals are printable
template <DType T>
std::ostream& operator<<(std::ostream& os, const interval<T>& in);

}  // namespace dwave::optimization

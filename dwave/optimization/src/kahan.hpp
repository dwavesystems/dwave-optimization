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

#include "dwave-optimization/typing.hpp"

namespace dwave::optimization {

/// An accumulator that uses Kahan summation to reduce the rounding error
/// that builds up over many additions and subtractions.
///
/// Only values that can be safely cast to `T` are accepted (see `can_cast`),
/// because precision lost before an addition cannot be compensated for.
template <std::floating_point T>
class kahan {
 public:
    /// Construct an accumulator with a sum of zero.
    constexpr kahan() noexcept : value_(0), compensator_(0) {}

    /// Construct an accumulator with the given starting sum and no compensation.
    constexpr kahan(const can_cast<T> auto value) noexcept : value_(value), compensator_(0) {}

    /// Add `other` to the sum, compensating for rounding error.
    constexpr kahan& operator+=(const can_cast<T> auto other) noexcept {
        T y = static_cast<T>(other) - compensator_;
        T t = value_ + y;
        compensator_ = (t - value_) - y;
        value_ = t;
        return *this;
    }

    /// Subtract `other` from the sum, compensating for rounding error.
    constexpr kahan& operator-=(const can_cast<T> auto other) noexcept {
        // Cast before negating, because negating a small integer promotes it to `int`.
        *this += -static_cast<T>(other);
        return *this;
    }

    /// Accumulators can be cast to their floating-point type, but only explicitly.
    constexpr explicit operator T() const noexcept { return value_; }

    /// We do not consider the compensator for checking equality.
    constexpr bool operator==(const kahan& other) const noexcept {
        return (this->value_ == other.value_);
    }

    /// Accumulators can be compared to values that can be safely cast to `T`.
    /// We do not consider the compensator for checking equality.
    constexpr friend bool operator==(const kahan& lhs, const can_cast<T> auto rhs) noexcept {
        return lhs.value_ == rhs;
    }

    /// The current sum.
    constexpr const T& value() const noexcept { return value_; }

    /// The current compensation, i.e. the (negated) rounding error not yet
    /// folded into the sum.
    constexpr const T& compensator() const noexcept { return compensator_; }

 private:
    T value_;
    T compensator_;
};

}  // namespace dwave::optimization

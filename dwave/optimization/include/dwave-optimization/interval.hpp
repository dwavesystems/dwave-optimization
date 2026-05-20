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

#include <algorithm>
#include <cassert>
#include <cmath>

#include "dwave-optimization/common.hpp"  // for ssize_t
#include "dwave-optimization/typing.hpp"

namespace dwave::optimization {

// dev note: true interval arithmetic requires "outward" rounding. For right
// now I am ignoring this in the interest of getting an API solidified, but
// we should support it ASAP. I have marked the places we need to make changes
// with a "numeric issues" comment for easy future reference.

template <DType T>
class interval {
 public:
    interval() noexcept : infimum(1), supremum(0) {};

    // todo: note
    interval(T value) noexcept : interval(value, value) {}

    interval(T infimum, T supremum) noexcept : infimum(infimum), supremum(supremum) {}

    static interval unbounded() {
        using limits = std::numeric_limits<T>;
        if constexpr (limits::has_infinity) {
            return interval(-limits::infinity(), limits::infinity());
        } else {
            return interval(limits::lowest(), limits::max());
        }
    }

    /// An interval evalutes to `true` if it is not empty.
    explicit operator bool() const noexcept { return infimum <= supremum; }

    bool operator==(const interval& rhs) const {
        return infimum == rhs.infimum and supremum == rhs.supremum;
    }

    interval& operator|=(DType auto scalar) { return *this |= interval<decltype(scalar)>(scalar); }
    template <DType U>
    interval& operator|=(const interval<U>& rhs) {
        // dev note: numeric issues
        if constexpr (std::floating_point<U> and std::integral<T>) {
            infimum = std::min<T>(infimum, std::floor(rhs.infimum));
            supremum = std::max<T>(supremum, std::ceil(rhs.supremum));
        } else {
            infimum = std::min<T>(infimum, rhs.infimum);
            supremum = std::max<T>(supremum, rhs.supremum);
        }
        return *this;
    }

    bool contains(auto&& value) const {
        // dev note: numeric issues
        return infimum <= value and value <= supremum;
    }

    /// The size of the interval is the max(supremum - infimum, 0)
    double size() const noexcept requires(std::floating_point<T>) {
        if (supremum >= infimum) return static_cast<double>(supremum) - infimum;
        return 0;
    }
    std::ptrdiff_t size() const noexcept requires(std::integral<T>) {
        if (supremum >= infimum) return static_cast<std::ptrdiff_t>(supremum) - infimum;
        return 0;
    }

    T infimum;
    T supremum;
};

}  // namespace dwave::optimization

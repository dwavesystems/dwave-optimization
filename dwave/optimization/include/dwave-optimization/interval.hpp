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
#include <iosfwd>
#include <type_traits>

#include "dwave-optimization/common.hpp"  // for ssize_t
#include "dwave-optimization/typing.hpp"

namespace dwave::optimization {

// dev note: true interval arithmetic requires "outward" rounding. For right
// now I am ignoring this and focusing on getting the API solidified, but
// we should support it ASAP. I have marked the places we need to make changes
// with a "numeric issues" comment for easy future reference.

template <DType T>
requires(not std::is_const_v<T>) class interval {
 public:
    // by default we construct an empty interval
    interval() = default;

    // rule of five stuff all works the way you'd expect
    interval(const interval&) = default;
    interval(interval&&) = default;
    ~interval() = default;
    interval& operator=(const interval&) = default;
    interval& operator=(interval&&) = default;

    // todo: note the difference between interval<int>(.5, .5) and interval<int>(interval<float>(.5,
    // .5))
    explicit interval(T value) noexcept : interval(value, value) {}
    interval(T infimum, T supremum) noexcept : infimum_(infimum), supremum_(supremum) {
        assert(not std::isnan(infimum_) and "infimum must not be nan");
        assert(not std::isnan(supremum_) and "supremum must not be nan");
    }

    template <DType U>
    explicit(std::integral<T> != std::integral<U>) interval(const interval<U>& rhs) : interval() {
        *this = rhs;
    }
    template <DType U>
    explicit(std::integral<T> != std::integral<U>) interval(interval<U>&& rhs) : interval() {
        *this = rhs;
    }

    template <DType U>
    interval& operator=(const interval<U>& rhs) {
        // dev note: numeric issues
        if constexpr (std::floating_point<U> and std::integral<T>) {
            infimum_ = std::floor(rhs.infimum());
            supremum_ = std::ceil(rhs.supremum());
        } else {
            infimum_ = rhs.infimum();
            supremum_ = rhs.supremum();
        }
        return *this;
    }

    /// An interval evalutes to `true` if it is not empty.
    explicit operator bool() const noexcept { return infimum_ <= supremum_; }

    bool operator==(const interval& rhs) const {
        return infimum_ == rhs.infimum_ and supremum_ == rhs.supremum_;
    }

    interval& operator|=(DType auto scalar) { return *this |= interval<decltype(scalar)>(scalar); }

    template <DType U>
    interval& operator|=(const interval<U>& rhs) {
        // if rhs is an empty interval, do nothing
        if (not static_cast<bool>(rhs)) return *this;

        // if lhs is an empty interval, then set it to rhs (which we know is not empty)
        if (not static_cast<bool>(*this)) return *this = rhs;

        // otherwise, adjust our interval to also contain the other.
        // dev note: numeric issues
        if constexpr (std::integral<T> and std::floating_point<U>) {
            infimum_ = std::min<T>(infimum_, std::floor(rhs.infimum()));
            supremum_ = std::max<T>(supremum_, std::ceil(rhs.supremum()));
        } else {
            infimum_ = std::min<T>(infimum_, rhs.infimum());
            supremum_ = std::max<T>(supremum_, rhs.supremum());
        }
        return *this;
    }

    template <DType U>
    friend auto operator|(interval lhs, const interval<U>& rhs) {
        interval<decltype(T() + U())> out(std::move(lhs));
        out |= rhs;
        return out;
    }

    template <DType U>
    interval& operator+=(const interval<U>& rhs) {
        // if lhs is an empty interval than the result is also empty
        if (not static_cast<bool>(*this)) return *this;

        // if rhs is an empty interval, then set lhs to rhs (making lhs empty)
        if (not static_cast<bool>(rhs)) return *this = rhs;

        // neither are empty! So add everything
        // dev note: numeric issues
        if constexpr (std::integral<T> and std::floating_point<U>) {
            infimum_ += std::floor(rhs.infimum());
            supremum_ += std::ceil(rhs.supremum());
        } else {
            infimum_ += rhs.infimum();
            supremum_ += rhs.supremum();
        }
        return *this;
    }

    template <DType U>
    friend auto operator+(interval lhs, const interval<U>& rhs) {
        interval<decltype(T() + U())> out(std::move(lhs));
        out += rhs;
        return out;
    }

    bool contains(DType auto&& value) const {
        // dev note: numeric issues
        return infimum_ <= value and value <= supremum_;
    }

    // Implement the get() to support the structured binding. It's not OK to
    // overload std functions so this needs to be found via ADL.
    // Unfortunately that prevents interval from satisfying the tuple-like
    // concept. Why can't we have nice things C++?? See note below for
    // more details.
    template <std::size_t I>
    friend const T& get(const interval& in) noexcept requires(I <= 1) {
        if constexpr (I == 0) return in.infimum_;
        if constexpr (I == 1) return in.supremum_;
    }
    template <std::size_t I>
    friend T&& get(interval&& in) noexcept requires(I <= 1) {
        if constexpr (I == 0) return std::move(in.infimum_);
        if constexpr (I == 1) return std::move(in.supremum_);
    }
    template <std::size_t I>
    friend const T&& get(const interval&& in) noexcept requires(I <= 1) {
        if constexpr (I == 0) return std::move(in.infimum_);
        if constexpr (I == 1) return std::move(in.supremum_);
    }

    const T& infimum() const noexcept { return infimum_; }

    /// The size of the interval is the max(supremum_ - infimum_, 0)
    double size() const noexcept requires(std::floating_point<T>) {
        if (supremum_ >= infimum_) return static_cast<double>(supremum_) - infimum_;
        return 0;
    }
    std::ptrdiff_t size() const noexcept requires(std::integral<T>) {
        if (supremum_ >= infimum_) return static_cast<std::ptrdiff_t>(supremum_) - infimum_;
        return 0;
    }

    const T& supremum() const noexcept { return supremum_; }

    static interval unbounded() {
        using limits = std::numeric_limits<T>;
        if constexpr (limits::has_infinity) {
            return interval(-limits::infinity(), limits::infinity());
        } else {
            return interval(limits::lowest(), limits::max());
        }
    }

 private:
    T infimum_ = 1;
    T supremum_ = 0;
};

std::ostream& operator<<(std::ostream& os, const interval<std::int64_t>& rhs);
std::ostream& operator<<(std::ostream& os, const interval<double>& rhs);

}  // namespace dwave::optimization

// Do a lot of convoluted stuff to support structured binding.
// Unfortunately this doesn't support the full tuple-like concept that would let us work
// out of the box with a lot of nice methods.
// See https://stackoverflow.com/questions/79660771/stdapply-to-a-custom-tuple-like-type
// See https://lists.isocpp.org/std-discussion/2021/09/1438.php
namespace std {

template <dwave::optimization::DType T>
struct tuple_size<dwave::optimization::interval<T>> : integral_constant<size_t, 2> {};

template <dwave::optimization::DType T>
struct tuple_element<0, dwave::optimization::interval<T>> {
    using type = T;
};

template <dwave::optimization::DType T>
struct tuple_element<1, dwave::optimization::interval<T>> {
    using type = T;
};

}  // namespace std

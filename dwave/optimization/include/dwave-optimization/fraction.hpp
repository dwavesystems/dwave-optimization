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
#include <concepts>
#include <iosfwd>
#include <numeric>
#include <stdexcept>

#include "dwave-optimization/common.hpp"  // for ssize_t

namespace dwave::optimization {

// A simple fraction class
class fraction {
 public:
    // fractions can be constructed from a single integer or a pair of integers
    constexpr fraction() noexcept : fraction(0) {}
    constexpr fraction(const std::integral auto n) noexcept : numerator_(n), denominator_(1) {}
    constexpr fraction(const std::integral auto numerator, const std::integral auto denominator)
            : numerator_(numerator), denominator_(denominator) {
        if (!denominator) throw std::invalid_argument("cannot divide by 0");
        reduce();
    }

    // fractions can be cast to doubles, but only explicitly
    constexpr explicit operator double() noexcept {
        return static_cast<double>(numerator_) / denominator_;
    }
    constexpr explicit operator ssize_t() noexcept { return numerator_ / denominator_; }

    // fractions can be compared to other fractions and to integers
    constexpr bool operator==(const fraction& other) const noexcept {
        return numerator_ == other.numerator_ && denominator_ == other.denominator_;
    }
    constexpr bool operator==(const std::integral auto n) const noexcept {
        return numerator_ == n && denominator_ == 1;
    }

    // fractions can be multiplied and divided in place.
    constexpr fraction& operator*=(const std::integral auto n) noexcept {
        numerator_ *= n;
        reduce();
        return *this;
    }
    constexpr fraction& operator*=(const fraction& other) noexcept {
        numerator_ *= other.numerator_;
        denominator_ *= other.denominator_;
        reduce();
        return *this;
    }
    constexpr friend fraction operator*(fraction lhs, const std::integral auto rhs) noexcept {
        lhs *= rhs;
        return lhs;
    }
    constexpr friend fraction operator*(const std::integral auto lhs, fraction rhs) noexcept {
        rhs *= lhs;
        return rhs;
    }
    constexpr friend fraction operator*(const fraction lhs, fraction rhs) noexcept {
        rhs *= lhs;
        return rhs;
    }
    constexpr fraction& operator/=(const std::integral auto n) {
        if (!n) throw std::invalid_argument("cannot divide by 0");
        denominator_ *= n;
        reduce();
        return *this;
    }
    constexpr friend fraction operator/(fraction lhs, const std::integral auto rhs) {
        lhs /= rhs;
        return lhs;
    }

    // fractions can be added
    constexpr friend fraction operator+(fraction lhs, const fraction& rhs) {
        lhs += rhs;
        return lhs;
    }
    constexpr friend fraction operator+(const fraction& lhs, const std::integral auto& rhs) {
        return lhs + fraction(rhs);
    }
    constexpr friend fraction operator+(const std::integral auto& lhs, const fraction& rhs) {
        return fraction(lhs) + rhs;
    }

    // fractions can be added inplace
    constexpr fraction& operator+=(const std::integral auto n) noexcept {
        (*this) += fraction(n);
        return *this;
    }

    constexpr fraction& operator+=(const fraction& other) noexcept {
        this->numerator_ =
                this->numerator_ * other.denominator_ + other.numerator_ * this->denominator_;
        this->denominator_ *= other.denominator_;
        reduce();
        return *this;
    }

    // fractions can be subtracted inplace
    constexpr fraction& operator-=(const std::integral auto n) noexcept {
        this->operator-=(fraction(n));
        return *this;
    }

    constexpr fraction& operator-=(const fraction& other) noexcept {
        this->operator+=(fraction(-other.numerator_, other.denominator_));
        return *this;
    }

    operator bool() const { return numerator_; }

    /// Fractions can be printed
    friend std::ostream& operator<<(std::ostream& os, const fraction& rhs);

    // dev note: obviously there are many more operators that we could add,
    // but they should be added as needed rather than eagerly.

    constexpr const ssize_t& numerator() const noexcept { return numerator_; }
    constexpr const ssize_t& denominator() const noexcept { return denominator_; }

    constexpr ssize_t ceil() const {
        assert(denominator_ > 0);
        if (numerator_ >= 0) {
            return (numerator_ / denominator_) + ((numerator_ % denominator_) > 0);
        } else {
            return numerator_ / denominator_;
        }
    }

 private:
    constexpr void reduce() noexcept {
        if (const ssize_t div = std::gcd(numerator_, denominator_); div != 1) {
            numerator_ /= div;
            denominator_ /= div;
        }
        if (denominator_ < 0) {
            numerator_ *= -1;
            denominator_ *= -1;
        }
    }

    ssize_t numerator_;
    ssize_t denominator_;
};

}  // namespace dwave::optimization

// Copyright 2023 D-Wave Systems Inc.
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

#include <iostream>
#include <numeric>
#include <vector>

namespace dwave::optimization {

// ssize_t is a posix definition. So let's add it for Windows.
// Match Python's Py_ssize_t definition which is itself defined to match ssize_t if it's present.
// https://github.com/python/cpython/blob/17cba55786a1b1e6b715b1a88ae1f9088f5d5999/PC/pyconfig.h.in#L200-L208
#if defined(_MSC_VER)
typedef __int64 ssize_t;
#endif

static_assert(sizeof(ssize_t) >= 1);  // check ssize_t exists

// backport unreachable from c++23
#if defined(_MSC_VER)
[[noreturn]] inline void unreachable() { __assume(false); }
#elif defined(__GNUC__) || defined(__GNUG__) || defined(__clang__)
[[noreturn]] inline void unreachable() { __builtin_unreachable(); }
#else
[[noreturn]] inline void unreachable() {}
#endif

class double_kahan {
 public:
    constexpr double_kahan() noexcept : value_(0), compensator_(0) {}
    constexpr double_kahan(double value) noexcept : value_(value), compensator_(0) {}

    constexpr double_kahan& operator+=(double other) noexcept {
        double y = other - compensator_;
        double t = value_ + y;
        compensator_ = (t - value_) - y;
        value_ = t;
        return *this;
    }

    constexpr double_kahan& operator-=(double other) noexcept {
        (*this += -other);
        return *this;
    }

    constexpr explicit operator double() const noexcept { return value_; }

    // We do not consider the compensator for checking equality.
    constexpr bool operator==(const double_kahan& other) const noexcept {
        return (this->value_ == other.value_);
    }
    constexpr friend bool operator==(const double_kahan& lhs, double rhs) noexcept {
        return lhs.value_ == rhs;
    }

    constexpr const double& value() const noexcept { return value_; }
    constexpr const double& compensator() const noexcept { return compensator_; }

 private:
    double value_;
    double compensator_;
};

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

    /// Fractions can be printed
    friend std::ostream& operator<<(std::ostream& os, const fraction& rhs) {
        os << "fraction(" << rhs.numerator();
        if (rhs.denominator() != 1) {
            os << ", " << rhs.denominator();
        }
        return os << ")";
    }

    // dev note: obviously there are many more operators that we could add,
    // but they should be added as needed rather than eagerly.

    constexpr const ssize_t& numerator() const noexcept { return numerator_; }
    constexpr const ssize_t& denominator() const noexcept { return denominator_; }

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

// type_list is partially based on
// https://github.com/lipk/cpp-typelist/blob/02eff2292fdf1ca73e8c09866c720d4b39473ed1
// under an MIT license.
//
//     MIT License
//
//     Copyright (c) 2018
//
//     Permission is hereby granted, free of charge, to any person obtaining a copy
//     of this software and associated documentation files (the "Software"), to deal
//     in the Software without restriction, including without limitation the rights
//     to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//     copies of the Software, and to permit persons to whom the Software is
//     furnished to do so, subject to the following conditions:
//
//     The above copyright notice and this permission notice shall be included in all
//     copies or substantial portions of the Software.
//
//     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//     OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//     SOFTWARE.

/// A ``type_list`` encodes a collection of types.
template <typename Type, typename... Types>
struct type_list {
    /// A new ``type_list`` created by adding a pointer to every type in the ``type_list``.
    using add_pointer = type_list<std::add_pointer_t<Type>, std::add_pointer_t<Types>...>;

    /// Return true of ``T`` is included in the ``type_list``.
    template <typename T>
    static constexpr bool contains() {
        if constexpr (std::same_as<Type, T>) return true;
        if constexpr (sizeof...(Types)) return type_list<Types...>::template contains<T>();
        return false;
    }

    /// Return the number of times that ``T`` appears in the ``type_list``.
    template <typename T>
    static constexpr std::size_t count() {
        constexpr std::size_t count = std::same_as<Type, T>;
        if constexpr (sizeof...(Types)) {
            return count + type_list<Types...>::template count<T>();
        }
        return count;
    }

    /// Return whether every type in the ``type_list`` is present in ``OtherTypeList``.
    template <typename OtherTypeList>   // this could be better specified
    static constexpr bool issubset() {  // match Python name
        bool subset = OtherTypeList::template contains<Type>();
        if constexpr (sizeof...(Types)) {
            return subset && type_list<Types...>::template issubset<OtherTypeList>();
        }
        return subset;
    }

    /// Return whether every type in the ``OtherTypeList`` is present in ``type_list``.
    template <typename OtherTypeList>     // this could be better specified
    static constexpr bool issuperset() {  // match Python name
        return OtherTypeList::template issubset<type_list<Type, Types...>>();
    }

    /// A new ``type_list`` created by removing a pointer from every type in the ``type_list``.
    using remove_pointer = type_list<std::remove_pointer_t<Type>, std::remove_pointer_t<Types>...>;

    /// Return the number of types.
    static constexpr std::size_t size() { return 1 + sizeof...(Types); }

    /// Convert the type_list to another class such as ``std::variant`` or
    /// ``std::tuple``.
    template <template <typename...> class U>
    using to = U<Type, Types...>;
};

// Forward declaration
struct Update;

void deduplicate_diff(std::vector<Update>& diff);

// Return whether the given double encodes an integer.
bool is_integer(const double& value);

}  // namespace dwave::optimization

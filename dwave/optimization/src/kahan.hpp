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

namespace dwave::optimization {

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

}  // namespace dwave::optimization

// Copyright 2026 D-Wave Systems Inc.
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

#include "dwave-optimization/cp/core/interval_array.hpp"

#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace dwave::optimization::cp {

template <>
IntervalArray<double>::IntervalArray(StateManager* sm, ssize_t min_size, ssize_t max_size) {
    if (min_size > max_size) {
        throw std::invalid_argument("Min size larger than max size");
    }

    assert(min_size <= max_size);
    min_size_ = sm->make_state_int(min_size);
    max_size_ = sm->make_state_int(max_size);

    // Set the bounds for each index
    min_.resize(max_size);
    max_.resize(max_size);
    for (ssize_t i = 0; i < max_size; ++i) {
        min_[i] = sm->make_state_real(-std::numeric_limits<double>::max() / 2);
        max_[i] = sm->make_state_real(std::numeric_limits<double>::max() / 2);
    }
}

template <>
IntervalArray<ssize_t>::IntervalArray(StateManager* sm, ssize_t min_size, ssize_t max_size) {
    if (min_size > max_size) {
        throw std::invalid_argument("Min size larger than max size");
    }

    assert(min_size <= max_size);
    min_size_ = sm->make_state_int(min_size);
    max_size_ = sm->make_state_int(max_size);

    min_.resize(max_size);
    max_.resize(max_size);
    for (size_t i = 0; i < min_.size(); ++i) {
        min_[i] = sm->make_state_int(0);
        max_[i] = sm->make_state_int((ssize_t)1 << 51);
    }
}

template <typename T>
IntervalArray<T>::IntervalArray(StateManager* sm, ssize_t size)
        : IntervalArray<T>(sm, size, size) {}

template <>
IntervalArray<double>::IntervalArray(StateManager* sm, ssize_t min_size, ssize_t max_size,
                                     double lb, double ub) {
    if (min_size > max_size) {
        throw std::invalid_argument("Min size larger than max size");
    }

    if (lb > ub) {
        throw std::invalid_argument("lower bound larger than upper bound");
    }

    assert(min_size <= max_size);
    min_size_ = sm->make_state_int(min_size);
    max_size_ = sm->make_state_int(max_size);

    min_.resize(max_size);
    max_.resize(max_size);

    for (size_t i = 0; i < min_.size(); ++i) {
        min_[i] = sm->make_state_real(lb);
        max_[i] = sm->make_state_real(ub);
    }
}

template <>
IntervalArray<ssize_t>::IntervalArray(StateManager* sm, ssize_t min_size, ssize_t max_size,
                                      double lb, double ub) {
    if (min_size > max_size) throw std::invalid_argument("Min size larger than max size");

    if (lb < std::numeric_limits<ssize_t>::min())
        throw std::invalid_argument("lower bound too small for ssize_t");
    if (ub > std::numeric_limits<ssize_t>::max())
        throw std::invalid_argument("upper bound too big for ssize_t");

    if (std::ceil(lb) > std::floor(ub)) {
        throw std::invalid_argument("lower bound larger than upper bound");
    }

    assert(min_size <= max_size);
    min_size_ = sm->make_state_int(min_size);
    max_size_ = sm->make_state_int(max_size);

    // Set the bounds for each index
    min_.resize(max_size);
    max_.resize(max_size);

    for (size_t i = 0; i < min_.size(); ++i) {
        min_[i] = sm->make_state_int(std::ceil(lb));
        max_[i] = sm->make_state_int(std::floor(ub));
    }
}

template <typename T>
IntervalArray<T>::IntervalArray(StateManager* sm, ssize_t size, double lb, double ub)
        : IntervalArray<T>(sm, size, size, lb, ub) {}

template <>
IntervalArray<double>::IntervalArray(StateManager* sm, ssize_t min_size, std::vector<double> lb,
                                     std::vector<double> ub) {
    if (lb.size() != ub.size()) {
        throw std::invalid_argument("lower bounds and upper bounds have different sizes");
    }

    if (min_size > static_cast<ssize_t>(lb.size())) {
        throw std::invalid_argument("Min size larger than max size");
    }

    assert(min_size <= static_cast<ssize_t>(lb.size()));
    min_size_ = sm->make_state_int(min_size);
    max_size_ = sm->make_state_int(static_cast<ssize_t>(lb.size()));

    min_.resize(lb.size());
    max_.resize(lb.size());
    for (size_t i = 0; i < lb.size(); ++i) {
        if (lb[i] > ub[i]) {
            throw std::invalid_argument("lower bound larger than upper bound");
        }

        min_[i] = sm->make_state_real(lb[i]);
        max_[i] = sm->make_state_real(ub[i]);
    }
}

template <>
IntervalArray<ssize_t>::IntervalArray(StateManager* sm, ssize_t min_size, std::vector<double> lb,
                                      std::vector<double> ub) {
    if (lb.size() != ub.size()) {
        throw std::invalid_argument("lower bounds and upper bounds have different sizes");
    }

    if (min_size > static_cast<ssize_t>(lb.size())) {
        throw std::invalid_argument("Min size larger than max size");
    }

    assert(min_size <= static_cast<ssize_t>(lb.size()));
    min_size_ = sm->make_state_int(min_size);
    max_size_ = sm->make_state_int(lb.size());

    min_.resize(lb.size());
    max_.resize(ub.size());

    for (size_t i = 0; i < lb.size(); ++i) {
        if (lb[i] < std::numeric_limits<ssize_t>::min())
            throw std::invalid_argument("lower bound too small for ssize_t");
        if (ub[i] > std::numeric_limits<ssize_t>::max())
            throw std::invalid_argument("upper bound too big for ssize_t");
        if (lb[i] > ub[i]) throw std::invalid_argument("lower bound larger than upper bound");
        min_[i] = sm->make_state_int(std::ceil(lb[i]));
        max_[i] = sm->make_state_int(std::floor(ub[i]));
    }
}

template <typename T>
IntervalArray<T>::IntervalArray(StateManager* sm, std::vector<double> lb, std::vector<double> ub)
        : IntervalArray<T>(sm, lb.size(), lb, ub) {}

template <typename T>
double IntervalArray<T>::min(int index) const {
    assert(index < static_cast<int>(min_.size()));
    return min_[index]->get_value();
}

template <typename T>
double IntervalArray<T>::max(int index) const {
    assert(index < static_cast<int>(max_.size()));
    return max_[index]->get_value();
}

template <>
double IntervalArray<double>::size(int index) const {
    assert(index < static_cast<int>(min_.size()));
    return max_[index]->get_value() - min_[index]->get_value();
}

template <>
double IntervalArray<ssize_t>::size(int index) const {
    assert(index < static_cast<int>(min_.size()));
    return max_[index]->get_value() - min_[index]->get_value() + 1;
}

template <>
bool IntervalArray<double>::is_bound(int index) const {
    return (this->size(index) == 0) and this->is_active(index);
}

template <>
bool IntervalArray<ssize_t>::is_bound(int index) const {
    return (this->size(index) == 1) and this->is_active(index);
}

template <>
bool IntervalArray<double>::contains(double value, int index) const {
    if (value > max_[index]->get_value()) return false;
    if (value < min_[index]->get_value()) return false;
    return true;
}

template <>
bool IntervalArray<ssize_t>::contains(double value, int index) const {
    if (value > max_[index]->get_value()) return false;
    if (value < min_[index]->get_value()) return false;
    if (std::floor(value) != std::ceil(value)) return false;
    return true;
}

template <typename T>
bool IntervalArray<T>::is_active(int index) const {
    return index < min_size_->get_value();
}

template <typename T>
bool IntervalArray<T>::maybe_active(int index) const {
    return index < max_size_->get_value();
}

template <>
CPStatus IntervalArray<double>::remove(double value, int index, DomainListener* l) {
    // can't really remove a double, even from the boundary
    return CPStatus::OK;
}

template <>
CPStatus IntervalArray<ssize_t>::remove(double value, int index, DomainListener* l) {
    // This is an interval so we can only remove from the boundary
    if (this->contains(value, index)) {
        // if there's only this value, then domain is wiped out
        if (this->is_bound(index) and this->is_active(index)) return CPStatus::Inconsistency;

        // The variable domain is fixed at the value we want to remove, but the variable is
        // optional. Then this variabe becomes inactive. We can set the maximum size of the domain
        // to the current index
        if (this->is_bound(index) and this->maybe_active(index)) {
            return this->update_max_size(index, l);
        }

        bool change_max = value == max_[index]->get_value();
        bool change_min = value == min_[index]->get_value();

        if (change_max) {
            max_[index]->set_value(value - 1);
            l->change_max(index);
        } else if (change_min) {
            min_[index]->set_value(value + 1);
            l->change_min(index);
        }
        l->change(index);
    }
    return CPStatus::OK;
}

template <>
CPStatus IntervalArray<double>::remove_above(double value, int index, DomainListener* l) {
    // Wipe-out all domain on a mandatory variable, inconsistency.
    if (min_[index]->get_value() > value and this->is_active(index)) return CPStatus::Inconsistency;

    // Wipe out of the domain of an optional variable. We can set the maximum size of the domain
    // to the current index
    if (min_[index]->get_value() > value and this->maybe_active(index)) {
        return this->update_max_size(index, l);
    }

    // nothing to do
    if (max_[index]->get_value() <= value) return CPStatus::OK;

    max_[index]->set_value(value);
    l->change_max(index);
    l->change(index);
    return CPStatus::OK;
}

template <>
CPStatus IntervalArray<ssize_t>::remove_above(double value, int index, DomainListener* l) {
    // Wipe-out all domain on a mandatory variable, inconsistency.
    if (min_[index]->get_value() > value and this->is_active(index)) return CPStatus::Inconsistency;

    // Wipe out of the domain of an optional variable. We can set the maximum size of the domain
    // to the current index
    if (min_[index]->get_value() > value and this->maybe_active(index)) {
        return this->update_max_size(index, l);
    }

    // nothing to do
    if (max_[index]->get_value() <= value) return CPStatus::OK;

    max_[index]->set_value(std::floor(value));
    l->change_max(index);
    l->change(index);
    return CPStatus::OK;
}

template <>
CPStatus IntervalArray<double>::remove_below(double value, int index, DomainListener* l) {
    // Wipe-out all domain on a mandatory variable, inconsistency.
    if (max_[index]->get_value() < value and this->is_active(index)) return CPStatus::Inconsistency;

    // Wipe out of the domain of an optional variable. We can set the maximum size of the domain
    // to the current index
    if (max_[index]->get_value() < value and this->maybe_active(index)) {
        return this->update_max_size(index, l);
    }

    // nothing to do
    if (min_[index]->get_value() >= value) return CPStatus::OK;

    min_[index]->set_value(value);
    l->change_min(index);
    l->change(index);
    return CPStatus::OK;
}

template <>
CPStatus IntervalArray<ssize_t>::remove_below(double value, int index, DomainListener* l) {
    // Wipe-out all domain on a mandatory variable, inconsistency.
    if (max_[index]->get_value() < value and this->is_active(index)) return CPStatus::Inconsistency;

    // Wipe out of the domain of an optional variable. We can set the maximum size of the domain
    // to the current index
    if (max_[index]->get_value() < value and this->maybe_active(index)) {
        return this->update_max_size(index, l);
    }

    // nothing to do
    if (min_[index]->get_value() >= value) return CPStatus::OK;

    min_[index]->set_value(std::ceil(value));
    l->change_min(index);
    l->change(index);
    return CPStatus::OK;
}

template <typename T>
CPStatus IntervalArray<T>::remove_all_but(double value, int index, DomainListener* l) {
    // If the value is not contained, wipe-out a mandatory variable, cause an inconsistency
    if (not this->contains(value, index) and this->is_active(index)) return CPStatus::Inconsistency;

    // Wipe out of the domain of an optional variable. We can set the maximum size of the domain
    // to the current index
    if (not this->contains(value, index) and this->maybe_active(index)) {
        return this->update_max_size(index, l);
    }

    if (this->contains(value, index) and this->is_bound(index)) {
        // nothing to do here, the domain is already fixed to this value
        return CPStatus::OK;
    }

    bool changed_min = (value == min_[index]->get_value());
    bool changed_max = (value == max_[index]->get_value());

    min_[index]->set_value(value);
    max_[index]->set_value(value);

    if (changed_max) l->change_max(index);
    if (changed_min) l->change_min(index);
    l->change(index);
    l->bind(index);

    return CPStatus::OK;
}

template <typename T>
CPStatus IntervalArray<T>::update_min_size(int new_min_size, DomainListener* l) {
    // Simple case, nothing to do
    if (new_min_size < min_size_->get_value()) return CPStatus::OK;

    // Wipeout of the domain for the size variable
    if (new_min_size > max_size_->get_value()) return CPStatus::Inconsistency;

    if (new_min_size != min_size_->get_value()) {
        min_size_->set_value(new_min_size);
        l->change_array_size(new_min_size);
    }
    return CPStatus::OK;
}

template <typename T>
CPStatus IntervalArray<T>::update_max_size(int new_max_size, DomainListener* l) {
    // Simple case, nothing to do
    if (new_max_size > max_size_->get_value()) return CPStatus::OK;

    // Wipeout of the domain for the size variable
    if (new_max_size < min_size_->get_value()) return CPStatus::Inconsistency;

    if (new_max_size != max_size_->get_value()) {
        max_size_->set_value(new_max_size);
        l->change_array_size(new_max_size);
    }
    return CPStatus::OK;
}

template class IntervalArray<double>;
template class IntervalArray<ssize_t>;

}  // namespace dwave::optimization::cp

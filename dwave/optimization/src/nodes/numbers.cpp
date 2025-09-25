// Copyright 2024 D-Wave Systems Inc.
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

#include "dwave-optimization/nodes/numbers.hpp"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>

#include "_state.hpp"

namespace dwave::optimization {

// Base class to be used as interfaces.

double const* NumberNode::buff(const State& state) const noexcept {
    return data_ptr<ArrayNodeStateData>(state)->buff();
}

std::span<const Update> NumberNode::diff(const State& state) const noexcept {
    return data_ptr<ArrayNodeStateData>(state)->diff();
}

double NumberNode::min() const { return min_; }

double NumberNode::max() const { return max_; }

void NumberNode::initialize_state(State& state, std::vector<double>&& number_data) const {
    if (number_data.size() != static_cast<size_t>(this->size())) {
        throw std::invalid_argument("Size of data provided does not match node size");
    }
    for (ssize_t index = 0, stop = this->size(); index < stop; ++index) {
        if (!is_valid(index, number_data[index])) {
            throw std::invalid_argument("Invalid data provided for node");
        }
    }

    emplace_data_ptr<ArrayNodeStateData>(state, std::move(number_data));
}

void NumberNode::initialize_state(State& state) const {
    std::vector<double> values;
    values.reserve(this->size());
    for (ssize_t i = 0, stop = this->size(); i < stop; ++i) {
        values.push_back(default_value(i));
    }
    initialize_state(state, values);
}

void NumberNode::commit(State& state) const noexcept {
    data_ptr<ArrayNodeStateData>(state)->commit();
}

void NumberNode::revert(State& state) const noexcept {
    data_ptr<ArrayNodeStateData>(state)->revert();
}

bool NumberNode::exchange(State& state, ssize_t i, ssize_t j) const {
    double value = data_ptr<ArrayNodeStateData>(state)->get(j);
    if ((lower_bound(i) > value) || (upper_bound(i) < value)) {
        return false;
    }
    value = data_ptr<ArrayNodeStateData>(state)->get(i);
    if ((lower_bound(j) > value) || (upper_bound(j) < value)) {
        return false;
    }
    return data_ptr<ArrayNodeStateData>(state)->exchange(i, j);
}

double NumberNode::get_value(State& state, ssize_t i) const {
    return data_ptr<ArrayNodeStateData>(state)->get(i);
}

bool NumberNode::clip_and_set_value(State& state, ssize_t index, double value) const {
    value = std::clamp(value, lower_bound(index), upper_bound(index));
    return data_ptr<ArrayNodeStateData>(state)->set(index, value);
}

// Integer Node ***************************************************************
template <bool maximum>
double get_extreme_bound(const std::optional<std::vector<double>>& bound, const int default_bound) {
    if (!bound) {
        return static_cast<double>(default_bound);
    }
    std::vector<double>::const_iterator it;
    if (maximum) {
        it = std::max_element(bound->begin(), bound->end());
    } else {
        it = std::min_element(bound->begin(), bound->end());
    }
    if (it != bound->end()) {
        return *it;
    }
    return static_cast<double>(default_bound);
}

void assign_bound(const std::optional<std::vector<double>>& bound, std::vector<double>& full_bound_,
                  double default_bound) {
    if (!bound) {
        full_bound_ = std::vector<double>{default_bound};
        return;
    }
    assert(!(bound->empty()));
    full_bound_ = std::move(*bound);
}

IntegerNode::IntegerNode(std::span<const ssize_t> shape,
                         std::optional<std::vector<double>> lower_bound,
                         std::optional<std::vector<double>> upper_bound)
        : NumberNode(shape, get_extreme_bound<false>(lower_bound, default_lower_bound),
                     get_extreme_bound<true>(upper_bound, default_upper_bound)) {
    assign_bound(lower_bound, full_lower_bound_, default_lower_bound);
    assign_bound(upper_bound, full_upper_bound_, default_upper_bound);
    bool index_wise_bound = false;
    // If lower bound is index-wise, it must be correct size
    if (full_lower_bound_.size() > 1) {
        index_wise_bound = true;
        if (static_cast<ssize_t>(full_lower_bound_.size()) != this->size()) {
            throw std::invalid_argument("lower_bound must match size of node");
        }
    }
    // If upper bound is index-wise, it must be correct size
    if (full_upper_bound_.size() > 1) {
        index_wise_bound = true;
        if (static_cast<ssize_t>(full_upper_bound_.size()) != this->size()) {
            throw std::invalid_argument("upper_bound must match size of node");
        }
    }
    if (min_ < minimum_lower_bound || max_ > maximum_upper_bound) {
        throw std::invalid_argument("range provided for integers exceeds supported range");
    }
    // if at least one of the bounds is index-wise, check that there are no
    // violations at any of the indices
    if (index_wise_bound) {
        for (ssize_t i = 0, stop = this->size(); i < stop; ++i) {
            if (this->lower_bound(i) > this->upper_bound(i)) {
                throw std::invalid_argument("Bounds of index " + std::to_string(i) + " clash");
            }
        }
    }
}

IntegerNode::IntegerNode(std::initializer_list<ssize_t> shape,
                         std::optional<std::vector<double>> lower_bound,
                         std::optional<std::vector<double>> upper_bound)
        : IntegerNode(std::span(shape), lower_bound, upper_bound) {}
IntegerNode::IntegerNode(ssize_t size, std::optional<std::vector<double>> lower_bound,
                         std::optional<std::vector<double>> upper_bound)
        : IntegerNode({size}, lower_bound, upper_bound) {}

IntegerNode::IntegerNode(std::span<const ssize_t> shape, double lower_bound,
                         std::optional<std::vector<double>> upper_bound)
        : IntegerNode(shape, std::vector<double>{lower_bound}, upper_bound) {}
IntegerNode::IntegerNode(std::initializer_list<ssize_t> shape, double lower_bound,
                         std::optional<std::vector<double>> upper_bound)
        : IntegerNode(std::span(shape), std::vector<double>{lower_bound}, upper_bound) {}
IntegerNode::IntegerNode(ssize_t size, double lower_bound,
                         std::optional<std::vector<double>> upper_bound)
        : IntegerNode({size}, std::vector<double>{lower_bound}, upper_bound) {}

IntegerNode::IntegerNode(std::span<const ssize_t> shape,
                         std::optional<std::vector<double>> lower_bound, double upper_bound)
        : IntegerNode(shape, lower_bound, std::vector<double>{upper_bound}) {}
IntegerNode::IntegerNode(std::initializer_list<ssize_t> shape,
                         std::optional<std::vector<double>> lower_bound, double upper_bound)
        : IntegerNode(std::span(shape), lower_bound, std::vector<double>{upper_bound}) {}
IntegerNode::IntegerNode(ssize_t size, std::optional<std::vector<double>> lower_bound,
                         double upper_bound)
        : IntegerNode({size}, lower_bound, std::vector<double>{upper_bound}) {}

IntegerNode::IntegerNode(std::span<const ssize_t> shape, double lower_bound, double upper_bound)
        : IntegerNode(shape, std::vector<double>{lower_bound}, std::vector<double>{upper_bound}) {}
IntegerNode::IntegerNode(std::initializer_list<ssize_t> shape, double lower_bound,
                         double upper_bound)
        : IntegerNode(std::span(shape), std::vector<double>{lower_bound},
                      std::vector<double>{upper_bound}) {}
IntegerNode::IntegerNode(ssize_t size, double lower_bound, double upper_bound)
        : IntegerNode({size}, std::vector<double>{lower_bound}, std::vector<double>{upper_bound}) {}

bool IntegerNode::integral() const { return true; }

double IntegerNode::lower_bound(ssize_t index) const {
    if (full_lower_bound_.size() == 1) {
        return full_lower_bound_[0];
    }
    assert(full_lower_bound_.size() > 1);
    return full_lower_bound_[index];
}

double IntegerNode::lower_bound() const {
    if (full_lower_bound_.size() > 1) {
        throw std::out_of_range(
                "IntegerNode has multiple lower bounds, use lower_bound(index) instead");
    }
    return full_lower_bound_[0];
}

double IntegerNode::upper_bound(ssize_t index) const {
    if (full_upper_bound_.size() == 1) {
        return full_upper_bound_[0];
    }
    assert(full_upper_bound_.size() > 1);
    return full_upper_bound_[index];
}

double IntegerNode::upper_bound() const {
    if (full_upper_bound_.size() > 1) {
        throw std::out_of_range(
                "IntegerNode has multiple uppper bounds, use upper_bound(index) instead");
    }
    return full_upper_bound_[0];
}

bool IntegerNode::is_valid(ssize_t index, double value) const {
    return (value >= lower_bound(index)) && (value <= upper_bound(index)) &&
           (std::round(value) == value);
}

bool IntegerNode::set_value(State& state, ssize_t index, double value) const {
    if (!is_valid(index, value)) {
        throw std::invalid_argument("Invalid integer value provided");
    }
    if (value >= lower_bound(index) && value <= upper_bound(index)) {
        return data_ptr<ArrayNodeStateData>(state)->set(index, value);
    }
    return false;
}

double IntegerNode::default_value(ssize_t index) const {
    return (lower_bound(index) <= 0 && upper_bound(index) >= 0) ? 0 : lower_bound(index);
}

// Binary Node ****************************************************************

template <bool upper_bound>
double get_bool_bound(const double bound) {
    if (upper_bound) {
        // reduce upper bound to fit boolean range
        if (bound >= 1.0) return 1.0;
        if (bound < 0) {
            throw std::invalid_argument("Upper bound is smaller than 0");
        }
        return 0.0;  // round down since 0 <= bound < 1
    } else {
        // increase lower bound to fit boolean range
        if (bound <= 0.0) return 0.0;
        if (bound > 1.0) {
            throw std::invalid_argument("Lower bound is greater than 1");
        }
        return 1.0;  // round up since 0 < bound <= 1
    }
}

template <bool upper_bound>
std::vector<double> limit_bound_to_bool_domain(const std::optional<std::vector<double>>& bound) {
    if (!bound) {
        // set default boolean bounds if user does not provide bounds
        double output = (upper_bound) ? 1.0 : 0.0;
        return std::vector<double>{output};
    }
    const ssize_t vec_size = bound->size();
    std::vector<double> output_vec;
    output_vec.reserve(vec_size);
    for (const double& value : bound.value()) {
        output_vec.emplace_back(get_bool_bound<upper_bound>(value));
    }
    return output_vec;
}

BinaryNode::BinaryNode(std::span<const ssize_t> shape,
                       std::optional<std::vector<double>> lower_bound,
                       std::optional<std::vector<double>> upper_bound)
        : IntegerNode(shape, limit_bound_to_bool_domain<false>(lower_bound),
                      limit_bound_to_bool_domain<true>(upper_bound)) {}

BinaryNode::BinaryNode(std::initializer_list<ssize_t> shape,
                       std::optional<std::vector<double>> lower_bound,
                       std::optional<std::vector<double>> upper_bound)
        : BinaryNode(std::span(shape), lower_bound, upper_bound) {}
BinaryNode::BinaryNode(ssize_t size, std::optional<std::vector<double>> lower_bound,
                       std::optional<std::vector<double>> upper_bound)
        : BinaryNode({size}, lower_bound, upper_bound) {}

BinaryNode::BinaryNode(std::span<const ssize_t> shape, double lower_bound,
                       std::optional<std::vector<double>> upper_bound)
        : BinaryNode(shape, std::vector<double>{lower_bound}, upper_bound) {}
BinaryNode::BinaryNode(std::initializer_list<ssize_t> shape, double lower_bound,
                       std::optional<std::vector<double>> upper_bound)
        : BinaryNode(std::span(shape), std::vector<double>{lower_bound}, upper_bound) {}
BinaryNode::BinaryNode(ssize_t size, double lower_bound,
                       std::optional<std::vector<double>> upper_bound)
        : BinaryNode({size}, std::vector<double>{lower_bound}, upper_bound) {}

BinaryNode::BinaryNode(std::span<const ssize_t> shape,
                       std::optional<std::vector<double>> lower_bound, double upper_bound)
        : BinaryNode(shape, lower_bound, std::vector<double>{upper_bound}) {}
BinaryNode::BinaryNode(std::initializer_list<ssize_t> shape,
                       std::optional<std::vector<double>> lower_bound, double upper_bound)
        : BinaryNode(std::span(shape), lower_bound, std::vector<double>{upper_bound}) {}
BinaryNode::BinaryNode(ssize_t size, std::optional<std::vector<double>> lower_bound,
                       double upper_bound)
        : BinaryNode({size}, lower_bound, std::vector<double>{upper_bound}) {}

BinaryNode::BinaryNode(std::span<const ssize_t> shape, double lower_bound, double upper_bound)
        : BinaryNode(shape, std::vector<double>{lower_bound}, std::vector<double>{upper_bound}) {}
BinaryNode::BinaryNode(std::initializer_list<ssize_t> shape, double lower_bound, double upper_bound)
        : BinaryNode(std::span(shape), std::vector<double>{lower_bound},
                     std::vector<double>{upper_bound}) {}
BinaryNode::BinaryNode(ssize_t size, double lower_bound, double upper_bound)
        : BinaryNode({size}, std::vector<double>{lower_bound}, std::vector<double>{upper_bound}) {}

bool BinaryNode::flip(State& state, ssize_t i) const {
    auto ptr = data_ptr<ArrayNodeStateData>(state);
    if (lower_bound(i) == upper_bound(i)) {
        // variable is fixed
        return false;
    }

    ptr->set(i, !ptr->get(i));
    return true;
}

bool BinaryNode::set(State& state, ssize_t i) const {
    if (upper_bound(i) >= 1) {
        assert(lower_bound(i) <= 1);
        return data_ptr<ArrayNodeStateData>(state)->set(i, 1.0);
    }
    return false;
}

bool BinaryNode::unset(State& state, ssize_t i) const {
    if (lower_bound(i) <= 0) {
        assert(upper_bound(i) >= 0);
        return data_ptr<ArrayNodeStateData>(state)->set(i, 0.0);
    }
    return false;
}

}  // namespace dwave::optimization

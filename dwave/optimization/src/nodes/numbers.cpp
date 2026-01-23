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
    initialize_state(state, std::move(values));
}

void NumberNode::commit(State& state) const noexcept {
    data_ptr<ArrayNodeStateData>(state)->commit();
}

void NumberNode::revert(State& state) const noexcept {
    data_ptr<ArrayNodeStateData>(state)->revert();
}

void NumberNode::exchange(State& state, ssize_t i, ssize_t j) const {
    auto ptr = data_ptr<ArrayNodeStateData>(state);
    // We expect the exchange to obey the index-wise bounds.
    assert(lower_bound(i) <= ptr->get(j));
    assert(upper_bound(i) >= ptr->get(j));
    assert(lower_bound(j) <= ptr->get(i));
    assert(upper_bound(j) >= ptr->get(i));
    // Assert that i and j are valid indices occurs in ptr->exchange().
    // Exchange occurs IFF (i != j) and (buffer[i] != buffer[j]).
    ptr->exchange(i, j);
}

double NumberNode::get_value(State& state, ssize_t i) const {
    return data_ptr<ArrayNodeStateData>(state)->get(i);
}

double NumberNode::lower_bound(ssize_t index) const {
    if (lower_bounds_.size() == 1) {
        return lower_bounds_[0];
    }
    assert(lower_bounds_.size() > 1);
    assert(0 <= index && index < static_cast<ssize_t>(lower_bounds_.size()));
    return lower_bounds_[index];
}

double NumberNode::lower_bound() const {
    if (lower_bounds_.size() > 1) {
        throw std::out_of_range(
                "Number array has multiple lower bounds, use lower_bound(index) instead");
    }
    return lower_bounds_[0];
}

double NumberNode::upper_bound(ssize_t index) const {
    if (upper_bounds_.size() == 1) {
        return upper_bounds_[0];
    }
    assert(upper_bounds_.size() > 1);
    assert(0 <= index && index < static_cast<ssize_t>(upper_bounds_.size()));
    return upper_bounds_[index];
}

double NumberNode::upper_bound() const {
    if (upper_bounds_.size() > 1) {
        throw std::out_of_range(
                "Number array has multiple upper bounds, use upper_bound(index) instead");
    }
    return upper_bounds_[0];
}

void NumberNode::clip_and_set_value(State& state, ssize_t index, double value) const {
    value = std::clamp(value, lower_bound(index), upper_bound(index));
    // Assert that i is a valid index occurs in data_ptr->set().
    // Set occurs IFF `value` != buffer[i] .
    data_ptr<ArrayNodeStateData>(state)->set(index, value);
}

template <bool maximum>
double get_extreme_index_wise_bound(const std::vector<double>& bound) {
    assert(bound.size() > 0);
    std::vector<double>::const_iterator it;
    if (maximum) {
        it = std::max_element(bound.begin(), bound.end());
    } else {
        it = std::min_element(bound.begin(), bound.end());
    }
    return *it;
}

void check_index_wise_bounds(const NumberNode& node, const std::vector<double>& lower_bounds_,
                             const std::vector<double>& upper_bounds_) {
    bool index_wise_bound = false;
    // If lower bound is index-wise, it must be correct size.
    if (lower_bounds_.size() > 1) {
        index_wise_bound = true;
        if (static_cast<ssize_t>(lower_bounds_.size()) != node.size()) {
            throw std::invalid_argument("lower_bound must match size of node");
        }
    }
    // If upper bound is index-wise, it must be correct size.
    if (upper_bounds_.size() > 1) {
        index_wise_bound = true;
        if (static_cast<ssize_t>(upper_bounds_.size()) != node.size()) {
            throw std::invalid_argument("upper_bound must match size of node");
        }
    }
    // If at least one of the bounds is index-wise, check that there are no
    // violations at any of the indices.
    if (index_wise_bound) {
        for (ssize_t i = 0, stop = node.size(); i < stop; ++i) {
            if (node.lower_bound(i) > node.upper_bound(i)) {
                throw std::invalid_argument("Bounds of index " + std::to_string(i) + " clash");
            }
        }
    }
}

NumberNode::NumberNode(std::span<const ssize_t> shape, std::vector<double> lower_bound,
                       std::vector<double> upper_bound)
        : ArrayOutputMixin(shape),
          min_(get_extreme_index_wise_bound<false>(lower_bound)),
          max_(get_extreme_index_wise_bound<true>(upper_bound)),
          lower_bounds_(std::move(lower_bound)),
          upper_bounds_(std::move(upper_bound)) {
    if ((shape.size() > 0) && (shape[0] < 0)) {
        throw std::invalid_argument("Number array cannot have dynamic size.");
    }

    if (max_ < min_) {
        throw std::invalid_argument("Invalid range for number array provided.");
    }

    check_index_wise_bounds(*this, lower_bounds_, upper_bounds_);
}

// Integer Node ***************************************************************

IntegerNode::IntegerNode(std::span<const ssize_t> shape,
                         std::optional<std::vector<double>> lower_bound,
                         std::optional<std::vector<double>> upper_bound)
        : NumberNode(shape,
                     lower_bound.has_value() ? std::move(*lower_bound)
                                             : std::vector<double>{default_lower_bound},
                     upper_bound.has_value() ? std::move(*upper_bound)
                                             : std::vector<double>{default_upper_bound}) {
    if (min_ < minimum_lower_bound || max_ > maximum_upper_bound) {
        throw std::invalid_argument("range provided for integers exceeds supported range");
    }
}

IntegerNode::IntegerNode(std::initializer_list<ssize_t> shape,
                         std::optional<std::vector<double>> lower_bound,
                         std::optional<std::vector<double>> upper_bound)
        : IntegerNode(std::span(shape), std::move(lower_bound), std::move(upper_bound)) {}
IntegerNode::IntegerNode(ssize_t size, std::optional<std::vector<double>> lower_bound,
                         std::optional<std::vector<double>> upper_bound)
        : IntegerNode({size}, std::move(lower_bound), std::move(upper_bound)) {}

IntegerNode::IntegerNode(std::span<const ssize_t> shape, double lower_bound,
                         std::optional<std::vector<double>> upper_bound)
        : IntegerNode(shape, std::vector<double>{lower_bound}, std::move(upper_bound)) {}
IntegerNode::IntegerNode(std::initializer_list<ssize_t> shape, double lower_bound,
                         std::optional<std::vector<double>> upper_bound)
        : IntegerNode(std::span(shape), std::vector<double>{lower_bound}, std::move(upper_bound)) {}
IntegerNode::IntegerNode(ssize_t size, double lower_bound,
                         std::optional<std::vector<double>> upper_bound)
        : IntegerNode({size}, std::vector<double>{lower_bound}, std::move(upper_bound)) {}

IntegerNode::IntegerNode(std::span<const ssize_t> shape,
                         std::optional<std::vector<double>> lower_bound, double upper_bound)
        : IntegerNode(shape, std::move(lower_bound), std::vector<double>{upper_bound}) {}
IntegerNode::IntegerNode(std::initializer_list<ssize_t> shape,
                         std::optional<std::vector<double>> lower_bound, double upper_bound)
        : IntegerNode(std::span(shape), std::move(lower_bound), std::vector<double>{upper_bound}) {}
IntegerNode::IntegerNode(ssize_t size, std::optional<std::vector<double>> lower_bound,
                         double upper_bound)
        : IntegerNode({size}, std::move(lower_bound), std::vector<double>{upper_bound}) {}

IntegerNode::IntegerNode(std::span<const ssize_t> shape, double lower_bound, double upper_bound)
        : IntegerNode(shape, std::vector<double>{lower_bound}, std::vector<double>{upper_bound}) {}
IntegerNode::IntegerNode(std::initializer_list<ssize_t> shape, double lower_bound,
                         double upper_bound)
        : IntegerNode(std::span(shape), std::vector<double>{lower_bound},
                      std::vector<double>{upper_bound}) {}
IntegerNode::IntegerNode(ssize_t size, double lower_bound, double upper_bound)
        : IntegerNode({size}, std::vector<double>{lower_bound}, std::vector<double>{upper_bound}) {}

bool IntegerNode::integral() const { return true; }

bool IntegerNode::is_valid(ssize_t index, double value) const {
    return (value >= lower_bound(index)) && (value <= upper_bound(index)) &&
           (std::round(value) == value);
}

void IntegerNode::set_value(State& state, ssize_t index, double value) const {
    // We expect `value` to obey the index-wise bounds and to be an integer.
    assert(lower_bound(index) <= value);
    assert(upper_bound(index) >= value);
    assert(value == std::round(value));
    // Assert that i is a valid index occurs in data_ptr->set().
    // Set occurs IFF `value` != buffer[i] .
    data_ptr<ArrayNodeStateData>(state)->set(index, value);
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
std::vector<double> limit_bound_to_bool_domain(std::optional<std::vector<double>>& bound) {
    // set default boolean bounds if user does not provide bounds
    if (!bound) return std::vector<double>{upper_bound ? 1.0 : 0.0};

    for (double& value : *bound) value = get_bool_bound<upper_bound>(value);
    return *bound;
}

BinaryNode::BinaryNode(std::span<const ssize_t> shape,
                       std::optional<std::vector<double>> lower_bound,
                       std::optional<std::vector<double>> upper_bound)
        : IntegerNode(shape, limit_bound_to_bool_domain<false>(lower_bound),
                      limit_bound_to_bool_domain<true>(upper_bound)) {}

BinaryNode::BinaryNode(std::initializer_list<ssize_t> shape,
                       std::optional<std::vector<double>> lower_bound,
                       std::optional<std::vector<double>> upper_bound)
        : BinaryNode(std::span(shape), std::move(lower_bound), std::move(upper_bound)) {}
BinaryNode::BinaryNode(ssize_t size, std::optional<std::vector<double>> lower_bound,
                       std::optional<std::vector<double>> upper_bound)
        : BinaryNode({size}, std::move(lower_bound), std::move(upper_bound)) {}

BinaryNode::BinaryNode(std::span<const ssize_t> shape, double lower_bound,
                       std::optional<std::vector<double>> upper_bound)
        : BinaryNode(shape, std::vector<double>{lower_bound}, std::move(upper_bound)) {}
BinaryNode::BinaryNode(std::initializer_list<ssize_t> shape, double lower_bound,
                       std::optional<std::vector<double>> upper_bound)
        : BinaryNode(std::span(shape), std::vector<double>{lower_bound}, std::move(upper_bound)) {}
BinaryNode::BinaryNode(ssize_t size, double lower_bound,
                       std::optional<std::vector<double>> upper_bound)
        : BinaryNode({size}, std::vector<double>{lower_bound}, std::move(upper_bound)) {}

BinaryNode::BinaryNode(std::span<const ssize_t> shape,
                       std::optional<std::vector<double>> lower_bound, double upper_bound)
        : BinaryNode(shape, std::move(lower_bound), std::vector<double>{upper_bound}) {}
BinaryNode::BinaryNode(std::initializer_list<ssize_t> shape,
                       std::optional<std::vector<double>> lower_bound, double upper_bound)
        : BinaryNode(std::span(shape), std::move(lower_bound), std::vector<double>{upper_bound}) {}
BinaryNode::BinaryNode(ssize_t size, std::optional<std::vector<double>> lower_bound,
                       double upper_bound)
        : BinaryNode({size}, std::move(lower_bound), std::vector<double>{upper_bound}) {}

BinaryNode::BinaryNode(std::span<const ssize_t> shape, double lower_bound, double upper_bound)
        : BinaryNode(shape, std::vector<double>{lower_bound}, std::vector<double>{upper_bound}) {}
BinaryNode::BinaryNode(std::initializer_list<ssize_t> shape, double lower_bound, double upper_bound)
        : BinaryNode(std::span(shape), std::vector<double>{lower_bound},
                     std::vector<double>{upper_bound}) {}
BinaryNode::BinaryNode(ssize_t size, double lower_bound, double upper_bound)
        : BinaryNode({size}, std::vector<double>{lower_bound}, std::vector<double>{upper_bound}) {}

void BinaryNode::flip(State& state, ssize_t i) const {
    auto ptr = data_ptr<ArrayNodeStateData>(state);
    // Variable should not be fixed.
    assert(lower_bound(i) != upper_bound(i));
    // Assert that i is a valid index occurs in ptr->set().
    // Set occurs IFF `value` != buffer[i] .
    ptr->set(i, !ptr->get(i));
}

void BinaryNode::set(State& state, ssize_t i) const {
    // We expect the set to obey the index-wise bounds.
    assert(upper_bound(i) == 1.0);
    // Assert that i is a valid index occurs in data_ptr->set().
    // Set occurs IFF `value` != buffer[i] .
    data_ptr<ArrayNodeStateData>(state)->set(i, 1.0);
}

void BinaryNode::unset(State& state, ssize_t i) const {
    // We expect the set to obey the index-wise bounds.
    assert(lower_bound(i) == 0.0);
    // Assert that i is a valid index occurs in data_ptr->set().
    // Set occurs IFF `value` != buffer[i] .
    data_ptr<ArrayNodeStateData>(state)->set(i, 0.0);
}

}  // namespace dwave::optimization

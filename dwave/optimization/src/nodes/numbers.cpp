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

#include "_state.hpp"

namespace dwave::optimization {

// Base class to be used as interfaces.

double const* NumberNode::buff(const State& state) const noexcept {
    return data_ptr<ArrayNodeStateData>(state)->buff();
}

std::span<const Update> NumberNode::diff(const State& state) const noexcept {
    return data_ptr<ArrayNodeStateData>(state)->diff();
}

void NumberNode::commit(State& state) const noexcept {
    data_ptr<ArrayNodeStateData>(state)->commit();
}

void NumberNode::revert(State& state) const noexcept {
    data_ptr<ArrayNodeStateData>(state)->revert();
}

double NumberNode::lower_bound() const { return lower_bound_; }
double NumberNode::upper_bound() const { return upper_bound_; }

std::pair<double, double> NumberNode::minmax(
        optional_cache_type<std::pair<double, double>>) const {
    return {lower_bound_, upper_bound_};
}

void NumberNode::initialize_state(State& state, std::vector<double>&& number_data) const {
    if (number_data.size() != static_cast<size_t>(this->size())) {
        throw std::invalid_argument("Size of data provided does not match node size");
    }
    if (auto it = std::find_if_not(number_data.begin(), number_data.end(),
                                   [&](double value) { return is_valid(value); });
        it != number_data.end()) {
        throw std::invalid_argument("Invalid data provided for node");
    }

    emplace_data_ptr<ArrayNodeStateData>(state, std::move(number_data));
}

void NumberNode::initialize_state(State& state) const {
    initialize_state(state, std::vector<double>(this->size(), default_value()));
}

// Specializations for the linear case
bool NumberNode::exchange(State& state, ssize_t i, ssize_t j) const {
    return data_ptr<ArrayNodeStateData>(state)->exchange(i, j);
}

double NumberNode::get_value(State& state, ssize_t i) const {
    return data_ptr<ArrayNodeStateData>(state)->get(i);
}

ssize_t NumberNode::linear_index(ssize_t x, ssize_t y) const {
    auto shape = this->shape();
    assert(this->ndim() == 2 && "Node must be of 2 dimensional for 2D indexing");
    assert(x >= 0 && x < shape[1] && "X index out of range");
    assert(y >= 0 && y < shape[0] && "Y index out of range");
    return x + y * shape[1];
}

// Integer Node

IntegerNode::IntegerNode() : IntegerNode({}) {}

IntegerNode::IntegerNode(std::span<const ssize_t> shape, std::optional<int> lower_bound,
                         std::optional<int> upper_bound)
        : NumberNode(shape, lower_bound.value_or(default_lower_bound),
                     upper_bound.value_or(default_upper_bound)) {
    if (lower_bound_ < minimum_lower_bound || upper_bound_ > maximum_upper_bound) {
        throw std::invalid_argument("range provided for integers exceeds supported range");
    }
}

IntegerNode::IntegerNode(std::initializer_list<ssize_t> shape, std::optional<int> lower_bound,
                         std::optional<int> upper_bound)
        : IntegerNode(std::span(shape), lower_bound, upper_bound) {}

IntegerNode::IntegerNode(ssize_t size, std::optional<int> lower_bound,
                         std::optional<int> upper_bound)
        : IntegerNode({size}, lower_bound, upper_bound) {}

bool IntegerNode::integral() const { return true; }

bool IntegerNode::is_valid(double value) const {
    return (value >= lower_bound()) && (value <= upper_bound()) && (std::round(value) == value);
}

double IntegerNode::default_value() const {
    return (lower_bound() <= 0 && upper_bound() >= 0) ? 0 : lower_bound();
}

bool IntegerNode::set_value(State& state, ssize_t i, int value) const {
    if (!is_valid(value)) {
        throw std::invalid_argument("Invalid integer value provided");
    }
    return data_ptr<ArrayNodeStateData>(state)->set(i, value);
}

// Binary Node

void BinaryNode::flip(State& state, ssize_t i) const {
    auto ptr = data_ptr<ArrayNodeStateData>(state);
    if (ptr->get(i)) {
        ptr->set(i, 0);
    } else {
        ptr->set(i, 1);
    }
}

bool BinaryNode::set(State& state, ssize_t i) const {
    return data_ptr<ArrayNodeStateData>(state)->set(i, 1);
}

bool BinaryNode::unset(State& state, ssize_t i) const {
    return data_ptr<ArrayNodeStateData>(state)->set(i, 0);
}

}  // namespace dwave::optimization

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
    for (ssize_t index = 0, stop = this->size(); index < stop; index++) {
        if (!is_valid(index, number_data[index])) {
            throw std::invalid_argument("Invalid data provided for node");
        }
    }

    emplace_data_ptr<ArrayNodeStateData>(state, std::move(number_data));
}

void NumberNode::initialize_state(State& state) const {
    std::vector<double> values;
    values.reserve(this->size());
    for (ssize_t i = 0; i < this->size(); i++) {
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

bool NumberNode::clip_and_set_value(State& state, ssize_t index, double value) const {
    value = std::clamp(value, lower_bound(index), upper_bound(index));
    return data_ptr<ArrayNodeStateData>(state)->set(index, value);
}

// Integer Node ***************************************************************
template <bool maximum>
double get_extreme_bound(const IntegerNode::bounds_t& bound, const int default_bound) {
    if (std::holds_alternative<std::nullopt_t>(bound)) {
        return static_cast<double>(default_bound);
    } else if (std::holds_alternative<double>(bound)) {
        return std::get<double>(bound);
    }

    assert(std::holds_alternative<const std::vector<double>>(bound));
    const std::vector<double>& bound_vec = std::get<const std::vector<double>>(bound);
    std::vector<double>::const_iterator it;
    if (maximum) {
        it = std::max_element(bound_vec.begin(), bound_vec.end());
    } else {
        it = std::min_element(bound_vec.begin(), bound_vec.end());
    }
    if (it == bound_vec.end()) {
        return static_cast<double>(default_bound);
    }

    return *it;
}

IntegerNode::IntegerNode(std::span<const ssize_t> shape, const bounds_t lower_bound,
                         const bounds_t upper_bound)
        : NumberNode(shape, get_extreme_bound<false>(lower_bound, default_lower_bound),
                     get_extreme_bound<true>(upper_bound, default_upper_bound)),
          full_lower_bound_(lower_bound),
          full_upper_bound_(upper_bound) {
    bool index_wise_bound = false;
    // If upper bound is index-wise, it must be correct size
    if (std::holds_alternative<const std::vector<double>>(full_lower_bound_)) {
        index_wise_bound = true;
        if (static_cast<ssize_t>(std::get<const std::vector<double>>(full_lower_bound_).size()) !=
            this->size()) {
            throw std::invalid_argument("lower_bound must match size of node");
        }
    }
    // If lower bound is index-wise, it must be correct size
    if (std::holds_alternative<const std::vector<double>>(full_upper_bound_)) {
        index_wise_bound = true;
        if (static_cast<ssize_t>(std::get<const std::vector<double>>(full_upper_bound_).size()) !=
            this->size()) {
            throw std::invalid_argument("upper_bound must match size of node");
        }
    }
    if (min_ < minimum_lower_bound || max_ > maximum_upper_bound) {
        throw std::invalid_argument("range provided for integers exceeds supported range");
    }
    // if at least one of the bounds is index-wise, check that there are no
    // violations at any of the indices
    if (index_wise_bound) {
        for (ssize_t i = 0; i < this->size(); i++) {
            if (this->lower_bound(i) > this->upper_bound(i)) {
                throw std::invalid_argument("Bounds of index " + std::to_string(i) + " clash");
            }
        }
    }
}

IntegerNode::IntegerNode(std::initializer_list<ssize_t> shape, const bounds_t lower_bound,
                         const bounds_t upper_bound)
        : IntegerNode(std::span(shape), lower_bound, upper_bound) {}

IntegerNode::IntegerNode(ssize_t size, const bounds_t lower_bound, const bounds_t upper_bound)
        : IntegerNode({size}, lower_bound, upper_bound) {}

// Handle user-defined integer bounds
IntegerNode::bounds_t convert_bound_to_double_format(const IntegerNode::ssize_bounds_t& bound) {
    if (std::holds_alternative<std::nullopt_t>(bound)) {
        return std::nullopt;
    } else if (std::holds_alternative<ssize_t>(bound)) {
        return static_cast<double>(std::get<ssize_t>(bound));
    }
    assert(std::holds_alternative<const std::vector<ssize_t>>(bound));
    std::vector<double> output_vec(std::get<const std::vector<ssize_t>>(bound).begin(),
                                   std::get<const std::vector<ssize_t>>(bound).end());
    return output_vec;
}

IntegerNode::IntegerNode(std::span<const ssize_t> shape, const ssize_bounds_t lower_bound,
                         const ssize_bounds_t upper_bound)
        : IntegerNode(shape, convert_bound_to_double_format(lower_bound),
                      convert_bound_to_double_format(upper_bound)) {}

IntegerNode::IntegerNode(std::initializer_list<ssize_t> shape, const ssize_bounds_t lower_bound,
                         const ssize_bounds_t upper_bound)
        : IntegerNode(std::span(shape), convert_bound_to_double_format(lower_bound),
                      convert_bound_to_double_format(upper_bound)) {}

IntegerNode::IntegerNode(ssize_t size, const ssize_bounds_t lower_bound,
                         const ssize_bounds_t upper_bound)
        : IntegerNode({size}, convert_bound_to_double_format(lower_bound),
                      convert_bound_to_double_format(upper_bound)) {}

bool IntegerNode::integral() const { return true; }

// Depending upon user input, lower_bound() may return non-integral values
double IntegerNode::lower_bound(ssize_t index) const {
    if (std::holds_alternative<std::nullopt_t>(full_lower_bound_)) {
        return static_cast<double>(default_lower_bound);
    } else if (std::holds_alternative<double>(full_lower_bound_)) {
        return static_cast<double>(std::get<double>(full_lower_bound_));
    }
    assert(std::holds_alternative<const std::vector<double>>(full_lower_bound_));
    return std::get<const std::vector<double>>(full_lower_bound_)[index];
}

// Depending upon user input, upper_bound() may return non-integral values
double IntegerNode::upper_bound(ssize_t index) const {
    if (std::holds_alternative<std::nullopt_t>(full_upper_bound_)) {
        return static_cast<double>(default_upper_bound);
    } else if (std::holds_alternative<double>(full_upper_bound_)) {
        return std::get<double>(full_upper_bound_);
    }
    assert(std::holds_alternative<const std::vector<double>>(full_upper_bound_));
    return std::get<const std::vector<double>>(full_upper_bound_)[index];
}

bool IntegerNode::is_valid(ssize_t index, double value) const {
    return (value >= lower_bound(index)) && (value <= upper_bound(index)) &&
           (std::round(value) == value);
}

double IntegerNode::default_value(ssize_t index) const {
    return (lower_bound(index) <= 0 && upper_bound(index) >= 0) ? 0 : lower_bound(index);
}

bool IntegerNode::set_value(State& state, ssize_t index, double value) const {
    if (!is_valid(index, value)) {
        throw std::invalid_argument("Invalid integer value provided");
    }
    return data_ptr<ArrayNodeStateData>(state)->set(index, value);
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
IntegerNode::bounds_t limit_bound_to_bool_domain(const IntegerNode::bounds_t& bound) {
    if (std::holds_alternative<std::nullopt_t>(bound)) {
        // set default boolean bounds if user does not provide bounds
        double output = (upper_bound) ? 1.0 : 0.0;
        return output;
    } else if (std::holds_alternative<double>(bound)) {
        return get_bool_bound<upper_bound>(std::get<double>(bound));
    }
    const ssize_t vec_size = std::get<const std::vector<double>>(bound).size();
    std::vector<double> output_vec;
    output_vec.reserve(vec_size);
    for (ssize_t index = 0; index < vec_size; index++) {
        output_vec.emplace_back(
                get_bool_bound<upper_bound>(std::get<const std::vector<double>>(bound)[index]));
    }
    return output_vec;
}

BinaryNode::BinaryNode(std::span<const ssize_t> shape, const bounds_t lower_bound,
                       const bounds_t upper_bound)
        : IntegerNode(shape, limit_bound_to_bool_domain<false>(lower_bound),
                      limit_bound_to_bool_domain<true>(upper_bound)) {}

BinaryNode::BinaryNode(std::initializer_list<ssize_t> shape, const bounds_t lower_bound,
                       const bounds_t upper_bound)
        : IntegerNode(std::span(shape), limit_bound_to_bool_domain<false>(lower_bound),
                      limit_bound_to_bool_domain<true>(upper_bound)) {}

BinaryNode::BinaryNode(ssize_t size, const bounds_t lower_bound, const bounds_t upper_bound)
        : IntegerNode({size}, limit_bound_to_bool_domain<false>(lower_bound),
                      limit_bound_to_bool_domain<true>(upper_bound)) {}

// handle integer inputs
BinaryNode::BinaryNode(std::span<const ssize_t> shape, const ssize_bounds_t lower_bound,
                       const ssize_bounds_t upper_bound)
        : BinaryNode(shape, convert_bound_to_double_format(lower_bound),
                     convert_bound_to_double_format(upper_bound)) {}

BinaryNode::BinaryNode(std::initializer_list<ssize_t> shape, const ssize_bounds_t lower_bound,
                       const ssize_bounds_t upper_bound)
        : BinaryNode(std::span(shape), convert_bound_to_double_format(lower_bound),
                     convert_bound_to_double_format(upper_bound)) {}

BinaryNode::BinaryNode(ssize_t size, const ssize_bounds_t lower_bound,
                       const ssize_bounds_t upper_bound)
        : BinaryNode({size}, convert_bound_to_double_format(lower_bound),
                     convert_bound_to_double_format(upper_bound)) {}

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

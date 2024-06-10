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

#pragma once

#include <algorithm>
#include <memory>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/graph.hpp"

namespace dwave::optimization {

/// A contiguous block of numbers.
class NumberNode : public Node, public ArrayOutputMixin<Array>, public Decision {
 public:
    NumberNode() = delete;

    // Overloads needed by the Array ABC **************************************

    double const* buff(const State&) const noexcept override;
    std::span<const Update> diff(const State& state) const noexcept override;

    void commit(State&) const noexcept override;
    void revert(State&) const noexcept override;

    double min() const override;
    double max() const override;
    double lower_bound() const;
    double upper_bound() const;

    // Overloads required by the Node ABC *************************************

    void initialize_state(State& state) const override;
    void initialize_state(State& state, RngAdaptor& rng) const;

    template <std::ranges::range R>
    void initialize_state(State& state, R&& number_data) const;

    void initialize_state(State& state, std::vector<double>&& number_data) const;

    // Specializations for the linear case
    bool exchange(State& state, ssize_t i, ssize_t j) const;
    double get_value(State& state, ssize_t i) const;

    ssize_t linear_index(ssize_t x, ssize_t y) const;

 protected:
    explicit NumberNode(std::span<const ssize_t> shape, double lower_bound, double upper_bound)
            : ArrayOutputMixin(shape), lower_bound_(lower_bound), upper_bound_(upper_bound) {
        if (upper_bound_ < lower_bound_) {
            throw std::invalid_argument("Invalid range for number array provided");
        }
    }

    virtual bool is_valid(double value) const = 0;
    virtual double generate_value(RngAdaptor& rng) const = 0;
    virtual double default_value() const = 0;
    virtual std::unique_ptr<NodeStateData> new_data_ptr(
            std::vector<double>&& number_data) const = 0;

    double lower_bound_;
    double upper_bound_;
};

template <std::ranges::range R>
void NumberNode::initialize_state(State& state, R&& number_data) const {
    int index = this->topological_index();
    assert(index >= 0 && "must be topologically sorted");
    assert(static_cast<int>(state.size()) > index && "unexpected state length");
    assert(state[index] == nullptr && "already initialized state");

    if (number_data.size() != static_cast<size_t>(this->size())) {
        throw std::invalid_argument("Size of data provided does not match node size");
    }
    if (auto it = std::find_if_not(number_data.begin(), number_data.end(),
                                   [&](double value) { return is_valid(value); });
        it != number_data.end()) {
        throw std::invalid_argument("Invalid data provided for node");
    }

    std::vector<double> number_data_d;
    std::copy(number_data.begin(), number_data.end(), std::back_inserter(number_data_d));
    initialize_state(state, std::move(number_data_d));
}

/// A contiguous block of integer numbers.
class IntegerNode : public NumberNode {
 public:
    static constexpr int minimum_lower_bound = -2000000000;
    static constexpr int maximum_upper_bound = -minimum_lower_bound;
    static constexpr int default_lower_bound = 0;
    static constexpr int default_upper_bound = maximum_upper_bound;

    explicit IntegerNode(std::span<const ssize_t> shape,
                         std::optional<int> lower_bound = default_lower_bound,
                         std::optional<int> upper_bound = default_upper_bound)
            : NumberNode(shape, lower_bound.value_or(default_lower_bound),
                         upper_bound.value_or(default_upper_bound)) {
        if (lower_bound_ < minimum_lower_bound || upper_bound_ > maximum_upper_bound) {
            throw std::invalid_argument("Range provided for integers exceeds supported range");
        }
    }

    explicit IntegerNode(std::initializer_list<ssize_t> shape,
                         std::optional<int> lower_bound = default_lower_bound,
                         std::optional<int> upper_bound = default_upper_bound)
            : IntegerNode(std::span(shape.begin(), shape.end()), lower_bound, upper_bound) {}

    void default_move(State& state, RngAdaptor& rng) const override;

    // Overloads needed by the Node ABC **************************************

    bool integral() const override;

    // Overloads needed by the NumberNode ABC **************************************

    bool is_valid(double value) const override;
    double generate_value(RngAdaptor& rng) const override;
    double default_value() const override;
    std::unique_ptr<NodeStateData> new_data_ptr(std::vector<double>&& number_data) const override;

    // Specializations for the linear case
    bool set_value(State& state, ssize_t i, int value) const;
};

/// A contiguous block of binary numbers.
class BinaryNode : public IntegerNode {
 public:
    explicit BinaryNode(std::initializer_list<ssize_t> shape) : IntegerNode(shape, 0, 1) {}
    explicit BinaryNode(std::span<const ssize_t> shape) : IntegerNode(shape, 0, 1) {}

    void default_move(State& state, RngAdaptor& rng) const override;

    // Overloads needed by the NumberNode ABC **************************************

    std::unique_ptr<NodeStateData> new_data_ptr(std::vector<double>&& number_data) const override;

    // Specializations for the linear case
    void flip(State& state, ssize_t i) const;
    bool set(State& state, ssize_t i) const;
    bool unset(State& state, ssize_t i) const;
};

}  // namespace dwave::optimization

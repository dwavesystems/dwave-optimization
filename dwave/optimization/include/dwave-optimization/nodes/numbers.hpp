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
class NumberNode : public ArrayOutputMixin<ArrayNode>, public DecisionNode {
 public:
    NumberNode() = delete;

    // Overloads needed by the Array ABC **************************************

    double const* buff(const State&) const noexcept override;
    std::span<const Update> diff(const State& state) const noexcept override;

    void commit(State&) const noexcept override;
    void revert(State&) const noexcept override;

    /// @copydoc Array::minmax()
    std::pair<double, double> minmax(
            optional_cache_type<std::pair<double, double>> cache = std::nullopt) const override;

    double lower_bound() const;
    double upper_bound() const;

    // Overloads required by the Node ABC *************************************

    // Initialize the state. Defaults to 0 if 0 is in range, otherwise defaults
    // to lower bound.
    void initialize_state(State& state) const override;

    // Initialize a state from an existing container, taking ownership.
    void initialize_state(State& state, std::vector<double>&& number_data) const;

    // Initialize a state from an existing container, making a copy.
    template <std::ranges::range R>
    void initialize_state(State& state, const R& values) const {
        return initialize_state(state, std::vector<double>(values.begin(), values.end()));
    }

    // Initialize the state of the node randomly
    template <std::uniform_random_bit_generator Generator>
    void initialize_state(State& state, Generator& rng) const {
        std::vector<double> values;

        const ssize_t size = this->size();
        values.reserve(size);

        if (integral()) {
            std::uniform_int_distribution<ssize_t> gen(lower_bound_, upper_bound_);
            for (ssize_t i = 0; i < size; ++i) values.emplace_back(gen(rng));
        } else {
            std::uniform_real_distribution<double> gen(lower_bound_, upper_bound_);
            for (ssize_t i = 0; i < size; ++i) values.emplace_back(gen(rng));
        }

        return initialize_state(state, std::move(values));
    }

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
    virtual double default_value() const = 0;

    double lower_bound_;
    double upper_bound_;
};

/// A contiguous block of integer numbers.
class IntegerNode : public NumberNode {
 public:
    static constexpr int minimum_lower_bound = -2000000000;
    static constexpr int maximum_upper_bound = -minimum_lower_bound;
    static constexpr int default_lower_bound = 0;
    static constexpr int default_upper_bound = maximum_upper_bound;

    // Default to a single scalar integer with default bounds
    IntegerNode();

    // Create an integer array with the given bounds. Defaulting to the
    // specified default bounds.
    IntegerNode(std::span<const ssize_t> shape,
                std::optional<int> lower_bound = std::nullopt,   // inclusive
                std::optional<int> upper_bound = std::nullopt);  // inclusive
    IntegerNode(std::initializer_list<ssize_t> shape,
                std::optional<int> lower_bound = std::nullopt,   // inclusive
                std::optional<int> upper_bound = std::nullopt);  // inclusive
    IntegerNode(ssize_t size,
                std::optional<int> lower_bound = std::nullopt,   // inclusive
                std::optional<int> upper_bound = std::nullopt);  // inclusive

    // Overloads needed by the Node ABC **************************************

    bool integral() const override;

    // Overloads needed by the NumberNode ABC **************************************

    bool is_valid(double value) const override;
    double default_value() const override;

    // Specializations for the linear case
    bool set_value(State& state, ssize_t i, int value) const;
};

/// A contiguous block of binary numbers.
class BinaryNode : public IntegerNode {
 public:
    explicit BinaryNode(std::initializer_list<ssize_t> shape) : IntegerNode(shape, 0, 1) {}
    explicit BinaryNode(std::span<const ssize_t> shape) : IntegerNode(shape, 0, 1) {}

    // Specializations for the linear case
    void flip(State& state, ssize_t i) const;
    bool set(State& state, ssize_t i) const;
    bool unset(State& state, ssize_t i) const;
};

}  // namespace dwave::optimization

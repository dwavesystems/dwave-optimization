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

#include <optional>
#include <ranges>
#include <span>
#include <utility>
#include <variant>
#include <vector>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/graph.hpp"

namespace dwave::optimization {

/// A contiguous block of numbers.
class NumberNode : public ArrayOutputMixin<ArrayNode>, public DecisionNode {
 public:
    NumberNode() = delete;

    // Overloads needed by the Array ABC **************************************

    // @copydoc Array::buff()
    double const* buff(const State&) const noexcept override;

    // @copydoc Array::diff()
    std::span<const Update> diff(const State& state) const noexcept override;

    /// @copydoc Array::min()
    double min() const override;

    /// @copydoc Array::max()
    double max() const override;

    // Overloads required by the Node ABC *************************************

    // @copydoc Node::commit()
    void commit(State&) const noexcept override;

    // @copydoc Node::revert()
    void revert(State&) const noexcept override;

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
            for (ssize_t i = 0; i < size; ++i) {
                std::uniform_int_distribution<ssize_t> gen(lower_bound(i), upper_bound(i));
                values.emplace_back(gen(rng));
            }
        } else {
            for (ssize_t i = 0; i < size; ++i) {
                std::uniform_real_distribution<double> gen(lower_bound(i), upper_bound(i));
                values.emplace_back(gen(rng));
            }
        }
        return initialize_state(state, std::move(values));
    }

    // NumberNode methods *****************************************************

    // Specializations for the linear case
    bool exchange(State& state, ssize_t i, ssize_t j) const;

    // return the value of index i in a given state
    double get_value(State& state, ssize_t i) const;

    ssize_t linear_index(ssize_t x, ssize_t y) const;

    // Lower bounds of value in a given index
    virtual double lower_bound(ssize_t index) const { return min_; }

    // Upper bounds of value in a given index
    virtual double upper_bound(ssize_t index) const { return max_; }

    // clip value in a given state to fall within upper_bound and lower_bound
    // in a given index
    bool clip_and_set_value(State& state, ssize_t index, double value) const;

 protected:
    explicit NumberNode(std::span<const ssize_t> shape, double minimum, double maximum)
            : ArrayOutputMixin(shape), min_(minimum), max_(maximum) {
        if (max_ < min_) {
            throw std::invalid_argument("Invalid range for number array provided");
        }
    }

    // return truth statement: 'value is allowed in a given index'
    virtual bool is_valid(ssize_t index, double value) const = 0;

    // default value in a given index
    virtual double default_value(ssize_t index) const = 0;

    double min_;
    double max_;
};

/// A contiguous block of integer numbers.
class IntegerNode : public NumberNode {
 public:
    // nullopt_t represents default
    using bounds_t = std::variant<const std::vector<double>, double, std::nullopt_t>;
    // nullopt_t represents default, used to handle *integer format* inputs
    using ssize_bounds_t = std::variant<const std::vector<ssize_t>, ssize_t, std::nullopt_t>;

    static constexpr int minimum_lower_bound = -2000000000;
    static constexpr int maximum_upper_bound = -minimum_lower_bound;
    static constexpr int default_lower_bound = 0;
    static constexpr int default_upper_bound = maximum_upper_bound;

    // Default to a single scalar integer with default bounds
    IntegerNode() : IntegerNode({}) {}

    // Create an integer array with the user-defined *double format* bounds.
    // Defaulting to the specified default bounds.
    IntegerNode(std::span<const ssize_t> shape, bounds_t lower_bound = std::nullopt,
                bounds_t upper_bound = std::nullopt);
    IntegerNode(std::initializer_list<ssize_t> shape, bounds_t lower_bound = std::nullopt,
                bounds_t upper_bound = std::nullopt);
    IntegerNode(ssize_t size, bounds_t lower_bound = std::nullopt,
                bounds_t upper_bound = std::nullopt);
    // Create an integer array with the user-defined *integer format* bounds.
    // Defaulting to the specified default bounds.
    IntegerNode(std::span<const ssize_t> shape, ssize_bounds_t lower_bound,
                ssize_bounds_t upper_bound = std::nullopt);
    IntegerNode(std::initializer_list<ssize_t> shape, ssize_bounds_t lower_bound,
                ssize_bounds_t upper_bound = std::nullopt);
    IntegerNode(ssize_t size, ssize_bounds_t lower_bound,
                ssize_bounds_t upper_bound = std::nullopt);
    // Create an integer array for these edge cases.
    // Defaulting to the specified default bounds.
    IntegerNode(std::span<const ssize_t> shape, std::nullopt_t, std::nullopt_t = std::nullopt)
            : IntegerNode(shape) {};
    IntegerNode(std::initializer_list<ssize_t> shape, std::nullopt_t, std::nullopt_t = std::nullopt)
            : IntegerNode(std::span(shape)) {};
    IntegerNode(ssize_t size, std::nullopt_t, std::nullopt_t = std::nullopt)
            : IntegerNode({size}) {};

    // Overloads needed by the Node ABC ***************************************

    bool integral() const override;

    // Overloads needed by the NumberNode ABC *********************************

    double lower_bound(ssize_t index) const override;

    double upper_bound(ssize_t index) const override;

    bool is_valid(ssize_t index, double value) const override;

    double default_value(ssize_t index) const override;

    // Specializations for the linear case
    bool set_value(State& state, ssize_t index, double value) const;

 private:
    bounds_t full_lower_bound_;
    bounds_t full_upper_bound_;
};

/// A contiguous block of binary numbers.
class BinaryNode : public IntegerNode {
 public:
    /// A binary scalar variable with lower_bound = 0.0 and upper_bound = 1.0
    BinaryNode() : BinaryNode({}) {}
    // Create a binary array with the user-defined *double format* bounds.
    // Defaulting to lower_bound = 0.0 and upper_bound = 1.0
    BinaryNode(std::span<const ssize_t> shape, bounds_t lower_bound = std::nullopt,
               bounds_t upper_bound = std::nullopt);
    BinaryNode(std::initializer_list<ssize_t> shape, bounds_t lower_bound = std::nullopt,
               bounds_t upper_bound = std::nullopt);
    BinaryNode(ssize_t size, bounds_t lower_bound = std::nullopt,
               bounds_t upper_bound = std::nullopt);
    // Create a binary array with the given *integer format* bounds.
    // Defaulting to lower_bound = 0.0 and upper_bound = 1.0
    BinaryNode(std::span<const ssize_t> shape, ssize_bounds_t lower_bound,
               ssize_bounds_t upper_bound = std::nullopt);
    BinaryNode(std::initializer_list<ssize_t> shape, ssize_bounds_t lower_bound,
               ssize_bounds_t upper_bound = std::nullopt);
    BinaryNode(ssize_t size, ssize_bounds_t lower_bound, ssize_bounds_t upper_bound = std::nullopt);
    // Create a binary array for these edge cases.
    // Defaulting to lower_bound = 0.0 and upper_bound = 1.0
    BinaryNode(std::span<const ssize_t> shape, std::nullopt_t, std::nullopt_t = std::nullopt)
            : BinaryNode(shape) {};
    BinaryNode(std::initializer_list<ssize_t> shape, std::nullopt_t, std::nullopt_t = std::nullopt)
            : BinaryNode(std::span(shape)) {};
    BinaryNode(ssize_t size, std::nullopt_t, std::nullopt_t = std::nullopt) : BinaryNode({size}) {};

    // Specializations for the linear case
    void flip(State& state, ssize_t i) const;
    bool set(State& state, ssize_t i) const;
    bool unset(State& state, ssize_t i) const;
};

}  // namespace dwave::optimization

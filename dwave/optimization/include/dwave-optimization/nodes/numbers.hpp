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

    // In the given state, swap the value of index i with the value of index j.
    // Returns `true` if the values at indices i and j change and `false`
    // otherwise.
    bool exchange(State& state, ssize_t i, ssize_t j) const;

    // Return the value of index i in a given state.
    double get_value(State& state, ssize_t i) const;

    // Lower bounds of value in a given index.
    virtual double lower_bound(ssize_t index) const { return min_; }
    virtual double lower_bound() const { return min_; }

    // Upper bounds of value in a given index.
    virtual double upper_bound(ssize_t index) const { return max_; }
    virtual double upper_bound() const { return max_; }

    // Clip value in a given state to fall within upper_bound and lower_bound
    // in a given index.
    bool clip_and_set_value(State& state, ssize_t index, double value) const;

 protected:
    explicit NumberNode(std::span<const ssize_t> shape, double minimum, double maximum)
            : ArrayOutputMixin(shape), min_(minimum), max_(maximum) {
        if (max_ < min_) {
            throw std::invalid_argument("Invalid range for number array provided");
        }
    }

    // Return truth statement: 'value is within the bounds of a given index'
    virtual bool is_valid(ssize_t index, double value) const = 0;

    // Default value in a given index
    virtual double default_value(ssize_t index) const = 0;

    double min_;
    double max_;
};

/// A contiguous block of integer numbers.
class IntegerNode : public NumberNode {
 public:
    static constexpr int minimum_lower_bound = -2000000000;
    static constexpr int maximum_upper_bound = -minimum_lower_bound;
    static constexpr int default_lower_bound = 0;
    static constexpr int default_upper_bound = maximum_upper_bound;

    // Default to a single scalar integer with default bounds
    IntegerNode() : IntegerNode({}) {}

    // Create an integer array with the user-defined bounds.
    // Defaulting to the specified default bounds.
    IntegerNode(std::span<const ssize_t> shape,
                std::optional<std::vector<double>> lower_bound = std::nullopt,
                std::optional<std::vector<double>> upper_bound = std::nullopt);
    IntegerNode(std::initializer_list<ssize_t> shape,
                std::optional<std::vector<double>> lower_bound = std::nullopt,
                std::optional<std::vector<double>> upper_bound = std::nullopt);
    IntegerNode(ssize_t size, std::optional<std::vector<double>> lower_bound = std::nullopt,
                std::optional<std::vector<double>> upper_bound = std::nullopt);

    IntegerNode(std::span<const ssize_t> shape, double lower_bound,
                std::optional<std::vector<double>> upper_bound = std::nullopt);
    IntegerNode(std::initializer_list<ssize_t> shape, double lower_bound,
                std::optional<std::vector<double>> upper_bound = std::nullopt);
    IntegerNode(ssize_t size, double lower_bound,
                std::optional<std::vector<double>> upper_bound = std::nullopt);

    IntegerNode(std::span<const ssize_t> shape, std::optional<std::vector<double>> lower_bound,
                double upper_bound);
    IntegerNode(std::initializer_list<ssize_t> shape,
                std::optional<std::vector<double>> lower_bound, double upper_bound);
    IntegerNode(ssize_t size, std::optional<std::vector<double>> lower_bound, double upper_bound);

    IntegerNode(std::span<const ssize_t> shape, double lower_bound, double upper_bound);
    IntegerNode(std::initializer_list<ssize_t> shape, double lower_bound, double upper_bound);
    IntegerNode(ssize_t size, double lower_bound, double upper_bound);

    // Overloads needed by the Node ABC ***************************************

    // @copydoc Node::integral()
    bool integral() const override;

    // Overloads needed by the NumberNode ABC *********************************

    // @copydoc NumberNode::lower_bound(). Depending upon user input, may
    // return non-integral values
    double lower_bound(ssize_t index) const override;
    double lower_bound() const override;

    // @copydoc NumberNode::upper_bound(). Depending upon user input, may
    // return non-integral values
    double upper_bound(ssize_t index) const override;
    double upper_bound() const override;

    // @copydoc NumberNode::is_valid()
    bool is_valid(ssize_t index, double value) const override;

    // IntegerNode methods ****************************************************

    // Set the value at the given index in the given state. Returns `true` if
    // the value at the index changed and `false` otherwise.
    bool set_value(State& state, ssize_t index, double value) const;

 protected:
    // Overloads needed by the Node ABC ***************************************

    // @copydoc NumberNode::default_value()
    double default_value(ssize_t index) const override;

 private:
    std::vector<double> full_lower_bound_;
    std::vector<double> full_upper_bound_;
};

/// A contiguous block of binary numbers.
class BinaryNode : public IntegerNode {
 public:
    /// A binary scalar variable with lower_bound = 0.0 and upper_bound = 1.0
    BinaryNode() : BinaryNode({}) {}

    // Create a binary array with the user-defined bounds.
    // Defaulting to lower_bound = 0.0 and upper_bound = 1.0
    BinaryNode(std::span<const ssize_t> shape,
               std::optional<std::vector<double>> lower_bound = std::nullopt,
               std::optional<std::vector<double>> upper_bound = std::nullopt);
    BinaryNode(std::initializer_list<ssize_t> shape,
               std::optional<std::vector<double>> lower_bound = std::nullopt,
               std::optional<std::vector<double>> upper_bound = std::nullopt);
    BinaryNode(ssize_t size, std::optional<std::vector<double>> lower_bound = std::nullopt,
               std::optional<std::vector<double>> upper_bound = std::nullopt);

    BinaryNode(std::span<const ssize_t> shape, double lower_bound,
               std::optional<std::vector<double>> upper_bound = std::nullopt);
    BinaryNode(std::initializer_list<ssize_t> shape, double lower_bound,
               std::optional<std::vector<double>> upper_bound = std::nullopt);
    BinaryNode(ssize_t size, double lower_bound,
               std::optional<std::vector<double>> upper_bound = std::nullopt);

    BinaryNode(std::span<const ssize_t> shape, std::optional<std::vector<double>> lower_bound,
               double upper_bound);
    BinaryNode(std::initializer_list<ssize_t> shape, std::optional<std::vector<double>> lower_bound,
               double upper_bound);
    BinaryNode(ssize_t size, std::optional<std::vector<double>> lower_bound, double upper_bound);

    BinaryNode(std::span<const ssize_t> shape, double lower_bound, double upper_bound);
    BinaryNode(std::initializer_list<ssize_t> shape, double lower_bound, double upper_bound);
    BinaryNode(ssize_t size, double lower_bound, double upper_bound);

    // Flip the value (0 -> 1 or 1 -> 0) at index i in the given state. Returns
    // `true` if the value at index i changed and `false` otherwise.
    bool flip(State& state, ssize_t i) const;

    // Set the value at index i to `true` in the given state. Returns `true` if
    // the value at index i changed and `false` otherwise.
    bool set(State& state, ssize_t i) const;

    // Set the at index i to `false` in the given state. Returns `true` if
    // the value at index i changed and `false` otherwise.
    bool unset(State& state, ssize_t i) const;
};

}  // namespace dwave::optimization

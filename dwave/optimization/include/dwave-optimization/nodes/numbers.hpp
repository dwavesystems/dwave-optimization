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

/// Allowable axis-wise bound operators.
enum BoundAxisOperator { Equal, LessEqual, GreaterEqual };

/// Class for stateless axis-wise bound information. Given an `axis`, define
/// constraints on the sum of the values in each slice along `axis`.
/// Constraints can be defined for ALL slices along `axis` or PER slice along
/// `axis`. Allowable operators are defined by `BoundAxisOperator`.
class BoundAxisInfo {
 public:
    /// To reduce the # of `IntegerNode` and `BinaryNode` constructors, we
    /// allow only one constructor.
    BoundAxisInfo(ssize_t axis, std::vector<BoundAxisOperator> axis_operators,
                  std::vector<double> axis_bounds);
    /// The bound axis
    const ssize_t axis;
    /// Operator for ALL axis slices (vector has length one) or operator*s* PER
    /// slice (length of vector is equal to the number of slices).
    const std::vector<BoundAxisOperator> operators;
    /// Bound for ALL axis slices (vector has length one) or bound*s* PER slice
    /// (length of vector is equal to the number of slices).
    const std::vector<double> bounds;

    /// Obtain the bound associated with a given slice along bound axis.
    double get_bound(const ssize_t slice) const;

    /// Obtain the operator associated with a given slice along bound axis.
    BoundAxisOperator get_operator(const ssize_t slice) const;
};

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
        if (bound_axes_info_.size() > 0) {
            throw std::invalid_argument("Cannot randomly initialize_state with bound axes");
        }

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
    void exchange(State& state, ssize_t i, ssize_t j) const;

    // Return the value of index i in a given state.
    double get_value(State& state, ssize_t i) const;

    // Lower bound of value in a given index.
    double lower_bound(ssize_t index) const;
    double lower_bound() const;

    // Upper bound of value in a given index.
    double upper_bound(ssize_t index) const;
    double upper_bound() const;

    // Clip value in a given state to fall within upper_bound and lower_bound
    // in a given index.
    void clip_and_set_value(State& state, ssize_t index, double value) const;

    /// Return pointer to the vector of axis-wise bounds
    const std::vector<BoundAxisInfo>& axis_wise_bounds() const;

    // Return a pointer to the vector containing the bound axis sums
    const std::vector<std::vector<double>>& bound_axis_sums(State& state) const;

 protected:
    explicit NumberNode(std::span<const ssize_t> shape, std::vector<double> lower_bound,
                        std::vector<double> upper_bound,
                        std::optional<std::vector<BoundAxisInfo>> bound_axes = std::nullopt);

    /// Return truth statement: 'value is valid in a given index'.
    virtual bool is_valid(ssize_t index, double value) const = 0;

    /// Default value in a given index.
    virtual double default_value(ssize_t index) const = 0;

    /// Update the running bound axis sums where `index` is changed by
    /// `value_change` in a given state.
    void update_bound_axis_slice_sums(State& state, const ssize_t index,
                                      const double value_change) const;

    /// Statelss global minimum and maximum of the values stored in NumberNode.
    double min_;
    double max_;

    /// Stateless index-wise upper and lower bounds.
    std::vector<double> lower_bounds_;
    std::vector<double> upper_bounds_;

    /// Stateless information on each bound axis.
    const std::vector<BoundAxisInfo> bound_axes_info_;
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

    // Create an integer array with the user-defined index- and axis-wise bounds.
    // Index-wise bounds default to the specified default bounds.
    IntegerNode(std::span<const ssize_t> shape,
                std::optional<std::vector<double>> lower_bound = std::nullopt,
                std::optional<std::vector<double>> upper_bound = std::nullopt,
                std::optional<std::vector<BoundAxisInfo>> bound_axes = std::nullopt);
    IntegerNode(std::initializer_list<ssize_t> shape,
                std::optional<std::vector<double>> lower_bound = std::nullopt,
                std::optional<std::vector<double>> upper_bound = std::nullopt,
                std::optional<std::vector<BoundAxisInfo>> bound_axes = std::nullopt);
    IntegerNode(ssize_t size, std::optional<std::vector<double>> lower_bound = std::nullopt,
                std::optional<std::vector<double>> upper_bound = std::nullopt,
                std::optional<std::vector<BoundAxisInfo>> bound_axes = std::nullopt);

    IntegerNode(std::span<const ssize_t> shape, double lower_bound,
                std::optional<std::vector<double>> upper_bound = std::nullopt,
                std::optional<std::vector<BoundAxisInfo>> bound_axes = std::nullopt);
    IntegerNode(std::initializer_list<ssize_t> shape, double lower_bound,
                std::optional<std::vector<double>> upper_bound = std::nullopt,
                std::optional<std::vector<BoundAxisInfo>> bound_axes = std::nullopt);
    IntegerNode(ssize_t size, double lower_bound,
                std::optional<std::vector<double>> upper_bound = std::nullopt,
                std::optional<std::vector<BoundAxisInfo>> bound_axes = std::nullopt);

    IntegerNode(std::span<const ssize_t> shape, std::optional<std::vector<double>> lower_bound,
                double upper_bound,
                std::optional<std::vector<BoundAxisInfo>> bound_axes = std::nullopt);
    IntegerNode(std::initializer_list<ssize_t> shape,
                std::optional<std::vector<double>> lower_bound, double upper_bound,
                std::optional<std::vector<BoundAxisInfo>> bound_axes = std::nullopt);
    IntegerNode(ssize_t size, std::optional<std::vector<double>> lower_bound, double upper_bound,
                std::optional<std::vector<BoundAxisInfo>> bound_axes = std::nullopt);

    IntegerNode(std::span<const ssize_t> shape, double lower_bound, double upper_bound,
                std::optional<std::vector<BoundAxisInfo>> bound_axes = std::nullopt);
    IntegerNode(std::initializer_list<ssize_t> shape, double lower_bound, double upper_bound,
                std::optional<std::vector<BoundAxisInfo>> bound_axes = std::nullopt);
    IntegerNode(ssize_t size, double lower_bound, double upper_bound,
                std::optional<std::vector<BoundAxisInfo>> bound_axes = std::nullopt);

    // Overloads needed by the Node ABC ***************************************

    // @copydoc Node::integral()
    bool integral() const override;

    // Overloads needed by the NumberNode ABC *********************************

    // @copydoc NumberNode::is_valid()
    bool is_valid(ssize_t index, double value) const override;

    // IntegerNode methods ****************************************************

    // Set the value at the given index in the given state.
    void set_value(State& state, ssize_t index, double value) const;

 protected:
    // Overloads needed by the Node ABC ***************************************

    // @copydoc NumberNode::default_value()
    double default_value(ssize_t index) const override;
};

/// A contiguous block of binary numbers.
class BinaryNode : public IntegerNode {
 public:
    /// A binary scalar variable with lower_bound = 0.0 and upper_bound = 1.0
    BinaryNode() : BinaryNode({}) {}

    // Create a binary array with the user-defined index- and axis-wise bounds.
    // Index-wise bounds default to lower_bound = 0.0 and upper_bound = 1.0.
    BinaryNode(std::span<const ssize_t> shape,
               std::optional<std::vector<double>> lower_bound = std::nullopt,
               std::optional<std::vector<double>> upper_bound = std::nullopt,
               std::optional<std::vector<BoundAxisInfo>> bound_axes = std::nullopt);
    BinaryNode(std::initializer_list<ssize_t> shape,
               std::optional<std::vector<double>> lower_bound = std::nullopt,
               std::optional<std::vector<double>> upper_bound = std::nullopt,
               std::optional<std::vector<BoundAxisInfo>> bound_axes = std::nullopt);
    BinaryNode(ssize_t size, std::optional<std::vector<double>> lower_bound = std::nullopt,
               std::optional<std::vector<double>> upper_bound = std::nullopt,
               std::optional<std::vector<BoundAxisInfo>> bound_axes = std::nullopt);

    BinaryNode(std::span<const ssize_t> shape, double lower_bound,
               std::optional<std::vector<double>> upper_bound = std::nullopt,
               std::optional<std::vector<BoundAxisInfo>> bound_axes = std::nullopt);
    BinaryNode(std::initializer_list<ssize_t> shape, double lower_bound,
               std::optional<std::vector<double>> upper_bound = std::nullopt,
               std::optional<std::vector<BoundAxisInfo>> bound_axes = std::nullopt);
    BinaryNode(ssize_t size, double lower_bound,
               std::optional<std::vector<double>> upper_bound = std::nullopt,
               std::optional<std::vector<BoundAxisInfo>> bound_axes = std::nullopt);

    BinaryNode(std::span<const ssize_t> shape, std::optional<std::vector<double>> lower_bound,
               double upper_bound,
               std::optional<std::vector<BoundAxisInfo>> bound_axes = std::nullopt);
    BinaryNode(std::initializer_list<ssize_t> shape, std::optional<std::vector<double>> lower_bound,
               double upper_bound,
               std::optional<std::vector<BoundAxisInfo>> bound_axes = std::nullopt);
    BinaryNode(ssize_t size, std::optional<std::vector<double>> lower_bound, double upper_bound,
               std::optional<std::vector<BoundAxisInfo>> bound_axes = std::nullopt);

    BinaryNode(std::span<const ssize_t> shape, double lower_bound, double upper_bound,
               std::optional<std::vector<BoundAxisInfo>> bound_axes = std::nullopt);
    BinaryNode(std::initializer_list<ssize_t> shape, double lower_bound, double upper_bound,
               std::optional<std::vector<BoundAxisInfo>> bound_axes = std::nullopt);
    BinaryNode(ssize_t size, double lower_bound, double upper_bound,
               std::optional<std::vector<BoundAxisInfo>> bound_axes = std::nullopt);

    // Flip the value (0 -> 1 or 1 -> 0) at index i in the given state.
    void flip(State& state, ssize_t i) const;

    // Set the value at index i to `true` in the given state.
    void set(State& state, ssize_t i) const;

    // Set the at index i to `false` in the given state.
    void unset(State& state, ssize_t i) const;
};

}  // namespace dwave::optimization

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
#include <stdexcept>
#include <utility>
#include <vector>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/graph.hpp"

namespace dwave::optimization {

/// A contiguous block of numbers.
class NumberNode : public ArrayOutputMixin<ArrayNode>, public DecisionNode {
 public:
    /// Stateless sum constraint information.
    ///
    /// A sum constraint constrains the sum of values within slices of the array.
    /// The slices are defined along `axis` when `axis` has a value. If
    /// `axis == std::nullopt`, the constraint is applied to the entire array,
    /// which is treated as a flat array with a single slice.
    ///
    /// Constraints may be defined either:
    /// - for ALL slices (the `operators` and `bounds` vectors have length 1), or
    /// - PER slice (their lengths equal the number of slices along `axis`).
    ///
    /// Each slice sum is constrained by an `Operator` and a corresponding `bound`.
    struct SumConstraint {
     public:
        /// Allowable operators.
        enum class Operator { Equal, LessEqual, GreaterEqual };

        /// To reduce the # of `IntegerNode` and `BinaryNode` constructors, we
        /// allow only one constructor.
        SumConstraint(std::optional<ssize_t> axis, std::vector<Operator> operators,
                      std::vector<double> bounds);

        /// Return the axis along which slices are defined.
        /// If `std::nullopt`, the sum constraint applies to the entire array.
        std::optional<ssize_t> axis() const { return axis_; };

        /// Obtain the bound associated with a given slice.
        double bound(const ssize_t slice) const;

        /// Obtain the operator associated with a given slice.
        Operator op(const ssize_t slice) const;

        /// The number of bounds.
        ssize_t num_bounds() const { return bounds_.size(); };

        /// The number of operators.
        ssize_t num_operators() const { return operators_.size(); };

     private:
        /// Axis along which slices are defined (`std::nullopt` = whole array).
        std::optional<ssize_t> axis_ = std::nullopt;
        /// Operator for ALL slices (vector has length one) or operators PER
        /// slice (length of vector is equal to the number of slices).
        std::vector<Operator> operators_;
        /// Bound for ALL slices (vector has length one) or bounds PER slice
        /// (length of vector is equal to the number of slices).
        std::vector<double> bounds_;
    };

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
        // Currently do not support random node initialization with sum constraints.
        if (sum_constraints_.size() > 0) {
            throw std::invalid_argument("Cannot randomly initialize_state with sum constraints.");
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

    /// @copydoc Node::propagate()
    void propagate(State& state) const override;

    // NumberNode methods *****************************************************

    // In the given state, swap the value of index i with the value of index j.
    // Users may pass the slices (per sum constraint) that each index lies on.
    void exchange(State& state, ssize_t i, ssize_t j,
                  std::optional<std::vector<ssize_t>> i_slices = std::nullopt,
                  std::optional<std::vector<ssize_t>> j_slices = std::nullopt) const;

    // Return the value of index i in a given state.
    double get_value(State& state, ssize_t i) const;

    // Lower bound of value in a given index.
    double lower_bound(ssize_t index) const;
    double lower_bound() const;

    // Upper bound of value in a given index.
    double upper_bound(ssize_t index) const;
    double upper_bound() const;

    // Clip value in a given state to fall within upper_bound and lower_bound
    // in a given index. Users may pass the slices (per sum constraint) that
    // each index lies on.
    void clip_and_set_value(State& state, ssize_t index, double value,
                            std::optional<std::vector<ssize_t>> slices = std::nullopt) const;

    /// Return the stateless sum constraints.
    const std::vector<SumConstraint>& sum_constraints() const;

    /// If the node is subject to sum constraints, we track the state
    /// dependent sum of the values within each slice per constraint. The
    /// returned vector is indexed in the same ordering as the constraints
    /// given by `sum_constraints()`.
    const std::vector<std::vector<double>>& sum_constraints_lhs(const State& state) const;

 protected:
    explicit NumberNode(std::span<const ssize_t> shape, std::vector<double> lower_bound,
                        std::vector<double> upper_bound,
                        std::vector<SumConstraint> sum_constraints = {});

    // Return truth statement: 'value is valid in a given index'.
    virtual bool is_valid(ssize_t index, double value) const = 0;

    // Default value in a given index.
    virtual double default_value(ssize_t index) const = 0;

    /// Statelss global minimum and maximum of the values stored in NumberNode.
    double min_;
    double max_;

    /// Stateless index-wise upper and lower bounds.
    std::vector<double> lower_bounds_;
    std::vector<double> upper_bounds_;

    /// Stateless sum constraints.
    std::vector<SumConstraint> sum_constraints_;
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

    // Create an integer array with the user-defined index-wise bounds and sum
    // constraints. Index-wise bounds default to the specified default bounds.
    // By default, there are no sum constraints.
    IntegerNode(std::span<const ssize_t> shape,
                std::optional<std::vector<double>> lower_bound = std::nullopt,
                std::optional<std::vector<double>> upper_bound = std::nullopt,
                std::vector<SumConstraint> sum_constraints = {});
    IntegerNode(std::initializer_list<ssize_t> shape,
                std::optional<std::vector<double>> lower_bound = std::nullopt,
                std::optional<std::vector<double>> upper_bound = std::nullopt,
                std::vector<SumConstraint> sum_constraints = {});
    IntegerNode(ssize_t size, std::optional<std::vector<double>> lower_bound = std::nullopt,
                std::optional<std::vector<double>> upper_bound = std::nullopt,
                std::vector<SumConstraint> sum_constraints = {});

    IntegerNode(std::span<const ssize_t> shape, double lower_bound,
                std::optional<std::vector<double>> upper_bound = std::nullopt,
                std::vector<SumConstraint> sum_constraints = {});
    IntegerNode(std::initializer_list<ssize_t> shape, double lower_bound,
                std::optional<std::vector<double>> upper_bound = std::nullopt,
                std::vector<SumConstraint> sum_constraints = {});
    IntegerNode(ssize_t size, double lower_bound,
                std::optional<std::vector<double>> upper_bound = std::nullopt,
                std::vector<SumConstraint> sum_constraints = {});

    IntegerNode(std::span<const ssize_t> shape, std::optional<std::vector<double>> lower_bound,
                double upper_bound, std::vector<SumConstraint> sum_constraints = {});
    IntegerNode(std::initializer_list<ssize_t> shape,
                std::optional<std::vector<double>> lower_bound, double upper_bound,
                std::vector<SumConstraint> sum_constraints = {});
    IntegerNode(ssize_t size, std::optional<std::vector<double>> lower_bound, double upper_bound,
                std::vector<SumConstraint> sum_constraints = {});

    IntegerNode(std::span<const ssize_t> shape, double lower_bound, double upper_bound,
                std::vector<SumConstraint> sum_constraints = {});
    IntegerNode(std::initializer_list<ssize_t> shape, double lower_bound, double upper_bound,
                std::vector<SumConstraint> sum_constraints = {});
    IntegerNode(ssize_t size, double lower_bound, double upper_bound,
                std::vector<SumConstraint> sum_constraints = {});

    // Overloads needed by the Node ABC ***************************************

    // @copydoc Node::integral()
    bool integral() const override;

    // Overloads needed by the NumberNode ABC *********************************

    // @copydoc NumberNode::is_valid()
    bool is_valid(ssize_t index, double value) const override;

    // IntegerNode methods ****************************************************

    // Set the value at the given index in the given state.
    // Users may pass the slices (per sum constraint) that each index lies on.
    void set_value(State& state, ssize_t index, double value,
                   std::optional<std::vector<ssize_t>> slices = std::nullopt) const;

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

    // Create a binary array with the user-defined index-wise bounds and sum
    // constraints. Index-wise bounds default to lower_bound = 0.0 and
    // upper_bound = 1.0. By default, there are no sum constraints.
    BinaryNode(std::span<const ssize_t> shape,
               std::optional<std::vector<double>> lower_bound = std::nullopt,
               std::optional<std::vector<double>> upper_bound = std::nullopt,
               std::vector<SumConstraint> sum_constraints = {});
    BinaryNode(std::initializer_list<ssize_t> shape,
               std::optional<std::vector<double>> lower_bound = std::nullopt,
               std::optional<std::vector<double>> upper_bound = std::nullopt,
               std::vector<SumConstraint> sum_constraints = {});
    BinaryNode(ssize_t size, std::optional<std::vector<double>> lower_bound = std::nullopt,
               std::optional<std::vector<double>> upper_bound = std::nullopt,
               std::vector<SumConstraint> sum_constraints = {});

    BinaryNode(std::span<const ssize_t> shape, double lower_bound,
               std::optional<std::vector<double>> upper_bound = std::nullopt,
               std::vector<SumConstraint> sum_constraints = {});
    BinaryNode(std::initializer_list<ssize_t> shape, double lower_bound,
               std::optional<std::vector<double>> upper_bound = std::nullopt,
               std::vector<SumConstraint> sum_constraints = {});
    BinaryNode(ssize_t size, double lower_bound,
               std::optional<std::vector<double>> upper_bound = std::nullopt,
               std::vector<SumConstraint> sum_constraints = {});

    BinaryNode(std::span<const ssize_t> shape, std::optional<std::vector<double>> lower_bound,
               double upper_bound, std::vector<SumConstraint> sum_constraints = {});
    BinaryNode(std::initializer_list<ssize_t> shape, std::optional<std::vector<double>> lower_bound,
               double upper_bound, std::vector<SumConstraint> sum_constraints = {});
    BinaryNode(ssize_t size, std::optional<std::vector<double>> lower_bound, double upper_bound,
               std::vector<SumConstraint> sum_constraints = {});

    BinaryNode(std::span<const ssize_t> shape, double lower_bound, double upper_bound,
               std::vector<SumConstraint> sum_constraints = {});
    BinaryNode(std::initializer_list<ssize_t> shape, double lower_bound, double upper_bound,
               std::vector<SumConstraint> sum_constraints = {});
    BinaryNode(ssize_t size, double lower_bound, double upper_bound,
               std::vector<SumConstraint> sum_constraints = {});

    /// ** Redefined NumberNode methods since BinaryNode has custom StateData **

    /// @copydoc Node::revert()
    void revert(State&) const noexcept override;

    /// Initialize the state. Defaults to 0 if 0 is in range, otherwise defaults
    /// to lower bound.
    void initialize_state(State& state) const override;

    /// Initialize a state from an existing container, taking ownership.
    void initialize_state(State& state, std::vector<double>&& number_data) const;

    /// Initialize a state from an existing container, making a copy.
    template <std::ranges::range R>
    void initialize_state(State& state, const R& values) const {
        return initialize_state(state, std::vector<double>(values.begin(), values.end()));
    }

    /// Initialize the state of the binary node randomly. Special handling if
    /// the node is subject to a sum constraint. In this case, for each slice
    /// (w.r.t the sum constraint) pick a random feasible sum that respects the
    /// sum constraint and perform a random assignment of buffer values within
    /// the slice such that the sum said values equals the chosen sum. Note:
    /// This method does not uniformly sample from the possible assignments
    /// that satisfy the sum constraint.
    template <std::uniform_random_bit_generator Generator>
    void initialize_state(State& state, Generator& rng) const {
        std::vector<double> values;
        const ssize_t size = this->size();

        if (sum_constraints_.size() == 0) {
            values.reserve(size);
            for (ssize_t i = 0; i < size; ++i) {
                std::uniform_int_distribution<ssize_t> gen(lower_bound(i), upper_bound(i));
                values.emplace_back(gen(rng));
            }
            return initialize_state(state, std::move(values));
        }

        if (sum_constraints().size() != 1)
            throw std::invalid_argument("Multiple sum constraints not yet supported.");
        // We need to construct a random initial state that adheres to the sum constraint.
        const std::span<const ssize_t> shape = this->shape();
        const ssize_t ndim = shape.size();
        const NumberNode::SumConstraint& constraint = sum_constraints().front();
        const std::optional<const ssize_t> axis = constraint.axis();
        // If the sum constraint is over entire array, the array is treated as a flat array with
        // one slice, otherwise, the # of slices is given by the size of the constrained axis.
        const ssize_t num_slices = axis.has_value() ? shape[*axis] : 1;
        const ssize_t slice_size = size / num_slices;
        values.resize(size);

        /// Reorder values of `span` such that the given `axis` is moved to the 0th index.
        /// If `axis` is std::nullopt, this method simply copies the span data as-is.
        auto shift_axis_data = [&ndim, &axis](const std::span<const ssize_t> span) {
            const ssize_t axis_val = axis.has_value() ? *axis : 0;
            std::vector<ssize_t> output;
            output.reserve(ndim);
            output.emplace_back(span[axis_val]);
            for (ssize_t i = 0; i < ndim; ++i) {
                if (i != axis_val) output.emplace_back(span[i]);
            }
            return output;
        };

        /// Reverse the operation defined by `shift_axis_data()`.
        auto undo_shift_axis_data = [&ndim, &axis](const std::span<const ssize_t> span) {
            const ssize_t axis_val = axis.has_value() ? *axis : 0;
            std::vector<ssize_t> output;
            output.reserve(ndim);
            ssize_t i_span = 1;
            for (ssize_t i = 0; i < ndim; ++i) {
                output.emplace_back(i == axis_val ? span[0] : span[i_span++]);
            }
            return output;
        };

        /// Random ssize_t in the interval [start, end].
        auto rand_ssize_t_in_range = [&rng](const ssize_t start, const ssize_t end) {
            assert(start <= end);
            std::uniform_int_distribution<ssize_t> dist(start, end);
            return dist(rng);
        };

        /// Given a `slice` w.r.t a sum constraint, the current sum `slice_sum`
        /// of the buffer values in the slice, and the maximum # of indices that
        /// can be flipped from 0 -> 1, pick a random number of flips to make
        /// that obey the constraint.
        auto rnd_num_flips = [&constraint, &rand_ssize_t_in_range](const ssize_t slice,
                                                                   const double slice_sum,
                                                                   const ssize_t max_flip) {
            // Difference between sum constraint bound and the current slice sum.
            const ssize_t bound_diff = constraint.bound(slice) - slice_sum;
            switch (constraint.op(slice)) {
                case NumberNode::SumConstraint::Operator::Equal:
                    // Minimum assignment to values in slice is greater than bound.
                    if (bound_diff < 0) throw std::invalid_argument("Infeasible sum constraint.");
                    // We must perform exactly `bound_diff` # of flips.
                    return bound_diff;
                case NumberNode::SumConstraint::Operator::LessEqual:
                    // Minimum assignment to values in slice is greater than bound.
                    if (bound_diff < 0) throw std::invalid_argument("Infeasible sum constraint.");
                    // We can perform at most min(bound_diff, max_flip) # of flips.
                    return rand_ssize_t_in_range(0, std::min<ssize_t>(bound_diff, max_flip));
                case NumberNode::SumConstraint::Operator::GreaterEqual:
                    // Maximum assignment to values in slice is smaller than bound.
                    if (max_flip < bound_diff)
                        throw std::invalid_argument("Infeasible sum constraint.");
                    // We must perform at least max(0, bound_diff) # of flips.
                    return rand_ssize_t_in_range(std::max<ssize_t>(0, bound_diff), max_flip);
                default:
                    assert(false && "Unexpected operator type.");
                    unreachable();
            }
        };
        // We need a way to iterate over each slice defined by the sum constraint.
        // If the sum constraint is along an axis, we adjust the shape and
        // strides such that the data for the constrained axis is moved to index 0.
        const std::vector<ssize_t> buff_shape = shift_axis_data(shape);
        const std::vector<ssize_t> buff_strides = shift_axis_data(strides());
        const BufferIterator<double, double, false> slice_0(values.data(), ndim, buff_shape.data(),
                                                            buff_strides.data());
        std::vector<ssize_t> cached_indices;  // Indices that can be flipped from 0 -> 1 in slice.
        cached_indices.reserve(slice_size);   // Size is bound by slice size.
        for (ssize_t slice = 0; slice < num_slices; ++slice) {  // Iterate over slices.
            ssize_t slice_sum = 0;  // The sum of the buffer values in the slice.
            // Iterate over all indices in the given slice, assign lower bound
            // to buffer value, increment `slice_sum` by buffer value, and
            // cache the index if the index can be flipped from a 0 -> 1.
            for (auto slice_it = slice_0 + slice * slice_size, slice_end = slice_it + slice_size;
                 slice_it != slice_end; ++slice_it) {
                // Determine the "true" index of `slice_it` given the node shape.
                const ssize_t index =
                        ravel_multi_index(undo_shift_axis_data(slice_it.location()), shape);
                *slice_it = lower_bound(index);  // Assign buffer value to lower bound.
                slice_sum += *slice_it;          // Increment slice sum by buffer value.
                // If buffer value is assigned 0 and upper bound is 1, cache index.
                if (*slice_it != upper_bound(index)) cached_indices.push_back(index);
            }

            const ssize_t cache_size = cached_indices.size();
            // Randomly flip indices in the slice from 0 to 1.
            for (ssize_t i = 0, stop = rnd_num_flips(slice, slice_sum, cache_size); i < stop; ++i) {
                // Pick a random index not yet flipped from cached_indices[i, end)
                const ssize_t j = rand_ssize_t_in_range(i, cache_size - 1);
                // Move value at index j to position i to prevent it being sampled again.
                std::swap(cached_indices[i], cached_indices[j]);
                const ssize_t index = cached_indices[i];
                assert(values[index] == 0.0 && upper_bound(index) == 1.0);
                values[index] = 1.0;
            }
            cached_indices.clear();
        }
        return initialize_state(state, std::move(values));
    }

    /// @copydoc NumberNode::exchange()
    void exchange(State& state, ssize_t i, ssize_t j,
                  std::optional<std::vector<ssize_t>> i_slices = std::nullopt,
                  std::optional<std::vector<ssize_t>> j_slices = std::nullopt) const;

    /// @copydoc NumberNode::clip_and_set_value()
    void clip_and_set_value(State& state, ssize_t index, double value,
                            std::optional<std::vector<ssize_t>> slices = std::nullopt) const;

    /// ** Redefined IntegerNode method since BinaryNode has custom StateData **

    /// @copydoc IntegerNode::set_value()
    void set_value(State& state, ssize_t index, double value,
                   std::optional<std::vector<ssize_t>> slices = std::nullopt) const;

    /// ************************** BinaryNode methods **************************

    // Flip the value (0 -> 1 or 1 -> 0) at `index` in the given state.
    // Users may pass the slices (per sum constraint) that `index` lies on.
    void flip(State& state, ssize_t index,
              std::optional<std::vector<ssize_t>> slices = std::nullopt) const;

    /// Given a state and slice (w.r.t a sum constraint), return the number of
    /// buffer values in the slice equal to 1.
    ssize_t num_true(const State& state, const ssize_t sum_constraint, const ssize_t slice) const;

    /// Given a state and slice (w.r.t a sum constraint), return the number of
    /// buffer values in the slice equal to 0.
    ssize_t num_false(const State& state, const ssize_t sum_constraint, const ssize_t slice) const;

    /// Given a state and slice (w.r.t a sum constraint), return the `i`th
    /// index in in the slice whose buffer value is 1.
    ssize_t ith_true_index(const State& state, const ssize_t sum_constraint, const ssize_t slice,
                           const ssize_t i) const;

    /// Given a state and slice (w.r.t a sum constraint), return the `i`th
    /// index in in the slice whose buffer value is 0.
    ssize_t ith_false_index(const State& state, const ssize_t sum_constraint, const ssize_t slice,
                            const ssize_t i) const;
};

}  // namespace dwave::optimization

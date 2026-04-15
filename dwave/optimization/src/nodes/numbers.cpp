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
#include <functional>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "_state.hpp"
#include "dwave-optimization/array.hpp"
#include "dwave-optimization/common.hpp"

namespace dwave::optimization {

NumberNode::SumConstraint::SumConstraint(std::optional<ssize_t> axis,
                                         std::vector<Operator> operators,
                                         std::vector<double> bounds)
        : axis_(axis), operators_(std::move(operators)), bounds_(std::move(bounds)) {
    const size_t num_operators = operators_.size();
    const size_t num_bounds = bounds_.size();

    if ((num_operators == 0) || (num_bounds == 0)) {
        throw std::invalid_argument("`operators` and `bounds` must have non-zero size.");
    }

    if (!axis_.has_value() && (num_operators != 1 || num_bounds != 1)) {
        throw std::invalid_argument(
                "If `axis` is undefined, `operators` and `bounds` must have size 1.");
    }

    // If `operators` and `bounds` are both defined PER slice along `axis`,
    // they must have the same size.
    if ((num_operators > 1) && (num_bounds > 1) && (num_bounds != num_operators)) {
        assert(axis.has_value());
        throw std::invalid_argument(
                "`operators` and `bounds` should have same size if neither has size 1.");
    }
}

double NumberNode::SumConstraint::get_bound(const ssize_t slice) const {
    assert(0 <= slice);
    if (bounds_.size() == 1) return bounds_[0];
    assert(slice < static_cast<ssize_t>(bounds_.size()));
    return bounds_[slice];
}

NumberNode::SumConstraint::Operator NumberNode::SumConstraint::get_operator(
        const ssize_t slice) const {
    assert(0 <= slice);
    if (operators_.size() == 1) return operators_[0];
    assert(slice < static_cast<ssize_t>(operators_.size()));
    return operators_[slice];
}

/// State dependent data attached to NumberNode
struct NumberNodeStateData : public ArrayNodeStateData {
 public:
    // User does not provide sum constraints.
    NumberNodeStateData(std::vector<double> input) : ArrayNodeStateData(std::move(input)) {}
    // User provides sum constraints.
    NumberNodeStateData(std::vector<double> input,
                        std::vector<std::vector<double>> sum_constraints_lhs)
            : ArrayNodeStateData(std::move(input)),
              sum_constraints_lhs(std::move(sum_constraints_lhs)) {}

    std::unique_ptr<NodeStateData> copy() const override {
        return std::make_unique<NumberNodeStateData>(*this);
    }

    /// Commit the state dependent data of NumberNode.
    void commit() {
        ArrayNodeStateData::commit();  // Commit changes to the buffer.
        slice_cache_.clear();          // Empty the slice cache.
    }

    /// Revert the state dependent data of NumberNode.
    void revert();

    /// Update the relevant sum constraints running sums (`lhs`) given that the
    /// value stored at `index` is changed by `difference`.
    void update(const NumberNode& node, const ssize_t index, const double difference);
    /// Users may pass the slices (per sum constraint) that `index` lies on.
    void update(const NumberNode& node, const ssize_t index, const double difference,
                std::vector<ssize_t> slices);

    /// For each sum constraint, track the sum of the values within each slice.
    /// `sum_constraints_lhs[i][j]` is the sum of the values within the `j`th slice
    /// along the `axis`* defined by the `i`th sum constraint.
    ///
    /// (*) If `axis == std::nullopt`, the constraint is applied to the entire
    /// array, which is treated as a flat array with a single slice.
    std::vector<std::vector<double>> sum_constraints_lhs;

 protected:
    /// When updating the buffer, we cache the slice per sum constraint that
    /// a given index lies on for efficient reverts().
    ///
    /// slice_cache_[i][j] = The slice of the `j`th sum constraint that the index
    ///                      of the `i`th update lies on.
    std::vector<std::vector<ssize_t>> slice_cache_;
};

void NumberNodeStateData::revert() {
    // Undo changes to `sum_constraints_lhs` given the `slice_cache_`.
    if (slice_cache_.size() > 0) {
        std::span<const Update> updates = ArrayNodeStateData::diff();
        assert(updates.size() == slice_cache_.size());

        // Iterate over the updates.
        for (size_t i = 0, stop = updates.size(); i < stop; ++i) {
            const double difference = updates[i].value - updates[i].old;
            const std::vector<ssize_t>& slices = slice_cache_[i];

            // Reverse the change applied to each running sum
            for (size_t j = 0, stop = slices.size(); j < stop; ++j) {
                sum_constraints_lhs[j][slices[j]] -= difference;
            }
        }
    }

    ArrayNodeStateData::revert();  // Revert changes to the buffer.
    slice_cache_.clear();          // Empty the slice cache.
}

void NumberNodeStateData::update(const NumberNode& node, const ssize_t index,
                                 const double difference) {
    const auto& sum_constraints = node.sum_constraints();
    assert(sum_constraints.size() != 0);  // Should only call where applicable.
    assert(difference != 0);              // Should not call when no change occurs.
    assert(sum_constraints.size() == sum_constraints_lhs.size());

    std::vector<ssize_t> cache_entry;  // Initialize the slice cache.
    cache_entry.reserve(sum_constraints.size());
    // Get multidimensional indices for `index` so we can identify the slices
    // `index` lies on per sum constraint.
    const std::vector<ssize_t> multi_index = unravel_index(index, node.shape());
    assert(sum_constraints.size() <= multi_index.size());
    // For each sum constraint.
    for (size_t i = 0, stop = sum_constraints.size(); i < stop; ++i) {
        const std::optional<const ssize_t> axis = sum_constraints[i].axis();
        /// Determine the "slice" that index lies on given the sum constraint.
        /// If `axis == std::nullopt`, the array is treated as a flat array with a
        /// single slice. Otherwise, the slice is defined by multi_index.
        assert(!axis.has_value() || *axis < static_cast<ssize_t>(multi_index.size()));
        const ssize_t slice = axis.has_value() ? multi_index[*axis] : 0;
        assert(0 <= slice && slice < static_cast<ssize_t>(sum_constraints_lhs[i].size()));
        sum_constraints_lhs[i][slice] += difference;  // Offset slice sum.
        cache_entry.push_back(slice);                 // Record the slice in the cache.
    }
    slice_cache_.emplace_back(std::move(cache_entry));  // Cache the slices.
}

void NumberNodeStateData::update(const NumberNode& node, const ssize_t index,
                                 const double difference, std::vector<ssize_t> slices) {
    const auto& sum_constraints = node.sum_constraints();
    assert(sum_constraints.size() != 0);  // Should only call where applicable.
    assert(difference != 0);              // Should not call when no change occurs.
    assert(sum_constraints.size() == sum_constraints_lhs.size());
    assert(sum_constraints.size() == slices.size());
    // For each sum constraint.
    for (size_t i = 0, stop = sum_constraints.size(); i < stop; ++i) {
        // Sanity check that the user provided slices for `index` are correct.
        assert(([&]() {
            const std::optional<const ssize_t> axis = sum_constraints[i].axis();
            /// Determine the "slice" that index lies on given the sum constraint.
            /// If `axis == std::nullopt`, the array is treated as a flat array with a
            /// single slice. Otherwise, the slice is defined by unravel_index().
            if (!axis.has_value()) return slices[i] == 0;
            return slices[i] == unravel_index(index, node.shape())[*axis];
        })());
        sum_constraints_lhs[i][slices[i]] += difference;  // Offset slice sum.
    }
    slice_cache_.emplace_back(std::move(slices));  // Cache the slices.
}

double const* NumberNode::buff(const State& state) const noexcept {
    return data_ptr<NumberNodeStateData>(state)->buff();
}

std::span<const Update> NumberNode::diff(const State& state) const noexcept {
    return data_ptr<NumberNodeStateData>(state)->diff();
}

double NumberNode::min() const { return min_; }

double NumberNode::max() const { return max_; }

/// Given a NumberNode and an assignment of its variables (`number_data`),
/// compute and return a vector containing the sum of values for each slice
/// along the specified `axis`*.
///
/// (*) If `axis == std::nullopt`, the constraint is applied to the entire
/// array, which is treated as a flat array with a single slice.
std::vector<std::vector<double>> get_sum_constraints_lhs(const NumberNode& node,
                                                         const std::vector<double>& number_data) {
    std::span<const ssize_t> node_shape = node.shape();
    const auto& sum_constraints = node.sum_constraints();
    const size_t num_sum_constraints = sum_constraints.size();
    assert(num_sum_constraints <= node_shape.size());
    assert(std::accumulate(node_shape.begin(), node_shape.end(), 1, std::multiplies<ssize_t>()) ==
           static_cast<ssize_t>(number_data.size()));

    // For each sum constraint, initialize the sum of the values contained in
    // each of its slice to 0.
    std::vector<std::vector<double>> sum_constraints_lhs;
    sum_constraints_lhs.reserve(num_sum_constraints);
    for (const NumberNode::SumConstraint& constraint : sum_constraints) {
        const std::optional<const ssize_t> axis = constraint.axis();
        assert(!axis.has_value() || *axis < static_cast<ssize_t>(node_shape.size()));
        /// If `axis == std::nullopt`, the array is treated as a flat array with a
        /// single slice. Otherwise, the # of slices along axis is given by node shape.
        const ssize_t num_slices = axis.has_value() ? node_shape[*axis] : 1;
        // Emplace an all zeros vector of size equal to the number of slices.
        sum_constraints_lhs.emplace_back(num_slices, 0.0);
    }

    // Define a BufferIterator for `number_data` given the shape and strides of
    // NumberNode and iterate over it.
    for (BufferIterator<double, double, true> it(number_data.data(), node_shape, node.strides());
         it != std::default_sentinel; ++it) {
        // Increment the sum of the appropriate slice per sum constraint.
        for (size_t i = 0; i < num_sum_constraints; ++i) {
            const std::optional<const ssize_t> axis = sum_constraints[i].axis();
            assert(!axis.has_value() || *axis < static_cast<ssize_t>(it.location().size()));
            /// Determine the "slice" that the iterator lies on given the sum constraint.
            const ssize_t slice = axis.has_value() ? it.location()[*axis] : 0;
            assert(0 <= slice && slice < static_cast<ssize_t>(sum_constraints_lhs[i].size()));
            sum_constraints_lhs[i][slice] += *it;
        }
    }
    return sum_constraints_lhs;
}

/// Determine whether the sum constraints are satisfied.
bool satisfies_sum_constraint(const std::vector<NumberNode::SumConstraint>& sum_constraints,
                              const std::vector<std::vector<double>>& sum_constraints_lhs) {
    assert(sum_constraints.size() == sum_constraints_lhs.size());
    // Iterate over each sum constraint.
    for (ssize_t i = 0, stop_i = static_cast<ssize_t>(sum_constraints.size()); i < stop_i; ++i) {
        const auto& constraint = sum_constraints[i];
        const auto& lhs = sum_constraints_lhs[i];

        // Return `false` if any slice does not satisfy the constraint.
        for (ssize_t slice = 0, stop_slice = static_cast<ssize_t>(lhs.size()); slice < stop_slice;
             ++slice) {
            switch (constraint.get_operator(slice)) {
                case NumberNode::SumConstraint::Operator::Equal:
                    if (lhs[slice] != constraint.get_bound(slice)) return false;
                    break;
                case NumberNode::SumConstraint::Operator::LessEqual:
                    if (lhs[slice] > constraint.get_bound(slice)) return false;
                    break;
                case NumberNode::SumConstraint::Operator::GreaterEqual:
                    if (lhs[slice] < constraint.get_bound(slice)) return false;
                    break;
                default:
                    assert(false && "Unexpected operator type.");
                    unreachable();
            }
        }
    }
    return true;
}

void NumberNode::initialize_state(State& state, std::vector<double>&& number_data) const {
    if (number_data.size() != static_cast<size_t>(size())) {
        throw std::invalid_argument("Size of data provided does not match node size");
    }

    for (ssize_t index = 0, stop = size(); index < stop; ++index) {
        if (!is_valid(index, number_data[index])) {
            throw std::invalid_argument("Invalid data provided for node");
        }
    }

    if (sum_constraints_.size() == 0) {  // No sum constraints to consider.
        emplace_data_ptr<NumberNodeStateData>(state, std::move(number_data));
    } else {
        // Given the assignment to NumberNode `number_data`, compute the sum
        // of the values within each slice per sum constraint.
        auto sum_constraints_lhs = get_sum_constraints_lhs(*this, number_data);

        if (!satisfies_sum_constraint(sum_constraints_, sum_constraints_lhs)) {
            throw std::invalid_argument("Initialized values do not satisfy sum constraint(s).");
        }

        emplace_data_ptr<NumberNodeStateData>(state, std::move(number_data),
                                              std::move(sum_constraints_lhs));
    }
}

/// Given a `span` (used for strides or shape data), reorder the values
/// of the span such that the given `axis` is moved to the 0th index.
std::vector<ssize_t> shift_axis_data(const std::span<const ssize_t> span, const ssize_t axis) {
    const ssize_t ndim = span.size();
    std::vector<ssize_t> output;
    output.reserve(ndim);
    output.emplace_back(span[axis]);

    for (ssize_t i = 0; i < ndim; ++i) {
        if (i != axis) output.emplace_back(span[i]);
    }
    return output;
}

/// Reverse the operation defined by `shift_axis_data()`.
std::vector<ssize_t> undo_shift_axis_data(const std::span<const ssize_t> span, const ssize_t axis) {
    const ssize_t ndim = span.size();
    std::vector<ssize_t> output;
    output.reserve(ndim);

    ssize_t i_span = 1;
    for (ssize_t i = 0; i < ndim; ++i) {
        if (i == axis)
            output.emplace_back(span[0]);
        else
            output.emplace_back(span[i_span++]);
    }
    return output;
}

/// Given a `lhs`, operator (`op`), and a `bound`, determine the non-negative amount
/// `delta` needed to be added to `lhs` to satisfy the constraint: (lhs+delta) op bound.
/// e.g. Given (lhs, op, bound) := (10, ==, 12), delta = 2
/// e.g. Given (lhs, op, bound) := (10, <=, 12), delta = 0
/// e.g. Given (lhs, op, bound) := (10, >=, 12), delta = 2
/// Throws an error if `delta` is negative (corresponding with an infeasible sum constraint)
double sum_constraint_delta(const double lhs, const NumberNode::SumConstraint::Operator op,
                            const double bound) {
    switch (op) {
        case NumberNode::SumConstraint::Operator::Equal:
            if (lhs > bound) throw std::invalid_argument("Infeasible sum constraint.");
            // If error was not thrown, return amount needed to satisfy constraint.
            return bound - lhs;
        case NumberNode::SumConstraint::Operator::LessEqual:
            if (lhs > bound) throw std::invalid_argument("Infeasible sum constraint.");
            // If error was not thrown, sum satisfies constraint.
            return 0.0;
        case NumberNode::SumConstraint::Operator::GreaterEqual:
            // If sum is less than bound, return the amount needed to equal it.
            // Otherwise, sum satisfies constraint.
            return (lhs < bound) ? (bound - lhs) : 0.0;
        default:
            assert(false && "Unexpected operator type.");
            unreachable();
    }
}

/// Given a NumberNode and exactly one sum constraint, assign values to
/// `values` (in-place) to satisfy the constraint. This method
/// A) Initially sets `values[i] = lower_bound(i)` for all i.
/// B) Incremements the values within each slice until they satisfy
/// the constraint (should this be possible).
void construct_state_given_exactly_one_sum_constraint(const NumberNode& node,
                                                      std::vector<double>& values) {
    const std::span<const ssize_t> node_shape = node.shape();
    const ssize_t ndim = node_shape.size();
    // 1) Initialize all elements to their lower bounds.
    for (ssize_t i = 0, stop = node.size(); i < stop; ++i) {
        values.push_back(node.lower_bound(i));
    }
    // 2) Determine the slice sums for the sum constraint. To improve performance,
    // compute sum during previous loop.
    assert(node.sum_constraints().size() == 1);
    const std::vector<double> lhs = get_sum_constraints_lhs(node, values).front();
    // Obtain the stateless sum constraint information.
    const NumberNode::SumConstraint& constraint = node.sum_constraints().front();
    const std::optional<const ssize_t> axis = constraint.axis();

    // Handle the case where the constraint applies to the entire array.
    if (!axis.has_value()) {
        assert(lhs.size() == 1);
        // Determine the amount needed to adjust the values within the array.
        double delta = sum_constraint_delta(lhs.front(), constraint.get_operator(0),
                                            constraint.get_bound(0));
        if (delta == 0) return;  // Bound is satisfied for entire array.

        for (ssize_t i = 0, stop = node.size(); i < stop; ++i) {
            // Determine allowable amount we can increment the value in at `i`.
            const double inc = std::min(delta, node.upper_bound(i) - values[i]);

            if (inc > 0) {  // Apply the increment to both `values` and `delta`.
                values[i] += inc;
                delta -= inc;
                if (delta == 0) break;  // Bound is satisfied for entire array.
            }
        }

        if (delta != 0) throw std::invalid_argument("Infeasible sum constraint.");
        return;
    }

    assert(axis.has_value() && 0 <= *axis && *axis < ndim);
    // We need a way to iterate over each slice along the constrainted axis and
    // adjust its values until they satisfy the constraint. We do this by
    // defining an iterator of `values` that traverses each slice one after
    // another. This is equivalent to adjusting the node's shape and strides
    // such that the data for the constrained axis is moved to position 0.
    const std::vector<ssize_t> buff_shape = shift_axis_data(node_shape, *axis);
    const std::vector<ssize_t> buff_strides = shift_axis_data(node.strides(), *axis);
    // Define an iterator for `values` corresponding with the beginning of
    // slice 0 along the constrained axis.
    const BufferIterator<double, double, false> slice_0_it(values.data(), ndim, buff_shape.data(),
                                                           buff_strides.data());
    // Determine the size of each slice along the constrained axis.
    const ssize_t slice_size = std::accumulate(buff_shape.begin() + 1, buff_shape.end(), 1.0,
                                               std::multiplies<ssize_t>());

    // 3) Iterate over each slice and adjust its values until they satisfy the
    // sum constraint.
    for (ssize_t slice = 0, stop = node_shape[*axis]; slice < stop; ++slice) {
        // Determine the amount needed to adjust the values within the slice.
        double delta = sum_constraint_delta(lhs[slice], constraint.get_operator(slice),
                                            constraint.get_bound(slice));
        if (delta == 0) continue;  // Sum constraint is satisfied for slice.
        assert(delta >= 0);        // Should only increment.

        // Determine how much we need to offset `slice_0_it` to get to the
        // first index in the given `slice`.
        const ssize_t offset = slice * slice_size;
        // Iterate over all indices in the given slice.
        for (auto slice_it = slice_0_it + offset, slice_end_it = slice_it + slice_size;
             slice_it != slice_end_it; ++slice_it) {
            assert(slice_it.location()[0] == slice);  // We should be in the right slice.
            // Determine the "true" index of `slice_it` given the node shape.
            ssize_t index =
                    ravel_multi_index(undo_shift_axis_data(slice_it.location(), *axis), node_shape);
            // Sanity check that we can correctly reverse the conversion.
            assert(std::ranges::equal(shift_axis_data(unravel_index(index, node_shape), *axis),
                                      slice_it.location()));
            assert(0 <= index && index < static_cast<ssize_t>(values.size()));
            // Determine allowable amount we can increment the value in at `index`.
            const double inc = std::min(delta, node.upper_bound(index) - *slice_it);

            if (inc > 0) {  // Apply the increment to both `it` and `delta`.
                *slice_it += inc;
                delta -= inc;
                if (delta == 0) break;  // Sum constraint is satisfied for slice.
            }
        }

        if (delta != 0) throw std::invalid_argument("Infeasible sum constraint.");
    }
}

void NumberNode::initialize_state(State& state) const {
    std::vector<double> values;
    values.reserve(size());

    if (sum_constraints_.size() == 0) {
        // No sum constraint to consider, initialize by default.
        for (ssize_t i = 0, stop = size(); i < stop; ++i) {
            values.push_back(default_value(i));
        }
        initialize_state(state, std::move(values));
    } else if (sum_constraints_.size() == 1) {
        construct_state_given_exactly_one_sum_constraint(*this, values);
        initialize_state(state, std::move(values));
    } else {
        assert(false && "Multiple sum constraints not yet supported.");
        unreachable();
    }
}

void NumberNode::propagate(State& state) const {
    // Should only propagate states that obey the sum constraint(s).
    assert(satisfies_sum_constraint(sum_constraints_, sum_constraints_lhs(state)));
    // Technically vestigial but will keep it for forms sake.
    for (const auto& sv : successors()) {
        sv->update(state, sv.index);
    }
}

void NumberNode::commit(State& state) const noexcept {
    data_ptr<NumberNodeStateData>(state)->commit();
}

void NumberNode::revert(State& state) const noexcept {
    data_ptr<NumberNodeStateData>(state)->revert();
}

void NumberNode::exchange(State& state, ssize_t i, ssize_t j,
                          std::optional<std::vector<ssize_t>> i_slices,
                          std::optional<std::vector<ssize_t>> j_slices) const {
    auto state_data = data_ptr<NumberNodeStateData>(state);
    // We expect the exchange to obey the index-wise bounds.
    assert(lower_bound(i) <= state_data->get(j));
    assert(upper_bound(i) >= state_data->get(j));
    assert(lower_bound(j) <= state_data->get(i));
    assert(upper_bound(j) >= state_data->get(i));
    // assert() that i and j are valid indices occurs in ptr->exchange().
    // State change occurs IFF (i != j) and (buffer[i] != buffer[j]).
    if (state_data->exchange(i, j)) {
        // If change occurred and sum constraint exist, update running sums.
        if (sum_constraints_.size() > 0) {
            const double difference = state_data->get(i) - state_data->get(j);
            if (i_slices.has_value()) {
                assert(j_slices.has_value());
                // Index i changed from (what is now) ptr->get(j) to ptr->get(i)
                state_data->update(*this, i, difference, *i_slices);
                // Index j changed from (what is now) ptr->get(i) to ptr->get(j)
                state_data->update(*this, j, -difference, *j_slices);
            } else {
                // Index i changed from (what is now) ptr->get(j) to ptr->get(i)
                state_data->update(*this, i, difference);
                // Index j changed from (what is now) ptr->get(i) to ptr->get(j)
                state_data->update(*this, j, -difference);
            }
        }
    }
}

double NumberNode::get_value(State& state, ssize_t i) const {
    return data_ptr<NumberNodeStateData>(state)->get(i);
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

void NumberNode::clip_and_set_value(State& state, ssize_t index, double value,
                                    std::optional<std::vector<ssize_t>> slices) const {
    auto state_data = data_ptr<NumberNodeStateData>(state);
    value = std::clamp(value, lower_bound(index), upper_bound(index));
    // assert() that i is a valid index occurs in ptr->set().
    // State change occurs IFF `value` != buffer[index].
    if (state_data->set(index, value)) {
        // If change occurred and sum constraint exist, update running sums.
        if (sum_constraints_.size() > 0) {
            if (slices.has_value()) {
                state_data->update(*this, index, value - diff(state).back().old, *slices);
            } else {
                state_data->update(*this, index, value - diff(state).back().old);
            }
        }
    }
}

const std::vector<NumberNode::SumConstraint>& NumberNode::sum_constraints() const {
    return sum_constraints_;
}

const std::vector<std::vector<double>>& NumberNode::sum_constraints_lhs(const State& state) const {
    return data_ptr<NumberNodeStateData>(state)->sum_constraints_lhs;
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

/// Check the user defined sum constraint(s).
void check_sum_constraints(const NumberNode& node) {
    const std::vector<NumberNode::SumConstraint>& sum_constraints = node.sum_constraints();
    if (sum_constraints.size() == 0) return;  // No sum constraints to check.

    const std::span<const ssize_t> shape = node.shape();
    // Used to assess if an axis is subject to multiple constraints.
    std::vector<bool> constrained_axis(shape.size(), false);
    // Used to assess if array is subject to multiple constraints.
    bool constrained_array = false;

    for (const NumberNode::SumConstraint& constraint : sum_constraints) {
        const std::optional<const ssize_t> axis = constraint.axis();
        const ssize_t num_operators = static_cast<ssize_t>(constraint.num_operators());
        const ssize_t num_bounds = static_cast<ssize_t>(constraint.num_bounds());

        // Handle the case where the constraint applies to the entire array.
        if (!axis.has_value()) {
            // Checked in SumConstraint constructor
            assert(num_operators == 1 && num_bounds == 1);

            if (constrained_array)
                throw std::invalid_argument(
                        "Cannot define multiple sum constraints for the entire number array.");
            constrained_array = true;
            continue;
        }

        assert(axis.has_value());

        if (*axis < 0 || *axis >= static_cast<ssize_t>(shape.size())) {
            throw std::invalid_argument("Invalid constrained axis given number array shape.");
        }

        if ((num_operators > 1) && (num_operators != shape[*axis])) {
            throw std::invalid_argument("Invalid number of operators given number array shape.");
        }

        if ((num_bounds > 1) && (num_bounds != shape[*axis])) {
            throw std::invalid_argument("Invalid number of bounds given number array shape.");
        }

        // Checked in SumConstraint constructor
        assert(num_operators == num_bounds || num_operators == 1 || num_bounds == 1);

        if (constrained_axis[*axis]) {
            throw std::invalid_argument(
                    "Cannot define multiple sum constraints for a single axis.");
        }
        constrained_axis[*axis] = true;
    }

    // *Currently*, we only support one sum constraint.
    if (sum_constraints.size() > 1) {
        throw std::invalid_argument("Can define at most one sum constraint per number array.");
    }

    // There are fasters ways to check whether the sum constraints are feasible.
    // For now, fully attempt to construct a state and throw if impossible.
    std::vector<double> values;
    values.reserve(node.size());
    construct_state_given_exactly_one_sum_constraint(node, values);
}

// Base class to be used as interfaces.
NumberNode::NumberNode(std::span<const ssize_t> shape, std::vector<double> lower_bound,
                       std::vector<double> upper_bound, std::vector<SumConstraint> sum_constraints)
        : ArrayOutputMixin(shape),
          min_(get_extreme_index_wise_bound<false>(lower_bound)),
          max_(get_extreme_index_wise_bound<true>(upper_bound)),
          lower_bounds_(std::move(lower_bound)),
          upper_bounds_(std::move(upper_bound)),
          sum_constraints_(std::move(sum_constraints)) {
    if ((shape.size() > 0) && (shape[0] < 0)) {
        throw std::invalid_argument("Number array cannot have dynamic size.");
    }

    if (max_ < min_) {
        throw std::invalid_argument("Invalid range for number array provided.");
    }

    check_index_wise_bounds(*this, lower_bounds_, upper_bounds_);
    check_sum_constraints(*this);
}

// Integer Node ***************************************************************

/// Check the user defined sum constraint for IntegerNode.
void check_sum_constraint_integrality(
        const std::vector<NumberNode::SumConstraint>& sum_constraints) {
    if (sum_constraints.size() == 0) return;  // No sum constraints to check.

    for (const NumberNode::SumConstraint& constraint : sum_constraints) {
        for (ssize_t slice = 0, stop = constraint.num_bounds(); slice < stop; ++slice) {
            const double bound = constraint.get_bound(slice);
            if (bound != std::floor(bound)) {
                throw std::invalid_argument(
                        "Sum constraint(s) for integral arrays must be integral.");
            }
        }
    }
}

IntegerNode::IntegerNode(std::span<const ssize_t> shape,
                         std::optional<std::vector<double>> lower_bound,
                         std::optional<std::vector<double>> upper_bound,
                         std::vector<SumConstraint> sum_constraints)
        : NumberNode(
                  shape,
                  lower_bound.has_value() ? std::move(*lower_bound)
                                          : std::vector<double>{default_lower_bound},
                  upper_bound.has_value() ? std::move(*upper_bound)
                                          : std::vector<double>{default_upper_bound},
                  (check_sum_constraint_integrality(sum_constraints), std::move(sum_constraints))) {
    if (min_ < minimum_lower_bound || max_ > maximum_upper_bound) {
        throw std::invalid_argument("range provided for integers exceeds supported range");
    }
}

IntegerNode::IntegerNode(std::initializer_list<ssize_t> shape,
                         std::optional<std::vector<double>> lower_bound,
                         std::optional<std::vector<double>> upper_bound,
                         std::vector<SumConstraint> sum_constraints)
        : IntegerNode(std::span(shape), std::move(lower_bound), std::move(upper_bound),
                      std::move(sum_constraints)) {}
IntegerNode::IntegerNode(ssize_t size, std::optional<std::vector<double>> lower_bound,
                         std::optional<std::vector<double>> upper_bound,
                         std::vector<SumConstraint> sum_constraints)
        : IntegerNode({size}, std::move(lower_bound), std::move(upper_bound),
                      std::move(sum_constraints)) {}

IntegerNode::IntegerNode(std::span<const ssize_t> shape, double lower_bound,
                         std::optional<std::vector<double>> upper_bound,
                         std::vector<SumConstraint> sum_constraints)
        : IntegerNode(shape, std::vector<double>{lower_bound}, std::move(upper_bound),
                      std::move(sum_constraints)) {}
IntegerNode::IntegerNode(std::initializer_list<ssize_t> shape, double lower_bound,
                         std::optional<std::vector<double>> upper_bound,
                         std::vector<SumConstraint> sum_constraints)
        : IntegerNode(std::span(shape), std::vector<double>{lower_bound}, std::move(upper_bound),
                      std::move(sum_constraints)) {}
IntegerNode::IntegerNode(ssize_t size, double lower_bound,
                         std::optional<std::vector<double>> upper_bound,
                         std::vector<SumConstraint> sum_constraints)
        : IntegerNode({size}, std::vector<double>{lower_bound}, std::move(upper_bound),
                      std::move(sum_constraints)) {}

IntegerNode::IntegerNode(std::span<const ssize_t> shape,
                         std::optional<std::vector<double>> lower_bound, double upper_bound,
                         std::vector<SumConstraint> sum_constraints)
        : IntegerNode(shape, std::move(lower_bound), std::vector<double>{upper_bound},
                      std::move(sum_constraints)) {}
IntegerNode::IntegerNode(std::initializer_list<ssize_t> shape,
                         std::optional<std::vector<double>> lower_bound, double upper_bound,
                         std::vector<SumConstraint> sum_constraints)
        : IntegerNode(std::span(shape), std::move(lower_bound), std::vector<double>{upper_bound},
                      std::move(sum_constraints)) {}
IntegerNode::IntegerNode(ssize_t size, std::optional<std::vector<double>> lower_bound,
                         double upper_bound, std::vector<SumConstraint> sum_constraints)
        : IntegerNode({size}, std::move(lower_bound), std::vector<double>{upper_bound},
                      std::move(sum_constraints)) {}

IntegerNode::IntegerNode(std::span<const ssize_t> shape, double lower_bound, double upper_bound,
                         std::vector<SumConstraint> sum_constraints)
        : IntegerNode(shape, std::vector<double>{lower_bound}, std::vector<double>{upper_bound},
                      std::move(sum_constraints)) {}
IntegerNode::IntegerNode(std::initializer_list<ssize_t> shape, double lower_bound,
                         double upper_bound, std::vector<SumConstraint> sum_constraints)
        : IntegerNode(std::span(shape), std::vector<double>{lower_bound},
                      std::vector<double>{upper_bound}, std::move(sum_constraints)) {}
IntegerNode::IntegerNode(ssize_t size, double lower_bound, double upper_bound,
                         std::vector<SumConstraint> sum_constraints)
        : IntegerNode({size}, std::vector<double>{lower_bound}, std::vector<double>{upper_bound},
                      std::move(sum_constraints)) {}

bool IntegerNode::integral() const { return true; }

bool IntegerNode::is_valid(ssize_t index, double value) const {
    return (value >= lower_bound(index)) && (value <= upper_bound(index)) &&
           (std::round(value) == value);
}

void IntegerNode::set_value(State& state, ssize_t index, double value,
                            std::optional<std::vector<ssize_t>> slices) const {
    auto state_data = data_ptr<NumberNodeStateData>(state);
    // We expect `value` to obey the index-wise bounds and to be an integer.
    assert(lower_bound(index) <= value);
    assert(upper_bound(index) >= value);
    assert(value == std::round(value));
    // assert() that i is a valid index occurs in ptr->set().
    // State change occurs IFF `value` != buffer[index].
    if (state_data->set(index, value)) {
        // If change occurred and sum constraint exist, update running sums.
        if (sum_constraints_.size() > 0) {
            if (slices.has_value()) {
                state_data->update(*this, index, value - diff(state).back().old, *slices);
            } else {
                state_data->update(*this, index, value - diff(state).back().old);
            }
        }
    }
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
                       std::optional<std::vector<double>> upper_bound,
                       std::vector<SumConstraint> sum_constraints)
        : IntegerNode(shape, limit_bound_to_bool_domain<false>(lower_bound),
                      limit_bound_to_bool_domain<true>(upper_bound), std::move(sum_constraints)) {}

BinaryNode::BinaryNode(std::initializer_list<ssize_t> shape,
                       std::optional<std::vector<double>> lower_bound,
                       std::optional<std::vector<double>> upper_bound,
                       std::vector<SumConstraint> sum_constraints)
        : BinaryNode(std::span(shape), std::move(lower_bound), std::move(upper_bound),
                     std::move(sum_constraints)) {}
BinaryNode::BinaryNode(ssize_t size, std::optional<std::vector<double>> lower_bound,
                       std::optional<std::vector<double>> upper_bound,
                       std::vector<SumConstraint> sum_constraints)
        : BinaryNode({size}, std::move(lower_bound), std::move(upper_bound),
                     std::move(sum_constraints)) {}

BinaryNode::BinaryNode(std::span<const ssize_t> shape, double lower_bound,
                       std::optional<std::vector<double>> upper_bound,
                       std::vector<SumConstraint> sum_constraints)
        : BinaryNode(shape, std::vector<double>{lower_bound}, std::move(upper_bound),
                     std::move(sum_constraints)) {}
BinaryNode::BinaryNode(std::initializer_list<ssize_t> shape, double lower_bound,
                       std::optional<std::vector<double>> upper_bound,
                       std::vector<SumConstraint> sum_constraints)
        : BinaryNode(std::span(shape), std::vector<double>{lower_bound}, std::move(upper_bound),
                     std::move(sum_constraints)) {}
BinaryNode::BinaryNode(ssize_t size, double lower_bound,
                       std::optional<std::vector<double>> upper_bound,
                       std::vector<SumConstraint> sum_constraints)
        : BinaryNode({size}, std::vector<double>{lower_bound}, std::move(upper_bound),
                     std::move(sum_constraints)) {}

BinaryNode::BinaryNode(std::span<const ssize_t> shape,
                       std::optional<std::vector<double>> lower_bound, double upper_bound,
                       std::vector<SumConstraint> sum_constraints)
        : BinaryNode(shape, std::move(lower_bound), std::vector<double>{upper_bound},
                     std::move(sum_constraints)) {}
BinaryNode::BinaryNode(std::initializer_list<ssize_t> shape,
                       std::optional<std::vector<double>> lower_bound, double upper_bound,
                       std::vector<SumConstraint> sum_constraints)
        : BinaryNode(std::span(shape), std::move(lower_bound), std::vector<double>{upper_bound},
                     std::move(sum_constraints)) {}
BinaryNode::BinaryNode(ssize_t size, std::optional<std::vector<double>> lower_bound,
                       double upper_bound, std::vector<SumConstraint> sum_constraints)
        : BinaryNode({size}, std::move(lower_bound), std::vector<double>{upper_bound},
                     std::move(sum_constraints)) {}

BinaryNode::BinaryNode(std::span<const ssize_t> shape, double lower_bound, double upper_bound,
                       std::vector<SumConstraint> sum_constraints)
        : BinaryNode(shape, std::vector<double>{lower_bound}, std::vector<double>{upper_bound},
                     std::move(sum_constraints)) {}
BinaryNode::BinaryNode(std::initializer_list<ssize_t> shape, double lower_bound, double upper_bound,
                       std::vector<SumConstraint> sum_constraints)
        : BinaryNode(std::span(shape), std::vector<double>{lower_bound},
                     std::vector<double>{upper_bound}, std::move(sum_constraints)) {}
BinaryNode::BinaryNode(ssize_t size, double lower_bound, double upper_bound,
                       std::vector<SumConstraint> sum_constraints)
        : BinaryNode({size}, std::vector<double>{lower_bound}, std::vector<double>{upper_bound},
                     std::move(sum_constraints)) {}

void BinaryNode::flip(State& state, ssize_t index,
                      std::optional<std::vector<ssize_t>> slices) const {
    auto state_data = data_ptr<NumberNodeStateData>(state);
    // Variable should not be fixed.
    assert(lower_bound(index) != upper_bound(index));
    // assert() that `index` is valid occurs in ptr->set().
    // State change occurs IFF `value` != buffer[index].
    if (state_data->set(index, !state_data->get(index))) {
        // If change occurred and sum constraint exist, update running sums.
        if (sum_constraints_.size() > 0) {
            // If value changed from 0 -> 1, update by 1.
            // If value changed from 1 -> 0, update by -1.
            if (slices.has_value()) {
                state_data->update(*this, index, (state_data->get(index) == 1) ? 1 : -1, *slices);
            } else {
                state_data->update(*this, index, (state_data->get(index) == 1) ? 1 : -1);
            }
        }
    }
}

}  // namespace dwave::optimization

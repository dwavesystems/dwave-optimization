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

BoundAxisInfo::BoundAxisInfo(ssize_t bound_axis, std::vector<BoundAxisOperator> axis_operators,
                             std::vector<double> axis_bounds)
        : axis(bound_axis), operators(std::move(axis_operators)), bounds(std::move(axis_bounds)) {
    const ssize_t num_operators = static_cast<ssize_t>(operators.size());
    const ssize_t num_bounds = static_cast<ssize_t>(bounds.size());

    // Null `operators` and `bounds` are not accepted.
    if ((num_operators == 0) || (num_bounds == 0)) {
        throw std::invalid_argument("Bad axis-wise bounds for axis: " + std::to_string(axis) +
                                    ", `operators` and `bounds` must each have non-zero size.");
    }

    // If `operators` and `bounds` are defined PER hyperslice along `axis`,
    // they must have the same size.
    if ((num_operators > 1) && (num_bounds > 1) && (num_bounds != num_operators)) {
        throw std::invalid_argument(
                "Bad axis-wise bounds for axis: " + std::to_string(axis) +
                ", `operators` and `bounds` should have same size if neither has size 1.");
    }
}

double BoundAxisInfo::get_bound(const ssize_t slice) const {
    assert(0 <= slice);
    if (bounds.size() == 0) return bounds[0];
    assert(slice < static_cast<ssize_t>(bounds.size()));
    return bounds[slice];
}

BoundAxisOperator BoundAxisInfo::get_operator(const ssize_t slice) const {
    assert(0 <= slice);
    if (operators.size() == 0) return operators[0];
    assert(slice < static_cast<ssize_t>(operators.size()));
    return operators[slice];
}

/// State dependant data attached to NumberNode
struct NumberNodeStateData : public ArrayNodeStateData {
    NumberNodeStateData(std::vector<double> input) : ArrayNodeStateData(std::move(input)) {}
    NumberNodeStateData(std::vector<double> input, std::vector<std::vector<double>> bound_axes_sums)
            : ArrayNodeStateData(std::move(input)),
              bound_axes_sums(std::move(bound_axes_sums)),
              prior_bound_axes_sums(this->bound_axes_sums) {}
    /// For each bound axis and for each hyperslice along said axis, we
    /// track the sum of the values within the hyperslice.
    /// bound_axes_sums[i][j] = "sum of the values within the jth
    ///                          hyperslice along the ith bound axis"
    std::vector<std::vector<double>> bound_axes_sums;
    // Store a copy for NumberNode::revert() and commit()
    std::vector<std::vector<double>> prior_bound_axes_sums;
};

double const* NumberNode::buff(const State& state) const noexcept {
    return data_ptr<NumberNodeStateData>(state)->buff();
}

std::span<const Update> NumberNode::diff(const State& state) const noexcept {
    return data_ptr<NumberNodeStateData>(state)->diff();
}

double NumberNode::min() const { return min_; }

double NumberNode::max() const { return max_; }

std::vector<std::vector<double>> get_bound_axes_sums(const NumberNode* node,
                                                     const std::vector<double>& number_data) {
    std::span<const ssize_t> node_shape = node->shape();
    const std::vector<BoundAxisInfo>& bound_axes_info = node->axis_wise_bounds();
    const ssize_t num_bound_axes = static_cast<ssize_t>(bound_axes_info.size());

    assert(num_bound_axes <= node_shape.size());
    assert(std::accumulate(node_shape.begin(), node_shape.end(), 1, std::multiplies<ssize_t>()) ==
           static_cast<ssize_t>(number_data.size()));

    // For each bound axis, initialize the sum of the values contained in each
    // of it's hyperslice to 0.
    std::vector<std::vector<double>> bound_axes_sums;
    bound_axes_sums.reserve(num_bound_axes);
    for (const BoundAxisInfo& axis_info : bound_axes_info) {
        assert(0 <= axis_info.axis && axis_info.axis < static_cast<ssize_t>(node_shape.size()));
        bound_axes_sums.emplace_back(node_shape[axis_info.axis], 0.0);
    }

    // Define a BufferIterator for number_data (contiguous block of doubles)
    // given the shape and strides of NumberNode.
    BufferIterator<double, double, true> it(number_data.data(), node_shape, node->strides());

    // Iterate over number_data.
    for (; it != std::default_sentinel; ++it) {
        // Increment the appropriate hyperslice along each bound axis.
        for (ssize_t bound_axis = 0; bound_axis < num_bound_axes; ++bound_axis) {
            const ssize_t axis = bound_axes_info[bound_axis].axis;
            assert(0 <= axis && axis < it.location().size());
            const ssize_t slice = it.location()[axis];
            assert(0 <= slice && slice < bound_axes_sums[bound_axis].size());
            bound_axes_sums[bound_axis][slice] += *it;
        }
    }

    return bound_axes_sums;
}

bool satisfies_axis_wise_bounds(const std::vector<BoundAxisInfo>& bound_axes_info,
                                const std::vector<std::vector<double>>& bound_axes_sums) {
    assert(bound_axes_info.size() == bound_axes_sums.size());
    // Check that each hyperslice satisfies the axis-wise bounds.
    for (ssize_t i = 0, stop_i = static_cast<ssize_t>(bound_axes_info.size()); i < stop_i; ++i) {
        const std::vector<double>& bound_axis_sums = bound_axes_sums[i];
        const BoundAxisInfo& bound_axis_info = bound_axes_info[i];

        for (ssize_t slice = 0, stop_slice = static_cast<ssize_t>(bound_axis_sums.size());
             slice < stop_slice; ++slice) {
            switch (bound_axis_info.get_operator(slice)) {
                case Equal:
                    if (bound_axis_sums[slice] != bound_axis_info.get_bound(slice)) return false;
                    break;
                case LessEqual:
                    if (bound_axis_sums[slice] > bound_axis_info.get_bound(slice)) return false;
                    break;
                case GreaterEqual:
                    if (bound_axis_sums[slice] < bound_axis_info.get_bound(slice)) return false;
                    break;
                default:
                    unreachable();
            }
        }
    }
    return true;
}

void NumberNode::initialize_state(State& state, std::vector<double>&& number_data) const {
    if (number_data.size() != static_cast<size_t>(this->size())) {
        throw std::invalid_argument("Size of data provided does not match node size");
    }

    for (ssize_t index = 0, stop = this->size(); index < stop; ++index) {
        if (!is_valid(index, number_data[index])) {
            throw std::invalid_argument("Invalid data provided for node");
        }
    }

    if (bound_axes_info_.size() == 0) {  // No bound axes to consider.
        emplace_data_ptr<NumberNodeStateData>(state, std::move(number_data));
        return;
    }

    std::vector<std::vector<double>> bound_axes_sums = get_bound_axes_sums(this, number_data);

    if (!satisfies_axis_wise_bounds(bound_axes_info_, bound_axes_sums)) {
        throw std::invalid_argument("Initialized values do not satisfy axis-wise bounds.");
    }

    emplace_data_ptr<NumberNodeStateData>(state, std::move(number_data),
                                          std::move(bound_axes_sums));
}

void construct_state_given_exactly_one_bound_axis(const NumberNode* node,
                                                  std::vector<double>& values) {
    const std::span<const ssize_t> node_shape = node->shape();
    const std::span<const ssize_t> node_strides = node->strides();
    assert(node_shape.size() == node_strides.size());
    const ssize_t ndim = node_shape.size();

    // We need to construct a state that satisfies the axis wise bounds.
    // First, initialize all elements to their lower bounds.
    for (ssize_t i = 0, stop = node->size(); i < stop; ++i) {
        values.push_back(node->lower_bound(i));
    }
    // Second, determine the hyperslice sums for the bound axis. This could be
    // done during the previous loop if we want to improve performance.
    assert(node->axis_wise_bounds().size() == 1);
    std::vector<double> bound_axis_sums = get_bound_axes_sums(node, values)[0];
    const BoundAxisInfo& bound_axis_info = node->axis_wise_bounds()[0];
    const ssize_t bound_axis = bound_axis_info.axis;
    assert(0 <= bound_axis && bound_axis < ndim);
    // Iterator to the beginning of `values`.
    BufferIterator<double, double, false> values_begin(values.data(), ndim, node_shape.data(),
                                                       node_strides.data());
    // Offset used to perterb `values_begin` to the first element of the
    // hyperslice along the given bound axis.
    std::vector<ssize_t> offset(ndim, 0);

    // Third, we iterate over each hyperslice and adjust its values until
    // it satisfies the axis-wise bounds.
    for (ssize_t slice = 0, stop = node_shape[bound_axis]; slice < stop; ++slice) {
        // Determine the amount we need to adjust the initialized values by
        // to satisfy the axis-wise bounds for the given hyperslice.
        double delta = 0;

        switch (bound_axis_info.get_operator(slice)) {
            case Equal:
                if (bound_axis_sums[slice] > bound_axis_info.get_bound(slice)) {
                    throw std::invalid_argument("Axis-wise bounds are infeasible.");
                }
                delta = bound_axis_info.get_bound(slice) - bound_axis_sums[slice];
                assert(delta >= 0);
                // If error was not thrown, either (delta > 0) and (sum <
                // bound) or (delta == 0) and (sum == bound).
                break;
            case LessEqual:
                if (bound_axis_sums[slice] > bound_axis_info.get_bound(slice)) {
                    throw std::invalid_argument("Axis-wise bounds are infeasible.");
                }
                // If error was not thrown, then (delta == 0) and (sum <= bound)
                break;
            case GreaterEqual:
                if (bound_axis_sums[slice] < bound_axis_info.get_bound(slice)) {
                    delta = bound_axis_info.get_bound(slice) - bound_axis_sums[slice];
                }
                assert(delta >= 0);
                // Either (delta == 0) and (sum >= bound) or (delta > 0) and
                // (sum < bound).
                break;
            default:
                unreachable();
        }

        if (delta == 0) continue;  // axis-wise bounds are satisfied for slice.

        // Define iterator to the cannonically least index in the given slice
        // along the bound axis.
        offset[bound_axis] = slice;
        BufferIterator<double, double, false> it = values_begin + offset;

        // Iterate over all remaining elements in values.
        for (; it != std::default_sentinel_t(); ++it) {
            // Only consider values that fall in the slice.
            if (it.location()[bound_axis] != slice) continue;

            // Determine the index of `it` from `values_begin`
            const ssize_t index = static_cast<ssize_t>(it - values_begin);
            assert(0 <= index && index < values.size());
            // Determine the amount we can increment the value in the given index.
            ssize_t inc = std::min(delta, node->upper_bound(index) - *it);

            if (inc > 0) {  // Apply the increment to both `it` and `delta`.
                *it += inc;
                delta -= inc;
                if (delta == 0) break; // Axis-wise bounds are now satisfied for slice.
            }
        }

        if (delta != 0) {
            throw std::invalid_argument("Axis-wise bounds are infeasible.");
        }
    }
}

void NumberNode::initialize_state(State& state) const {
    std::vector<double> values;
    values.reserve(this->size());

    if (bound_axes_info_.size() == 0) {  // No bound axes to consider
        for (ssize_t i = 0, stop = this->size(); i < stop; ++i) {
            values.push_back(default_value(i));
        }
        initialize_state(state, std::move(values));
        return;
    }

    if (bound_axes_info_.size() != 1) {
        throw std::invalid_argument("Cannot initialize state with multiple bound axes.");
    }

    construct_state_given_exactly_one_bound_axis(this, values);
    initialize_state(state, std::move(values));
}

void NumberNode::commit(State& state) const noexcept {
    auto node_data = data_ptr<NumberNodeStateData>(state);
    // Manually store a copy of bound_axes_sums.
    node_data->prior_bound_axes_sums = node_data->bound_axes_sums;
    node_data->commit();
}

void NumberNode::revert(State& state) const noexcept {
    auto node_data = data_ptr<NumberNodeStateData>(state);
    // Manually reset bound_axes_sums.
    node_data->bound_axes_sums = node_data->prior_bound_axes_sums;
    node_data->revert();
}

void NumberNode::exchange(State& state, ssize_t i, ssize_t j) const {
    auto ptr = data_ptr<NumberNodeStateData>(state);
    // We expect the exchange to obey the index-wise bounds.
    assert(lower_bound(i) <= ptr->get(j));
    assert(upper_bound(i) >= ptr->get(j));
    assert(lower_bound(j) <= ptr->get(i));
    assert(upper_bound(j) >= ptr->get(i));
    // Assert that i and j are valid indices occurs in ptr->exchange().
    // Exchange occurs IFF (i != j) and (buffer[i] != buffer[j]).
    if (ptr->exchange(i, j)) {
        // If the values at indices i and j were exchanged, update the bound
        // axis sums.
        const double difference = ptr->get(i) - ptr->get(j);
        // Index i changed from (what is now) ptr->get(j) to ptr->get(i)
        update_bound_axis_slice_sums(state, i, difference);
        // Index j changed from (what is now) ptr->get(i) to ptr->get(j)
        update_bound_axis_slice_sums(state, j, -difference);
        assert(satisfies_axis_wise_bounds(bound_axes_info_, ptr->bound_axes_sums));
    }
}

double NumberNode::get_value(State& state, ssize_t i) const {
    return data_ptr<NumberNodeStateData>(state)->get(i);
}

double NumberNode::lower_bound(ssize_t index) const {
    if (lower_bounds_.size() == 1) {
        return lower_bounds_[0];
    }
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
    auto ptr = data_ptr<NumberNodeStateData>(state);
    value = std::clamp(value, lower_bound(index), upper_bound(index));
    // Assert that i is a valid index occurs in data_ptr->set().
    // Set occurs IFF `value` != buffer[i] .
    if (ptr->set(index, value)) {
        // Update the bound axis sums.
        update_bound_axis_slice_sums(state, index, value - diff(state).back().old);
        assert(satisfies_axis_wise_bounds(bound_axes_info_, ptr->bound_axes_sums));
    }
}

const std::vector<BoundAxisInfo>& NumberNode::axis_wise_bounds() const { return bound_axes_info_; }

const std::vector<std::vector<double>>& NumberNode::bound_axis_sums(State& state) const {
    return data_ptr<NumberNodeStateData>(state)->bound_axes_sums;
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

/// Check the user defined axis-wise bounds for NumberNode
void check_axis_wise_bounds(const std::vector<BoundAxisInfo>& bound_axes_info,
                            const std::span<const ssize_t> shape) {
    if (bound_axes_info.size() == 0) return;  // No bound axes to check.

    // Used to asses if an axis have been bound multiple times.
    std::vector<bool> axis_bound(shape.size(), false);

    // For each set of bound axis data
    for (const BoundAxisInfo& bound_axis_info : bound_axes_info) {
        const ssize_t axis = bound_axis_info.axis;

        if (axis < 0 || axis >= static_cast<ssize_t>(shape.size())) {
            throw std::invalid_argument(
                    "Invalid bound axis: " + std::to_string(axis) +
                    ". Note, negative indexing is not supported for axis-wise bounds.");
        }

        // The number of operators defined for the given bound axis
        const ssize_t num_operators = static_cast<ssize_t>(bound_axis_info.operators.size());
        if ((num_operators > 1) && (num_operators != shape[axis])) {
            throw std::invalid_argument(
                    "Invalid number of axis-wise operators along axis: " + std::to_string(axis) +
                    " given axis size: " + std::to_string(shape[axis]));
        }

        // The number of operators defined for the given bound axis
        const ssize_t num_bounds = static_cast<ssize_t>(bound_axis_info.bounds.size());
        if ((num_bounds > 1) && (num_bounds != shape[axis])) {
            throw std::invalid_argument(
                    "Invalid number of axis-wise bounds along axis: " + std::to_string(axis) +
                    " given axis size: " + std::to_string(shape[axis]));
        }

        // Checked in BoundAxisInfo constructor
        assert(num_operators == num_bounds || num_operators == 1 || num_bounds == 1);

        if (axis_bound[axis]) {
            throw std::invalid_argument(
                    "Cannot define multiple axis-wise bounds for a single axis.");
        }
        axis_bound[axis] = true;
    }

    // *Currently*, we only support axis-wise bounds for up to one axis.
    if (bound_axes_info.size() > 1) {
        throw std::invalid_argument("Axis-wise bounds are supported for at most one axis.");
    }
}

// Base class to be used as interfaces.
NumberNode::NumberNode(std::span<const ssize_t> shape, std::vector<double> lower_bound,
                       std::vector<double> upper_bound,
                       std::optional<std::vector<BoundAxisInfo>> bound_axes)
        : ArrayOutputMixin(shape),
          min_(get_extreme_index_wise_bound<false>(lower_bound)),
          max_(get_extreme_index_wise_bound<true>(upper_bound)),
          lower_bounds_(std::move(lower_bound)),
          upper_bounds_(std::move(upper_bound)),
          bound_axes_info_(bound_axes ? std::move(*bound_axes) : std::vector<BoundAxisInfo>{}) {
    if ((shape.size() > 0) && (shape[0] < 0)) {
        throw std::invalid_argument("Number array cannot have dynamic size.");
    }

    if (max_ < min_) {
        throw std::invalid_argument("Invalid range for number array provided.");
    }

    check_index_wise_bounds(*this, lower_bounds_, upper_bounds_);
    check_axis_wise_bounds(bound_axes_info_, this->shape());
}

void NumberNode::update_bound_axis_slice_sums(State& state, const ssize_t index,
                                              const double value_change) const {
    const auto& bound_axes_info = bound_axes_info_;
    if (bound_axes_info.size() == 0) return;  // No axis-wise bounds to satisfy

    // Get multidimensional indices for `index` so we can identify the slices
    // `index` lies on per bound axis.
    const std::vector<ssize_t> multi_index = unravel_index(index, this->shape());
    assert(bound_axes_info.size() <= multi_index.size());
    // Get the hyperslice sums of all bound axes.
    auto& bound_axes_sums = data_ptr<NumberNodeStateData>(state)->bound_axes_sums;
    assert(bound_axes_info.size() == bound_axes_sums.size());

    for (ssize_t bound_axis = 0, stop = static_cast<ssize_t>(bound_axes_info.size());
         bound_axis < stop; ++bound_axis) {
        assert(bound_axes_info[bound_axis].axis < static_cast<ssize_t>(multi_index.size()));
        // Get the slice along the bound axis the `value_change` occurs in
        const ssize_t slice = multi_index[bound_axes_info[bound_axis].axis];
        assert(slice < static_cast<ssize_t>(bound_axes_sums[bound_axis].size()));
        // Offset running sum in slice
        bound_axes_sums[bound_axis][slice] += value_change;
    }
}

// Integer Node ***************************************************************

/// Check the user defined axis-wise bounds for IntegerNode
void check_integrality_of_axis_wise_bounds(const std::vector<BoundAxisInfo>& bound_axes_info) {
    if (bound_axes_info.size() == 0) return;  // No bound axes to check.

    for (const BoundAxisInfo& bound_axis_info : bound_axes_info) {
        for (const double& bound : bound_axis_info.bounds) {
            if (bound != std::round(bound)) {
                throw std::invalid_argument(
                        "Axis wise bounds for integral number arrays must be intregral.");
            }
        }
    }
}

IntegerNode::IntegerNode(std::span<const ssize_t> shape,
                         std::optional<std::vector<double>> lower_bound,
                         std::optional<std::vector<double>> upper_bound,
                         std::optional<std::vector<BoundAxisInfo>> bound_axes)
        : NumberNode(shape,
                     lower_bound.has_value() ? std::move(*lower_bound)
                                             : std::vector<double>{default_lower_bound},
                     upper_bound.has_value() ? std::move(*upper_bound)
                                             : std::vector<double>{default_upper_bound},
                     std::move(bound_axes)) {
    if (min_ < minimum_lower_bound || max_ > maximum_upper_bound) {
        throw std::invalid_argument("range provided for integers exceeds supported range");
    }

    check_integrality_of_axis_wise_bounds(bound_axes_info_);
}

IntegerNode::IntegerNode(std::initializer_list<ssize_t> shape,
                         std::optional<std::vector<double>> lower_bound,
                         std::optional<std::vector<double>> upper_bound,
                         std::optional<std::vector<BoundAxisInfo>> bound_axes)
        : IntegerNode(std::span(shape), std::move(lower_bound), std::move(upper_bound),
                      std::move(bound_axes)) {}
IntegerNode::IntegerNode(ssize_t size, std::optional<std::vector<double>> lower_bound,
                         std::optional<std::vector<double>> upper_bound,
                         std::optional<std::vector<BoundAxisInfo>> bound_axes)
        : IntegerNode({size}, std::move(lower_bound), std::move(upper_bound),
                      std::move(bound_axes)) {}

IntegerNode::IntegerNode(std::span<const ssize_t> shape, double lower_bound,
                         std::optional<std::vector<double>> upper_bound,
                         std::optional<std::vector<BoundAxisInfo>> bound_axes)
        : IntegerNode(shape, std::vector<double>{lower_bound}, std::move(upper_bound),
                      std::move(bound_axes)) {}
IntegerNode::IntegerNode(std::initializer_list<ssize_t> shape, double lower_bound,
                         std::optional<std::vector<double>> upper_bound,
                         std::optional<std::vector<BoundAxisInfo>> bound_axes)
        : IntegerNode(std::span(shape), std::vector<double>{lower_bound}, std::move(upper_bound),
                      std::move(bound_axes)) {}
IntegerNode::IntegerNode(ssize_t size, double lower_bound,
                         std::optional<std::vector<double>> upper_bound,
                         std::optional<std::vector<BoundAxisInfo>> bound_axes)
        : IntegerNode({size}, std::vector<double>{lower_bound}, std::move(upper_bound),
                      std::move(bound_axes)) {}

IntegerNode::IntegerNode(std::span<const ssize_t> shape,
                         std::optional<std::vector<double>> lower_bound, double upper_bound,
                         std::optional<std::vector<BoundAxisInfo>> bound_axes)
        : IntegerNode(shape, std::move(lower_bound), std::vector<double>{upper_bound},
                      std::move(bound_axes)) {}
IntegerNode::IntegerNode(std::initializer_list<ssize_t> shape,
                         std::optional<std::vector<double>> lower_bound, double upper_bound,
                         std::optional<std::vector<BoundAxisInfo>> bound_axes)
        : IntegerNode(std::span(shape), std::move(lower_bound), std::vector<double>{upper_bound},
                      std::move(bound_axes)) {}
IntegerNode::IntegerNode(ssize_t size, std::optional<std::vector<double>> lower_bound,
                         double upper_bound, std::optional<std::vector<BoundAxisInfo>> bound_axes)
        : IntegerNode({size}, std::move(lower_bound), std::vector<double>{upper_bound},
                      std::move(bound_axes)) {}

IntegerNode::IntegerNode(std::span<const ssize_t> shape, double lower_bound, double upper_bound,
                         std::optional<std::vector<BoundAxisInfo>> bound_axes)
        : IntegerNode(shape, std::vector<double>{lower_bound}, std::vector<double>{upper_bound},
                      std::move(bound_axes)) {}
IntegerNode::IntegerNode(std::initializer_list<ssize_t> shape, double lower_bound,
                         double upper_bound, std::optional<std::vector<BoundAxisInfo>> bound_axes)
        : IntegerNode(std::span(shape), std::vector<double>{lower_bound},
                      std::vector<double>{upper_bound}, std::move(bound_axes)) {}
IntegerNode::IntegerNode(ssize_t size, double lower_bound, double upper_bound,
                         std::optional<std::vector<BoundAxisInfo>> bound_axes)
        : IntegerNode({size}, std::vector<double>{lower_bound}, std::vector<double>{upper_bound},
                      std::move(bound_axes)) {}

bool IntegerNode::integral() const { return true; }

bool IntegerNode::is_valid(ssize_t index, double value) const {
    return (value >= lower_bound(index)) && (value <= upper_bound(index)) &&
           (std::round(value) == value);
}

void IntegerNode::set_value(State& state, ssize_t index, double value) const {
    auto ptr = data_ptr<NumberNodeStateData>(state);
    // We expect `value` to obey the index-wise bounds and to be an integer.
    assert(lower_bound(index) <= value);
    assert(upper_bound(index) >= value);
    assert(value == std::round(value));
    // Assert that i is a valid index occurs in data_ptr->set().
    // set() occurs IFF `value` != buffer[i].
    if (ptr->set(index, value)) {
        // Update the bound axis.
        update_bound_axis_slice_sums(state, index, value - diff(state).back().old);
        assert(satisfies_axis_wise_bounds(bound_axes_info_, ptr->bound_axes_sums));
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
                       std::optional<std::vector<BoundAxisInfo>> bound_axes)
        : IntegerNode(shape, limit_bound_to_bool_domain<false>(lower_bound),
                      limit_bound_to_bool_domain<true>(upper_bound), bound_axes) {}

BinaryNode::BinaryNode(std::initializer_list<ssize_t> shape,
                       std::optional<std::vector<double>> lower_bound,
                       std::optional<std::vector<double>> upper_bound,
                       std::optional<std::vector<BoundAxisInfo>> bound_axes)
        : BinaryNode(std::span(shape), std::move(lower_bound), std::move(upper_bound),
                     std::move(bound_axes)) {}
BinaryNode::BinaryNode(ssize_t size, std::optional<std::vector<double>> lower_bound,
                       std::optional<std::vector<double>> upper_bound,
                       std::optional<std::vector<BoundAxisInfo>> bound_axes)
        : BinaryNode({size}, std::move(lower_bound), std::move(upper_bound),
                     std::move(bound_axes)) {}

BinaryNode::BinaryNode(std::span<const ssize_t> shape, double lower_bound,
                       std::optional<std::vector<double>> upper_bound,
                       std::optional<std::vector<BoundAxisInfo>> bound_axes)
        : BinaryNode(shape, std::vector<double>{lower_bound}, std::move(upper_bound),
                     std::move(bound_axes)) {}
BinaryNode::BinaryNode(std::initializer_list<ssize_t> shape, double lower_bound,
                       std::optional<std::vector<double>> upper_bound,
                       std::optional<std::vector<BoundAxisInfo>> bound_axes)
        : BinaryNode(std::span(shape), std::vector<double>{lower_bound}, std::move(upper_bound),
                     std::move(bound_axes)) {}
BinaryNode::BinaryNode(ssize_t size, double lower_bound,
                       std::optional<std::vector<double>> upper_bound,
                       std::optional<std::vector<BoundAxisInfo>> bound_axes)
        : BinaryNode({size}, std::vector<double>{lower_bound}, std::move(upper_bound),
                     std::move(bound_axes)) {}

BinaryNode::BinaryNode(std::span<const ssize_t> shape,
                       std::optional<std::vector<double>> lower_bound, double upper_bound,
                       std::optional<std::vector<BoundAxisInfo>> bound_axes)
        : BinaryNode(shape, std::move(lower_bound), std::vector<double>{upper_bound},
                     std::move(bound_axes)) {}
BinaryNode::BinaryNode(std::initializer_list<ssize_t> shape,
                       std::optional<std::vector<double>> lower_bound, double upper_bound,
                       std::optional<std::vector<BoundAxisInfo>> bound_axes)
        : BinaryNode(std::span(shape), std::move(lower_bound), std::vector<double>{upper_bound},
                     std::move(bound_axes)) {}
BinaryNode::BinaryNode(ssize_t size, std::optional<std::vector<double>> lower_bound,
                       double upper_bound, std::optional<std::vector<BoundAxisInfo>> bound_axes)
        : BinaryNode({size}, std::move(lower_bound), std::vector<double>{upper_bound},
                     std::move(bound_axes)) {}

BinaryNode::BinaryNode(std::span<const ssize_t> shape, double lower_bound, double upper_bound,
                       std::optional<std::vector<BoundAxisInfo>> bound_axes)
        : BinaryNode(shape, std::vector<double>{lower_bound}, std::vector<double>{upper_bound},
                     std::move(bound_axes)) {}
BinaryNode::BinaryNode(std::initializer_list<ssize_t> shape, double lower_bound, double upper_bound,
                       std::optional<std::vector<BoundAxisInfo>> bound_axes)
        : BinaryNode(std::span(shape), std::vector<double>{lower_bound},
                     std::vector<double>{upper_bound}, std::move(bound_axes)) {}
BinaryNode::BinaryNode(ssize_t size, double lower_bound, double upper_bound,
                       std::optional<std::vector<BoundAxisInfo>> bound_axes)
        : BinaryNode({size}, std::vector<double>{lower_bound}, std::vector<double>{upper_bound},
                     std::move(bound_axes)) {}

void BinaryNode::flip(State& state, ssize_t i) const {
    auto ptr = data_ptr<NumberNodeStateData>(state);
    // Variable should not be fixed.
    assert(lower_bound(i) != upper_bound(i));
    // Assert that i is a valid index occurs in ptr->set().
    // set() occurs IFF `value` != buffer[i].
    if (ptr->set(i, !ptr->get(i))) {
        // If value changed from 0 -> 1, update the bound axis sums by 1.
        // If value changed from 1 -> 0, update the bound axis sums by -1.
        update_bound_axis_slice_sums(state, i, (ptr->get(i) == 1) ? 1 : -1);
        assert(satisfies_axis_wise_bounds(bound_axes_info_, ptr->bound_axes_sums));
    }
}

void BinaryNode::set(State& state, ssize_t i) const {
    auto ptr = data_ptr<NumberNodeStateData>(state);
    // We expect the set to obey the index-wise bounds.
    assert(upper_bound(i) == 1.0);
    // Assert that i is a valid index occurs in ptr->set().
    // set() occurs IFF `value` != buffer[i].
    if (ptr->set(i, 1.0)) {
        // If value changed from 0 -> 1, update the bound axis sums by 1.
        update_bound_axis_slice_sums(state, i, 1.0);
        assert(satisfies_axis_wise_bounds(bound_axes_info_, ptr->bound_axes_sums));
    }
}

void BinaryNode::unset(State& state, ssize_t i) const {
    auto ptr = data_ptr<NumberNodeStateData>(state);
    // We expect the set to obey the index-wise bounds.
    assert(lower_bound(i) == 0.0);
    // Assert that i is a valid index occurs in ptr->set().
    // set occurs IFF `value` != buffer[i].
    if (ptr->set(i, 0.0)) {
        // If value changed from 1 -> 0, update the bound axis sums by -1.
        update_bound_axis_slice_sums(state, i, -1.0);
        assert(satisfies_axis_wise_bounds(bound_axes_info_, ptr->bound_axes_sums));
    }
}

}  // namespace dwave::optimization

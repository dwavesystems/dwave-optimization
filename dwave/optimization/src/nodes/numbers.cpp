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
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "_state.hpp"
#include "dwave-optimization/array.hpp"

namespace dwave::optimization {

BoundAxisInfo::BoundAxisInfo(ssize_t bound_axis, std::vector<BoundAxisOperator> axis_operators,
                             std::vector<double> axis_bounds)
        : axis(bound_axis), operators(std::move(axis_operators)), bounds(std::move(axis_bounds)) {
    const ssize_t num_operators = operators.size();
    const ssize_t num_bounds = bounds.size();

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
    if (bounds.size() == 1) return bounds[0];
    assert(slice < bounds.size());
    return bounds[slice];
}

BoundAxisOperator BoundAxisInfo::get_operator(const ssize_t slice) const {
    assert(0 <= slice);
    if (operators.size() == 1) return operators[0];
    assert(slice < operators.size());
    return operators[slice];
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

struct NumberNodeDataHelper_ {
    NumberNodeDataHelper_(std::vector<double> input, const std::span<const ssize_t>& shape,
                          const std::span<const ssize_t>& strides,
                          const std::vector<BoundAxisInfo>& bound_axes_info)
            : values(std::move(input)) {
        if (bound_axes_info.empty()) return;  // No axis sums to compute.
        compute_bound_axis_hyperslice_sums(shape, strides, bound_axes_info);
    }

    /// Variable assignment to NumberNode
    std::vector<double> values;
    /// For each bound axis and for each hyperslice along said axis, we track
    /// the sum of the values within the hyperslice.
    /// bound_axes_sums[i][j] = "sum of the values within the jth hyperslice along
    ///                          the ith bound axis"
    std::vector<std::vector<double>> bound_axes_sums;

    /// Determine the sum of the values of each hyperslice along each bound
    /// axis given the variable assignment of NumberNode.
    void compute_bound_axis_hyperslice_sums(const std::span<const ssize_t>& shape,
                                            const std::span<const ssize_t>& strides,
                                            const std::vector<BoundAxisInfo>& bound_axes_info) {
        const ssize_t num_bound_axes = bound_axes_info.size();
        bound_axes_sums.reserve(num_bound_axes);

        // For each variable assignment of NumberNode (stored in values), we
        // need to add the variables value to the running sum for each
        // hyperslice it is contained in (and that we are tracking). For each
        // such variable i and each bound axis j, we can identify which
        // hyperslice i lies in along j via `unravel_index(i, shape)[j]`.
        // However, this is inefficient. Instead we track the running
        // multidimensional index (for each bound axis we care about) and
        // adjust it based on the strides of the NumberNode array as
        // we iterate over the variable assignments of the NumberNode.
        //
        // To do this easily, we first compute the element strides from the
        // byte strides of the NumberNode array. Formally
        // element_strides[i] = "# of elements need get to the next hyperslice
        //                       along the ith bound axis"
        const ssize_t bytes_per_element = static_cast<ssize_t>(sizeof(double));
        std::vector<ssize_t> element_strides;
        element_strides.reserve(num_bound_axes);
        // A running stride counter for each bound axis.
        // When remaining_axis_strides[i] = 0, we have moved to the next
        // hyperslice along the ith bound axis.
        std::vector<ssize_t> remaining_axis_strides;
        remaining_axis_strides.reserve(num_bound_axes);

        // For each bound axis
        for (ssize_t i = 0; i < num_bound_axes; ++i) {
            const ssize_t bound_axis = bound_axes_info[i].axis;
            assert(0 <= bound_axis && bound_axis < shape.size());

            const ssize_t num_axis_slices = shape[bound_axis];
            // Initialize the sums for each hyperslice along the bound axis.
            bound_axes_sums.emplace_back(std::vector<double>(num_axis_slices, 0.0));

            // Update element stride data
            assert(strides[bound_axis] % bytes_per_element == 0);
            element_strides.emplace_back(strides[bound_axis] / bytes_per_element);
            // Initialize by the total # of element_strides along the bound axis
            remaining_axis_strides.push_back(element_strides[i]);
        }

        // Running hyperslice index per bound axis
        std::vector<ssize_t> hyperslice_index(num_bound_axes, 0);

        // Iterate over variable assignments of NumberNode.
        for (ssize_t i = 0, stop = static_cast<ssize_t>(values.size()); i < stop; ++i) {
            // Iterate over the bound axes.
            for (ssize_t j = 0; j < num_bound_axes; ++j) {
                const ssize_t bound_axis = bound_axes_info[j].axis;
                // Check the computation of the hyperslice
                assert(unravel_index(i, shape)[bound_axis] == hyperslice_index[j]);
                // Accumulate sum in hyperslice along jth bound axis
                bound_axes_sums[j][hyperslice_index[j]] += values[i];

                // Update running multidimensional index
                if (--remaining_axis_strides[j] == 0) {
                    // Moved to next hyperslice, reset `remaining_axis_strides`
                    remaining_axis_strides[j] = element_strides[j];

                    // Increment the multi_index along bound axis modulo the #
                    // of hyperslice along said axis
                    if (++hyperslice_index[j] == shape[bound_axis]) {
                        hyperslice_index[j] = 0;
                    }
                }
            }
        }
    }
};

// State dependant data attached to NumberNode

struct NumberNodeStateData : public ArrayNodeStateData {
    NumberNodeStateData(std::vector<double> input, const std::span<const ssize_t>& shape,
                        const std::span<const ssize_t>& strides,
                        const std::vector<BoundAxisInfo>& bound_axis_info)
            : NumberNodeStateData(
                      NumberNodeDataHelper_(std::move(input), shape, strides, bound_axis_info)) {}

    NumberNodeStateData(NumberNodeDataHelper_&& helper)
            : ArrayNodeStateData(std::move(helper.values)),
              bound_axes_sums(helper.bound_axes_sums),
              prior_bound_axes_sums(std::move(helper.bound_axes_sums)) {}

    std::vector<std::vector<double>> bound_axes_sums;
    // Store a copy for NumberNode::revert()
    std::vector<std::vector<double>> prior_bound_axes_sums;
};

/// Check the user defined axis-wise bounds for NumberNode
void check_axis_wise_bounds(const std::vector<BoundAxisInfo>& bound_axes_info,
                            const std::span<const ssize_t> shape) {
    if (bound_axes_info.size() == 0) {  // No bound axes to check.
        return;
    }

    // Used to asses if an axis have been bound multiple times.
    std::vector<bool> axis_bound(shape.size(), false);

    // For each set of bound axis data
    for (const BoundAxisInfo& bound_axis_info : bound_axes_info) {
        const ssize_t axis = bound_axis_info.axis;

        if (axis < 0 || axis >= shape.size()) {
            throw std::invalid_argument(
                    "Invalid bound axis: " + std::to_string(axis) +
                    ". Note, negative indexing is not supported for axis-wise bounds.");
        }

        // The number of operators defined for the given bound axis
        const ssize_t num_operators = bound_axis_info.operators.size();
        if ((num_operators > 1) && (num_operators != shape[axis])) {
            throw std::invalid_argument(
                    "Invalid number of axis-wise operators along axis: " + std::to_string(axis) +
                    " given axis shape: " + std::to_string(shape[axis]));
        }

        // The number of operators defined for the given bound axis
        const ssize_t num_bounds = bound_axis_info.bounds.size();
        if ((num_bounds > 1) && (num_bounds != shape[axis])) {
            throw std::invalid_argument(
                    "Invalid number of axis-wise bounds along axis: " + std::to_string(axis) +
                    " given axis shape: " + std::to_string(shape[axis]));
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

double const* NumberNode::buff(const State& state) const noexcept {
    return data_ptr<NumberNodeStateData>(state)->buff();
}

std::span<const Update> NumberNode::diff(const State& state) const noexcept {
    return data_ptr<NumberNodeStateData>(state)->diff();
}

double NumberNode::min() const { return min_; }

double NumberNode::max() const { return max_; }

void NumberNode::initialize_state(State& state, std::vector<double>&& number_data) const {
    if (number_data.size() != static_cast<size_t>(this->size())) {
        throw std::invalid_argument("Size of data provided does not match node size");
    }
    for (ssize_t index = 0, stop = this->size(); index < stop; ++index) {
        if (!is_valid(index, number_data[index])) {
            throw std::invalid_argument("Invalid data provided for node");
        }
    }

    emplace_data_ptr<NumberNodeStateData>(state, std::move(number_data), this->shape(),
                                          this->strides(), this->bound_axes_info_);

    if (!this->satisfies_axis_wise_bounds(state)) {
        throw std::invalid_argument("Initialized values do not satisfy axis-wise bounds.");
    }
}

void NumberNode::initialize_state(State& state) const {
    std::vector<double> values;
    values.reserve(this->size());
    for (ssize_t i = 0, stop = this->size(); i < stop; ++i) {
        values.push_back(default_value(i));
    }
    initialize_state(state, std::move(values));
}

void NumberNode::commit(State& state) const noexcept {
    auto node_data = data_ptr<NumberNodeStateData>(state);
    node_data->commit();
    // Manually store a copy of axis_sums
    node_data->prior_bound_axes_sums = node_data->bound_axes_sums;
}

void NumberNode::revert(State& state) const noexcept {
    auto node_data = data_ptr<NumberNodeStateData>(state);
    node_data->revert();
    // Manually reset axis_sums
    node_data->bound_axes_sums = node_data->prior_bound_axes_sums;
}

void NumberNode::exchange(State& state, ssize_t i, ssize_t j) const {
    auto ptr = data_ptr<ArrayNodeStateData>(state);
    // We expect the exchange to obey the index-wise bounds.
    assert(lower_bound(i) <= ptr->get(j));
    assert(upper_bound(i) >= ptr->get(j));
    assert(lower_bound(j) <= ptr->get(i));
    assert(upper_bound(j) >= ptr->get(i));
    // Assert that i and j are valid indices occurs in ptr->exchange().
    // Exchange occurs IFF (i != j) and (buffer[i] != buffer[j]).
    ptr->exchange(i, j);
}

double NumberNode::get_value(State& state, ssize_t i) const {
    return data_ptr<ArrayNodeStateData>(state)->get(i);
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

void NumberNode::clip_and_set_value(State& state, ssize_t index, double value) const {
    value = std::clamp(value, lower_bound(index), upper_bound(index));
    // Assert that i is a valid index occurs in data_ptr->set().
    // Set occurs IFF `value` != buffer[i] .
    data_ptr<ArrayNodeStateData>(state)->set(index, value);
}

ssize_t NumberNode::num_bound_axes() const {
    return static_cast<ssize_t>(bound_axes_info_.size());
};

const BoundAxisInfo* NumberNode::bound_axis_info(const ssize_t axis) const {
    assert(axis >= 0 && axis < bound_axes_info_.size());
    return &bound_axes_info_[axis];
};

ssize_t NumberNode::num_hyperslice_along_bound_axis(State& state, const ssize_t axis) const {
    assert(axis >= 0 && axis < data_ptr<NumberNodeStateData>(state)->bound_axes_sums.size());
    return data_ptr<NumberNodeStateData>(state)->bound_axes_sums[axis].size();
}

double NumberNode::bound_axis_hyperslice_sum(State& state, const ssize_t axis,
                                             const ssize_t slice) const {
    assert(axis >= 0 && slice >= 0);
    assert(axis < data_ptr<NumberNodeStateData>(state)->bound_axes_sums.size());
    assert(slice < data_ptr<NumberNodeStateData>(state)->bound_axes_sums[axis].size());
    return data_ptr<NumberNodeStateData>(state)->bound_axes_sums[axis][slice];
}

// /// Check whether the axis-wise bound is satisfied for the given hyperslice
// void check_hyperslice(const BoundAxisInfo& bound_axis_info, const ssize_t slice,
//                       const double slice_sum) {
//     const double rhs_bound = bound_axis_info.get_bound(slice);
//     std::cout << slice_sum;
//
//     switch (bound_axis_info.get_operator(slice)) {
//         case Equal:
//             std::cout << " == " << rhs_bound << std::endl;
//             if (slice_sum == rhs_bound) return;
//         case LessEqual:
//             std::cout << " <= " << rhs_bound << std::endl;
//             if (slice_sum <= rhs_bound) return;
//         case GreaterEqual:
//             std::cout << " >= " << rhs_bound << std::endl;
//             if (slice_sum >= rhs_bound) return;
//         default:
//             throw std::invalid_argument("Invalid axis-wise bound operator");
//     }
//
//     throw std::invalid_argument("Initialized state does not satisfy axis-wise bounds.");
// }

bool NumberNode::satisfies_axis_wise_bounds(State& state) const {
    const ssize_t num_bound_axes = this->num_bound_axes();
    if (num_bound_axes == 0) return true;  // No bounds to satisfy

    // Grab the hyperslice sums of all bound axes
    const std::vector<std::vector<double>>& bound_axes_sums =
            data_ptr<NumberNodeStateData>(state)->bound_axes_sums;
    assert(num_bound_axes == bound_axes_sums.size());

    for (ssize_t bound_axis = 0; bound_axis < num_bound_axes; ++bound_axis) {
        // Grab the stateless axis-wise bound data for the bound axis
        const BoundAxisInfo& bound_axis_info = this->bound_axes_info_[bound_axis];

        // Grab the sums of all hyperslices along the bound axis
        const std::vector<double>& bound_axis_sums = bound_axes_sums[bound_axis];

        // For each hyperslice along said axis
        for (ssize_t slice = 0, stop = bound_axis_sums.size(); slice < stop; ++slice) {
            const double rhs_bound = bound_axis_info.get_bound(slice);
            const double slice_sum = bound_axis_sums[slice];

            // Check whether the axis-wise bound is satisfied for the given hyperslice
            switch (bound_axis_info.get_operator(slice)) {
                case Equal:
                    if (slice_sum != rhs_bound) return false;
                    continue;
                case LessEqual:
                    if (slice_sum > rhs_bound) return false;
                    continue;
                case GreaterEqual:
                    if (slice_sum < rhs_bound) return false;
                    continue;
                default:
                    throw std::invalid_argument("Invalid axis-wise bound operator");
            }
        }
    }

    return true;
}

// Integer Node ***************************************************************

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
    // We expect `value` to obey the index-wise bounds and to be an integer.
    assert(lower_bound(index) <= value);
    assert(upper_bound(index) >= value);
    assert(value == std::round(value));
    // Assert that i is a valid index occurs in data_ptr->set().
    // Set occurs IFF `value` != buffer[i] .
    data_ptr<ArrayNodeStateData>(state)->set(index, value);
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
    auto ptr = data_ptr<ArrayNodeStateData>(state);
    // Variable should not be fixed.
    assert(lower_bound(i) != upper_bound(i));
    // Assert that i is a valid index occurs in ptr->set().
    // Set occurs IFF `value` != buffer[i] .
    ptr->set(i, !ptr->get(i));
}

void BinaryNode::set(State& state, ssize_t i) const {
    // We expect the set to obey the index-wise bounds.
    assert(upper_bound(i) == 1.0);
    // Assert that i is a valid index occurs in data_ptr->set().
    // Set occurs IFF `value` != buffer[i] .
    data_ptr<ArrayNodeStateData>(state)->set(i, 1.0);
}

void BinaryNode::unset(State& state, ssize_t i) const {
    // We expect the set to obey the index-wise bounds.
    assert(lower_bound(i) == 0.0);
    // Assert that i is a valid index occurs in data_ptr->set().
    // Set occurs IFF `value` != buffer[i] .
    data_ptr<ArrayNodeStateData>(state)->set(i, 0.0);
}

}  // namespace dwave::optimization

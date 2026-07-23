// Copyright 2024 D-Wave Inc.
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

#include "dwave-optimization/nodes/flow.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

#include "_state.hpp"

namespace dwave::optimization {

// todo: consider promoting this function to some public namespace.
// NOTE: this does not check that dynamic arrays have the same (dynamic) size
// (i.e. does not use sizeinfo()), only that their remaining static dimensions
// are equivalent.
std::span<const ssize_t> same_shape(
    const Array* node_ptr,
    std::convertible_to<const Array*> auto... node_ptrs
) {
    if (!node_ptr) throw std::invalid_argument("node pointer cannot be nullptr");

    // successively check that any remaining args have the same shape
    auto check_shape_equal = [&node_ptr](const Array* ptr) {
        if (!ptr) throw std::invalid_argument("node pointer cannot be nullptr");

        if (!std::ranges::equal(node_ptr->shape(), ptr->shape())) {
            throw std::invalid_argument("all arrays must have the same shape");
        }
    };
    (check_shape_equal(node_ptrs), ...);

    return node_ptr->shape();
}

/// ExtractNode

ExtractNode::ExtractNode(ArrayNode* condition_ptr, ArrayNode* arr_ptr) :
    ArrayOutputMixin({-1}),
    condition_ptr_(condition_ptr),
    arr_ptr_(arr_ptr),
    values_info_(arr_ptr_),
    sizeinfo_(this, 0, condition_ptr_->sizeinfo().max) {
    if (condition_ptr_->sizeinfo() != arr_ptr_->sizeinfo()) {
        throw std::invalid_argument("condition and arr must have the same size");
    }

    add_predecessor_(condition_ptr);
    add_predecessor_(arr_ptr);
}

double const* ExtractNode::buff(const State& state) const {
    return data_ptr_<ArrayNodeStateData>(state)->buff();
}

void ExtractNode::commit(State& state) const { data_ptr_<ArrayNodeStateData>(state)->commit(); }

std::span<const Update> ExtractNode::diff(const State& state) const {
    return data_ptr_<ArrayNodeStateData>(state)->diff();
}

void ExtractNode::initialize_state(State& state) const {
    const std::ranges::view auto condition = condition_ptr_->view(state);
    const std::ranges::view auto arr = arr_ptr_->view(state);

    std::vector<double> values;
    values.reserve(condition.size());

    for (
        auto cit = condition.begin(), arrit = arr.begin(); cit != std::default_sentinel;
        ++cit, ++arrit
    ) {
        if (*cit) {
            values.emplace_back(*arrit);
        }
    }

    emplace_data_ptr_<ArrayNodeStateData>(state, std::move(values));
}

bool ExtractNode::integral() const { return values_info_.integral; }

double ExtractNode::max() const { return values_info_.max; }

double ExtractNode::min() const { return values_info_.min; }

void ExtractNode::propagate(State& state) const {
    auto node_data = data_ptr_<ArrayNodeStateData>(state);

    // Nothing to do in this case
    if (condition_ptr_->diff(state).empty() && arr_ptr_->diff(state).empty()) return;

    auto get_index = [](const Update& update) { return update.index; };

    // Get the minimum changed index
    auto cond_view = condition_ptr_->diff(state) | std::views::transform(get_index);
    auto arr_view = arr_ptr_->diff(state) | std::views::transform(get_index);
    auto min_cond_it = std::ranges::min_element(cond_view);
    auto min_arr_it = std::ranges::min_element(arr_view);

    ssize_t min_changed_idx = -1;
    if (min_cond_it != cond_view.end() && min_arr_it != arr_view.end()) {
        min_changed_idx = std::min(*min_cond_it, *min_arr_it);
    } else if (min_cond_it != cond_view.end()) {
        min_changed_idx = *min_cond_it;
    } else if (min_arr_it != arr_view.end()) {
        min_changed_idx = *min_arr_it;
    }
    assert(min_changed_idx != -1 && "one of the arrays should have an update");

    const std::ranges::view auto condition = condition_ptr_->view(state);
    const std::ranges::view auto arr = arr_ptr_->view(state);

    // Count the trues before this index
    auto add_true = [](ssize_t acc, double val) -> ssize_t { return acc + static_cast<bool>(val); };
    ssize_t count =
        std::accumulate(condition.begin(), condition.begin() + min_changed_idx, 0, add_true);

    // Get the new values
    std::vector<double> new_values;
    for (
        auto cit = condition.begin() + min_changed_idx, arrit = arr.begin() + min_changed_idx;
        cit != std::default_sentinel;
        ++cit, ++arrit
    ) {
        if (*cit) new_values.push_back(*arrit);
    }

    node_data->assign(std::move(new_values), count);
}

void ExtractNode::replace_predecessor_(ssize_t index, Node* node_ptr) {
    Node::replace_predecessor_(index, node_ptr);

    if (index == 0) {
        condition_ptr_ = dynamic_cast<ArrayNode*>(node_ptr);
        assert(condition_ptr_ != nullptr);
    } else {
        assert(index == 1);
        arr_ptr_ = dynamic_cast<ArrayNode*>(node_ptr);
        assert(arr_ptr_ != nullptr);
    }
}

void ExtractNode::revert(State& state) const { data_ptr_<ArrayNodeStateData>(state)->revert(); }

std::span<const ssize_t> ExtractNode::shape(const State& state) const {
    return std::span(&data_ptr_<ArrayNodeStateData>(state)->size(), 1);
}

ssize_t ExtractNode::size(const State& state) const {
    return data_ptr_<ArrayNodeStateData>(state)->size();
}

ssize_t ExtractNode::size_diff(const State& state) const {
    return data_ptr_<ArrayNodeStateData>(state)->size_diff();
}

SizeInfo ExtractNode::sizeinfo() const { return this->sizeinfo_; }

/// ArgWhereNode

// Validate the predecessor and return the (static) output shape (num_nonzero, ndim).
std::vector<ssize_t> argwhere_shape(const ArrayNode* array_ptr) {
    if (!array_ptr) throw std::invalid_argument("node pointer cannot be nullptr");
    if (array_ptr->ndim() < 1) {
        throw std::invalid_argument(
            "cannot take the non-zero indices of a scalar (0d) array"
        );
    }
    // The number of non-zero elements is state-dependent, so axis 0 is dynamic.
    return {Array::DYNAMIC_SIZE, array_ptr->ndim()};
}

// The output holds one row of `ndim` indices per non-zero element, so its size
// is `ndim` times the number of non-zero elements, which ranges from 0 to the
// size of the predecessor.
SizeInfo argwhere_sizeinfo(const Array* self, const Array* array_ptr) {
    const std::optional<ssize_t> arr_max = array_ptr->sizeinfo().max;
    std::optional<ssize_t> max = std::nullopt;
    if (arr_max.has_value()) max = arr_max.value() * array_ptr->ndim();
    return SizeInfo(self, 0, max);
}

// The output values are indices into the predecessor. The smallest is 0. The
// largest is one less than the size of the predecessor's largest dimension.
std::pair<double, double> argwhere_minmax(const Array* array_ptr) {
    const std::span<const ssize_t> shape = array_ptr->shape();

    // the product of the fixed (non-axis-0) dimensions
    const ssize_t rest = std::reduce(shape.begin() + 1, shape.end(), 1, std::multiplies<ssize_t>());

    // the maximum possible length of the (possibly dynamic) axis 0
    ssize_t axis0;
    if (rest == 0) {
        axis0 = 0;  // the array is empty, so there are never any indices
    } else if (const std::optional<ssize_t> arr_max = array_ptr->sizeinfo().max;
               arr_max.has_value()) {
        axis0 = arr_max.value() / rest;
    } else {
        axis0 = std::numeric_limits<ssize_t>::max();
    }

    double max = static_cast<double>(axis0) - 1;
    for (const ssize_t& dim : shape | std::views::drop(1)) {
        max = std::max<double>(max, static_cast<double>(dim) - 1);
    }
    return {0.0, std::max<double>(0.0, max)};
}

// Append the multi-index of a single element - the C-order flat `index`
// unravelled according to `shape` - to the flattened output buffer.
void argwhere_emplace_multi_index(std::vector<double>& out, ssize_t index,
                                  std::span<const ssize_t> shape) {
    const ssize_t ndim = shape.size();
    const ssize_t offset = out.size();
    out.resize(offset + ndim);
    for (ssize_t axis = ndim - 1; axis >= 0; --axis) {
        // shape[axis] > 0 here: if any dimension were 0 the array would be
        // empty and this function would not be called.
        out[offset + axis] = index % shape[axis];
        index /= shape[axis];
    }
}

// The state holds the flattened (num_nonzero, ndim) index buffer as well as the
// state-dependent shape.
struct ArgWhereNodeData : ArrayNodeStateData {
    ArgWhereNodeData(std::vector<double>&& values, ssize_t ndim) noexcept :
            ArrayNodeStateData(std::move(values)), shape_{0, ndim} {
        update_shape();
    }

    std::unique_ptr<NodeStateData> copy() const override {
        return std::make_unique<ArgWhereNodeData>(*this);
    }

    // Recompute the (dynamic) number of rows from the current buffer size.
    void update_shape() { shape_[0] = this->size() / shape_[1]; }

    std::span<const ssize_t> shape() const {
        return std::span<const ssize_t>(shape_.data(), shape_.size());
    }

    // shape_[0] is the number of non-zero indices (dynamic), shape_[1] is the
    // (fixed) number of dimensions of the predecessor.
    std::array<ssize_t, 2> shape_;
};

ArgWhereNode::ArgWhereNode(ArrayNode* array_ptr) :
        ArrayOutputMixin(argwhere_shape(array_ptr)),
        array_ptr_(array_ptr),
        sizeinfo_(argwhere_sizeinfo(this, array_ptr)),
        minmax_(argwhere_minmax(array_ptr)) {
    add_predecessor_(array_ptr);
}

double const* ArgWhereNode::buff(const State& state) const {
    return data_ptr_<ArgWhereNodeData>(state)->buff();
}

void ArgWhereNode::commit(State& state) const { data_ptr_<ArgWhereNodeData>(state)->commit(); }

std::span<const Update> ArgWhereNode::diff(const State& state) const {
    return data_ptr_<ArgWhereNodeData>(state)->diff();
}

void ArgWhereNode::initialize_state(State& state) const {
    const std::span<const ssize_t> shape = array_ptr_->shape(state);
    const std::ranges::view auto arr = array_ptr_->view(state);

    std::vector<double> values;
    ssize_t index = 0;
    for (auto it = arr.begin(); it != std::default_sentinel; ++it, ++index) {
        if (static_cast<bool>(*it)) argwhere_emplace_multi_index(values, index, shape);
    }

    emplace_data_ptr_<ArgWhereNodeData>(state, std::move(values), array_ptr_->ndim());
}

bool ArgWhereNode::integral() const { return true; }

double ArgWhereNode::max() const { return minmax_.second; }

double ArgWhereNode::min() const { return minmax_.first; }

void ArgWhereNode::propagate(State& state) const {
    const std::span<const Update> arr_diff = array_ptr_->diff(state);
    if (arr_diff.empty()) return;

    auto node_data = data_ptr_<ArgWhereNodeData>(state);

    // Find the smallest flat index in the predecessor that changed. Every
    // element before it keeps both its truthiness and its (fixed) multi-index,
    // so the corresponding output rows are unchanged.
    const ssize_t min_changed = std::ranges::min(
        arr_diff | std::views::transform([](const Update& update) { return update.index; })
    );

    const std::span<const ssize_t> shape = array_ptr_->shape(state);
    const std::ranges::view auto arr = array_ptr_->view(state);
    const ssize_t ndim = array_ptr_->ndim();

    // Count the non-zero elements strictly before min_changed. Each contributes
    // one already-correct row of `ndim` values to the front of the buffer.
    auto is_nonzero = [](double value) { return static_cast<bool>(value); };
    const ssize_t count = std::count_if(arr.begin(), arr.begin() + min_changed, is_nonzero);

    // Recompute the rows for every element from min_changed onwards.
    std::vector<double> values;
    ssize_t index = min_changed;
    for (auto it = arr.begin() + min_changed; it != std::default_sentinel; ++it, ++index) {
        if (is_nonzero(*it)) argwhere_emplace_multi_index(values, index, shape);
    }

    node_data->assign(std::move(values), count * ndim);
    node_data->update_shape();
}

void ArgWhereNode::replace_predecessor_(ssize_t index, Node* node_ptr) {
    Node::replace_predecessor_(index, node_ptr);

    assert(index == 0);
    array_ptr_ = dynamic_cast<ArrayNode*>(node_ptr);
    assert(array_ptr_ != nullptr);
}

void ArgWhereNode::revert(State& state) const {
    auto node_data = data_ptr_<ArgWhereNodeData>(state);
    node_data->revert();
    node_data->update_shape();
}

std::span<const ssize_t> ArgWhereNode::shape(const State& state) const {
    return data_ptr_<ArgWhereNodeData>(state)->shape();
}

ssize_t ArgWhereNode::size(const State& state) const {
    return data_ptr_<ArgWhereNodeData>(state)->size();
}

ssize_t ArgWhereNode::size_diff(const State& state) const {
    return data_ptr_<ArgWhereNodeData>(state)->size_diff();
}

SizeInfo ArgWhereNode::sizeinfo() const { return sizeinfo_; }

/// WhereNode

struct WhereNodeData : ArrayNodeStateData {
    // Initialize the state with the values given
    explicit WhereNodeData(const std::ranges::view auto& values) noexcept :
        ArrayNodeStateData(std::vector<double>(values.begin(), values.begin() + values.size())) {}

    explicit WhereNodeData(std::vector<double>&& values) noexcept :
        ArrayNodeStateData(std::move(values)) {}

    // Update the buffer according to the given diffs
    void apply_diffs(
        std::ranges::view auto&& condition,
        std::span<const Update> condition_diff,
        std::ranges::view auto&& x,
        std::span<const Update> x_diff,
        std::ranges::view auto&& y,
        std::span<const Update> y_diff
    ) {
        // rather than doing a lot of fancy things to track the various changes, let's
        // just get the indices that have been updated in at least one predecessor and
        // recalculate those from scratch

        // We'll ignore any updates at indices greater than the final size
        ssize_t final_size = condition.size();
        auto relevant_index = [&final_size](const Update& up) { return up.index < final_size; };

        std::vector<ssize_t> indices;
        {
            indices.reserve(condition_diff.size() + x_diff.size() + y_diff.size());

            // dump any changed indices
            for (const Update& update : std::views::filter(condition_diff, relevant_index)) {
                indices.emplace_back(update.index);
            }
            for (const Update& update : std::views::filter(x_diff, relevant_index)) {
                indices.emplace_back(update.index);
            }
            for (const Update& update : std::views::filter(y_diff, relevant_index)) {
                indices.emplace_back(update.index);
            }

            // sort and de-duplicate
            std::sort(indices.begin(), indices.end());
            indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
        }

        ssize_t previous_index = 0;
        auto cit = condition.begin();
        auto xit = x.begin();
        auto yit = y.begin();
        for (const ssize_t& index : indices) {
            // advance all of the iterators to the given index
            {
                ssize_t distance = index - previous_index;
                cit += distance;
                xit += distance;
                yit += distance;
                previous_index = index;
            }

            const double& value = (*cit) ? *xit : *yit;  // value from x or y based on condition
            this->set(index, value, true);
        }

        this->trim_to(final_size);
    }
};

SizeInfo wherenode_calculate_sizeinfo(const Array* node_ptr, const Array* condition_ptr) {
    if (!node_ptr->dynamic()) return SizeInfo(node_ptr->size());

    // NOTE: could maybe do something with the min/max of x and y?
    if (condition_ptr->size() == 1) return SizeInfo(node_ptr);

    // all three predecessor arrays should be the same (dynamic) size
    return condition_ptr->sizeinfo();
}

WhereNode::WhereNode(ArrayNode* condition_ptr, ArrayNode* x_ptr, ArrayNode* y_ptr) :
    ArrayOutputMixin(same_shape(x_ptr, y_ptr)),
    condition_ptr_(condition_ptr),
    x_ptr_(x_ptr),
    y_ptr_(y_ptr),
    values_info_({x_ptr, y_ptr}),
    sizeinfo_(wherenode_calculate_sizeinfo(this, condition_ptr_)) {
    // x and y where checked for nullptr by same_shape() above
    if (!condition_ptr_) throw std::invalid_argument("node pointer cannot be nullptr");

    // If `condition` is a single number, then we broadcast to x/y and don't care about their shapes
    // Otherwise, we need all to be the same shape and size
    if (condition_ptr_->size() != 1) {
        same_shape(condition_ptr_, x_ptr_);  // x/y have already been checked
        SizeInfo cond_size = condition_ptr_->sizeinfo().substitute(100);
        if (cond_size != x_ptr_->sizeinfo().substitute(100) ||
            cond_size != y_ptr->sizeinfo().substitute(100)) {
            throw std::invalid_argument(
                "If condition is not of size 1, condition, x and y must all be the same size"
            );
        }
    }

    add_predecessor_(condition_ptr);
    add_predecessor_(x_ptr);
    add_predecessor_(y_ptr);
}

double const* WhereNode::buff(const State& state) const {
    return data_ptr_<WhereNodeData>(state)->buff();
}

void WhereNode::commit(State& state) const { data_ptr_<WhereNodeData>(state)->commit(); }

std::span<const Update> WhereNode::diff(const State& state) const {
    return data_ptr_<WhereNodeData>(state)->diff();
}

void WhereNode::initialize_state(State& state) const {
    if (condition_ptr_->size() != 1) {
        // `condition` has the same shape as x/y and isn't a single value
        const std::ranges::view auto condition = condition_ptr_->view(state);
        const std::ranges::view auto x = x_ptr_->view(state);
        const std::ranges::view auto y = y_ptr_->view(state);

        std::vector<double> values;
        values.reserve(condition.size());

        // zip would be very nice here...
        for (
            auto cit = condition.begin(), xit = x.begin(), yit = y.begin();
            cit != std::default_sentinel;
            ++cit, ++xit, ++yit
        ) {
            values.emplace_back((*cit) ? *xit : *yit);
        }

        emplace_data_ptr_<WhereNodeData>(state, std::move(values));
    } else if (condition_ptr_->buff(state)[0]) {
        // `condition` is a single value and is currently selecting x
        // so our state is just a copy of x's
        emplace_data_ptr_<WhereNodeData>(state, x_ptr_->view(state));
    } else {
        // `condition` is a single value and is currently selecting y
        // so our state is just a copy of y's
        emplace_data_ptr_<WhereNodeData>(state, y_ptr_->view(state));
    }
}

bool WhereNode::integral() const { return values_info_.integral; }

double WhereNode::max() const { return values_info_.max; }

double WhereNode::min() const { return values_info_.min; }

// Given a list of updates on a single `conditional`, did we end up flipping?
bool _flipped(std::span<const Update> diff) {
    bool flip = false;
    for (const auto& [index, old, new_] : diff) {
        assert(index == 0);  // should all be on a single value
        if (static_cast<bool>(old) != static_cast<bool>(new_)) flip = !flip;
    }
    return flip;
}

void WhereNode::propagate(State& state) const {
    auto node_data = data_ptr_<WhereNodeData>(state);

    if (condition_ptr_->size() != 1) {
        // `condition` is an array

        node_data->apply_diffs(
            condition_ptr_->view(state),
            condition_ptr_->diff(state),
            x_ptr_->view(state),
            x_ptr_->diff(state),
            y_ptr_->view(state),
            y_ptr_->diff(state)
        );

    } else if (_flipped(condition_ptr_->diff(state))) {
        // `condition` is a single value and it changed
        // so let's just assume we're updating everything in our buffer

        if (condition_ptr_->buff(state)[0]) {
            // we're now pointing to x
            node_data->assign(x_ptr_->view(state));
        } else {
            // we're now pointing to y
            node_data->assign(y_ptr_->view(state));
        }

    } else {
        // `condition` is a single value and it didn't change
        // so we update ourselves to match the relevant predecessor

        if (condition_ptr_->buff(state)[0]) {
            // we're pointing to x, so update ourselves according to x
            node_data->update(x_ptr_->diff(state));
        } else {
            // we're pointing to y, so update ourselves according to y
            node_data->update(y_ptr_->diff(state));
        }
    }
}

void WhereNode::replace_predecessor_(ssize_t index, Node* node_ptr) {
    Node::replace_predecessor_(index, node_ptr);

    if (index == 0) {
        condition_ptr_ = dynamic_cast<ArrayNode*>(node_ptr);
        assert(condition_ptr_ != nullptr);
    } else if (index == 1) {
        x_ptr_ = dynamic_cast<ArrayNode*>(node_ptr);
        assert(x_ptr_ != nullptr);
    } else {
        assert(index == 2);
        y_ptr_ = dynamic_cast<ArrayNode*>(node_ptr);
        assert(y_ptr_ != nullptr);
    }
}

void WhereNode::revert(State& state) const { data_ptr_<WhereNodeData>(state)->revert(); }

std::span<const ssize_t> WhereNode::shape(const State& state) const {
    if (!this->dynamic()) return this->shape();

    // in which case our shape is determined by `condition`
    if (condition_ptr_->size() == 1) {
        // in this case our shape is determined by `condition`
        return condition_ptr_->buff(state)[0] ? x_ptr_->shape(state) : y_ptr_->shape(state);
    }

    return condition_ptr_->shape(state);
}

ssize_t WhereNode::size(const State& state) const {
    ssize_t size = this->size();
    if (size >= 0) return size;

    if (condition_ptr_->size() == 1) {
        // in this case our size is determined by `condition`
        size = condition_ptr_->buff(state)[0] ? x_ptr_->size(state) : y_ptr_->size(state);
    } else {
        // all the arrays should have equal dynamic size
        size = condition_ptr_->size(state);
    }

    // should match our current buffer
    assert(static_cast<ssize_t>(data_ptr_<WhereNodeData>(state)->size()) == size);

    return size;
}

ssize_t WhereNode::size_diff(const State& state) const {
    if (!this->dynamic()) return 0;
    return data_ptr_<WhereNodeData>(state)->size_diff();
}

SizeInfo WhereNode::sizeinfo() const { return this->sizeinfo_; }

}  // namespace dwave::optimization

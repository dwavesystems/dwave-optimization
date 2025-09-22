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
#include <optional>
#include <ranges>
#include <span>
#include <vector>

#include "_state.hpp"

namespace dwave::optimization {

// todo: consider promoting this function to some public namespace.
// NOTE: this does not check that dynamic arrays have the same (dynamic) size
// (i.e. does not use sizeinfo()), only that their remaining static dimensions
// are equivalent.
std::span<const ssize_t> same_shape(const Array* node_ptr,
                                    std::convertible_to<const Array*> auto... node_ptrs) {
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

ExtractNode::ExtractNode(ArrayNode* condition_ptr, ArrayNode* arr_ptr)
        : ArrayOutputMixin({-1}),
          condition_ptr_(condition_ptr),
          arr_ptr_(arr_ptr),
          values_info_(arr_ptr_),
          sizeinfo_(this, 0, condition_ptr_->sizeinfo().max) {
    if (condition_ptr_->sizeinfo() != arr_ptr_->sizeinfo()) {
        throw std::invalid_argument("condition and arr must have the same size");
    }

    add_predecessor(condition_ptr);
    add_predecessor(arr_ptr);
}

double const* ExtractNode::buff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->buff();
}

void ExtractNode::commit(State& state) const { data_ptr<ArrayNodeStateData>(state)->commit(); }

std::span<const Update> ExtractNode::diff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->diff();
}

void ExtractNode::initialize_state(State& state) const {
    const Array::View condition = condition_ptr_->view(state);
    const Array::View arr = arr_ptr_->view(state);

    std::vector<double> values;
    values.reserve(condition.size());

    for (auto cit = condition.begin(), arrit = arr.begin(), end = condition.end(); cit != end;
         ++cit, ++arrit) {
        if (*cit) {
            values.emplace_back(*arrit);
        }
    }

    emplace_data_ptr<ArrayNodeStateData>(state, std::move(values));
}

bool ExtractNode::integral() const { return values_info_.integral; }

double ExtractNode::max() const { return values_info_.max; }

double ExtractNode::min() const { return values_info_.min; }

void ExtractNode::propagate(State& state) const {
    auto node_data = data_ptr<ArrayNodeStateData>(state);

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

    const Array::View condition = condition_ptr_->view(state);
    const Array::View arr = arr_ptr_->view(state);

    // Count the trues before this index
    auto add_true = [](ssize_t acc, double val) -> ssize_t { return acc + static_cast<bool>(val); };
    ssize_t count =
            std::accumulate(condition.begin(), condition.begin() + min_changed_idx, 0, add_true);

    // Get the new values
    std::vector<double> new_values;
    for (auto cit = condition.begin() + min_changed_idx, arrit = arr.begin() + min_changed_idx,
              end = condition.end();
         cit != end; ++cit, ++arrit) {
        if (*cit) new_values.push_back(*arrit);
    }

    node_data->assign(new_values, count);
}

void ExtractNode::revert(State& state) const { data_ptr<ArrayNodeStateData>(state)->revert(); }

std::span<const ssize_t> ExtractNode::shape(const State& state) const {
    return std::span(&data_ptr<ArrayNodeStateData>(state)->size(), 1);
}

ssize_t ExtractNode::size(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->size();
}

ssize_t ExtractNode::size_diff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->size_diff();
}

SizeInfo ExtractNode::sizeinfo() const { return this->sizeinfo_; }

/// WhereNode

struct WhereNodeData : ArrayNodeStateData {
    // Initialize the state with the values given
    explicit WhereNodeData(const Array::View values) noexcept
            : ArrayNodeStateData(std::vector<double>(values.begin(), values.end())) {}

    explicit WhereNodeData(std::vector<double>&& values) noexcept
            : ArrayNodeStateData(std::move(values)) {}

    // Update the buffer according to the given diffs
    void apply_diffs(const Array::View condition, std::span<const Update> condition_diff,
                     const Array::View x, std::span<const Update> x_diff, const Array::View y,
                     std::span<const Update> y_diff) {
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

WhereNode::WhereNode(ArrayNode* condition_ptr, ArrayNode* x_ptr, ArrayNode* y_ptr)
        : ArrayOutputMixin(same_shape(x_ptr, y_ptr)),
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
                    "If condition is not of size 1, condition, x and y must all be the same size");
        }
    }

    add_predecessor(condition_ptr);
    add_predecessor(x_ptr);
    add_predecessor(y_ptr);
}

double const* WhereNode::buff(const State& state) const {
    return data_ptr<WhereNodeData>(state)->buff();
}

void WhereNode::commit(State& state) const { data_ptr<WhereNodeData>(state)->commit(); }

std::span<const Update> WhereNode::diff(const State& state) const {
    return data_ptr<WhereNodeData>(state)->diff();
}

void WhereNode::initialize_state(State& state) const {
    if (condition_ptr_->size() != 1) {
        // `condition` has the same shape as x/y and isn't a single value
        const Array::View condition = condition_ptr_->view(state);
        const Array::View x = x_ptr_->view(state);
        const Array::View y = y_ptr_->view(state);

        std::vector<double> values;
        values.reserve(condition.size());

        // zip would be very nice here...
        for (auto cit = condition.begin(), xit = x.begin(), yit = y.begin(), end = condition.end();
             cit != end; ++cit, ++xit, ++yit) {
            values.emplace_back((*cit) ? *xit : *yit);
        }

        emplace_data_ptr<WhereNodeData>(state, std::move(values));
    } else if (condition_ptr_->buff(state)[0]) {
        // `condition` is a single value and is currently selecting x
        // so our state is just a copy of x's
        emplace_data_ptr<WhereNodeData>(state, x_ptr_->view(state));
    } else {
        // `condition` is a single value and is currently selecting y
        // so our state is just a copy of y's
        emplace_data_ptr<WhereNodeData>(state, y_ptr_->view(state));
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
    auto node_data = data_ptr<WhereNodeData>(state);

    if (condition_ptr_->size() != 1) {
        // `condition` is an array

        node_data->apply_diffs(condition_ptr_->view(state), condition_ptr_->diff(state),
                               x_ptr_->view(state), x_ptr_->diff(state), y_ptr_->view(state),
                               y_ptr_->diff(state));

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

void WhereNode::revert(State& state) const { data_ptr<WhereNodeData>(state)->revert(); }

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
    assert(static_cast<ssize_t>(data_ptr<WhereNodeData>(state)->size()) == size);

    return size;
}

ssize_t WhereNode::size_diff(const State& state) const {
    if (!this->dynamic()) return 0;
    return data_ptr<WhereNodeData>(state)->size_diff();
}

SizeInfo WhereNode::sizeinfo() const { return this->sizeinfo_; }

}  // namespace dwave::optimization

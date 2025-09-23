// Copyright 2025 D-Wave
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

#include "dwave-optimization/nodes/sorting.hpp"

#include <set>

#include "_state.hpp"

namespace dwave::optimization {

/// ArgSortNode

struct ArgSortNodeDataHelper_ {
    ArgSortNodeDataHelper_(std::vector<double> values) {
        for (ssize_t index = 0, stop = values.size(); index < stop; index++) {
            order.emplace(values[index], index);
        }

        for (const auto& [_, index] : order) {
            indices.push_back(index);
        }
    }

    std::vector<double> indices;
    std::set<std::pair<double, ssize_t>> order;
};

struct ArgSortNodeData : public ArrayNodeStateData {
    ArgSortNodeData(std::vector<double> values)
            : ArgSortNodeData(ArgSortNodeDataHelper_(std::move(values))) {}
    ArgSortNodeData(ArgSortNodeDataHelper_&& helper)
            : ArrayNodeStateData(std::move(helper.indices)), order(std::move(helper.order)) {}

    /// Pairs are <value in the original array, index of the value>
    std::set<std::pair<double, ssize_t>> order;
    std::vector<Update> predecessor_updates;
};

ArgSortNode::ArgSortNode(ArrayNode* arr_ptr)
        : ArrayOutputMixin(arr_ptr->shape()),
          arr_ptr_(arr_ptr),
          minmax_(0, static_cast<double>(arr_ptr_->sizeinfo().max.value_or(
                                                 std::numeric_limits<ssize_t>::max()) -
                                         1)),
          sizeinfo_(arr_ptr_->sizeinfo()) {
    add_predecessor(arr_ptr);
}

double const* ArgSortNode::buff(const State& state) const {
    return data_ptr<ArgSortNodeData>(state)->buff();
}

void ArgSortNode::commit(State& state) const { data_ptr<ArgSortNodeData>(state)->commit(); }

std::span<const Update> ArgSortNode::diff(const State& state) const {
    return data_ptr<ArgSortNodeData>(state)->diff();
}

void ArgSortNode::initialize_state(State& state) const {
    const Array::View arr = arr_ptr_->view(state);

    emplace_data_ptr<ArgSortNodeData>(state, std::vector<double>{arr.begin(), arr.end()});
}

bool ArgSortNode::integral() const { return true; }

double ArgSortNode::min() const { return minmax_.first; }

double ArgSortNode::max() const { return minmax_.second; }

void ArgSortNode::propagate(State& state) const {
    auto node_data = data_ptr<ArgSortNodeData>(state);

    auto pred_diff = arr_ptr_->diff(state);

    // Save a copy of the predecessor's updates so we can use them in case we
    // need to revert the changes to the ordering
    node_data->predecessor_updates.assign(pred_diff.begin(), pred_diff.end());

    // Make the modifications to the std::set based on the updates.
    for (const Update& update : pred_diff) {
        if (!update.placed()) {
            node_data->order.erase(std::make_pair(update.old, update.index));
        }
        if (!update.removed()) {
            node_data->order.insert(std::make_pair(update.value, update.index));
        }
    }

    // Assign the new order as determined by the ordering of the std::set.
    // A further optimization could be to track the earliest modified (final) index
    // and only assign from there which might help when all the updates only affect
    // the end region of the final ordering.
    node_data->assign(
            node_data->order |
            std::views::transform([](const std::pair<double, ssize_t>& p) { return p.second; }));
}

void ArgSortNode::revert(State& state) const {
    auto node_data = data_ptr<ArgSortNodeData>(state);

    // Revert the changes to `order` by going over the predecessor's previous updates in reverse
    for (const Update& update : node_data->predecessor_updates | std::views::reverse) {
        if (!update.placed()) {
            node_data->order.insert(std::make_pair(update.old, update.index));
        }
        if (!update.removed()) {
            node_data->order.erase(std::make_pair(update.value, update.index));
        }
    }

    node_data->predecessor_updates.clear();
    node_data->revert();
}

std::span<const ssize_t> ArgSortNode::shape(const State& state) const {
    return arr_ptr_->shape(state);
}

ssize_t ArgSortNode::size(const State& state) const {
    return data_ptr<ArgSortNodeData>(state)->size();
}

ssize_t ArgSortNode::size_diff(const State& state) const {
    return data_ptr<ArgSortNodeData>(state)->size_diff();
}

SizeInfo ArgSortNode::sizeinfo() const { return this->sizeinfo_; }

}  // namespace dwave::optimization

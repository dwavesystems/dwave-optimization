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

#include "dwave-optimization/nodes/set_routines.hpp"

#include <unordered_set>

#include "_state.hpp"

namespace dwave::optimization {

/// IsInNode
struct IsInNodeDataHelper_ {
    IsInNodeDataHelper_(std::vector<double> element, std::vector<double> test_elements) {
        for (const double val : test_elements) {
            test_elements_counter[val] += 1;
        }
        for (ssize_t index = 0, stop = element.size(); index < stop; index++) {
            element_isin.emplace_back(test_elements_counter.contains(element[index]));
            element_indices[element[index]].insert(index);
        }
    }
    // # of test_elements per value
    std::unordered_map<double, ssize_t> test_elements_counter;
    // all indices of elements with a given value
    std::unordered_map<double, std::unordered_set<ssize_t>> element_indices;
    // Indicator vector: element_isin[i] == true iff element[i] is in test_elements
    std::vector<ssize_t> element_isin;
};

struct IsInNodeData : public ArrayNodeStateData {
    IsInNodeData(std::vector<double> element, std::vector<double> test_elements)
            : IsInNodeData(IsInNodeDataHelper_(std::move(element), std::move(test_elements))) {}
    IsInNodeData(IsInNodeDataHelper_&& helper)
            : ArrayNodeStateData(std::move(helper.element_isin)),
              test_elements_counter(std::move(helper.test_elements_counter)),
              element_indices(std::move(helper.element_indices)) {}

    std::unordered_map<double, ssize_t> test_elements_counter;
    std::unordered_map<double, std::unordered_set<ssize_t>> element_indices;
    std::vector<Update> element_updates;
    std::vector<Update> test_element_updates;
};

IsInNode::IsInNode(ArrayNode* element_ptr, ArrayNode* test_elements_ptr)
        : ArrayOutputMixin(element_ptr->shape()),
          element_ptr_(element_ptr),
          test_elements_ptr_(test_elements_ptr) {
    add_predecessor(element_ptr);
    add_predecessor(test_elements_ptr);
}

double const* IsInNode::buff(const State& state) const {
    return data_ptr<IsInNodeData>(state)->buff();
}

void IsInNode::commit(State& state) const { data_ptr<IsInNodeData>(state)->commit(); }

std::span<const Update> IsInNode::diff(const State& state) const {
    return data_ptr<IsInNodeData>(state)->diff();
}

void IsInNode::initialize_state(State& state) const {
    const Array::View element = element_ptr_->view(state);
    const Array::View test_elements = test_elements_ptr_->view(state);

    emplace_data_ptr<IsInNodeData>(state, std::vector<double>{element.begin(), element.end()},
                                   std::vector<double>{test_elements.begin(), test_elements.end()});
}

bool IsInNode::integral() const { return true; }  // all values are True/False

std::pair<double, double> IsInNode::minmax(
        optional_cache_type<std::pair<double, double>> cache) const {
    return memoize(cache, [&]() {
        return std::make_pair(0.0, 1.0);  // all values are True/False
    });
}

void IsInNode::propagate(State& state) const {
    auto test_elements_diff = test_elements_ptr_->diff(state);
    auto element_diff = element_ptr_->diff(state);

    if (test_elements_diff.empty() && element_diff.empty()) {
        return;  // nothing to do
    }

    // Current node data
    auto node_data = data_ptr<IsInNodeData>(state);
    auto& element_indices = node_data->element_indices;
    auto& test_elements_counter = node_data->test_elements_counter;

    // Used to track if elements are added/removed from test_elements
    std::unordered_set<double> del_test_elements;
    std::unordered_set<double> add_test_elements;

    // Save a copy of updates so we can use them in case we need to revert
    node_data->test_element_updates.assign(test_elements_diff.begin(), test_elements_diff.end());
    node_data->element_updates.assign(element_diff.begin(), element_diff.end());

    // Handle changes to `test_elements`
    for (const Update& update : test_elements_diff) {
        if (!update.placed()) {  // i.e. changed or removed
            test_elements_counter[update.old] -= 1;
            // Deleting `update.old` from test_elements results in no index of
            // test_elements having the value `update.old`. Record it to update
            // the containment of values from element later.
            if (test_elements_counter[update.old] == 0) {
                del_test_elements.insert(update.old);
                // This catches the case of a user changing the same index of
                // test_elements multiple times.
                if (add_test_elements.contains(update.old)) {
                    add_test_elements.erase(update.old);
                }
            }
        }
        if (!update.removed()) {  // i.e. changed or added
            test_elements_counter[update.value] += 1;
            // Previously, `update.value` did not occur in test_elements.
            // Therefore, we record it to update the containment of values from
            // element later.
            if (test_elements_counter[update.value] == 1) {
                add_test_elements.insert(update.value);
                // This catches the case of a user changing the same index of
                // test_elements multiple times.
                if (del_test_elements.contains(update.value)) {
                    del_test_elements.erase(update.value);
                }
            }
        }
    }

    // Handle changes to `element`
    for (const Update& update : element_diff) {
        if (!update.placed()) {  // i.e. changed or removed
            element_indices[update.old].erase(update.index);
            // Remove key if set is empty to reduce bloating
            if (element_indices[update.old].empty()) {
                element_indices.erase(update.old);
            }
        }
        if (!update.removed()) {  // i.e. changed or added
            element_indices[update.value].insert(update.index);
            node_data->set(update.index, test_elements_counter[update.value], true);
        }
    }

    // Handle del_test_elements and add_test_elements
    for (const double& key : del_test_elements) {
        // Each index of `element` with the value `key` is no longer in `test_elements`
        if (element_indices.contains(key)) {
            for (const auto index : element_indices[key]) {
                node_data->set(index, 0.0);
            }
        }
        // Remove key to reduce bloating
        test_elements_counter.erase(key);
    }
    for (const double& key : add_test_elements) {
        // Each index of `element` with the value `key` is now in `test_elements`
        if (element_indices.contains(key)) {
            for (const auto index : element_indices[key]) {
                node_data->set(index, 1.0);
            }
        }
    }

    // Resize if necessary
    if (element_ptr_->size_diff(state) != 0) {
        node_data->trim_to(element_ptr_->size(state));
    }
}

void IsInNode::revert(State& state) const {
    auto node_data = data_ptr<IsInNodeData>(state);
    auto& element_indices = node_data->element_indices;
    auto& test_elements_counter = node_data->test_elements_counter;

    // Revert the changes to `test_elements`
    for (const Update& update : node_data->test_element_updates | std::views::reverse) {
        if (!update.placed()) {  // i.e. removed or changed
            test_elements_counter[update.old] += 1;
        }
        if (!update.removed()) {  // i.e. placed or changed
            test_elements_counter[update.value] -= 1;
            // Remove key to reduce bloating
            if (test_elements_counter[update.value] == 0) {
                test_elements_counter.erase(update.value);
            }
        }
    }

    // Revert the changes to `element`
    for (const Update& update : node_data->element_updates | std::views::reverse) {
        if (!update.placed()) {  // i.e. removed or changed
            element_indices[update.old].insert(update.index);
        }
        if (!update.removed()) {  // i.e. placed or changed
            element_indices[update.value].erase(update.index);
            // Remove key if set is empty to reduce bloating
            if (element_indices[update.value].empty()) {
                element_indices.erase(update.value);
            }
        }
    }

    node_data->element_updates.clear();
    node_data->test_element_updates.clear();
    node_data->revert();
}

std::span<const ssize_t> IsInNode::shape(const State& state) const {
    return element_ptr_->shape(state);
}

ssize_t IsInNode::size(const State& state) const { return data_ptr<IsInNodeData>(state)->size(); }

ssize_t IsInNode::size_diff(const State& state) const {
    return data_ptr<IsInNodeData>(state)->size_diff();
}

SizeInfo IsInNode::sizeinfo() const { return element_ptr_->sizeinfo(); }

}  // namespace dwave::optimization

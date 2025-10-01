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

#include <vector>

#include "_state.hpp"

namespace dwave::optimization {

// IsInNode *******************************************************************
struct IsInNodeSetData {
    IsInNodeSetData() = default;

    // # of indices from `test_elements` with a given value
    ssize_t test_elements_count;  // default is 0
    // All indices of `element` with a given value
    std::vector<ssize_t> element_indices;  // default is empty vector
};

struct IsInNodeDataHelper_ {
    IsInNodeDataHelper_(std::vector<double> element, std::vector<double> test_elements) {
        element_isin.reserve(element.size());

        for (const double& val : test_elements) {
            set_data[val].test_elements_count += 1;
        }
        for (ssize_t index = 0, stop = element.size(); index < stop; ++index) {
            IsInNodeSetData& set_data_struct = set_data[element[index]];
            set_data_struct.element_indices.push_back(index);
            element_isin.emplace_back(set_data_struct.test_elements_count > 0);
        }
    }

    std::unordered_map<double, IsInNodeSetData> set_data;
    // element_isin[i] == true iff `element[i]` is in`test_elements`
    std::vector<bool> element_isin;
};

struct IsInNodeData : public ArrayNodeStateData {
    IsInNodeData(std::vector<double> element, std::vector<double> test_elements)
            : IsInNodeData(IsInNodeDataHelper_(std::move(element), std::move(test_elements))) {}
    IsInNodeData(IsInNodeDataHelper_&& helper)
            : ArrayNodeStateData(std::move(helper.element_isin)),
              set_data(std::move(helper.set_data)) {}

    // Used to track if elements are added/removed from `test_elements`
    // during IsInNode::propagate()
    std::unordered_map<double, bool> add_or_rm_test_elements;
    // For each value (key) in `element` and `test_elements`, store a
    // IsInNodeSetData struct.
    std::unordered_map<double, IsInNodeSetData> set_data;
    // Needed for IsInNode::revert()
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

bool IsInNode::integral() const { return true; }  // All values are true/false

double IsInNode::min() const { return 0.0; }  // All values are true/false

double IsInNode::max() const { return 1.0; }  // All values are true/false

inline void rm_index(IsInNodeData*& node_data, const ssize_t index, const double key) {
    auto set_data_it = node_data->set_data.find(key);
    // find() should be successful since we need to remove the index
    assert(set_data_it != node_data->set_data.end());
    // Vector containing indices of `element` with value `key`
    std::vector<ssize_t>& indices_vec = set_data_it->second.element_indices;
    // Find index of `indices_vec` equal to `rm_index`
    auto index_ptr = std::find(indices_vec.begin(), indices_vec.end(), index);
    // find() should be successful since we need to remove the index
    assert(index_ptr != indices_vec.end());
    // swap and pop
    *index_ptr = indices_vec.back();
    indices_vec.pop_back();

    // Reduce bloating
    if (indices_vec.empty() && set_data_it->second.test_elements_count == 0) {
        node_data->set_data.erase(set_data_it);
    }
}

void IsInNode::propagate(State& state) const {
    const std::span<const Update> test_elements_diff = test_elements_ptr_->diff(state);
    const std::span<const Update> element_diff = element_ptr_->diff(state);

    if (test_elements_diff.empty() && element_diff.empty()) {
        return;  // Nothing to do
    }

    IsInNodeData* node_data = data_ptr<IsInNodeData>(state);

    // Save a copy of the updates for IsInNode::revert()
    node_data->test_element_updates.assign(test_elements_diff.begin(), test_elements_diff.end());
    node_data->element_updates.assign(element_diff.begin(), element_diff.end());

    // Handle changes to `test_elements`
    for (const Update& update : test_elements_diff) {
        if (!update.placed()) {
            // i.e. changed or removed.
            ssize_t& count = node_data->set_data[update.old].test_elements_count;
            assert(count > 0 && "update.old should be in test_elements_counter");
            count -= 1;

            if (count == 0) {
                // Deleting `update.old` from `test_elements` results in no index
                // of `test_elements` having the value `update.old`. Record it to
                // update the containment of `element` later.
                node_data->add_or_rm_test_elements.insert_or_assign(update.old, false);
            }
        }
        if (!update.removed()) {
            // i.e. changed or added.
            ssize_t& count = node_data->set_data[update.value].test_elements_count;
            count += 1;

            if (count == 1) {
                // Previously, `update.value` did not occur in `test_elements`.
                // Record it to update the containment of `element` later.
                node_data->add_or_rm_test_elements.insert_or_assign(update.value, true);
            }
        }
    }

    // Handle changes to `element`
    for (const Update& update : element_diff) {
        if (!update.placed()) {  // i.e. changed or removed
            rm_index(node_data, update.index, update.old);
        }
        if (!update.removed()) {  // i.e. changed or added
            IsInNodeSetData& set_data_struct = node_data->set_data[update.value];
            set_data_struct.element_indices.push_back(update.index);
            // Record whether or not the value of `element` at `update.index` is
            // in `test_elements`. Allow emplace_back()
            node_data->set(update.index, set_data_struct.test_elements_count > 0, true);
        }
    }

    // `add_or_rm_test_elements` contains all unique values of `test_elements`
    // that are new or fully removed
    for (const auto& [key, assignment] : node_data->add_or_rm_test_elements) {
        auto set_data_it = node_data->set_data.find(key);
        // find() should be successful since we need to remove the index
        assert(set_data_it != node_data->set_data.end());
        // Vector containing indices of `element` with value `key`
        const std::vector<ssize_t>& indices_vec = set_data_it->second.element_indices;

        if (!indices_vec.empty()) {
            for (const ssize_t& index : indices_vec) {
                node_data->set(index, assignment);
            }
            // Reduce bloating
        } else if (!assignment) {
            node_data->set_data.erase(set_data_it);
        }
    }
    node_data->add_or_rm_test_elements.clear();

    // Resize if necessary
    if (element_ptr_->size_diff(state) != 0) {
        node_data->trim_to(element_ptr_->size(state));
    }
}

void IsInNode::revert(State& state) const {
    IsInNodeData* node_data = data_ptr<IsInNodeData>(state);

    // Revert the changes to `test_elements`
    for (const Update& update : node_data->test_element_updates | std::views::reverse) {
        if (!update.placed()) {  // i.e. removed or changed
            node_data->set_data[update.old].test_elements_count += 1;
        }
        if (!update.removed()) {  // i.e. changed or removed.
            auto set_data_it = node_data->set_data.find(update.value);
            // find() should be successful since we need to remove the index
            assert(set_data_it != node_data->set_data.end());
            set_data_it->second.test_elements_count -= 1;
            // Reduce bloating
            if (set_data_it->second.test_elements_count == 0 &&
                set_data_it->second.element_indices.empty()) {
                node_data->set_data.erase(set_data_it);
            }
        }
    }

    // Revert the changes to `element`
    for (const Update& update : node_data->element_updates | std::views::reverse) {
        if (!update.placed()) {  // i.e. removed or changed
            node_data->set_data[update.old].element_indices.push_back(update.index);
        }
        if (!update.removed()) {  // i.e. placed or changed
            rm_index(node_data, update.index, update.value);
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

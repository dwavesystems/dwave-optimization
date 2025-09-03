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

#include "_state.hpp"

namespace dwave::optimization {

/// IsInNode
struct IsInNodeDataHelper_ {
    IsInNodeDataHelper_(std::vector<double> element, std::vector<double> test_elements) {
        element_isin.reserve(element.size());

        for (const double& val : test_elements) {
            test_elements_counter[val] += 1;
        }
        for (ssize_t index = 0, stop = element.size(); index < stop; index++) {
            element_isin.emplace_back(test_elements_counter.contains(element[index]));
            element_indices[element[index]].push_back(index);
        }
    }
    std::unordered_map<double, ssize_t> test_elements_counter;
    std::unordered_map<double, std::vector<ssize_t>> element_indices;
    // element_isin[i] == true iff `element[i]` is in`test_elements`
    std::vector<bool> element_isin;
};

struct IsInNodeData : public ArrayNodeStateData {
    IsInNodeData(std::vector<double> element, std::vector<double> test_elements)
            : IsInNodeData(IsInNodeDataHelper_(std::move(element), std::move(test_elements))) {}
    IsInNodeData(IsInNodeDataHelper_&& helper)
            : ArrayNodeStateData(std::move(helper.element_isin)),
              test_elements_counter(std::move(helper.test_elements_counter)),
              element_indices(std::move(helper.element_indices)) {}

    // # of elements from `test_elements` with a given value (key)
    std::unordered_map<double, ssize_t> test_elements_counter;
    // All indices of `element` with a given value (key)
    std::unordered_map<double, std::vector<ssize_t>> element_indices;

    // Used to track if elements are added/removed from `test_elements`
    // during IsInNode::propagate()
    std::unordered_map<double, bool> add_or_rm_test_elements;

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

std::pair<double, double> IsInNode::minmax(
        optional_cache_type<std::pair<double, double>> cache) const {
    return memoize(cache, [&]() {
        return std::make_pair(0.0, 1.0);  // All values are true/false
    });
}

inline void IsInNode::rm_index_from_element_indices_key(IsInNodeData*& node_data,
                                                        const ssize_t rm_index,
                                                        const double key) const {
    // Get iterator containing indices of `element` with value `key`
    // Note: find() must be successful
    auto indices_iter = node_data->element_indices.find(key);
    assert(indices_iter != node_data->element_indices.end());
    // Vector containing indices of `element` with value `key`
    std::vector<ssize_t>& indices_vec = indices_iter->second;
    // Find index of `indices_vec` equal to `rm_index`
    // Note: find() must be successful
    auto index_ptr = std::find(indices_vec.begin(), indices_vec.end(), rm_index);
    assert(index_ptr != indices_vec.end());
    // swap and pop
    *index_ptr = indices_vec.back();
    indices_vec.pop_back();
    // Uncomment to reduce bloating
    // if (indices_vec.empty()) {
    //     node_data->element_indices.erase(indices_iter);
    // }
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
            auto count = node_data->test_elements_counter.find(update.old);
            // `update.old` must be in `test_elements_counter`
            assert(count != node_data->test_elements_counter.end());
            count->second -= 1;

            if (count->second == 0) {
                // Deleting `update.old` from `test_elements` results in no index
                // of `test_elements` having the value `update.old`. Record it to
                // update the containment of `element` later.
                node_data->add_or_rm_test_elements.insert_or_assign(update.old, false);
            }
        }
        if (!update.removed()) {
            // i.e. changed or added. Note: `update.value` may not occur in
            // `test_elements_counter`, hence try_emplace
            auto [count, _] = node_data->test_elements_counter.try_emplace(update.value, 0);
            count->second += 1;

            if (count->second == 1) {
                // Previously, `update.value` did not occur in `test_elements`.
                // Record it to update the containment of `element` later.
                node_data->add_or_rm_test_elements.insert_or_assign(update.value, true);
            }
        }
    }

    // Handle changes to `element`
    for (const Update& update : element_diff) {
        if (!update.placed()) {  // i.e. changed or removed
            rm_index_from_element_indices_key(node_data, update.index, update.old);
        }
        if (!update.removed()) {  // i.e. changed or added
            node_data->element_indices[update.value].push_back(update.index);
            // Record whether or not the value of `element` at `update.index` is
            // in `test_elements`. Allow emplace_back()
            node_data->set(update.index, node_data->test_elements_counter.contains(update.value),
                           true);
        }
    }

    // `add_or_rm_test_elements` contains all unique values of `test_elements`
    // that are new or fully removed
    for (const auto& [key, assignment] : node_data->add_or_rm_test_elements) {
        // Find `key` (should it exist) in `element_indices`
        auto indices_iter = node_data->element_indices.find(key);

        if (indices_iter != node_data->element_indices.end()) {
            // indices_iter->second is vector containing indices of `element` with
            // value `key`
            for (const ssize_t& index : indices_iter->second) {
                node_data->set(index, assignment);
            }
        }
        // Uncomment to reduce bloating
        // if (!assignment) {
        //     node_data->test_elements_counter.erase(key);
        // }
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
            node_data->test_elements_counter[update.old] += 1;
        }
        if (!update.removed()) {  // i.e. changed or removed.
            auto count = node_data->test_elements_counter.find(update.value);
            // `update.value` must be in `test_elements_counter`
            assert(count != node_data->test_elements_counter.end());
            count->second -= 1;
            // Uncomment to reduce bloating
            // if (count->second == 0) {
            //     node_data->test_elements_counter.erase(count);
            // }
        }
    }

    // Revert the changes to `element`
    for (const Update& update : node_data->element_updates | std::views::reverse) {
        if (!update.placed()) {  // i.e. removed or changed
            node_data->element_indices[update.old].push_back(update.index);
        }
        if (!update.removed()) {  // i.e. placed or changed
            rm_index_from_element_indices_key(node_data, update.index, update.value);
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

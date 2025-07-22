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

#include "dwave-optimization/nodes/statistics.hpp"

#include <span>

#include "dwave-optimization/array.hpp"

namespace dwave::optimization {

MeanNode::MeanNode(ArrayNode *arr_ptr) : ScalarOutputMixin<ArrayNode, true>(), arr_ptr_(arr_ptr) {
    add_predecessor(arr_ptr);
}

void MeanNode::initialize_state(State &state) const {
    // If state is empty, default value for mean is zero.
    if (arr_ptr_->size(state) == 0) {
        emplace_data_ptr<ScalarOutputMixinStateData>(state, 0.0);
        return;
    }
    // Otherwise, compute mean of values in array
    double sum = 0.0;
    for (const auto val : arr_ptr_->view(state)) {
        sum += val;
    }
    const double mean = sum / static_cast<double>(arr_ptr_->size(state));
    emplace_data_ptr<ScalarOutputMixinStateData>(state, mean);
}

bool MeanNode::integral() const { return false; }

std::pair<double, double> MeanNode::minmax(
        optional_cache_type<std::pair<double, double>> cache) const {
    return memoize(cache, [&]() {
        // If state is empty, default value for mean is zero.
        if (arr_ptr_->size() == 0) {
            return std::make_pair(0.0, 0.0);
        }
        // Othwerwise, ArrayNode is non-empty and therefore has min and max
        return std::make_pair(arr_ptr_->min(), arr_ptr_->max());
    });
}

void MeanNode::propagate(State &state) const {
    const std::span<const Update> arr_updates = arr_ptr_->diff(state);

    // If no update, mean is unchanged
    if (arr_updates.empty()) {
        return;
    }

    auto node_data = data_ptr<ScalarOutputMixinStateData>(state);
    const double state_size = static_cast<double>(arr_ptr_->size(state));

    // If state is empty, default value for mean is zero.
    if (state_size == 0.0) {
        node_data->set(0.0);
        return;
    }

    // Compute size of ArrayNode prior to change
    const double state_size_prior = state_size - static_cast<double>(arr_ptr_->size_diff(state));
    // Compute the sum of ArrayNode prior to change
    double sum = node_data->update.value * (state_size_prior);

    if (state_size == state_size_prior) {
        // no value was removed or placed
        for (const Update &u : arr_updates) {
            sum += u.value - u.old;
        }
    } else {
        for (const Update &u : arr_updates) {
            if (u.removed()) {
                sum -= u.old;
            } else if (u.placed()) {
                sum += u.value;
            } else {
                sum += u.value - u.old;
            }
        }
    }
    // Compute new mean
    node_data->set(sum / state_size);
}

}  // namespace dwave::optimization

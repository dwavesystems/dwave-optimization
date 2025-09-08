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

std::pair<double, double> calculate_values_minmax_(const Array *arr_ptr) {
    // Predecessor is static or dynamic but always non-empty. Therefore,
    // mean is always computed and will fall within min/max of predecessor.
    if ((arr_ptr->size() > 0) || (arr_ptr->sizeinfo().min.value_or(0) > 0)) {
        return std::make_pair(arr_ptr->min(), arr_ptr->max());
    }
    // Predecessor is static and empty. Therefore, mean will always be
    // default of 0.0.
    if (arr_ptr->size() == 0) {
        return std::make_pair(0.0, 0.0);
    }
    // Predecessor is dynamic and possibly empty. Therefore, mean will be
    // default value of 0.0 (i.e. predecessor is empty) or fall within the
    // min/max of predecessor. If necessary, extend meannode min/max.
    return std::make_pair(std::min(arr_ptr->min(), 0.0), std::max(arr_ptr->max(), 0.0));
}

MeanNode::MeanNode(ArrayNode *arr_ptr)
        : ScalarOutputMixin<ArrayNode, true>(),
          arr_ptr_(arr_ptr),
          minmax_(calculate_values_minmax_(arr_ptr_)) {
    add_predecessor(arr_ptr);
}

void MeanNode::initialize_state(State &state) const {
    if (arr_ptr_->size(state) == 0) {
        emplace_state(state, 0.0);
        return;
    }
    // Otherwise, compute mean of values in array
    double sum = 0.0;
    for (const auto val : arr_ptr_->view(state)) {
        sum += val;
    }
    emplace_state(state, sum / static_cast<double>(arr_ptr_->size(state)));
}

bool MeanNode::integral() const { return false; }

double MeanNode::min() const { return this->minmax_.first; }

double MeanNode::max() const { return this->minmax_.second; }

void MeanNode::propagate(State &state) const {
    const std::span<const Update> arr_updates = arr_ptr_->diff(state);

    // If no update, mean is unchanged
    if (arr_updates.empty()) {
        return;
    }

    const ssize_t state_size = arr_ptr_->size(state);

    if (state_size == 0) {
        set_state(state, 0.0);
        return;
    }

    // Compute size of ArrayNode prior to change
    const ssize_t state_size_prior = state_size - arr_ptr_->size_diff(state);

    // Compute the sum of ArrayNode prior to change
    double sum = data_ptr<ScalarOutputMixinStateData>(state)->update.value *
                 static_cast<double>(state_size_prior);

    for (const Update &u : arr_updates) {
        if (u.removed()) {
            sum -= u.old;
        } else if (u.placed()) {
            sum += u.value;
        } else {
            sum += u.value - u.old;
        }
    }
    set_state(state, sum / static_cast<double>(state_size));
}
}  // namespace dwave::optimization

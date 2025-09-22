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

#include "dwave-optimization/nodes/softmax.hpp"

#include <cassert>
#include <cmath>
#include <ranges>
#include <vector>

#include "_state.hpp"

namespace dwave::optimization {

struct SoftMaxNodeDataHelper_ {
    SoftMaxNodeDataHelper_(std::vector<double> input) {
        // Compute softmax.
        denominator = 0.0;
        for (ssize_t i = 0, stop = input.size(); i < stop; ++i) {
            double exp_val = std::exp(input[i]);
            values.push_back(exp_val);
            denominator += exp_val;
        }
        for (double& val : values) {
            val /= denominator;
        }
        // Give prior_denominator an initial value.
        prior_denominator = denominator;
    }

    std::vector<double> values;
    double denominator;
    double prior_denominator;
};

struct SoftMaxNodeStateData : public ArrayNodeStateData {
    SoftMaxNodeStateData(std::vector<double> input)
            : SoftMaxNodeStateData(SoftMaxNodeDataHelper_(std::move(input))) {}

    SoftMaxNodeStateData(SoftMaxNodeDataHelper_&& helper)
            : ArrayNodeStateData(std::move(helper.values)),
              denominator(helper.denominator),
              prior_denominator(helper.prior_denominator) {}

    double denominator;
    double prior_denominator;
    std::vector<bool> index_changed;
};

SoftMaxNode::SoftMaxNode(ArrayNode* arr_ptr)
        : ArrayOutputMixin(arr_ptr->shape()), arr_ptr_(arr_ptr), sizeinfo_(arr_ptr_->sizeinfo()) {
    add_predecessor(arr_ptr);
}

double const* SoftMaxNode::buff(const State& state) const {
    return data_ptr<SoftMaxNodeStateData>(state)->buff();
}

void SoftMaxNode::commit(State& state) const {
    return data_ptr<SoftMaxNodeStateData>(state)->commit();
}

std::span<const Update> SoftMaxNode::diff(const State& state) const {
    return data_ptr<SoftMaxNodeStateData>(state)->diff();
}

void SoftMaxNode::initialize_state(State& state) const {
    const Array::View arr = arr_ptr_->view(state);
    emplace_data_ptr<SoftMaxNodeStateData>(state, std::vector<double>{arr.begin(), arr.end()});
}

bool SoftMaxNode::integral() const { return false; }

// Softmax function forms probability space, the values of which sum to 1.0.
double SoftMaxNode::min() const { return 0.0; }

double SoftMaxNode::max() const { return 1.0; }

void SoftMaxNode::propagate(State& state) const {
    // Store updates to predecessor in vector
    std::vector<Update> arr_updates(arr_ptr_->diff(state).begin(), arr_ptr_->diff(state).end());

    if (arr_updates.empty()) {
        return;
    }

    // To update node, we need to compute the new softmax denominator
    auto node_data = data_ptr<SoftMaxNodeStateData>(state);
    const double prior_denominator = node_data->denominator;
    double new_denominator = prior_denominator;
    // We only want to compute an exponential once per index
    deduplicate_diff(arr_updates);
    // Record for which indices we need to recompute the exponential
    node_data->index_changed.resize(arr_ptr_->size(state), false);

    for (const Update& u : arr_updates) {
        // Offset by contribution to prior_denominator. When possible, avoid
        // calling exp() function by multiplying elements by their denominator.
        if (!u.placed()) {  // i.e. removed or changed.
            new_denominator -= node_data->get(u.index) * prior_denominator;
        }
        if (!u.removed()) {  // i.e. placed or changed
            new_denominator += std::exp(u.value);
            // Due to deduplicate_diff, this should be false
            assert(!node_data->index_changed[u.index]);
            node_data->index_changed[u.index] = true;
        }
    }

    // Technically, we could check whether prior_denominator == denominator
    // to avoid extra updates. However, this edge case is unlikely.
    const double scale = prior_denominator / new_denominator;
    for (ssize_t i = 0, stop = arr_ptr_->size(state); i < stop; ++i) {
        if (node_data->index_changed[i]) {
            node_data->set(i, std::exp(arr_ptr_->view(state)[i]) / new_denominator, true);
            node_data->index_changed[i] = false;
        } else {
            node_data->set(i, node_data->get(i) * scale);
        }
    }

    // Resize if necessary
    if (arr_ptr_->size_diff(state) != 0) {
        node_data->trim_to(arr_ptr_->size(state));
    }

    node_data->denominator = new_denominator;
    node_data->prior_denominator = prior_denominator;
}

void SoftMaxNode::revert(State& state) const {
    auto node_data = data_ptr<SoftMaxNodeStateData>(state);
    node_data->revert();
    // Manually reset denominator
    node_data->denominator = node_data->prior_denominator;
}

std::span<const ssize_t> SoftMaxNode::shape(const State& state) const {
    return arr_ptr_->shape(state);
}

ssize_t SoftMaxNode::size(const State& state) const {
    return data_ptr<SoftMaxNodeStateData>(state)->size();
}

ssize_t SoftMaxNode::size_diff(const State& state) const {
    return data_ptr<SoftMaxNodeStateData>(state)->size_diff();
}

SizeInfo SoftMaxNode::sizeinfo() const { return this->sizeinfo_; }

}  // namespace dwave::optimization

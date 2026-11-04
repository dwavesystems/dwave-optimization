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

#include "_state.hpp"
#include "dwave-optimization/array.hpp"

namespace dwave::optimization {

namespace {
// Setting maximum size to prevent large array bincount outputs
static constexpr int bincount_maximum_size_ = 10001;
static constexpr std::string_view bincount_error_noninteger_ = "array values must be integer";
static constexpr std::string_view bincount_error_negative_ = "array values must be non-negative";
static const std::string bincount_error_too_large_ =
    "array values must be less than " + std::to_string(bincount_maximum_size_ - 1);

ssize_t bincount_get_size_(ArrayNode* arr_ptr) {
    if (arr_ptr->max() >= static_cast<double>(bincount_maximum_size_)) {
        // This will throw in the constructor body
        return 0;
    }

    return static_cast<ssize_t>(arr_ptr->max()) + 1;
}
}  // namespace

BinCountNode::BinCountNode(ArrayNode* arr_ptr) :
    BinCountNode(arr_ptr, bincount_get_size_(arr_ptr)) {}

BinCountNode::BinCountNode(ArrayNode* arr_ptr, ssize_t size) :
    ArrayOutputMixin<EqualityMixin<ArrayNode>>(size), arr_ptr_(arr_ptr), size_(size) {
    // Check validity of predecessor node
    if (!arr_ptr->integral()) {
        throw std::invalid_argument(bincount_error_noninteger_.data());
    }
    if (arr_ptr->min() < 0.0) {
        throw std::invalid_argument(bincount_error_negative_.data());
    }
    if (arr_ptr->max() >= bincount_maximum_size_) {
        throw std::invalid_argument(bincount_error_too_large_.data());
    }
    add_predecessor_(arr_ptr);
}

void BinCountNode::initialize_state(State& state) const {
    std::vector<double> data;
    data.resize(size_, 0.0);

    // Compute bincount
    for (const double& val : arr_ptr_->view(state)) {
        // Assert validity of data
        assert(is_integer(val));
        assert(val >= 0);
        assert(val < static_cast<double>(bincount_maximum_size_));
        int val_int = val;
        data[val_int] += 1.0;
    }

    emplace_data_ptr_<ArrayNodeStateData>(state, std::move(data));
}

double const* BinCountNode::buff(const State& state) const {
    return data_ptr_<ArrayNodeStateData>(state)->buff();
}

void BinCountNode::commit(State& state) const {
    return data_ptr_<ArrayNodeStateData>(state)->commit();
}

std::span<const Update> BinCountNode::diff(const State& state) const {
    return data_ptr_<ArrayNodeStateData>(state)->diff();
}

bool BinCountNode::integral() const { return true; }

double BinCountNode::min() const { return 0.0; }

double BinCountNode::max() const {
    return arr_ptr_->sizeinfo().max.value_or(std::numeric_limits<ssize_t>::max());
}

void BinCountNode::propagate(State& state) const {
    const std::span<const Update> i_updates = arr_ptr_->diff(state);

    // If no update, bincount is unchanged
    if (i_updates.empty()) {
        return;
    }

    auto* bincount_data = data_ptr_<ArrayNodeStateData>(state);

    // Update bin counts using new and old values
    for (const auto& update : i_updates) {
        assert(
            std::isnan(update.value) ||
            (is_integer(update.value) && update.value >= 0 && update.value < bincount_maximum_size_)
        );
        if (!update.removed()) {
            int value_int = update.value;
            bincount_data->set(value_int, bincount_data->get(value_int) + 1);
        }
        if (!update.placed()) {
            int old_int = update.old;
            bincount_data->set(old_int, bincount_data->get(old_int) - 1);
        }
    }
}

void BinCountNode::replace_predecessor_(ssize_t index, Node* node_ptr) {
    Node::replace_predecessor_(index, node_ptr);

    ArrayNode* array_ptr = dynamic_cast<ArrayNode*>(node_ptr);
    size_ = bincount_get_size_(array_ptr);
    assert(array_ptr != nullptr);
    assert(std::ranges::equal(arr_ptr_->shape(), array_ptr->shape()));

    assert(index == 0);
    arr_ptr_ = array_ptr;
}

void BinCountNode::revert(State& state) const {
    auto data = data_ptr_<ArrayNodeStateData>(state);
    data->revert();
}

SizeInfo BinCountNode::sizeinfo() const { return SizeInfo(size_); }

std::pair<double, double> calculate_values_minmax_(const Array* arr_ptr) {
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

MeanNode::MeanNode(ArrayNode* arr_ptr) :
    ScalarOutputMixin<EqualityMixin<ArrayNode>, true>(),
    arr_ptr_(arr_ptr),
    minmax_(calculate_values_minmax_(arr_ptr_)) {
    add_predecessor_(arr_ptr);
}

void MeanNode::initialize_state(State& state) const {
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

void MeanNode::propagate(State& state) const {
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
    double sum = data_ptr_<ScalarOutputMixinStateData>(state)->update.value *
                 static_cast<double>(state_size_prior);

    for (const Update& u : arr_updates) {
        if (!u.placed()) {  // i.e. removed or changed.
            sum -= u.old;
        }
        if (!u.removed()) {  // i.e. placed or changed.
            sum += u.value;
        }
    }
    set_state(state, sum / static_cast<double>(state_size));
}

void MeanNode::replace_predecessor_(ssize_t index, Node* node_ptr) {
    Node::replace_predecessor_(index, node_ptr);

    ArrayNode* array_ptr = dynamic_cast<ArrayNode*>(node_ptr);
    assert(array_ptr != nullptr);
    assert(std::ranges::equal(arr_ptr_->shape(), array_ptr->shape()));

    assert(index == 0);
    arr_ptr_ = array_ptr;
}

}  // namespace dwave::optimization

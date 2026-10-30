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
static constexpr int bincount_maximum_size_ = 2000000000;
static constexpr std::string_view bincount_error_noninteger_ = "array values must be integer";
static constexpr std::string_view bincount_error_negative_ = "array values must be non-negative";
static constexpr std::string_view bincount_error_toolarge_ = "array values too large";
}  // namespace

struct BinCountNodeDataHelper_ {
    BinCountNodeDataHelper_(std::vector<double> input) : values(std::vector<double>()), size(0) {
        // If input is empty data is an empty vector (default values)
        if (input.size() == 0) {
            return;
        }
        // Determine size from input
        double max_val = *std::max_element(input.begin(), input.end());
        if (max_val > static_cast<double>(bincount_maximum_size_)) {
            throw std::invalid_argument(bincount_error_toolarge_.data());
        }
        int max_val_int = max_val;
        values.resize(max_val_int + 1, 0.0);

        // Compute bincount
        for (const double& val : input) {
            // Verify that the values are non-negative integers
            if (!is_integer(val)) {
                throw std::invalid_argument(bincount_error_noninteger_.data());
            } else if (val < 0) {
                throw std::invalid_argument(bincount_error_negative_.data());
            }
            int val_int = val;
            values[val_int] += 1.0;
        }
        size = max_val + 1;
    }

    std::vector<double> values;
    ssize_t size;
};

class BinCountNodeData : public ArrayNodeStateData {
 public:
    BinCountNodeData(std::vector<double> input) :
        BinCountNodeData(BinCountNodeDataHelper_(std::move(input))) {}

    BinCountNodeData(BinCountNodeDataHelper_&& helper) :
        ArrayNodeStateData(std::move(helper.values)), size(helper.size) {}

    std::span<const ssize_t> shape() const { return std::span<const ssize_t>(&size, 1); }

    void grow(ssize_t index) {
        // Add zero entries until index is the last entry
        assert(index >= size);
        std::vector<Update> grow_updates;
        grow_updates.reserve(index - size + 1);
        while (index >= size) {
            Update update(size, std::numeric_limits<double>::signaling_NaN(), 0.0);
            grow_updates.push_back(update);
            ++size;
        }

        update(std::move(grow_updates));
    }

    void shrink(ssize_t index) {
        // Remove zero entries until index is the last entry
        assert(index < size);
        std::vector<Update> shrink_updates;
        shrink_updates.reserve(size - index - 1);
        while (index < size) {
            Update update(size, 0.0, std::numeric_limits<double>::signaling_NaN());
            shrink_updates.push_back(update);
            --size;
        }

        update(std::move(shrink_updates));
    }

    void bincount_revert() {
        // Need to update internal size value as well
        size -= size_diff();
        this->revert();
    }

    ssize_t size;
};

BinCountNode::BinCountNode(ArrayNode* i_ptr) :
    // Because output size depends on maximum input value, BinCount array is always dynamic
    ArrayOutputMixin<EqualityMixin<ArrayNode>>(-1),
    i_ptr_(i_ptr),
    sizeinfo_(SizeInfo(this, 0, bincount_maximum_size_)) {
    add_predecessor_(i_ptr);
}

void BinCountNode::initialize_state(State& state) const {
    emplace_data_ptr_<BinCountNodeData>(
        state, std::vector<double>{i_ptr_->begin(state), i_ptr_->end(state)}
    );
}

double const* BinCountNode::buff(const State& state) const {
    return data_ptr_<BinCountNodeData>(state)->buff();
}

void BinCountNode::commit(State& state) const {
    return data_ptr_<BinCountNodeData>(state)->commit();
}

std::span<const Update> BinCountNode::diff(const State& state) const {
    return data_ptr_<BinCountNodeData>(state)->diff();
}

bool BinCountNode::integral() const { return true; }

double BinCountNode::min() const { return 0.0; }

double BinCountNode::max() const { return sizeof(ssize_t); }

void BinCountNode::propagate(State& state) const {
    const std::span<const Update> i_updates = i_ptr_->diff(state);

    // If no update, bincount is unchanged
    if (i_updates.empty()) {
        return;
    }

    auto bincount_data = data_ptr_<BinCountNodeData>(state);
    int max_bin = bincount_data->size - 1;

    // First look through updates to check if we need to change the size of the output
    for (const auto& update : i_updates) {
        // Check for validity of new values
        assert(
            std::isnan(update.value) ||
            (is_integer(update.value) && update.value >= 0 && update.value < bincount_maximum_size_)
        );

        int value_int = update.value;
        max_bin = std::max(max_bin, value_int);
    }

    // If new max_bin is larger than the previous maximum, resize to max_bin + 1
    if (max_bin >= bincount_data->size) {
        bincount_data->grow(max_bin);
    }

    // Update bin counts using new and old values
    for (const auto& update : i_updates) {
        if (!update.removed()) {
            int value_int = update.value;
            bincount_data->set(value_int, bincount_data->get(value_int) + 1);
        }
        if (!update.placed()) {
            int old_int = update.old;
            bincount_data->set(old_int, bincount_data->get(old_int) - 1);
        }
    }

    // Trim bincount output to last non-zero entry
    ssize_t last_nonzero = bincount_data->size - 1;
    while (bincount_data->get(last_nonzero) == 0.0) {
        --last_nonzero;
    }
    if (last_nonzero < bincount_data->size - 1) {
        bincount_data->shrink(last_nonzero);
    }
}

void BinCountNode::replace_predecessor_(ssize_t index, Node* node_ptr) {
    Node::replace_predecessor_(index, node_ptr);

    ArrayNode* array_ptr = dynamic_cast<ArrayNode*>(node_ptr);
    assert(array_ptr != nullptr);
    assert(std::ranges::equal(i_ptr_->shape(), array_ptr->shape()));

    assert(index == 0);
    i_ptr_ = array_ptr;
}

void BinCountNode::revert(State& state) const {
    auto data = data_ptr_<BinCountNodeData>(state);
    data->bincount_revert();
}

std::span<const ssize_t> BinCountNode::shape(const State& state) const {
    return data_ptr_<BinCountNodeData>(state)->shape();
}

ssize_t BinCountNode::size(const State& state) const {
    return data_ptr_<BinCountNodeData>(state)->size;
}

ssize_t BinCountNode::size_diff(const State& state) const {
    return data_ptr_<BinCountNodeData>(state)->size_diff();
}

SizeInfo BinCountNode::sizeinfo() const { return sizeinfo_; }

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

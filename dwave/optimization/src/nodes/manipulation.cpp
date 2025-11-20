// Copyright 2024 D-Wave
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

#include "dwave-optimization/nodes/manipulation.hpp"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

#include "_state.hpp"
#include "dwave-optimization/array.hpp"
#include "dwave-optimization/graph.hpp"

namespace dwave::optimization {

// Return the linear offsets that need to be applied to each update when broadcasting from
// `from_shape` to `to_shape`.
std::vector<ssize_t> diff_offsets(std::span<const ssize_t> from_shape,
                                  std::span<const ssize_t> to_shape) {
    std::vector<ssize_t> offsets{0};  // for an identity broadcast there is only one offset

    auto expand = [](const std::vector<ssize_t>& offsets, ssize_t size, ssize_t multiplier) {
        assert(size > 0);
        assert(multiplier > 0);
        std::vector<ssize_t> new_offsets;
        for (const ssize_t& offset : offsets) {
            for (ssize_t i = 0; i < size; ++i) {
                new_offsets.emplace_back(offset + i * multiplier);
            }
        }

        return new_offsets;
    };

    assert(from_shape.size() <= to_shape.size());

    auto from_it = from_shape.rbegin();
    auto to_it = to_shape.rbegin();
    ssize_t multiplier = 1;
    for (const auto stop = from_shape.rend(); from_it != stop; ++from_it, ++to_it) {
        if (*from_it != *to_it) {
            assert(*from_it == 1);  // should be checked in constructor
            offsets = expand(offsets, *to_it, multiplier);
        }

        multiplier *= *to_it;
    }

    // all of the dimensions we're prepending need offsets
    for (auto stop = to_shape.rend(); to_it != stop; ++to_it) {
        offsets = expand(offsets, *to_it, multiplier);
        multiplier *= *to_it;
    }

    // We need it to be sorted in order to potentially support dynamic nodes
    // Also it's more convenient to debug.
    std::ranges::sort(offsets);

    return offsets;
}

class BroadcastToNodeData : public NodeStateData {
 public:
    BroadcastToNodeData(std::vector<ssize_t>&& diff_offsets)
            : diff_offsets(std::move(diff_offsets)) {}

    virtual void commit() { diff.clear(); }
    virtual void revert() { diff.clear(); }

    const std::vector<ssize_t> diff_offsets;

    std::vector<Update> diff;
};

class DynamicBroadcastToNodeData : public BroadcastToNodeData {
 public:
    DynamicBroadcastToNodeData(std::vector<ssize_t>&& diff_offsets, std::vector<ssize_t>&& shape)
            : BroadcastToNodeData(std::move(diff_offsets)), shape(std::move(shape)) {}

    virtual void commit() {
        BroadcastToNodeData::commit();
        old_shape = shape;
        size_diff = 0;
    }

    virtual void revert() {
        BroadcastToNodeData::revert();
        shape = old_shape;
        size_diff = 0;
    }

    std::vector<ssize_t> shape;
    std::vector<ssize_t> old_shape = shape;  // for easy reverting

    ssize_t size_diff = 0;
};

BroadcastToNode::BroadcastToNode(ArrayNode* array_ptr, std::initializer_list<ssize_t> shape)
        : BroadcastToNode(array_ptr, std::span(shape)) {}

BroadcastToNode::BroadcastToNode(ArrayNode* array_ptr, std::span<const ssize_t> shape)
        : array_ptr_(array_ptr),
          ndim_(shape.size()),
          shape_(std::make_unique<ssize_t[]>(ndim_)),
          strides_(std::make_unique<ssize_t[]>(ndim_)),
          values_info_(array_ptr) {
    // Fill in our shape_ and then make it accessible locally as a span
    std::copy(shape.begin(), shape.end(), shape_.get());
    const std::span<const ssize_t> target_shape(shape_.get(), ndim_);

    // Make our strides locally accessible as a span.
    std::span<ssize_t> target_strides(strides_.get(), ndim_);

    // Finally get our source shape and strides
    const auto source_shape = array_ptr_->shape();
    const auto source_strides = array_ptr_->strides();

    // Alright, let's check that our source and target shapes are compatible
    // and at the same time fill in our strides

    if (source_shape.size() > target_shape.size()) {
        throw std::invalid_argument("array has more dimensions (" +
                                    std::to_string(source_shape.size()) +
                                    ") than can be broadast to " + shape_to_string(target_shape));
    }

    if (array_ptr_->dynamic() &&
        (source_shape.size() != target_shape.size() || target_shape[0] >= 0)) {
        throw std::invalid_argument(
                "dynamic arrays can only be broadcast to another dynamic shape with the same "
                "number of dimensions");
    }

    // Walk backwards through all four arrays. Zip would be nice here...
    auto rit_sshape = source_shape.rbegin();
    auto rit_sstrides = source_strides.rbegin();
    auto rit_tshape = target_shape.rbegin();
    auto rit_tstrides = target_strides.rbegin();
    for (const auto stop = source_shape.rend(); rit_sshape != stop;
         ++rit_tshape, ++rit_tstrides, ++rit_sshape, ++rit_sstrides) {
        if (*rit_sshape == *rit_tshape) {
            // source and target match in dimension size, strides are therefore
            // preserved
            *rit_tstrides = *rit_sstrides;
        } else if (*rit_sshape == 1) {
            // source dimension size is 1, so we can stretch it out
            *rit_tstrides = 0;  // stretched
        } else {
            // not compatible!
            throw std::invalid_argument("array of shape " + shape_to_string(source_shape) +
                                        " could not be broadcast to " +
                                        shape_to_string(target_shape));
        }
    }

    // All of the dimensions we're adding at the "front" get 0 strides
    for (auto stop = target_shape.rend(); rit_tshape != stop; ++rit_tshape, ++rit_tstrides) {
        *rit_tstrides = 0;
    }

    add_predecessor(array_ptr);
}

double const* BroadcastToNode::buff(const State& state) const { return array_ptr_->buff(state); }

void BroadcastToNode::commit(State& state) const { data_ptr<BroadcastToNodeData>(state)->commit(); }

std::span<const Update> BroadcastToNode::diff(const State& state) const {
    return data_ptr<BroadcastToNodeData>(state)->diff;
}

void BroadcastToNode::initialize_state(State& state) const {
    std::vector<ssize_t> offsets = diff_offsets(array_ptr_->shape(), this->shape());

    if (ndim_ && shape_[0] < 0) {
        // dynamic
        // In addition to storing the offsets, when we're dynamic we need to store our current
        // shape/size as well. Luckily, they are easy to calculate because we cannot broadcast
        // along shape[0].

        std::vector<ssize_t> shape(this->shape().begin(), this->shape().end());
        assert(this->ndim() > 0);
        assert(array_ptr_->ndim() > 0);
        assert(shape[0] == -1);
        shape[0] = array_ptr_->shape(state)[0];
        assert(shape[0] >= 0);

        emplace_data_ptr<DynamicBroadcastToNodeData>(state, std::move(offsets), std::move(shape));
    } else {
        // not dynamic
        emplace_data_ptr<BroadcastToNodeData>(state, std::move(offsets));
    }
}

bool BroadcastToNode::integral() const { return values_info_.integral; }

double BroadcastToNode::min() const { return values_info_.min; }

double BroadcastToNode::max() const { return values_info_.max; }

ssize_t BroadcastToNode::ndim() const { return ndim_; }

void BroadcastToNode::propagate(State& state) const {
    BroadcastToNodeData* data_ptr = this->data_ptr<BroadcastToNodeData>(state);

    const auto from_diff = array_ptr_->diff(state);
    if (from_diff.empty()) return;  // exit early if nothing to propagate

    auto& to_diff = data_ptr->diff;
    assert(to_diff.empty());  // should be cleared between propagations

    const auto& diff_offsets = data_ptr->diff_offsets;

    ssize_t size_diff = 0;
    for (Update update : deduplicate_diff_view(from_diff)) {
        // we need to convert from our predecessor's index to ours
        const ssize_t index = convert_predecessor_index_(update.index);
        assert(([&]() {
                   std::vector<ssize_t> multi_index =
                           unravel_index(update.index, array_ptr_->shape());
                   multi_index.insert(multi_index.begin(), this->ndim() - array_ptr_->ndim(), 0);
                   const ssize_t assert_index = ravel_multi_index(multi_index, this->shape());
                   return assert_index == index;
               })() &&
               "Bad conversion of predecessor index");

        if (update.placed()) {
            for (const ssize_t offset : diff_offsets) {
                update.index = index + offset;
                to_diff.emplace_back(update);
                size_diff += 1;
            }
        } else if (update.removed()) {
            for (const ssize_t offset : diff_offsets | std::views::reverse) {
                update.index = index + offset;
                to_diff.emplace_back(update);
                size_diff -= 1;
            }
        } else {
            for (const ssize_t offset : diff_offsets) {
                update.index = index + offset;
                to_diff.emplace_back(update);
            }
        }
    }

    // we also need to update the shape/sizediff, if we're dynamic
    if (dynamic()) {
        auto* dynamic_data_ptr = this->data_ptr<DynamicBroadcastToNodeData>(state);
        dynamic_data_ptr->shape[0] = array_ptr_->shape(state)[0];
        dynamic_data_ptr->size_diff = size_diff;
    }

    if (to_diff.size()) Node::propagate(state);
}

ssize_t BroadcastToNode::convert_predecessor_index_(ssize_t index) const {
    assert(this->ndim() >= array_ptr_->ndim() && "incorrect # of dimensions");
    assert(index >= 0 && "index must be non-negative");  // NumPy raises here so we assert

    std::span<const ssize_t> array_shape = array_ptr_->shape();
    std::span<const ssize_t> node_shape = this->shape();

    if (array_shape.empty()) {
        assert(index == 0);  // otherwise it's out-of-bounds
        return index;
    }

    // Shape iterators, initialized to the last element in their resp. ranges.
    auto array_shape_it = std::ranges::end(array_shape) - 1;
    auto node_shape_it = std::ranges::end(node_shape) - 1;

    ssize_t flat_index = 0;
    ssize_t multiplier = 1;

    // We traverse the dimensions (and shape) of the predecessor array in
    // reverse order up to and *not* including the 0th dimension while also
    // traversing the BroadcastToNode shape in reverse.
    for (ssize_t dim = std::ranges::size(array_shape) - 1; dim > 0;
         --dim, --array_shape_it, --node_shape_it) {
        assert(0 <= dim && "All dimensions except the first must be non-negative");
        assert(array_shape_it != array_shape.begin() - 1 && "Bad array shape iterator");

        // Contribution of `index` in the given dimension `dim`
        const ssize_t multidimensional_index = index % *array_shape_it;
        index /= *array_shape_it;

        // NumPy supports "clip" and "wrap" which we could add support for
        // but for now let's just assert.
        assert(0 <= multidimensional_index && multidimensional_index < *node_shape_it &&
               "Multidimensional_index exceeds node shape");
        assert(node_shape_it != node_shape.begin() - 1 && "Bad node shape iterator");

        // determine the contribution of `multidimensional_index` to flat index
        flat_index += multidimensional_index * multiplier;
        // this contribution is defined by the node shape
        multiplier *= *node_shape_it;
    }

    // Check if the index is out of bounds for non-dynamic shapes and assert
    assert(array_shape[0] < 0 || index < array_shape[0]);

    // Handle the contribution of the 0th dimension of the predecessor.
    flat_index += index * multiplier;

    return flat_index;
}

void BroadcastToNode::revert(State& state) const { data_ptr<BroadcastToNodeData>(state)->revert(); }

std::span<const ssize_t> BroadcastToNode::shape() const {
    return std::span<const ssize_t>(shape_.get(), ndim_);
}

std::span<const ssize_t> BroadcastToNode::shape(const State& state) const {
    if (!ndim_ || shape_[0] >= 0) return shape();  // not dynamic
    return data_ptr<DynamicBroadcastToNodeData>(state)->shape;
}

ssize_t BroadcastToNode::size() const {
    if (ndim_ && shape_[0] < 0) return -1;  // dynamic
    return std::accumulate(shape_.get(), shape_.get() + ndim_, 1, std::multiplies<ssize_t>());
}

ssize_t BroadcastToNode::size(const State& state) const {
    if (!ndim_ || shape_[0] >= 0) return size();  // not dynamic
    const auto& shape = data_ptr<DynamicBroadcastToNodeData>(state)->shape;
    assert(shape.size() > 0 && shape[0] >= 0);
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<ssize_t>());
}

ssize_t BroadcastToNode::size_diff(const State& state) const {
    if (!ndim_ || shape_[0] >= 0) return 0;  // not dynamic
    return data_ptr<DynamicBroadcastToNodeData>(state)->size_diff;
}

std::span<const ssize_t> BroadcastToNode::strides() const {
    return std::span<const ssize_t>(strides_.get(), ndim_);
}

std::vector<ssize_t> make_concatenate_shape(std::span<ArrayNode*> array_ptrs, ssize_t axis);

double const* ConcatenateNode::buff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->buff();
}

void ConcatenateNode::commit(State& state) const { data_ptr<ArrayNodeStateData>(state)->commit(); }

ConcatenateNode::ConcatenateNode(std::span<ArrayNode*> array_ptrs, const ssize_t axis)
        : ArrayOutputMixin(make_concatenate_shape(array_ptrs, axis)),
          axis_(axis),
          array_ptrs_(array_ptrs.begin(), array_ptrs.end()),
          values_info_(std::ranges::transform_view(array_ptrs,
                                                   [](auto ptr) -> const Array* { return ptr; })) {
    // Compute buffer start position for each input array
    array_starts_.reserve(array_ptrs.size());
    array_starts_.emplace_back(0);
    for (ssize_t arr_i = 1, stop = array_ptrs.size(); arr_i < stop; ++arr_i) {
        auto subshape = array_ptrs_[arr_i - 1]->shape().last(this->ndim() - axis_);
        ssize_t prod =
                std::accumulate(subshape.begin(), subshape.end(), 1, std::multiplies<ssize_t>());
        array_starts_.emplace_back(prod + array_starts_[arr_i - 1]);
    }

    for (auto it = array_ptrs.begin(), stop = array_ptrs.end(); it != stop; ++it) {
        if ((*it)->dynamic()) {
            throw std::invalid_argument("concatenate input arrays cannot be dynamic");
        }

        this->add_predecessor((*it));
    }
}

std::span<const Update> ConcatenateNode::diff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->diff();
}

void ConcatenateNode::initialize_state(State& state) const {
    std::vector<double> values;
    values.resize(size());

    for (ssize_t arr_i = 0, stop = array_ptrs_.size(); arr_i < stop; ++arr_i) {
        // Create a view into our buffer with the same shape as
        // our input array starting at the correct place
        auto view_it = Array::iterator(values.data() + array_starts_[arr_i], this->ndim(),
                                       array_ptrs_[arr_i]->shape().data(), this->strides().data());

        std::copy(array_ptrs_[arr_i]->begin(state), array_ptrs_[arr_i]->end(state), view_it);
    }

    emplace_data_ptr<ArrayNodeStateData>(state, std::move(values));
}

std::vector<ssize_t> make_concatenate_shape(std::span<ArrayNode*> array_ptrs, ssize_t axis) {
    // One or more arrays must be given
    if (array_ptrs.size() < 1) {
        throw std::invalid_argument("need at least one array to concatenate");
    }

    for (auto it = std::next(array_ptrs.begin()), stop = array_ptrs.end(); it != stop; ++it) {
        // Arrays must have the same number of dimensions
        if ((*std::prev(it))->ndim() != (*it)->ndim()) {
            throw std::invalid_argument(
                    "all the input arrays must have the same number of dimensions," +
                    std::string(" but the array at index ") +
                    std::to_string(std::distance(array_ptrs.begin(), std::prev(it))) + " has " +
                    std::to_string((*std::prev(it))->ndim()) +
                    " dimension(s) and the array at index " +
                    std::to_string(std::distance(array_ptrs.begin(), it)) + " has " +
                    std::to_string((*it)->ndim()) + " dimension(s)");
        }

        // Array shapes must be the same except for on the concatenation axis
        for (ssize_t i = 0, stop = (*it)->ndim(); i < stop; ++i) {
            if (i != axis) {
                if ((*std::prev(it))->shape()[i] != (*it)->shape()[i]) {
                    throw std::invalid_argument(
                            "all the input array dimensions except for the concatenation" +
                            std::string(" axis must match exactly, but along dimension ") +
                            std::to_string(i) + ", the array at index " +
                            std::to_string(std::distance(array_ptrs.begin(), std::prev(it))) +
                            " has size " + std::to_string((*std::prev(it))->shape()[i]) +
                            " and the array at index " +
                            std::to_string(std::distance(array_ptrs.begin(), it)) + " has size " +
                            std::to_string((*it)->shape()[i]));
                }
            }
        }
    }

    // Axis must be in range 0..ndim-1
    // We can do this check on the first input array since we at
    // this point know they all have the same number of dimensions
    if (!(0 <= axis && axis < array_ptrs.front()->ndim())) {
        throw std::invalid_argument("axis " + std::to_string(axis) +
                                    std::string(" is out of bounds for array of dimension ") +
                                    std::to_string(array_ptrs.front()->ndim()));
    }

    // The shape of the input arrays, which will be the
    // same except for possibly on the concatenation axis
    std::span<const ssize_t> shape0 = array_ptrs.front()->shape();
    std::vector<ssize_t> shape(shape0.begin(), shape0.end());

    // On the concatenation axis we sum the axis dimension sizes
    for (auto it = std::next(array_ptrs.begin()), stop = array_ptrs.end(); it != stop; ++it) {
        shape[axis] = shape[axis] + (*it)->shape()[axis];
    }

    return shape;
}

bool ConcatenateNode::integral() const { return values_info_.integral; }

double ConcatenateNode::min() const { return values_info_.min; }

double ConcatenateNode::max() const { return values_info_.max; }

void ConcatenateNode::propagate(State& state) const {
    auto ptr = data_ptr<ArrayNodeStateData>(state);

    for (ssize_t arr_i = 0, stop = array_ptrs_.size(); arr_i < stop; ++arr_i) {
        auto view_it =
                Array::const_iterator(ptr->buff() + array_starts_[arr_i], this->ndim(),
                                      array_ptrs_[arr_i]->shape().data(), this->strides().data());

        for (auto update : array_ptrs_[arr_i]->diff(state)) {
            assert(!update.placed() && !update.removed() && "no dynamic support implemented");
            auto update_it = view_it + update.index;
            ssize_t buffer_index = &*update_it - ptr->buff();
            assert(*update_it == update.old);
            ptr->set(buffer_index, update.value);
        }
    }
}

void ConcatenateNode::revert(State& state) const { data_ptr<ArrayNodeStateData>(state)->revert(); }

CopyNode::CopyNode(ArrayNode* array_ptr)
        : ArrayOutputMixin(array_ptr->shape()), array_ptr_(array_ptr), values_info_(array_ptr_) {
    this->add_predecessor(array_ptr);
}

double const* CopyNode::buff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->buff();
}

void CopyNode::commit(State& state) const { data_ptr<ArrayNodeStateData>(state)->commit(); }

std::span<const Update> CopyNode::diff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->diff();
}

void CopyNode::initialize_state(State& state) const {
    emplace_data_ptr<ArrayNodeStateData>(state, array_ptr_->view(state));
}

bool CopyNode::integral() const { return values_info_.integral; }

double CopyNode::min() const { return values_info_.min; }

double CopyNode::max() const { return values_info_.max; }

void CopyNode::propagate(State& state) const {
    data_ptr<ArrayNodeStateData>(state)->update(array_ptr_->diff(state));
}

void CopyNode::revert(State& state) const { data_ptr<ArrayNodeStateData>(state)->revert(); }

std::span<const ssize_t> CopyNode::shape(const State& state) const {
    return array_ptr_->shape(state);
}

ssize_t CopyNode::size(const State& state) const { return array_ptr_->size(state); }

ssize_t CopyNode::size_diff(const State& state) const { return array_ptr_->size_diff(state); }

// A PutNode needs to track its buffer as well as a mask of which elements in the
// original array are currently overwritten.
// We use ArrayStateData for the buffer
class PutNodeState : private ArrayStateData, public NodeStateData {
 public:
    explicit PutNodeState(std::vector<double>&& values, std::vector<ssize_t>&& mask) noexcept
            : ArrayStateData(std::move(values)), mask_(std::move(mask)) {}

    using ArrayStateData::buff;

    void commit() {
        assert(ambiguities_.empty() && "commiting with ambiguities still present!");
        ArrayStateData::commit();
        mask_diff_.clear();
    }

    std::unique_ptr<NodeStateData> copy() const override {
        return std::make_unique<PutNodeState>(*this);
    }

    // decrement the mask count at index and set the state to array_value if
    // the output is no longer masked
    void decrement_mask(ssize_t index, double array_value) {
        assert(0 <= index && static_cast<std::size_t>(index) < mask_.size());
        assert(mask_[index] >= 1);
        mask_[index] -= 1;
        mask_diff_.emplace_back(index, -1);
        if (mask_[index]) {
            // uh oh, we found a case where we are not certain what value
            // we should be propagating! So we save it as ambiguous so we
            // can fix it later
            ambiguities_.emplace(index);
        } else {
            // we've gone from masked to not, an easy case.
            ArrayStateData::set(index, array_value);
        }
    }

    using ArrayStateData::diff;

    // increment the mask count at index and set the state to masked_value
    void increment_mask(ssize_t index, double masked_value) {
        assert(0 <= index && static_cast<std::size_t>(index) < mask_.size());
        assert(mask_[index] >= 0);
        mask_[index] += 1;
        mask_diff_.emplace_back(index, +1);
        ArrayStateData::set(index, masked_value);
    }

    std::span<const ssize_t> mask() const { return mask_; }

    std::size_t num_ambiguities() const { return ambiguities_.size(); }

    // Sometimes we can be uncertain about some indices. So we save them as
    // ambiguities and do a big expensive repair at the end.
    void resolve_ambiguities(const Array::View& indices, const Array::View& values) {
        if (ambiguities_.empty()) return;  // nothing to do

        const auto first = indices.begin();
        const auto last = indices.end();

        for (const ssize_t& index : ambiguities_) {
            if (!mask_[index]) continue;  // the ambiguity was eventually resolved

            // go looking for this value in index. It's expensive.
            auto it = std::find(first, last, index);

            assert(it != last);  // this should have been covered by the mask check above

            // ok, we found a match, so now we need to look that value up in values
            ArrayStateData::set(index, values[it - first]);
        }

        ambiguities_.clear();
    }

    void revert() {
        assert(ambiguities_.empty() && "reverting with ambiguities still present!");
        ArrayStateData::revert();
        // doing it in reverse shouldn't matter, but it's a good habit
        for (const auto& [index, change] : mask_diff_ | std::views::reverse) {
            mask_[index] -= change;
        }
        mask_diff_.clear();
    }

    // Incorporate an update to the base array, ignoring it if the value is
    // currently masked
    void update_array(const Update& update) {
        assert(!update.placed() && "base array cannot be dynamic");
        assert(!update.removed() && "base array cannot be dynamic");
        assert(0 <= update.index && static_cast<std::size_t>(update.index) < mask_.size());

        // only update the array if the value is not masked
        if (!mask_[update.index]) ArrayStateData::set(update.index, update.value);
    }

    // update the output at index to masked_value if the array is masked
    void update_mask(ssize_t index, double masked_value) {
        assert(index >= 0);
        if (static_cast<std::size_t>(index) >= mask_.size())
            return;  // in this case they can be out of bounds

        if (mask_[index]) ArrayStateData::set(index, masked_value);
    }

 private:
    std::vector<ssize_t> mask_;
    std::vector<std::tuple<ssize_t, std::int8_t>> mask_diff_;

    // When a value is masked more than once, it can be ambiguous what happens
    // when only one of the masks is removed. In that case we save the index
    // in the array where we're not certain and do a repair before propagating.
    // We use a unordered_set to avoid rechecking the same index multiple times.
    std::unordered_set<ssize_t> ambiguities_;
};

PutNode::PutNode(ArrayNode* array_ptr, ArrayNode* indices_ptr, ArrayNode* values_ptr)
        : ArrayOutputMixin(array_ptr ? array_ptr->shape() : std::span<ssize_t>{}),
          array_ptr_(array_ptr),
          indices_ptr_(indices_ptr),
          values_ptr_(values_ptr),
          values_info_({array_ptr, values_ptr}) {
    if (!array_ptr_ || !indices_ptr_ || !values_ptr_) {
        throw std::invalid_argument("given ArrayNodes cannot be nullptr");
    }

    if (array_ptr_->dynamic()) {
        throw std::invalid_argument("array cannot be dynamic");
    }
    if (array_ptr_->ndim() < 1) {
        throw std::invalid_argument("array cannot be a scalar");
    }

    if (indices_ptr_->ndim() != 1 || values_ptr_->ndim() != 1) {
        throw std::invalid_argument("indices and values must be 1-dimensional");
    }
    if (!indices_ptr_->integral()) {
        throw std::invalid_argument("indices may not contain non-integer values");
    }

    if (!array_shape_equal(indices_ptr_, values_ptr_)) {
        throw std::invalid_argument(
                "shape mismatch: indices and values must always be the same size");
    }

    if (indices_ptr_->min() < 0) {
        throw std::out_of_range("indices may not be less than 0");
    }
    if (indices_ptr_->max() >= array_ptr_->size()) {
        throw std::out_of_range("indices may not exceed the size of the array");
    }

    add_predecessor(array_ptr);
    add_predecessor(indices_ptr);
    add_predecessor(values_ptr);
}

double const* PutNode::buff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->buff();
}

void PutNode::commit(State& state) const { return data_ptr<PutNodeState>(state)->commit(); }

std::span<const Update> PutNode::diff(const State& state) const {
    return data_ptr<PutNodeState>(state)->diff();
}

void PutNode::initialize_state(State& state) const {
    // Begin by copying the array
    std::vector<double> values(array_ptr_->begin(state), array_ptr_->end(state));

    // We also track how many times each index is currently overwritten
    std::vector<ssize_t> mask(values.size(), 0);

    // Now traverse the indices and the values, overwriting where necessary
    {
        // Indices and values should always be the same size. This is checked
        // by the constructor.
        assert(indices_ptr_->size(state) == values_ptr_->size(state));

        auto ind = indices_ptr_->begin(state);
        auto v = values_ptr_->begin(state);
        for (const auto end = indices_ptr_->end(state); ind != end; ++ind, ++v) {
            assert(0 <= *ind && *ind < values.size());
            values[*ind] = *v;
            mask[*ind] += 1;
        }
    }

    emplace_data_ptr<PutNodeState>(state, std::move(values), std::move(mask));
}

std::span<const ssize_t> PutNode::mask(const State& state) const {
    return data_ptr<PutNodeState>(state)->mask();
}

bool PutNode::integral() const { return values_info_.integral; }

double PutNode::min() const { return values_info_.min; }

double PutNode::max() const { return values_info_.max; }

void PutNode::propagate(State& state) const {
    auto ptr = data_ptr<PutNodeState>(state);

    // these should always be synced
    assert(indices_ptr_->size(state) == values_ptr_->size(state));

    // first go through the updates to the indices
    {
        auto array_iterator = array_ptr_->begin(state);
        auto values_iterator = values_ptr_->begin(state);

        const ssize_t values_size = values_ptr_->size(state);

        for (const Update& update : indices_ptr_->diff(state)) {
            // if the update is not a placement, that means that we're removing
            // update.old from the list of masked indices and thereby potentially
            // overwriting the value propagated by the PutNode
            if (!update.placed()) {
                assert(is_integer(update.old));
                const ssize_t array_index = update.old;
                assert(0 <= array_index && array_index < array_ptr_->size(state));

                ptr->decrement_mask(array_index, *(array_iterator + array_index));
            }

            // if the update is not a removal, that means that we're adding
            // update.value to the list of masked indices and thereby potentially
            // overwriting the value propagated by the PutNode
            if (!update.removed()) {
                assert(is_integer(update.value));
                const ssize_t array_index = update.value;
                assert(0 <= array_index && array_index < array_ptr_->size(state));

                if (update.index < values_size) {
                    ptr->increment_mask(array_index, *(values_iterator + update.index));
                } else {
                    // there is an edge case here where we grow and then shrink. In
                    // this case the mask value we're placing will get removed later
                    // so we just put a placeholder value in there
                    ptr->increment_mask(array_index, 0);
                }
            }
        }

        if (ptr->num_ambiguities()) {
            ptr->resolve_ambiguities(indices_ptr_->view(state), values_ptr_->view(state));
        }
    }

    // next go through the updates to the values. Some of these might be redundant
    // to the ones we already did for the indices, but that's OK
    {
        auto index_iterator = indices_ptr_->begin(state);

        for (const Update& update : values_ptr_->diff(state)) {
            if (update.removed()) continue;  // should have already been handled by the indexer
            assert(is_integer(*(index_iterator + update.index)));

            ptr->update_mask(*(index_iterator + update.index), update.value);
        }
    }

    // finally incorporate changes from the base array.
    for (const Update& update : array_ptr_->diff(state)) {
        assert(!update.placed() && "base array cannot be dynamic");
        assert(!update.removed() && "base array cannot be dynamic");
        ptr->update_array(update);
    }

    // if we have made any changes, then call our successor's update method(s)
    if (ptr->diff().size()) Node::propagate(state);
}

void PutNode::revert(State& state) const { return data_ptr<PutNodeState>(state)->revert(); }

// Reshape allows one shape dimension to be -1. In that case the size is inferred.
// We do that inference here.
std::vector<ssize_t> infer_reshape(Array* array_ptr, std::vector<ssize_t>&& shape) {
    // If the base array is dynamic, we might allow the first dimension to be -1.
    // So let's defer to the various constructors to check correctness.
    if (array_ptr->dynamic()) return shape;

    // Check if there are any -1s, and if not fallback to other input checking.
    auto it = std::ranges::find(shape, -1);
    if (it == shape.end()) return shape;

    // Get the product of the shape and negate it (to exclude the -1)
    auto prod = -std::reduce(shape.begin(), shape.end(), 1, std::multiplies<ssize_t>());

    // If the product is <=0, then we have another negative number or a 0. In which
    // case we just fall back to other error checking.
    if (prod <= 0) return shape;

    // Ok, we can officially overwrite the -1.
    // Don't worry about the case that prod doesn't divide array_ptr->size(), other
    // error checking will catch that case.
    *it = array_ptr->size() / prod;

    return shape;
}

class DynamicReshapeNodeData : public NodeStateData {
 public:
    // shape is the dynamic shape, i.e. leads with a -1
    // size is the actual size, used to infer the actual shape
    DynamicReshapeNodeData(std::span<const ssize_t> shape, ssize_t size)
            : shape_(shape.begin(), shape.end()),
              size_(size),
              row_size_(std::accumulate(shape_.begin() + 1, shape_.end(), 1,
                                        std::multiplies<ssize_t>())) {
        assert(size_ % row_size_ == 0);
        if (shape_.size()) shape_[0] = size_ / row_size_;
    }

    void commit() { old_size_ = size_; }

    void revert() { set_size(old_size_); }

    void set_size(ssize_t size) {
        assert(size % row_size_ == 0);
        size_ = size;
        if (shape_.size()) shape_[0] = size / row_size_;
    }

    std::span<const ssize_t> shape() const { return shape_; }

    ssize_t size() const { return size_; }

    ssize_t size_diff() const { return size_ - old_size_; }

 private:
    std::vector<ssize_t> shape_;
    ssize_t size_;
    ssize_t old_size_ = size_;

    ssize_t row_size_;
};

ReshapeNode::ReshapeNode(ArrayNode* node_ptr, std::vector<ssize_t>&& shape)
        : ArrayOutputMixin(infer_reshape(node_ptr, std::move(shape))),
          array_ptr_(node_ptr),
          values_info_(array_ptr_),
          sizeinfo_(array_ptr_->sizeinfo()) {
    // Don't (yet) support non-contiguous predecessors.
    // In some cases with non-contiguous predecessors we need to make a copy.
    // See https://github.com/dwavesystems/dwave-optimization/issues/16
    // There are also cases where we want reshape non-contiguous nodes.
    if (!array_ptr_->contiguous()) {
        throw std::invalid_argument("cannot reshape a non-contiguous array");
    }

    if (array_ptr_->dynamic()) {
        // We allow reshaping dyanamic arrays if the size of each "row" of the new shape evenly
        // divides the size of each "row" of the old. E.g.,
        // * (-1, 2) -> (-1,)
        // * (-1, 4) -> (-1, 2)
        // * (-1, 4) -> (-1, 2, 2)
        // are all OK, but
        // * (-1, 2) -> (-1, 4)
        // is not.

        if (!this->dynamic()) {
            throw std::invalid_argument("cannot reshape a dynamic array to a fixed size");
        }

        const auto array_shape = array_ptr_->shape();
        const auto new_shape = this->shape();

        ssize_t array_row_size = std::reduce(array_shape.begin() + 1, array_shape.end(), 1,
                                             std::multiplies<ssize_t>());
        ssize_t new_row_size =
                std::reduce(new_shape.begin() + 1, new_shape.end(), 1, std::multiplies<ssize_t>());

        if (array_row_size % new_row_size) {
            throw std::invalid_argument("cannot reshape array of shape " +
                                        shape_to_string(array_shape) + " into shape " +
                                        shape_to_string(new_shape));
        }
    } else if (this->dynamic()) {
        // If our base array is not dynamic but the user is trying to make it dynamic
        throw std::invalid_argument("cannot reshape to a dynamic array");
    }

    // one -1 was already replaced by infer_shape
    if (std::ranges::any_of(this->shape() | std::views::drop(1),
                            [](const ssize_t& dim) { return dim < 0; })) {
        throw std::invalid_argument("can only specify one unknown dimension");
    }

    // works for dynamic as well
    if (this->size() != array_ptr_->size()) {
        // Use the same error message as NumPy
        throw std::invalid_argument("cannot reshape array of size " +
                                    std::to_string(array_ptr_->size()) + " into shape " +
                                    shape_to_string(this->shape()));
    }

    this->add_predecessor(node_ptr);
}

double const* ReshapeNode::buff(const State& state) const { return array_ptr_->buff(state); }

void ReshapeNode::commit(State& state) const {
    if (!this->dynamic()) return;  // stateless
    return data_ptr<DynamicReshapeNodeData>(state)->commit();
}

std::span<const Update> ReshapeNode::diff(const State& state) const {
    return array_ptr_->diff(state);
}

void ReshapeNode::initialize_state(State& state) const {
    if (!this->dynamic()) return Node::initialize_state(state);  // stateless
    emplace_data_ptr<DynamicReshapeNodeData>(state, this->shape(), array_ptr_->size(state));
}

bool ReshapeNode::integral() const { return values_info_.integral; }

double ReshapeNode::min() const { return values_info_.min; }

double ReshapeNode::max() const { return values_info_.max; }

void ReshapeNode::propagate(State& state) const {
    if (!this->dynamic()) return;  // stateless
    data_ptr<DynamicReshapeNodeData>(state)->set_size(array_ptr_->size(state));
}

void ReshapeNode::revert(State& state) const {
    if (!this->dynamic()) return;  // stateless
    return data_ptr<DynamicReshapeNodeData>(state)->revert();
}

std::span<const ssize_t> ReshapeNode::shape(const State& state) const {
    if (!this->dynamic()) return this->shape();  // stateless
    return data_ptr<DynamicReshapeNodeData>(state)->shape();
}

ssize_t ReshapeNode::size(const State& state) const {
    if (!this->dynamic()) return this->size();  // stateless
    return data_ptr<DynamicReshapeNodeData>(state)->size();
}

SizeInfo ReshapeNode::sizeinfo() const { return this->sizeinfo_; }

ssize_t ReshapeNode::size_diff(const State& state) const {
    if (!this->dynamic()) return 0;  // stateless
    return data_ptr<DynamicReshapeNodeData>(state)->size_diff();
}

bool is_fill_never_used(const Array* array_ptr, ssize_t this_size) {
    // determine whether our fill value will ever matter. We need this info later for
    // min/max/integral. We could do this lazily, but let's just be proactive for
    // simplicity
    if (array_ptr->dynamic()) {
        // We'll never use the fill value if the minimum size of our predecessor
        // array is greater than or equal to our size.
        const ssize_t min_size = array_ptr->sizeinfo().substitute(100).min.value_or(0);
        return min_size >= this_size;
    }

    // If our predecessor is not dynamic, then this is simple to figure out
    return array_ptr->size() >= this_size;
}

ValuesInfo resize_compute_values_info(const Array* array_ptr, ssize_t size, double fill_value) {
    bool fill_never_used = is_fill_never_used(array_ptr, size);

    bool integral = (fill_never_used || is_integer(fill_value)) && array_ptr->integral();

    auto min = array_ptr->min();
    auto max = array_ptr->max();

    if (!fill_never_used) {
        min = std::min(min, fill_value);
        max = std::max(max, fill_value);
    }

    return {min, max, integral};
}

ResizeNode::ResizeNode(ArrayNode* array_ptr, std::vector<ssize_t>&& shape, double fill_value)
        : ArrayOutputMixin(shape),
          array_ptr_(array_ptr),
          fill_value_(fill_value),
          values_info_(resize_compute_values_info(array_ptr_, this->size(), fill_value_)) {
    // our incoming array can be any shape/size, but we cannot be dynamic
    if (this->dynamic()) throw std::invalid_argument("cannot resize to a dynamic shape");

    add_predecessor(array_ptr);
}

double const* ResizeNode::buff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->buff();
}

void ResizeNode::commit(State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->commit();
}

std::span<const Update> ResizeNode::diff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->diff();
}

void ResizeNode::initialize_state(State& state) const {
    const ssize_t size = this->size();  // the desired size of our state
    assert(size >= 0);                  // we're never dynamic

    std::vector<double> values;
    values.reserve(size);

    // Fill in from our predecessor, up to our size.
    // In c++23 we could use append_range(...) which would be nicer.
    for (const auto& v : array_ptr_->view(state) | std::views::take(size)) {
        values.emplace_back(v);
    }

    // Now fill in everything else with our fill value
    assert(!is_fill_never_used(array_ptr_, size) ||
           values.size() == static_cast<std::size_t>(size));
    values.resize(size, fill_value_);

    // Finally create the state
    emplace_data_ptr<ArrayNodeStateData>(state, std::move(values));
}

bool ResizeNode::integral() const { return values_info_.integral; }

double ResizeNode::min() const { return values_info_.min; }

double ResizeNode::max() const { return values_info_.max; }

void ResizeNode::propagate(State& state) const {
    const ssize_t size = this->size();  // the desired size of our state
    assert(size >= 0);                  // we're never dynamic

    auto data_ptr = this->data_ptr<ArrayNodeStateData>(state);
    assert(data_ptr);  // should never be nullptr

    for (const Update& update : array_ptr_->diff(state)) {
        const auto& [index, _, value] = update;

        if (index >= size) continue;  // don't care, it's out of range

        if (update.removed()) {
            // replace anything that was removed with our fill value
            data_ptr->set(index, fill_value_);
        } else {
            // either a placement or a change. We don't care either way!
            data_ptr->set(index, value);
        }
    }

    if (data_ptr->diff().size()) Node::propagate(state);
}

void ResizeNode::revert(State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->revert();
}

// Return the lhs % rhs, but always as a positive number
ssize_t positive_modulus_(ssize_t lhs, ssize_t rhs) {
    lhs %= rhs;
    if (lhs < 0) lhs += rhs;
    return lhs;
}

RollNode::RollNode(ArrayNode* array_ptr, ssize_t shift, std::vector<ssize_t> axis)
        : RollNode(array_ptr, std::vector<ssize_t>{shift}, std::move(axis)) {}

RollNode::RollNode(ArrayNode* array_ptr, std::vector<ssize_t> shift, std::vector<ssize_t> axis)
        : ArrayOutputMixin(array_ptr->shape()),
          array_ptr_(array_ptr),
          shift_(std::move(shift)),
          axis_(std::move(axis)),
          values_info_(array_ptr),
          sizeinfo_(array_ptr_->sizeinfo()) {
    // we moved the argument shift so let's get it back as a reference
    std::vector<ssize_t>& shift_ref = std::get<std::vector<ssize_t>>(shift_);

    if (axis_.empty()) {
        // If the axis is empty then we're shifting as a flat array and therefore only
        // want a single shift value
        if (shift_ref.size() != 1) {
            throw std::invalid_argument("unexpected number of shifts (" +
                                        std::to_string(shift_ref.size()) + "), expected 1");
        }
    } else if (shift_ref.size() == 1) {
        // we're broadcasting to the axes in this case
    } else {
        // axis is not empty so it must be the same length as the shift
        if (shift_ref.size() != axis_.size()) {
            throw std::invalid_argument("shift and axis must have the same length");
        }
    }

    // axis and shift are consistent, but is axis consistent with our array?
    for (const ssize_t& ax : axis_) {
        if (ax < 0 or array_ptr_->ndim() <= ax) throw std::invalid_argument("axis out of bounds");
    }

    add_predecessor(array_ptr);
}

RollNode::RollNode(ArrayNode* array_ptr, ArrayNode* shift_ptr, std::vector<ssize_t> axis)
        : ArrayOutputMixin(array_ptr->shape()),
          array_ptr_(array_ptr),
          shift_(shift_ptr),
          axis_(std::move(axis)),
          values_info_(array_ptr),
          sizeinfo_(array_ptr_->sizeinfo()) {
    if (shift_ptr->dynamic()) throw std::invalid_argument("shift may not be dynamic");
    if (shift_ptr->ndim() >= 2) throw std::invalid_argument("shift must be 0 or 1 dimensional");

    if (axis_.empty()) {
        if (shift_ptr->size() != 1) {
            throw std::invalid_argument("unexpected number of shifts (" +
                                        std::to_string(shift_ptr->size()) + "), expected 1");
        }
    } else if (shift_ptr->size() == 1) {
        // we're broadcasting to the axes in this case
    } else {
        if (shift_ptr->size() != static_cast<ssize_t>(axis.size())) {
            throw std::invalid_argument("shift and axis must have the same length");
        }
    }

    // axis and shift are consistent, but is axis consistent with our array?
    for (const ssize_t& ax : axis_) {
        if (ax < 0 or array_ptr_->ndim() <= ax) throw std::invalid_argument("axis out of bounds");
    }

    add_predecessor(array_ptr);
    add_predecessor(shift_ptr);
}

std::span<const ssize_t> RollNode::axes() const { return axis_; }

double const* RollNode::buff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->buff();
}

void RollNode::commit(State& state) const { data_ptr<ArrayNodeStateData>(state)->commit(); }

std::span<const Update> RollNode::diff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->diff();
}

void RollNode::initialize_state(State& state) const {
    // Get the predecessor's state as a vector that will become our state.
    std::vector<double> buffer(array_ptr_->begin(state), array_ptr_->end(state));

    if (axis_.empty()) {
        // We're shifting everything as a flat array
        const auto [shift, changed] = shift_diff_(state);
        assert(not changed);  // not in a propagation
        rotate_(buffer, shift);
    } else {
        const auto [shifts, changed] = shifts_diff_(state);
        assert(not changed);  // not in a propagation
        rotate_(buffer, shape(state), shifts);
    }

    emplace_data_ptr<ArrayNodeStateData>(state, std::move(buffer));
}

bool RollNode::integral() const { return values_info_.integral; }

double RollNode::min() const { return values_info_.min; }

double RollNode::max() const { return values_info_.max; }

void RollNode::propagate(State& state) const {
    ArrayNodeStateData* const state_ptr = data_ptr<ArrayNodeStateData>(state);

    if (axis_.empty()) {
        // We're shifting everything as a flat array
        const auto [shift, changed] = shift_diff_(state);
        const ssize_t size = array_ptr_->size(state);

        if (not changed and array_ptr_->size_diff(state) == 0) {
            // if neither our size nor our shift changed, then we just need to
            // update the individual indices and it is efficient to do so.

            auto transform = [&shift, &size](Update update) -> Update {
                update.index = positive_modulus_(update.index + shift, size);
                return update;
            };

            if (array_ptr_->dynamic()) {
                // One last case we need to worry about. If the array may have grown and then
                // shrunk, we need to deduplicate the diff.
                // We'd really like to use some of the nice C++20 range stuff and
                // deduplicate_diff_view(), but alas some of the old Python images
                // don't support deduplicate_diff_view with std::ranges::transform.
                // So we have to do it manually with a copy.
                auto diff = array_ptr_->diff(state);
                std::vector<Update> updates(diff.begin(), diff.end());
                deduplicate_diff(updates);
                state_ptr->update(updates | std::views::transform(transform));
            } else {
                // Otherwise we just propagate the diff like normal under the assumption
                // that our predecessor was efficient (not always true but nice to believe).
                state_ptr->update(std::ranges::transform_view(array_ptr_->diff(state), transform));
            }
        } else {
            // Either our size or our shift has changed, so we might as well re-calculate
            // the whole thing.
            // dev note: in the case that our size changed, we could save some effort here
            // but for now I am going with the simple and safe approach.
            std::vector<double> buffer(array_ptr_->begin(state), array_ptr_->end(state));
            rotate_(buffer, shift);
            state_ptr->assign(buffer);
        }
    } else {
        // Get the shifts
        const auto [shifts, changed] = shifts_diff_(state);

        if (not changed and array_ptr_->size_diff(state) == 0) {
            const auto shape = array_ptr_->shape(state);

            auto transform = [&shape, &shifts](Update update) -> Update {
                ssize_t multiplier = 1;

                for (ssize_t axis = shape.size() - 1; axis >= 0; --axis) {
                    // get the amount we wish to shift the index by
                    const ssize_t shift = shifts[axis] * multiplier;

                    // but that shift needs to happen within the current axis
                    // so we need the size of that area
                    const ssize_t size = shape[axis] * multiplier;

                    // now we shift the index, staying careful to stay "within"
                    // the current axis
                    auto res = std::div(update.index, size);

                    // res.quot * size is the index at the beginning of the current axis
                    // we then shift the res.rem modulus the size of the axis
                    update.index = res.quot * size + positive_modulus_(res.rem + shift, size);

                    // we can now increase the multiplier before going on
                    multiplier *= shape[axis];
                }

                return update;
            };

            // Unlike the flat shift case we always deduplicate because each shift
            // operation is relatively expensive.
            // See note there about the use of deduplicate_diff
            auto diff = array_ptr_->diff(state);
            std::vector<Update> updates(diff.begin(), diff.end());
            deduplicate_diff(updates);
            state_ptr->update(updates | std::views::transform(transform));
        } else {
            // Either our size or our shifts have changed, so we might as well re-calculate
            // the whole thing.
            std::vector<double> buffer(array_ptr_->begin(state), array_ptr_->end(state));
            rotate_(buffer, shape(state), shifts);
            state_ptr->assign(buffer);
        }
    }
}

void RollNode::revert(State& state) const { data_ptr<ArrayNodeStateData>(state)->revert(); }

// Act on the given array *in place*.
void RollNode::rotate_(std::span<double> array, ssize_t shift) {
    // we want to turn shift into a number >=0,<array.size()
    shift = positive_modulus_(shift, array.size());

    // if the net is no shift, then exit early
    if (shift == 0) return;

    // for consistency wiht NumPy we do a right rotation
    std::rotate(array.rbegin(), array.rbegin() + shift, array.rend());
}

void RollNode::rotate_(std::span<double> array, std::span<const ssize_t> shape,
                       std::span<const ssize_t> shifts) {
    // make sure our inputs are all consistent with eachother as a sanity check
    assert(shifts.size() == shape.size());
    assert(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies()) ==
           static_cast<ssize_t>(array.size()));

    const ssize_t array_size = array.size();

    ssize_t multiplier = 1;
    for (ssize_t axis = shape.size() - 1; axis >= 0; --axis) {
        // get the amount we wish to shift each sub-array by
        const ssize_t shift = shifts[axis] * multiplier;

        // also the size of each sub-array that we'll be shifting
        const ssize_t roll_size = shape[axis] * multiplier;

        for (ssize_t offset = 0; offset < array_size; offset += roll_size) {
            rotate_(array.subspan(offset, roll_size), shift);
        }

        multiplier *= shape[axis];
    }
}

std::span<const ssize_t> RollNode::shape(const State& state) const {
    if (not this->dynamic()) return this->shape();
    return array_ptr_->shape(state);  // same as predecessor
}

const std::variant<const Array*, std::vector<ssize_t>>& RollNode::shift() const { return shift_; }

std::tuple<ssize_t, bool> RollNode::shift_diff_(const State& state) const {
    assert(axis_.empty());

    if (std::holds_alternative<const Array*>(shift_)) {
        const Array* const shift_ptr = std::get<const Array*>(shift_);
        assert(shift_ptr->size() == 1);

        // first get the shift
        const ssize_t shift = shift_ptr->view(state)[0];

        // next see if there has been a change since last time
        const auto diff = shift_ptr->diff(state);

        return {shift, not(diff.empty() or diff[0].old == shift)};
    }
    assert(std::holds_alternative<std::vector<ssize_t>>(shift_));
    const auto& shift = std::get<std::vector<ssize_t>>(shift_);
    assert(shift.size() == 1);
    return {shift[0], false};
}

std::tuple<std::vector<ssize_t>, bool> RollNode::shifts_diff_(const State& state) const {
    assert(not axis_.empty());  // not defined when shifting as a flat array

    std::vector<ssize_t> shifts(ndim(), 0);

    if (std::holds_alternative<const Array*>(shift_)) {
        const Array* const shifts_ptr = std::get<const Array*>(shift_);
        const auto view = shifts_ptr->view(state);

        if (shifts_ptr->size() == 1) {
            // broadcasting
            for (ssize_t i = 0, stop = axis_.size(); i < stop; ++i) {
                shifts[axis_[i]] += view[0];
            }
        } else {
            for (ssize_t i = 0, stop = axis_.size(); i < stop; ++i) {
                shifts[axis_[i]] += view[i];
            }
        }

        // If there is any diff, we assume that at least one of our indices
        // have changed
        return {shifts, not shifts_ptr->diff(state).empty()};
    }

    const std::vector<ssize_t>& shifts_ref = std::get<std::vector<ssize_t>>(shift_);

    if (shifts_ref.size() == 1) {
        for (ssize_t i = 0, stop = axis_.size(); i < stop; ++i) {
            shifts[axis_[i]] += shifts_ref[0];
        }
    } else {
        for (ssize_t i = 0, stop = axis_.size(); i < stop; ++i) {
            shifts[axis_[i]] += shifts_ref[i];
        }
    }

    return {shifts, false};
}

ssize_t RollNode::size(const State& state) const {
    if (ssize_t size = this->size(); size >= 0) return size;
    return array_ptr_->size(state);  // same size as predecessor always
}

SizeInfo RollNode::sizeinfo() const { return sizeinfo_; }

ssize_t RollNode::size_diff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->size_diff();
}

SizeNode::SizeNode(ArrayNode* node_ptr)
        : array_ptr_(node_ptr),
          minmax_(array_ptr_->sizeinfo().min.value_or(0),
                  array_ptr_->sizeinfo().max.value_or(std::numeric_limits<ssize_t>::max())) {
    this->add_predecessor(node_ptr);
}

void SizeNode::initialize_state(State& state) const {
    emplace_state(state, array_ptr_->size(state));
}

double SizeNode::min() const { return minmax_.first; }

double SizeNode::max() const { return minmax_.second; }

void SizeNode::propagate(State& state) const { set_state(state, array_ptr_->size(state)); }

// TransposeNode **************************************************************

ArrayNode* TransposeNode::predeccesor_check_(ArrayNode* array_ptr) const {
    // Can take the transpose of a dynamic vector but not a dyanmic (>=2)-D
    // array since the latter would result in a dynamic array which is dynamic
    // in an axis other than the 0th axis.
    if ((array_ptr->ndim() >= 2) && (array_ptr->dynamic())) {
        throw std::invalid_argument("Cannot take transpose of a dynamic (>=2)-D Array");
    }
    return array_ptr;
}

// a TransposeNodes shape and strides are the reverse of its predecessor
std::unique_ptr<ssize_t[]> reverse_span_helper(const std::span<const ssize_t> span,
                                               const ssize_t size) {
    std::unique_ptr<ssize_t[]> reverse_span = std::make_unique<ssize_t[]>(size);
    std::reverse_copy(span.begin(), span.end(), reverse_span.get());
    return reverse_span;
}

TransposeNode::TransposeNode(ArrayNode* array_ptr)
        : array_ptr_(predeccesor_check_(array_ptr)),
          ndim_(array_ptr->ndim()),
          shape_(reverse_span_helper(array_ptr->shape(), ndim_)),
          strides_(reverse_span_helper(array_ptr->strides(), ndim_)),
          contiguous_(Array::is_contiguous(ndim_, shape_.get(), strides_.get())),
          values_info_(array_ptr) {
    add_predecessor(array_ptr);
}

// this node simply points to the predecessor buff
double const* TransposeNode::buff(const State& state) const { return array_ptr_->buff(state); }

ssize_t TransposeNode::ndim() const { return ndim_; }

std::span<const ssize_t> TransposeNode::shape(const State& state) const {
    if (ndim_ <= 1) {  // predecessor is vector and may be dynamic
        return array_ptr_->shape(state);
    }
    // predecessor is (>=2)-D array and shape is static
    return std::span<const ssize_t>(shape_.get(), ndim_);
}

std::span<const ssize_t> TransposeNode::shape() const {
    if (ndim_ <= 1) {  // predecessor is vector and may be dynamic
        return array_ptr_->shape();
    }
    // predecessor is (>=2)-D array and shape is fixed
    return std::span<const ssize_t>(shape_.get(), ndim_);
}

std::span<const ssize_t> TransposeNode::strides() const {
    return std::span<const ssize_t>(strides_.get(), ndim_);
}

ssize_t TransposeNode::size() const { return array_ptr_->size(); }

ssize_t TransposeNode::size(const State& state) const { return array_ptr_->size(state); }

double TransposeNode::min() const { return values_info_.min; }

double TransposeNode::max() const { return values_info_.max; }

bool TransposeNode::integral() const { return values_info_.integral; }

bool TransposeNode::contiguous() const { return contiguous_; }

// For saving the diff data on the node
class TransposeNodeDiffData : public NodeStateData {
 public:
    TransposeNodeDiffData() {}

    void commit() { diff.clear(); }
    void revert() { diff.clear(); }

    std::vector<Update> diff;
};

std::span<const Update> TransposeNode::diff(const State& state) const {
    // If the predecessor is a vector, the transpose does nothing and the diff
    // of this node is simply the diff of the predecessor node.
    if (ndim_ <= 1) {  // predecessor is vector
        return array_ptr_->diff(state);
    }
    // Otherwise, we use the stored diff data.
    return data_ptr<TransposeNodeDiffData>(state)->diff;
}

ssize_t TransposeNode::size_diff(const State& state) const { return array_ptr_->size_diff(state); }

void TransposeNode::initialize_state(State& state) const {
    if (ndim_ <= 1) {
        return Node::initialize_state(state);  // stateless
    }
    // Construct diff data if predecessor is (>=2)-D array
    emplace_data_ptr<TransposeNodeDiffData>(state);
}

Update TransposeNode::convert_predecessor_update_(Update update) const {
    if (ndim_ <= 1) {  // predecessor is vector
        return update;
    }

    const std::span<const ssize_t> array_shape = array_ptr_->shape();
    ssize_t transpose_flat_index = 0;
    // when constructing a flat index of the transpose, it is helpful to know
    // the # of indices contributed when you move along a fixed axes.
    // `transpose_axis_index_stride` is initialized by the # of indices
    // contributed when moving along the 0th axis of the transpose.
    ssize_t transpose_axis_index_stride = std::accumulate(
            array_shape.begin(), array_shape.end() - 1, 1, std::multiplies<ssize_t>());

    // traverse the predecessor axes in backward (reverse) order and the
    // transpose axes in forward order
    for (ssize_t i = ndim_ - 1; i >= 0; --i) {
        // grab predecessor shape along the ith axis
        const ssize_t axis_shape = array_shape[i];
        assert(0 <= axis_shape &&
               "all dimensions of (>=2)-D array must be non-negative for transpose operation");
        // determine the multidimensional index of `flat_index` along the ith
        // axis of predecessor. Note: this is the multidimensional index along
        // the (ndim_ - 1 - i)th axis of the transpose
        const ssize_t multidimensional_index = update.index % axis_shape;
        // reassign flat_index to the correct index along the (i - 1)th axes of predecessor
        update.index /= axis_shape;

        // weight the multidimensional index along the (ndim_ - 1 - i)th axis
        // of the transpose by # of indices contributed by moving along axis
        transpose_flat_index += multidimensional_index * transpose_axis_index_stride;

        // recall we are traversing the tranpose axes in forward order.
        // the # of indices contributed by moving along the (ndim - 2 - i)th
        // axis is the same as (the # of indices contributed by moving along the
        // (ndim_ - 1 - i)th axis) / shape(ndim_ - i - 1)
        transpose_axis_index_stride /= array_shape[ndim_ - i - 1];
    }

    update.index = transpose_flat_index;

    return update;
}

void TransposeNode::propagate(State& state) const {
    const std::span<const Update> array_diff = array_ptr_->diff(state);

    if (array_diff.empty() || ndim_ <= 1) {
        return;  // Nothing to do or predecessor is vector (transpose of vector is vector)
    }

    // Predecessor is a non-dynamic (>=2)-D array.
    std::vector<Update>& transpose_diff = data_ptr<TransposeNodeDiffData>(state)->diff;
    assert(transpose_diff.size() == 0);
    transpose_diff.reserve(array_ptr_->size_diff(state));

    for (const Update& u : array_diff) {
        assert(([&]() {
            // make a copy of the update
            Update u_copy = u;
            // convert flat index of predecessor update to multidimensional indices
            std::vector<ssize_t> multi_index = unravel_index(u_copy.index, array_ptr_->shape());
            // reverse multidimensional indices to obtain the multidimensional
            // transpose indices
            std::reverse(multi_index.begin(), multi_index.end());
            // convert multidimensional transpose indices to transpose flat index
            // and check conversion
            return ravel_multi_index(multi_index, this->shape()) ==
                   convert_predecessor_update_(u_copy).index;
        })());
        // Make a copy of the update and convert the index to the respective
        // transpose index
        transpose_diff.emplace_back(convert_predecessor_update_(u));
    }
}

void TransposeNode::commit(State& state) const {
    if (ndim_ > 1) {
        data_ptr<TransposeNodeDiffData>(state)->commit();
    }  // otherwise, stateless
};

void TransposeNode::revert(State& state) const {
    if (ndim_ > 1) {
        data_ptr<TransposeNodeDiffData>(state)->revert();
    }  // otherwise, stateless
}

}  // namespace dwave::optimization

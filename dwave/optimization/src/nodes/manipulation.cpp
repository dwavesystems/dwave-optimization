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

#include <tuple>
#include <unordered_set>

#include "dwave-optimization/nodes/manipulation.hpp"

#include "_state.hpp"

namespace dwave::optimization {

std::vector<ssize_t> make_concatenate_shape(std::span<ArrayNode*> array_ptrs, ssize_t axis);

double const* ConcatenateNode::buff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->buff();
}

void ConcatenateNode::commit(State& state) const { data_ptr<ArrayNodeStateData>(state)->commit(); }

ConcatenateNode::ConcatenateNode(std::span<ArrayNode*> array_ptrs, const ssize_t axis)
        : ArrayOutputMixin(make_concatenate_shape(array_ptrs, axis)),
          axis_(axis),
          array_ptrs_(array_ptrs.begin(), array_ptrs.end()) {
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

bool ConcatenateNode::integral() const {
    return std::ranges::all_of(array_ptrs_, [](ArrayNode* ptr){ return ptr->integral(); });
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

std::pair<double, double> ConcatenateNode::minmax(
        optional_cache_type<std::pair<double, double>> cache) const {
    return memoize(cache, [&]() {
        auto min = [&cache](const ArrayNode* ptr) { return ptr->minmax(cache).first; };
        auto max = [&cache](const ArrayNode* ptr) { return ptr->minmax(cache).second; };
        return std::make_pair(std::ranges::min(array_ptrs_ | std::views::transform(min)),
                              std::ranges::max(array_ptrs_ | std::views::transform(max)));
    });
}

void ConcatenateNode::propagate(State& state) const {
    auto ptr = data_ptr<ArrayNodeStateData>(state);

    for (ssize_t arr_i = 0, stop = array_ptrs_.size(); arr_i < stop; ++arr_i) {
        auto view_it = Array::iterator(ptr->buff() + array_starts_[arr_i], this->ndim(),
                                       array_ptrs_[arr_i]->shape().data(), this->strides().data());

        for (auto diff : array_ptrs_[arr_i]->diff(state)) {
            assert(!diff.placed() && !diff.removed() && "no dynamic support implemented");
            auto update_it = view_it + diff.index;
            ssize_t buffer_index = &*update_it - ptr->buffer.data();
            assert(*update_it == diff.old);
            ptr->updates.emplace_back(buffer_index, *update_it, diff.value);
            *update_it = diff.value;
        }
    }
}

void ConcatenateNode::revert(State& state) const { data_ptr<ArrayNodeStateData>(state)->revert(); }

CopyNode::CopyNode(ArrayNode* array_ptr)
        : ArrayOutputMixin(array_ptr->shape()), array_ptr_(array_ptr) {
    this->add_predecessor(array_ptr);
}

double const* CopyNode::buff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->buff();
}

void CopyNode::commit(State& state) const { data_ptr<ArrayNodeStateData>(state)->commit(); }

std::span<const Update> CopyNode::diff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->diff();
}

bool CopyNode::integral() const { return array_ptr_->integral(); }

void CopyNode::initialize_state(State& state) const {
    emplace_data_ptr<ArrayNodeStateData>(state, array_ptr_->view(state));
}

std::pair<double, double> CopyNode::minmax(
        optional_cache_type<std::pair<double, double>> cache) const {
    return memoize(cache, [&]() { return array_ptr_->minmax(cache); });
}

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
          values_ptr_(values_ptr) {
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

bool PutNode::integral() const {
    // Because our underlying storage medium is double, we need both sources of
    // values to be integral. If we had a typed array then we could coerce values
    // to match.
    return array_ptr_->integral() && values_ptr_->integral();
}

std::span<const ssize_t> PutNode::mask(const State& state) const {
    return data_ptr<PutNodeState>(state)->mask();
}

std::pair<double, double> PutNode::minmax(
        optional_cache_type<std::pair<double, double>> cache) const {
    return memoize(cache, [&]() {
        auto [alow, ahigh] = array_ptr_->minmax(cache);
        auto [vlow, vhigh] = values_ptr_->minmax(cache);
        return std::make_pair(std::min<double>(alow, vlow), std::max<double>(ahigh, vhigh));
    });
}

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
    // if the base array is dynamic, we might allow the first dimension to be negative
    // 1. So let's defer to the various constructors.
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

ReshapeNode::ReshapeNode(ArrayNode* node_ptr, std::vector<ssize_t>&& shape)
        : ArrayOutputMixin(infer_reshape(node_ptr, std::move(shape))), array_ptr_(node_ptr) {
    // Don't (yet) support non-contiguous predecessors.
    // In some cases with non-contiguous predecessors we need to make a copy.
    // See https://github.com/dwavesystems/dwave-optimization/issues/200
    // There are also cases where we want reshape non-contiguous nodes.
    if (!array_ptr_->contiguous()) {
        throw std::invalid_argument("cannot reshape a non-contiguous array");
    }

    // Don't (yet) support dynamic predecessors.
    // We could support reshaping "down", e.g. (-1, 2) -> (-1,).
    // But we cannot support reshaping "up", e.g. (-1,) -> (-1, 2).
    // This is because in that case we would need the predecessor to grow/shrink
    // by a multiple of two each time.
    if (array_ptr_->dynamic()) {
        throw std::invalid_argument("cannot reshape a dynamic array");
    }

    // NumPy let's you use -1 in exactly one axis which is then inferred from
    // the others. We could support that in the future, including the dynamic
    // case.
    if (this->dynamic()) {
        throw std::invalid_argument("cannot reshape to a dynamic array");
    }

    // one -1 was already replaced by infer_shape
    if (std::ranges::any_of(this->shape() | std::views::drop(1),
                            [](const ssize_t& dim) { return dim < 0; })) {
        throw std::invalid_argument("can only specify one unknown dimension");
    }

    if (this->size() != array_ptr_->size()) {
        // Use the same error message as NumPy
        throw std::invalid_argument("cannot reshape array of size " +
                                    std::to_string(array_ptr_->size()) + " into shape " +
                                    shape_to_string(this->shape()));
    }

    this->add_predecessor(node_ptr);
}

double const* ReshapeNode::buff(const State& state) const { return array_ptr_->buff(state); }

void ReshapeNode::commit(State& state) const {}  // stateless node

std::span<const Update> ReshapeNode::diff(const State& state) const {
    return array_ptr_->diff(state);
}

bool ReshapeNode::integral() const { return array_ptr_->integral(); }

std::pair<double, double> ReshapeNode::minmax(
        optional_cache_type<std::pair<double, double>> cache) const {
    return memoize(cache, [&]() { return array_ptr_->minmax(cache); });
}

void ReshapeNode::revert(State& state) const {}  // stateless node

class SizeNodeData : public ScalarNodeStateData {
 public:
    explicit SizeNodeData(std::integral auto value) : ScalarNodeStateData(value) {}
    void set(std::integral auto value) { ScalarNodeStateData::set(value); }
};

SizeNode::SizeNode(ArrayNode* node_ptr) : array_ptr_(node_ptr) { this->add_predecessor(node_ptr); }

double const* SizeNode::buff(const State& state) const {
    return data_ptr<SizeNodeData>(state)->buff();
}

void SizeNode::commit(State& state) const { return data_ptr<SizeNodeData>(state)->commit(); }

std::span<const Update> SizeNode::diff(const State& state) const {
    return data_ptr<SizeNodeData>(state)->diff();
}

void SizeNode::initialize_state(State& state) const {
    emplace_data_ptr<SizeNodeData>(state, array_ptr_->size(state));
}

std::pair<double, double> SizeNode::minmax(
        optional_cache_type<std::pair<double, double>> cache) const {
    return memoize(cache, [&]() {
        const ssize_t size = array_ptr_->size();
        if (size >= 0) return std::pair<double, double>(size, size);

        double low = array_ptr_->sizeinfo().min.value_or(0);
        double high = array_ptr_->sizeinfo().max.value_or(std::numeric_limits<ssize_t>::max());

        return std::make_pair(low, high);
    });
}

void SizeNode::propagate(State& state) const {
    return data_ptr<SizeNodeData>(state)->set(array_ptr_->size(state));
}

void SizeNode::revert(State& state) const { return data_ptr<SizeNodeData>(state)->revert(); }

}  // namespace dwave::optimization

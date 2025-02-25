// Copyright 2023 D-Wave Systems Inc.
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

#include "dwave-optimization/nodes/indexing.hpp"

#include <algorithm>
#include <ranges>
#include <unordered_set>

#include "_state.hpp"
#include "dwave-optimization/nodes/constants.hpp"
#include "dwave-optimization/state.hpp"
#include "dwave-optimization/utils.hpp"

namespace dwave::optimization {

// Generic data storage for nodes that index other nodes.
struct IndexingNodeData : NodeStateData {
    IndexingNodeData(std::vector<ssize_t>&& offsets, std::vector<double>&& values) noexcept
            : offsets(offsets), data(values), old_size(offsets.size()) {}

    void commit() {
        diff.clear();
        old_offsets.clear();
        old_size = offsets.size();
    }

    void revert() {
        // If we shrank, we need to undo the resizing
        if (old_size > static_cast<ssize_t>(offsets.size())) {
            offsets.resize(old_size);
            data.resize(old_size);
        }

        // Walk backwards through the diff/offsets and unapply them to the state
        auto it = old_offsets.rbegin();
        for (auto& [index, old, _] : diff | std::views::reverse) {
            offsets[index] = *it;
            data[index] = old;
            ++it;
        }

        // if we grew, we need to undo the resizing
        if (old_size < static_cast<ssize_t>(offsets.size())) {
            offsets.resize(old_size);
            data.resize(old_size);
        }

        // Clear the diff
        diff.clear();
        old_offsets.clear();
    }

    // Track the indices in the accessed array. These are stored as offsets
    // relative to the beginning of the array.
    std::vector<ssize_t> offsets;

    // The array values as a contiguous dataset
    std::vector<double> data;

    // Updates to the values, as well as the old indices so we can revert
    std::vector<Update> diff;
    std::vector<ssize_t> old_offsets;  // index linked to diff!
    ssize_t old_size;
};

// AdvancedIndexingNode *******************************************************

struct AdvancedIndexingNodeData : NodeStateData {
 public:
    AdvancedIndexingNodeData(std::vector<ssize_t>&& offsets, std::vector<double>&& values,
                             bool maintain_reverse_offset_map) noexcept
            : data(values),
              old_offsets_size(offsets.size()),
              old_data_size(data.size()),
              offsets_(offsets),
              maintain_reverse_offset_map(maintain_reverse_offset_map) {
        for (ssize_t idx = 0; idx < static_cast<ssize_t>(offsets_.size()); ++idx) {
            add_to_reverse(idx, offsets_[idx]);
        }
    }

    void commit() {
        diff.clear();
        offsets_diff.clear();
        old_offsets_size = offsets_.size();
        old_data_size = data.size();
    }

    void revert() {
        // If we shrank, we need to undo the resizing
        if (old_offsets_size > static_cast<ssize_t>(offsets_.size())) {
            offsets_.resize(old_offsets_size);
        }

        if (old_data_size > static_cast<ssize_t>(data.size())) {
            data.resize(old_data_size);
        }

        // Walk backwards through the diff/offsets and unapply them to the state
        for (const auto& update : offsets_diff | std::views::reverse) {
            offsets_[update.index] = update.old;

            if (!update.removed()) delete_from_reverse(update.index, update.value);
            if (!update.placed()) add_to_reverse(update.index, update.old);
        }

        for (auto& [index, old, _] : diff | std::views::reverse) {
            data[index] = old;
        }

        // If we grew, we need to undo the resizing
        if (old_offsets_size < static_cast<ssize_t>(offsets_.size())) {
            offsets_.resize(old_offsets_size);
        }

        if (old_data_size < static_cast<ssize_t>(data.size())) {
            data.resize(old_data_size);
        }

        // Clear the diff
        diff.clear();
        offsets_diff.clear();
    }

    void add_to_offset(ssize_t index, ssize_t delta) {
        ssize_t old_offset = offsets_[index];
        offsets_[index] += delta;
        offsets_diff.emplace_back(index, old_offset, offsets_[index]);

        delete_from_reverse(index, old_offset);
        add_to_reverse(index, offsets_[index]);
    }

    void place_offset(ssize_t offset) {
        offsets_diff.emplace_back(Update::placement(offsets_.size(), offset));
        add_to_reverse(offsets_.size(), offset);
        offsets_.push_back(offset);
    }

    void remove_offset() {
        assert(offsets_.size() > 0);
        ssize_t index = offsets_.size() - 1;
        offsets_diff.emplace_back(Update::removal(index, offsets_.back()));
        delete_from_reverse(index, offsets_.back());
        offsets_.pop_back();
    }

    ssize_t offsets_size() const { return offsets_.size(); }

    // May return nullptr to represent offset not tracked
    const std::vector<ssize_t>* get_offset_idxs(ssize_t offset) const {
        assert(maintain_reverse_offset_map && "reverse map must be enabled");

        const auto it = reverse_offsets.find(offset);
        if (it != reverse_offsets.end()) return &((*it).second);

        return nullptr;
    }

    const std::vector<ssize_t>& offsets() const { return offsets_; }

    // The array values as a contiguous dataset
    std::vector<double> data;

    // Updates to the values, as well as the old indices so we can revert
    std::vector<Update> diff;
    std::vector<Update> offsets_diff;
    ssize_t old_offsets_size;
    ssize_t old_data_size;

 private:
    void delete_from_reverse(ssize_t index, ssize_t offset) {
        if (!maintain_reverse_offset_map) return;

        const auto it = reverse_offsets.find(offset);
        assert(it != reverse_offsets.end());
        auto& idxs = (*it).second;
        if (idxs.back() != index) {
            auto idxs_it = std::find(idxs.begin(), idxs.end(), index);
            assert(idxs_it != idxs.end());
            *idxs_it = idxs.back();
        }
        idxs.pop_back();
        if (idxs.size() == 0) {
            reverse_offsets.erase(it);
        }
    }

    void add_to_reverse(ssize_t index, ssize_t offset) {
        if (!maintain_reverse_offset_map) return;

        reverse_offsets[offset].push_back(index);
    }

    // Track the indices in the accessed array. These are stored as offsets
    // relative to the beginning of the array.
    std::vector<ssize_t> offsets_;
    std::unordered_map<ssize_t, std::vector<ssize_t>> reverse_offsets;
    const bool maintain_reverse_offset_map;
};

struct AdvancedIndexingNode::IndexParser_ {
    IndexParser_(Array* array_ptr, std::vector<array_or_slice>&& indices)
            : indices_(std::move(indices)) {
        // This may happen if the dynamic_cast to Array from Node fails in the
        // AdvancedIndexingNode constructor
        if (array_ptr == nullptr) {
            throw std::invalid_argument("AdvancedIndexingNode must be given an Array to index");
        }

        // Confirm that we match the number of dimensions
        assert(array_ptr->ndim() >= 0);  // Should always be true
        if (static_cast<std::size_t>(array_ptr->ndim()) < indices_.size()) {
            // NumPy handles this case, we could as well
            throw std::invalid_argument(std::string("too few indices for array: array is ") +
                                        std::to_string(array_ptr->ndim()) + "-dimensional, but " +
                                        std::to_string(indices_.size()) + " were indexed");
        }
        if (static_cast<std::size_t>(array_ptr->ndim()) > indices_.size()) {
            throw std::invalid_argument(std::string("too many indices for array: array is ") +
                                        std::to_string(array_ptr->ndim()) + "-dimensional, but " +
                                        std::to_string(indices_.size()) + " were indexed");
        }

        if (array_ptr->ndim() == 0) return;

        bool encountered_array = false;
        bool encountered_slice_after_array = false;
        Array* first_array = nullptr;

        this->ndim = 0;
        this->subspace_stride = array_ptr->itemsize();

        for (size_t idx = 0; idx < indices_.size(); ++idx) {
            const array_or_slice& index = indices_[idx];
            this->ndim += std::holds_alternative<Slice>(index);

            if (std::holds_alternative<ArrayNode*>(index)) {
                ArrayNode* indexing_array_ptr = std::get<ArrayNode*>(index);

                // Test whether we can index with the given array. This is probably a bit
                // over-strict and may need to be loosened over time.

                // NumPy error message would be something like
                // IndexError: index -100 is out of bounds for axis 0 with size 4
                // so we attempt to match that as closely as possible

                if (!indexing_array_ptr->integral()) {
                    throw std::out_of_range(
                            std::string("index may not contain non-integer values for axis ") +
                            std::to_string(idx));
                }
                if (indexing_array_ptr->min() < 0) {
                    // todo: support negative values
                    auto msg = std::string("index's smallest possible value ") +
                               std::to_string(static_cast<ssize_t>(indexing_array_ptr->min())) +
                               std::string(" is out of bounds for axis ") + std::to_string(idx);

                    // if we're not dynamic then add the size information
                    if (array_ptr->shape()[idx] >= 0) {
                        msg += std::string(" with size ") + std::to_string(array_ptr->shape()[idx]);
                    }

                    throw std::out_of_range(msg);
                }
                if (idx == 0 && array_ptr->shape()[0] < 0) {
                    // 100 is a magic number
                    auto sizeinfo = array_ptr->sizeinfo().substitute(100);
                    if (!sizeinfo.min.has_value()) {
                        throw std::invalid_argument("indexed array has unknown minimum size");
                    }

                    // from the min size of the dynamic array, get the min size of the first axis
                    ssize_t min_axis_size = (sizeinfo.min.value() * array_ptr->itemsize()) /
                                            array_ptr->strides()[0];

                    if (indexing_array_ptr->max() >= min_axis_size) {
                        throw std::out_of_range(
                                std::string("index's largest possible value ") +
                                std::to_string(static_cast<ssize_t>(indexing_array_ptr->max())) +
                                std::string(" is out of bounds for axis ") + std::to_string(idx) +
                                std::string(" with minimum size ") + std::to_string(min_axis_size));
                    }
                } else if (indexing_array_ptr->max() >= array_ptr->shape()[idx]) {
                    throw std::out_of_range(
                            std::string("index's largest possible value ") +
                            std::to_string(static_cast<ssize_t>(indexing_array_ptr->max())) +
                            std::string(" is out of bounds for axis ") + std::to_string(idx) +
                            std::string(" with size ") + std::to_string(array_ptr->shape()[idx]));
                }

                ssize_t a_ndim = indexing_array_ptr->ndim();
                if (!encountered_array) {
                    indexing_arrays_ndim = a_ndim;
                    first_array = indexing_array_ptr;
                    first_array_index = idx;
                } else if (!array_shape_equal(first_array, indexing_array_ptr)) {
                    throw std::invalid_argument(
                            "shape mismatch: indexing arrays could not be broadcast together");
                } else if (a_ndim != indexing_arrays_ndim) {
                    throw std::invalid_argument(
                            "dimension mismatch: indexing arrays could not be broadcast together");
                }

                encountered_array = true;
                if (encountered_slice_after_array) {
                    // Encountered an array after a slice after an array
                    bullet1mode = true;
                }
            } else if (encountered_array) {
                encountered_slice_after_array = true;
            }
        }

        this->ndim += indexing_arrays_ndim;

        if (!encountered_array) {
            throw std::invalid_argument(
                    "there must be at least one array-type index to use AdvancedIndexingNode");
        }

        if (bullet1mode && array_ptr->dynamic() && std::holds_alternative<Slice>(indices_[0])) {
            throw std::invalid_argument(
                    "Indexed Array cannot be dynamic with an empty slice in the first dimension");
        }

        assert(first_array != nullptr);

        if (!bullet1mode && std::holds_alternative<ArrayNode*>(indices_[0])) {
            // Technically either mode can handle this case (all array indices grouped at
            // the start) but we will use bullet1mode to allow these arrays to be dynamic
            bullet1mode = true;
        }

        if (indexing_arrays_ndim == 0) {
            // Again, technically this could be the bullet 2 case, the output is the same
            // for both but bullet1mode can handle this more easily.
            bullet1mode = true;
        }

        if (indexing_arrays_ndim > 1) {
            throw std::invalid_argument(
                    "Advanced indexing currently only supports 0- and 1-dimensional indexing "
                    "arrays");
        }

        simple_array_strides = shape_to_strides(array_ptr->ndim(), array_ptr->shape().data());
        for (ssize_t i = 0; i < array_ptr->ndim(); ++i) {
            simple_array_strides[i] /= array_ptr->itemsize();
        }

        if (this->ndim > 0) {
            this->shape = std::make_unique<ssize_t[]>(ndim);

            if (bullet1mode) {
                for (ssize_t idx = 0; idx < first_array->ndim(); ++idx) {
                    this->shape[idx] = first_array->shape()[idx];
                }

                ssize_t slice_idx = first_array->ndim();
                for (size_t idx = 0; idx < indices_.size(); ++idx) {
                    if (std::holds_alternative<Slice>(indices_[idx])) {
                        assert(slice_idx < this->ndim);
                        assert(std::get<Slice>(indices_[idx]).empty());

                        // Could fit the shape here e.g.
                        // std::get<Slice>(indices_[idx]).fit(array_ptr->shape()[idx]).size();
                        // but will just assume it's an empty slice for now.
                        this->shape[slice_idx++] = array_ptr->shape()[idx];
                    }
                }
            } else {
                // bullet2mode
                if (indexing_arrays_ndim != 1) {
                    throw std::invalid_argument(
                            "Currently only 1d indexing arrays are supported when using in-place "
                            "combined indexing");
                }
                size_t slice_idx = 0;
                bool hit_first_array = false;
                for (size_t idx = 0; idx < indices_.size(); ++idx) {
                    if (std::holds_alternative<Slice>(indices_[idx])) {
                        assert(std::get<Slice>(indices_[idx]).empty());
                        // Could fit the shape here e.g.
                        // std::get<Slice>(indices_[idx]).fit(array_ptr->shape()[idx]).size();
                        // but will just assume it's an empty slice for now.

                        this->shape[slice_idx++] = array_ptr->shape()[idx];
                    } else if (!hit_first_array) {
                        assert(std::holds_alternative<ArrayNode*>(indices_[idx]));
                        hit_first_array = true;

                        // Fill in the intermediate shape which is the same as the shape of the
                        // indexing arrays
                        for (ssize_t indexing_array_axis = 0;
                             indexing_array_axis < indexing_arrays_ndim; ++indexing_array_axis) {
                            this->shape[slice_idx++] = first_array->shape()[indexing_array_axis];
                        }
                    }

                    if (std::holds_alternative<ArrayNode*>(indices_[idx]) &&
                        std::get<ArrayNode*>(indices_[idx])->dynamic()) {
                        throw std::invalid_argument(
                                "Indexing arrays cannot be dynamic when using in-place combined "
                                "indexing");
                    }
                }
            }

            strides = shape_to_strides(this->ndim, shape.get());
            if (indexing_arrays_ndim == 0) {
                subspace_stride = ndim > 0 ? strides[0] * shape[0] : array_ptr->itemsize();
            } else {
                subspace_stride = bullet1mode ? strides[indexing_arrays_ndim - 1]
                                              : strides[first_array_index];
            }
        }
    }

    // "Bullet 1 mode" refers to NumPy combined indexing where there is any slice between
    // indexing arrays, e.g. A[:, i, :, j]. It is a reference to the two cases laid out
    // in bullets in the NumPy docs here:
    // https://numpy.org/doc/stable/user/basics.indexing.html#combining-advanced-and-basic-indexing
    bool bullet1mode = false;
    std::vector<array_or_slice> indices_;
    ssize_t ndim = 0;
    ssize_t indexing_arrays_ndim = -1;
    ssize_t first_array_index;
    ssize_t subspace_stride;
    std::unique_ptr<ssize_t[]> strides = nullptr;
    std::unique_ptr<ssize_t[]> shape = nullptr;
    std::unique_ptr<ssize_t[]> simple_array_strides = nullptr;
};

AdvancedIndexingNode::AdvancedIndexingNode(ArrayNode* array_ptr,
                                           std::vector<array_or_slice> indices)
        : AdvancedIndexingNode(array_ptr, IndexParser_(array_ptr, std::move(indices))) {}

AdvancedIndexingNode::AdvancedIndexingNode(ArrayNode* array_ptr, IndexParser_&& parser)
        : array_ptr_(array_ptr),
          ndim_(parser.ndim),
          strides_(std::move(parser.strides)),
          shape_(std::move(parser.shape)),
          array_item_strides_(std::move(parser.simple_array_strides)),
          size_(Array::shape_to_size(ndim_, shape_.get())),
          indices_(std::move(parser.indices_)),
          indexing_arrays_ndim_(parser.indexing_arrays_ndim),
          bullet1mode_(parser.bullet1mode),
          first_array_index_(parser.first_array_index),
          subspace_stride_(parser.subspace_stride) {
    assert(!array_ptr->ndim() || array_item_strides_);

    // Now actually add them. This way if there is an error thrown we're not
    // causing segfaults
    add_predecessor(array_ptr);

    for (const array_or_slice& index : indices_) {
        if (std::holds_alternative<ArrayNode*>(index)) {
            add_predecessor(std::get<ArrayNode*>(index));
        }
    }

    if (dynamic()) {
        // Copy over the shape to dynamic_shape_. Then the first index of dynamic_shape_
        // will be rewritten as the output changes size.
        dynamic_shape_ = std::make_unique<ssize_t[]>(ndim());
        std::copy(shape_.get(), shape_.get() + ndim(), dynamic_shape_.get());
    }
}

double const* AdvancedIndexingNode::buff(const State& state) const {
    return data_ptr<AdvancedIndexingNodeData>(state)->data.data();
}
std::span<const Update> AdvancedIndexingNode::diff(const State& state) const {
    return data_ptr<AdvancedIndexingNodeData>(state)->diff;
}

void AdvancedIndexingNode::fill_subspace(State& state, ssize_t array_offset,
                                         std::vector<double>& data, ssize_t index_in_arrays) const {
    ssize_t starting_output_axis = bullet1mode_ ? indexing_arrays_ndim_ : 0;
    fill_subspace_recurse<false, false, false>(array_ptr_->shape(state), array_ptr_->buff(state),
                                               array_offset * static_cast<ssize_t>(itemsize()), 0,
                                               data, index_in_arrays * subspace_stride_,
                                               starting_output_axis, nullptr);
}

template <bool placement, bool removal>
void AdvancedIndexingNode::fill_subspace(State& state, ssize_t array_item_offset,
                                         std::vector<double>& data, ssize_t index_in_arrays,
                                         std::vector<Update>& diff) const {
    ssize_t starting_output_axis = bullet1mode_ ? indexing_arrays_ndim_ : 0;
    fill_subspace_recurse<true, placement, removal>(
            array_ptr_->shape(state), array_ptr_->buff(state),
            array_item_offset * static_cast<ssize_t>(itemsize()), 0, data,
            index_in_arrays * subspace_stride_, starting_output_axis, &diff);
}

template <bool placement, bool removal>
void AdvancedIndexingNode::fill_axis0_subspace(State& state, ssize_t axis0_index,
                                               ssize_t array_item_offset, std::vector<double>& data,
                                               ssize_t index_in_arrays,
                                               std::vector<Update>& diff) const {
    assert(!bullet1mode_ && "fill_axis0_subspace should only be called with bullet 2 mode");
    ssize_t array_offset = axis0_index * array_ptr_->strides()[0] + array_item_offset * itemsize();
    size_t data_offset = axis0_index * strides()[0] + index_in_arrays * subspace_stride_;
    fill_subspace_recurse<true, placement, removal>(array_ptr_->shape(state),
                                                    array_ptr_->buff(state), array_offset, 1, data,
                                                    data_offset, 1, &diff);
}

template <bool fill_diff, bool placement, bool removal>
void AdvancedIndexingNode::fill_subspace_recurse(const std::span<const ssize_t>& array_shape,
                                                 const double* array_buffer, ssize_t array_offset,
                                                 ssize_t array_dim, std::vector<double>& data,
                                                 ssize_t data_offset, size_t output_dim,
                                                 std::vector<Update>* diff) const {
    static_assert(!(!fill_diff && (placement || removal)));
    static_assert(!(placement && removal));

    if (array_dim == static_cast<ssize_t>(indices_.size())) {
        // Base case. Copy the value from `array_buffer` to `data`, updating the diff
        // as necessary. Or pop/push to the data array if placing/removing.
        ssize_t data_index = data_offset / static_cast<ssize_t>(itemsize());
        double new_val = array_buffer[array_offset / static_cast<ssize_t>(itemsize())];
        if constexpr (placement) {
            if constexpr (fill_diff) diff->emplace_back(Update::placement(data_index, new_val));
            assert(data_index == static_cast<ssize_t>(data.size()));
            data.push_back(new_val);
        } else if constexpr (removal) {
            assert(data_index == static_cast<ssize_t>(data.size() - 1));
            if constexpr (fill_diff)
                diff->emplace_back(Update::removal(data_index, data[data_index]));
            data.pop_back();
        } else {
            assert(data_index < static_cast<ssize_t>(data.size()));
            if constexpr (fill_diff) diff->emplace_back(data_index, data[data_index], new_val);
            data[data_index] = new_val;
        }
    } else {
        assert(array_dim < static_cast<ssize_t>(indices_.size()));
        if (std::holds_alternative<Slice>(indices_[array_dim])) {
            assert(static_cast<ssize_t>(output_dim) < ndim_);
            // Iterate over this dimension backward if doing removals, otherwise forward
            ssize_t axis_size = array_shape[array_dim];
            assert(axis_size >= 0);
            if (!removal) {
                for (ssize_t i = 0; i < axis_size; ++i) {
                    fill_subspace_recurse<fill_diff, placement, removal>(
                            array_shape, array_buffer,
                            array_offset + i * array_ptr_->strides()[array_dim], array_dim + 1,
                            data, data_offset + i * strides()[output_dim], output_dim + 1, diff);
                }
            } else {
                for (ssize_t i = array_shape[array_dim] - 1; i >= 0; --i) {
                    fill_subspace_recurse<fill_diff, placement, removal>(
                            array_shape, array_buffer,
                            array_offset + i * array_ptr_->strides()[array_dim], array_dim + 1,
                            data, data_offset + i * strides()[output_dim], output_dim + 1, diff);
                }
            }
        } else {
            // NOTE: I think output_dim should be incremented by the ndim of the indexing arrays
            if (!bullet1mode_ && array_dim == first_array_index_) output_dim++;
            fill_subspace_recurse<fill_diff, placement, removal>(array_shape, array_buffer,
                                                                 array_offset, array_dim + 1, data,
                                                                 data_offset, output_dim, diff);
        }
    }
}

void AdvancedIndexingNode::initialize_state(State& state) const {
    assert(array_ptr_->ndim() == static_cast<ssize_t>(indices_.size()));
    assert(array_ptr_->ndim() >= 1);

    auto array_strides = array_ptr_->strides();

    const ssize_t size = std::get<ArrayNode*>(indices_[first_array_index_])->size(state);

    std::vector<ssize_t> offsets(size);
    if (size) {
        for (ssize_t index = 0; index < static_cast<ssize_t>(indices_.size()); ++index) {
            if (std::holds_alternative<ArrayNode*>(indices_[index])) {
                assert(array_strides[index] % static_cast<ssize_t>(itemsize()) == 0);

                assert(index < static_cast<ssize_t>(array_strides.size()));
                ssize_t stride = array_strides[index] / static_cast<ssize_t>(itemsize());

                auto it = offsets.begin();
                for (auto& index_val : std::get<ArrayNode*>(indices_[index])->view(state)) {
                    *it += index_val * stride;
                    ++it;
                }
            }
        }
    }

    ssize_t array_axis0_size = array_ptr_->size(state) / array_item_strides()[0];

    // Now get the values
    ssize_t data_size = bullet1mode_ ? size * (subspace_stride_ / itemsize())
                                     : array_axis0_size * (strides()[0] / itemsize());
    std::vector<double> data(data_size);

    for (size_t idx = 0; idx < offsets.size(); ++idx) {
        fill_subspace(state, offsets[idx], data, idx);
    }

    bool main_array_is_constant_node = dynamic_cast<const ConstantNode*>(array_ptr_) != nullptr;

    emplace_data_ptr<AdvancedIndexingNodeData>(state, std::move(offsets), std::move(data),
                                               !main_array_is_constant_node);
}

void AdvancedIndexingNode::commit(State& state) const {
    data_ptr<AdvancedIndexingNodeData>(state)->commit();
}

void AdvancedIndexingNode::revert(State& state) const {
    data_ptr<AdvancedIndexingNodeData>(state)->revert();
}

std::pair<double, double> AdvancedIndexingNode::minmax(
        optional_cache_type<std::pair<double, double>> cache) const {
    return memoize(cache, [&]() { return array_ptr_->minmax(cache); });
}

ssize_t AdvancedIndexingNode::size(const State& state) const {
    return dynamic() ? data_ptr<AdvancedIndexingNodeData>(state)->data.size() : this->size();
}

SizeInfo AdvancedIndexingNode::sizeinfo() const {
    if (!dynamic()) return SizeInfo(size());
    // when we get around to supporting broadcasting this will need to change
    assert(predecessors().size() >= 2);
    assert(!dynamic_cast<ArrayNode*>(predecessors()[0])->dynamic() &&
           "sizeinfo for dynamic base arrays not supported");
    return SizeInfo(dynamic_cast<ArrayNode*>(predecessors()[1]));
}

std::span<const ssize_t> AdvancedIndexingNode::shape(const State& state) const {
    if (!dynamic()) return shape();
    assert(this->ndim() >= 1);
    dynamic_shape_[0] = this->size(state) / (strides()[0] / itemsize());
    return std::span<const ssize_t>(dynamic_shape_.get(), this->ndim());
}

std::pair<ssize_t, ssize_t> get_mapped_index(
        const std::vector<AdvancedIndexingNode::array_or_slice>& indices,
        const std::span<const ssize_t>& strides, const std::span<const ssize_t>& item_strides,
        const std::span<const ssize_t>& shape, const std::span<const ssize_t>& mapped_item_strides,
        ssize_t itemsize, ssize_t index) {
    ssize_t offset = 0;
    ssize_t subspace_index = 0;
    ssize_t mapped_axis = 0;
    bool hit_first_array = false;
    assert(indices.size() == strides.size());
    assert(indices.size() == item_strides.size());
    assert(indices.size() == shape.size());
    assert(item_strides.empty() || item_strides.data() != nullptr);
    for (ssize_t i = 0; i < static_cast<ssize_t>(indices.size()); ++i) {
        ssize_t item_stride = item_strides[i];
        ssize_t axis_index = (index / item_stride) % shape[i];
        if (std::holds_alternative<ArrayNode*>(indices[i])) {
            offset += axis_index * strides[i] / itemsize;
            // Only increase the mapped axis on the first array indexer. We just skip
            // over all the axes due to the indexer arrays by increasing mapped_axis
            // by their dimension.
            mapped_axis += hit_first_array ? 0 : std::get<ArrayNode*>(indices[i])->ndim();
            hit_first_array = true;
        } else {
            assert(mapped_axis < static_cast<ssize_t>(mapped_item_strides.size()));
            subspace_index += axis_index * mapped_item_strides[mapped_axis++] / itemsize;
        }
    }
    return std::make_pair(offset, subspace_index);
}

void AdvancedIndexingNode::propagate(State& state) const {
    // Pull the data into this namespce
    auto node_data = data_ptr<AdvancedIndexingNodeData>(state);
    auto& data = node_data->data;
    auto& diff = node_data->diff;
    auto& offsets_diff = node_data->offsets_diff;

    assert(diff.empty() && "diff not empty, probably calling propagate twice");

    // All indexers are handled in essentially the same way. We look at the final size
    // of the indexer arrays (`new_size`), and ignore all updates with an index larger
    // than this size (possible because arrays may grow past the size and then shrink
    // again). We use the size of `offsets` to keep track of the output size during
    // this loop. Then we loop over the indexer arrays, and then the updates in their diffs.
    // If in an update index is equal to this size, it becomes a new placement
    // for our new diff (these should always be coming from placements of the first indexer,
    // but crucially, not all placements of the first indexer turn into placements for
    // our node). Finally, all other updates are handled as normal updates to our output
    // even if they are placements or removals from an indexer array. If the update from
    // the indexer is a placement is a placement/removal, we replace the new/old value
    // (NaN) with 0, such that the delta calculation of the offset is still correct.
    //
    // If the final size of the output array has shrunk, after handling the updates from
    // the first indexer, we add all the removal updates in a row. No other removals should
    // be necessary.

    const auto array_strides = array_ptr_->strides();

    assert(predecessors().size() >= 2 && "number of predecessors should be at least 2");

    // New size of the indexing arrays
    ssize_t new_indexers_size = dynamic_cast<Array*>(predecessors()[1])->size(state);

    // Handle all the updates to the indexing arrays, modifying the `offsets`
    for (ssize_t dim = 0; dim < array_ptr_->ndim(); ++dim) {
        if (!std::holds_alternative<ArrayNode*>(indices_[dim])) continue;
        Array* indexer = std::get<ArrayNode*>(indices_[dim]);

        ssize_t stride = array_strides[dim] / itemsize();

        assert(new_indexers_size == indexer->size(state) &&
               "indexer has a different size than the first");

        for (const Update& update : indexer->diff(state)) {
            // Ignore all updates with index larger than the final size
            if (update.index >= new_indexers_size) {
                continue;
            }
            if (update.index == static_cast<ssize_t>(node_data->offsets_size())) {
                // This means a new placement for us, and should always come from a placement
                // on the first indexer
                assert(update.placed() && "Trying to grow at index larger than current output");
                assert(dim == first_array_index_ &&
                       "New placement should only come from first indexer");

                ssize_t new_offset = stride * update.value;
                node_data->place_offset(new_offset);
            } else {
                // All other updates (including placements and removals) get handled the same
                // way
                assert(update.index < static_cast<ssize_t>(node_data->offsets_size()) &&
                       "Update with index more than one larger than current size");

                ssize_t old_indexer_val = update.old;
                ssize_t new_indexer_val = update.value;
                // Replace NaNs with 0, as they represent no change to the offset
                if (update.placed()) {
                    old_indexer_val = 0;
                } else if (update.removed()) {
                    new_indexer_val = 0;
                }

                ssize_t offset_delta = stride * (new_indexer_val - old_indexer_val);
                node_data->add_to_offset(update.index, offset_delta);
            }
        }

        if (dim == first_array_index_) {
            // If we shrank, we now need to add diff for that
            for (ssize_t index = static_cast<ssize_t>(node_data->offsets_size()) - 1;
                 index >= new_indexers_size; --index) {
                node_data->remove_offset();
            }
        }
    }

    // Now that `offsets` has been updated, we update the data and diff for each subspace
    // that has changed

    // Collapse offset updates so there is one per index
    deduplicate_diff(node_data->offsets_diff);

    if (bullet1mode_) {
        bool parent_array_changed = array_ptr_->diff(state).size() > 0;
        std::unordered_set<ssize_t> offset_idxs_updated;

        // Handle the normal updates and placements where the offsets differ
        for (const auto& offset_update : offsets_diff) {
            if (offset_update.placed()) {
                fill_subspace<true, false>(state, offset_update.value, data, offset_update.index,
                                           diff);
            } else if (offset_update.removed()) {
                break;
            } else {
                fill_subspace<false, false>(state, offset_update.value, data, offset_update.index,
                                            diff);
            }
            if (parent_array_changed) offset_idxs_updated.insert(offset_update.index);
        }

        // Handle the removals
        for (const auto& offset_update : offsets_diff | std::views::reverse) {
            if (!offset_update.removed()) break;
            fill_subspace<false, true>(state, offset_update.value, data, offset_update.index, diff);
        }

        // Handle any updates to the parent array that weren't already pulled in by
        // filling changed subspaces
        ssize_t parent_size = array_ptr_->size(state);
        for (const auto& update : array_ptr_->diff(state)) {
            // Skip any update larger than the current size as it should have been
            // handled by the removals above
            if (update.index >= parent_size) continue;
            // First part is the offset, second part is the subspace index
            std::pair<ssize_t, ssize_t> mapped_index =
                    get_mapped_index(indices_, array_strides, array_item_strides(),
                                     array_ptr_->shape(state), strides(), itemsize(), update.index);
            // This will return a nullptr if there currently are no mapped offsets
            const auto* idxs = node_data->get_offset_idxs(mapped_index.first);
            if (idxs != nullptr) {
                for (const ssize_t& idx : *idxs) {
                    if (offset_idxs_updated.contains(idx)) continue;
                    const ssize_t item_stride = subspace_stride_ / itemsize();
                    const ssize_t new_index = idx * item_stride + mapped_index.second;
                    assert(data[new_index] == update.old);
                    diff.emplace_back(new_index, update.old, update.value);
                    data[new_index] = update.value;
                }
            }
        }
    } else {
        ssize_t new_parent_size = array_ptr_->size(state);
        ssize_t old_parent_size = new_parent_size - array_ptr_->size_diff(state);
        ssize_t unchanged_parent_size = std::min(new_parent_size, old_parent_size);

        ssize_t old_axis0_size = old_parent_size / array_item_strides()[0];
        ssize_t new_axis0_size = new_parent_size / array_item_strides()[0];
        ssize_t unchanged_axis0_size = std::min(new_axis0_size, old_axis0_size);
        // First handle any changes to the size of the parent array
        if (new_axis0_size > old_axis0_size) {
            // Growing, so need placements
            for (ssize_t axis0_idx = old_axis0_size; axis0_idx < new_axis0_size; ++axis0_idx) {
                for (ssize_t offset_idx = 0; offset_idx < node_data->offsets_size(); ++offset_idx) {
                    fill_axis0_subspace<true, false>(state, axis0_idx,
                                                     node_data->offsets()[offset_idx], data,
                                                     offset_idx, diff);
                }
            }
        }

        if (new_axis0_size < old_axis0_size) {
            // Shrinking, so need removals
            for (ssize_t axis0_idx = old_axis0_size - 1; axis0_idx >= new_axis0_size; --axis0_idx) {
                for (ssize_t offset_idx = node_data->offsets_size() - 1; offset_idx >= 0;
                     --offset_idx) {
                    fill_axis0_subspace<false, true>(state, axis0_idx,
                                                     node_data->offsets()[offset_idx], data,
                                                     offset_idx, diff);
                }
            }
        }

        // Now handle the changes to offsets for axis0 indices that are have not just been placed
        std::unordered_set<ssize_t> offset_idxs_updated;
        for (const auto& offset_update : offsets_diff) {
            for (ssize_t axis0_idx = 0; axis0_idx < unchanged_axis0_size; ++axis0_idx) {
                assert(!offset_update.placed());
                assert(!offset_update.removed());
                fill_axis0_subspace<false, false>(state, axis0_idx, offset_update.value, data,
                                                  offset_update.index, diff);
            }
            offset_idxs_updated.insert(offset_update.index);
        }

        // Finally, handle any updates to the parent array that have not been covered above
        for (const auto& update : array_ptr_->diff(state)) {
            // Skip any update larger than the unchanged size
            if (update.index >= unchanged_parent_size) continue;

            // First part is the offset, second part is the subspace index
            std::pair<ssize_t, ssize_t> mapped_index =
                    get_mapped_index(indices_, array_strides, array_item_strides(),
                                     array_ptr_->shape(state), strides(), itemsize(), update.index);
            const auto* idxs = node_data->get_offset_idxs(mapped_index.first);
            if (idxs != nullptr) {
                for (const auto& idx : *idxs) {
                    if (offset_idxs_updated.contains(idx)) continue;
                    ssize_t item_stride = subspace_stride_ / itemsize();
                    ssize_t new_index = idx * item_stride + mapped_index.second;
                    assert(data[new_index] == update.old ||
                           (std::isnan(data[new_index]) && update.placed()));
                    diff.emplace_back(new_index, update.old_or(0), update.value_or(0));
                    data[new_index] = update.value;
                }
            }
        }
    }

    // Only signal successors if we actually have something to propagate
    if (diff.size()) Node::propagate(state);
}

ssize_t AdvancedIndexingNode::size_diff(const State& state) const {
    if (this->dynamic()) {
        auto ptr = data_ptr<AdvancedIndexingNodeData>(state);

        return static_cast<ssize_t>(ptr->data.size()) - ptr->old_data_size;
    }
    return 0;
}

// BasicIndexingNode **********************************************************

// Helper class for determining the shape/strides/start of the view induced by
// the given indices
struct BasicIndexingNode::IndexParser_ {
    IndexParser_(Array* array_ptr, std::vector<slice_or_int>&& indices) {
        // some input validation
        assert(array_ptr->ndim() >= 0);  // should always be true
        if (static_cast<std::size_t>(array_ptr->ndim()) > indices.size()) {
            throw std::invalid_argument("too few indices for the array");
        }
        if (static_cast<std::size_t>(array_ptr->ndim()) < indices.size()) {
            throw std::invalid_argument("too many indices for the array");
        }

        if (!array_ptr->contiguous()) {
            // we should handle this at the Python level
            throw std::invalid_argument("cannot slice a non-contiguous array");
        }

        // first count the number of dimensions and use that to allocate the
        // shape and strides
        for (const slice_or_int& index : indices) {
            this->ndim += std::holds_alternative<Slice>(index);
        }
        if (this->ndim) {
            this->strides = std::make_unique<ssize_t[]>(ndim);
            this->shape = std::make_unique<ssize_t[]>(ndim);
        }

        // now do the parsing
        const auto array_strides = array_ptr->strides();
        const auto array_shape = array_ptr->shape();

        ssize_t array_dim = 0;  // current dim in the array
        ssize_t view_dim = 0;   // current dim in the view
        for (const slice_or_int& index : indices) {
            assert(array_dim >= view_dim);  // view.ndim <= array.ndim

            if (std::holds_alternative<Slice>(index)) {
                const Slice& slice = std::get<Slice>(index);

                if (slice.step == 0) {
                    throw std::invalid_argument("step cannot be 0");
                }
                if (slice.step < 0) {
                    // todo: only disallow this on dynamic arrays
                    throw std::invalid_argument("step cannot be negative");
                }

                if (array_dim == 0 && array_ptr->dynamic()) {
                    // Handle the dynamic case

                    // First dimension is dynamic
                    this->shape[view_dim] = DYNAMIC_SIZE;
                    this->strides[view_dim] = slice.step * array_strides[array_dim];

                    this->axis0_slice = slice;

                    // Skip start for dynamic axis 0 as it will be adjusted later with
                    // the predecessor's state
                } else {
                    assert(array_shape[array_dim] >= 0);

                    // fit the slice to the array dimension. This handles
                    // negative indices and the like
                    Slice fitted = slice.fit(array_shape[array_dim]);

                    // Fit the slice to the actual dimension size to get the new size
                    this->shape[view_dim] = fitted.size();

                    // For the strides, we only care about the slice's step times the
                    // array's strides
                    this->strides[view_dim] = fitted.step * array_strides[array_dim];

                    // For the start, it's defined by the (fitted) slice's start
                    // Note that the start is in terms of doubles, but strides are in terms
                    // of bytes, so we have to be careful
                    // We
                    this->start += fitted.start * array_strides[array_dim] / array_ptr->itemsize();
                }

                ++view_dim;
            } else {
                ssize_t idx = std::get<ssize_t>(index);

                if (array_dim == 0 && array_ptr->dynamic()) {
                    throw std::invalid_argument(
                            "integer index not allowed on first dimension of dynamic array");
                } else if (idx < 0) {
                    // Negative indexing on a dimension with a known size
                    // We do one pass of fitting, and then raise an error if it's still
                    // negative or out of range
                    idx += array_shape[array_dim];
                }

                if (idx < 0 || idx >= array_shape[array_dim]) {
                    // todo: better error message? Needs testing from Python
                    throw std::invalid_argument("index out of range");
                }

                // Integer indices only affect the start, they drop the dimension
                // so there is no strides/shape to worry about for this axis
                this->start += idx * array_strides[array_dim] / array_ptr->itemsize();
            }

            ++array_dim;  // we are always stepping through the array
        }
    }

    ssize_t ndim = 0;
    std::unique_ptr<ssize_t[]> strides = nullptr;
    std::unique_ptr<ssize_t[]> shape = nullptr;

    ssize_t start = 0;

    std::optional<Slice> axis0_slice;
};

BasicIndexingNode::BasicIndexingNode(ArrayNode* array_ptr, std::vector<slice_or_int> indices)
        : BasicIndexingNode(array_ptr, IndexParser_(array_ptr, std::move(indices))) {}

BasicIndexingNode::BasicIndexingNode(ArrayNode* array_ptr, IndexParser_&& parser)
        : array_ptr_(array_ptr),
          ndim_(parser.ndim),
          strides_(std::move(parser.strides)),
          shape_(std::move(parser.shape)),
          start_(parser.start),
          size_(Array::shape_to_size(ndim_, shape_.get())),
          axis0_slice_(parser.axis0_slice),
          contiguous_(Array::is_contiguous(ndim_, shape_.get(), strides_.get())) {
    add_predecessor(array_ptr);
}

struct BasicIndexingNodeData : NodeStateData {
    BasicIndexingNodeData(const BasicIndexingNode* node) {
        if (node->dynamic()) {
            dynamic_shape = std::make_unique<ssize_t[]>(node->ndim());
            // First dim will be set later
            std::copy(node->shape().begin(), node->shape().end(), dynamic_shape.get());
        }
    };

    std::vector<Update> diff;
    // Used to track diffs for dynamic size
    ssize_t previous_size;

    // The state dependent shape when indexing a dynamic array. Will be nullptr when
    // predecessor array is fixed shape.
    std::unique_ptr<ssize_t[]> dynamic_shape = nullptr;

    Slice fitted_first_slice;

    // This will be used to cache the full output in the case of a negative start index.
    // Keeping a full cache not only keeps the implementation simpler, but because any
    // change of size to the predecessor array means (most definitely) that the entire
    // output of the BasicIndexingNode will change, it is likely close to optimal (unless
    // we have a case where we need to do basic indexing across a large range of a dynamic
    // array that very rarely changes size).
    //
    // By stable sorting the predecessor updates and carefully processing a new diff,
    // we could make have an implementation with a O(n + mlogm) cost on resize where n is
    // the size of the output array and m is the number of predecessor updates. Then we
    // could have a O(m logm) cost for the case when the predecessor does not change size.
    std::vector<double> full_cache_;
};

void BasicIndexingNode::update_dynamic_shape(State& state) const {
    assert(dynamic());
    assert(axis0_slice_.has_value() && "first dim slice needs to be set");
    assert(array_ptr_->ndim() >= 1);

    // parse the slice
    const Slice& slice = axis0_slice_.value();
    const ssize_t& first_dim_size = array_ptr_->shape(state)[0];

    auto node_data = data_ptr<BasicIndexingNodeData>(state);

    node_data->fitted_first_slice = slice.fit(first_dim_size);

    // Fit the slice to the actual dimension size to get the new size
    node_data->dynamic_shape[0] = node_data->fitted_first_slice.size();
}

double const* BasicIndexingNode::buff(const State& state) const {
    if (!dynamic()) {
        assert(!axis0_slice_.has_value());
        return array_ptr_->buff(state) + start_;
    }

    const auto node_data = data_ptr<BasicIndexingNodeData>(state);
    return array_ptr_->buff(state) + start_ + dynamic_start(node_data->fitted_first_slice.start);
}

void BasicIndexingNode::commit(State& state) const {
    auto node_data = data_ptr<BasicIndexingNodeData>(state);
    node_data->diff.clear();
    node_data->previous_size = size(state);
    // reset the cache
    node_data->full_cache_.assign(begin(state), end(state));
}

std::span<const Update> BasicIndexingNode::diff(const State& state) const {
    return data_ptr<BasicIndexingNodeData>(state)->diff;
}

std::vector<BasicIndexingNode::slice_or_int> BasicIndexingNode::infer_indices() const {
    std::vector<slice_or_int> indices;

    ssize_t view_dim = 0;
    const ssize_t view_ndim = this->ndim();

    ssize_t array_dim = 0;
    const ssize_t array_ndim = array_ptr_->ndim();

    if (dynamic()) {
        indices.emplace_back(*axis0_slice_);

        view_dim += 1;
        array_dim += 1;
    }

    auto array_strides = array_ptr_->strides();
    auto view_strides = this->strides();

    auto array_shape = array_ptr_->shape();
    auto view_shape = this->shape();

    ssize_t flat_start = start_;

    // we don't currently have support for views with a greater number of dimensions
    // than their viewed array. Once we do, the logic here will need to be revisted
    assert(view_ndim <= array_ndim && "not implemented yet");
    if (view_ndim > array_ndim) throw std::logic_error("not implemented (yet)");

    // ok, let's traverse the remaining dimensions to find the strides
    for (; array_dim < array_ndim; ++array_dim) {
        // calculate the start of this dimension.
        const ssize_t num_per_row = array_strides[array_dim] / itemsize();
        const ssize_t start = flat_start / num_per_row;
        flat_start %= num_per_row;

        if (view_dim >= view_ndim || view_strides[view_dim] < array_strides[array_dim]) {
            // The strides are smaller in the view, or not present at all, which
            // means we have a constant index.
            assert(view_ndim < array_ndim);
            indices.emplace_back(start);

        } else {
            // Otherwise we must be a slice
            assert(view_dim < view_ndim);  // still within range

            const ssize_t step = view_strides[view_dim] / array_strides[array_dim];
            const ssize_t stop = start + view_shape[view_dim] * step;
            Slice slice = Slice(start, stop, step).fit(array_shape[array_dim]);
            indices.emplace_back(slice);

            ++view_dim;
        }
    }

    return indices;
}

void BasicIndexingNode::initialize_state(State& state) const {
    // we're a view, so we don't really need state other than for the updates
    emplace_data_ptr<BasicIndexingNodeData>(state, this);
    if (this->dynamic()) {
        update_dynamic_shape(state);
        auto node_data = data_ptr<BasicIndexingNodeData>(state);
        node_data->previous_size = size(state);

        if (axis0_slice_.value().start < 0) {
            // Initialize the full output cache
            node_data->full_cache_.assign(begin(state), end(state));
        }
    }
}

ssize_t BasicIndexingNode::dynamic_start(ssize_t slice_start) const {
    return slice_start * strides_[0] / static_cast<ssize_t>(sizeof(double));
}

ssize_t get_smallest_size_during_diff(ssize_t initial_size, const std::span<const Update> diff) {
    ssize_t minimum_size = initial_size;
    ssize_t size = initial_size;
    for (const auto& update : diff) {
        if (update.placed()) {
            size++;
        } else if (update.removed()) {
            size--;
            assert(size >= 0);
            minimum_size = std::min(size, minimum_size);
        }
    }

    return minimum_size;
}

std::pair<double, double> BasicIndexingNode::minmax(
        optional_cache_type<std::pair<double, double>> cache) const {
    return memoize(cache, [&]() { return array_ptr_->minmax(cache); });
}

void BasicIndexingNode::propagate(State& state) const {
    auto node_data = data_ptr<BasicIndexingNodeData>(state);
    auto& diff = node_data->diff;
    assert(diff.size() == 0 && "calling propagate on an node with pending updates");

    if (contiguous_ && dynamic() && axis0_slice_.value().start < 0) {
        assert(this->start_ == 0);
        // Do a full recompute of the output using the current data. This could be avoided
        // when the size of the parent array hasn't changed.
        auto& full_cache = node_data->full_cache_;
        assert(full_cache.size() ==
                       axis0_slice_.value()
                               .fit(array_ptr_->size(state) - array_ptr_->size_diff(state))
                               .size() &&
               "cache is the wrong size");

        update_dynamic_shape(state);

        ssize_t i = 0;  // Index on the cache
        auto it = begin(state);
        for (; i < static_cast<ssize_t>(full_cache.size()) && it != end(state); ++i, ++it) {
            if (full_cache[i] != *it) {
                diff.emplace_back(i, full_cache[i], *it);
            }
        }

        for (; it != end(state); ++it, ++i) {
            // Output array has grown
            diff.emplace_back(Update::placement(static_cast<double>(i), *it));
        }

        for (ssize_t j = static_cast<ssize_t>(full_cache.size()) - 1; j >= i; --j) {
            // Output array has shrunk
            diff.emplace_back(Update::removal(static_cast<double>(j), full_cache[j]));
        }

        assert(size(state) == i);

    } else if (contiguous_ && dynamic()) {
        // Handle the case of dynamic array with non-negative start.
        //
        // The parent array's diff may contain a series of placements and removals
        // (interspersed with with mutations of the value at a given index) that make
        // it difficult to track the correct old/new values for the diff that the
        // basic indexing node must produce. For example, with a negative end, a
        // placement on the parent array may mean that the basic indexing node needs
        // to translate that placement into placement at an offset index, so the old/new
        // values from the parent's `Update` cannot be reused. A previous update may
        // have modified the value at that offset index, so we cannot necessarily use the
        // current value of the parent at that index for the old value of the `Update`.
        //
        // To address this and other subtle issues with translating the parent's diff
        // in a one-to-one way, we instead separate out two ranges of indices where we
        // handle the parent updates in a special way, and then handle the rest in the
        // straightforward one-to-one fashion. The two special ranges are:
        //
        // 1. The "delta" range: the range of indices where we know we will have net
        // growth or shrinking of the basic indexing node. For example, if the parent
        // array A (net) grows from size 5 to 10, and our basic indexing is computing
        // A[3:7], then we will have a net growth of size 2 to 4, and the delta range
        // will consist of the (parent) indices 5:7. We can then handle parent updates
        // in this range by using the old value of the first update we see for the new
        // old value, and otherwise just replacing the "new" value. Note that if we are
        // net shrinking, we will have to construct the corresponding `Update`s in
        // reverse order.
        //
        // 2. The "false removal" range: the range of (parent) indices where the parent
        // shrinks down to, but eventually grows past again. This range will never
        // overlap with the "delta" range. We have to treat this range in a special way
        // as well because placements/removals in this range would be out of order if we
        // parsed them one-to-one and simply appended them to the diff (after the updates
        // for the "delta" range). Updates in this range are handled with similar logic
        // to the "delta" range, but they should all be normal updates (no
        // placements/removals) and we should never need to look up an old value from
        // from the current parent's array.
        //
        // Handling the updates in these ranges is done by pre-allocating an array of
        // `Update`s for the diff of the size of the "delta" range plus the size of the
        // "false removal" range. All other updates within the overall range of the
        // basic indexing node simply get appended (always as normal updates) to the
        // diff array.
        //
        // If we are net growing, we simply fill in the `Update`s in the delta range
        // as placements using the parent's current data directly for the `new` values.
        //
        // We then loop through the parent's diff. If the parent `Update` is outside the
        // delta and the false removal range but within the rest of our new slice, we
        // parse it normally and append it to the diff vector.
        //
        // If it is inside the delta, and we are net shrinking, we set the old value
        // if is the first value we've seen at that index, and otherwise ignore the
        // update.
        //
        // If it is inside the false removal range, we use old value for the old value
        // of our corresponding update if it is the first update seen at that index.
        // Then always overwrite the new value with the new value of the parent update.
        //
        // All other updates are ignored.
        //
        // Finally, if net shrinking, we loop through the `Update`s for the delta one
        // more time, and if any have unset `old` values, we know we can safely look it
        // up in the parent's current data. This should only happen if we have a negative
        // stop on our main slice.

        assert(this->start_ == 0);

        ssize_t old_parent_size = array_ptr_->size(state) - array_ptr_->size_diff(state);
        Slice previous_slice = axis0_slice_.value().fit(old_parent_size);
        if (previous_slice.size() == 0) {
            previous_slice = Slice(axis0_slice_.value().start, axis0_slice_.value().start);
        }
        Slice new_slice = axis0_slice_.value().fit(array_ptr_->size(state));
        if (new_slice.size() == 0) {
            new_slice = Slice(axis0_slice_.value().start, axis0_slice_.value().start);
        }

        ssize_t smallest_parent_size =
                get_smallest_size_during_diff(old_parent_size, array_ptr_->diff(state));

        bool growing = new_slice.size() > previous_slice.size();
        bool shrinking = new_slice.size() < previous_slice.size();

        // Define the range of the "delta", where values need to be placed/removed,
        // indexed with reference to the parent array
        Slice delta = Slice(new_slice.stop, new_slice.stop);
        if (growing) {
            delta = Slice(previous_slice.stop, new_slice.stop);
        } else if (shrinking) {
            delta = Slice(new_slice.stop, previous_slice.stop);
        }

        ssize_t false_removals_start = std::max(smallest_parent_size, new_slice.start);
        ssize_t false_removals_size = std::max(ssize_t(0), delta.start - false_removals_start);
        assert(false_removals_size >= 0);

        // The first "delta size" updates will be used to track for placement/removal indices.
        // An extra `false_removals_size` updates are added after that which will handle the
        // "false" removals
        diff.reserve(delta.size() + false_removals_size);
        if (growing) {
            for (ssize_t i = delta.start; i < delta.stop; ++i) {
                diff.emplace_back(i - new_slice.start, Update::nothing, Update::nothing);
            }
        } else if (shrinking) {
            for (ssize_t i = delta.stop - 1; i >= delta.start; --i) {
                diff.emplace_back(i - new_slice.start, Update::nothing, Update::nothing);
            }
        }

        for (ssize_t i = 0; i < false_removals_size; ++i) {
            diff.emplace_back(false_removals_start + i - new_slice.start, Update::nothing,
                              Update::nothing);
        }

        if (growing) {
            // Simply retrieve the new values from the new data of the parent
            for (std::size_t i = 0, stop = delta.size(); i < stop; ++i) {
                diff[i].value = array_ptr_->view(state)[delta.start + i];
            }
        }

        for (const auto& update : array_ptr_->diff(state)) {
            bool within_delta = update.index >= delta.start && update.index < delta.stop;

            if (shrinking && within_delta) {
                // We're overall shrinking, and the update from the parent has an index
                // within the delta range
                auto delta_index = delta.stop - update.index - 1;
                // NOTE: don't want to use `placed()` here because it's unclear what
                // that should return in the case of `Update(x, nan, nan)`
                if (std::isnan(diff[delta_index].old)) {
                    // We haven't seen this index yet, so the `old` of this update should
                    // be the `old` of our translated index
                    assert(diff[delta_index].index == update.index - new_slice.start);
                    diff[delta_index].old = update.old;
                }
            } else if (update.index < delta.start && update.index >= false_removals_start) {
                // Within the "false removals" range
                auto index = delta.size() + update.index - false_removals_start;
                assert(index >= delta.size());
                assert(index < delta.size() + false_removals_size);
                assert(diff[index].index == update.index - new_slice.start);

                // NOTE: don't want to use `removed()` here because it's unclear what
                // that should return in the case of `Update(x, nan, nan)`
                if (std::isnan(diff[index].old)) {
                    assert(!std::isnan(update.old));
                    diff[index].old = update.old;
                }
                diff[index].value = update.value;
            } else if (update.index < delta.start && update.index >= new_slice.start) {
                diff.emplace_back(update.index - new_slice.start, update.old, update.value);
            }
        }

        if (shrinking) {
            // Fill in the missing old values for the removals in the delta range,
            // directly using the parent array's data.
            for (std::size_t i = 0, stop = delta.size(); i < stop; ++i) {
                // NOTE: don't want to use `placed()` here because it's unclear what
                // that should return in the case of `Update(x, nan, nan)`
                if (std::isnan(diff[i].old)) {
                    diff[i].old = array_ptr_->view(state)[delta.stop - i - 1];
                }
            }
        }

        update_dynamic_shape(state);
    } else if (contiguous_) {
        // This is simple, it's just a numeric offset

        const ssize_t start = this->start_;
        const ssize_t stop = start + this->size(state);

        // A few sanity checks...
        assert(start >= 0);
        assert(stop >= start);
        assert(stop <= array_ptr_->size(state));

        for (const auto& [index, old, value] : array_ptr_->diff(state)) {
            if (index < start) continue;  // before the range we care about
            if (index >= stop) continue;  // after the range we care about

            diff.emplace_back(index - start, old, value);
        }

    } else if (!dynamic() || (axis0_slice_.value().start >= 0 && axis0_slice_.value().stop >= 0)) {
        // Handle all other cases of static sized arrays (contiguous and non-contiguous), and the
        // case of dynamic but with positive start and end
        assert(!this->contiguous());
        assert(array_ptr_->contiguous() && "doesn't support non-contiguous arrays");

        if (dynamic()) update_dynamic_shape(state);

        const ssize_t start = this->start_;
        // distance between pointers in the parent array. Relies on parent
        // being contiguous
        assert(array_ptr_->contiguous());
        const ssize_t stop = &*this->end(state) - &*array_ptr_->begin(state);

        // A few sanity checks...
        assert(start >= 0);
        assert(stop >= start);
        assert(this->ndim() > 0);  // if we're not contiguous we cannot be a scalar
        // some strided stop after the end to make the math work out.
        assert(stop <= array_ptr_->size(state) + this->strides()[0] / this->itemsize());

        for (auto [index, old, value] : array_ptr_->diff(state)) {
            if (index < start) continue;  // before the range we care about
            if (index >= stop) continue;  // after the range we care about

            index -= start;

            bool skip = false;
            for (ssize_t stride : this->strides()) {
                assert(stride % this->itemsize() == 0);
                // again, we rely on the parent being contiguous!
                assert(array_ptr_->contiguous());
                const ssize_t div = stride / this->itemsize();

                if (index % div) {
                    // we are "between" steps
                    skip = true;
                    break;
                }

                index /= div;
            }
            if (skip) continue;

            assert(index >= 0 && index < this->size(state));

            diff.emplace_back(index, old, value);
        }
    } else {
        assert(false && "not yet implemented");
    }

    // Only signal successors if we actually have something to propagate
    if (diff.size()) Node::propagate(state);
}

void BasicIndexingNode::revert(State& state) const {
    auto node_data = data_ptr<BasicIndexingNodeData>(state);
    node_data->diff.clear();
    if (dynamic()) {
        // todo this is only safe if revert() has been called on the predecessor array
        // first, which isn't safe to assume
        update_dynamic_shape(state);
        if (axis0_slice_.value().start < 0) {
            assert(contiguous_ && "not yet implemented");
            node_data->full_cache_.resize(node_data->previous_size);
        }
    }
}

ssize_t BasicIndexingNode::size(const State& state) const {
    if (size_ >= 0) return size_;

    return Array::shape_to_size(ndim_, data_ptr<BasicIndexingNodeData>(state)->dynamic_shape.get());
}

SizeInfo BasicIndexingNode::sizeinfo() const {
    if (size_ >= 0) return SizeInfo(size_);

    auto sizeinfo = SizeInfo(array_ptr_);

    // Get the multiplier on the size of the array
    {
        // Determine how many elements are in each row of the array. Where "row" here means the
        // first axis.
        // We use the fact that the parent is contiguous to simplify this calculation.
        assert(array_ptr_->contiguous());
        assert(this->ndim() > 0);
        ssize_t array_num_per_row = array_ptr_->strides()[0] / array_ptr_->itemsize();

        sizeinfo.multiplier /= array_num_per_row;

        // Determine how many elements are in each row of the resulting array.
        // Here we may be strided, so we use the shape to figure it out.
        auto shape = this->shape();
        ssize_t num_per_row =
                std::reduce(shape.begin() + 1, shape.end(), 1, std::multiplies<ssize_t>());

        sizeinfo.multiplier *= num_per_row;

        // handle the step
        assert(axis0_slice_);            // dynamic so this should be present
        assert(axis0_slice_->step > 0);  // constructor prevents otherwise for now

        sizeinfo.multiplier /= axis0_slice_->step;
    }

    // Get the offset and bounds
    {
        assert(axis0_slice_);  // dynamic so this should be present

        if (axis0_slice_->step > 0) {
            if (axis0_slice_->start > 0) {
                // having a positive start imposes a size offset
                assert((sizeinfo.multiplier * axis0_slice_->start).denominator() == 1);
                sizeinfo.offset -= static_cast<ssize_t>(sizeinfo.multiplier * axis0_slice_->start);

            } else if (axis0_slice_->start < 0) {
                // a negative start imposes a max size
                sizeinfo.max = static_cast<ssize_t>(sizeinfo.multiplier * -1 * axis0_slice_->start);
            }

            // This is where having Slice's values be optional<ssize_t> would be useful.
            // For now let's just hard code it.
            constexpr ssize_t MAX = std::numeric_limits<ssize_t>::max();

            if (axis0_slice_->stop > 0 && axis0_slice_->stop != MAX) {
                // having a positive stop imposes a max size
                sizeinfo.max = static_cast<ssize_t>(sizeinfo.multiplier * axis0_slice_->stop);
            } else if (axis0_slice_->stop < 0) {
                // having a negative stop imposes a size offset
                sizeinfo.offset += static_cast<ssize_t>(sizeinfo.multiplier * axis0_slice_->stop);
            }

        } else {
            // this is currently disallowed by the constructor
            assert(false && "not implemented yet");
            unreachable();
        }
    }

    return sizeinfo;
}

ssize_t BasicIndexingNode::size_diff(const State& state) const {
    if (size_ >= 0) return 0;

    auto ptr = data_ptr<BasicIndexingNodeData>(state);
    return size(state) - ptr->previous_size;
}

std::span<const ssize_t> BasicIndexingNode::shape(const State& state) const {
    if (size_ >= 0) return BasicIndexingNode::shape();

    return std::span<const ssize_t>(data_ptr<BasicIndexingNodeData>(state)->dynamic_shape.get(),
                                    ndim_);
}

// PermutationNode ************************************************************

PermutationNode::PermutationNode(ArrayNode* array_ptr, ArrayNode* order_ptr)
        : ArrayOutputMixin(array_ptr->shape()), array_ptr_(array_ptr), order_ptr_(order_ptr) {
    std::span<const ssize_t> array_shape = array_ptr_->shape();

    // For now, we are only going to support permutation on constant nodes
    if (!dynamic_cast<ConstantNode*>(array_ptr)) {
        throw std::invalid_argument("array must be a ConstantNode");
    }
    if (array_ptr_->ndim() < 1) {
        throw std::invalid_argument("array must not be a scalar");
    }
    if (!std::equal(array_shape.begin() + 1, array_shape.end(), array_shape.begin())) {
        throw std::invalid_argument(
                "array must be square - that is every dimension must have the same size");
    }

    if (order_ptr_->dynamic()) {
        throw std::invalid_argument("order's size must be fixed");
    }
    if (!order_ptr_->integral()) {
        throw std::invalid_argument("order must take integral values");
    }
    if (order_ptr_->ndim() != 1) {
        throw std::invalid_argument("order must be a 1d array");
    }

    if (array_shape[0] != order_ptr_->size()) {
        throw std::invalid_argument("array shape and order size mismatch");
    }
    if (order_ptr_->max() > array_ptr_->size()) {
        throw std::invalid_argument("order may have values out of range");
    }

    this->add_predecessor(array_ptr);
    this->add_predecessor(order_ptr);
}

void PermutationNode::commit(State& state) const { data_ptr<IndexingNodeData>(state)->commit(); }

double const* PermutationNode::buff(const State& state) const {
    return data_ptr<IndexingNodeData>(state)->data.data();
}

std::span<const Update> PermutationNode::diff(const State& state) const {
    return data_ptr<IndexingNodeData>(state)->diff;
}

void PermutationNode::initialize_state(State& state) const {
    const std::span<const ssize_t> strides = array_ptr_->strides();

    const ssize_t n = this->shape()[0];
    const ssize_t ndim = this->ndim();

    // first work out all of the offsets in the array
    std::vector<ssize_t> offsets;
    {
        assert(this->size() >= 0);
        offsets.reserve(this->size());

        // this may be strided, so for speed we just dump it into a vector once
        std::vector<ssize_t> order(order_ptr_->begin(state), order_ptr_->end(state));

        std::vector<ssize_t> i(ndim);
        std::vector<ssize_t> o(ndim, order[0]);
        do {
            offsets.emplace_back(std::inner_product(o.begin(), o.end(), strides.begin(), 0) /
                                 sizeof(double));

            for (ssize_t dim = ndim - 1; dim >= 0; --dim) {
                if (++i[dim] != n) {
                    o[dim] = order[i[dim]];
                    break;
                }
                i[dim] = 0;
                o[dim] = order[0];
            }
        } while (std::any_of(i.begin(), i.end(), std::identity()));
    }

    // now get the values
    std::vector<double> values;
    {
        values.reserve(this->size());

        const double* start = array_ptr_->buff(state);
        for (const ssize_t offset : offsets) {
            values.emplace_back(start[offset]);
        }
    }

    emplace_data_ptr<IndexingNodeData>(state, std::move(offsets), std::move(values));
}

void PermutationNode::propagate(State& state) const {
    auto ptr = data_ptr<IndexingNodeData>(state);

    auto& offsets = ptr->offsets;
    auto& values = ptr->data;
    auto& updates = ptr->diff;
    auto& old_offsets = ptr->old_offsets;

    assert(updates.size() == 0 && "called propagate on a node with pending updates");

    // incorporate changes to the order
    auto order_diff = order_ptr_->diff(state);
    if (order_diff.size()) {
        const ssize_t ndim = this->ndim();
        const ssize_t n = this->shape()[0];

        const std::span<const ssize_t> strides = array_ptr_->strides();
        const double* start = array_ptr_->buff(state);

        ssize_t mul = std::pow(n, ndim - 1);
        ssize_t step = 1;

        for (ssize_t dim = 0; dim < ndim; ++dim) {
            for (auto& [index, old, neo] : order_diff) {
                ssize_t stride = strides[dim] / array_ptr_->itemsize();

                for (ssize_t i = index * mul, end = i + step * n; i < end; i += step) {
                    old_offsets.emplace_back(offsets[i]);

                    offsets[i] += stride * (neo - old);

                    // We do end up updating some of the indices more than once. We could
                    // instead put placeholder values, then later do a deduplication step
                    // and then value update. If we find that we're doing lots of redundant
                    // work
                    double old_value = values[i];
                    values[i] = start[offsets[i]];

                    updates.emplace_back(i, old_value, values[i]);
                }
            }

            mul /= n;
            step *= n;
        }
    }

    // incorporate changes to the array
    assert(array_ptr_->diff(state).size() == 0 && "not implemented yet");  // todo

    // Only signal successors if we actually have something to propagate
    if (updates.size()) Node::propagate(state);
}

void PermutationNode::revert(State& state) const { data_ptr<IndexingNodeData>(state)->revert(); }

}  // namespace dwave::optimization

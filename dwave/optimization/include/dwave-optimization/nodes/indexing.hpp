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

#pragma once

#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/graph.hpp"

namespace dwave::optimization {

// Following NumPy's naming convention, there are three types of indexing.
//  * Basic https://numpy.org/doc/stable/user/basics.indexing.html#basic-indexing
//      In basic indexing, the indexes are all slices and/or ints. The resulting array will have
//      a shape defined by those slices.
//  * Advanced indexing https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing
//      In advanced indexing, the indices are all arrays of the same shape. The resulting array
//      will itself have the same shape as the index arrays
//  * Combined
//      https://numpy.org/doc/stable/user/basics.indexing.html#combining-advanced-and-basic-indexing
//      A combination of int/slice/array indices. The rules are... more complicated.

class AdvancedIndexingNode : public Node, public Array {
 public:
    // Indices are some combination of arrays and slices
    using array_or_slice = std::variant<Array*, Slice>;

    // Runtime constructor that can be used from Cython/Python
    AdvancedIndexingNode(Node* node_ptr, std::vector<array_or_slice> indices);

    // Templated constructor to be used from C++
    // Just calls the runtime constructor for simplicity, but we could implement
    // a faster version by using more compile-time information
    template <ArrayNode T, class... Indices>
    explicit AdvancedIndexingNode(T* array_ptr, Indices... indices)
            : AdvancedIndexingNode(static_cast<Node*>(array_ptr), make_indices(indices...)) {}

    // Array overloads
    ssize_t ndim() const noexcept override { return ndim_; }
    double const* buff(const State& state) const override;
    std::span<const Update> diff(const State& state) const override;
    ssize_t size() const override { return size_; }
    ssize_t size(const State& state) const override;
    std::span<const ssize_t> shape() const override { return std::span(shape_.get(), ndim_); }
    std::span<const ssize_t> shape(const State& state) const override;
    ssize_t size_diff(const State& state) const override;
    SizeInfo sizeinfo() const override;
    std::span<const ssize_t> strides() const override { return std::span(strides_.get(), ndim_); }
    bool contiguous() const override { return true; }

    // Node overloads
    void commit(State& state) const override;
    void initialize_state(State& state) const override;
    void propagate(State& state) const override;
    void revert(State& state) const override;

    // AdvancedIndexingHelper methods
    std::span<const array_or_slice> indices() const { return indices_; }

 private:
    struct IndexParser_;
    AdvancedIndexingNode(Array* array_ptr, IndexParser_&& parser);

    // Convert the indices from a parameter pack to a vector of arrays.
    // This allows us to use the runtime constructor from the templated constructor.
    template <class... Indices>
    static std::vector<array_or_slice> make_indices(Indices&... indices) {
        std::vector<array_or_slice> output;
        (output.emplace_back(indices), ...);
        return output;
    }

    void fill_subspace(State& state, ssize_t array_offset, std::vector<double>& data,
                       ssize_t index_in_arrays) const;

    template <bool placement, bool removal>
    void fill_subspace(State& state, ssize_t array_offset, std::vector<double>& data,
                       ssize_t index_in_arrays, std::vector<Update>& diff) const;

    template <bool placement, bool removal>
    void fill_axis0_subspace(State& state, ssize_t axis0_index, ssize_t array_item_offset,
                             std::vector<double>& data, ssize_t index_in_arrays,
                             std::vector<Update>& diff) const;

    template <bool fill_diff, bool placement, bool removal>
    void fill_subspace_recurse(const std::span<const ssize_t>& shape, const double* array_buffer,
                               ssize_t array_offset, ssize_t array_dim, std::vector<double>& data,
                               ssize_t data_offset, size_t output_dim, std::vector<Update>*) const;

    std::span<const ssize_t> array_item_strides() const {
        return std::span(array_item_strides_.get(), array_ptr_->ndim());
    }

    // We could do dynamic_cast<Array*>(predecessors[0]), but since there's
    // only one we just go ahead and hold a direct pointer
    const Array* array_ptr_;

    const ssize_t ndim_;
    std::unique_ptr<ssize_t[]> strides_;
    std::unique_ptr<ssize_t[]> shape_;
    std::unique_ptr<ssize_t[]> dynamic_shape_;

    std::unique_ptr<ssize_t[]> array_item_strides_;

    const ssize_t size_;

    const std::vector<array_or_slice> indices_;
    const ssize_t indexing_arrays_ndim_;

    // "Bullet 1 mode" refers to NumPy combined indexing where there is any slice between
    // indexing arrays, e.g. A[:, i, :, j]. It is a reference to the two cases laid out
    // in bullets in the NumPy docs here:
    // https://numpy.org/doc/stable/user/basics.indexing.html#combining-advanced-and-basic-indexing
    const bool bullet1mode_;

    const ssize_t first_array_index_;
    const ssize_t subspace_stride_;
};

// Basic indexing is indexing by some combination of ints and slices.
// See https://numpy.org/doc/stable/user/basics.indexing.html#basic-indexing
// BasicIndex nodes always have exactly one predecessor representing the array.
// to be indexed.
class BasicIndexingNode : public Node, public Array {
 public:
    // Indices are some combination of slices and integers.
    using slice_or_int = std::variant<Slice, ssize_t>;

    // Runtime constructor that can be used from Cython/Python
    BasicIndexingNode(Node* array_ptr, std::vector<slice_or_int> indices);

    // Templated constructor to be used from C++
    // Just calls the runtime constructor for simplicity, but we could implement
    // a faster version by using more compile-time information
    template <ArrayNode T, class... Indices>
    explicit BasicIndexingNode(T* array_ptr, Indices... indices)
            : BasicIndexingNode(static_cast<Node*>(array_ptr), make_indices(indices...)) {}

    // Overloads needed by the Array ABC **************************************

    ssize_t ndim() const noexcept override { return ndim_; }

    std::span<const ssize_t> shape(const State& state) const override;
    std::span<const ssize_t> shape() const override { return std::span(shape_.get(), ndim_); }

    std::span<const ssize_t> strides() const override { return std::span(strides_.get(), ndim_); }

    ssize_t size(const State& state) const override;
    ssize_t size() const override { return size_; }

    // Information about the values are all inherited from the array
    bool integral() const override { return array_ptr_->integral(); }
    double min() const override { return array_ptr_->min(); }
    double max() const override { return array_ptr_->max(); }

    double const* buff(const State& state) const override;

    std::span<const Update> diff(const State& state) const override;

    ssize_t size_diff(const State& state) const override;

    SizeInfo sizeinfo() const override;

    bool contiguous() const override { return contiguous_; }

    // Overloads required by the Node ABC *************************************

    // don't need to overload update()

    void propagate(State& state) const override;

    void initialize_state(State& state) const override;

    // Never any changes to revert or commit
    void commit(State&) const override;
    void revert(State&) const override;

    // Node-specific methods

    // Infer the indices used to create the node.
    std::vector<slice_or_int> infer_indices() const;

 private:
    // Private constructor using an intermediate object
    struct IndexParser_;
    BasicIndexingNode(Array* array_ptr, IndexParser_&& parser);

    // Convert the indices from a parameter pack to a vector of variants.
    // This allows us to use the runtime constructor from the templated constructor.
    template <class... Indices>
    static std::vector<slice_or_int> make_indices(Indices&... indices) {
        std::vector<slice_or_int> output;

        auto parse = [&output](auto index) {
            if constexpr (std::is_integral<decltype(index)>::value) {
                output.push_back(static_cast<ssize_t>(index));
            } else {
                static_assert(std::is_same<Slice, decltype(index)>::value);
                output.push_back(index);
            }
        };
        (parse(indices), ...);
        return output;
    }

    void update_dynamic_shape(State& state) const;
    ssize_t dynamic_start(ssize_t slice_start) const;

    // We could do dynamic_cast<Array*>(predecessors[0]), but since there's
    // only one we just go ahead and hold a direct pointer
    const Array* array_ptr_;

    // The number of dimensions after applying the slice(s)
    const ssize_t ndim_;
    std::unique_ptr<ssize_t[]> strides_;
    std::unique_ptr<ssize_t[]> shape_;

    // the start of the view in the array_ptr's array
    const ssize_t start_;

    // The size of the array after applying slice(s)
    const ssize_t size_;

    // An optional slice on the first dimension, saved for when the predecessor array
    // is dynamic
    const std::optional<Slice> axis0_slice_;

    const bool contiguous_;
};

class PermutationNode : public Node, public ArrayOutputMixin<Array> {
 public:
    // We use this style rather than a template to support Cython later
    PermutationNode(Node* array_ptr, Node* order_ptr);

    double const* buff(const State& state) const override;
    std::span<const Update> diff(const State& state) const override;

    void initialize_state(State& state) const override;
    void commit(State& state) const override;
    void revert(State& state) const override;

    void propagate(State& state) const override;

 private:
    // we could dynamically cast each time, but it's easier to just keep separate
    // pointers to the "array" part of the two predecessors
    const Array* array_ptr_;
    const Array* order_ptr_;
};

class ReshapeNode : public Node, public ArrayOutputMixin<Array> {
 public:
    ReshapeNode(Node* node_ptr, std::initializer_list<ssize_t> shape);
    ReshapeNode(Node* node_ptr, std::span<const ssize_t> shape);

    double const* buff(const State& state) const override;
    void commit(State& state) const override;
    std::span<const Update> diff(const State& state) const override;
    void revert(State& state) const override;

 private:
    // we could dynamically cast each time, but it's easier to just keep separate
    // pointer to the "array" part of the predecessor
    const Array* array_ptr_;
};

}  // namespace dwave::optimization

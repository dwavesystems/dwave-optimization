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

#include <algorithm>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/graph.hpp"

namespace dwave::optimization {

/// A contiguous block of numbers.
class ConstantNode : public ArrayOutputMixin<ArrayNode> {
 public:
    // Default constructor - defaults to an empty 1d array.
    ConstantNode() noexcept : ConstantNode({0}) {}

    // A single scalar value
    explicit ConstantNode(double value) : ConstantNode({}) { *buffer_ptr_ = value; }

    // A pointer to another array or similar. In this case the ConstantNode will be a *view*
    // rather than a container. That is it does not manage the lifespan of the array data.
    ConstantNode(double* data_ptr, std::initializer_list<ssize_t> shape)
            : ArrayOutputMixin(shape), own_data_(false), buffer_ptr_(data_ptr) {}
    ConstantNode(double* data_ptr, std::span<const ssize_t> shape)
            : ArrayOutputMixin(shape), own_data_(false), buffer_ptr_(data_ptr) {}

    /// Create a ConstantNode by copying the contents of a range
    template <std::ranges::sized_range Range>
    explicit ConstantNode(Range&& values)
            : ConstantNode(std::forward<Range>(values), {static_cast<ssize_t>(values.size())}) {}

    /// Create a ConstantNode by copying the contents of a range and interpreting
    /// it as the given shape
    template <std::ranges::sized_range Range>
    ConstantNode(Range&& values, std::initializer_list<ssize_t> shape) : ConstantNode(shape) {
        const ssize_t size = this->size();
        if (size < 0) {
            throw std::invalid_argument("ConstantNode cannot be dynamic");
        }
        if (values.size() < static_cast<std::size_t>(size)) {
            throw std::out_of_range("too few values for the given shape");
        }
        std::copy(values.begin(), std::next(values.begin(), size), buffer_ptr_);
    }

    ~ConstantNode() {
        // Only deallocate in the instance that it owns its own data
        if (own_data_) buffer_alloc_.deallocate(buffer_ptr_, size());
    }

    bool own_data() const noexcept { return own_data_; }

    // Stateless access to the underlying data.
    std::span<const double> data() const noexcept {
        return std::span<const double>(buffer_ptr_, this->size());
    }

    // Overload of the buff() method. The first one is a stateless overload
    double const* buff() const { return buffer_ptr_; }
    double const* buff(const State&) const override { return buffer_ptr_; }

    // Overloads needed by the Array ABC **************************************

    // There are never any updates to propagate
    std::span<const Update> diff(const State& state) const noexcept override { return {}; }

    // Whether the values in the array can be interpreted as integers.
    // Returns ``true`` for an empty array. This has the slightly odd effect of
    // making empty arrays logical, because we also return 0 for min/max.
    // Note that this is an O(size) function call, whereas for most other nodes it
    // is O(1).
    bool integral() const override;

    // The maximum and minimum values of elements in the array.
    // Returns ``0.0`` for an empty array.
    // Note that this is an O(size) function call, whereas for most other nodes it
    // is O(1).
    std::pair<double, double> minmax(
            optional_cache_type<std::pair<double, double>> cache = std::nullopt) const override;

    // Overloads required by the Node ABC *************************************

    // This node never needs to update its successors
    void propagate(State&) const noexcept override {}

    // Constants don't have predecessors so no one should be calling update().
    // Always throws a logic_error.
    [[noreturn]] void update(State& state, int index) const override;

    // Never any changes to revert or commit
    void commit(State&) const noexcept override {}
    void revert(State&) const noexcept override {}

 private:
    // Allocate the memory to hold shape worth of doubles, but don't populate it
    explicit ConstantNode(std::initializer_list<ssize_t> shape)
            : ArrayOutputMixin(shape), own_data_(true) {
        buffer_ptr_ = buffer_alloc_.allocate(this->size());
    }

    // Allocator for the buffer. Does not consume any memory
    std::allocator<double> buffer_alloc_;

    // If True, the node is responsible for deallocating the data, otherwise
    // it is just a view
    const bool own_data_;

    // The beginning of the array data. The ConstantNode is unusual because it
    // holds its values on the object itself rather than in a State.
    double* buffer_ptr_;

    // Information about the values in the buffer
    struct BufferStats {
        BufferStats() = delete;
        explicit BufferStats(std::span<const double> buffer);

        bool integral;
        double min;
        double max;
    };
    mutable std::optional<BufferStats> buffer_stats_;
};

}  // namespace dwave::optimization

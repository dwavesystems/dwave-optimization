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
#include <span>
#include <vector>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/graph.hpp"

namespace dwave::optimization {

/// A contiguous block of numbers.
class ConstantNode : public Node, public ArrayOutputMixin<Array> {
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

    // Create a ConstantNode by copying the contents of a vector
    template <class T>
    explicit ConstantNode(const std::vector<T>& other)
            : ConstantNode(other, {static_cast<ssize_t>(other.size())}) {}

    // Create a ConstantNode by copying the contents of a vector and interpreting
    // it as the given shape
    template <class U>
    ConstantNode(const std::vector<U>& other, std::initializer_list<ssize_t> shape)
            : ConstantNode(shape) {
        std::copy(other.begin(), other.begin() + size(), buffer_ptr_);
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
    constexpr std::span<const Update> diff(const State& state) const noexcept override {
        return {};
    }

    // Overloads required by the Node ABC *************************************

    // This node never needs to update its successors
    constexpr void propagate(State&) const noexcept override {}

    // Never any changes to revert or commit
    constexpr void commit(State&) const noexcept override {}
    constexpr void revert(State&) const noexcept override {}

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
};

}  // namespace dwave::optimization

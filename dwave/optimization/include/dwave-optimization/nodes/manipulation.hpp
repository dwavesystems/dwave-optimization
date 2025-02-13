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

/// Array manipulation routines.

#pragma once

#include <span>
#include <vector>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/state.hpp"

namespace dwave::optimization {

class ConcatenateNode : public ArrayOutputMixin<ArrayNode> {
 public:
    explicit ConcatenateNode(std::span<ArrayNode*> array_ptrs, ssize_t axis);
    explicit ConcatenateNode(std::ranges::contiguous_range auto&& array_ptrs, ssize_t axis)
            : ConcatenateNode(std::span<ArrayNode*>(array_ptrs), axis) {}

    double const* buff(const State& state) const override;
    void commit(State& state) const override;
    std::span<const Update> diff(const State& state) const override;
    void initialize_state(State& state) const override;
    bool integral() const override;

    /// @copydoc Array::minmax()
    std::pair<double, double> minmax(
            optional_cache_type<std::pair<double, double>> cache = std::nullopt) const override;

    void propagate(State& state) const override;
    void revert(State& state) const override;

    ssize_t axis() const { return axis_; }

 private:
    ssize_t axis_;
    std::vector<ArrayNode*> array_ptrs_;
    std::vector<ssize_t> array_starts_;
};

/// An array node that is a contiguous copy of its predecessor.
class CopyNode : public ArrayOutputMixin<ArrayNode> {
 public:
    explicit CopyNode(ArrayNode* array_ptr);

    /// @copydoc Array::buff()
    double const* buff(const State& state) const override;

    /// @copydoc Node::commit()
    void commit(State& state) const override;

    /// @copydoc Array::diff()
    std::span<const Update> diff(const State& state) const override;

    /// @copydoc Node::initialize_state()
    void initialize_state(State& state) const override;

    /// @copydoc Array::integral()
    bool integral() const override;

    /// @copydoc Array::minmax()
    std::pair<double, double> minmax(
            optional_cache_type<std::pair<double, double>> cache = std::nullopt) const override;

    /// @copydoc Node::propagate()
    void propagate(State& state) const override;

    /// @copydoc Node::revert()
    void revert(State& state) const override;

    using ArrayOutputMixin::shape;

    /// @copydoc Array::shape()
    std::span<const ssize_t> shape(const State& state) const override;

    using ArrayOutputMixin::size;

    /// @copydoc Array::size()
    ssize_t size(const State& state) const override;

    /// @copydoc Array::size_diff()
    ssize_t size_diff(const State& state) const override;

 private:
    const Array* array_ptr_;
};

/// Replaces specified elements of an array with the given values.
///
/// The indexing works on the flattened array. Translated to NumPy, PutNode is
/// roughly equivalent to
/// @code{.py}
/// def put(array, indices, values):
///     array = array.copy()
///     array.flat[indices] = values
///     return array
/// @endcode
///
/// In the case of duplicate indices, the most recent update will be propagated.
class PutNode : public ArrayOutputMixin<ArrayNode> {
 public:
    /// Constructor for PutNode.
    ///
    /// @param array_ptr The target array. Cannot be dynamic.
    /// @param indices_ptr,values_ptr The indices and values to be replaced in
    ///   the target array. ``indices_ptr`` and ``values_ptr`` must be
    ///   1-dimensional and must always have the same number of elements.
    PutNode(ArrayNode* array_ptr, ArrayNode* indices_ptr, ArrayNode* values_ptr);

    /// @copydoc Array::buff()
    double const* buff(const State& state) const override;

    /// @copydoc Node::commit()
    void commit(State& state) const override;

    /// @copydoc Array::diff()
    std::span<const Update> diff(const State&) const override;

    /// @copydoc Node::initialize_state()
    void initialize_state(State& state) const override;

    /// @copydoc Array::integral()
    bool integral() const override;

    /// Return the number of indices currently "covering" each element in the array.
    std::span<const ssize_t> mask(const State& state) const;

    /// @copydoc Array::minmax()
    std::pair<double, double> minmax(
            optional_cache_type<std::pair<double, double>> cache = std::nullopt) const override;

    /// @copydoc Node::propagate()
    void propagate(State& state) const override;

    /// @copydoc Node::revert()
    void revert(State& state) const override;

 private:
    const Array* array_ptr_;
    const Array* indices_ptr_;
    const Array* values_ptr_;
};


/// Propagates the values of its predecessor, interpreted into a different shape.
class ReshapeNode : public ArrayOutputMixin<ArrayNode> {
 public:
    /// Constructor for ReshapeNode.
    ///
    /// @param array_ptr The array to be reshaped. May not be dynamic.
    /// @param shape The new shape. Must have the same size as the original shape.
    ReshapeNode(ArrayNode* array_ptr, std::vector<ssize_t>&& shape);

    /// Constructor for ReshapeNode.
    ///
    /// @param array_ptr The array to be reshaped. May not be dynamic.
    /// @param shape The new shape. Must have the same size as the original shape.
    template <std::ranges::range Range>
    ReshapeNode(ArrayNode* node_ptr, Range&& shape)
            : ReshapeNode(node_ptr, std::vector<ssize_t>(shape.begin(), shape.end())) {}

    /// @copydoc Array::buff()
    double const* buff(const State& state) const override;

    /// @copydoc Node::commit()
    void commit(State& state) const override;

    /// @copydoc Array::diff()
    std::span<const Update> diff(const State& state) const override;

    /// @copydoc Array::integral()
    bool integral() const override;

    /// @copydoc Array::minmax()
    std::pair<double, double> minmax(
            optional_cache_type<std::pair<double, double>> cache = std::nullopt) const override;

    /// @copydoc Node::revert()
    void revert(State& state) const override;

 private:
    // we could dynamically cast each time, but it's easier to just keep separate
    // pointer to the "array" part of the predecessor
    const Array* array_ptr_;
};

class SizeNode : public ScalarOutputMixin<ArrayNode> {
 public:
    explicit SizeNode(ArrayNode* node_ptr);

    double const* buff(const State& state) const override;

    void commit(State& state) const override;

    std::span<const Update> diff(const State&) const override;

    void initialize_state(State& state) const override;

    // SizeNode's value is always a non-negative integer.
    bool integral() const override { return true; }

    /// @copydoc Array::minmax()
    std::pair<double, double> minmax(
            optional_cache_type<std::pair<double, double>> cache = std::nullopt) const override;

    void propagate(State& state) const override;

    void revert(State& state) const override;

 private:
    // we could dynamically cast each time, but it's easier to just keep separate
    // pointer to the "array" part of the predecessor
    const Array* array_ptr_;
};

}  // namespace dwave::optimization

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
#include <variant>
#include <vector>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/state.hpp"

namespace dwave::optimization {

class BroadcastToNode : public ArrayNode {
 public:
    BroadcastToNode(ArrayNode* array_ptr, std::initializer_list<ssize_t> shape);
    BroadcastToNode(ArrayNode* array_ptr, std::span<const ssize_t> shape);

    /// @copydoc Array::buff()
    double const* buff(const State& state) const override;

    /// @copydoc Node::commit()
    void commit(State& state) const override;

    /// Broadcast nodes are always treated as non-contiguous.
    bool contiguous() const override { return false; }

    /// @copydoc Array::diff()
    std::span<const Update> diff(const State& state) const override;

    /// @copydoc Node::initialize_state()
    void initialize_state(State& state) const override;

    /// @copydoc Array::integral()
    bool integral() const override;

    /// @copydoc Array::min()
    double min() const override;

    /// @copydoc Array::max()
    double max() const override;

    /// @copydoc Array::ndim()
    ssize_t ndim() const override;

    /// @copydoc Node::propagate()
    void propagate(State& state) const override;

    /// @copydoc Node::revert()
    void revert(State& state) const override;

    /// @copydoc Array::shape()
    std::span<const ssize_t> shape() const override;
    std::span<const ssize_t> shape(const State& state) const override;

    /// @copydoc Array::size()
    ssize_t size() const override;
    ssize_t size(const State& state) const override;

    /// @copydoc Array::size_diff()
    ssize_t size_diff(const State& state) const override;

    /// @copydoc Array::strides()
    std::span<const ssize_t> strides() const override;

 private:
    /// Translate a linear index of the predecessor into a linear index of the
    /// BroadcastToNode.
    ssize_t convert_predecessor_index_(ssize_t index) const;

    ArrayNode* array_ptr_;

    ssize_t ndim_;
    std::unique_ptr<ssize_t[]> shape_;
    std::unique_ptr<ssize_t[]> strides_;

    const ValuesInfo values_info_;
};

class ConcatenateNode : public ArrayOutputMixin<ArrayNode> {
 public:
    explicit ConcatenateNode(std::span<ArrayNode*> array_ptrs, ssize_t axis);
    explicit ConcatenateNode(std::ranges::contiguous_range auto&& array_ptrs, ssize_t axis)
            : ConcatenateNode(std::span<ArrayNode*>(array_ptrs), axis) {}

    double const* buff(const State& statfe) const override;
    void commit(State& state) const override;
    std::span<const Update> diff(const State& state) const override;
    void initialize_state(State& state) const override;

    /// @copydoc Array::integral()
    bool integral() const override;

    /// @copydoc Array::min()
    double min() const override;

    /// @copydoc Array::max()
    double max() const override;

    void propagate(State& state) const override;
    void revert(State& state) const override;

    ssize_t axis() const { return axis_; }

 private:
    ssize_t axis_;
    std::vector<ArrayNode*> array_ptrs_;
    std::vector<ssize_t> array_starts_;

    const ValuesInfo values_info_;
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

    /// @copydoc Array::min()
    double min() const override;

    /// @copydoc Array::max()
    double max() const override;

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

    const ValuesInfo values_info_;
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

    /// @copydoc Array::min()
    double min() const override;

    /// @copydoc Array::max()
    double max() const override;

    /// @copydoc Node::propagate()
    void propagate(State& state) const override;

    /// @copydoc Node::revert()
    void revert(State& state) const override;

 private:
    const Array* array_ptr_;
    const Array* indices_ptr_;
    const Array* values_ptr_;

    const ValuesInfo values_info_;
};

/// Propagates the values of its predecessor, interpreted into a different shape.
class ReshapeNode : public ArrayOutputMixin<ArrayNode> {
 public:
    /// Constructor for ReshapeNode.
    ///
    /// @param array_ptr The array to be reshaped.
    /// @param shape The new shape. Must have the same size as the original shape.
    ReshapeNode(ArrayNode* array_ptr, std::vector<ssize_t>&& shape);

    /// Constructor for ReshapeNode.
    ///
    /// @param array_ptr The array to be reshaped.
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

    /// @copydoc Node::initialize_state()
    void initialize_state(State& state) const override;

    /// @copydoc Array::integral()
    bool integral() const override;

    /// @copydoc Array::min()
    double min() const override;

    /// @copydoc Array::max()
    double max() const override;

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

    /// @copydoc Array::sizeinfo()
    SizeInfo sizeinfo() const override;

    /// @copydoc Array::size_diff()
    ssize_t size_diff(const State& state) const override;

 private:
    // we could dynamically cast each time, but it's easier to just keep separate
    // pointer to the "array" part of the predecessor
    const Array* array_ptr_;

    const ValuesInfo values_info_;
    const SizeInfo sizeinfo_;
};

/// Reshape a node to a specific non-dynamic shape. Use fill_value for any missing
/// values.
class ResizeNode : public ArrayOutputMixin<ArrayNode> {
 public:
    /// Constructor for ResizeNode.
    ///
    /// @param array_ptr The array to be resized.
    /// @param shape The new shape. Must not be dynamic.
    /// @param fill_value The value to use for missing values.
    ResizeNode(ArrayNode* array_ptr, std::vector<ssize_t>&& shape, double fill_value = 0);

    /// Constructor for ResizeNode.
    ///
    /// @param array_ptr The array to be resized.
    /// @param shape The new shape. Must not be dynamic.
    /// @param fill_value The value to use for missing values.
    template <std::ranges::range Range>
    ResizeNode(ArrayNode* node_ptr, Range&& shape, double fill_value = 0)
            : ResizeNode(node_ptr, std::vector<ssize_t>(shape.begin(), shape.end()), fill_value) {}

    /// @copydoc Array::buff()
    double const* buff(const State& state) const override;

    /// @copydoc Node::commit()
    void commit(State& state) const override;

    /// @copydoc Array::diff()
    std::span<const Update> diff(const State& state) const override;

    /// The fill value.
    double fill_value() const { return fill_value_; }

    /// @copydoc Node::initialize_state()
    void initialize_state(State& state) const override;

    /// @copydoc Array::integral()
    bool integral() const override;

    /// @copydoc Array::min()
    double min() const override;

    /// @copydoc Array::max()
    double max() const override;

    /// @copydoc Node::initialize_state()
    void propagate(State& state) const override;

    /// @copydoc Node::revert()
    void revert(State& state) const override;

 private:
    const Array* array_ptr_;

    const double fill_value_;

    const ValuesInfo values_info_;
};

class RollNode : public ArrayOutputMixin<ArrayNode> {
 public:
    /// Construct a RollNode.
    ///
    /// `shift` is a single integer, a vector of shifts per-axis, or an array encoding
    /// the same.
    ///
    /// If `axis` is empty then the array is treated as flat while rolling. Otherwise
    /// the given axes are shifted.
    RollNode(ArrayNode* array_ptr, ssize_t shift, std::vector<ssize_t> axis = {});
    RollNode(ArrayNode* array_ptr, std::vector<ssize_t> shift, std::vector<ssize_t> axis = {});
    RollNode(ArrayNode* array_ptr, ArrayNode* shift, std::vector<ssize_t> axis = {});

    /// The axes upon which the roll is performed. If empty the array is treated as flat
    /// when rolling.
    std::span<const ssize_t> axes() const;

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

    /// @copydoc Array::min()
    double min() const override;

    /// @copydoc Array::max()
    double max() const override;

    /// @copydoc Node::initialize_state()
    void propagate(State& state) const override;

    /// @copydoc Node::revert()
    void revert(State& state) const override;

    using ArrayOutputMixin::shape;

    /// @copydoc Array::shape()
    std::span<const ssize_t> shape(const State& state) const override;

    /// The shift value used by the node.
    const std::variant<const Array*, std::vector<ssize_t>>& shift() const;

    using ArrayOutputMixin::size;

    /// @copydoc Array::size()
    ssize_t size(const State& state) const override;

    /// @copydoc Array::sizeinfo()
    SizeInfo sizeinfo() const override;

    /// @copydoc Array::size_diff()
    ssize_t size_diff(const State& state) const override;

 private:
    // Rotate the given array by shift in-place.
    static void rotate_(std::span<double> array, ssize_t shift);

    // Rotate the given array with the given shift by the shift given for each
    // axis. Acts in-place.
    static void rotate_(std::span<double> array, std::span<const ssize_t> shape,
                        std::span<const ssize_t> shifts);

    // Return the current shift, and whether is changed since the last propagation.
    std::tuple<ssize_t, bool> shift_diff_(const State& state) const;

    // Return the current shifts for each axis and whether any have changed since
    // the last propagation
    std::tuple<std::vector<ssize_t>, bool> shifts_diff_(const State& state) const;

    const Array* array_ptr_;

    std::variant<const Array*, std::vector<ssize_t>> shift_;

    std::vector<ssize_t> axis_;

    const ValuesInfo values_info_;
    const SizeInfo sizeinfo_;
};

class SizeNode : public ScalarOutputMixin<ArrayNode, true> {
 public:
    explicit SizeNode(ArrayNode* node_ptr);

    void initialize_state(State& state) const override;

    // SizeNode's value is always a non-negative integer.
    bool integral() const override { return true; }

    /// @copydoc Array::min()
    double min() const override;

    /// @copydoc Array::max()
    double max() const override;

    void propagate(State& state) const override;

 private:
    // we could dynamically cast each time, but it's easier to just keep separate
    // pointer to the "array" part of the predecessor
    const Array* array_ptr_;

    const std::pair<double, double> minmax_;
};

// Compute the transpose of predecessor
class TransposeNode : public ArrayNode {
 public:
    TransposeNode(ArrayNode* array_ptr);

    // Overloads needed by the Array ABC **************************************

    /// @copydoc Array::buff()
    double const* buff(const State& state) const override;

    /// @copydoc Array::ndim()
    ssize_t ndim() const override;

    /// @copydoc Array::shape()
    std::span<const ssize_t> shape(const State& state) const override;
    std::span<const ssize_t> shape() const override;

    /// @copydoc Array::strides()
    std::span<const ssize_t> strides() const override;

    /// @copydoc Array::size()
    ssize_t size() const override;
    ssize_t size(const State& state) const override;

    /// @copydoc Array::min()
    double min() const override;

    /// @copydoc Array::max()
    double max() const override;

    /// @copydoc Array::integral()
    bool integral() const override;

    /// @copydoc Array::contiguous()
    bool contiguous() const override;

    /// @copydoc Array::diff()
    std::span<const Update> diff(const State& state) const override;

    /// @copydoc Array::size_diff()
    ssize_t size_diff(const State& state) const override;

    // Overloads required by the Node ABC *************************************

    void initialize_state(State& state) const override;

    void propagate(State& state) const override;

    void commit(State&) const override;

    void revert(State&) const override;

 private:
    const Array* array_ptr_;

    const ssize_t ndim_;
    const std::unique_ptr<ssize_t[]> shape_;
    const std::unique_ptr<ssize_t[]> strides_;
    const bool contiguous_;
    const ValuesInfo values_info_;

    ArrayNode* predeccesor_check_(ArrayNode* array_ptr) const;
    Update convert_predecessor_update_(Update update) const;
};

}  // namespace dwave::optimization

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
#include <array>
#include <cmath>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/functional.hpp"
#include "dwave-optimization/graph.hpp"

namespace dwave::optimization {

template <class BinaryOp>
class BinaryOpNode : public ArrayOutputMixin<ArrayNode> {
 public:
    // We need at least two nodes, and they must be the same shape
    BinaryOpNode(ArrayNode* a_ptr, ArrayNode* b_ptr);

    double const* buff(const State& state) const override;
    std::span<const Update> diff(const State& state) const override;
    bool integral() const override;

    /// @copydoc Array::minmax()
    std::pair<double, double> minmax(
            optional_cache_type<std::pair<double, double>> cache = std::nullopt) const override;

    using ArrayOutputMixin::shape;
    std::span<const ssize_t> shape(const State& state) const override;

    using ArrayOutputMixin::size;
    ssize_t size(const State& state) const override;

    ssize_t size_diff(const State& state) const override;

    SizeInfo sizeinfo() const override;

    void commit(State& state) const override;
    void revert(State& state) const override;
    void initialize_state(State& state) const override;
    void propagate(State& state) const override;

    // The predecessors of the operation, as Array*.
    std::span<Array* const> operands() {
        assert(predecessors().size() == operands_.size());
        return operands_;
    }
    std::span<const Array* const> operands() const {
        assert(predecessors().size() == operands_.size());
        return operands_;
    }

 private:
    BinaryOp op;

    // There are redundant, because we could dynamic_cast each time from
    // predecessors(), but this is more performant
    std::array<Array* const, 2> operands_;
};

// We follow NumPy naming convention rather than C++ to distinguish between
// binary operations and reduce operations.
// https://numpy.org/doc/stable/reference/routines.math.html
using AddNode = BinaryOpNode<std::plus<double>>;
using AndNode = BinaryOpNode<std::logical_and<double>>;
using DivideNode = BinaryOpNode<std::divides<double>>;
using EqualNode = BinaryOpNode<std::equal_to<double>>;
using LessEqualNode = BinaryOpNode<std::less_equal<double>>;
using MultiplyNode = BinaryOpNode<std::multiplies<double>>;
using MaximumNode = BinaryOpNode<functional::max<double>>;
using MinimumNode = BinaryOpNode<functional::min<double>>;
using ModulusNode = BinaryOpNode<functional::modulus<double>>;
using OrNode = BinaryOpNode<std::logical_or<double>>;
using SafeDivideNode = BinaryOpNode<functional::safe_divides<double>>;
using SubtractNode = BinaryOpNode<std::minus<double>>;
using XorNode = BinaryOpNode<functional::logical_xor<double>>;

template <class BinaryOp>
class NaryOpNode : public ArrayOutputMixin<ArrayNode> {
 public:
    // Need at least one node to start with to determine the shape
    explicit NaryOpNode(ArrayNode* node_ptr);
    explicit NaryOpNode(std::span<ArrayNode*> node_ptrs);
    explicit NaryOpNode(ArrayNode* node_ptr, std::convertible_to<ArrayNode*> auto... node_ptrs)
            : NaryOpNode(node_ptr) {
        (add_node(node_ptrs), ...);
    }

    void add_node(ArrayNode* node_ptr);

    double const* buff(const State& state) const override;
    std::span<const Update> diff(const State& state) const override;
    bool integral() const override;

    /// @copydoc Array::minmax()
    std::pair<double, double> minmax(
            optional_cache_type<std::pair<double, double>> cache = std::nullopt) const override;

    void commit(State& state) const override;
    void revert(State& state) const override;
    void initialize_state(State& state) const override;
    void propagate(State& state) const override;

    // The predecessors of the operation, as Array*.
    std::span<Array* const> operands() {
        assert(predecessors().size() == operands_.size());
        return operands_;
    }
    std::span<const Array* const> operands() const {
        assert(predecessors().size() == operands_.size());
        return operands_;
    }

 private:
    BinaryOp op;

    std::vector<Array*> operands_;
};

using NaryAddNode = NaryOpNode<std::plus<double>>;
using NaryMaximumNode = NaryOpNode<functional::max<double>>;
using NaryMinimumNode = NaryOpNode<functional::min<double>>;
using NaryMultiplyNode = NaryOpNode<std::multiplies<double>>;

/// TODO: support multiple axes
template <class BinaryOp>
class PartialReduceNode : public ArrayOutputMixin<ArrayNode> {
 public:
    // Runtime constructor
    PartialReduceNode(ArrayNode* array_ptr, std::span<const ssize_t> axes, double init);
    PartialReduceNode(ArrayNode* array_ptr, std::initializer_list<ssize_t> axes, double init);
    PartialReduceNode(ArrayNode* array_ptr, ssize_t axis, double init);

    // Some operations have known default values so we can create them regardless of
    // whether or not the array is dynamic.
    // Others will raise an error for non-dynamic arrays.
    explicit PartialReduceNode(ArrayNode* array_ptr, std::span<const ssize_t> axes);
    explicit PartialReduceNode(ArrayNode* array_ptr, std::initializer_list<ssize_t> axes);
    explicit PartialReduceNode(ArrayNode* array_ptr, ssize_t axis);

    std::span<const ssize_t> axes() const;
    double const* buff(const State& state) const override;

    void commit(State& state) const override;
    std::span<const Update> diff(const State& state) const override;
    void initialize_state(State& state) const override;

    bool integral() const override;

    /// @copydoc Array::minmax()
    std::pair<double, double> minmax(
            optional_cache_type<std::pair<double, double>> cache = std::nullopt) const override;

    // The predecessor of the reduction, as an Array*.
    std::span<Array* const> operands() {
        assert(predecessors().size() == 1);
        return std::span<Array* const, 1>(&array_ptr_, 1);
    }
    std::span<const Array* const> operands() const {
        assert(predecessors().size() == 1);
        return std::span<const Array* const, 1>(&array_ptr_, 1);
    }

    void propagate(State& state) const override;
    void revert(State& state) const override;

    using ArrayOutputMixin::shape;
    std::span<const ssize_t> shape(const State& state) const override;

    using ArrayOutputMixin::size;
    ssize_t size(const State& state) const override;
    ssize_t size_diff(const State& state) const override;

    const std::optional<double> init;

 private:
    BinaryOp op;

    // There are redundant, because we could dynamic_cast each time from
    // predecessors(), but this is more performant
    Array* const array_ptr_;

    // The axis along which to do the
    std::unique_ptr<ssize_t[]> axes_ = nullptr;

    template <class Range>
    static std::unique_ptr<ssize_t[]> make_axes(Range&& axes) noexcept {
        if (axes.size() == 0) return nullptr;
        auto ptr = std::make_unique<ssize_t[]>(axes.size());
        std::copy(axes.begin(), axes.end(), ptr.get());
        return ptr;
    }

    /// Map the parent index to the affected array index (linear)
    ssize_t map_parent_index(const State& state, ssize_t parent_flat_index) const;

    /// Convert linear index to indices for each dimension
    std::vector<ssize_t> parent_strides_c_;
};

using PartialProdNode = PartialReduceNode<std::multiplies<double>>;
using PartialSumNode = PartialReduceNode<std::plus<double>>;

template <class BinaryOp>
class ReduceNode : public ScalarOutputMixin<ArrayNode> {
 public:
    ReduceNode(ArrayNode* node_ptr, double init);

    // Some operations have known default values so we can create them regardless of
    // whether or not the array is dynamic.
    // Others will raise an error for non-dynamic arrays.
    explicit ReduceNode(ArrayNode* node_ptr);

    double const* buff(const State& state) const override;
    std::span<const Update> diff(const State& state) const override;
    bool integral() const override;

    /// @copydoc Array::minmax()
    std::pair<double, double> minmax(
            optional_cache_type<std::pair<double, double>> cache = std::nullopt) const override;

    void commit(State& state) const override;
    void revert(State& state) const override;
    void initialize_state(State& state) const override;
    void propagate(State& state) const override;

    // The predecessor of the reduction, as an Array*.
    std::span<Array* const> operands() {
        assert(predecessors().size() == 1);
        return std::span<Array* const, 1>(&array_ptr_, 1);
    }
    std::span<const Array* const> operands() const {
        assert(predecessors().size() == 1);
        return std::span<const Array* const, 1>(&array_ptr_, 1);
    }

    const std::optional<double> init;

 private:
    BinaryOp op;

    // Calculate the output value based on the state of the predecessor
    double reduce(const State& state) const;

    // There are redundant, because we could dynamic_cast each time from
    // predecessors(), but this is more performant
    Array* const array_ptr_;
};

// We follow NumPy naming convention rather than C++ to distinguish between
// binary operations and reduce operations.
// https://numpy.org/doc/stable/reference/routines.math.html
using AllNode = ReduceNode<std::logical_and<double>>;
using AnyNode = ReduceNode<std::logical_or<double>>;
using MaxNode = ReduceNode<functional::max<double>>;
using MinNode = ReduceNode<functional::min<double>>;
using ProdNode = ReduceNode<std::multiplies<double>>;
using SumNode = ReduceNode<std::plus<double>>;

// SoftMaxNode *****************************************************************

class SoftMaxNode : public ArrayOutputMixin<ArrayNode> {
 public:
    SoftMaxNode(ArrayNode* arr_ptr);

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

    using Array::shape;

    /// @copydoc Array::shape()
    std::span<const ssize_t> shape(const State& state) const override;

    using Array::size;

    /// @copydoc Array::size()
    ssize_t size(const State& state) const override;

    /// @copydoc Array::size_diff()
    ssize_t size_diff(const State& state) const override;

    /// @copydoc Array::sizeinfo()
    SizeInfo sizeinfo() const override;

 private:
    // these are redundant, but convenient
    const Array* arr_ptr_;
};

template <class UnaryOp>
class UnaryOpNode : public ArrayOutputMixin<ArrayNode> {
 public:
    explicit UnaryOpNode(ArrayNode* node_ptr);

    double const* buff(const State& state) const override;
    std::span<const Update> diff(const State& state) const override;
    bool integral() const override;

    /// @copydoc Array::minmax()
    std::pair<double, double> minmax(
            optional_cache_type<std::pair<double, double>> cache = std::nullopt) const override;

    using ArrayOutputMixin::shape;
    std::span<const ssize_t> shape(const State& state) const override;
    using ArrayOutputMixin::size;
    ssize_t size(const State& state) const override;
    ssize_t size_diff(const State& state) const override;
    SizeInfo sizeinfo() const override;

    void commit(State& state) const override;
    void revert(State& state) const override;
    void initialize_state(State& state) const override;
    void propagate(State& state) const override;

    // The predecessor of the operation, as an Array*.
    std::span<Array* const> operands() {
        assert(predecessors().size() == 1);
        return std::span<Array* const, 1>(&array_ptr_, 1);
    }
    std::span<const Array* const> operands() const {
        assert(predecessors().size() == 1);
        return std::span<const Array* const, 1>(&array_ptr_, 1);
    }

 private:
    UnaryOp op;

    // There are redundant, because we could dynamic_cast each time from
    // predecessors(), but this is more performant
    Array* const array_ptr_;
};

using AbsoluteNode = UnaryOpNode<functional::abs<double>>;
using CosNode = UnaryOpNode<functional::cos<double>>;
using ExpitNode = UnaryOpNode<functional::expit<double>>;
using ExpNode = UnaryOpNode<functional::exp<double>>;
using LogNode = UnaryOpNode<functional::log<double>>;
using LogicalNode = UnaryOpNode<functional::logical<double>>;
using NegativeNode = UnaryOpNode<std::negate<double>>;
using NotNode = UnaryOpNode<std::logical_not<double>>;
using RintNode = UnaryOpNode<functional::rint<double>>;
using SinNode = UnaryOpNode<functional::sin<double>>;
using SquareNode = UnaryOpNode<functional::square<double>>;
using SquareRootNode = UnaryOpNode<functional::square_root<double>>;

}  // namespace dwave::optimization

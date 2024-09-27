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
#include <functional>
#include <optional>
#include <vector>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/graph.hpp"

namespace dwave::optimization {

namespace functional {

template <class T>
struct abs {
    constexpr T operator()(const T& x) const { return std::abs(x); }
};

template <class T>
struct logical {
    constexpr bool operator()(const T& x) const { return x; }
};

template <class T>
struct max {
    constexpr T operator()(const T& x, const T& y) const { return std::max(x, y); }
};

template <class T>
struct min {
    constexpr T operator()(const T& x, const T& y) const { return std::min(x, y); }
};

template <class T>
struct square {
    constexpr T operator()(const T& x) const { return x * x; }
};

}  // namespace functional

template <class BinaryOp>
class BinaryOpNode : public ArrayOutputMixin<ArrayNode> {
 public:
    // We need at least two nodes, and they must be the same shape
    BinaryOpNode(ArrayNode* a_ptr, ArrayNode* b_ptr);

    double const* buff(const State& state) const override;
    std::span<const Update> diff(const State& state) const override;
    bool integral() const override;
    double max() const override;
    double min() const override;

    using ArrayOutputMixin::shape;
    std::span<const ssize_t> shape(const State& state) const override;

    using ArrayOutputMixin::size;
    ssize_t size(const State& state) const override;

    ssize_t size_diff(const State& state) const override;

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
    using op = BinaryOp;

    // There are redundant, because we could dynamic_cast each time from
    // predecessors(), but this is more performant
    std::array<Array* const, 2> operands_;
};

// We follow NumPy naming convention rather than C++ to distinguish between
// binary operations and reduce operations.
// https://numpy.org/doc/stable/reference/routines.math.html
using AddNode = BinaryOpNode<std::plus<double>>;
using AndNode = BinaryOpNode<std::logical_and<double>>;
using EqualNode = BinaryOpNode<std::equal_to<double>>;
using LessEqualNode = BinaryOpNode<std::less_equal<double>>;
using MultiplyNode = BinaryOpNode<std::multiplies<double>>;
using MaximumNode = BinaryOpNode<functional::max<double>>;
using MinimumNode = BinaryOpNode<functional::min<double>>;
using OrNode = BinaryOpNode<std::logical_or<double>>;
using SubtractNode = BinaryOpNode<std::minus<double>>;

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
    double max() const override;
    double min() const override;

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
    using op = BinaryOp;

    std::vector<Array*> operands_;
};

using NaryAddNode = NaryOpNode<std::plus<double>>;
using NaryMaximumNode = NaryOpNode<functional::max<double>>;
using NaryMinimumNode = NaryOpNode<functional::min<double>>;
using NaryMultiplyNode = NaryOpNode<std::multiplies<double>>;

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
    double max() const override;
    double min() const override;

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
    using op = BinaryOp;

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
using MaxNode = ReduceNode<functional::max<double>>;
using MinNode = ReduceNode<functional::min<double>>;
using ProdNode = ReduceNode<std::multiplies<double>>;
using SumNode = ReduceNode<std::plus<double>>;

template <class UnaryOp>
class UnaryOpNode : public ArrayOutputMixin<ArrayNode> {
 public:
    explicit UnaryOpNode(ArrayNode* node_ptr);

    double const* buff(const State& state) const override;
    std::span<const Update> diff(const State& state) const override;
    bool integral() const override;
    double max() const override;
    double min() const override;

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
    using op = UnaryOp;

    // There are redundant, because we could dynamic_cast each time from
    // predecessors(), but this is more performant
    Array* const array_ptr_;
};

using AbsoluteNode = UnaryOpNode<functional::abs<double>>;
using LogicalNode = UnaryOpNode<functional::logical<double>>;
using NegativeNode = UnaryOpNode<std::negate<double>>;
using NotNode = UnaryOpNode<std::logical_not<double>>;
using SquareNode = UnaryOpNode<functional::square<double>>;

}  // namespace dwave::optimization

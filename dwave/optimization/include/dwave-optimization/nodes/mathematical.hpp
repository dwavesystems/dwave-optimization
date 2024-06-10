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
#include <functional>
#include <optional>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/graph.hpp"

namespace dwave::optimization {

namespace functional {

template <class T>
struct abs {
    constexpr T operator()(const T& x) const { return std::abs(x); }
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
class BinaryOpNode : public Node, public ArrayOutputMixin<Array> {
 public:
    // We need at least two nodes, and they must be the same shape
    BinaryOpNode(Node* a_ptr, Node* b_ptr);

    double const* buff(const State& state) const override;
    std::span<const Update> diff(const State& state) const override;
    bool integral() const override;
    double max() const override;
    double min() const override;

    void commit(State& state) const override;
    void revert(State& state) const override;
    void initialize_state(State& state) const override;
    void propagate(State& state) const override;

 private:
    using op = BinaryOp;

    // There are redundant, because we could dynamic_cast each time from
    // predecessors(), but that gets tedious as well as having a very minor
    // performance hit. So we go ahead and hold dedicated pointers here.
    const Array* const lhs_ptr_;
    const Array* const rhs_ptr_;
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
class NaryOpNode : public Node, public ArrayOutputMixin<Array> {
 public:
    // Need at least one node to start with to determine the shape
    NaryOpNode(Node* node_ptr);

    NaryOpNode(std::vector<Node*>& node_ptrs);

    void add_node(Node* node_ptr);

    double const* buff(const State& state) const override;
    std::span<const Update> diff(const State& state) const override;
    bool integral() const override;
    double max() const override;
    double min() const override;

    void commit(State& state) const override;
    void revert(State& state) const override;
    void initialize_state(State& state) const override;
    void propagate(State& state) const override;

 private:
    using op = BinaryOp;

    static Node*& verify_node_list_(std::vector<Node*>& node_ptrs) {
        if (!node_ptrs.size()) {
            throw std::invalid_argument("Must supply at least one predecessor");
        }
        return node_ptrs[0];
    }
};

using NaryAddNode = NaryOpNode<std::plus<double>>;
using NaryMaximumNode = NaryOpNode<functional::max<double>>;
using NaryMinimumNode = NaryOpNode<functional::min<double>>;
using NaryMultiplyNode = NaryOpNode<std::multiplies<double>>;

template <class BinaryOp>
class ReduceNode : public Node, public ScalarOutputMixin<Array> {
 public:
    template <ArrayNode T>
    ReduceNode(T* array_ptr, double init) : init(init), array_ptr_(array_ptr) {
        add_predecessor(array_ptr);
    }

    ReduceNode(Node* node_ptr, double init)
            : init(init), array_ptr_(dynamic_cast<Array*>(node_ptr)) {
        if (!array_ptr_) {
            throw std::invalid_argument("node_ptr must be an Array");
        }
        add_predecessor(node_ptr);
    }

    // Some operations have known default values so we can create them regardless of
    // whether or not the array is dynamic.
    // Others will raise an error for non-dynamic arrays.
    template <class T>
    explicit ReduceNode(T* array_ptr);

    double const* buff(const State& state) const override;
    std::span<const Update> diff(const State& state) const override;
    bool integral() const override;
    double max() const override;
    double min() const override;

    void commit(State& state) const override;
    void revert(State& state) const override;
    void initialize_state(State& state) const override;
    void propagate(State& state) const override;

    const std::optional<double> init;

 private:
    using op = BinaryOp;

    // Calculate the output value based on the state of the predecessor
    double reduce(const State& state) const;

    // This is redundant, because we could dynamic_cast each time from
    // predecessors(), but that gets tedious as well as having a very minor
    // performance hit. So we go ahead and hold a dedicated pointer here.
    const Array* const array_ptr_;
};

template <>
template <class T>
ReduceNode<std::multiplies<double>>::ReduceNode(T* array_ptr) : ReduceNode(array_ptr, 1) {}

template <>
template <class T>
ReduceNode<std::plus<double>>::ReduceNode(T* array_ptr) : ReduceNode(array_ptr, 0) {}

template <class BinaryOp>
template <class T>
ReduceNode<BinaryOp>::ReduceNode(T* array_ptr)
        : init(), array_ptr_(dynamic_cast<Array*>(array_ptr)) {
    if (array_ptr_->dynamic()) {
        throw std::invalid_argument(
                "cannot do a reduction on a dynamic array with an operation that has no identity "
                "without supplying an initial value");
    } else if (array_ptr_->size() < 1) {
        throw std::invalid_argument(
                "cannot do a reduction on an empty array with an operation that has no identity "
                "without supplying an initial value");
    }

    add_predecessor(array_ptr);
}

// We follow NumPy naming convention rather than C++ to distinguish between
// binary operations and reduce operations.
// https://numpy.org/doc/stable/reference/routines.math.html
using AllNode = ReduceNode<std::logical_and<double>>;
using MaxNode = ReduceNode<functional::max<double>>;
using MinNode = ReduceNode<functional::min<double>>;
using ProdNode = ReduceNode<std::multiplies<double>>;
using SumNode = ReduceNode<std::plus<double>>;

template <class UnaryOp>
class UnaryOpNode : public Node, public ArrayOutputMixin<Array> {
 public:
    template <ArrayNode T>
    UnaryOpNode(T* array_ptr) : ArrayOutputMixin(array_ptr->shape()), array_ptr_(array_ptr) {
        add_predecessor(array_ptr);
    }

    UnaryOpNode(Node* node_ptr) : ArrayOutputMixin(dynamic_cast<Array*>(node_ptr)->shape()),
                                  array_ptr_(dynamic_cast<Array*>(node_ptr)) {
        if (!array_ptr_) {
            throw std::invalid_argument("node_ptr must be an Array");
        }
        add_predecessor(node_ptr);
    }

    double const* buff(const State& state) const override;
    std::span<const Update> diff(const State& state) const override;

    void commit(State& state) const override;
    void revert(State& state) const override;
    void initialize_state(State& state) const override;
    void propagate(State& state) const override;

 private:
    using op = UnaryOp;

    // This is redundant, because we could dynamic_cast each time from
    // predecessors(), but that gets tedious as well as having a very minor
    // performance hit. So we go ahead and hold a dedicated pointer here.
    const Array* const array_ptr_;
};

using AbsoluteNode = UnaryOpNode<functional::abs<double>>;
using NegativeNode = UnaryOpNode<std::negate<double>>;
using SquareNode = UnaryOpNode<functional::square<double>>;

}  // namespace dwave::optimization

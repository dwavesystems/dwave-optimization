// Copyright 2024 D-Wave Inc.
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

#include <variant>
#include <vector>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/nodes/constants.hpp"
#include "dwave-optimization/nodes/mathematical.hpp"

namespace dwave::optimization {

// Performs an n-ary element-wise reduction operation on the 1d array operands.
//
// The operation is taken in as another (separate) `Graph`, which is expected
// to have n + 1 `InputNode`s, where n is the number of operands. The extra
// `InputNode` will be used for the previous/initial value of the output of
// the reduction. Following the convention of `numpy.ufunc.reduce()` and
// `std::accumulate()`, the special input should be the first `InputNode`
// on the given `Graph`, with the remaining inputs used for the values of
// the operands.
class NaryReduceNode : public ArrayOutputMixin<ArrayNode> {
 public:
    // Initial value can either be a double or another node
    using array_or_double = std::variant<ArrayNode*, double>;

    NaryReduceNode(Graph&& expression, const std::vector<ArrayNode*>& operands, array_or_double initial);

    NaryReduceNode(Graph&& expression, const std::vector<ArrayNode*>& operands, double initial) : NaryReduceNode(std::move(expression), operands, array_or_double(initial)) {};
    NaryReduceNode(Graph&& expression, const std::vector<ArrayNode*>& operands, ArrayNode* initial) : NaryReduceNode(std::move(expression), operands, array_or_double(initial)) {};

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

    std::span<const ssize_t> shape(const State& state) const override;

    ssize_t size(const State& state) const override;
    ssize_t size_diff(const State& state) const override;
    SizeInfo sizeinfo() const override;

    void swap_expression(Graph&& other) { std::swap(expression_, other); };

    const array_or_double initial;

 private:
    double evaluate_expression(State& register_) const;

    double get_initial_value(const State& state) const;

    std::span<const InputNode* const> operand_inputs() const;
    const InputNode* const reduction_input() const;

    Graph expression_;
    const std::vector<ArrayNode*> operands_;
    const ArrayNode* output_;
};

void validate_expression(const Graph& expression);

void validate_naryreduce_arguments(const Graph& expression, const std::vector<ArrayNode*> operands);

using NaryReduceSupportedNodes =
        std::variant<const InputNode*, const ConstantNode*, const MaximumNode*, const NegativeNode*,
                     const AddNode*, const SubtractNode*, const MultiplyNode*>;

}  // namespace dwave::optimization

// Copyright 2025 D-Wave Inc.
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

#include <memory>
#include <variant>
#include <vector>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/nodes/binaryop.hpp"
#include "dwave-optimization/nodes/constants.hpp"
#include "dwave-optimization/nodes/unaryop.hpp"
#include "dwave-optimization/type_list.hpp"

namespace dwave::optimization {

// Performs an n-ary element-wise accumulate operation on the 1d array operands.
//
// The operation is taken in as another (separate) `Graph`, which is expected
// to have n + 1 `InputNode`s, where n is the number of operands. The extra
// `InputNode` will be used for the previous/initial value of the output of
// the accumulate. Following the convention of `numpy.ufunc.accumulate()` and
// `std::accumulate()`, the special input should be the first `InputNode`
// on the given `Graph`, with the remaining inputs used for the values of
// the operands.
class AccumulateZipNode : public ArrayOutputMixin<ArrayNode> {
 public:
    // Initial value can either be a double or another node
    using array_or_double = std::variant<ArrayNode*, double>;

    /// The node types that are allowed to be used in the expression.
    using supported_node_types =
            type_list<AddNode, AndNode, ConstantNode, DivideNode, EqualNode, InputNode,
                      LessEqualNode, MaximumNode, MinimumNode, MultiplyNode, NegativeNode, OrNode,
                      SubtractNode, XorNode>;

    AccumulateZipNode(std::shared_ptr<Graph> expression_ptr,
                      const std::vector<ArrayNode*>& operands, array_or_double initial);

    AccumulateZipNode(Graph&& expression, const std::vector<ArrayNode*>& operands,
                      array_or_double initial)
            : AccumulateZipNode(std::make_shared<Graph>(std::move(expression)), operands, initial) {
    }
    AccumulateZipNode(Graph&& expression, const std::vector<ArrayNode*>& operands, double initial)
            : AccumulateZipNode(std::move(expression), operands, array_or_double(initial)) {}
    AccumulateZipNode(Graph&& expression, const std::vector<ArrayNode*>& operands,
                      ArrayNode* initial)
            : AccumulateZipNode(std::move(expression), operands, array_or_double(initial)) {}

    /// @copydoc Array::buff()
    double const* buff(const State& state) const override;

    /// Do a "dry run" of the constructor and raise any errors that constructor would
    /// raise.
    static void check(const Graph& expression, std::span<const ArrayNode* const> operands,
                      array_or_double initial);

    /// @copydoc Node::commit()
    void commit(State& state) const override;

    /// @copydoc Array::diff()
    std::span<const Update> diff(const State& state) const override;

    /// Access the underlying shared_ptr holding the Graph.
    /// Modifying the Graph leads to undefined behavior.
    std::shared_ptr<Graph>& expression_ptr() { return expression_ptr_; }

    /// @copydoc Node::initialize_state()
    void initialize_state(State& state) const override;

    /// @copydoc Array::integral()
    bool integral() const override;

    /// @copydoc Array::max()
    double max() const override;

    /// @copydoc Array::min()
    double min() const override;

    /// @copydoc Node::propagate()
    void propagate(State& state) const override;

    /// @copydoc Node::revert()
    void revert(State& state) const override;

    std::span<const ssize_t> shape(const State& state) const override;

    ssize_t size(const State& state) const override;
    ssize_t size_diff(const State& state) const override;
    SizeInfo sizeinfo() const override;

    const array_or_double initial;

 private:
    double evaluate_expression(State& register_) const;

    double get_initial_value(const State& state) const;

    std::span<const InputNode* const> operand_inputs() const;
    const InputNode* const accumulate_input() const;

    std::shared_ptr<Graph> expression_ptr_;
    const std::vector<ArrayNode*> operands_;

    const SizeInfo sizeinfo_;
};

}  // namespace dwave::optimization

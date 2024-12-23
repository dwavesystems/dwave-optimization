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

#include <vector>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/graph.hpp"

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
    // Runtime constructor that can be used from Cython/Python
    NaryReduceNode(Graph&& expression, const std::vector<ArrayNode*>& operands, double initial);

    double const* buff(const State& state) const override;
    void commit(State& state) const override;
    std::span<const Update> diff(const State& state) const override;
    void initialize_state(State& state) const override;
    bool integral() const override;
    double max() const override;
    double min() const override;
    void propagate(State& state) const override;
    void revert(State& state) const override;
    std::span<const ssize_t> shape(const State& state) const override;
    ssize_t size(const State& state) const override;
    ssize_t size_diff(const State& state) const override;
    SizeInfo sizeinfo() const override;

    void swap_expression(Graph&& other) { std::swap(expression_, other); };

    const double initial;

 private:
    double evaluate_expression(State& register_) const;

    std::span<const InputNode* const> operand_inputs() const;
    const InputNode* const reduction_input() const;

    Graph expression_;
    const std::vector<ArrayNode*> operands_;
    const ArrayNode* output_;
};

Graph validate_expression(Graph&& expression);

}  // namespace dwave::optimization

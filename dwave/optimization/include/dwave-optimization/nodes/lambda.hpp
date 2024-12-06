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

class NaryReduceNode : public ArrayOutputMixin<ArrayNode> {
 public:
    // Runtime constructor that can be used from Cython/Python
    NaryReduceNode(Graph&& expression, const std::vector<InputNode*>& inputs,
                   const ArrayNode* output, const std::vector<double>& initial_values,
                   const std::vector<ArrayNode*>& operands);

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

 private:
    double evaluate_expression(State& register_) const;

    const Graph expression_;
    const std::vector<InputNode*> inputs_;
    const ArrayNode* output_;
    const std::vector<ArrayNode*> operands_;
    const std::vector<double> initial_values_;
};

}  // namespace dwave::optimization

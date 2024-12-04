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

#include <vector>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/graph.hpp"

namespace dwave::optimization {

// InputNode acts like a placeholder or store of data very similar to ConstantNode,
// with the key different being that its contents *may* change in between propagations.
// However, it is not a decision variable--instead its use cases are acting as an "input"
// for "models as functions", or for placeholders in large models where (otherwise constant)
// data changes infrequently (e.g. a scheduling problem with a preference matrix).
//
// Currently there is no "default" way to initialize the state, so its must be initialized
// explicitly with some data.
class InputNode : public ArrayOutputMixin<ArrayNode> {
 public:
    explicit InputNode(std::span<const ssize_t> shape, double min, double max, bool integral)
            : ArrayOutputMixin(shape), min_(min), max_(max), integral_(integral) {};

    explicit InputNode(std::initializer_list<ssize_t> shape, double min, double max, bool integral)
            : ArrayOutputMixin(shape), min_(min), max_(max), integral_(integral) {};

    explicit InputNode()
            : InputNode({}, -std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::infinity(), false) {};

    bool integral() const override { return integral_; };

    double max() const override { return max_; };
    double min() const override { return min_; };

    void initialize_state(State& state) const override {
        throw std::logic_error(
                "InputNode must have state explicity initialized (with `initialize_state(state, "
                "data)`)");
    }

    void initialize_state(State& state, std::span<const double> data) const;

    double const* buff(const State&) const override;

    std::span<const Update> diff(const State& state) const noexcept override;

    void propagate(State& state) const noexcept override {};
    void commit(State& state) const noexcept override;
    void revert(State& state) const noexcept override;

    void assign(State& state, const std::vector<double>& new_values) const;
    void assign(State& state, std::span<const double> new_values) const;

 private:
    double min_, max_;
    bool integral_;
};

class NaryReduceNode : public ArrayOutputMixin<ArrayNode> {
 public:
    // Runtime constructor that can be used from Cython/Python
    NaryReduceNode(Graph&& expression, const std::vector<InputNode*>& inputs,
                   const ArrayNode* output, const std::vector<double>& initial_values,
                   const std::vector<ArrayNode*>& operands);

    // Array overloads
    double const* buff(const State& state) const override;
    std::span<const Update> diff(const State& state) const override;
    ssize_t size(const State& state) const override;
    std::span<const ssize_t> shape(const State& state) const override;
    ssize_t size_diff(const State& state) const override;
    SizeInfo sizeinfo() const override;

    // Information about the values are all inherited from the array
    bool integral() const override;
    double min() const override;
    double max() const override;

    // Node overloads
    void commit(State& state) const override;
    void initialize_state(State& state) const override;
    void propagate(State& state) const override;
    void revert(State& state) const override;

 private:
    double evaluate_expression(State& register_) const;

    const Graph expression_;
    const std::vector<InputNode*> inputs_;
    const ArrayNode* output_;
    const std::vector<ArrayNode*> operands_;
    const std::vector<double> initial_values_;
};

}  // namespace dwave::optimization

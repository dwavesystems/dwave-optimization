// Copyright 2024 D-Wave Systems Inc.
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

#include <iostream>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/graph.hpp"

namespace dwave::optimization {

class ArrayValidationNode : public Node {
 public:
    explicit ArrayValidationNode(ArrayNode* node_ptr);

    // Node overloads
    void commit(State& state) const override;
    void initialize_state(State& state) const override;
    void propagate(State& state) const override;
    void revert(State& state) const override;

 private:
    const ArrayNode* array_ptr;

#ifdef NDEBUG
    static constexpr bool do_logging = false;
#else
    static constexpr bool do_logging = true;
#endif
};

class DynamicArrayTestingNode : public ArrayOutputMixin<ArrayNode>, public DecisionNode {
 public:
    DynamicArrayTestingNode(std::initializer_list<ssize_t> shape);

    // Overloads needed by the Array ABC **************************************

    std::span<const Update> diff(const State& state) const override;

    double const* buff(const State&) const noexcept override;

    using ArrayOutputMixin::size;  // for size()
    ssize_t size(const State& state) const override;

    using ArrayOutputMixin::shape;  // for shape()
    std::span<const ssize_t> shape(const State& state) const override;

    ssize_t size_diff(const State& state) const override;

    void initialize_state(State& state) const override;
    void initialize_state(State& state, std::initializer_list<double> values) const;
    void initialize_state(State& state, std::span<const double> values) const;

    // Overloads required by the Node ABC *************************************

    void commit(State&) const override;
    void revert(State&) const override;
    void update(State&, int) const override;

    // Overloads required by the Decision ABC
    void default_move(State& state, RngAdaptor& rng) const override;

    // State mutation methods *************************************************

    void grow(State& state, std::span<const double> values) const;
    void grow(State& state, std::initializer_list<double> values) const;
    void set(State& state, ssize_t index, double value) const;
    void shrink(State& state) const;

    void random_move(State& state, RngAdaptor& rng) const;
    void random_moves(State& state, RngAdaptor& rng, size_t max_changes) const;

 private:
    const std::span<const ssize_t> shape_;
};

}  // namespace dwave::optimization

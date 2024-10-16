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
#include <vector>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/state.hpp"

namespace dwave::optimization {

class ReshapeNode : public ArrayOutputMixin<ArrayNode> {
 public:
    ReshapeNode(ArrayNode* node_ptr, std::span<const ssize_t> shape);
    ReshapeNode(ArrayNode* array_ptr, std::vector<ssize_t>&& shape);

    double const* buff(const State& state) const override;
    void commit(State& state) const override;
    std::span<const Update> diff(const State& state) const override;
    void revert(State& state) const override;

 private:
    // we could dynamically cast each time, but it's easier to just keep separate
    // pointer to the "array" part of the predecessor
    const Array* array_ptr_;
};


class SizeNode : public ScalarOutputMixin<ArrayNode> {
 public:
    explicit SizeNode(ArrayNode* node_ptr);

    double const* buff(const State& state) const override;

    void commit(State& state) const override;

    std::span<const Update> diff(const State&) const override;

    void initialize_state(State& state) const override;

    // SizeNode's value is always a non-negative integer.
    bool integral() const override { return true; }

    double max() const override;

    double min() const override;

    void propagate(State& state) const override;

    void revert(State& state) const override;

 private:
    // we could dynamically cast each time, but it's easier to just keep separate
    // pointer to the "array" part of the predecessor
    const Array* array_ptr_;
};

}  // namespace dwave::optimization

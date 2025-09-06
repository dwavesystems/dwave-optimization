// Copyright 2025 D-Wave
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

#include <cassert>
#include <concepts>
#include <functional>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/functional.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/state.hpp"

namespace dwave::optimization {

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

    void add_node(ArrayNode* node_ptr, bool recompute_statistics = true);

    double const* buff(const State& state) const override;
    std::span<const Update> diff(const State& state) const override;

    /// @copydoc Array::integral()
    bool integral() const override;

    /// @copydoc Array::min()
    double min() const override;

    /// @copydoc Array::max()
    double max() const override;

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

    // Note that this is not const as it may change as we add nodes
    ValuesInfo values_info_ = ValuesInfo(0, 0, true);
};

using NaryAddNode = NaryOpNode<std::plus<double>>;
using NaryMaximumNode = NaryOpNode<functional::max<double>>;
using NaryMinimumNode = NaryOpNode<functional::min<double>>;
using NaryMultiplyNode = NaryOpNode<std::multiplies<double>>;

}  // namespace dwave::optimization

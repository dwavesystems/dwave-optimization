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
#include <functional>
#include <optional>
#include <span>
#include <utility>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/functional.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/state.hpp"

namespace dwave::optimization {

template <class UnaryOp>
class UnaryOpNode : public ArrayOutputMixin<ArrayNode> {
 public:
    explicit UnaryOpNode(ArrayNode* node_ptr);

    double const* buff(const State& state) const override;
    std::span<const Update> diff(const State& state) const override;

    /// @copydoc Array::integral()
    bool integral() const override;

    /// @copydoc Array::min()
    double min() const override;

    /// @copydoc Array::max()
    double max() const override;

    using ArrayOutputMixin::shape;
    std::span<const ssize_t> shape(const State& state) const override;
    using ArrayOutputMixin::size;
    ssize_t size(const State& state) const override;
    ssize_t size_diff(const State& state) const override;
    SizeInfo sizeinfo() const override;

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
    UnaryOp op;

    // There are redundant, because we could dynamic_cast each time from
    // predecessors(), but this is more performant
    Array* const array_ptr_;

    const ValuesInfo values_info_;
    const SizeInfo sizeinfo_;
};

using AbsoluteNode = UnaryOpNode<functional::abs<double>>;
using CosNode = UnaryOpNode<functional::cos<double>>;
using ExpitNode = UnaryOpNode<functional::expit<double>>;
using ExpNode = UnaryOpNode<functional::exp<double>>;
using LogNode = UnaryOpNode<functional::log<double>>;
using LogicalNode = UnaryOpNode<functional::logical<double>>;
using NegativeNode = UnaryOpNode<std::negate<double>>;
using NotNode = UnaryOpNode<std::logical_not<double>>;
using RintNode = UnaryOpNode<functional::rint<double>>;
using SinNode = UnaryOpNode<functional::sin<double>>;
using SquareNode = UnaryOpNode<functional::square<double>>;
using SquareRootNode = UnaryOpNode<functional::square_root<double>>;

}  // namespace dwave::optimization

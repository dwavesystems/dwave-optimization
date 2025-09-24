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

#include <array>
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

template <class BinaryOp>
class BinaryOpNode : public ArrayOutputMixin<ArrayNode> {
 public:
    // We need at least two nodes, and they must be the same shape
    BinaryOpNode(ArrayNode* a_ptr, ArrayNode* b_ptr);

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

    // There are redundant, because we could dynamic_cast each time from
    // predecessors(), but this is more performant
    std::array<Array* const, 2> operands_;

    const ValuesInfo values_info_;
    const SizeInfo sizeinfo_;
};

// We follow NumPy naming convention rather than C++ to distinguish between
// binary operations and reduce operations.
// https://numpy.org/doc/stable/reference/routines.math.html
using AddNode = BinaryOpNode<std::plus<double>>;
using AndNode = BinaryOpNode<std::logical_and<double>>;
using DivideNode = BinaryOpNode<std::divides<double>>;
using EqualNode = BinaryOpNode<std::equal_to<double>>;
using LessEqualNode = BinaryOpNode<std::less_equal<double>>;
using MaximumNode = BinaryOpNode<functional::max<double>>;
using MinimumNode = BinaryOpNode<functional::min<double>>;
using ModulusNode = BinaryOpNode<functional::modulus<double>>;
using MultiplyNode = BinaryOpNode<std::multiplies<double>>;
using OrNode = BinaryOpNode<std::logical_or<double>>;
using SafeDivideNode = BinaryOpNode<functional::safe_divides<double>>;
using SubtractNode = BinaryOpNode<std::minus<double>>;
using XorNode = BinaryOpNode<functional::logical_xor<double>>;

}  // namespace dwave::optimization

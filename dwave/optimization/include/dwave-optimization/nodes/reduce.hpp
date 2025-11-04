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
#include <initializer_list>
#include <optional>
#include <span>
#include <vector>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/functional.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/state.hpp"

namespace dwave::optimization {

template <class BinaryOp>
class ReduceNode : public ArrayOutputMixin<ArrayNode> {
 public:
    ReduceNode(ArrayNode* array_ptr);

    ReduceNode(ArrayNode* array_ptr, std::initializer_list<ssize_t> axes,
               std::optional<double> initial = std::nullopt);
    ReduceNode(ArrayNode* array_ptr, std::span<const ssize_t> axes,
               std::optional<double> initial = std::nullopt);

    /// The axes over which the reduction is performed.
    std::span<const ssize_t> axes() const { return axes_; }

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

    /// @copydoc Array::max()
    double max() const override;

    /// @copydoc Array::min()
    double min() const override;

    // The predecessor of the reduction, as an Array*.
    std::span<Array* const> operands() {
        assert(predecessors().size() == 1);
        return std::span<Array* const, 1>(&array_ptr_, 1);
    }
    std::span<const Array* const> operands() const {
        assert(predecessors().size() == 1);
        return std::span<const Array* const, 1>(&array_ptr_, 1);
    }

    /// @copydoc Node::propagate()
    void propagate(State& state) const override;

    /// @copydoc Node::revert()
    void revert(State& state) const override;

    /// @copydoc Array::shape()
    using ArrayOutputMixin::shape;
    std::span<const ssize_t> shape(const State& state) const override;

    /// @copydoc Array::size()
    using ArrayOutputMixin::size;
    ssize_t size(const State& state) const override;

    /// @copydoc Array::size_diff()
    ssize_t size_diff(const State& state) const override;

    /// The initial value if one was provided.
    /// Otherwise uses the first element in the reduction.
    const std::optional<double> initial;

 private:
    // Perform a reduction over the reduction space associated with the given
    // index. The return type is determined by the reduction_type, see
    // functional_.hpp
    auto reduce_(const State& state, ssize_t index) const;

    BinaryOp op;

    Array* const array_ptr_;

    // The axes we're reducing in a sorted, unique vector.
    // An empty vector means we're reducing over everything
    std::vector<ssize_t> axes_;

    // The minimum and maximum (inclusive) value that might be returned as well
    // as whether we're integral or not.
    const ValuesInfo values_info_;

    // During propagation, we need to take a predecessor's `update` and apply
    // it to the correct `index` in the ReduceNode buffer. This method
    // determines the correct buffer `index` given an `update.index`.
    ssize_t convert_predecessor_index_(ssize_t index) const;
};

using AllNode = ReduceNode<std::logical_and<double>>;
using AnyNode = ReduceNode<std::logical_or<double>>;
using MaxNode = ReduceNode<functional::max<double>>;
using MinNode = ReduceNode<functional::min<double>>;
using ProdNode = ReduceNode<std::multiplies<double>>;
using SumNode = ReduceNode<std::plus<double>>;

}  // namespace dwave::optimization

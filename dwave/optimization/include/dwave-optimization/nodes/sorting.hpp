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

#include <span>
#include <vector>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/state.hpp"

namespace dwave::optimization {

class ArgSortNode : public ArrayOutputMixin<ArrayNode> {
 public:
    ArgSortNode(ArrayNode* arr_ptr);

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

    /// @copydoc Array::min()
    double min() const override;

    /// @copydoc Array::max()
    double max() const override;

    /// @copydoc Node::propagate()
    void propagate(State& state) const override;

    /// @copydoc Node::revert()
    void revert(State& state) const override;

    using Array::shape;

    /// @copydoc Array::shape()
    std::span<const ssize_t> shape(const State& state) const override;

    using Array::size;

    /// @copydoc Array::size()
    ssize_t size(const State& state) const override;

    /// @copydoc Array::size_diff()
    ssize_t size_diff(const State& state) const override;

    /// @copydoc Array::sizeinfo()
    SizeInfo sizeinfo() const override;

 private:
    // these are redundant, but convenient
    const Array* arr_ptr_;

    const std::pair<double, double> minmax_;
    const SizeInfo sizeinfo_;
};

}  // namespace dwave::optimization

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

#include <span>
#include <utility>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/graph.hpp"

namespace dwave::optimization {

/// Choose elements from x or y depending on condition.
///
/// `condition` must be either a scalar array or the same shape as `x` and `y`.
/// `x` and `y` must have the same shape, including dynamic.
/// dynamically sized `condition`s are not allowed.
class WhereNode : public ArrayOutputMixin<ArrayNode> {
 public:
    WhereNode(ArrayNode* condition_ptr, ArrayNode* x_ptr, ArrayNode* y_ptr);

    double const* buff(const State& state) const override;
    void commit(State& state) const override;
    std::span<const Update> diff(const State& state) const override;
    void initialize_state(State& state) const override;
    bool integral() const override;

    /// @copydoc Array::minmax()
    std::pair<double, double> minmax(
            optional_cache_type<std::pair<double, double>> cache = std::nullopt) const override;

    void propagate(State& state) const override;
    void revert(State& state) const override;
    using Array::shape;
    std::span<const ssize_t> shape(const State& state) const override;
    using Array::size;
    ssize_t size(const State& state) const override;
    ssize_t size_diff(const State& state) const override;

 private:
    // these are redundant, but convenient
    const Array* condition_ptr_;
    const Array* x_ptr_;
    const Array* y_ptr_;
};

}  // namespace dwave::optimization

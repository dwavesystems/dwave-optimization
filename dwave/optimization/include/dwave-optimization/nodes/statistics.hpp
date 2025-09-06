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

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/state.hpp"

namespace dwave::optimization {

class MeanNode : public ScalarOutputMixin<ArrayNode, true> {
 public:
    MeanNode(ArrayNode* arr_ptr);

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

 private:
    // these are redundant, but convenient
    const Array* arr_ptr_;

    const std::pair<double, double> minmax_;
};

}  // namespace dwave::optimization

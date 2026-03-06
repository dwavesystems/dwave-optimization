// Copyright 2026 D-Wave Systems Inc.
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

#include "dwave-optimization/cp/core/advisor.hpp"

namespace dwave::optimization::cp {
Advisor::Advisor(Propagator* p, ssize_t p_input, std::unique_ptr<IndexTransform> index_transform)
        : p_(p), p_input_(p_input), index_transform_(std::move(index_transform)) {}

void Advisor::notify(CPPropagatorsState& p_state, ssize_t i) const {
    std::vector<ssize_t> out;
    index_transform_->affected(i, out);
    for (ssize_t j : out) {
        p_->mark_index(p_state, j);
    }
}

Propagator* Advisor::get_propagator() const { return p_; }
}  // namespace dwave::optimization::cp

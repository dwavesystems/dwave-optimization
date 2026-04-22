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
#pragma once

#include "dwave-optimization/cp/core/cpvar.hpp"
#include "dwave-optimization/cp/core/propagator.hpp"
#include "dwave-optimization/nodes/indexing.hpp"

namespace dwave::optimization::cp {

struct BasicIndexingForwardTransform : IndexTransform {
    BasicIndexingForwardTransform(const ArrayNode* array_ptr, const BasicIndexingNode* bi_ptr);

    void affected(ssize_t i, std::vector<ssize_t>& out) override;

    const ArrayNode* array_ptr_;
    const BasicIndexingNode* bi_ptr_;
    std::vector<BasicIndexingNode::slice_or_int> slices;
};

class BasicIndexingPropagator : public Propagator {
 public:
    BasicIndexingPropagator(ssize_t index, CPVar* array, CPVar* basic_indexing);

    void initialize_state(CPState& state) const override;
    CPStatus propagate(CPPropagatorsState& p_state, CPVarsState& v_state) const override;

 private:
    CPVar* array_;
    CPVar* basic_indexing_;
};

}  // namespace dwave::optimization::cp

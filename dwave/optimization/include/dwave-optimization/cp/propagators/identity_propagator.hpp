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

namespace dwave::optimization::cp {

/// This is an identity propagator, it does nothing, leaves the domains untouched but it is used for
/// testing the "delta" propagation, where we mark indices of the constraints array to propagate.
class ElementWiseIdentityPropagator : public Propagator {
 public:
    // constructor
    ElementWiseIdentityPropagator(ssize_t index, CPVar* var);

    void initialize_state(CPState& state) const override;
    CPStatus propagate(CPPropagatorsState& p_state, CPVarsState& v_state) const override;

 private:
    CPVar* var_;
};

/// This is an identity propagator, it does nothing, leaves the domains untouched but it is used for
/// testing the "delta" propagation, where we mark indices of the constraints array to propagate.
class ReductionIdentityPropagator : public Propagator {
 public:
    // constructor
    ReductionIdentityPropagator(ssize_t index, CPVar* var);

    void initialize_state(CPState& state) const override;
    CPStatus propagate(CPPropagatorsState& p_state, CPVarsState& v_state) const override;

 private:
    CPVar* var_;
};
}  // namespace dwave::optimization::cp

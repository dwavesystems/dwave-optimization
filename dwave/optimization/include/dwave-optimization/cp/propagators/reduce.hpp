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

template <class BinaryOp>
class ReducePropagator : public Propagator {
 public:
    ReducePropagator(ssize_t index, CPVar* in, CPVar* out);
    void initialize_state(CPState& state) const override;
    CPStatus propagate(CPPropagatorsState& p_state, CPVarsState& v_state) const override;

 private:
    // The variables entering the binary op
    CPVar *in_, *out_;
};

using SumPropagator = ReducePropagator<std::plus<double>>;

/// Dynamic Reduce Propagator.
/// Uses the binary op to filter the domains of the input and output variable.
/// This propagator assumes full reduction of the array, it does not reduce over some axes.
template <class BinaryOp>
class DynamicReducePropagator : public Propagator {
 public:
    DynamicReducePropagator(ssize_t index, CPVar* in, CPVar* out);
    void initialize_state(CPState& state) const override;
    CPStatus propagate(CPPropagatorsState& p_state, CPVarsState& v_state) const override;

 private:
    // The variables entering the binary op
    CPVar *in_, *out_;
};

using DynamicSumPropagator = DynamicReducePropagator<std::plus<double>>;

}  // namespace dwave::optimization::cp

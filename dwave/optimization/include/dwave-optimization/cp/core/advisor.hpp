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
#include "dwave-optimization/cp/core/index_transform.hpp"
#include "dwave-optimization/cp/core/propagator.hpp"
namespace dwave::optimization::cp {

/// Utility class to help schedule propagators when variable change their domains
/// In our settings variables and constraints are arrays, so we need a way to:
/// - trigger propagator indices when variable indices' domains change
/// - map variable index to propagator
class Advisor {
 public:
    Advisor(Propagator* p, ssize_t p_input, std::unique_ptr<IndexTransform> index_transform);

    void notify(CPPropagatorsState& p_state, ssize_t i) const;

    Propagator* get_propagator() const;

 private:
    // The propagator the advisor is watching
    Propagator* p_;

    // Which input
    const ssize_t p_input_;

    // Maps of the observed variable indices to the observed propagator indices
    std::unique_ptr<IndexTransform> index_transform_;
};

}  // namespace dwave::optimization::cp

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

#include "dwave-optimization/cp/core/engine.hpp"

namespace dwave::optimization::cp {

CPStatus CPEngine::fix_point(CPState& state) const {
    CPStatus status = CPStatus::OK;

    CPPropagatorsState& p_state = state.propagator_state_;

    while (state.propagation_queue_.size() > 0) {
        Propagator* p = state.propagation_queue_.front();
        status = this->propagate(state, p);

        if (status == CPStatus::Inconsistency) {
            while (state.propagation_queue_.size() > 0) {
                state.propagation_queue_.front()->set_scheduled(p_state, false);
                state.propagation_queue_.pop_front();
            }

            // inconsistency detected, return!
            return status;
        }
    }

    return status;
}

CPStatus CPEngine::propagate(CPState& state, Propagator* p) const {
    auto& p_state = state.propagator_state_;
    auto& v_state = state.var_state_;
    return p->propagate(p_state, v_state);
}

StateManager* CPEngine::get_state_manager(const CPState& state) const {
    return state.get_state_manager();
}

}  // namespace dwave::optimization::cp

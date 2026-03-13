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

#include "dwave-optimization/cp/propagators/identity_propagator.hpp"

#include <deque>
namespace dwave::optimization::cp {

ElementWiseIdentityPropagator::ElementWiseIdentityPropagator(ssize_t index, CPVar* var)
        : Propagator(index) {
    // TODO: not supporting dynamic variables for now
    assert(var->min_size() == var->max_size());
    var_ = var;
}

void ElementWiseIdentityPropagator::initialize_state(CPState& state) const {
    CPPropagatorsState& p_state = state.get_propagators_state();
    assert(propagator_index_ >= 0);
    assert(propagator_index_ < static_cast<ssize_t>(p_state.size()));
    p_state[propagator_index_] =
            std::make_unique<PropagatorData>(state.get_state_manager(), var_->max_size());
}

CPStatus ElementWiseIdentityPropagator::propagate(CPPropagatorsState& p_state,
                                                  CPVarsState& v_state) const {
    auto data = data_ptr<PropagatorData>(p_state);
    assert(data->num_indices_to_process() > 0);
    std::deque<ssize_t> indices_to_process = data->indices_to_process();
    while (data->num_indices_to_process() > 0) {
        ssize_t i = indices_to_process.front();
        indices_to_process.pop_front();
        data->set_scheduled(false, i);
        /// Note: this is where other propagator would filter domains..
    }
    return CPStatus::OK;
}

ReductionIdentityPropagator::ReductionIdentityPropagator(ssize_t index, CPVar* var)
        : Propagator(index) {
    // TODO: not supporting dynamic variables for now
    assert(var->min_size() == var->max_size());
    var_ = var;
}

void ReductionIdentityPropagator::initialize_state(CPState& state) const {
    CPPropagatorsState& p_state = state.get_propagators_state();
    assert(propagator_index_ >= 0);
    assert(propagator_index_ < static_cast<ssize_t>(p_state.size()));
    p_state[propagator_index_] = std::make_unique<PropagatorData>(state.get_state_manager(), 0);
}

CPStatus ReductionIdentityPropagator::propagate(CPPropagatorsState& p_state,
                                                CPVarsState& v_state) const {
    auto data = data_ptr<PropagatorData>(p_state);
    std::deque<ssize_t> indices_to_process = data->indices_to_process();
    assert(indices_to_process.size() == 1);
    while (data->num_indices_to_process() > 0) {
        ssize_t i = indices_to_process.front();
        indices_to_process.pop_front();
        data->set_scheduled(false, i);
        /// Note: this is where other propagator would filter domains..
    }
    return CPStatus::OK;
}

}  // namespace dwave::optimization::cp

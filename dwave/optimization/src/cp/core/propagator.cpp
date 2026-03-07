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

#include "dwave-optimization/cp/core/propagator.hpp"
namespace dwave::optimization::cp {

void PropagatorData::mark_index(ssize_t index) {
    assert(index < static_cast<ssize_t>(indices.is_scheduled.size()));
    if (indices.is_scheduled[index]) return;
    indices.is_scheduled[index] = true;
    indices.to_process.push_back(index);
}

bool Propagator::scheduled(const CPPropagatorsState& state) const {
    assert(propagator_index_ >= 0);
    assert(propagator_index_ < static_cast<ssize_t>(state.size()));
    return state[propagator_index_]->scheduled();
}

void Propagator::set_scheduled(CPPropagatorsState& state, bool scheduled) const {
    assert(propagator_index_ >= 0);
    assert(propagator_index_ < static_cast<ssize_t>(state.size()));
    state[propagator_index_]->set_scheduled(scheduled);
}

bool Propagator::active(const CPPropagatorsState& state) const {
    assert(propagator_index_ >= 0);
    assert(propagator_index_ < static_cast<ssize_t>(state.size()));
    return state[propagator_index_]->active();
}

void Propagator::set_active(CPPropagatorsState& state, bool active) const {
    assert(propagator_index_ >= 0);
    assert(propagator_index_ < static_cast<ssize_t>(state.size()));
    state[propagator_index_]->set_active(active);
}

void Propagator::mark_index(CPPropagatorsState& state, ssize_t index) const {
    assert(propagator_index_ >= 0);
    assert(index >= 0);
    state[propagator_index_]->mark_index(index);
}
}  // namespace dwave::optimization::cp

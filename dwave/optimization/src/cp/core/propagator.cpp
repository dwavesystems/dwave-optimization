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

template <std::derived_from<PropagatorData> PData>
PData* Propagator::data_ptr(CPPropagatorsState& state) const {
    assert(propagator_index_ >= 0);
    assert(propagator_index_ < static_cast<ssize_t>(state.size()));
    return static_cast<PData*>(state[propagator_index_].get());
}

template <std::derived_from<PropagatorData> PData>
const PData* Propagator::data_ptr(const CPPropagatorsState& state) const {
    assert(propagator_index_ >= 0);
    assert(propagator_index_ < static_cast<ssize_t>(state.size()));
    return static_cast<const PData*>(state[propagator_index_].get());
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
}  // namespace dwave::optimization::cp

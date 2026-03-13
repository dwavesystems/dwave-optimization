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

// ------- PropagatorData --------

void PropagatorData::mark_index(ssize_t index) {
    assert(index < static_cast<ssize_t>(is_scheduled_.size()));
    if (is_scheduled_[index] or not active_[index]) return;
    is_scheduled_[index] = true;
    to_process_.push_back(index);
}

bool PropagatorData::scheduled() const { return scheduled_; }

bool PropagatorData::scheduled(ssize_t i) const {
    assert(i < static_cast<ssize_t>(is_scheduled_.size()));
    return is_scheduled_[i];
}

void PropagatorData::set_scheduled(bool scheduled) { scheduled_ = scheduled; }

void PropagatorData::set_scheduled(bool scheduled, ssize_t i) {
    assert(i < static_cast<ssize_t>(is_scheduled_.size()));
    is_scheduled_[i] = scheduled;
}

bool PropagatorData::active(ssize_t i) const {
    assert(i < static_cast<ssize_t>(active_.size()));
    return active_[i]->get_value();
}

void PropagatorData::set_active(bool active, ssize_t i) {
    assert(i < static_cast<ssize_t>(active_.size()));
    active_[i]->set_value(active);
}

// ------- Propagator --------

bool Propagator::scheduled(const CPPropagatorsState& state) const {
    assert(propagator_index_ >= 0);
    assert(propagator_index_ < static_cast<ssize_t>(state.size()));
    return state[propagator_index_]->scheduled();
}

bool Propagator::scheduled(const CPPropagatorsState& state, int i) const {
    assert(propagator_index_ >= 0);
    assert(propagator_index_ < static_cast<ssize_t>(state.size()));
    return state[propagator_index_]->scheduled(i);
}

void Propagator::set_scheduled(CPPropagatorsState& state, bool scheduled) const {
    assert(propagator_index_ >= 0);
    assert(propagator_index_ < static_cast<ssize_t>(state.size()));
    state[propagator_index_]->set_scheduled(scheduled);
}

bool Propagator::active(const CPPropagatorsState& state, int i) const {
    assert(propagator_index_ >= 0);
    assert(propagator_index_ < static_cast<ssize_t>(state.size()));
    return state[propagator_index_]->active(i);
}

void Propagator::set_active(CPPropagatorsState& state, bool active, int i) const {
    assert(propagator_index_ >= 0);
    assert(propagator_index_ < static_cast<ssize_t>(state.size()));
    state[propagator_index_]->set_active(active, i);
}

void Propagator::mark_index(CPPropagatorsState& state, ssize_t index) const {
    assert(propagator_index_ >= 0);
    assert(index >= 0);
    state[propagator_index_]->mark_index(index);
}

ssize_t Propagator::num_indices_to_process(const CPPropagatorsState& state) const {
    assert(propagator_index_ >= 0);
    return state[propagator_index_]->num_indices_to_process();
}
}  // namespace dwave::optimization::cp

#include "dwave-optimization/cp/core/propagator.hpp"

namespace dwave::optimization::cp {

template <std::derived_from<PropagatorData> PData>
PData* Propagator::data_ptr(CPPropagatorsState& state) const {
    assert(propagator_index_ >= 0);
    assert(propagator_index_ < state.size());
    return static_cast<PData*>(state[propagator_index_].get());
}

template <std::derived_from<PropagatorData> PData>
const PData* Propagator::data_ptr(const CPPropagatorsState& state) const {
    assert(propagator_index_ >= 0);
    assert(propagator_index_ < state.size());
    return static_cast<const PData*>(state[propagator_index_].get());
}

bool Propagator::scheduled(const CPPropagatorsState& state) const {
    assert(propagator_index_ >= 0);
    assert(propagator_index_ < state.size());
    return state[propagator_index_]->scheduled();
}

void Propagator::set_scheduled(CPPropagatorsState& state, bool scheduled) const {
    assert(propagator_index_ >= 0);
    assert(propagator_index_ < state.size());
    state[propagator_index_]->set_scheduled(scheduled);
}

bool Propagator::active(const CPPropagatorsState& state) const {
    assert(propagator_index_ >= 0);
    assert(propagator_index_ < state.size());
    return state[propagator_index_]->active();
}

bool Propagator::set_active(CPPropagatorsState& state, bool active) const {
    assert(propagator_index_ >= 0);
    assert(propagator_index_ < state.size());
    state[propagator_index_]->set_active(active);
}
}  // namespace dwave::optimization::cp

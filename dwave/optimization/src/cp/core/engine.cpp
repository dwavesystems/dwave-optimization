#include "dwave-optimization/cp/core/engine.hpp"

namespace dwave::optimization::cp {

CPStatus CPEngine::fix_point(CPState& state) const {
    CPStatus status = CPStatus::OK;

    CPPropagatorsState& p_state = state.propagator_state_;
    CPVarsState& v_state = state.var_state_;

    while (state.propagation_queue_.size() > 0) {
        Propagator* p = state.propagation_queue_.front();
        status = this->propagate(state, p);

        if (status == CPStatus::Inconsistency) {
            while (state.propagation_queue_.size() > 0) {
                Propagator* p = state.propagation_queue_.front();
                // state.propagation_queue_.front()->set_scheduled(state, false);
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
    p->propagate(p_state, v_state);
}

StateManager* CPEngine::get_state_manager(const CPState& state) const {
    return state.get_state_manager();
}

}  // namespace dwave::optimization::cp

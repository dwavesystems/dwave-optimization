#include "dwave-optimization/cp/core/cpvar.hpp"

#include <cassert>
namespace dwave::optimization::cp {

CPVar::CPVar(const CPModel& model, const dwave::optimization::ArrayNode* node_ptr, int index)
        : model_(model), node_(node_ptr), cp_var_index_(index) {}

// ------ CPVar -------

double CPVar::min(const CPVarsState& state, int index) const {
    const CPVarData* data = data_ptr<CPVarData>(state);
    return data->min(index);
}

double CPVar::max(const CPVarsState& state, int index) const {
    const CPVarData* data = data_ptr<CPVarData>(state);
    return data->max(index);
}

double CPVar::size(const CPVarsState& state, int index) const {
    const CPVarData* data = data_ptr<CPVarData>(state);
    return data->size(index);
}

bool CPVar::is_bound(const CPVarsState& state, int index) const {
    const CPVarData* data = data_ptr<CPVarData>(state);
    return data->is_bound(index);
}

bool CPVar::contains(const CPVarsState& state, double value, int index) const {
    const CPVarData* data = data_ptr<CPVarData>(state);
    return data->contains(value, index);
}

CPStatus CPVar::remove(CPVarsState& state, double value, int index) const {
    CPVarData* data = data_ptr<CPVarData>(state);
    return data->remove(value, index);
}

CPStatus CPVar::remove_above(CPVarsState& state, double value, int index) const {
    CPVarData* data = data_ptr<CPVarData>(state);
    return data->remove_above(value, index);
}

CPStatus CPVar::remove_below(CPVarsState& state, double value, int index) const {
    CPVarData* data = data_ptr<CPVarData>(state);
    return data->remove_below(value, index);
}

CPStatus CPVar::assign(CPVarsState& state, double value, int index) const {
    CPVarData* data = data_ptr<CPVarData>(state);
    return data->remove_all_but(value, index);
}

void CPVar::schedule_all(CPState& state, const std::vector<Propagator*>& propagators) const {
    for (auto p : propagators) {
        state.schedule(p);
    }
}

void CPVar::propagate_on_assignment(Propagator* p) { on_bind.push_back(p); }

void CPVar::propagate_on_bounds_change(Propagator* p) { on_bounds.push_back(p); }

void CPVar::propagate_on_domain_change(Propagator* p) { on_domain.push_back(p); }

void CPVar::initialize_state(CPState& state) const {
    assert(this->cp_var_index_ >= 0);
    assert(state.var_state_.size() > cp_var_index_);

    // TODO: for now we don't handle dynamically sized nodes
    assert(node_->size() > 0);

    state.var_state_[cp_var_index_] = std::make_unique<CPVarData>(
            state.get_state_manager(), node_->size(), node_->min(), node_->max(),
            std::make_unique<Listener>(this, state), node_->integral());
}

}  // namespace dwave::optimization::cp

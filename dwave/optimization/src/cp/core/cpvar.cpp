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

#include "dwave-optimization/cp/core/cpvar.hpp"

#include <cassert>
namespace dwave::optimization::cp {

// ------ CPVar -------

CPVar::CPVar(const CPModel& model, const dwave::optimization::ArrayNode* node_ptr, int index)
        : model_(model), node_(node_ptr), cp_var_index_(index) {}

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

bool CPVar::is_active(const CPVarsState& state, int index) const {
    const CPVarData* data = data_ptr<CPVarData>(state);
    return data->is_active(index);
}

bool CPVar::maybe_active(const CPVarsState& state, int index) const {
    const CPVarData* data = data_ptr<CPVarData>(state);
    return data->maybe_active(index);
}

ssize_t CPVar::min_size(const CPVarsState& state) const {
    const CPVarData* data = data_ptr<CPVarData>(state);
    return data->min_size();
}

ssize_t CPVar::max_size(const CPVarsState& state) const {
    const CPVarData* data = data_ptr<CPVarData>(state);
    return data->max_size();
}

ssize_t CPVar::min_size() const {
    if (node_->size() > 0) {
        // Static array
        return node_->size();
    }
    SizeInfo info = node_->sizeinfo();
    // The minimum size value existence should be guaranteed at construction of the CPVar
    assert(info.min.has_value());
    return info.min.value();
}

ssize_t CPVar::max_size() const {
    if (node_->size() > 0) {
        // Static array
        return node_->size();
    }
    SizeInfo info = node_->sizeinfo();
    // The maximum size value existence should be guaranteed at construction of the CPVar
    assert(info.max.has_value());
    return info.max.value();
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

CPStatus CPVar::set_min_size(CPVarsState& state, int new_min_size) const {
    CPVarData* data = data_ptr<CPVarData>(state);
    return data->update_min_size(new_min_size);
}

CPStatus CPVar::set_max_size(CPVarsState& state, int new_max_size) const {
    CPVarData* data = data_ptr<CPVarData>(state);
    return data->update_max_size(new_max_size);
}

void CPVar::schedule_all(CPState& state, const std::vector<Advisor>& advisors, ssize_t i) const {
    CPPropagatorsState& p_state = state.get_propagators_state();
    for (const Advisor& adv : advisors) {
        adv.notify(p_state, i);
        state.schedule(adv.get_propagator());
    }
}

void CPVar::propagate_on_assignment(Advisor&& adv) { on_bind.push_back(std::move(adv)); }

void CPVar::propagate_on_bounds_change(Advisor&& adv) { on_bounds.push_back(std::move(adv)); }

void CPVar::propagate_on_domain_change(Advisor&& adv) { on_domain.push_back(std::move(adv)); }

void CPVar::initialize_state(CPState& state) const {
    assert(this->cp_var_index_ >= 0);
    assert(static_cast<ssize_t>(state.var_state_.size()) > cp_var_index_);

    // TODO: for now we don't handle dynamically sized nodes

    ssize_t max_size, min_size;

    if (node_->size() > 0) {
        // Static array
        max_size = node_->size();
        min_size = node_->size();
    } else {
        // Dynamic arrays
        SizeInfo sizeinfo = node_->sizeinfo();
        if (not sizeinfo.max.has_value()) {
            throw std::invalid_argument("Array has no max size available");
        }

        if (not sizeinfo.min.has_value()) {
            throw std::invalid_argument("Array has no min size available");
        }

        max_size = sizeinfo.max.value();
        min_size = sizeinfo.min.value();
    }

    state.var_state_[cp_var_index_] = std::make_unique<CPVarData>(
            state.get_state_manager(), min_size, max_size, node_->min(), node_->max(),
            std::make_unique<Listener>(this, state), node_->integral());
}

}  // namespace dwave::optimization::cp

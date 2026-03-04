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

#include "dwave-optimization/cp/state/cpvar_state.hpp"

#include <type_traits>

#include "dwave-optimization/cp/core/interval_array.hpp"

namespace dwave::optimization::cp {

CPVarData::CPVarData(StateManager* sm, ssize_t min_size, ssize_t max_size, double lb, double ub,
                     std::unique_ptr<DomainListener> listener, bool integral) {
    // Set a real interval
    if (integral) {
        domains_ = std::make_unique<IntIntervalArray>(sm, min_size, max_size, lb, ub);
    } else {
        domains_ = std::make_unique<RealIntervalArray>(sm, min_size, max_size, lb, ub);
    }

    listen_ = std::move(listener);
}

size_t CPVarData::num_domains() const { return domains_->num_domains(); }

double CPVarData::min(int index) const { return domains_->min(index); }

double CPVarData::max(int index) const { return domains_->max(index); }

double CPVarData::size(int index) const { return domains_->size(index); }

bool CPVarData::is_bound(int index) const { return domains_->is_bound(index); }

bool CPVarData::contains(double value, int index) const { return domains_->contains(value, index); }

bool CPVarData::is_active(int index) const { return domains_->is_active(index); }

bool CPVarData::maybe_active(int index) const { return domains_->maybe_active(index); }

ssize_t CPVarData::min_size() const { return domains_->min_size(); }

ssize_t CPVarData::max_size() const { return domains_->max_size(); }

CPStatus CPVarData::remove(double value, int index) {
    return domains_->remove(value, index, listen_.get());
}

CPStatus CPVarData::remove_above(double value, int index) {
    return domains_->remove_above(value, index, listen_.get());
}

CPStatus CPVarData::remove_below(double value, int index) {
    return domains_->remove_below(value, index, listen_.get());
}

CPStatus CPVarData::remove_all_but(double value, int index) {
    return domains_->remove_all_but(value, index, listen_.get());
}

CPStatus CPVarData::update_min_size(ssize_t new_min_size) {
    return domains_->update_min_size(new_min_size, listen_.get());
}

CPStatus CPVarData::update_max_size(ssize_t new_max_size) {
    return domains_->update_max_size(new_max_size, listen_.get());
}

}  // namespace dwave::optimization::cp

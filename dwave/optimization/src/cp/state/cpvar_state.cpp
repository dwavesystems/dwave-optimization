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

CPVarData::CPVarData(DomainArrayVariant&& domains, std::unique_ptr<DomainListener> listener)
        : domains_(domains) {
    listen_ = std::move(listener);
}

size_t CPVarData::num_domains() const { return DomainDispatcher::num_domains(domains_); }

double CPVarData::min(int index) const { return DomainDispatcher::min(domains_, index); }

double CPVarData::max(int index) const { return DomainDispatcher::max(domains_, index); }

double CPVarData::size(int index) const { return DomainDispatcher::size(domains_, index); }

bool CPVarData::is_bound(int index) const { return DomainDispatcher::is_bound(domains_, index); }

bool CPVarData::contains(double value, int index) const {
    return DomainDispatcher::contains(domains_, value, index);
}

bool CPVarData::is_active(int index) const { return DomainDispatcher::is_active(domains_, index); }

bool CPVarData::maybe_active(int index) const {
    return DomainDispatcher::maybe_active(domains_, index);
}

ssize_t CPVarData::min_size() const { return DomainDispatcher::min_size(domains_); }

ssize_t CPVarData::max_size() const { return DomainDispatcher::max_size(domains_); }

CPStatus CPVarData::remove(double value, int index) {
    return DomainDispatcher::remove(domains_, value, index, listen_.get());
}

CPStatus CPVarData::remove_above(double value, int index) {
    return DomainDispatcher::remove_above(domains_, value, index, listen_.get());
}

CPStatus CPVarData::remove_below(double value, int index) {
    return DomainDispatcher::remove_below(domains_, value, index, listen_.get());
}

CPStatus CPVarData::remove_all_but(double value, int index) {
    return DomainDispatcher::remove_all_but(domains_, value, index, listen_.get());
}

CPStatus CPVarData::update_min_size(ssize_t new_min_size) {
    return DomainDispatcher::update_min_size(domains_, new_min_size, listen_.get());
}

CPStatus CPVarData::update_max_size(ssize_t new_max_size) {
    return DomainDispatcher::update_max_size(domains_, new_max_size, listen_.get());
}

}  // namespace dwave::optimization::cp

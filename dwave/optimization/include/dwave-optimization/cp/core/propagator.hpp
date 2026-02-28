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

#pragma once

#include <memory>
#include <vector>

#include "dwave-optimization/cp/state/cpvar_state.hpp"
#include "dwave-optimization/cp/state/state.hpp"

namespace dwave::optimization::cp {

// internal structure of propagators
class PropagatorData {
 public:
    virtual ~PropagatorData() = default;
    bool scheduled() const { return scheduled_; }
    void set_scheduled(bool scheduled) { scheduled_ = scheduled; }
    bool active() const { return active_->get_value(); }
    void set_active(bool active) { active_->set_value(active); }

    // indices of the constraint (corresponding to this propagator) to propagate.
    std::vector<ssize_t> indices;

 protected:
    bool scheduled_;
    StateBool* active_;
};

using CPPropagatorsState = std::vector<std::unique_ptr<PropagatorData>>;

class Propagator {
 public:
    virtual ~Propagator() = default;

    Propagator(int index) : propagator_index_(index) {}

    template <std::derived_from<PropagatorData> PData>
    PData* data_ptr(CPPropagatorsState& state) const;

    template <std::derived_from<PropagatorData> PData>
    const PData* data_ptr(const CPPropagatorsState& state) const;

    bool scheduled(const CPPropagatorsState& state) const;

    void set_scheduled(CPPropagatorsState& state, bool scheduled) const;

    bool active(const CPPropagatorsState& state) const;

    bool set_active(CPPropagatorsState& state, bool active) const;

    virtual CPStatus propagate(CPPropagatorsState& p_state, CPVarsState& v_state) = 0;

 private:
    const ssize_t propagator_index_;
};

}  // namespace dwave::optimization::cp

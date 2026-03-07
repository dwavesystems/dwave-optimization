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

#include <deque>
#include <memory>
#include <vector>

#include "dwave-optimization/cp/state/cpvar_state.hpp"
#include "dwave-optimization/cp/state/state.hpp"

namespace dwave::optimization::cp {

   // forward declaration

   class CPState;


// internal structure of propagators
class PropagatorData {
 protected:
    /// Helper class to handle the indices to process
    struct ProcessingIndexHelper {
        // constructor
        ProcessingIndexHelper(ssize_t n) { is_scheduled.resize(n, false); }

        std::deque<ssize_t> to_process;
        std::vector<bool> is_scheduled;
    };

 public:
    virtual ~PropagatorData() = default;

    PropagatorData(StateManager* sm, ssize_t constraint_size) : indices(constraint_size) {
      active_ = sm->make_state_bool(true);
      scheduled_ = false;
    }

    /// Check whether the propagator is already scheduled
    bool scheduled() const { return scheduled_; }

    /// Set the scheduled status of the propagator
    void set_scheduled(bool scheduled) { scheduled_ = scheduled; }

    /// Check whether the propagator is active (not active when the constraint is entailed)
    bool active() const { return active_->get_value(); }
    void set_active(bool active) { active_->set_value(active); }

    // indices of the constraint (corresponding to this propagator) to propagate.
    void mark_index(ssize_t index);

    ProcessingIndexHelper indices;

 protected:
    bool scheduled_;

    // TODO: should this be a vector? Probably yes
    StateBool* active_;
};

using CPPropagatorsState = std::vector<std::unique_ptr<PropagatorData>>;

class Propagator {
 public:
    virtual ~Propagator() = default;

    Propagator(int index) : propagator_index_(index) {}

    template <std::derived_from<PropagatorData> PData>
    PData* data_ptr(CPPropagatorsState& state) const {
        assert(propagator_index_ >= 0);
        assert(propagator_index_ < static_cast<ssize_t>(state.size()));
        return static_cast<PData*>(state[propagator_index_].get());
    }

    template <std::derived_from<PropagatorData> PData>
    const PData* data_ptr(const CPPropagatorsState& state) const {
        assert(propagator_index_ >= 0);
        assert(propagator_index_ < static_cast<ssize_t>(state.size()));
        return static_cast<const PData*>(state[propagator_index_].get());
    }

    bool scheduled(const CPPropagatorsState& state) const;

    void set_scheduled(CPPropagatorsState& state, bool scheduled) const;

    bool active(const CPPropagatorsState& state) const;

    void set_active(CPPropagatorsState& state, bool active) const;

    virtual CPStatus propagate(CPPropagatorsState& p_state, CPVarsState& v_state) const = 0;

    virtual void initialize_state(CPState& state) const = 0;

    void mark_index(CPPropagatorsState& p_state, ssize_t index) const;

 protected:
    // Index of the propagator to access the respective propagator state
    const ssize_t propagator_index_;
};

}  // namespace dwave::optimization::cp

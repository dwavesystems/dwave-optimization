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
class Propagator;

// internal structure of propagators
class PropagatorData {
 public:
    virtual ~PropagatorData() = default;

    PropagatorData(StateManager* sm, ssize_t constraint_size) : n_(constraint_size) {
        scheduled_ = false;
        is_scheduled_.resize(constraint_size, false);
        active_.resize(constraint_size);
        for (ssize_t i = 0; i < constraint_size; ++i) {
            active_[i] = sm->make_state_bool(true);
        }
    }

    /// Check whether the propagator is already scheduled
    bool scheduled() const;
    bool scheduled(ssize_t i) const;

    /// Set the scheduled status of the propagator
    void set_scheduled(bool scheduled);
    void set_scheduled(bool scheduled, ssize_t);

    /// Check whether the propagator is active (not active when the constraint is entailed)
    bool active(ssize_t i) const;
    void set_active(bool active, ssize_t i);

    // indices of the constraint (corresponding to this propagator) to propagate.
    void mark_index(ssize_t index);

    ssize_t num_indices_to_process() const { return to_process_.size(); }
    std::deque<ssize_t>& indices_to_process() { return to_process_; }

 protected:
    // size of the constraint the propagator is associated with
    const ssize_t n_;
    bool scheduled_;
    std::deque<ssize_t> to_process_;
    std::vector<bool> is_scheduled_;
    std::vector<StateBool*> active_;
};

using CPPropagatorsState = std::vector<std::unique_ptr<PropagatorData>>;

/// Base class for propagators. A propagator, or filtering function, implements the inference done
/// on the variables domains from a constraint.
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

    bool scheduled(const CPPropagatorsState& state, int i) const;

    void set_scheduled(CPPropagatorsState& state, bool scheduled) const;

    bool active(const CPPropagatorsState& state, int i) const;

    void set_active(CPPropagatorsState& state, bool active, int i) const;

    /// Implementation of the filtering algorithm
    /// IMPORTANT: The fix-point engine implementation in engine.hpp assumes that the propagate
    /// function is idempotent
    virtual CPStatus propagate(CPPropagatorsState& p_state, CPVarsState& v_state) const = 0;

    virtual void initialize_state(CPState& state) const = 0;

    void mark_index(CPPropagatorsState& p_state, ssize_t index) const;

    ssize_t num_indices_to_process(const CPPropagatorsState& state) const;

 protected:
    // Index of the propagator to access the respective propagator state
    const ssize_t propagator_index_;
};

}  // namespace dwave::optimization::cp

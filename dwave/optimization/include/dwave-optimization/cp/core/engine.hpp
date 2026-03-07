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

#include "dwave-optimization/cp/core/propagator.hpp"
#include "dwave-optimization/cp/core/status.hpp"
#include "dwave-optimization/cp/state/cpvar_state.hpp"
#include "dwave-optimization/cp/state/state_manager.hpp"

namespace dwave::optimization::cp {

// forward declaration
class CPVar;
class CPState;

class CPEngine {
 public:
    CPStatus fix_point(CPState& state) const;
    CPStatus propagate(CPState& state, Propagator* p) const;
    StateManager* get_state_manager(const CPState& state) const;
};

// Class that holds all the data used during the CP search
class CPState {
    friend CPEngine;
    friend CPVar;

 public:
    CPState(std::unique_ptr<StateManager> sm, ssize_t num_variables, ssize_t num_propagators)
            : sm_(std::move(sm)) {
        var_state_.resize(num_variables);
        propagator_state_.resize(num_propagators);
    }

    void schedule(Propagator* p) {
        if (p->active(propagator_state_) and not p->scheduled(propagator_state_)) {
            p->set_scheduled(propagator_state_, true);
            propagation_queue_.push_back(p);
        }
    }

    StateManager* get_state_manager() const { return sm_.get(); }

    const CPVarsState& get_variables_state() const { return var_state_; }
    CPVarsState& get_variables_state() { return var_state_; }

    const CPPropagatorsState& get_propagators_state() const { return propagator_state_; }
    CPPropagatorsState& get_propagators_state() { return propagator_state_; }

 protected:
    // Manager that handles the backtracking stack
    std::unique_ptr<StateManager> sm_;

    // Internal state for the CP variables
    CPVarsState var_state_;

    // Internal state for the Propagators
    CPPropagatorsState propagator_state_;

    // Propagation queue
    std::deque<Propagator*> propagation_queue_;
};

}  // namespace dwave::optimization::cp

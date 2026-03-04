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

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/cp/core/engine.hpp"
#include "dwave-optimization/cp/core/propagator.hpp"
#include "dwave-optimization/cp/core/status.hpp"
#include "dwave-optimization/cp/state/cpvar_state.hpp"
#include "dwave-optimization/graph.hpp"

namespace dwave::optimization::cp {

// forward declaration
class CPVar;

class CPModel {
 public:
    template <class VarType, class... Args>
    VarType* emplace_variable(Args&&... args) {
        static_assert(std::is_base_of_v<CPVar, VarType>);

        if (locked_) {
            throw std::logic_error("cannot add a variable to a locked CP model");
        }

        // Construct via make_unique so we can allow the constructor to throw
        auto uptr = std::make_unique<VarType>(std::forward<Args&&>(args)...);
        VarType* ptr = uptr.get();

        // Pass ownership of the lifespan to nodes_
        variables_.emplace_back(std::move(uptr));
        return ptr;  // return the observing pointer
    }

    template <class PropagatorType, class... Args>
    PropagatorType* emplace_propagator(Args&&... args) {
        static_assert(std::is_base_of_v<Propagator, PropagatorType>);

        if (locked_) {
            throw std::logic_error("cannot add a variable to a locked CP model");
        }

        // Construct via make_unique so we can allow the constructor to throw
        auto uptr = std::make_unique<PropagatorType>(std::forward<Args&&>(args)...);
        PropagatorType* ptr = uptr.get();

        // Pass ownership of the lifespan to nodes_
        propagators_.emplace_back(std::move(uptr));
        return ptr;  // return the observing pointer
    }

    template <class SM>
    CPState initialize_state() const {
        static_assert(std::is_base_of_v<StateManager, SM>);
        std::unique_ptr<SM> sm = std::make_unique<SM>();
        CPState state = CPState(std::move(sm), num_variables(), num_propagators());
        return state;
    }

    ssize_t num_variables() const { return variables_.size(); }

    ssize_t num_propagators() const { return propagators_.size(); }

 protected:
    bool locked_ = false;
    std::vector<std::unique_ptr<Propagator>> propagators_;
    std::vector<std::unique_ptr<CPVar>> variables_;
};

/// Interface for CP variables, allows query and modify their domain. It assumes a flattened array
/// of variables.
class CPVar {
 public:
    virtual ~CPVar() = default;

    // constructor
    CPVar(const CPModel& model, const dwave::optimization::ArrayNode* node_ptr, int index);

    const CPModel& get_model() const { return model_; }

    template <std::derived_from<CPVarData> StateData>
    StateData* data_ptr(CPVarsState& state) const {
        assert(cp_var_index_ >= 0);
        return static_cast<StateData*>(state[cp_var_index_].get());
    }

    template <std::derived_from<CPVarData> StateData>
    const StateData* data_ptr(const CPVarsState& state) const {
        assert(cp_var_index_ >= 0);
        return static_cast<const StateData*>(state[cp_var_index_].get());
    }

    // query the domains
    double min(const CPVarsState& state, int index) const;
    double max(const CPVarsState& state, int index) const;
    double size(const CPVarsState& state, int index) const;
    bool is_bound(const CPVarsState& state, int index) const;
    bool contains(const CPVarsState& state, double value, int index) const;
    bool is_active(const CPVarsState& state, int index) const;
    bool maybe_active(const CPVarsState& state, int index) const;
    ssize_t max_size(const CPVarsState& state) const;
    ssize_t min_size(const CPVarsState& state) const;

    // actions on the domains
    CPStatus remove(CPVarsState& state, double value, int index) const;
    CPStatus remove_above(CPVarsState& state, double value, int index) const;
    CPStatus remove_below(CPVarsState& state, double value, int index) const;
    CPStatus assign(CPVarsState& state, double value, int index) const;
    CPStatus set_min_size(CPVarsState& state, int index) const;
    CPStatus set_max_size(CPVarsState& state, int index) const;

    // actions for the propagation engine
    void schedule_all(CPState& state, const std::vector<Propagator*>& propagators) const;

    // Note: maybe we need the state here.. especially if we want to attach propagators dynamically
    // during the search...
    // Another note: it is als possible to use a vector or struct where the struct holds the
    // propagator and the event type (as an enum) that triggers it. Still considering what to do
    // here..
    void propagate_on_domain_change(Propagator* p);
    void propagate_on_bounds_change(Propagator* p);
    void propagate_on_assignment(Propagator* p);

    void initialize_state(CPState& state) const;

    /// Note: these could be state stacks that can be updated through the search. Keeping them as
    /// simple vectors (static in terms of the search for now)
    // They represent the propagators to trigger when the variable gets-assigned/changes
    // domain/bounds change
    std::vector<Propagator*> on_bind;
    std::vector<Propagator*> on_domain;
    std::vector<Propagator*> on_bounds;
    std::vector<Propagator*> on_array_size_change;

 protected:
    const CPModel& model_;

    // but should I have this?
    const dwave::optimization::ArrayNode* node_;
    const ssize_t cp_var_index_;

    class Listener : public DomainListener {
     public:
        Listener(const CPVar* var, CPState& state) : var_(var), state_(state) {}

        // domain listener overrides

        void bind() override { var_->schedule_all(state_, var_->on_bind); }

        void change() override { var_->schedule_all(state_, var_->on_domain); }

        void change_min() override { var_->schedule_all(state_, var_->on_bounds); }

        void change_max() override { var_->schedule_all(state_, var_->on_bounds); }

        void change_array_size() override {
            var_->schedule_all(state_, var_->on_array_size_change);
        }

     private:
        const CPVar* var_;
        CPState& state_;
    };
};
}  // namespace dwave::optimization::cp

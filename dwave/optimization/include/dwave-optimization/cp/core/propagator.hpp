#pragma once

#include <memory>
#include <vector>

//#include "dwave-optimization/cp/core/status.hpp"
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

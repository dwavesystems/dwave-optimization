#pragma once
#include <functional>

#include "dwave-optimization/cp/state/state.hpp"

namespace dwave::optimization::cp {
class StateManager {
 public:
    virtual ~StateManager() = default;

    virtual int get_level() = 0;
    virtual void with_new_state(std::function<void()> body) = 0;
    virtual void save_state() = 0;
    virtual void restore_state() = 0;
    virtual void restore_state_until(int level) = 0;
    virtual StateInt* make_state_int(int init_value) = 0;
    virtual StateBool* make_state_bool(bool init_value) = 0;
    virtual StateReal* make_state_real(double init_value) = 0;
};
}  // namespace dwave::optimization::cp

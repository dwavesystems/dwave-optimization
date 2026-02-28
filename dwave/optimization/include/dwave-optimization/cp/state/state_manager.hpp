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

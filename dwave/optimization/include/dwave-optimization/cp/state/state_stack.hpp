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

#include "dwave-optimization/cp/state/state.hpp"
#include "dwave-optimization/cp/state/state_manager.hpp"

namespace dwave::optimization::cp {
template <class T>
class StateStack {
 public:
    // constructor
    StateStack(StateManager* sm);

    void push(T* elem);

    int size() const;
    T* get(int index) const;

 protected:
    // ownership of the objects T is shared between the objects owning the stack_
    //  std::vector<std::shared_ptr<T>> stack_;
    std::vector<T*> stack_;
    // the size_ int is owned by the state manager
    StateInt* size_;
};
}  // namespace dwave::optimization::cp

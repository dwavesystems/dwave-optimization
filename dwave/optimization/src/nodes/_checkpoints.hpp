// Copyright 2026 D-Wave
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

#include <variant>

#include "dwave-optimization/state.hpp"

namespace dwave::optimization {

class CheckpointableState;

class LinkedListCheckpoint : public NodeStateCheckpoint {
 public:
    LinkedListCheckpoint() = delete;
    // We're not moveable or copyable because NodeStateCheckpoint is not.

    LinkedListCheckpoint(CheckpointableState& state);

    ~LinkedListCheckpoint() override;

 protected:  // todo: private?
    friend CheckpointableState;

    LinkedListCheckpoint* prev_ptr_;

    // Is usually not nullptr unless the state has been destructed
    std::variant<LinkedListCheckpoint*, CheckpointableState*> next_ptr_;
};

class CheckpointableState : public NodeStateData {
 public:
    CheckpointableState() = default;

    CheckpointableState(const CheckpointableState& other) {
        assert(false);
    }
    CheckpointableState(CheckpointableState&&) = default;

    CheckpointableState& operator=(const CheckpointableState&) {
        assert(false);
    }
    CheckpointableState& operator=(CheckpointableState&&) = default;

    ~CheckpointableState();

 protected:
    template <std::derived_from<LinkedListCheckpoint> T>
    T* checkpoint_ptr() {
        return static_cast<T*>(prev_ptr_);
    }

 private:  // todo: private?
    friend LinkedListCheckpoint;

    // The name is a bit confusing, but by making it match LinkedListCheckpoint::prev_ptr_
    // it makes the implementations of the various visit methods clearer.
    LinkedListCheckpoint* prev_ptr_ = nullptr;  // Will be nullptr if there are no checkpoints
};

}  // namespace dwave::optimization

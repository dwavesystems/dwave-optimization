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

#include "_checkpoints.hpp"

namespace dwave::optimization {

// Place self between the state and any checkpoint it's currently holding
LinkedListCheckpoint::LinkedListCheckpoint(CheckpointableState& state) :
    prev_ptr_(state.prev_ptr_), next_ptr_(&state) {
    if (prev_ptr_ != nullptr) prev_ptr_->next_ptr_ = this;
    state.prev_ptr_ = this;
}

LinkedListCheckpoint::~LinkedListCheckpoint() {
    if (prev_ptr_ != nullptr) prev_ptr_->next_ptr_ = next_ptr_;

    // Now make sure next_ptr is pointing to prev_ptr (which can be null)
    std::visit(
        [&](auto* next_ptr) -> void {
            if (next_ptr == nullptr) return;  // state was destructed first
            next_ptr->prev_ptr_ = prev_ptr_;
        },
        next_ptr_
    );
}

CheckpointableState::~CheckpointableState() {
    if (prev_ptr_ == nullptr) return;  // nothing to clean up
    prev_ptr_->next_ptr_ = static_cast<CheckpointableState*>(nullptr);
}

}  // namespace dwave::optimization

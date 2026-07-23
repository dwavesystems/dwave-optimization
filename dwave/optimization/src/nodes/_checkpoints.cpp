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

CheckpointableState::~CheckpointableState() {
    if (prev_ptr_ == nullptr) return;  // nothing to clean up
    prev_ptr_->next_ptr_ = static_cast<CheckpointableState*>(nullptr);
}

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

DiffCheckpoint::DiffCheckpoint(CheckpointableState& state, ssize_t drop) :
    LinkedListCheckpoint(state), updates_(), drop_(drop) {}

DiffCheckpoint::DiffCheckpoint(CheckpointableState& state, std::span<const Update> diff) :
    LinkedListCheckpoint(state), updates_(), drop_(diff.size()) {
    if (auto* prev_ptr = static_cast<DiffCheckpoint*>(prev_ptr_)) {
        prev_ptr->commit_updates(std::vector<Update>(diff.begin(), diff.end()));
        assert(prev_ptr->drop_ == 0);
    }
}

DiffCheckpoint::~DiffCheckpoint() {
    // if we're the oldest checkpoint, just let whatever information we're
    // holding get destructed with us
    if (prev_ptr_ == nullptr) return;

    // otherwise we need to transfer our info over
    auto* prev_ptr = static_cast<DiffCheckpoint*>(prev_ptr_);
    assert(prev_ptr->drop_ == 0);
    for (auto& updates : updates_) prev_ptr->commit_updates(std::move(updates));
    prev_ptr->drop_ = drop_;
}

void DiffCheckpoint::commit_updates(std::vector<Update> updates) {
    assert(0 <= drop_ and static_cast<size_t>(drop_) <= updates.size());

    if (drop_) {
        updates.erase(updates.begin(), updates.begin() + drop_);
        drop_ = 0;
    }

    updates_.emplace_back(std::move(updates));
}

void DiffCheckpoint::revert_updates(std::vector<Update> updates) {
    assert(0 <= drop_ and static_cast<size_t>(drop_) <= updates.size());

    if (not drop_) return;  // nothing to do

    // We want to track the updates that would revert the changes from the
    // current state.
    // In C++23 we could use assign_range() which would be nicer
    auto relevant = std::move(updates) | std::views::take(drop_) | std::views::reverse |
                    std::views::transform([](const Update& up) { return up.inverse(); });
    updates_.emplace_back(relevant.begin(), relevant.end());

    drop_ = 0;
}

}  // namespace dwave::optimization

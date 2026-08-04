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

#include <span>
#include <variant>
#include <vector>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/state.hpp"

namespace dwave::optimization {

class CheckpointableState;

// A LinkedListCheckpoint is one checkpoint in a chain of checkpoints implemented
// as a doubly-linked list.
class LinkedListCheckpoint : public NodeStateCheckpoint {
 public:
    LinkedListCheckpoint() = delete;
    // We're not moveable or copy-able because NodeStateCheckpoint is not.

    LinkedListCheckpoint(CheckpointableState& state);

    ~LinkedListCheckpoint() override;

 protected:
    friend CheckpointableState;

    // The next-oldest checkpoint in the chain. Can be nullptr which indicates
    // that this is the oldest checkpoint.
    LinkedListCheckpoint* prev_ptr_;

    // The next-newest checkpoint in the chain or, if this is the newest
    // newest checkpoint, will point to the node state.
    // Is usually not nullptr unless the state has been destructed before the
    // checkpoint has.
    std::variant<LinkedListCheckpoint*, CheckpointableState*> next_ptr_;
};

// A mixin class for states to work with LinkedListCheckpoints.
class CheckpointableState {
 public:
    CheckpointableState() = default;

    // When CheckpointableState is copied, we don't want the new state to inherit
    // its checkpoints.
    CheckpointableState(const CheckpointableState&) {}

    CheckpointableState(CheckpointableState&&) = default;

    CheckpointableState& operator=(const CheckpointableState&) = delete;
    CheckpointableState& operator=(CheckpointableState&&) = default;

    ~CheckpointableState();

 protected:
    template <std::derived_from<LinkedListCheckpoint> T>
    T* checkpoint_ptr() {
        return static_cast<T*>(prev_ptr_);
    }

 private:
    friend LinkedListCheckpoint;

    // The name is a bit confusing, but by making it match LinkedListCheckpoint::prev_ptr_
    // it makes the implementations of the various visit methods clearer.
    // Will be nullptr if there are no checkpoints
    LinkedListCheckpoint* prev_ptr_ = nullptr;
};

// A DiffCheckpoint is a type of linked list checkpoint that stores the diffs
// since it was created.
class DiffCheckpoint : public LinkedListCheckpoint {
 public:
    DiffCheckpoint(CheckpointableState& state, std::span<const Update> diff);

    ~DiffCheckpoint() override;

    // Add updates associated with a commit to the checkpoint. The checkpoint
    // therefore stores the information it needs to later undo those changes.
    void commit_updates(std::vector<Update> updates);

    // Clear all the updates held by the checkpoint and return them to the
    // caller.
    auto detach_updates() {
        auto updates = std::move(updates_) | std::views::join;
        assert(updates_.empty());
        return updates;
    }

    // The current "drop". The drop is used when a checkpoint is created while
    // a node has some mutations already applied. This tells the checkpoint
    // how to handle the diff associated with those mutations, i.e., the ones
    // the checkpoint shouldn't be tracking. 
    ssize_t& drop() { return drop_; }

    // Add updates associated with a revert to the checkpoint. The checkpoint
    // therefore stores the information it needs to later undo those changes.
    void revert_updates(std::vector<Update> updates);

 protected:
    DiffCheckpoint(CheckpointableState& state, ssize_t drop);

 private:
    // We store the updates as a vector-of-vectors in order to make them fast
    // to append.
    std::vector<std::vector<Update>> updates_;

    // See drop() docstring.
    ssize_t drop_;
};

}  // namespace dwave::optimization

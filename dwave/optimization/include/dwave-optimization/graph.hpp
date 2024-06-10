// Copyright 2023 D-Wave Systems Inc.
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

#include <algorithm>
#include <cassert>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include "dwave-optimization/state.hpp"
#include "dwave-optimization/utils.hpp"
#include "dwave-optimization/vartypes.hpp"

namespace dwave::optimization {

class Array;
class Node;

// We don't want this interface to be opinionated about what type of rng we're using.
// So we create this class to do type erasure on RNGs.
// The performance won't be great, but this is only intended to be used for default
// moves.
class RngAdaptor {
 public:
    // use a fixed width so we can be sure of the min and max
    using result_type = std::uint32_t;

    // By default just use Mersenne Twister
    RngAdaptor() : RngAdaptor(std::mt19937()) {}

    template <class Generator>
    explicit RngAdaptor(Generator&& r)
            : rng_(engine_type<std::decay_t<Generator>>(std::forward<Generator>(r))) {}

    static constexpr result_type min() { return std::numeric_limits<result_type>::min(); }
    static constexpr result_type max() { return std::numeric_limits<result_type>::max(); }
    result_type operator()() const { return rng_(); }

 private:
    template <class Generator>
    using engine_type = std::independent_bits_engine<Generator, 32, result_type>;

    std::function<result_type()> rng_;
};
// Confirm this will work with rngs
static_assert(std::uniform_random_bit_generator<RngAdaptor>);

// A decision is a class that has at least one move.
struct Decision {
    virtual void default_move(State& state, RngAdaptor& rng) const = 0;
};

class Graph {
 public:
    Graph();
    ~Graph();

    template <class NodeType, class... Args>
    NodeType* emplace_node(Args&&... args);

    State initialize_state() const;
    State initialize_state();  // topologically sorts first
    void initialize_state(State& state) const;
    void initialize_state(State& state);  // topologically sorts first
    State empty_state() const;
    State empty_state();  // topologically sorts first

    // Initialize the state of the given node and all predecessors recursively.
    static void recursive_initialize(State& state, const Node* ptr);
    // Reset the state of the given node and all successors recursively.
    static void recursive_reset(State& state, const Node* ptr);

    // Sort the nodes topologically. This "locks" the model in that nodes cannot
    // be added to a topologically sorted model without invalidating the topological
    // ordering.
    void topological_sort();

    // Reset the topological sort. Note that the decision variables will still
    // have a topological ordering.
    void reset_topological_sort();

    // Whether or not the model is currently topologically sorted.
    // Models that are topologically sorted cannot be modified.
    bool topologically_sorted() const noexcept { return topologically_sorted_; }

    std::span<const std::unique_ptr<Node>> nodes() const { return nodes_; }

    // Given the source (changing) nodes, update the model incrementally and accept the changes
    // according to the accept function.
    void propose(
            State& state, std::vector<const Node*> sources,
            std::function<bool(const Graph&, State&)> accept = [](const Graph&, State&) {
                return true;
            }) const;

    // Get the descendants of the source nodes, that is, all nodes that can be visited starting from
    // the sources.
    static std::vector<const Node*> descendants(State& state, std::vector<const Node*> sources);

    // Call the propagate method for each node that changes.
    void propagate(State& state, std::span<const Node*> changed) const;

    // Commit the changes on each changed node.
    void commit(State& state, std::span<const Node*> changed) const;

    // Revert the changes on each changed node.
    void revert(State& state, std::span<const Node*> changed) const;

    // The number of decision nodes.
    ssize_t num_decisions() const noexcept { return num_decisions_; }

    // The number of nodes in the model.
    ssize_t num_nodes() const noexcept { return nodes_.size(); }

    // The number of constraints in the model.
    ssize_t num_constraints() const noexcept { return constraints_.size(); }

    // Specifying the objective node. Must be an array output
    void set_objective(Array* objective_ptr);
    // void reset_objective();

    void add_constraint(Array* constraint_ptr);
    std::span<Array*> constraints() { return constraints_; }
    std::span<Array const * const> constraints() const { return constraints_; }

    double energy(const State& state) const;
    bool feasible(const State& state) const;

    // Return a vector to the decisions. Note this is a copy, not a view!
    // The graph must be topologically sorted
    std::vector<const Decision*> decisions() const;

 private:
    static void visit_(Node* n_ptr, int* count_ptr);

    std::vector<std::unique_ptr<Node>> nodes_;

    // The number of decision variables in the model
    ssize_t num_decisions_ = 0;

    Array* objective_ptr_ = nullptr;
    std::vector<Array*> constraints_;

    // Track whether the model is currently topologically sorted
    bool topologically_sorted_ = false;
};

class Node {
 public:
    struct SuccessorView {
        SuccessorView(Node* ptr, int index) noexcept : ptr(ptr), index(index) {}

        // Can dereference just like a pointer
        Node& operator*() { return *ptr; }
        Node& operator*() const { return *ptr; }
        Node* operator->() { return ptr; }
        Node* operator->() const { return ptr; }

        // Can implicitly cast. Though we might want to require explicit casting
        // later if this is becoming confusing
        operator Node*() { return ptr; }
        operator Node*() const { return ptr; }

        Node* ptr;
        int index;  // the index of self in the successor
    };

    Node() noexcept : expired_ptr_(new bool(false)) {}
    virtual ~Node() { *expired_ptr_ = true; }

    const std::vector<Node*>& predecessors() const { return predecessors_; }
    const std::vector<SuccessorView>& successors() const { return successors_; }

    virtual void initialize_state(State& state) const;

    // The current topological index. Will be negative if unsorted.
    ssize_t topological_index() const { return topological_index_; }

    // Incorporate any update(s) from predecessor nodes and then call the update()
    // method of any successors.
    // Note that calling update() on successors must happen *AFTER* the state
    // has been updated.
    // By default, this method calls the update() method of all succesors.
    // Subclasses will in general want to update their state and/or filter
    // the list of successors that need to be updated.
    virtual void propagate(State& state) const {
        for (const auto& sv : successors()) {
            sv->update(state, sv.index);
        }
    }

    virtual void commit(State& state) const = 0;
    virtual void revert(State& state) const = 0;

    // Called by predecessor nodes to signal that they have changes that need
    // to be incorporated into this node's state.
    // By default this method does nothing.
    // Subclasses will in general want to track which predececessors have been
    // updated and may even want to eagerly incorporate changes.
    virtual void update(State& state, int index) const {}

    // Return a shared pointer to a bool value. When the node is destructed
    // the bool will be set to True
    std::shared_ptr<bool> expired_ptr() const { return expired_ptr_; }

    friend void Graph::topological_sort();
    friend void Graph::reset_topological_sort();
    template <class NodeType, class... Args>
    friend NodeType* Graph::emplace_node(Args&&... args);

 protected:
    // For use by non-dynamic node constructors.
    Node(std::initializer_list<Node*> nodes) noexcept
            : predecessors_(nodes), expired_ptr_(new bool(false)) {
        int idx = 0;
        for (Node* ptr : predecessors_) {
            ptr->successors_.emplace_back(this, idx);
            ++idx;
        }
    }

    /// Add a predecessor node. Adds itself to the predecessor as a successor.
    void add_predecessor(Node* predecessor) {
        assert(this->topological_index_ <= 0 &&
               "cannot add a predecessor to a topologically sorted node");
        predecessor->successors_.emplace_back(this, this->predecessors_.size());
        this->predecessors_.emplace_back(predecessor);
    }

    /// Add a successor node. Adds itself to the successor as a predecessor.
    void add_successor(Node* successor) {
        assert(successor->topological_index_ <= 0 && "cannot add a topologically sorted successor");
        this->successors_.emplace_back(successor, successor->predecessors_.size());
        successor->predecessors_.emplace_back(this);
    }

    template <class StateData>
    StateData* data_ptr(State& state) const {
        int index = topological_index();
        assert(index >= 0 && "must be topologically sorted");
        assert(static_cast<int>(state.size()) > index && "unexpected state length");
        assert(state[index] != nullptr && "uninitialized state");

        return static_cast<StateData*>(state[index].get());
    }
    template <class StateData>
    const StateData* data_ptr(const State& state) const {
        int index = topological_index();
        assert(index >= 0 && "must be topologically sorted");
        assert(static_cast<int>(state.size()) > index && "unexpected state length");
        assert(state[index] != nullptr && "uninitialized state");

        return static_cast<const StateData*>(state[index].get());
    }

 private:
    ssize_t topological_index_ = -1;  // negative is unset

    std::vector<Node*> predecessors_;
    std::vector<SuccessorView> successors_;  // todo: successor view

    // Used to signal whether the node is expired to observers. Should be set
    // to true when the Node is deallocated.
    std::shared_ptr<bool> expired_ptr_;
};

template <class NodeType, class... Args>
NodeType* Graph::emplace_node(Args&&... args) {
    static_assert(std::is_base_of_v<Node, NodeType>);

    if (topologically_sorted_) {
        // "locked" is a python concept, but we use it rather than "topologically sorted"
        // to avoid a lot of fiddling with error handling.
        throw std::logic_error("cannot add a symbol to a locked model");
    }

    // Construct via make_unique so we can allow the constructor to throw
    nodes_.emplace_back(std::make_unique<NodeType>(std::forward<Args&&>(args)...));
    NodeType* ptr = static_cast<NodeType*>(nodes_.back().get());

    // Decisions get their topological index assigned immediately, in the order of insertion
    if constexpr (std::is_base_of_v<Decision, NodeType>) {
        ptr->topological_index_ = num_decisions_++;
    }

    return ptr;  // return the observing pointer
}

// Useful concepts
// todo: consider moving to another header file

template <typename T>
concept ArrayNode = std::is_base_of<Array, T>::value && std::is_base_of<Node, T>::value;

}  // namespace dwave::optimization

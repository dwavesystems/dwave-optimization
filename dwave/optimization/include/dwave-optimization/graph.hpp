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
#include <string>
#include <utility>
#include <vector>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/common.hpp"
#include "dwave-optimization/state.hpp"

namespace dwave::optimization {

class ArrayNode;
class Node;
class DecisionNode;
class InputNode;

// A decision is an independent variable in the model. The value(s) to be optimized.
struct Decision {};

class Graph {
 public:
    Graph() noexcept = default;
    ~Graph() noexcept = default;

    // We disallow copy construction and assignment because it would only be
    // a shallow copy/assignment in terms of the underlying nodes.
    // We could implement these in the future if desired but it would be quite
    // non-trivial.
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    Graph(Graph&&) noexcept = default;
    Graph& operator=(Graph&&) noexcept = default;

    /// Add a constraint node.
    void add_constraint(ArrayNode* constraint_ptr);

    /// Call commit on every `Node` in the `Graph`.
    void commit(State& state) const;

    /// Commit the changes on each changed node.
    void commit(State& state, std::span<const Node*> changed) const;
    void commit(State& state, std::vector<const Node*>&& changed) const;

    /// Return the constraints of the graph as a span of nodes.
    std::span<ArrayNode* const> constraints() noexcept { return constraints_; }
    std::span<const ArrayNode* const> constraints() const noexcept { return constraints_; }

    /// Retrieve all of the decisions in the model
    std::span<DecisionNode* const> decisions() noexcept { return decisions_; }
    std::span<const DecisionNode* const> decisions() const noexcept { return decisions_; }

    /// Get the descendants of the source nodes, that is, all nodes that can be visited starting
    /// from the sources.
    static std::vector<const Node*> descendants(State& state, std::vector<const Node*> sources);
    std::vector<const Node*> descendants(std::vector<const Node*> sources) const;

    /// Add a new node to the graph.
    template <class NodeType, class... Args>
    NodeType* emplace_node(Args&&... args);

    /// Create a new "empty" state, that is a state with no nodes initialized.
    /// The `const` version will fail if the model is not topologically sorted,
    /// the non-`const` version will topologically sort the model.
    State empty_state() const;
    State empty_state();

    /// Calculate the current energy of the model, that is the output of the
    /// objective node.
    double energy(const State& state) const;

    /// Return `true` if the graph is feasible. A graph is feasible if all constraints
    /// output `true`.
    bool feasible(const State& state) const;

    /// Create a new state with all nodes initialized by their default initializer.
    /// The `const` version will fail if the model is not topologically sorted,
    /// the non-`const` version will topologically sort the model.
    State initialize_state() const;
    State initialize_state();

    /// Given a state, initialize any nodes that have not already been initialized.
    /// The `const` version will fail if the model is not topologically sorted,
    /// the non-`const` version will topologically sort the model.
    void initialize_state(State& state) const;
    void initialize_state(State& state);

    /// Return a span over all of the input nodes in the model.
    std::span<InputNode* const> inputs() noexcept { return inputs_; }
    std::span<const InputNode* const> inputs() const noexcept { return inputs_; }

    /// All of the nodes in the graph.
    std::span<const std::unique_ptr<Node>> nodes() const { return nodes_; }

    /// The number of decision nodes.
    ssize_t num_decisions() const noexcept { return decisions_.size(); }

    /// The number of nodes in the model.
    ssize_t num_nodes() const noexcept { return nodes_.size(); }

    /// The number of constraints in the model.
    ssize_t num_constraints() const noexcept { return constraints_.size(); }

    /// The number of input nodes in the model.
    ssize_t num_inputs() const noexcept { return inputs_.size(); }

    /// Return a pointer to the node that is the objective of the model.
    /// Will return nullptr if there is no objective set.
    ArrayNode* objective() noexcept { return objective_ptr_; }
    const ArrayNode* objective() const noexcept { return objective_ptr_; }

    /// Call propagate on every `Node` in the `Graph`.
    void propagate(State& state) const;

    /// Call the propagate method on each node in changed. Note this does not call propagate on
    /// the descendents of changed.
    void propagate(State& state, std::span<const Node*> changed) const;
    void propagate(State& state, std::vector<const Node*>&& changed) const;

    /// Given the source (changing) nodes, update the model incrementally and accept the changes
    /// according to the accept function.
    void propose(
        State& state,
        std::vector<const Node*> sources,
        std::function<bool(const Graph&, State&)> accept = [](const Graph&, State&) { return true; }
    ) const;

    /// Initialize the state of the given node and all predecessors recursively.
    static void recursive_initialize(State& state, const Node* ptr);
    /// Reset the state of the given node and all successors recursively.
    static void recursive_reset(State& state, const Node* ptr);

    /// Remove unused nodes from the graph.
    ///
    /// This method will reset the topological sort if there is one.
    ///
    /// A node is considered unused if all of the following are true:
    /// * It is not a decision.
    /// * It is not an ancestor of the objective.
    /// * It is not an ancestor of a constraint.
    /// * It has no "listeners" on its expired_ptr. Set ``ignore_listeners`` to
    ///   ``true`` to disable this condition.
    ///
    /// Returns the number of nodes removed from the graph.
    ssize_t remove_unused_nodes(bool ignore_listeners = false);

    /// Call revert on every `Node` in the `Graph`.
    void revert(State& state) const;

    /// Revert the changes on each changed node.
    void revert(State& state, std::span<const Node*> changed) const;
    void revert(State& state, std::vector<const Node*>&& changed) const;

    /// Reset the topological sort. Note that the decision variables will still
    /// have a topological ordering.
    void reset_topological_sort();

    /// Specify the objective node. Must be an array with a single element.
    /// To unset the objective provide nullptr.
    void set_objective(ArrayNode* objective_ptr);

    /// Sort the nodes topologically. This "locks" the model in that nodes cannot
    /// be added to a topologically sorted model without invalidating the topological
    /// ordering.
    void topological_sort();

    /// Whether or not the model is currently topologically sorted.
    /// Models that are topologically sorted cannot be modified.
    bool topologically_sorted() const noexcept { return topologically_sorted_; }

 private:
    std::vector<std::unique_ptr<Node>> nodes_;

    // The nodes with important semantic meanings to the model.
    // All of these pointers are non-owning!
    ArrayNode* objective_ptr_ = nullptr;
    std::vector<ArrayNode*> constraints_;
    std::vector<DecisionNode*> decisions_;
    std::vector<InputNode*> inputs_;

    // Track whether the model is currently topologically sorted
    bool topologically_sorted_ = false;
};

class Node {
 public:
    struct SuccessorView {
        // Cannot construct a SuccessorView without a Node/index
        SuccessorView() = delete;
        ~SuccessorView() = default;

        // Move/copy work normally
        SuccessorView(const SuccessorView&) = default;
        SuccessorView(SuccessorView&&) = default;
        SuccessorView& operator=(const SuccessorView&) = default;
        SuccessorView& operator=(SuccessorView&&) = default;

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

    // Nodes cannot be moved or copied.
    Node(const Node&) = delete;
    Node(Node&&) noexcept = delete;
    Node& operator=(const Node&) = delete;
    Node& operator=(Node&&) noexcept = delete;

    /// Methods for interrogating nodes as strings. Useful for error messages
    /// and debugging. We roughly follow Python's scheme of repr() and str()
    /// printing different information.
    virtual std::string classname() const;

    /// Commit any changing updates to the node.
    virtual void commit(State& state) const = 0;

    /// Return true if the node's state is deterministic - that is it's uniquely
    /// derived from its predecessors. Defaults to `true`, except for decisions.
    virtual bool deterministic_state() const { return true; }

    /// Return a shared pointer to a bool value. When the node is destructed
    /// the bool will be set to True
    std::shared_ptr<const bool> expired_ptr() const { return expired_ptr_; }

    /// Initialize the state of the node.
    virtual void initialize_state(State& state) const;

    /// Return the number of "listeners", that is the number of shared_ptrs,
    /// as returned by expired_ptr().
    ssize_t num_listeners() const {
        assert(expired_ptr_.use_count() >= 1);  // self counts as 1
        return static_cast<ssize_t>(expired_ptr_.use_count()) - 1;
    }

    /// Return predecessors of the node.
    const std::vector<Node*>& predecessors() const { return predecessors_; }

    /// Incorporate any update(s) from predecessor nodes and then call the update()
    /// method of any successors.
    /// Note that calling update() on successors must happen *AFTER* the state
    /// has been updated.
    /// By default, this method calls the update() method of all succesors.
    /// Subclasses will in general want to update their state and/or filter
    /// the list of successors that need to be updated.
    virtual void propagate(State& state) const {
        for (const auto& sv : successors()) {
            sv->update(state, sv.index);
        }
    }

    /// @copydoc Node::classname()
    virtual std::string repr() const;

    /// Revert any pending changes to the node.
    virtual void revert(State& state) const = 0;

    /// @copydoc Node::classname()
    virtual std::string str() const;

    /// Return the successors of the node.
    const std::vector<SuccessorView>& successors() const { return successors_; }

    /// The current topological index. Will be negative if unsorted.
    ssize_t topological_index() const { return topological_index_; }

    /// Called by predecessor nodes to signal that they have changes that need
    /// to be incorporated into this node's state.
    /// By default this method does nothing.
    /// Subclasses will in general want to track which predececessors have been
    /// updated and may even want to eagerly incorporate changes.
    virtual void update(State& state, int index) const {}

    /// Nodes are printable
    friend std::ostream& operator<<(std::ostream& os, const Node& node);

    friend void Graph::topological_sort();
    friend void Graph::reset_topological_sort();
    template <class NodeType, class... Args>
    friend NodeType* Graph::emplace_node(Args&&... args);
    friend ssize_t Graph::remove_unused_nodes(bool ignore_listeners);

 protected:
    // For use by non-dynamic node constructors.
    Node(std::initializer_list<Node*> nodes) noexcept :
        predecessors_(nodes), expired_ptr_(new bool(false)) {
        int idx = 0;
        for (Node* ptr : predecessors_) {
            ptr->successors_.emplace_back(this, idx);
            ++idx;
        }
    }

    /// Add a predecessor node. Adds itself to the predecessor as a successor.
    void add_predecessor_(Node* predecessor) {
        assert(
            this->topological_index_ <= 0 &&
            "cannot add a predecessor to a topologically sorted node"
        );
        predecessor->successors_.emplace_back(this, this->predecessors_.size());
        this->predecessors_.emplace_back(predecessor);
    }

    template <std::derived_from<NodeStateData> StateData>
    StateData* data_ptr_(State& state) const {
        const ssize_t index = topological_index();
        assert(index >= 0 and "must be topologically sorted");
        assert(state.size() > static_cast<std::size_t>(index) and "unexpected state length");
        assert(state[index] != nullptr and "uninitialized state");

        return static_cast<StateData*>(state[index].get());
    }
    template <std::derived_from<NodeStateData> StateData>
    const StateData* data_ptr_(const State& state) const {
        const ssize_t index = topological_index();
        assert(index >= 0 and "must be topologically sorted");
        assert(state.size() > static_cast<std::size_t>(index) and "unexpected state length");
        assert(state[index] != nullptr and "uninitialized state");

        return static_cast<const StateData*>(state[index].get());
    }

    template <std::derived_from<NodeStateData> StateData, class... Args>
    void emplace_data_ptr_(State& state, Args&&... args) const {
        const ssize_t index = topological_index();
        assert(index >= 0 and "must be topologically sorted");
        assert(state.size() > static_cast<std::size_t>(index) and "unexpected state length");
        assert(state[index] == nullptr and "already initialized state");

        state[index] = std::make_unique<StateData>(std::forward<Args&&>(args)...);
    }

    // Whether this node type is elegible to be removed from the model
    virtual bool removable_() const { return true; }

 private:
    ssize_t topological_index_ = -1;  // negative is unset

    std::vector<Node*> predecessors_;
    std::vector<SuccessorView> successors_;

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
    auto uptr = std::make_unique<NodeType>(std::forward<Args&&>(args)...);
    NodeType* ptr = uptr.get();

    // Pass ownership of the lifespan to nodes_
    nodes_.emplace_back(std::move(uptr));

    // Decisions get their topological index assigned immediately, in the order of insertion
    if constexpr (std::is_base_of_v<Decision, NodeType>) {
        static_assert(std::is_base_of_v<DecisionNode, NodeType>);
        ptr->topological_index_ = decisions_.size();
        decisions_.emplace_back(ptr);
    } else if constexpr (std::is_base_of_v<InputNode, NodeType>) {
        inputs_.emplace_back(ptr);
    }

    return ptr;  // return the observing pointer
}

class ArrayNode : public Array, public virtual Node {};
class DecisionNode : public Decision, public virtual Node {
 public:
    /// Decision nodes by definition do not have a deterministic state.
    bool deterministic_state() const final { return false; }

    /// Decisions don't have predecessors so no one should be calling update().
    /// Always throws a logic_error.
    [[noreturn]] void update(State& state, int index) const override;

 protected:
    /// In general we do not allow decisions to be removed from models.
    bool removable_() const override { return false; }
};

}  // namespace dwave::optimization

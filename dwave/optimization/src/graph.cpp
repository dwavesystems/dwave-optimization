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

#include "dwave-optimization/graph.hpp"

#include <algorithm>
#include <deque>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <utility>


#if defined( __has_include ) && __has_include(<cxxabi.h>)
#define _HAS_CXXABI
#include <cxxabi.h>
#endif

#include "dwave-optimization/array.hpp"

namespace dwave::optimization {

// Graph **********************************************************************

void Graph::topological_sort() {
    if (topologically_sorted_) return;

    // The decisions already have their topological index assigned

    // Find a topological ordering of all of the non-decisions
    // The extra `visit` parameter is to allow for recursive lambdas
    // We iterate over successors in reverse order to preserve their original ordering
    // after sorting based on the topological index (which is counting down during DFS).
    auto visit = [](Node* n_ptr, int* count_ptr, auto&& visit) -> void {
        if (n_ptr->topological_index_ >= 0) return;
        if (n_ptr->topological_index_ == -2) throw std::logic_error("has cycles");

        // decisions should already be sorted
        assert(dynamic_cast<Decision*>(n_ptr) == nullptr && "unsorted decisions node");

        n_ptr->topological_index_ = -2;

        for (Node* m_ptr : n_ptr->successors() | std::views::reverse) {
            visit(m_ptr, count_ptr, visit);
        }

        n_ptr->topological_index_ = *count_ptr;
        *count_ptr -= 1;
    };

    int count = num_nodes() - 1;
    for (const std::unique_ptr<Node>& node_ptr : nodes_ | std::views::reverse) {
        // todo: iterative version
        visit(node_ptr.get(), &count, visit);
    }

    // Check that we have no gaps in our range.
    // Because the decisions were already sorted, that should determine the
    // count after assigning all others
    assert(count == num_decisions() - 1);

    // for later convenience, we sort the nodes_ by index
    // This moves all of the decisions to the front
    std::sort(nodes_.begin(), nodes_.end(),
              [](const std::unique_ptr<Node>& n_ptr, const std::unique_ptr<Node>& m_ptr) {
                  return n_ptr->topological_index_ < m_ptr->topological_index_;
              });

    topologically_sorted_ = true;
}

void Graph::reset_topological_sort() {
    if (!topologically_sorted_) return;  // already unsorted

    // The decisions must have a consistent topological index, so we don't reset them
    std::for_each(
            // std::execution::par_unseq,  // todo: performance testing. Probably won't help
            nodes_.begin() + num_decisions(), nodes_.end(), [](const std::unique_ptr<Node>& n_ptr) {
                assert(dynamic_cast<Decision*>(n_ptr.get()) == nullptr);  // not a decision
                n_ptr->topological_index_ = -1;
            });

    topologically_sorted_ = false;
}

State Graph::initialize_state() const {
    auto state = empty_state();
    initialize_state(state);
    return state;
}

State Graph::initialize_state() {
    topological_sort();
    return static_cast<const Graph*>(this)->initialize_state();
}

void Graph::initialize_state(State& state) const {
    assert(static_cast<int>(state.size()) == num_nodes() && "unexpected state length");
    assert(topologically_sorted_ && "graph must be topologically sorted");

    for (int i = 0, end = num_nodes(); i < end; ++i) {
        if (state[i]) continue;  // should this clear any pending changes?

        nodes_[i]->initialize_state(state);
    }
}

void Graph::recursive_initialize(State& state, const Node* ptr) {
    ssize_t index = ptr->topological_index();

    if (index < 0) {
        throw std::logic_error("cannot initialize a node that has not been topologically sorted");
    }

    // make sure that the state is large enough
    if (index >= static_cast<ssize_t>(state.size())) {
        state.resize(index + 1);
    }

    if (state[index]) return;  // it's been created already

    // otherwise, make sure all predecessors are initialized
    for (const dwave::optimization::Node* pred_ptr : ptr->predecessors()) {
        recursive_initialize(state, pred_ptr);
    }

    ptr->initialize_state(state);
}

void Graph::recursive_reset(State& state, const Node* ptr) {
    ssize_t index = ptr->topological_index();

    if (index < 0) {
        throw std::logic_error("cannot reset a node that has not been topologically sorted");
    }

    // In the case that we're past the end of the state, we're by definition reset!
    if (index >= static_cast<ssize_t>(state.size())) {
        return;
    }

    // We've already been reset so nothing to do
    if (!state[index]) return;

    // Otherwise, reset our own state and then all of our successors
    state[index].reset();

    for (const dwave::optimization::Node* successor_ptr : ptr->successors()) {
        if (successor_ptr->topological_index() < 0) continue;  // nothing to reset
        recursive_reset(state, successor_ptr);
    }
}

void Graph::initialize_state(State& state) {
    topological_sort();
    static_cast<const Graph*>(this)->initialize_state(state);
}

State Graph::empty_state() const { return State(num_nodes()); }
State Graph::empty_state() {
    topological_sort();
    return static_cast<const Graph*>(this)->empty_state();
}

void Graph::set_objective(ArrayNode* objective_ptr) {
    // nullptr is an unset objective, so we allow it.
    if (objective_ptr != nullptr && objective_ptr->size() != 1) {
        throw std::invalid_argument("objective must have a single output");
    }
    this->objective_ptr_ = objective_ptr;
}

void Graph::add_constraint(ArrayNode* constraint_ptr) {
    if (!constraint_ptr->logical()) {
        throw std::invalid_argument("constraint must have a logical output");
    }
    // todo: we could substitute an AND on the user's behalf, or we could allow
    // multidimensional constraints as a way of supporting many constraints at once
    if (constraint_ptr->size() != 1) {
        throw std::invalid_argument(
                "The truth value of an array with more than one element is ambiguous");
    }

    constraints_.emplace_back(constraint_ptr);
}

double Graph::energy(const State& state) const {
    if (objective_ptr_ == nullptr) return 0;
    return objective_ptr_->view(state).front();
}
bool Graph::feasible(const State& state) const {
    for (const Array* ptr : constraints_) {
        assert(ptr->size(state) == 1);
        if (!(ptr->view(state)[0])) return false;
    }
    return true;
}

// Performs Breadth First Search and sort the nodes visited according to their topological number.
std::vector<const Node*> Graph::descendants(State& state, std::vector<const Node*> sources) {
    // Perform BFS starting from the sources
    ssize_t exploration_index = 0;

    while (exploration_index != static_cast<ssize_t>(sources.size())) {
        const Node* n_ptr = sources[exploration_index++];

        for (Node* m_ptr : n_ptr->successors()) {
            if (state[m_ptr->topological_index()]->mark) continue;

            state[m_ptr->topological_index()]->mark = true;
            sources.emplace_back(m_ptr);
        }
    }

    // Sort the nodes according to topological number
    std::sort(sources.begin(), sources.end(), [](const Node* n_ptr, const Node* m_ptr) {
        return n_ptr->topological_index() < m_ptr->topological_index();
    });

    // Note: this was incorporated in the propagate method before, after calling each node propagate
    // method. This could cause a small performance regression.
    for (const Node* node_ptr : sources) {
        state[node_ptr->topological_index()]->mark = false;
    }

    return sources;
}

void Graph::propagate(State& state) const {
    std::ranges::for_each(nodes(), [&state](const auto& ptr) { ptr->propagate(state); });
}

void Graph::propagate(State& state, std::span<const Node*> queue_to_update) const {
    for (const Node* node_ptr : queue_to_update) {
        node_ptr->propagate(state);

        // Note: commented because we split the methods.
        // We might incur in a small performance degradation.
        // state[node_ptr->topological_index()]->mark = false;
    }
}

void Graph::propagate(State& state, std::vector<const Node*>&& changed) const {
    return propagate(state, std::span(changed));
}

void Graph::commit(State& state) const {
    std::ranges::for_each(nodes(), [&state](const auto& ptr) { ptr->commit(state); });
}

void Graph::commit(State& state, std::span<const Node*> changed) const {
    std::for_each(
            // std::execution::par_unseq,  // todo: test performance. Might help!
            changed.begin(), changed.end(), [&state](const Node* n_ptr) { n_ptr->commit(state); });

    assert(([&state, &changed]() -> bool {
        for (const Node* n_ptr : changed) {
            if (dynamic_cast<const Array*>(n_ptr) == nullptr) continue;
            if (!dynamic_cast<const Array*>(n_ptr)->diff(state).empty()) return false;
        }
        return true;
    })());
}

void Graph::commit(State& state, std::vector<const Node*>&& changed) const {
    commit(state, std::span(changed));
}

void Graph::revert(State& state) const {
    std::ranges::for_each(nodes(), [&state](const auto& ptr) { ptr->revert(state); });
}

void Graph::revert(State& state, std::span<const Node*> changed) const {
    std::for_each(
            // std::execution::par_unseq,  // todo: test performance. Might help!
            changed.begin(), changed.end(), [&state](const Node* n_ptr) { n_ptr->revert(state); });

    assert(([&state, &changed]() -> bool {
        for (const Node* n_ptr : changed) {
            if (dynamic_cast<const Array*>(n_ptr) == nullptr) continue;
            if (!dynamic_cast<const Array*>(n_ptr)->diff(state).empty()) return false;
        }
        return true;
    })());
}

void Graph::revert(State& state, std::vector<const Node*>&& changed) const {
    revert(state, std::span(changed));
}

// Note: we pass the vector of changed nodes by value as we expect it to be rather small. Revisit if
// it becomes too expensive.
void Graph::propose(State& state, std::vector<const Node*> sources,
                    std::function<bool(const Graph&, State&)> accept) const {
    // Perform BFS to mark the nodes to update
    auto changed = descendants(state, std::move(sources));

    // Propagate the updates
    propagate(state, changed);

    // Check the acceptance function for whether to commit the changes or not.
    if (accept(*this, state)) {
        commit(state, changed);
    } else {
        revert(state, changed);
    }
}

ssize_t Graph::remove_unused_nodes(bool ignore_listeners) {
    // Establish a topological ordering. We'll use the fact that the node list
    // is ordered, but we're going to mess with the topological indices!
    topological_sort();

    // We'll use this later to calculate how many nodes we removed
    const ssize_t num_nodes_before = this->num_nodes();

    // We generally want to avoid touching decisions
    const ssize_t num_decisions = this->num_decisions();

    // Mark the nodes that we plan to keep regardless of their number of successors
    // For performance we store a signalling value in the topological_index of the
    // nodes. We need to take some care to not override the topological index of
    // any decisions.
    constexpr ssize_t keep = -2;
    {
        // all constraints are kept
        for (ArrayNode* ptr : constraints_) {
            if (ptr->topological_index_ < num_decisions) continue;  // we'll always keep these
            ptr->topological_index_ = keep;
        }

        // as is the objective if it exists
        if (objective_ptr_ && objective_ptr_->topological_index_ >= num_decisions) {
            objective_ptr_->topological_index_ = keep;
        }

        // If we're not ignoring listeners, we check if anyone is "listening" to
        // the expired_ptr_. If so, we keep the node.
        if (!ignore_listeners) {
            for (auto& ptr : nodes_ | std::views::drop(num_decisions)) {
                if (ptr->expired_ptr_.use_count() > 1) ptr->topological_index_ = keep;
            }
        }
    }

    // Now walk backwards through the topologically sorted node list
    // removing any nodes with no successors that we haven't marked.
    for (auto& ptr : nodes_ | std::views::drop(num_decisions) | std::views::reverse) {
        if (!ptr->removable()) continue;                // some nodes can never be removed
        if (ptr->topological_index_ == keep) continue;  // we marked these to keep
        if (ptr->successors().size() > 0) continue;     // this node is used by other nodes

        // Remove ptr from its predecessor's successor vectors.
        // This very temporarily leaves the node in an invalid state, which is
        // why we do it here rather than via a method.
        for (auto pred_ptr : ptr->predecessors_) {
            std::erase_if(pred_ptr->successors_,
                          [&ptr](const Node::SuccessorView& sv) { return sv.ptr == ptr.get(); });
        }

        // now delete the node by clearing the unique_ptr
        ptr.reset();
    }

    // Traverse the nodes_ one last time removing any nullptrs we created
    std::erase_if(nodes_, [](const auto& ptr) { return !ptr; });

    // Undo all of the weird stuff we did with the topological indices
    reset_topological_sort();

    // Finally return the number of nodes that have been removed!
    return num_nodes_before - this->num_nodes();
}

// Node ***********************************************************************

void Node::initialize_state(State& state) const {
    assert(topological_index_ >= 0 && "must be topologically sorted");
    assert(static_cast<int>(state.size()) > topological_index_ && "unexpected state length");
    assert(state[topological_index_] == nullptr && "already initialized state");

    state[topological_index_] = std::make_unique<NodeStateData>();
}

// We try to avoid needing to override this method in every class by doing
// some compiler-specific stuff.
std::string Node::classname() const {
    // get the compiler-specific name
    const char* compiler_name = typeid(*this).name();

#if defined(_HAS_CXXABI)
    // if cxxabi.h is available, we can demangle
    std::string name;

    // __cxa_demangle requires us to manage the lifespan of the returned pointer
    // so we do this a bit carefully
    {
        int status = 0;
        std::size_t size = 0;
        char* demangled = abi::__cxa_demangle(compiler_name, NULL, &size, &status);
        assert(status == 0);  // these are type names so this should always work
        if (demangled != nullptr) {
            name = std::string(demangled);
            free(demangled);
        } else {
            // if the status assert above passed we should never get here, but just in case
            name = std::string(compiler_name);
        }
    }
#else
    // otherwise just use the compiler name as-is.
    std::string name(compiler_name);
#endif

    // try to strip dwave::optimization:: from the beginning if it's there
    static const std::string namespace_name = "dwave::optimization::";
    if (const auto found = name.find(namespace_name); found != std::string::npos) {
        name.replace(found, namespace_name.size(), "");
    }

    return name;
}

std::string Node::repr() const {
    std::ostringstream oss;  // we use this for os-specific pointer formatting

    // by default we do a Python-style repr print
    oss << "<" << classname() << " at " << this;

    // if we're topologically sorted, we print that too
    if (topological_index_ >= 0) oss << ", topological_index=" << topological_index_;

    oss << ">";
    return oss.str();
}

std::string Node::str() const { return classname(); }

std::ostream& operator<<(std::ostream& os, const Node& node) { return os << node.repr(); }

// DecisionNode ***************************************************************

[[noreturn]] void DecisionNode::update(State& state, int index) const {
    throw std::logic_error("update() called on a decisison variable");
}

}  // namespace dwave::optimization

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
#include <chrono>
#include <deque>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <utility>

#if defined(__has_include) and __has_include(<cxxabi.h>)
#define _HAS_CXXABI
#include <cxxabi.h>
#endif

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/nodes/collections.hpp"
#include "dwave-optimization/nodes/constants.hpp"
#include "dwave-optimization/nodes/inputs.hpp"

namespace dwave::optimization {

void Graph::add_constraint(ArrayNode* constraint_ptr) {
    if (not constraint_ptr->logical()) {
        throw std::invalid_argument("constraint must have a logical output");
    }
    // todo: we could substitute an AND on the user's behalf, or we could allow
    // multidimensional constraints as a way of supporting many constraints at once
    if (constraint_ptr->size() != 1) {
        throw std::invalid_argument(
            "The truth value of an array with more than one element is ambiguous"
        );
    }

    constraints_.emplace_back(constraint_ptr);
}

void Graph::commit(State& state) const {
    std::ranges::for_each(nodes(), [&state](const auto& ptr) { ptr->commit(state); });
}

void Graph::commit(State& state, std::span<const Node*> changed) const {
    std::for_each(
        // std::execution::par_unseq,  // todo: test performance. Might help!
        changed.begin(),
        changed.end(),
        [&state](const Node* n_ptr) { n_ptr->commit(state); }
    );

    assert(([&state, &changed]() -> bool {
        for (const Node* n_ptr : changed) {
            if (dynamic_cast<const Array*>(n_ptr) == nullptr) continue;
            if (not dynamic_cast<const Array*>(n_ptr)->diff(state).empty()) return false;
        }
        return true;
    })());
}

void Graph::commit(State& state, std::vector<const Node*>&& changed) const {
    commit(state, std::span(changed));
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

std::vector<const Node*> Graph::descendants(std::vector<const Node*> sources) const {
    State state;
    for (ssize_t i = 0, stop = num_nodes(); i < stop; ++i) {
        state.emplace_back(std::make_unique<NodeStateData>());
    }
    return descendants(state, sources);
}

State Graph::empty_state() const { return State(num_nodes()); }

State Graph::empty_state() {
    topological_sort();
    return static_cast<const Graph*>(this)->empty_state();
}

double Graph::energy(const State& state) const {
    if (objective_ptr_ == nullptr) return 0;
    return objective_ptr_->view(state).front();
}
bool Graph::feasible(const State& state) const {
    for (const Array* ptr : constraints_) {
        assert(ptr->size(state) == 1);
        if (not(ptr->view(state)[0])) return false;
    }
    return true;
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
    assert(static_cast<int>(state.size()) == num_nodes() and "unexpected state length");
    assert(topologically_sorted_ and "graph must be topologically sorted");

    for (int i = 0, end = num_nodes(); i < end; ++i) {
        if (state[i]) continue;  // should this clear any pending changes?

        nodes_[i]->initialize_state(state);
    }
}

void Graph::initialize_state(State& state) {
    topological_sort();
    static_cast<const Graph*>(this)->initialize_state(state);
}

std::span<const DecisionNode*> Graph::mutated(State& state) const {
    state.mutated_nodes_.clear();
    for (const DecisionNode* dec_ptr : decisions()) {
        if (const ArrayNode* arr_ptr = dynamic_cast<const ArrayNode*>(dec_ptr); arr_ptr) {
            if (not arr_ptr->diff(state).empty()) {
                state.mutated_nodes_.push_back(dec_ptr);
            }
        } else if (
            dynamic_cast<const DisjointListsNode*>(dec_ptr) or
            dynamic_cast<const DisjointBitSetsNode*>(dec_ptr)
        ) {
            for (const Node* suc_ptr : dec_ptr->successors()) {
                const ArrayNode* arr_ptr = dynamic_cast<const ArrayNode*>(suc_ptr);
                assert(arr_ptr and "all successors should be array nodes");
                if (not arr_ptr->diff(state).empty()) {
                    state.mutated_nodes_.push_back(dec_ptr);
                    break;
                }
            }
        } else {
            assert(false and "unknown decision node type");
            unreachable();
        }
    }

    return state.mutated_nodes_;
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

// Note: we pass the vector of changed nodes by value as we expect it to be rather small. Revisit if
// it becomes too expensive.
void Graph::propose(
    State& state,
    std::vector<const Node*> sources,
    std::function<bool(const Graph&, State&)> accept
) const {
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
    if (not state[index]) return;

    // Otherwise, reset our own state and then all of our successors
    state[index].reset();

    for (const dwave::optimization::Node* successor_ptr : ptr->successors()) {
        if (successor_ptr->topological_index() < 0) continue;  // nothing to reset
        recursive_reset(state, successor_ptr);
    }
}

ssize_t Graph::remove_redundant_nodes(bool ignore_listeners, double time_limit_s) {
    if (topologically_sorted_) throw std::logic_error("cannot remove nodes from a locked model");

    // This function can get quite expensive, so we have a timeout we'll try to respect.
    const auto stop_time =
        std::chrono::steady_clock::now() + std::chrono::duration<double>(time_limit_s);
    auto out_of_time = [&stop_time]() -> bool {
        return std::chrono::steady_clock::now() >= stop_time;
    };

    // We need to know how many nodes we started with in order to know how many we removed
    const ssize_t num_nodes = this->num_nodes();

    // We want the nodes in topological order, and we will respect that order
    // while we're transferring successors.
    // We will, however, abuse the topological_index_ to store information
    topological_sort();

    // We'll set the topological index to one of these values to track what we've
    // already done.
    // They are arbitrary values, but we steer away from -1 because that's used to
    // indicate unsorted
    constexpr ssize_t seen = -2;  // already checked for duplicates
    constexpr ssize_t drop = -3;  // marked for deletion

    // The actions that we always want to do before returning, whether it's because
    // we hit the time limit or because we have no more work to do.
    auto cleanup = [&]() -> ssize_t {
        // we should only ever have swapped the objective_ptr
        assert(objective_ptr_ == nullptr or objective_ptr_->topological_index() != drop);

        // Whether a node was dropped
        auto dropped = [&drop, &ignore_listeners](const auto& ptr) {
            if (not ignore_listeners and ptr->num_listeners() > 0) return false;
            assert(ptr->topological_index_ != drop or ptr->successors_.empty());
            return ptr->topological_index() == drop;
        };

        // Go through our "special" nodes and drop anything
        assert(std::ranges::none_of(decisions_, dropped));  // should never be dropped
        assert(std::ranges::none_of(inputs_, dropped));  // should never be dropped
        std::erase_if(constants_, dropped);

        std::erase_if(constraints_, dropped);
        assert(objective_ptr_ == nullptr or objective_ptr_->topological_index_ != drop);

        // This is the step that actually deallocates the node
        for (auto& uptr : nodes_) {
            if (not dropped(uptr)) continue;

            // Remove the node from its predecessor's successor vectors.
            // This leaves the node in an invalid state, very briefly
            for (auto* pred_ptr : uptr->predecessors_) {
                [[maybe_unused]] ssize_t num_removed = pred_ptr->remove_successor_(uptr.get());
                assert(num_removed > 0);
            }

            // And then reset the pointer, thereby dellocating the node
            uptr.reset();

        }
        // Finally, remove the nullptrs from the nodelist
        std::erase_if(nodes_, [](const auto& uptr) { return not uptr; });

        // Reset the topological index of everything that's left so we clean up
        // everything
        reset_topological_sort();

        // Finally, report the number of nodes we dropped
        return num_nodes - this->num_nodes();
    };

    // Given two equal nodes, transfer successors and the objective marker
    // from `from_ptr` to `to_ptr`. This does not fix the constraints, we handle
    // that as part of `cleanup()`.
    auto transfer = [&](Node* from_ptr, Node* to_ptr) -> void {
        assert(from_ptr->topological_index_ > to_ptr->topological_index_);

        // Transfer the successors
        to_ptr->take_successors(*from_ptr);

        // Mark from for dropping
        from_ptr->topological_index_ = drop;

        // We also want to fix the objective if relevant
        if (objective_ptr_ != nullptr and static_cast<Node*>(objective_ptr_) == from_ptr) {
            objective_ptr_ = dynamic_cast<ArrayNode*>(to_ptr);
            assert(objective_ptr_ != nullptr);
            assert(objective_ptr_->size() == 1);
        }
    };

    // Ok, all that setup done, now let's start checking for redundancy.

    // Decisions are never redundant, so we skip over them

    // Nor are inputs, so likewise we skip them

    // The first set of nodes we're worried about are the constants - the only
    // class of root node that can have redundancy
    for (ssize_t i = 0, num_constants = constants_.size(); i < num_constants; ++i) {
        if (constants_[i]->topological_index_ == drop) continue;

        for (ssize_t j = i + 1; j < num_constants; ++j) {
            if (constants_[j]->topological_index_ == drop) continue;

            // the topological indices of our constants should not yet be touched
            // and because they are added in order, they should always be ascending
            assert(constants_[i]->topological_index_ >= 0);
            assert(constants_[j]->topological_index_ >= 0);
            assert(constants_[i]->topological_index_ < constants_[j]->topological_index_);

            // Check against or current time limit before doing the potentially
            // expensive (O(buffer_size)) equality check.
            if (out_of_time()) return cleanup();

            // nothing to do if they are not equal
            if (not constants_[i]->equal_to(*constants_[j])) continue;

            transfer(constants_[j], constants_[i]);
        }

        constants_[i]->topological_index_ = seen;
    }

    // For each node, we check pairwise among all of its successors.
    // This is because for nodes to be equal, they *must* have the same set of
    // predecessors.
    for (auto& uptr : nodes_) {
        auto& successors = uptr->successors();

        for (ssize_t i = 0, num_successors = successors.size(); i < num_successors; ++i) {
            // We've already seen or dropped this node
            if (successors[i]->topological_index_ < 0) continue;

            for (ssize_t j = i + 1; j < num_successors; ++j) {
                // We've already seen or dropped this node
                if (successors[j]->topological_index_ < 0) continue;

                // A node cannot be redundant with itself
                if (successors[i] == successors[j]) continue;

                // Check against or current time limit before doing the potentially
                // expensive (O(num_nodes^2)) equality check.
                if (out_of_time()) return cleanup();

                // If lhs != rhs there's nothing to do, so keep looking
                if (not successors[i]->equal_to(*successors[j])) continue;

                // We have a redundant node!

                // We want to transfer to the node with the lower topological order
                if (successors[i]->topological_index_ < successors[j]->topological_index_) {
                    transfer(successors[j], successors[i]);
                } else {
                    transfer(successors[i], successors[j]);
                    break;  // stop comparing i to other nodes because we dropped it
                }
            }

            // Ok, we've checked i against everything, so if we didn't drop it
            // we can mark it as seen
            if (successors[i]->topological_index_ >= 0) successors[i]->topological_index_ = seen;
        }
    }

    return cleanup();
}

ssize_t Graph::remove_unused_nodes(bool ignore_listeners) {
    if (topologically_sorted_) throw std::logic_error("cannot remove nodes from a locked model");

    // Establish a topological ordering. We'll use the fact that the node list
    // is ordered, but we're going to mess with the topological indices!
    topological_sort();

    // Specifically we'll be marking the nodes with either
    constexpr ssize_t keep = -2;  // always keep
    constexpr ssize_t drop = -3;  // to be dropped

    // We'll use this later to calculate how many nodes we removed
    const ssize_t num_nodes = this->num_nodes();

    // Mark the nodes that we plan to keep regardless of their number of successors
    // For performance we store a signalling value in the topological_index of the
    // nodes.

    // First up our roots - specifically the decisions are always kept.
    for (auto* ptr : decisions_) ptr->topological_index_ = keep;

    // Constants and Inputs are allowed to be removed

    // Next up our important leaves.
    for (auto* ptr : constraints_) ptr->topological_index_ = keep;
    if (objective_ptr_) objective_ptr_->topological_index_ = keep;

    // Some nodes are just not allowed to be removed for other reasons
    for (auto& uptr : nodes_) {
        if (not uptr->removable_()) uptr->topological_index_ = keep;
    }

    // Also, any nodes that have a listener, that is a node is holding a Node::expired_ptr()
    if (not ignore_listeners) {
        for (auto& uptr : nodes_) {
            if (uptr->num_listeners() > 0) uptr->topological_index_ = keep;
        }
    }

    // Having established what we're keeping, time to mark stuff for removal!

    for (auto& uptr : nodes_ | std::views::reverse) {
        if (uptr->topological_index_ == keep) continue;  // we marked these to keep
        if (uptr->successors().size() > 0) continue;     // this node is used by other nodes

        // We have a node with no successors and that we haven't marked it as important.
        // So let's mark it to be dropped later.

        // Remove the node from its predecessor's successor vectors.
        // This leaves the node in an invalid state, until we delete it later.
        for (auto* pred_ptr : uptr->predecessors_) {
            [[maybe_unused]] ssize_t num_removed = pred_ptr->remove_successor_(uptr.get());
            assert(num_removed > 0);
        }

        uptr->topological_index_ = drop;
    }

    // Now let's start actually dropping stuff
    auto dropped = [&drop](const auto& ptr) { return ptr->topological_index_ == drop; };

    assert(std::ranges::none_of(decisions_, dropped));
    std::erase_if(inputs_, dropped);
    std::erase_if(constants_, dropped);

    assert(objective_ptr_ == nullptr or objective_ptr_->topological_index_ != drop);
    assert(std::ranges::none_of(constraints_, dropped));

    // Traverse the nodes_ one last time, this is what actualy deallocates the nodes.
    std::erase_if(nodes_, dropped);

    // Let's fix the topological indices by first fixing the decision topological
    // indices that we overwrote then calling reset_topological_sort() to fix
    // everything else.
    {
        int index = 0;
        for (auto* ptr : decisions_) ptr->topological_index_ = index++;
    }
    reset_topological_sort();

    // Finally return the number of nodes that have been removed
    return num_nodes - this->num_nodes();
}

void Graph::reset_topological_sort() {
    if (not topologically_sorted_) return;  // already unsorted

    // The decisions must have a consistent topological index, so we don't reset them
    std::for_each(
        // std::execution::par_unseq,  // todo: performance testing. Probably won't help
        nodes_.begin() + num_decisions(),
        nodes_.end(),
        [](const std::unique_ptr<Node>& n_ptr) {
            assert(dynamic_cast<Decision*>(n_ptr.get()) == nullptr);  // not a decision
            n_ptr->topological_index_ = -1;
        }
    );

    topologically_sorted_ = false;
}

void Graph::revert(State& state) const {
    std::ranges::for_each(nodes(), [&state](const auto& ptr) { ptr->revert(state); });
}

void Graph::revert(State& state, std::span<const Node*> changed) const {
    std::for_each(
        // std::execution::par_unseq,  // todo: test performance. Might help!
        changed.begin(),
        changed.end(),
        [&state](const Node* n_ptr) { n_ptr->revert(state); }
    );

    assert(([&state, &changed]() -> bool {
        for (const Node* n_ptr : changed) {
            if (dynamic_cast<const Array*>(n_ptr) == nullptr) continue;
            if (not dynamic_cast<const Array*>(n_ptr)->diff(state).empty()) return false;
        }
        return true;
    })());
}

void Graph::revert(State& state, std::vector<const Node*>&& changed) const {
    revert(state, std::span(changed));
}

void Graph::set_objective(ArrayNode* objective_ptr) {
    // nullptr is an unset objective, so we allow it.
    if (objective_ptr != nullptr and objective_ptr->size() != 1) {
        throw std::invalid_argument("objective must have a single output");
    }
    this->objective_ptr_ = objective_ptr;
}

void Graph::topological_sort() {
    if (topologically_sorted_) return;

    // Check that all of our non-decisions have a negative label
    assert(std::ranges::all_of(nodes_, [](const auto& uptr) {
        if (dynamic_cast<DecisionNode*>(uptr.get())) {
            // A decision node
            return uptr->topological_index_ >= 0;  // specific values checked later
        } else {
            // Not a decision
            return uptr->topological_index_ < 0;
        }
    }));

    // Assign the topological order of our root nodes. Because decisions are first
    // they always get the same topological ordering and have already had that order
    // assigned.
    {
        // Double check the decisions are what we expect.
        assert([&]() -> bool {
            int index = 0;
            for (auto* ptr : decisions_) {
                if (ptr->topological_index_ != index++) return false;
            }
            return true;
        }());

        int index = decisions_.size();  // the decisions are already ordered
        for (auto* ptr : inputs_) ptr->topological_index_ = index++;
        for (auto* ptr : constants_) ptr->topological_index_ = index++;
    }

    // Find a topological ordering of all of the non-decisions
    // The extra `visit` parameter is to allow for recursive lambdas
    // We iterate over successors in reverse order to preserve their original ordering
    // after sorting based on the topological index (which is counting down during DFS).
    auto visit = [](Node* n_ptr, int* count_ptr, auto&& visit) -> void {
        if (n_ptr->topological_index_ >= 0) return;
        if (n_ptr->topological_index_ == -2) throw std::logic_error("has cycles");

        // decisions should already be sorted
        assert(dynamic_cast<Decision*>(n_ptr) == nullptr and "unsorted decisions node");

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
    assert(
        count == static_cast<int>(decisions_.size()) + static_cast<int>(inputs_.size()) +
                     static_cast<int>(constants_.size()) - 1
    );

    // for later convenience, we sort the nodes_ by index
    // This moves all of the decisions to the front
    std::sort(
        nodes_.begin(),
        nodes_.end(),
        [](const std::unique_ptr<Node>& n_ptr, const std::unique_ptr<Node>& m_ptr) {
            return n_ptr->topological_index_ < m_ptr->topological_index_;
        }
    );

    topologically_sorted_ = true;
}

void Node::initialize_state(State& state) const {
    assert(topological_index_ >= 0 and "must be topologically sorted");
    assert(static_cast<int>(state.size()) > topological_index_ and "unexpected state length");
    assert(state[topological_index_] == nullptr and "already initialized state");

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

ssize_t Node::remove_successor_(const Node* ptr) {
    ssize_t before = successors_.size();
    std::erase_if(successors_, [&ptr](const SuccessorView& sv) { return sv.ptr == ptr; });
    return before - successors_.size();
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

void Node::take_successors(Node& from) {
    assert(this != &from and "a node cannot take successors from itself");

    for (const auto& sv : from.successors_) {
        sv.ptr->replace_predecessor_(sv.index, this);
        this->successors_.emplace_back(sv);
    }
    from.successors_.clear();
}

void Node::replace_predecessor_(ssize_t previous_index, Node* node_ptr) {
    assert(0 <= previous_index);
    assert(static_cast<size_t>(previous_index) < predecessors_.size());
    predecessors_[previous_index] = node_ptr;
}

std::ostream& operator<<(std::ostream& os, const Node& node) { return os << node.repr(); }

[[noreturn]] void DecisionNode::update(State& state, int index) const {
    throw std::logic_error("update() called on a decisison variable");
}

}  // namespace dwave::optimization

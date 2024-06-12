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

#include <span>
#include <vector>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/graph.hpp"

namespace dwave::optimization {

/// A base class for various collections. Cannot be used directly - it has no
/// public constructors.
/// Subclasses must implement an overload of Node::initialize_state() and
/// Decision::default_move()
class CollectionNode : public Node, public ArrayOutputMixin<Array>, public Decision {
 public:
    CollectionNode() = delete;

    // Set the node's initial state explicitly
    void initialize_state(State& state, std::vector<double> contents) const;
    using Node::initialize_state;  // inherit the default overload

    // Overloads needed by the Array ABC **************************************

    double const* buff(const State& state) const override;
    std::span<const Update> diff(const State& state) const override;

    constexpr bool integral() const override { return true; }

    double min() const noexcept override { return 0; }
    double max() const noexcept override { return max_value_ - 1; }

    using ArrayOutputMixin::size;  // for size()
    ssize_t size(const State& state) const override;

    using ArrayOutputMixin::shape;  // for shape()
    std::span<const ssize_t> shape(const State& state) const override;

    ssize_t size_diff(const State& state) const override;

    // Overloads required by the Node ABC *************************************

    void commit(State&) const override;
    void revert(State&) const override;

    void update[[noreturn]](State&, int) const override {
        throw std::logic_error("update() called on a decisison variable");
    }

    // Exchange the values in the list at index i and index j
    // Note that for variable-length collections these indices might not
    // be in the "visible" range.
    void exchange(State& state, ssize_t i, ssize_t j) const;

    // Rotate the elements between src_idx and dest_idx with dest_idx getting the value of src_idx.
    // Equivalent change through exchange operation will be more computationally expensive.
    void rotate(State& state, ssize_t dest_idx, ssize_t src_idx) const;

    // Grow the size of the collection by one
    void grow(State& state) const;

    // Shrink the size of the collection by one
    void shrink(State& state) const;

    // Information about the size of the collection.
    SizeInfo sizeinfo() const override;

 protected:
    CollectionNode(ssize_t max_value, ssize_t min_size, ssize_t max_size)
            : Node(),
              ArrayOutputMixin((min_size == max_size) ? max_size : Array::DYNAMIC_SIZE),
              Decision(),
              max_value_(max_value),
              min_size_(min_size),
              max_size_(max_size) {
        if (min_size < 0 || max_size < 0) {
            throw std::invalid_argument("a collection cannot contain fewer than 0 elements");
        }
        if (min_size > max_size) {
            throw std::invalid_argument("min_size cannot be greater than max_size");
        }
        if (max_size > max_value) {
            throw std::invalid_argument("a collection cannot be larger than its maximum value");
        }
    }

    // The collection draws its values from [0, max_value_)
    const ssize_t max_value_;

    // min_size_ <= |collection| <= max_size_
    const ssize_t min_size_;
    const ssize_t max_size_;
};

// A disjoint set node, implemented with a bit set for each set. The bit sets are
// actually just arrays of doubles, such that the successor nodes can be normal binary
// Array output nodes.
// After adding this node to the DAG, you must add `num_disjoint_sets` more
// successor nodes of type `DisjointBitSetNode`, which is a special node meant to
// output its respective disjoint set as an array.
class DisjointBitSetsNode : public Node, public Decision {
 public:
    // `primary_set_size` is the size of the primary set that the node will partition,
    // i.e. the set `range(primary_set_size)`.
    DisjointBitSetsNode(ssize_t primary_set_size, ssize_t num_disjoint_sets);

    void initialize_state(State& state) const override;

    // Set the node's initial state explicitly
    void initialize_state(State& state, const std::vector<std::vector<double>>& contents) const;

    // Overloads required by the Node ABC *************************************

    void commit(State&) const override;
    void revert(State&) const override;

    void update[[noreturn]](State&, int) const override {
        throw std::logic_error("update() called on a decisison variable");
    }

    // Overloads required by the Decision ABC

    void default_move(State& state, RngAdaptor& rng) const override;

    // Disjoint-Bitset-specific methods ********************************************

    void swap_between_sets(State& state, ssize_t from_disjoint_set, ssize_t to_disjoint_set,
                           ssize_t element_i) const;

    ssize_t get_containing_set_index(State& state, ssize_t element_i) const;

    ssize_t primary_set_size() const { return primary_set_size_; }

    ssize_t num_disjoint_sets() const { return num_disjoint_sets_; }

 protected:
    const ssize_t primary_set_size_;
    const ssize_t num_disjoint_sets_;
};

// Successor node for the output of `DisjointBitSetsNode`
class DisjointBitSetNode : public Node, public ArrayOutputMixin<Array> {
 public:
    explicit DisjointBitSetNode(DisjointBitSetsNode* disjoint_bit_sets_node)
            : Node(),
              ArrayOutputMixin(disjoint_bit_sets_node->primary_set_size()),
              disjoint_bit_sets_node(disjoint_bit_sets_node),
              set_index_(disjoint_bit_sets_node->successors().size()),
              primary_set_size_(disjoint_bit_sets_node->primary_set_size()) {
        if (set_index_ >= disjoint_bit_sets_node->num_disjoint_sets()) {
            throw std::length_error("disjoint-bit-set node already has all output nodes");
        }
        add_predecessor(disjoint_bit_sets_node);
    }

    // Overloads needed by the Array ABC **************************************

    double const* buff(const State&) const override;
    std::span<const Update> diff(const State& state) const override;

    constexpr bool integral() const override { return true; };

    double min() const noexcept override { return 0; };
    double max() const noexcept override { return 1; };

    // Overloads required by the Node ABC *************************************

    void commit(State&) const override{};
    void revert(State&) const override{};
    void update(State&, int) const override{};

    ssize_t set_index() const noexcept { return set_index_; };

 protected:
    const DisjointBitSetsNode* disjoint_bit_sets_node;
    const ssize_t set_index_;
    const ssize_t primary_set_size_;
};

// A "disjoint lists" node, i.e. a set of disjoint sets that also maintain an item
// ordering. After adding this node to the DAG, you must add `num_disjoint_lists` more
// successor nodes of type `DisjointListNode`, which is a special node meant to
// output its respective disjoint list as an array.
class DisjointListsNode : public Node, public Decision {
 public:
    // `primary_set_size` is the size of the primary set that the node will partition,
    // i.e. the set `range(primary_set_size)`.
    DisjointListsNode(ssize_t primary_set_size, ssize_t num_disjoint_lists);

    void initialize_state(State& state) const override;

    // Set the node's initial state explicitly
    void initialize_state(State& state, std::vector<std::vector<double>> contents) const;

    // Overloads required by the Node ABC *************************************

    void commit(State&) const override;
    void revert(State&) const override;

    void update[[noreturn]](State&, int) const override {
        throw std::logic_error("update() called on a decisison variable");
    }

    // Overloads required by the Decision ABC

    void default_move(State& state, RngAdaptor& rng) const override;

    // Disjoint-list-specific methods ********************************************
    ssize_t get_disjoint_list_size(State& state, ssize_t list_index) const;

    void swap_in_list(State& state, ssize_t disjoint_list, ssize_t element_i,
                      ssize_t element_j) const;

    void pop_to_list(State& state, ssize_t from_disjoint_list, ssize_t element_i,
                     ssize_t to_disjoint_list, ssize_t element_j) const;

    ssize_t num_disjoint_lists() const { return num_disjoint_lists_; }
    ssize_t primary_set_size() const { return primary_set_size_; }

 protected:
    const ssize_t primary_set_size_;
    const ssize_t num_disjoint_lists_;
};

// Successor node for the output of `DisjointListsNode`
class DisjointListNode : public Node, public ArrayOutputMixin<Array> {
 public:
    explicit DisjointListNode(DisjointListsNode* disjoint_list_node);

    // Overloads needed by the Array ABC **************************************

    double const* buff(const State& state) const override;
    std::span<const Update> diff(const State& state) const override;

    constexpr bool integral() const override { return true; }

    double min() const noexcept override { return 0; }
    double max() const noexcept override { return primary_set_size_ - 1; }

    using ArrayOutputMixin::size;  // for size()
    ssize_t size(const State& state) const override;

    using ArrayOutputMixin::shape;  // for shape()
    std::span<const ssize_t> shape(const State& state) const override;

    // Information about the size of the collection.
    SizeInfo sizeinfo() const override;

    ssize_t size_diff(const State& state) const override;

    // Overloads required by the Node ABC *************************************

    void commit(State&) const override{};
    void revert(State&) const override{};
    void update(State&, int) const override{};

    ssize_t list_index() const noexcept { return list_index_; };

 protected:
    const DisjointListsNode* disjoint_list_node_ptr;
    const ssize_t list_index_;
    const ssize_t primary_set_size_;
};

// A list node is an ordered collection of unique integers.
class ListNode : public CollectionNode {
 public:
    // Create a ListNode that is always a permutation of range(n)
    explicit ListNode(ssize_t n) : CollectionNode(n, n, n) {}
    explicit ListNode(ssize_t n, ssize_t min_size, ssize_t max_size)
            : CollectionNode(n, min_size, max_size) {}

    void default_move(State& state, RngAdaptor& rng) const override;

    // A ListNode's initial state defaults to range(n)
    void initialize_state(State& state) const override;
    using CollectionNode::initialize_state;  // for explicit initialization
};

// A set node is an unordered subset of range(n)
// Note that "unordered" means that order is considered meaningless, the state
// itself is still an array and therefore has an order.
class SetNode : public CollectionNode {
 public:
    /// Create a set node with elements drawn from range(n)
    explicit SetNode(ssize_t n) : CollectionNode(n, 0, n) {}

    explicit SetNode(ssize_t n, ssize_t min_size, ssize_t max_size)
            : CollectionNode(n, min_size, max_size) {}

    void default_move(State& state, RngAdaptor& rng) const override;

    // A SetNode's default initial state is empty
    void initialize_state(State& state) const override;
    using CollectionNode::initialize_state;  // for explicit initialization
};

}  // namespace dwave::optimization

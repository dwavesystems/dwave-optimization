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
#include "dwave-optimization/nodes/collections.hpp"

#include <ranges>
#include <unordered_set>
#include <utility>

namespace dwave::optimization {

struct CollectionStateData : NodeStateData {
    explicit CollectionStateData(ssize_t n) : CollectionStateData(n, n) {}

    CollectionStateData(ssize_t n, ssize_t size) : size(size) {
        assert(size <= n);
        for (ssize_t i = 0; i < n; ++i) {
            elements.push_back(i);
        }
    }

    explicit CollectionStateData(std::vector<double> elements)
            : elements(std::move(elements)), size(this->elements.size()) {}

    CollectionStateData(std::vector<double> elements, ssize_t size)
            : elements(std::move(elements)), size(size) {}

    void commit() {
        previous.clear();
        all_updates.clear();
        previous_size = size;
    }

    void exchange(ssize_t i, ssize_t j) {
        assert(i >= 0 && static_cast<std::size_t>(i) < elements.size());
        assert(j >= 0 && static_cast<std::size_t>(j) < elements.size());

        std::swap(elements[i], elements[j]);

        all_updates.emplace_back(i, elements[j], elements[i]);
        all_updates.emplace_back(j, elements[i], elements[j]);

        // only track changes within the visible part of the array
        if (i < size && j < size) {
            previous.emplace_back(i, elements[j], elements[i]);
            previous.emplace_back(j, elements[i], elements[j]);
        } else if (j < size) {
            previous.emplace_back(j, elements[i], elements[j]);
        } else if (i < size) {
            previous.emplace_back(i, elements[j], elements[i]);
        }  // if neither are visible we don't need to track it
    }

    void rotate(ssize_t dest_idx, ssize_t src_idx) {
        // We want the rotate function to be called only on the visible part of the list.
        assert(src_idx >= 0 && src_idx < size);
        assert(dest_idx >= 0 && dest_idx < size);

        // Elements move from left to right.
        if (src_idx > dest_idx) {
            auto prev = elements[src_idx];
            for (ssize_t i = dest_idx; i <= src_idx; i++) {
                std::swap(elements[i], prev);
                all_updates.emplace_back(i, prev, elements[i]);
                previous.emplace_back(i, prev, elements[i]);
            }
        } else if (src_idx < dest_idx) {
            auto prev = elements[src_idx];
            for (ssize_t i = dest_idx; i >= src_idx; i--) {
                std::swap(elements[i], prev);
                all_updates.emplace_back(i, prev, elements[i]);
                previous.emplace_back(i, prev, elements[i]);
            }
        }
    }

    void grow() {
        assert(size < static_cast<ssize_t>(elements.size()));
        previous.emplace_back(Update::placement(size, elements[size]));
        ++size;
    }

    void revert() {
        // Un-apply any changes by working backwards through all updates.
        // If we end up enforcing previous being sorted and unique later then
        // we could do this any order (or better in parallel).

        for (const Update& update : all_updates | std::views::reverse) {
            elements[update.index] = update.old;
        }

        previous.clear();
        all_updates.clear();
        size = previous_size;
    }

    void shrink() {
        assert(size > 0);
        --size;
        previous.emplace_back(Update::removal(size, elements[size]));
    }

    std::unique_ptr<NodeStateData> copy() const override {
        return std::make_unique<CollectionStateData>(*this);
    }

    std::vector<double> elements;
    std::vector<Update> previous;
    std::vector<Update> all_updates;
    ssize_t size;
    ssize_t previous_size = size;
};

void CollectionNode::commit(State& state) const { data_ptr<CollectionStateData>(state)->commit(); }

double const* CollectionNode::buff(const State& state) const {
    auto ptr = data_ptr<CollectionStateData>(state);
    return ptr->elements.data();
}

std::span<const Update> CollectionNode::diff(const State& state) const {
    return data_ptr<CollectionStateData>(state)->previous;
}

void CollectionNode::exchange(State& state, ssize_t i, ssize_t j) const {
    if (i == j) return;
    data_ptr<CollectionStateData>(state)->exchange(i, j);  // handles the asserts
}

void CollectionNode::rotate(State& state, ssize_t dest_idx, ssize_t src_idx) const {
    if (src_idx == dest_idx) return;
    data_ptr<CollectionStateData>(state)->rotate(dest_idx, src_idx);  // handles the asserts
}

void CollectionNode::grow(State& state) const {
    assert(this->dynamic());
    assert(data_ptr<CollectionStateData>(state)->size < max_size_);
    data_ptr<CollectionStateData>(state)->grow();
}

void CollectionNode::initialize_state(State& state, std::vector<double> contents) const {
    int index = this->topological_index();
    assert(index >= 0 && "must be topologically sorted");
    assert(static_cast<int>(state.size()) > index && "unexpected state length");
    assert(state[index] == nullptr && "already initialized state");

    if (static_cast<ssize_t>(contents.size()) < min_size_) {
        throw std::invalid_argument("contents is shorter than the List's minimum size");
    }
    if (static_cast<ssize_t>(contents.size()) > max_size_) {
        throw std::invalid_argument("contents is longer than the List's maximum size");
    }

    for (const auto& val : contents) {
        if (ssize_t(val) != val) throw std::invalid_argument("contents must be integral");
        if (val < 0) throw std::invalid_argument("contents must be non-negative");
        if (val >= max_value_) throw std::invalid_argument("contents too large for the collection");
    }

    // now confirm that we have a permutation
    std::unordered_set<double> set(contents.begin(), contents.end());
    if (set.size() < contents.size()) {
        throw std::invalid_argument("contents must be unique");
    }

    // finally, augment contents with the rest of the values so it's a permutation of range(n)
    for (ssize_t i = 0; i < max_value_; ++i) {
        if (set.count(static_cast<double>(i))) continue;  // already present
        contents.emplace_back(i);
    }

    assert(static_cast<ssize_t>(contents.size()) == max_value_);

    state[index] = std::make_unique<CollectionStateData>(std::move(contents), set.size());
}

void CollectionNode::revert(State& state) const { data_ptr<CollectionStateData>(state)->revert(); }

std::span<const ssize_t> CollectionNode::shape(const State& state) const {
    if (this->dynamic()) {
        const auto ptr = data_ptr<CollectionStateData>(state);
        return std::span<const ssize_t>(&(ptr->size), 1);
    } else {
        assert(this->min_size_ == this->max_size_);
        return std::span<const ssize_t>(&(this->min_size_), 1);
    }
}

void CollectionNode::shrink(State& state) const {
    assert(this->dynamic());
    assert(data_ptr<CollectionStateData>(state)->size > min_size_);
    data_ptr<CollectionStateData>(state)->shrink();
}

ssize_t CollectionNode::size(const State& state) const {
    if (this->dynamic()) {
        return data_ptr<CollectionStateData>(state)->size;
    }
    return this->size();
}

SizeInfo CollectionNode::sizeinfo() const {
    if (!dynamic())  return SizeInfo(size());
    return SizeInfo(this, min_size_, max_size_);
}

ssize_t CollectionNode::size_diff(const State& state) const {
    if (this->dynamic()) {
        auto ptr = data_ptr<CollectionStateData>(state);
        return ptr->size - ptr->previous_size;
    }
    return 0;
}

struct DisjointBitSetsNodeData : NodeStateData {
    DisjointBitSetsNodeData(ssize_t primary_set_size, ssize_t num_disjoint_sets)
            : primary_set_size(primary_set_size), num_disjoint_sets(num_disjoint_sets) {
        data.resize(primary_set_size * num_disjoint_sets, 0);
        diffs.resize(num_disjoint_sets);

        // Put all elements in the first set
        for (ssize_t i = 0; i < primary_set_size; ++i) {
            data[i] = 1;
        }
    }

    DisjointBitSetsNodeData(ssize_t primary_set_size, ssize_t num_disjoint_sets, const std::vector<std::vector<double>>& contents)
            : primary_set_size(primary_set_size), num_disjoint_sets(num_disjoint_sets) {
        if (static_cast<ssize_t>(contents.size()) != num_disjoint_sets) {
            throw std::invalid_argument("must provide correct number of sets");
        }

        data.resize(primary_set_size * num_disjoint_sets, 0);
        diffs.resize(num_disjoint_sets);

        ssize_t num_elements = 0;

        for (ssize_t set_index = 0; set_index < num_disjoint_sets; ++set_index) {
            if (static_cast<ssize_t>(contents[set_index].size()) != primary_set_size) {
                throw std::invalid_argument("provided vector for set must have size equal to the number of elements");
            }
            for (ssize_t el_index = 0; el_index < primary_set_size; ++el_index) {
                if (!(contents[set_index][el_index] == 0 || contents[set_index][el_index] == 1)) {
                    throw std::invalid_argument("provided set must be binary valued");
                }
                data[set_index * primary_set_size + el_index] = contents[set_index][el_index];
                num_elements += static_cast<ssize_t>(contents[set_index][el_index]);
            }
        }

        if (num_elements != primary_set_size) {
            throw std::invalid_argument("disjoint set elements must be in exactly one bit-set once");
        }
    }

    void swap_between_sets(ssize_t from_disjoint_set, ssize_t to_disjoint_set, ssize_t element) {
        double& el0 = data[from_disjoint_set * primary_set_size + element];
        double& el1 = data[to_disjoint_set * primary_set_size + element];

        if (el0 != el1) {
            diffs[from_disjoint_set].emplace_back(element, el0, el1);
            diffs[to_disjoint_set].emplace_back(element, el1, el0);
            std::swap(el0, el1);
        }
    }

    ssize_t get_containing_set_index(ssize_t element) const {
        assert(element >= 0 && element < primary_set_size);
        for (ssize_t set_index = 0; set_index < num_disjoint_sets; ++set_index) {
            if (data[set_index * primary_set_size + element]) return set_index;
        }

        assert(false && "disjoint set elements must be in exactly one bit-set once");
        unreachable();
    }

    void commit() {
        for (auto& diff : diffs) {
            diff.clear();
        }
    }

    void revert() {
        for (ssize_t set_index = 0; set_index < num_disjoint_sets; ++set_index) {
            for (const auto& update : diffs[set_index] | std::views::reverse) {
                data[set_index * primary_set_size + update.index] = update.old;
            }
            diffs[set_index].clear();
        }
    }

    double const* get_data(ssize_t set_index) {
        return &data[set_index * primary_set_size];
    }

    ssize_t primary_set_size;
    ssize_t num_disjoint_sets;
    std::vector<double> data;
    std::vector<std::vector<Update>> diffs;
};

void DisjointBitSetsNode::initialize_state(State& state) const {
    int index = this->topological_index();
    state[index] = std::make_unique<DisjointBitSetsNodeData>(primary_set_size_, num_disjoint_sets_);
}

void DisjointBitSetsNode::initialize_state(State& state, const std::vector<std::vector<double>>& contents) const {
    int index = this->topological_index();
    state[index] = std::make_unique<DisjointBitSetsNodeData>(primary_set_size_, num_disjoint_sets_, contents);
}

void DisjointBitSetsNode::commit(State& state) const {
    data_ptr<DisjointBitSetsNodeData>(state)->commit();
}

void DisjointBitSetsNode::revert(State& state) const {
    data_ptr<DisjointBitSetsNodeData>(state)->revert();
}

void DisjointBitSetsNode::default_move(State& state, RngAdaptor& rng) const {
    std::uniform_int_distribution<std::size_t> dist_sets(0, num_disjoint_sets_ - 1);
    std::uniform_int_distribution<std::size_t> dist_el(0, primary_set_size_ - 1);

    swap_between_sets(state, dist_sets(rng), dist_sets(rng), dist_el(rng));
}

void DisjointBitSetsNode::swap_between_sets(State& state, ssize_t from_disjoint_set, ssize_t to_disjoint_set, ssize_t element) const {
    data_ptr<DisjointBitSetsNodeData>(state)->swap_between_sets(from_disjoint_set, to_disjoint_set, element);
}

ssize_t DisjointBitSetsNode::get_containing_set_index(State& state, ssize_t element) const {
    return data_ptr<DisjointBitSetsNodeData>(state)->get_containing_set_index(element);
}

double const* DisjointBitSetNode::buff(const State& state) const {
    int index = disjoint_bit_sets_node->topological_index();
    DisjointBitSetsNodeData* pred_data = static_cast<DisjointBitSetsNodeData*>(state[index].get());
    return pred_data->get_data(set_index_);
}

std::span<const Update> DisjointBitSetNode::diff(const State& state) const {
    int index = disjoint_bit_sets_node->topological_index();
    DisjointBitSetsNodeData* pred_data = static_cast<DisjointBitSetsNodeData*>(state[index].get());
    return pred_data->diffs[set_index_];
}

struct DisjointListStateData : NodeStateData {
    DisjointListStateData(ssize_t primary_set_size, ssize_t num_disjoint_lists)
            : primary_set_size(primary_set_size) {
        lists.resize(num_disjoint_lists);
        all_list_updates.resize(num_disjoint_lists);
        list_sizes.resize(num_disjoint_lists);
        previous_list_sizes.resize(num_disjoint_lists, 0);

        if (primary_set_size < 0) {
            throw std::invalid_argument("number of primary elements must be non-negative");
        }
        if (num_disjoint_lists < 0) {
            throw std::invalid_argument("number of disjoint lists must be non-negative");
        }

        // put all elements in the first list
        for (ssize_t i = 0; i < primary_set_size; ++i) {
            lists[0].push_back(i);
        }
        previous_list_sizes[0] = primary_set_size;
    }

    explicit DisjointListStateData(size_t primary_set_size, size_t num_disjoint_lists,
                                   std::vector<std::vector<double>> lists_)
            : primary_set_size(primary_set_size),
              lists(std::move(lists_)),
              all_list_updates(num_disjoint_lists),
              list_sizes(num_disjoint_lists),
              previous_list_sizes(num_disjoint_lists) {
        if (lists.size() != num_disjoint_lists) {
            throw std::invalid_argument("must provide the correct number of disjoint lists");
        }

        std::unordered_set<size_t> elements;
        for (const auto& list : lists) {
            for (const auto& el : list) {
                size_t int_el = static_cast<size_t>(el);
                if (el < 0.0 || static_cast<double>(int_el) != el) {
                    throw std::invalid_argument(
                            "disjoint list elements must be integral and non-negative");
                }
                if (int_el >= primary_set_size) {
                    throw std::invalid_argument(
                            "disjoint list elements must be belong in the range [0, "
                            "primary_set_size)");
                }
                auto [_, inserted] = elements.insert(int_el);
                if (!inserted) {
                    throw std::invalid_argument(
                            "disjoint list elements must be in exactly one list once");
                }
            }
        }
        if (elements.size() != primary_set_size) {
            throw std::invalid_argument(
                    "disjoint lists must contain all elements in the range [0, primary_set_size)");
        }

        for (ssize_t i = 0; i < static_cast<ssize_t>(lists.size()); ++i) {
            previous_list_sizes[i] = lists[i].size();
        }
    }

    void commit() {
        for (size_t list_index = 0; list_index < lists.size(); ++list_index) {
            auto& updates = all_list_updates[list_index];
            if (updates.size() > 0) {
                previous_list_sizes[list_index] = lists[list_index].size();
                updates.clear();
            }
        }
    }

    void swap_in_list(ssize_t list_index, ssize_t element_i, ssize_t element_j) {
        // Swap two items in the same list
        auto& list = lists[list_index];
        assert(element_i >= 0 && static_cast<std::size_t>(element_i) < list.size());
        assert(element_j >= 0 && static_cast<std::size_t>(element_j) < list.size());

        std::swap(list[element_i], list[element_j]);

        all_list_updates[list_index].emplace_back(element_i, list[element_j], list[element_i]);
        all_list_updates[list_index].emplace_back(element_j, list[element_i], list[element_j]);
    }

    void pop_to_list(ssize_t from_list_index, ssize_t element_i, ssize_t to_list_index,
                     ssize_t element_j) {
        // Pop an item from one list and insert it into another
        auto& from_list = lists[from_list_index];
        auto& to_list = lists[to_list_index];
        assert(element_i >= 0 && static_cast<std::size_t>(element_i) < from_list.size());
        assert(element_j >= 0 && static_cast<std::size_t>(element_j) <= to_list.size());

        auto value = from_list[element_i];

        // Iteratively swap in each value to complete the deletion, adding an update
        // each time
        auto last_index_from = from_list.size() - 1;
        for (size_t i = element_i; i < last_index_from; i++) {
            all_list_updates[from_list_index].emplace_back(i, from_list[i], from_list[i + 1]);
            from_list[i] = from_list[i + 1];
        }
        // Remove the final item
        all_list_updates[from_list_index].emplace_back(
                Update::removal(last_index_from, from_list[last_index_from]));
        from_list.pop_back();

        // Iteratively move each item over for the insert, adding an update each time
        for (size_t i = element_j; i < to_list.size(); i++) {
            all_list_updates[to_list_index].emplace_back(i, to_list[i], value);
            std::swap(to_list[i], value);
        }
        // Push the final item
        all_list_updates[to_list_index].emplace_back(Update::placement(to_list.size(), value));
        to_list.push_back(value);
    }

    void revert() {
        // Un-apply any changes by working backwards through all updates.

        for (size_t list_index = 0; list_index < lists.size(); ++list_index) {
            auto& updates = all_list_updates[list_index];
            auto& list = lists[list_index];
            for (const Update& update : updates | std::views::reverse) {
                if (update.placed()) {
                    list.erase(list.begin() + update.index);
                } else if (update.removed()) {
                    list.insert(list.begin() + update.index, update.old);
                } else {
                    list[update.index] = update.old;
                }
            }
            updates.clear();
        }
    }

    std::unique_ptr<NodeStateData> copy() const override {
        return std::make_unique<DisjointListStateData>(*this);
    }

    ssize_t primary_set_size;

    std::vector<std::vector<double>> lists;
    std::vector<std::vector<Update>> all_list_updates;
    std::vector<ssize_t> list_sizes;  // used for the span returned by shape()
    std::vector<ssize_t> previous_list_sizes;
};

void DisjointListsNode::initialize_state(State& state) const {
    int index = this->topological_index();
    state[index] = std::make_unique<DisjointListStateData>(this->primary_set_size(),
                                                           this->num_disjoint_lists());
}

void DisjointListsNode::initialize_state(State& state,
                                         std::vector<std::vector<double>> contents) const {
    int index = this->topological_index();
    assert(index >= 0 && "must be topologically sorted");
    assert(static_cast<int>(state.size()) > index && "unexpected state length");
    assert(state[index] == nullptr && "already initialized state");

    state[index] = std::make_unique<DisjointListStateData>(
            this->primary_set_size(), this->num_disjoint_lists(), std::move(contents));
}

void DisjointListsNode::commit(State& state) const {
    data_ptr<DisjointListStateData>(state)->commit();
}

void DisjointListsNode::revert(State& state) const {
    data_ptr<DisjointListStateData>(state)->revert();
}

ssize_t DisjointListsNode::get_disjoint_list_size(State& state, ssize_t list_index) const {
    auto data = data_ptr<DisjointListStateData>(state);
    auto size = data->lists[list_index].size();
    return size;
    }

void DisjointListsNode::swap_in_list(State& state, ssize_t list_index, ssize_t element_i,
                                     ssize_t element_j) const {
    if (element_i == element_j) return;

    data_ptr<DisjointListStateData>(state)->swap_in_list(list_index, element_i, element_j);
}

void DisjointListsNode::pop_to_list(State& state, ssize_t from_list_index, ssize_t element_i,
                                    ssize_t to_list_index, ssize_t element_j) const {
    data_ptr<DisjointListStateData>(state)->pop_to_list(from_list_index, element_i, to_list_index,
                                                        element_j);
}

void DisjointListsNode::default_move(State& state, RngAdaptor& rng) const {
    auto data = data_ptr<DisjointListStateData>(state);

    // choose a list to pop from (list_i) randomly weighted by how many items are in it,
    // and another list to pop to (list_j) uniformly randomly
    size_t list_i = 0;

    std::uniform_real_distribution<double> uni_dist(0.0, 1.0);

    size_t num_items = 0;
    for (size_t list_index = 0; list_index < data->lists.size(); list_index++) {
        auto size = data->lists[list_index].size();
        if (size > 0) {
            num_items += size;
            if (uni_dist(rng) <= static_cast<double>(size) / num_items) {
                list_i = list_index;
            }
        }
    }

    std::uniform_int_distribution<std::size_t> dist_lists(0, data->lists.size() - 1);
    size_t list_j = dist_lists(rng);

    if (list_i != list_j) {
        std::uniform_int_distribution<std::size_t> eli_dist(0, data->lists[list_i].size() - 1);
        std::uniform_int_distribution<std::size_t> elj_dist(0, data->lists[list_j].size());

        this->pop_to_list(state, list_i, eli_dist(rng), list_j, elj_dist(rng));
    } else {
        std::uniform_int_distribution<std::size_t> el_dist(0, data->lists[list_i].size() - 1);

        this->swap_in_list(state, list_i, el_dist(rng), el_dist(rng));
    }
}

DisjointListNode::DisjointListNode(DisjointListsNode* disjoint_list_node)
        : Node(),
          ArrayOutputMixin(Array::DYNAMIC_SIZE),
          disjoint_list_node_ptr(disjoint_list_node),
          list_index_(disjoint_list_node->successors().size()),
          primary_set_size_(disjoint_list_node->primary_set_size()) {
    if (list_index_ >= disjoint_list_node->num_disjoint_lists()) {
        throw std::length_error("disjoint-list node already has all output nodes");
    }

    add_predecessor(disjoint_list_node);
}

double const* DisjointListNode::buff(const State& state) const {
    int index = disjoint_list_node_ptr->topological_index();
    DisjointListStateData* data = static_cast<DisjointListStateData*>(state[index].get());
    return data->lists[list_index_].data();
}

std::span<const Update> DisjointListNode::diff(const State& state) const {
    int index = disjoint_list_node_ptr->topological_index();
    DisjointListStateData* data = static_cast<DisjointListStateData*>(state[index].get());
    return data->all_list_updates[list_index_];
}

ssize_t DisjointListNode::size(const State& state) const {
    int index = disjoint_list_node_ptr->topological_index();
    DisjointListStateData* data = static_cast<DisjointListStateData*>(state[index].get());
    return data->lists[list_index_].size();
}

std::span<const ssize_t> DisjointListNode::shape(const State& state) const {
    int index = disjoint_list_node_ptr->topological_index();
    DisjointListStateData* data = static_cast<DisjointListStateData*>(state[index].get());
    data->list_sizes[list_index_] = data->lists[list_index_].size();
    return std::span<const ssize_t>(&(data->list_sizes[list_index_]), 1);
}

SizeInfo DisjointListNode::sizeinfo() const {
    assert(dynamic());
    return SizeInfo(this, 0, primary_set_size_);
}

ssize_t DisjointListNode::size_diff(const State& state) const {
    int index = disjoint_list_node_ptr->topological_index();
    DisjointListStateData* data = static_cast<DisjointListStateData*>(state[index].get());
    return data->lists[list_index_].size() - data->previous_list_sizes[list_index_];
}

void ListNode::default_move(State& state, RngAdaptor& rng) const {
    std::uniform_int_distribution<std::size_t> dist(0, this->size(state) - 1);
    // This might mean we're selecting the same variable twice.
    // But for large-ish lists that should be quite rare and anyway it's
    // pretty harmless.
    this->exchange(state, dist(rng), dist(rng));
}

void ListNode::initialize_state(State& state) const {
    int index = this->topological_index();
    assert(index >= 0 && "must be topologically sorted");
    assert(static_cast<int>(state.size()) > index && "unexpected state length");
    assert(state[index] == nullptr && "already initialized state");

    state[index] = std::make_unique<CollectionStateData>(max_size_);
}

void SetNode::default_move(State& state, RngAdaptor& rng) const {
    // 50/50 that we grow or shrink
    const ssize_t size = this->size(state);

    std::uniform_int_distribution<int> coin(0, 1);

    if (size < max_size_ && coin(rng)) {
        // Grow

        // first randomly shuffle the value we're going to add
        std::uniform_int_distribution<int> unused(size, max_size_ - 1);
        this->exchange(state, size, unused(rng));

        // then grow
        this->grow(state);
    } else if (size > min_size_) {
        // Shrink
        this->shrink(state);
    }
}

void SetNode::initialize_state(State& state) const {
    int index = this->topological_index();
    assert(index >= 0 && "must be topologically sorted");
    assert(static_cast<int>(state.size()) > index && "unexpected state length");
    assert(state[index] == nullptr && "already initialized state");

    // default to range(min_size_)
    state[index] = std::make_unique<CollectionStateData>(max_value_, min_size_);
}

}  // namespace dwave::optimization

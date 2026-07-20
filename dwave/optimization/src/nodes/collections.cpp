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

#include <memory>
#include <ranges>
#include <unordered_set>
#include <utility>

#include "_checkpoints.hpp"

namespace dwave::optimization {

// Given a collection, check that it is a valid sub-permutation of range(n),
// and then "augment" it so that the values not in collection are appended onto
// the end.
std::vector<double> augment_collection_(std::vector<double> values, const ssize_t n) {
    // First check that our given values has the values we expect
    if (not std::ranges::all_of(values, is_integer)) {
        throw std::invalid_argument("values values must be integers");
    }
    if (std::ranges::any_of(values, [](const auto& val) -> bool { return val < 0; })) {
        throw std::invalid_argument("values values must be non-negative");
    }
    if (std::ranges::any_of(values, [&n](const auto& val) -> bool { return val >= n; })) {
        throw std::invalid_argument("values values must be less than " + std::to_string(n));
    }

    // Next check that we have a sub-permutation of range(n)

    // We'll track how many times each value appears in the values.
    // We avoid the weird vector<bool> overload for performance even though
    // that's all we need here.
    std::vector<signed char> count(n, 0);

    for (const auto& val : values) {
        const size_t i = static_cast<size_t>(val);  // relies on previous checks

        if (count[i]) {
            throw std::invalid_argument(
                "values must be a subset of range(" + std::to_string(n) + ")"
            );
        }

        ++count[i];
    }

    // If values has the same size as n, then we're a permutation rather than
    // a sub-permutation and we can return
    if (static_cast<ssize_t>(values.size()) == n) return values;  // already unique

    // Otherwise add any "missing" values to the values.
    for (ssize_t i = 0; i < n; ++i) {
        if (not count[i]) values.emplace_back(i);
    }

    return values;
}

class CollectionStateData_;

class CollectionCheckpoint_ : public LinkedListCheckpoint {
 public:
    CollectionCheckpoint_() = delete;

    CollectionCheckpoint_(CollectionStateData_& state);

    ~CollectionCheckpoint_() override {
        // if we're the oldest checkpoint, just let whatever information we're
        // holding get destructed with us
        if (prev_ptr_ == nullptr) return;

        // otherwise we need to transfer our info over
        auto* prev_ptr = static_cast<CollectionCheckpoint_*>(prev_ptr_);
        assert(prev_ptr->drop_ == 0);
        for (auto& updates : updates_) prev_ptr->commit_updates(std::move(updates));
        prev_ptr->drop_ = drop_;
    }

    // detach the updates as a flattened view (in the forward order)
    auto detach_updates() {
        auto updates = std::move(updates_) | std::views::join;
        assert(updates_.empty());
        return updates;
    }

    ssize_t& drop() { return drop_; }
    ssize_t drop() const { return drop_; }

    // Track the updates associated with a commit
    void commit_updates(std::vector<Update> updates) {
        assert(0 <= drop_ and static_cast<size_t>(drop_) <= updates.size());

        if (not drop_) {
            updates_.emplace_back(std::move(updates));
            return;
        }

        // Otherwise we only want to take the updates up to drop
        // In C++23 we could use assign_range() which would be nicer
        auto relevant = std::move(updates) | std::views::drop(drop_);
        updates_.emplace_back(relevant.begin(), relevant.end());
        drop_ = 0;
    }

    // Track the updates associated with a revert
    void revert_updates(std::vector<Update> updates) {
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

    ssize_t size() { return size_; }

    bool valid() const override { return true; }

 private:
    std::vector<std::vector<Update>> updates_;
    ssize_t drop_;

    ssize_t size_;
};

class CollectionStateData_ : public CheckpointableState {
 public:
    explicit CollectionStateData_(ssize_t n) : CollectionStateData_(n, n) {}

    CollectionStateData_(ssize_t n, ssize_t size) : size_(size), previous_size_(size_) {
        assert(0 <= n and "n must be positive");
        assert(0 <= size_ and size_ <= n and "size must be in [0, n] inclusive");
        for (ssize_t i = 0; i < n; ++i) {
            elements_.push_back(i);
        }
    }

    CollectionStateData_(std::vector<double> elements, ssize_t size) :
        elements_(std::move(elements)), size_(size), previous_size_(size_) {
        assert(0 <= size_ and static_cast<size_t>(size_) <= elements_.size());
    }

    void assign(std::vector<double>&& values, ssize_t size) {
        // this should have been checked already by the CollectionNode
        assert(values.size() == elements_.size());

        // First shrink the visible part down so that we're correctly tracking
        // our visible/invisible changes.
        while (this->size_ > size) shrink();

        // Then, let's note the public changes to the visible part of the buffer
        for (ssize_t i = 0, stop = std::min(this->size_, size); i < stop; ++i) {
            updates_.emplace_back(i, elements_[i], values[i]);
        }

        // Next, actually swap the buffers (tracking the changes for a later revert)
        for (ssize_t i = 0, n = elements_.size(); i < n; ++i) {
            if (elements_[i] == values[i]) continue;  // no change
            all_updates_.emplace_back(i, elements_[i], values[i]);
        }
        std::swap(elements_, values);

        // Finally do any growing we need to do.
        while (this->size_ < size) grow();

        assert(this->size_ == size);
    }

    void assign(std::unique_ptr<NodeStateCheckpoint>& checkpoint) {
        // convert the checkpoint into something we can read
        auto* checkpoint_ptr = static_cast<CollectionCheckpoint_*>(checkpoint.get());

        // Right now, you can only revert to the most recent checkpoint. It's
        // pretty straightforward to support going further back, but this is all
        // we need right now.
        assert(this->checkpoint_ptr<CollectionCheckpoint_>() == checkpoint_ptr);

        // Ok, let's get ourselves to the same place as the checkpoint

        // we want to minimize the size of the visible buffer, so let's shrink ourselves
        // if we need to
        while (size_ > checkpoint_ptr->size()) shrink();

        for (const auto& [idx, old, _] : checkpoint_ptr->detach_updates() | std::views::reverse) {
            if (elements_[idx] == old) continue;  // nothing to do

            all_updates_.emplace_back(idx, elements_[idx], old);
            if (idx < size_) updates_.emplace_back(idx, elements_[idx], old);

            elements_[idx] = old;
        }

        // now that we've filled in our buffer, grow until we're the correct size
        while (size_ < checkpoint_ptr->size()) grow();

        // update the "drop" value of the checkpoint so that our next commit doesn't
        // add all of the changes we just added
        checkpoint_ptr->drop() = all_updates_.size();
    }

    const double* buff() const { return elements_.data(); }

    std::unique_ptr<NodeStateCheckpoint> checkpoint() {
        return std::make_unique<CollectionCheckpoint_>(*this);
    }

    void commit() {
        updates_.clear();

        if (auto* checkpoint_ptr = this->checkpoint_ptr<CollectionCheckpoint_>()) {
            checkpoint_ptr->commit_updates(std::move(all_updates_));
            assert(all_updates_.empty());
        } else {
            all_updates_.clear();
        }

        previous_size_ = size_;
    }

    std::unique_ptr<NodeStateData> copy() const override {
        return std::make_unique<CollectionStateData_>(*this);
    }

    std::span<const Update> diff() const { return updates_; }

    void exchange(ssize_t i, ssize_t j) {
        assert(i >= 0 and static_cast<size_t>(i) < elements_.size());
        assert(j >= 0 and static_cast<size_t>(j) < elements_.size());

        std::swap(elements_[i], elements_[j]);

        all_updates_.emplace_back(i, elements_[j], elements_[i]);
        all_updates_.emplace_back(j, elements_[i], elements_[j]);

        // only track changes within the visible part of the array
        if (i < size_ and j < size_) {
            updates_.emplace_back(i, elements_[j], elements_[i]);
            updates_.emplace_back(j, elements_[i], elements_[j]);
        } else if (j < size_) {
            updates_.emplace_back(j, elements_[i], elements_[j]);
        } else if (i < size_) {
            updates_.emplace_back(i, elements_[j], elements_[i]);
        }  // if neither are visible we don't need to track it
    }

    void grow() {
        assert(size_ < static_cast<ssize_t>(elements_.size()));
        updates_.emplace_back(Update::placement(size_, elements_[size_]));
        ++size_;
    }

    void revert() {
        updates_.clear();

        // Un-apply any changes by working backwards through all updates.
        // If we end up enforcing updates being sorted and unique later then
        // we could do this any order (or better in parallel).
        for (const Update& update : all_updates_ | std::views::reverse) {
            elements_[update.index] = update.old;
        }

        if (auto* checkpoint_ptr = this->checkpoint_ptr<CollectionCheckpoint_>()) {
            checkpoint_ptr->revert_updates(std::move(all_updates_));
            assert(all_updates_.empty());
        } else {
            all_updates_.clear();
        }

        size_ = previous_size_;
    }

    void rotate(ssize_t dest_idx, ssize_t src_idx) {
        // We want the rotate function to be called only on the visible part of the list.
        assert(src_idx >= 0 and src_idx < size_);
        assert(dest_idx >= 0 and dest_idx < size_);

        // Elements move from left to right.
        if (src_idx > dest_idx) {
            auto prev = elements_[src_idx];
            for (ssize_t i = dest_idx; i <= src_idx; i++) {
                std::swap(elements_[i], prev);
                all_updates_.emplace_back(i, prev, elements_[i]);
                updates_.emplace_back(i, prev, elements_[i]);
            }
        } else if (src_idx < dest_idx) {
            auto prev = elements_[src_idx];
            for (ssize_t i = dest_idx; i >= src_idx; i--) {
                std::swap(elements_[i], prev);
                all_updates_.emplace_back(i, prev, elements_[i]);
                updates_.emplace_back(i, prev, elements_[i]);
            }
        }
    }

    std::span<const ssize_t> shape() const { return std::span<const ssize_t>(&size_, 1); }

    void shrink() {
        assert(size_ > 0);
        --size_;
        updates_.emplace_back(Update::removal(size_, elements_[size_]));
    }

    ssize_t size() const { return size_; }

    ssize_t size_diff() const { return size_ - previous_size_; }

 private:
    friend CollectionCheckpoint_;

    // The elements in the collection
    std::vector<double> elements_;

    // The updates to the *visible* elements
    std::vector<Update> updates_;

    // The updates to the buffer, including the changes that are not currently
    // visible to the user
    std::vector<Update> all_updates_;

    // The current size of visible buffer, as well as the size after the last
    // commit/revert
    ssize_t size_;
    ssize_t previous_size_;
};

CollectionCheckpoint_::CollectionCheckpoint_(CollectionStateData_& state) :
    LinkedListCheckpoint(state),
    updates_(),
    drop_(state.all_updates_.size()),  // so we ignore any updates added before we're made
    size_(state.size()) {
    if (auto* prev_checkpoint = static_cast<CollectionCheckpoint_*>(prev_ptr_)) {
        prev_checkpoint->commit_updates(state.all_updates_);
        assert(prev_checkpoint->drop() == 0);
    }
}

CollectionNode::CollectionNode(ssize_t max_value, ssize_t min_size, ssize_t max_size) :
    ArrayOutputMixin((min_size == max_size) ? max_size : Array::DYNAMIC_SIZE),
    max_value_(max_value),
    min_size_(min_size),
    max_size_(max_size) {
    if (min_size < 0 or max_size < 0) {
        throw std::invalid_argument("a collection cannot contain fewer than 0 elements");
    }
    if (min_size > max_size) {
        throw std::invalid_argument("min_size cannot be greater than max_size");
    }
    if (max_size > max_value) {
        throw std::invalid_argument("a collection cannot be larger than its maximum value");
    }
}

void CollectionNode::assign(State& state, std::vector<double> values) const {
    const ssize_t size = values.size();
    if (size < min_size_) throw std::invalid_argument("values does not contain enough values");
    if (size > max_size_) throw std::invalid_argument("values contains too many values");

    // Check that the values are a proper subset and fill out the invisible part
    auto augemented = augment_collection_(std::move(values), max_value_);

    data_ptr_<CollectionStateData_>(state)->assign(std::move(augemented), size);
}

void CollectionNode::assign_from_checkpoint(
    State& state,
    std::unique_ptr<NodeStateCheckpoint>& checkpoint
) const {
    data_ptr_<CollectionStateData_>(state)->assign(checkpoint);
}
void CollectionNode::assign_from_checkpoint(
    State& state,
    std::unique_ptr<NodeStateCheckpoint>&& checkpoint
) const {
    assign_from_checkpoint(state, checkpoint);  // call the lvalue version
    checkpoint.reset();
}

std::unique_ptr<NodeStateCheckpoint> CollectionNode::checkpoint(State& state) const {
    return data_ptr_<CollectionStateData_>(state)->checkpoint();
}

void CollectionNode::commit(State& state) const {
    data_ptr_<CollectionStateData_>(state)->commit();
}

const double* CollectionNode::buff(const State& state) const {
    return data_ptr_<CollectionStateData_>(state)->buff();
}

std::span<const Update> CollectionNode::diff(const State& state) const {
    return data_ptr_<CollectionStateData_>(state)->diff();
}

void CollectionNode::exchange(State& state, ssize_t i, ssize_t j) const {
    if (i == j) return;
    data_ptr_<CollectionStateData_>(state)->exchange(i, j);  // handles the asserts
}

void CollectionNode::rotate(State& state, ssize_t dest_idx, ssize_t src_idx) const {
    if (src_idx == dest_idx) return;
    data_ptr_<CollectionStateData_>(state)->rotate(dest_idx, src_idx);  // handles the asserts
}

void CollectionNode::grow(State& state) const {
    assert(this->dynamic());
    assert(data_ptr_<CollectionStateData_>(state)->size() < max_size_);
    data_ptr_<CollectionStateData_>(state)->grow();
}

void CollectionNode::initialize_state(State& state, std::vector<double> values) const {
    const ssize_t size = values.size();
    if (size < min_size_) throw std::invalid_argument("values does not contain enough values");
    if (size > max_size_) throw std::invalid_argument("values contains too many values");

    // Check that the values are a proper subset and fill out the invisible part
    auto augemented = augment_collection_(std::move(values), max_value_);

    emplace_data_ptr_<CollectionStateData_>(state, std::move(augemented), size);
}

bool CollectionNode::integral() const { return true; }

double CollectionNode::min() const { return 0; }

double CollectionNode::max() const {
    if (max_value_ == 0) return 0;
    return max_value_ - 1;
}

void CollectionNode::revert(State& state) const {
    data_ptr_<CollectionStateData_>(state)->revert();
}

std::span<const ssize_t> CollectionNode::shape(const State& state) const {
    if (not dynamic()) {
        assert(min_size_ == max_size_);
        return std::span<const ssize_t>(&min_size_, 1);
    }
    return data_ptr_<CollectionStateData_>(state)->shape();
}

void CollectionNode::shrink(State& state) const {
    assert(dynamic());
    assert(size(state) > min_size_);
    data_ptr_<CollectionStateData_>(state)->shrink();
}

ssize_t CollectionNode::size(const State& state) const {
    if (ssize_t size = this->size(); size >= 0) {
        assert(data_ptr_<CollectionStateData_>(state)->size() == size);
        return size;
    }
    return data_ptr_<CollectionStateData_>(state)->size();
}

SizeInfo CollectionNode::sizeinfo() const {
    if (not dynamic()) return SizeInfo(size());
    return SizeInfo(this, min_size_, max_size_);
}

ssize_t CollectionNode::size_diff(const State& state) const {
    if (not dynamic()) {
        assert(data_ptr_<CollectionStateData_>(state)->size_diff() == 0);
        return 0;
    }
    return data_ptr_<CollectionStateData_>(state)->size_diff();
}

struct DisjointBitSetsNodeData_ : NodeStateData {
    DisjointBitSetsNodeData_(ssize_t primary_set_size, ssize_t num_disjoint_sets) :
        primary_set_size(primary_set_size), num_disjoint_sets(num_disjoint_sets) {
        data.resize(primary_set_size * num_disjoint_sets, 0);
        diffs.resize(num_disjoint_sets);

        // Put all elements in the first set
        for (ssize_t i = 0; i < primary_set_size; ++i) {
            data[i] = 1;
        }
    }

    DisjointBitSetsNodeData_(
        ssize_t primary_set_size,
        ssize_t num_disjoint_sets,
        const std::vector<std::vector<double>>& contents
    ) :
        primary_set_size(primary_set_size), num_disjoint_sets(num_disjoint_sets) {
        if (static_cast<ssize_t>(contents.size()) != num_disjoint_sets) {
            throw std::invalid_argument("must provide correct number of sets");
        }

        data.resize(primary_set_size * num_disjoint_sets, 0);
        diffs.resize(num_disjoint_sets);

        ssize_t num_elements = 0;

        for (ssize_t set_index = 0; set_index < num_disjoint_sets; ++set_index) {
            if (static_cast<ssize_t>(contents[set_index].size()) != primary_set_size) {
                throw std::invalid_argument(
                    "provided vector for set must have size equal to the number of elements"
                );
            }
            for (ssize_t el_index = 0; el_index < primary_set_size; ++el_index) {
                if (not(contents[set_index][el_index] == 0 or contents[set_index][el_index] == 1)) {
                    throw std::invalid_argument("provided set must be binary valued");
                }
                data[set_index * primary_set_size + el_index] = contents[set_index][el_index];
                num_elements += static_cast<ssize_t>(contents[set_index][el_index]);
            }
        }

        if (num_elements != primary_set_size) {
            throw std::invalid_argument(
                "disjoint set elements must be in exactly one bit-set once"
            );
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
        assert(element >= 0 and element < primary_set_size);
        for (ssize_t set_index = 0; set_index < num_disjoint_sets; ++set_index) {
            if (data[set_index * primary_set_size + element]) return set_index;
        }

        assert(false and "disjoint set elements must be in exactly one bit-set once");
        unreachable();
    }

    void commit() {
        for (auto& diff : diffs) {
            diff.clear();
        }
    }

    std::unique_ptr<NodeStateData> copy() const override {
        return std::make_unique<DisjointBitSetsNodeData_>(*this);
    }

    void revert() {
        for (ssize_t set_index = 0; set_index < num_disjoint_sets; ++set_index) {
            for (const auto& update : diffs[set_index] | std::views::reverse) {
                data[set_index * primary_set_size + update.index] = update.old;
            }
            diffs[set_index].clear();
        }
    }

    const double* get_data(ssize_t set_index) { return &data[set_index * primary_set_size]; }

    ssize_t primary_set_size;
    ssize_t num_disjoint_sets;
    std::vector<double> data;
    std::vector<std::vector<Update>> diffs;
};

DisjointBitSetsNode::DisjointBitSetsNode(ssize_t primary_set_size, ssize_t num_disjoint_sets) :
    primary_set_size_(primary_set_size), num_disjoint_sets_(num_disjoint_sets) {
    if (primary_set_size < 0) throw std::invalid_argument("primary_set_size must be non-negative");
    if (num_disjoint_sets < 1) throw std::invalid_argument("num_disjoint_sets must be positive");
}

void DisjointBitSetsNode::initialize_state(State& state) const {
    emplace_data_ptr_<DisjointBitSetsNodeData_>(state, primary_set_size_, num_disjoint_sets_);
}

void DisjointBitSetsNode::initialize_state(
    State& state,
    const std::vector<std::vector<double>>& contents
) const {
    emplace_data_ptr_<DisjointBitSetsNodeData_>(
        state, primary_set_size_, num_disjoint_sets_, contents
    );
}

void DisjointBitSetsNode::commit(State& state) const {
    data_ptr_<DisjointBitSetsNodeData_>(state)->commit();
}

void DisjointBitSetsNode::revert(State& state) const {
    data_ptr_<DisjointBitSetsNodeData_>(state)->revert();
}

void DisjointBitSetsNode::swap_between_sets(
    State& state,
    ssize_t from_disjoint_set,
    ssize_t to_disjoint_set,
    ssize_t element
) const {
    data_ptr_<DisjointBitSetsNodeData_>(state)->swap_between_sets(
        from_disjoint_set, to_disjoint_set, element
    );
}

ssize_t DisjointBitSetsNode::get_containing_set_index(State& state, ssize_t element) const {
    return data_ptr_<DisjointBitSetsNodeData_>(state)->get_containing_set_index(element);
}

DisjointBitSetNode::DisjointBitSetNode(DisjointBitSetsNode* disjoint_bit_sets_node) :
    ArrayOutputMixin(disjoint_bit_sets_node->primary_set_size()),
    disjoint_bit_sets_node_(disjoint_bit_sets_node),
    set_index_(disjoint_bit_sets_node->successors().size()),
    primary_set_size_(disjoint_bit_sets_node_->primary_set_size()) {
    if (set_index_ >= disjoint_bit_sets_node_->num_disjoint_sets()) {
        throw std::length_error("disjoint-bit-set node already has all output nodes");
    }
    add_predecessor_(disjoint_bit_sets_node);
}

const double* DisjointBitSetNode::buff(const State& state) const {
    int index = disjoint_bit_sets_node_->topological_index();
    DisjointBitSetsNodeData_* pred_data =
        static_cast<DisjointBitSetsNodeData_*>(state[index].get());
    return pred_data->get_data(set_index_);
}

std::span<const Update> DisjointBitSetNode::diff(const State& state) const {
    int index = disjoint_bit_sets_node_->topological_index();
    DisjointBitSetsNodeData_* pred_data =
        static_cast<DisjointBitSetsNodeData_*>(state[index].get());
    return pred_data->diffs[set_index_];
}

bool DisjointBitSetNode::integral() const { return true; }

double DisjointBitSetNode::min() const { return 0; }

double DisjointBitSetNode::max() const { return 1; }

struct DisjointListStateData_ : NodeStateData {
    DisjointListStateData_(ssize_t primary_set_size, ssize_t num_disjoint_lists) :
        primary_set_size(primary_set_size) {
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

    explicit DisjointListStateData_(
        size_t primary_set_size,
        size_t num_disjoint_lists,
        std::vector<std::vector<double>> lists_
    ) :
        primary_set_size(primary_set_size),
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
                if (el < 0.0 or static_cast<double>(int_el) != el) {
                    throw std::invalid_argument(
                        "disjoint list elements must be integral and non-negative"
                    );
                }
                if (int_el >= primary_set_size) {
                    throw std::invalid_argument(
                        "disjoint list elements must be belong in the range [0, "
                        "primary_set_size)"
                    );
                }
                auto [_, inserted] = elements.insert(int_el);
                if (not inserted) {
                    throw std::invalid_argument(
                        "disjoint list elements must be in exactly one list once"
                    );
                }
            }
        }
        if (elements.size() != primary_set_size) {
            throw std::invalid_argument(
                "disjoint lists must contain all elements in the range [0, primary_set_size)"
            );
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

    void rotate_in_list(ssize_t list_index, ssize_t dest_idx, ssize_t src_idx) {
        // We want the rotate function to be called only on the visible part of the list.
        auto& list = lists[list_index];
        assert(src_idx >= 0 and static_cast<size_t>(src_idx) < list.size());
        assert(dest_idx >= 0 and static_cast<size_t>(dest_idx) < list.size());

        // Elements move from left to right.
        if (src_idx > dest_idx) {
            auto prev = list[src_idx];
            for (ssize_t i = dest_idx; i <= src_idx; i++) {
                std::swap(list[i], prev);
                all_list_updates[list_index].emplace_back(i, prev, list[i]);
            }
        } else if (src_idx < dest_idx) {
            auto prev = list[src_idx];
            for (ssize_t i = dest_idx; i >= src_idx; i--) {
                std::swap(list[i], prev);
                all_list_updates[list_index].emplace_back(i, prev, list[i]);
            }
        }
    }

    // "Manually" set the full state of one of the disjoint lists. This will not check that
    // the new state is a valid configuration (given the state of the other lists), so it
    // is up to the caller to ensure the final state of all the lists is correct. Only emits
    // `Update`s for indices in the final list that actually changed.
    void set_state(ssize_t list_index, const std::span<const double>& new_values) {
        auto& list = lists[list_index];
        auto& diff = all_list_updates[list_index];

        const ssize_t overlap_length = std::min<ssize_t>(list.size(), new_values.size());

        {
            auto bit = list.begin();
            auto vit = new_values.begin();
            for (ssize_t index = 0; index < overlap_length; ++index, ++bit, ++vit) {
                if (*bit == *vit) continue;  // no change
                diff.emplace_back(index, *bit, *vit);
                *bit = *vit;
            }
        }

        // next walk backwards through the excess list, if there is any, removing as we go
        {
            for (ssize_t index = list.size() - 1; index >= overlap_length; --index) {
                diff.emplace_back(Update::removal(index, list[index]));
            }
            list.resize(overlap_length);
        }

        // finally walk forward through the excess new_values, if there are any, adding them to the
        // list
        {
            auto vit = new_values.begin() + list.size();
            list.reserve(new_values.size());
            for (
                ssize_t index = list.size(), stop = new_values.size(); index < stop; ++index, ++vit
            ) {
                diff.emplace_back(Update::placement(index, *vit));
                list.emplace_back(*vit);
            }
        }
    }

    void swap_in_list(ssize_t list_index, ssize_t element_i, ssize_t element_j) {
        // Swap two items in the same list
        auto& list = lists[list_index];
        assert(element_i >= 0 and static_cast<size_t>(element_i) < list.size());
        assert(element_j >= 0 and static_cast<size_t>(element_j) < list.size());

        std::swap(list[element_i], list[element_j]);

        all_list_updates[list_index].emplace_back(element_i, list[element_j], list[element_i]);
        all_list_updates[list_index].emplace_back(element_j, list[element_i], list[element_j]);
    }

    void pop_to_list(
        ssize_t from_list_index,
        ssize_t element_i,
        ssize_t to_list_index,
        ssize_t element_j
    ) {
        // Pop an item from one list and insert it into another
        auto& from_list = lists[from_list_index];
        auto& to_list = lists[to_list_index];
        assert(element_i >= 0 and static_cast<size_t>(element_i) < from_list.size());
        assert(element_j >= 0 and static_cast<size_t>(element_j) <= to_list.size());

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
            Update::removal(last_index_from, from_list[last_index_from])
        );
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
        return std::make_unique<DisjointListStateData_>(*this);
    }

    ssize_t primary_set_size;

    std::vector<std::vector<double>> lists;
    std::vector<std::vector<Update>> all_list_updates;
    std::vector<ssize_t> list_sizes;  // used for the span returned by shape()
    std::vector<ssize_t> previous_list_sizes;
};

DisjointListsNode::DisjointListsNode(ssize_t primary_set_size, ssize_t num_disjoint_lists) :
    primary_set_size_(primary_set_size), num_disjoint_lists_(num_disjoint_lists) {
    if (primary_set_size < 0) throw std::invalid_argument("primary_set_size must be non-negative");
    if (num_disjoint_lists < 1) throw std::invalid_argument("num_disjoint_lists must be positive");
}

void DisjointListsNode::initialize_state(State& state) const {
    emplace_data_ptr_<DisjointListStateData_>(
        state, this->primary_set_size(), this->num_disjoint_lists()
    );
}

void DisjointListsNode::initialize_state(
    State& state,
    std::vector<std::vector<double>> contents
) const {
    emplace_data_ptr_<DisjointListStateData_>(
        state, this->primary_set_size(), this->num_disjoint_lists(), std::move(contents)
    );
}

void DisjointListsNode::commit(State& state) const {
    data_ptr_<DisjointListStateData_>(state)->commit();
}

void DisjointListsNode::revert(State& state) const {
    data_ptr_<DisjointListStateData_>(state)->revert();
}

void DisjointListsNode::propagate(State& state) const {
#ifndef NDEBUG
    auto data = data_ptr_<DisjointListStateData_>(state);
    std::vector<bool> items(this->primary_set_size());
    for (auto const& list : data->lists) {
        for (const auto& item : list) {
            assert(item >= 0);
            assert(item < this->primary_set_size());
            assert(items[item] == false);
            items[item] = true;
        }
    }

    for (const bool& found : items) {
        assert(found);
    }
#endif

    Node::propagate(state);
}

ssize_t DisjointListsNode::get_disjoint_list_size(State& state, ssize_t list_index) const {
    auto data = data_ptr_<DisjointListStateData_>(state);
    auto size = data->lists[list_index].size();
    return size;
}

void DisjointListsNode::rotate_in_list(
    State& state,
    ssize_t list_index,
    ssize_t src_idx,
    ssize_t dest_idx
) const {
    if (src_idx == dest_idx) return;
    data_ptr_<DisjointListStateData_>(state)->rotate_in_list(list_index, src_idx, dest_idx);
}

void DisjointListsNode::set_state(
    State& state,
    ssize_t list_index,
    const std::span<const double>& new_values
) const {
    data_ptr_<DisjointListStateData_>(state)->set_state(list_index, new_values);
}

void DisjointListsNode::swap_in_list(
    State& state,
    ssize_t list_index,
    ssize_t element_i,
    ssize_t element_j
) const {
    if (element_i == element_j) return;

    data_ptr_<DisjointListStateData_>(state)->swap_in_list(list_index, element_i, element_j);
}

void DisjointListsNode::pop_to_list(
    State& state,
    ssize_t from_list_index,
    ssize_t element_i,
    ssize_t to_list_index,
    ssize_t element_j
) const {
    data_ptr_<DisjointListStateData_>(state)->pop_to_list(
        from_list_index, element_i, to_list_index, element_j
    );
}

DisjointListNode::DisjointListNode(DisjointListsNode* disjoint_list_node) :
    ArrayOutputMixin(Array::DYNAMIC_SIZE),
    disjoint_list_node_ptr_(disjoint_list_node),
    list_index_(disjoint_list_node->successors().size()),
    primary_set_size_(disjoint_list_node->primary_set_size()) {
    if (list_index_ >= disjoint_list_node->num_disjoint_lists()) {
        throw std::length_error("disjoint-list node already has all output nodes");
    }

    add_predecessor_(disjoint_list_node);
}

const double* DisjointListNode::buff(const State& state) const {
    int index = disjoint_list_node_ptr_->topological_index();
    DisjointListStateData_* data = static_cast<DisjointListStateData_*>(state[index].get());
    return data->lists[list_index_].data();
}

std::span<const Update> DisjointListNode::diff(const State& state) const {
    int index = disjoint_list_node_ptr_->topological_index();
    DisjointListStateData_* data = static_cast<DisjointListStateData_*>(state[index].get());
    return data->all_list_updates[list_index_];
}

bool DisjointListNode::integral() const { return true; }

double DisjointListNode::min() const { return 0; }

double DisjointListNode::max() const { return primary_set_size_ - 1; }

ssize_t DisjointListNode::size(const State& state) const {
    int index = disjoint_list_node_ptr_->topological_index();
    DisjointListStateData_* data = static_cast<DisjointListStateData_*>(state[index].get());
    return data->lists[list_index_].size();
}

std::span<const ssize_t> DisjointListNode::shape(const State& state) const {
    int index = disjoint_list_node_ptr_->topological_index();
    DisjointListStateData_* data = static_cast<DisjointListStateData_*>(state[index].get());
    data->list_sizes[list_index_] = data->lists[list_index_].size();
    return std::span<const ssize_t>(&(data->list_sizes[list_index_]), 1);
}

SizeInfo DisjointListNode::sizeinfo() const {
    assert(dynamic());
    return SizeInfo(this, 0, primary_set_size_);
}

ssize_t DisjointListNode::size_diff(const State& state) const {
    int index = disjoint_list_node_ptr_->topological_index();
    DisjointListStateData_* data = static_cast<DisjointListStateData_*>(state[index].get());
    return data->lists[list_index_].size() - data->previous_list_sizes[list_index_];
}

void ListNode::initialize_state(State& state) const {
    emplace_data_ptr_<CollectionStateData_>(state, max_value_, min_size_);
}

void SetNode::initialize_state(State& state) const {
    emplace_data_ptr_<CollectionStateData_>(state, max_value_, min_size_);
}

}  // namespace dwave::optimization

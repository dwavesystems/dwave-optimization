// Copyright 2025 D-Wave
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

#include "dwave-optimization/nodes/binaryop.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

#include "_state.hpp"
#include "dwave-optimization/array.hpp"
#include "dwave-optimization/common.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/operator.hpp"
#include "dwave-optimization/state.hpp"

namespace dwave::optimization {

template <class BinaryOp>
double const* BinaryOpNode<BinaryOp>::buff(const State& state) const {
    return this->template data_ptr_<ArrayNodeStateData>(state)->buff();
}

template <class BinaryOp>
std::span<const Update> BinaryOpNode<BinaryOp>::diff(const State& state) const {
    return this->template data_ptr_<ArrayNodeStateData>(state)->diff();
}

template <class BinaryOp>
void BinaryOpNode<BinaryOp>::commit(State& state) const {
    this->template data_ptr_<ArrayNodeStateData>(state)->commit();
}

template <class BinaryOp>
void BinaryOpNode<BinaryOp>::initialize_state(State& state) const {
    auto* lhs_ptr = operands_[0];
    auto* rhs_ptr = operands_[1];

    std::vector<double> values;

    if (std::ranges::equal(lhs_ptr->shape(state), rhs_ptr->shape(state))) {
        // This is the easy case - all we need to do is iterate over both as flat arrays
        values.reserve(lhs_ptr->size(state));

        for (const auto& [x, y] : std::views::zip(lhs_ptr->view(state), rhs_ptr->view(state))) {
            values.emplace_back(op(x, y));  // order is important
        }

    } else if (lhs_ptr->size() == 1) {
        values.reserve(rhs_ptr->size(state));

        const double lhs = lhs_ptr->view(state).front();

        for (const double val : rhs_ptr->view(state)) {
            values.emplace_back(op(lhs, val));
        }

    } else if (rhs_ptr->size() == 1) {
        values.reserve(lhs_ptr->size(state));

        const double rhs = rhs_ptr->view(state).front();

        for (const double val : lhs_ptr->view(state)) {
            values.emplace_back(op(val, rhs));
        }

    } else {
        // this case is complicated we need to "stretch" dimensions into each other
        assert(false and "not yet implemented");
        std::unreachable();
    }

    this->template emplace_data_ptr_<ArrayNodeStateData>(state, std::move(values));
}

template <class BinaryOp>
void BinaryOpNode<BinaryOp>::propagate(State& state) const {
    const Array* lhs_ptr = operands_[0];
    const Array* rhs_ptr = operands_[1];
    std::span<const Update> lhs_diff = lhs_ptr->diff(state);
    std::span<const Update> rhs_diff = rhs_ptr->diff(state);

    // If there are no updates, return early.
    if (lhs_diff.empty() and rhs_diff.empty()) return;

    auto ptr = this->template data_ptr_<ArrayNodeStateData>(state);

    if (std::ranges::equal(lhs_ptr->shape(state), rhs_ptr->shape(state))) {
        // The easy case, just go through both predecessors making updates.

        // Handle the dynamic case by copying and deduplicating both diffs,
        // and then get a span pointing to the copies to use instead of
        // lhs_ptr/rhs_ptr->diff directly.
        std::vector<Update> lhs_diff_copy;
        std::vector<Update> rhs_diff_copy;
        if (lhs_ptr->dynamic()) {
            assert(rhs_ptr->dynamic());
            lhs_diff_copy = lhs_diff | views::deduplicate_diff;
            rhs_diff_copy = rhs_diff | views::deduplicate_diff;
            lhs_diff = std::span(lhs_diff_copy);
            rhs_diff = std::span(rhs_diff_copy);
        }

        if (lhs_diff.size() and rhs_diff.size()) {
            // Both modified
            auto lit = lhs_ptr->begin(state);
            auto rit = rhs_ptr->begin(state);

            auto apply_op = [this, &lit, &rit](const Update& up) {
                if (up.removed()) {
                    return up;
                }
                // Use the update's old value in case it's a placement.
                // ArrayNodeStateData.update() otherwise ignores the old value
                return Update(up.index, up.old, op(*(lit + up.index), *(rit + up.index)));
            };

            auto is_standard_update = [](const Update& up) {
                return not up.removed() and not up.placed();
            };

            ptr->update(std::move(lhs_diff) | std::views::transform(apply_op));
            // For the RHS, we have already dealt with growing/shrinking the array,
            // so we just ignore all updates that are placements/removals.
            ptr->update(
                std::move(rhs_diff) | std::views::filter(is_standard_update) |
                std::views::transform(apply_op)
            );
        } else if (lhs_diff.size()) {
            // LHS modified, but not RHS
            auto rit = rhs_ptr->begin(state);
            for (const auto& [index, _, value] : lhs_diff) {
                ptr->set(index, op(value, *(rit + index)));
            }
        } else if (rhs_diff.size()) {
            // RHS modified, but not LHS
            auto lit = lhs_ptr->begin(state);
            for (const auto& [index, _, value] : rhs_diff) {
                ptr->set(index, op(*(lit + index), value));
            }
        }
    } else if (lhs_ptr->size() == 1) {
        // lhs is a single value being broadcast to the rhs array.

        // Create a unary version of our binary op.
        const double lhs = lhs_ptr->view(state).front();
        auto unary_func = std::bind(op, lhs, std::placeholders::_1);

        if (lhs_diff.size()) {
            // The lhs has changed, so in this case we're probably changing
            // everything, so just overwrite the state entirely.
            auto rhs_view = rhs_ptr->view(state);
            ptr->assign(rhs_view | std::views::transform(unary_func));
        } else {
            // The lhs did not change, so go through the changes on the rhs and apply them
            auto update_func = [&unary_func](Update update) {
                if (not update.removed()) update.value = unary_func(update.value);
                return update;
            };
            ptr->update(std::move(rhs_diff) | std::views::transform(update_func));
        }
    } else if (rhs_ptr->size() == 1) {
        // rhs is a single value being broadcast to the lhs array

        // create a unary version of our binary op
        const double rhs = rhs_ptr->view(state).front();
        auto unary_func = std::bind(op, std::placeholders::_1, rhs);

        if (rhs_diff.size()) {
            // The rhs has changed, so in this case we're probably changing
            // everything, so just overwrite the state entirely.
            auto lhs_view = lhs_ptr->view(state);
            ptr->assign(lhs_view | std::views::transform(unary_func));
        } else {
            // The rhs did not change, so go through the changes on the lhs and apply them
            auto update_func = [&unary_func](Update update) {
                if (not update.removed()) update.value = unary_func(update.value);
                return update;
            };
            ptr->update(std::move(lhs_diff) | std::views::transform(update_func));
        }
    } else {
        // this case is complicated we need to "stretch" dimensions into eachother
        assert(false and "not yet implemented");
        std::unreachable();
    }

    if (ptr->diff().size()) Node::propagate(state);
}

template <class BinaryOp>
void BinaryOpNode<BinaryOp>::replace_predecessor_(ssize_t previous_index, Node* node_ptr) {
    Node::replace_predecessor_(previous_index, node_ptr);

    assert(0 <= previous_index and previous_index < 2);
    operands_[previous_index] = dynamic_cast<ArrayNode*>(node_ptr);
    assert(operands_[previous_index] != nullptr);
}

template <class BinaryOp>
void BinaryOpNode<BinaryOp>::revert(State& state) const {
    this->template data_ptr_<ArrayNodeStateData>(state)->revert();
}

template <class BinaryOp>
std::span<const ssize_t> BinaryOpNode<BinaryOp>::shape(const State& state) const {
    if (not this->dynamic()) return this->shape();

    const auto [lhs, rhs] = operands_;

    // If we're broadcasting we know which size we're looking at
    if (lhs->size() == 1) return rhs->shape(state);
    if (rhs->size() == 1) return lhs->shape(state);

    // Dev note: it's very tempting to put an assert here that checks
    // that lhs.shape == rhs.shape, but that can make calling shape()
    // very expensive in some cases.
    // Having this commented out makes the second if-branch above redundant, but
    // I am leaving this in for clarity and future debugging.
    // assert(std::ranges::equal(lhs->shape(state), rhs->shape(state)));

    return lhs->shape(state);
}

template <class BinaryOp>
ssize_t BinaryOpNode<BinaryOp>::size(const State& state) const {
    if (const ssize_t size = this->size(); size >= 0) return size;

    const auto [lhs, rhs] = operands_;

    // If we're broadcasting we know which size we're looking at
    if (lhs->size() == 1) return rhs->size(state);
    if (rhs->size() == 1) return lhs->size(state);

    // Dev note: it's very tempting to put an assert here that checks
    // that lhs.size == rhs.size, but that can make calling size()
    // very expensive in some cases.
    // Having this commented out makes the second if-branch above redundant, but
    // I am leaving this in for clarity and future debugging.
    // assert(lhs->size(state) == rhs->size(state)));

    return lhs->size(state);
}

template <class BinaryOp>
ssize_t BinaryOpNode<BinaryOp>::size_diff(const State& state) const {
    return this->template data_ptr_<ArrayNodeStateData>(state)->size_diff();
}

template class BinaryOpNode<op::add>;
template class BinaryOpNode<op::divide>;
template class BinaryOpNode<op::equal>;
template class BinaryOpNode<op::less_equal>;
template class BinaryOpNode<op::logical_and>;
template class BinaryOpNode<op::logical_or>;
template class BinaryOpNode<op::logical_xor>;
template class BinaryOpNode<op::maximum>;
template class BinaryOpNode<op::minimum>;
template class BinaryOpNode<op::multiply>;
template class BinaryOpNode<op::remainder>;
template class BinaryOpNode<op::subtract>;
template class BinaryOpNode<op::binary_expression_op>;

}  // namespace dwave::optimization

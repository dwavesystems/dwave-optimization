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

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/common.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/interval.hpp"
#include "dwave-optimization/operator.hpp"
#include "dwave-optimization/state.hpp"

namespace dwave::optimization {

/// Apply a unary function to the predecessor node.
template <class Op>
requires(Op::nout == 1) class OperatorNode : public ArrayOutputMixin<ArrayNode> {
 public:
    template <std::convertible_to<ArrayNode*>... ArrayPtrs>
    explicit OperatorNode(ArrayPtrs... array_ptrs) : OperatorNode(Op{}, array_ptrs...) {}

    template <std::convertible_to<ArrayNode*>... ArrayPtrs>
    requires(sizeof...(ArrayPtrs) == Op::nin)
    explicit OperatorNode(Op&& op, ArrayPtrs... array_ptrs) :
        ArrayOutputMixin(broadcast_shapes(array_ptrs->shape()...)),
        op(std::move(op)),
        operands_{array_ptrs...},
        operand_strides_{
            broadcast_strides_to(array_ptrs->shape(), array_ptrs->strides(), shape())...
        },
        domain_(calculate_domain_(op, array_ptrs...)),
        integral_(calculate_integral_(array_ptrs...)),
        sizeinfo_(calculate_sizeinfo_(array_ptrs...)) {
        (add_predecessor_(array_ptrs), ...);
    }

    /// @copydoc Array::buff()
    double const* buff(const State& state) const override {
        return data_ptr_<OperatorNodeState_>(state)->buffer.data();
    }

    /// @copydoc Node::commit()
    void commit(State& state) const override {
        auto* state_ptr = data_ptr_<OperatorNodeState_>(state);
        state_ptr->diff.clear();
        state_ptr->previous_size = state_ptr->buffer.size();
    }

    /// @copydoc Array::diff()
    std::span<const Update> diff(const State& state) const override {
        return data_ptr_<OperatorNodeState_>(state)->diff;
    }

    /// @copydoc Node::equal_to()
    bool equal_to(const Node& rhs) const override {
        const OperatorNode* rhs_ptr = dynamic_cast<const OperatorNode*>(&rhs);
        if (rhs_ptr == nullptr) return false;  // not same type so not equal
        return this->equal_to(*rhs_ptr);       // use the equal_to(const OperatorNode&) overload
    }
    bool equal_to(const OperatorNode& rhs) const {
        // If we're the same type, then we just need to check we have the same predecessors
        if constexpr (Op::commutative) {
            return std::ranges::is_permutation(operands_, rhs.operands_);
        } else {
            return std::ranges::equal(operands_, rhs.operands_);
        }
    }

    /// @copydoc Node::initialize_state()
    void initialize_state(State& state) const override {
        emplace_data_ptr_<OperatorNodeState_>(state, apply_(state), shape_(state));
    }

    /// @copydoc Array::integral()
    bool integral() const override { return integral_; }

    /// @copydoc Array::max()
    double max() const override { return domain_.supremum; }

    /// @copydoc Array::min()
    double min() const override { return domain_.infimum; }

    /// The predecessor of the operation
    std::span<ArrayNode* const> operands() { return operands_; }
    std::span<const ArrayNode* const> operands() const { return operands_; }

    /// @copydoc Node::propagate()
    void propagate(State& state) const override {
        std::vector<std::span<const Update>> diffs;
        for (const ArrayNode* array_ptr : operands_) {
            const auto diff = array_ptr->diff(state);

            // If there are no changes, then don't bother saving the diff for
            // our propagation
            if (diff.empty()) continue;

            // If we have an array with size 1 and it's diff is non-empty, then
            // it's being broadcast and if it's value changed we just need to
            // recalculate the entire array
            if (array_ptr->size() == 1) {
                if (diff[0].old == diff.back().value) continue;  // net nothing changed

                auto* state_ptr = data_ptr_<OperatorNodeState_>(state);
                state_ptr->assign(apply_(state));
                if (not state_ptr->diff.empty()) Node::propagate(state);
                return;
            }

            diffs.emplace_back(std::move(diff));  // kind of pointless to move a span but...
        }

        // if (diffs.empty()) {
        //     assert(false and "not yet implemented");
        // }

        auto* state_ptr = data_ptr_<OperatorNodeState_>(state);

        if (dynamic()) {
            const auto shape = shape_(state);

            if (shape[0] != state_ptr->shape[0]) {
                const ssize_t size = state_ptr->row_size * shape[0];

                assert(0 <= size);
                if (static_cast<std::size_t>(size) < state_ptr->buffer.size()) {
                    assert(false and "not yet implemented");
                }

                // we add nans to the buffer and mark them as unseen under
                // the assumption that subsequent updates will trigger us
                // to update the buffer value and correctly form the Update.
                state_ptr->buffer.resize(size, Update::nothing);
                state_ptr->seen.resize(size, false);

                state_ptr->shape[0] = shape[0];
            }
        }

        const ssize_t size = state_ptr->buffer.size();
        assert(state_ptr->buffer.size() == state_ptr->seen.size());
        assert(state_ptr->diff.empty());

        auto views = predecessor_views_(state);  // maybe update to take shape?

        for (const auto& diff : diffs) {
            for (const auto& [index, _, __] : diff) {
                assert(0 <= index);

                if (size <= index) continue;           // we shrank
                if (state_ptr->seen[index]) continue;  // we've already recalculated this one

                auto value = std::apply([&](auto&... views) { return op(views[index]...); }, views);

                state_ptr->diff.emplace_back(index, state_ptr->buffer[index], value);
                state_ptr->buffer[index] = value;
                state_ptr->seen[index] += 1;  // use += 1 to track that we saw each one once
            }
        }

        // todo: check that all of the nans have been removed from the buffer
        assert(std::ranges::none_of(state_ptr->buffer, [](auto& val) { return std::isnan(val); }));
        assert(std::ranges::all_of(state_ptr->seen, [](auto& val) {
            return 0 <= val and val <= 1;
        }));

        if (not state_ptr->diff.empty()) Node::propagate(state);
    }

    /// @copydoc Node::revert()
    void revert(State& state) const override {
        auto* state_ptr = data_ptr_<OperatorNodeState_>(state);

        const ssize_t size = state_ptr->previous_size;

        state_ptr->buffer.resize(size);
        state_ptr->seen.resize(size);
        for (const auto& [index, old, _] : state_ptr->diff | std::views::reverse) {
            assert(0 <= index);

            if (size <= index) continue;  // we already removed this

            assert(state_ptr->seen[index] == 1);

            state_ptr->buffer[index] = old;
            state_ptr->seen[index] -= 1;

            // assert(false and "not yet implemented");
        }
        state_ptr->diff.clear();

        std::ranges::none_of(state_ptr->seen, [](auto v) { return v; });

        // now adjust the shape
        if (dynamic()) {
            assert(0 <= state_ptr->previous_num_rows_);
            assert(0 < ndim());
            state_ptr->shape[0] = state_ptr->previous_num_rows_;
        }
    }

    /// @copydoc Array::shape()
    std::span<const ssize_t> shape(const State& state) const override {
        if (not dynamic()) return shape();
        return std::span<const ssize_t>(data_ptr_<OperatorNodeState_>(state)->shape.get(), ndim());
    }
    using ArrayOutputMixin::shape;

    /// @copydoc Array::size()
    ssize_t size(const State& state) const override {
        if (not dynamic()) return size();
        return data_ptr_<OperatorNodeState_>(state)->buffer.size();
    }
    using ArrayOutputMixin::size;

    /// @copydoc Array::size_diff()
    ssize_t size_diff(const State& state) const override {
        const auto* state_ptr = data_ptr_<OperatorNodeState_>(state);
        return state_ptr->buffer.size() - state_ptr->previous_size;
    }

    /// @copydoc Array::sizeinfo()
    SizeInfo sizeinfo() const override { return this->sizeinfo_; }

 private:
    // It would be more convenient to use ArrayStateData from _state.hpp
    // but we don't currently have that in a public header.
    // If that changes in the future, or we add default state support
    // https://github.com/dwavesystems/dwave-optimization/issues/629 then
    // we should use that.
    struct OperatorNodeState_ : NodeStateData {
        OperatorNodeState_() = delete;

        template <std::ranges::range R>
        OperatorNodeState_(R&& values, std::span<const ssize_t> shape) :
            buffer(std::ranges::to<std::vector<double>>(std::forward<R>(values))),
            seen(buffer.size(), false),
            previous_size(buffer.size()),
            shape(shape.empty() ? nullptr : std::make_unique<ssize_t[]>(shape.size())),
            previous_num_rows_{shape.empty() ? -1 : shape[0]},
            row_size{
                std::ranges::fold_left(shape | std::views::drop(1), ssize_t{1}, op::multiply{})
            },
            diff() {
            std::copy(shape.begin(), shape.end(), this->shape.get());
        }

        void assign(std::vector<double> new_buffer) {
            const ssize_t overlap = std::min<ssize_t>(buffer.size(), new_buffer.size());

            for (ssize_t i = 0; i < overlap; ++i) {
                if (buffer[i] != new_buffer[i]) {
                    diff.emplace_back(i, buffer[i], new_buffer[i]);
                    seen[i] += 1;
                }
            }

            // If the new buffer is larger than the old, add the appends
            for (ssize_t i = overlap, stop = new_buffer.size(); i < stop; ++i) {
                assert(false);
            }

            // If the new buffer is shorter, then we need to work backwards
            for (ssize_t i = buffer.size() - 1, stop = overlap; stop <= i; ++i) {
                assert(false);
            }

            std::swap(buffer, new_buffer);

            // handle the size
            if (shape) {
                assert(false);
            }
        }

        std::vector<double> buffer;
        std::vector<signed char> seen;
        ssize_t previous_size;

        std::unique_ptr<ssize_t[]> shape;
        ssize_t previous_num_rows_;
        ssize_t row_size;

        std::vector<Update> diff;
    };

    // Apply op to the arrays. At some point this should be moved upstream to the operator mixins.
    std::vector<double> apply_(const State& state) const {
        // The scalar case is special because our iterators don't love 0 dimensional arrays
        // so we work with the pointer instead.
        if (this->ndim() == 0) {
            return {std::apply(
                [&](auto&&... array_ptrs) -> double { return op((*(array_ptrs->buff(state)))...); },
                operands_
            )};
        }

        // An array of views, one for each predecessor, with appropriate broadcasting
        const auto views = predecessor_views_(state);

        // Now, we zip together all of the views and apply the op
        std::vector<double> buffer;
        for (const auto& args : std::apply(std::views::zip, views)) {
            buffer.emplace_back(std::apply(op, args));
        }

        return buffer;
    }

    // Calculate the min/max of the op applied to array_ptr
    static interval<double> calculate_domain_(
        const Op& op,
        std::convertible_to<const Array*> auto... array_ptrs
    ) {
        static_assert(1 <= sizeof...(array_ptrs));

        auto interval_extension = [](const Array* ptr) {
            return interval<double>(ptr->min(), ptr->max());
        };

        // Ensure that our arrays output values within the domain of our
        // operator. Otherwise raise an error.
        {
            std::array<const Array* const, Op::nin> operands{array_ptrs...};
            for (ssize_t i = 0; i < Op::nin; ++i) {
                if (interval_extension(operands[i]) <= op.template domain<double>[i]) {
                    // then we're fine
                } else {
                    throw std::invalid_argument(
                        "operand " + std::to_string(i) +
                        "'s can take values out of range for the operator"
                    );
                }
            }
        }

        return op(interval_extension(array_ptrs)...);
    }

    // Determine whether the op applied to array_ptr will always result in an
    // integral output.
    static bool calculate_integral_(std::convertible_to<const Array*> auto... array_ptrs) {
        static_assert(1 <= sizeof...(array_ptrs));

        // rint() actually always returns a floating point. But it is an integral
        // floating point. This is a place where having proper dtypes would be
        // very very nice.
        if constexpr (std::same_as<Op, op::rint>) return true;

        // This has to be one of the least readable bits of code I've ever written.
        // The idea is that if *all* of our predecessors are integral, we want to see the type
        // that would result from `op(*(int() for _ in range(len(array_ptrs))))`. I.e., does
        // calling our op with all integer inputs result in an integer?
        // Otherwise we ask the same question, but given all double inputs.
        if (std::ranges::all_of(std::array{array_ptrs->integral()...}, std::identity{})) {
            return std::integral<decltype(op([](auto&) -> int { return {}; }(array_ptrs)...))>;
        } else {
            return std::integral<decltype(op([](auto&) -> double { return {}; }(array_ptrs)...))>;
        }
    }

    static SizeInfo calculate_sizeinfo_(
        const Array* array_ptr,
        std::convertible_to<const Array*> auto... array_ptrs
    ) {
        SizeInfo lhs_sizeinfo = array_ptr->sizeinfo();

        if constexpr (sizeof...(array_ptrs) > 0) {
            // // in C++26 we can just index the pack, but until then...
            // auto first = [](const Array* array_ptr, auto...) {
            //     return array_ptr;
            // };

            // const Array* rhs_array_ptr = first(array_ptrs...);
            SizeInfo rhs_sizeinfo = calculate_sizeinfo_(array_ptrs...);

            if (lhs_sizeinfo == rhs_sizeinfo) {
                // do nothing
            } else if (lhs_sizeinfo == 1) {
                assert(false and "not yet implemented");
            } else if (rhs_sizeinfo == 1) {
                assert(false and "not yet implemented");
            } else {
                assert(false and "not yet implemented");
            }
        }

        return lhs_sizeinfo;
    }

    auto predecessor_views_(const State& state) const {
        const auto shape = shape_(state);

        // Not the most readable code block...
        // What this is doing is creating an array of views, one for each operand array, and doing
        // the appropriate broadcasting.
        using view_type = decltype(view(state));
        return [&]<std::size_t... I>(std::index_sequence<I...>) {
            return std::array<view_type, Op::nin>{std::ranges::subrange(
                const_iterator(operands_[I]->buff(state), shape, operand_strides_[I]),
                std::default_sentinel
            )...};
        }(std::make_index_sequence<Op::nin>{});
    }

    void replace_predecessor_(ssize_t index, Node* node_ptr) override {
        Node::replace_predecessor_(index, node_ptr);
        operands_[index] = dynamic_cast<ArrayNode*>(node_ptr);
        assert(operands_[index] != nullptr);
    }

    // Get our shape by reading the relevant predecessor. This should only
    // ever be called during an initialization or a propagation.
    std::span<const ssize_t> shape_(const State& state) const {
        static_assert(Op::nin > 0);

        // TODO: rewrite this for more operands
        if constexpr (Op::nin == 1) {
            return operands_[0]->shape(state);
        } else if (Op::nin == 2) {
            const auto [lhs, rhs] = operands_;

            if (lhs->size() == 1) return rhs->shape(state);
            if (rhs->size() == 1) return lhs->shape(state);

            return lhs->shape(state);
        } else {
            assert(false and "not yet implemented");
        }
    }

    Op op;

    // There are redundant, because we could dynamic_cast each time from
    // predecessors(), but this is more performant
    std::array<ArrayNode*, Op::nin> operands_;

    std::array<std::vector<ssize_t>, Op::nin> operand_strides_;

    interval<double> domain_;  // the range of possible output values
    bool integral_;            // whether the node will always output integral values

    const SizeInfo sizeinfo_;
};

}  // namespace dwave::optimization

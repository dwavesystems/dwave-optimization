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

#include <cassert>
#include <concepts>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/functional.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/interval.hpp"
#include "dwave-optimization/state.hpp"

namespace dwave::optimization {

/// Apply a unary function to the predecessor node.
template <class UnaryOp>
class UnaryOpNode : public ArrayOutputMixin<ArrayNode> {
 public:
    template <class... Args>
    explicit UnaryOpNode(ArrayNode* node_ptr, Args&&... args) :
        ArrayOutputMixin(node_ptr->shape()),
        op(std::forward<Args>(args)...),
        array_ptr_(node_ptr),
        domain_(calculate_domain_(node_ptr)),
        integral_(calculate_integral_(node_ptr)),
        sizeinfo_(array_ptr_->sizeinfo()) {
        add_predecessor_(node_ptr);
    }

    /// @copydoc Array::buff()
    double const* buff(const State& state) const override {
        return data_ptr_<UnaryOpState_>(state)->buffer.data();
    }

    /// @copydoc Node::commit()
    void commit(State& state) const override {
        auto* state_ptr = data_ptr_<UnaryOpState_>(state);
        state_ptr->diff.clear();
        state_ptr->previous_size = state_ptr->buffer.size();
    }

    /// @copydoc Array::diff()
    std::span<const Update> diff(const State& state) const override {
        return data_ptr_<UnaryOpState_>(state)->diff;
    }

    /// @copydoc Node::equal_to()
    bool equal_to(const Node& rhs) const override {
        const UnaryOpNode* rhs_ptr = dynamic_cast<const UnaryOpNode*>(&rhs);
        if (rhs_ptr == nullptr) return false;  // not same type so not equal
        return this->equal_to(*rhs_ptr);       // use the equal_to(const UnaryOpNode&) overload
    }
    bool equal_to(const UnaryOpNode& rhs) const {
        // If we're the same type, then we just need to check we have the same predecessor
        return this->array_ptr_ == rhs.array_ptr_;
    }

    /// @copydoc Node::initialize_state()
    void initialize_state(State& state) const override {
        emplace_data_ptr_<UnaryOpState_>(
            state, array_ptr_->view(state) | std::views::transform(op), array_ptr_->shape(state)
        );
    }

    /// @copydoc Array::integral()
    bool integral() const override { return integral_; }

    /// @copydoc Array::max()
    double max() const override { return domain_.supremum; }

    /// @copydoc Array::min()
    double min() const override { return domain_.infimum; }

    /// The predecessor of the operation
    std::span<ArrayNode* const> operands() {
        assert(predecessors().size() == 1);
        return std::span<ArrayNode* const, 1>(&array_ptr_, 1);
    }
    std::span<const ArrayNode* const> operands() const {
        assert(predecessors().size() == 1);
        return std::span<const ArrayNode* const, 1>(&array_ptr_, 1);
    }

    /// @copydoc Node::propagate()
    void propagate(State& state) const override {
        const auto array_diff = array_ptr_->diff(state);

        // If there are no updates the handle, return early.
        if (array_diff.empty()) return;

        auto* state_ptr = data_ptr_<UnaryOpState_>(state);
        auto& buffer = state_ptr->buffer;
        auto& diff = state_ptr->diff;

        for (const auto& update : array_diff) {
            assert(0 <= update.index);

            if (update.removed()) {
                assert(static_cast<size_t>(update.index) + 1 == buffer.size());

                diff.emplace_back(Update::removal(update.index, buffer[update.index]));
                buffer.pop_back();
            } else if (update.placed()) {
                assert(static_cast<size_t>(update.index) == buffer.size());

                buffer.emplace_back(op(update.value));
                diff.emplace_back(Update::placement(update.index, buffer.back()));
            } else {
                assert(static_cast<size_t>(update.index) < buffer.size());

                double& old = buffer[update.index];
                double value = op(update.value);

                if (old == value) continue;  // no change to update

                diff.emplace_back(update.index, old, value);
                old = value;
            }
        }

        if (ndim()) state_ptr->shape[0] = array_ptr_->shape(state)[0];

        if (not diff.empty()) Node::propagate(state);
    }

    /// @copydoc Node::revert()
    void revert(State& state) const override {
        auto* state_ptr = data_ptr_<UnaryOpState_>(state);
        std::vector<double>& buffer = state_ptr->buffer;
        const ssize_t size = state_ptr->previous_size;
        std::vector<Update>& diff = state_ptr->diff;
        ssize_t* shape = state_ptr->shape.get();

        const ssize_t propagated_size = buffer.size();

        buffer.resize(size);
        for (const auto& [index, old, _] : diff | std::views::reverse) {
            assert(0 <= index);
            if (size <= index) continue;
            buffer[index] = old;
        }
        diff.clear();

        // Adjust our shape back to what it should be, while avoiding reading our
        // predecessor.
        if (size != propagated_size) {
            assert(0 < this->ndim());  // if our size changed this must be true
            shape[0] = buffer.size();

            // we could cache the divisor, but that feels like overkill in the context
            // of a revert
            for (ssize_t i = 1, ndim = this->ndim(); i < ndim; ++i) shape[0] /= shape[i];
        }
    }

    /// @copydoc Array::shape()
    std::span<const ssize_t> shape(const State& state) const override {
        return std::span<const ssize_t>(data_ptr_<UnaryOpState_>(state)->shape.get(), ndim());
    }
    using ArrayOutputMixin::shape;

    /// @copydoc Array::size()
    ssize_t size(const State& state) const override {
        return data_ptr_<UnaryOpState_>(state)->buffer.size();
    }
    using ArrayOutputMixin::size;

    /// @copydoc Array::size_diff()
    ssize_t size_diff(const State& state) const override {
        const auto* state_ptr = data_ptr_<UnaryOpState_>(state);
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
    struct UnaryOpState_ : NodeStateData {
        UnaryOpState_() = delete;

        template <std::ranges::range R>
        UnaryOpState_(R&& values, std::span<const ssize_t> shape) :
            buffer(std::ranges::to<std::vector<double>>(std::forward<R>(values))),
            previous_size(buffer.size()),
            shape(shape.size() ? std::make_unique<ssize_t[]>(shape.size()) : nullptr),
            diff() {
            for (ssize_t i = 0, ndim = shape.size(); i < ndim; ++i) {
                this->shape[i] = shape[i];
            }
        }

        std::vector<double> buffer;
        ssize_t previous_size;

        std::unique_ptr<ssize_t[]> shape;

        std::vector<Update> diff;
    };

    // Calculate the min/max of the op applied to array_ptr
    static interval<double> calculate_domain_(Array* array_ptr) {
        interval<double> array_domain = interval<double>(array_ptr->min(), array_ptr->max());

        if constexpr (std::same_as<UnaryOp, functional::square_root>) {
            if (not(array_domain <= UnaryOp::template domain<double>)) {
                throw std::invalid_argument("SquareRoot's predecessors must be non-negative");
            }
        }
        if constexpr (std::same_as<UnaryOp, functional::log>) {
            if (not(array_domain <= UnaryOp::template domain<double>)) {
                throw std::invalid_argument("Log's predecessors must be non-negative");
            }
        }

        // All the other ufuncs should have unbounded domains but as a sanity check...
        assert(array_domain <= UnaryOp::template domain<double>);

        // The range of our UnaryOp, i.e. the output min/max of our UnaryOpNode
        return UnaryOp{}(array_domain);
    }

    // Determine whether the op applied to array_ptr will always result in an
    // integral output.
    static bool calculate_integral_(const Array* array_ptr) {
        // rint() actually always returns a floating point. But it is an intergral
        // floating point. This is a place where having proper dtypes would be
        // very very nice.
        if constexpr (std::same_as<UnaryOp, functional::rint>) return true;

        // Otherwise, we ask our op what an integer input would result in.
        if (array_ptr->integral()) {
            return std::integral<decltype(UnaryOp{}(int()))>;
        } else {
            return std::integral<decltype(UnaryOp{}(double()))>;
        }
    }

    void replace_predecessor_(ssize_t index, Node* node_ptr) override {
        Node::replace_predecessor_(index, node_ptr);
        array_ptr_ = dynamic_cast<ArrayNode*>(node_ptr);
        assert(array_ptr_ != nullptr);
    }

    UnaryOp op;

    // This is redundant, because we could dynamic_cast each time from
    // predecessors(), but this is more performant
    ArrayNode* array_ptr_;

    interval<double> domain_;  // the range of possible output values
    bool integral_;            // whether the node will always output integral values

    const SizeInfo sizeinfo_;
};

using AbsoluteNode = UnaryOpNode<functional::absolute>;
extern template class UnaryOpNode<functional::absolute>;

using CosNode = UnaryOpNode<functional::cos>;
extern template class UnaryOpNode<functional::cos>;

using ExpNode = UnaryOpNode<functional::exp>;
extern template class UnaryOpNode<functional::exp>;

using ExpitNode = UnaryOpNode<functional::expit>;
extern template class UnaryOpNode<functional::expit>;

using LogNode = UnaryOpNode<functional::log>;
extern template class UnaryOpNode<functional::log>;

using LogicalNode = UnaryOpNode<functional::logical>;
extern template class UnaryOpNode<functional::logical>;

using NegativeNode = UnaryOpNode<functional::negative>;
extern template class UnaryOpNode<functional::negative>;

using NotNode = UnaryOpNode<functional::logical_not>;
extern template class UnaryOpNode<functional::logical_not>;

using RintNode = UnaryOpNode<functional::rint>;
extern template class UnaryOpNode<functional::rint>;

using SinNode = UnaryOpNode<functional::sin>;
extern template class UnaryOpNode<functional::sin>;

using SquareNode = UnaryOpNode<functional::square>;
extern template class UnaryOpNode<functional::square>;

using SquareRootNode = UnaryOpNode<functional::square_root>;
extern template class UnaryOpNode<functional::square_root>;

using TanhNode = UnaryOpNode<functional::tanh>;
extern template class UnaryOpNode<functional::tanh>;

}  // namespace dwave::optimization

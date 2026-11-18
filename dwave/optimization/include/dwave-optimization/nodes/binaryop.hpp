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
#include <span>
#include <stdexcept>
#include <utility>
#include <variant>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/common.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/interval.hpp"
#include "dwave-optimization/operator.hpp"
#include "dwave-optimization/state.hpp"

namespace dwave::optimization {

template <class BinaryOp>
class BinaryOpNode : public ArrayOutputMixin<EqualityMixin<ArrayNode, BinaryOpNode<BinaryOp>>> {
 public:
    BinaryOpNode(BinaryOp&& op, ArrayNode* lhs_ptr, ArrayNode* rhs_ptr) :
        ArrayOutputMixin<EqualityMixin<ArrayNode, BinaryOpNode<BinaryOp>>>(
            broadcast_shapes(lhs_ptr->shape(), rhs_ptr->shape())
        ),
        op(std::move(op)),
        operands_({lhs_ptr, rhs_ptr}),
        domain_(calculate_domain_(op, operands_[0], operands_[1])),
        integral_(calculate_integral_(op, operands_[0], operands_[1])),
        sizeinfo_(calculate_sizeinfo_(shape(), operands_[0], operands_[1])) {
        this->add_predecessor_(lhs_ptr);
        this->add_predecessor_(rhs_ptr);
    }

    BinaryOpNode(ArrayNode* lhs_ptr, ArrayNode* rhs_ptr) : BinaryOpNode({}, lhs_ptr, rhs_ptr) {}

    /// @copydoc Array::buff()
    const double* buff(const State& state) const override;

    /// @copydoc Node::commit()
    void commit(State& state) const override;

    /// @copydoc Array::diff()
    std::span<const Update> diff(const State& state) const override;

    /// Two BinaryOpNodes are equal if they have the same operation and the
    /// same predecessors. If the BinaryOp is commutative, then permutations
    /// of predecessors are allowed.
    bool equal_to(const BinaryOpNode& rhs) const override {
        if (op != rhs.op) return false;

        if constexpr (BinaryOp::commutative) {
            return std::ranges::is_permutation(operands_, rhs.operands_);
        } else {
            return std::ranges::equal(operands_, rhs.operands_);
        }
    }

    /// @copydoc Node::initialize_state()
    void initialize_state(State& state) const override;

    /// @copydoc Array::integral()
    bool integral() const override { return integral_; }

    /// @copydoc Array::max()
    double max() const override { return domain_.supremum; }

    /// @copydoc Array::min()
    double min() const override { return domain_.infimum; }

    /// The predecessors of the operation, as Array*.
    std::span<Array* const> operands() {
        assert(this->predecessors().size() == operands_.size());
        return operands_;
    }
    std::span<const Array* const> operands() const {
        assert(this->predecessors().size() == operands_.size());
        return operands_;
    }

    /// @copydoc Node::propagate()
    void propagate(State& state) const override;

    /// @copydoc Node::revert()
    void revert(State& state) const override;

    /// @copydoc Array::shape()
    std::span<const ssize_t> shape(const State& state) const override;
    using ArrayOutputMixin<EqualityMixin<ArrayNode, BinaryOpNode<BinaryOp>>>::shape;

    /// @copydoc Array::size()
    ssize_t size(const State& state) const override;
    using ArrayOutputMixin<EqualityMixin<ArrayNode, BinaryOpNode<BinaryOp>>>::size;

    /// @copydoc Array::size_diff()
    ssize_t size_diff(const State& state) const override;

    /// @copydoc Array::sizeinfo()
    SizeInfo sizeinfo() const override { return this->sizeinfo_; }

 private:
    static interval<double> calculate_domain_(
        BinaryOp& op,
        const Array* lhs_ptr,
        const Array* rhs_ptr
    ) {
        std::variant<interval<double>, interval<long long>> lhs_enclosure;
        if (lhs_ptr->integral()) {
            lhs_enclosure = interval<long long>(lhs_ptr->min(), lhs_ptr->max());
        } else {
            lhs_enclosure = interval<double>(lhs_ptr->min(), lhs_ptr->max());
        }

        std::variant<interval<double>, interval<long long>> rhs_enclosure;
        if (rhs_ptr->integral()) {
            rhs_enclosure = interval<long long>(rhs_ptr->min(), rhs_ptr->max());
        } else {
            rhs_enclosure = interval<double>(rhs_ptr->min(), rhs_ptr->max());
        }

        return std::visit(
            [&op](const auto& lhs, const auto& rhs) -> interval<double> { return op(lhs, rhs); },
            lhs_enclosure,
            rhs_enclosure
        );
    }

    static bool calculate_integral_(BinaryOp& op, const Array* lhs_ptr, const Array* rhs_ptr) {
        if (lhs_ptr->integral() and rhs_ptr->integral()) {
            return std::integral<decltype(op(int{}, int{}))>;
        } else if (lhs_ptr->integral()) {
            return std::integral<decltype(op(int{}, double{}))>;
        } else if (rhs_ptr->integral()) {
            return std::integral<decltype(op(double{}, int{}))>;
        } else {
            return std::integral<decltype(op(double{}, double{}))>;
        }
    }

    // Determine the size of the resulting array.
    // Raises an error if they are incompatible.
    // Requires that lhs and rhs be broadcastable.
    static SizeInfo calculate_sizeinfo_(
        std::span<const ssize_t> shape,
        const Array* lhs_ptr,
        const Array* rhs_ptr
    ) {
        // Do some checks to ensure these array nodes can be used together with this op
        // We support limited broadcasting - one side must be a scalar.
        // If one size is a scalar, we also support dynamic arrays.
        // Otherwise both arrays must be the same shape and not be dynamic
        if (lhs_ptr->size() == 1 or rhs_ptr->size() == 1) {
            // This is allowed. Note that in this case, since `broadcast_shapes()`
            // was called prior to `calculate_values_minmax()` in the constructor
            // of this node, we do not need to check `lhs_ptr->shape()` against
            // `rhs_ptr->shape()` since `broadcast_shapes()` has certified that we
            // can broadcast safely.
        } else if (lhs_ptr->sizeinfo().substitute(100) != rhs_ptr->sizeinfo().substitute(100)) {
            throw std::invalid_argument("arrays must have the same size or one must be a scalar");
        } else if (
            lhs_ptr->shape().size() != rhs_ptr->shape().size() or
            not std::equal(
                lhs_ptr->shape().begin(), lhs_ptr->shape().end(), rhs_ptr->shape().begin()
            )
        ) {
            throw std::invalid_argument("arrays must have the same shape or one must be a scalar");
        }

        // If the shape isn't dynamic, then the size if fixed and so is SizeInfo
        if (shape.empty() or 0 <= shape[0]) {
            return SizeInfo(std::ranges::fold_left(shape, ssize_t{1}, op::multiply{}));
        }

        if (lhs_ptr->dynamic() and rhs_ptr->dynamic()) {
            assert(lhs_ptr->sizeinfo() == rhs_ptr->sizeinfo());
            return lhs_ptr->sizeinfo();
        } else if (lhs_ptr->dynamic()) {
            assert(rhs_ptr->size() == 1);
            return lhs_ptr->sizeinfo();
        } else if (rhs_ptr->dynamic()) {
            assert(lhs_ptr->size() == 1);
            return rhs_ptr->sizeinfo();
        }

        // not possible for us to be dynamic and none of our predecessors to be
        assert(false and "not implemeted");
        std::unreachable();
    }

    void replace_predecessor_(ssize_t previous_index, Node* node_ptr) override;

    [[no_unique_address]] BinaryOp op;

    // There are redundant, because we could dynamic_cast each time from
    // predecessors(), but this is more performant
    std::array<Array*, 2> operands_;

    const interval<double> domain_;  // the range of possible output values
    const bool integral_;            // whether the node will always output integral values

    const SizeInfo sizeinfo_;
};

// We follow NumPy naming convention rather than C++ to distinguish between
// binary operations and reduce operations.
// https://numpy.org/doc/stable/reference/routines.math.html
using AddNode = BinaryOpNode<op::add>;
extern template class BinaryOpNode<op::add>;

using AndNode = BinaryOpNode<op::logical_and>;
extern template class BinaryOpNode<op::logical_and>;

using DivideNode = BinaryOpNode<op::divide>;
extern template class BinaryOpNode<op::divide>;

using EqualNode = BinaryOpNode<op::equal>;
extern template class BinaryOpNode<op::equal>;

using LessEqualNode = BinaryOpNode<op::less_equal>;
extern template class BinaryOpNode<op::less_equal>;

using MaximumNode = BinaryOpNode<op::maximum>;
extern template class BinaryOpNode<op::maximum>;

using MinimumNode = BinaryOpNode<op::minimum>;
extern template class BinaryOpNode<op::minimum>;

using ModulusNode = BinaryOpNode<op::remainder>;
extern template class BinaryOpNode<op::remainder>;

using MultiplyNode = BinaryOpNode<op::multiply>;
extern template class BinaryOpNode<op::multiply>;

using OrNode = BinaryOpNode<op::logical_or>;
extern template class BinaryOpNode<op::logical_or>;

using SubtractNode = BinaryOpNode<op::subtract>;
extern template class BinaryOpNode<op::subtract>;

using XorNode = BinaryOpNode<op::logical_xor>;
extern template class BinaryOpNode<op::logical_xor>;

}  // namespace dwave::optimization

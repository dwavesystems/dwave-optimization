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

#include <cmath>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "_state.hpp"

namespace dwave::optimization {

template <class BinaryOp>
std::pair<double, double> calculate_values_minmax(const Array* lhs_ptr, const Array* rhs_ptr) {
    // Do some checks to ensure these array nodes can be used together with this op
    // We support limited broadcasting - one side must be a scalar.
    // If one size is a scalar, we also support dynamic arrays.
    // Otherwise both arrays must be the same shape and not be dynamic
    if (lhs_ptr->size() == 1 || rhs_ptr->size() == 1) {
        // this is allowed
    } else if (lhs_ptr->sizeinfo().substitute(100) != rhs_ptr->sizeinfo().substitute(100)) {
        throw std::invalid_argument("arrays must have the same shape or one must be a scalar");
    }

    if constexpr (std::is_same<BinaryOp, std::divides<double>>::value) {
        bool strictly_negative = rhs_ptr->min() < 0 && rhs_ptr->max() < 0;
        bool strictly_positive = rhs_ptr->min() > 0 && rhs_ptr->max() > 0;
        if (!strictly_negative && !strictly_positive) {
            throw std::invalid_argument(
                    "Divide's denominator predecessor must be either strictly positive or strictly "
                    "negative");
        }
    }

    // If the output of the operation is boolean, then min/max is simple
    using result_type = typename std::invoke_result<BinaryOp, double&, double&>::type;
    if constexpr (std::same_as<result_type, bool>) {
        return {false, true};
    }

    // Otherwise the min and max depend on the predecessors

    const auto lhs_low = lhs_ptr->min();
    const auto lhs_high = lhs_ptr->max();
    const auto rhs_low = rhs_ptr->min();
    const auto rhs_high = rhs_ptr->max();

    auto op = BinaryOp();

    // these can result in inf. If we update propagation/initialization to handle
    // that case we should update these as well.
    if constexpr (std::same_as<BinaryOp, std::divides<double>> ||
                  std::same_as<BinaryOp, std::multiplies<double>>) {
        // The constructor should prevent us from getting here, but just in case...
        assert((!std::same_as<BinaryOp, std::divides<double>> || rhs_low != 0));
        assert((!std::same_as<BinaryOp, std::divides<double>> || rhs_high != 0));

        // just get all possible combinations of values
        std::array<double, 4> combos{op(lhs_low, rhs_low), op(lhs_low, rhs_high),
                                     op(lhs_high, rhs_low), op(lhs_high, rhs_high)};

        return std::make_pair(std::ranges::min(combos), std::ranges::max(combos));
    }
    if constexpr (std::same_as<BinaryOp, functional::safe_divides<double>>) {
        // safe_divide is, well, safe. So we start by calculating all combos.
        // Though there are some possible other values depending on our rhs.
        std::vector<double> combos = {op(lhs_low, rhs_low), op(lhs_low, rhs_high),
                                      op(lhs_high, rhs_low), op(lhs_high, rhs_high)};

        if (rhs_ptr->integral()) {
            if (rhs_low < 0 && 0 < rhs_high) {
                combos.emplace_back(op(1, 0));
            }
            if (rhs_low < -1 && -1 < rhs_high) {
                combos.emplace_back(op(lhs_low, -1));
                combos.emplace_back(op(lhs_high, -1));
            }
            if (rhs_low < 1 && 1 < rhs_high) {
                combos.emplace_back(op(lhs_low, 1));
                combos.emplace_back(op(lhs_high, 1));
            }
        } else {
            if (rhs_low < 0 && 0 < rhs_high) {
                // rhs's range includes zero, but it's not currently accounted for
                // so let's do that.
                // combos.emplace_back(op(1, 0));  // redundant because we're already max range
                combos.emplace_back(std::numeric_limits<double>::max());
                combos.emplace_back(std::numeric_limits<double>::lowest());
            } else if (rhs_low == 0 && rhs_high != 0) {
                // We can get very close to dividing by 0
                combos.emplace_back(std::copysign(std::numeric_limits<double>::max(), lhs_low));
                combos.emplace_back(std::copysign(std::numeric_limits<double>::max(), lhs_high));
            } else if (rhs_low != 0 && rhs_high == 0) {
                // We can get very close to dividing by -0
                combos.emplace_back(std::copysign(std::numeric_limits<double>::max(), -lhs_low));
                combos.emplace_back(std::copysign(std::numeric_limits<double>::max(), -lhs_high));
            }
        }

        return std::make_pair(std::ranges::min(combos), std::ranges::max(combos));
    }
    if constexpr (std::same_as<BinaryOp, functional::max<double>> ||
                  std::same_as<BinaryOp, functional::min<double>> ||
                  std::same_as<BinaryOp, std::plus<double>>) {
        return std::make_pair(op(lhs_low, rhs_low), op(lhs_high, rhs_high));
    }
    if constexpr (std::same_as<BinaryOp, std::minus<double>>) {
        return std::make_pair(lhs_low - rhs_high, lhs_high - rhs_low);
    }
    if constexpr (std::same_as<BinaryOp, functional::modulus<double>>) {
        // Lower bound is the smallest negative absolute value
        return std::make_pair(-rhs_high < rhs_low ? -rhs_high : rhs_low,
                              -rhs_low > rhs_high ? -rhs_low : rhs_high);
    }

    assert(false && "not implemeted yet");
    unreachable();
}

template <class BinaryOp>
bool calculate_integral(const Array* lhs_ptr, const Array* rhs_ptr) {
    using result_type = typename std::invoke_result<BinaryOp, double&, double&>::type;

    if constexpr (std::is_integral<result_type>::value) {
        return true;
    }

    // The mathematical operations require a bit more fiddling.

    if constexpr (std::is_same<BinaryOp, std::divides<double>>::value ||
                  std::is_same<BinaryOp, functional::safe_divides<double>>::value) {
        return false;
    }
    if constexpr (std::is_same<BinaryOp, functional::max<double>>::value ||
                  std::is_same<BinaryOp, functional::min<double>>::value ||
                  std::is_same<BinaryOp, std::minus<double>>::value ||
                  std::is_same<BinaryOp, functional::modulus<double>>::value ||
                  std::is_same<BinaryOp, std::multiplies<double>>::value ||
                  std::is_same<BinaryOp, std::plus<double>>::value) {
        return lhs_ptr->integral() && rhs_ptr->integral();
    }

    assert(false && "not implemeted yet");
    unreachable();
}

template <class BinaryOp>
BinaryOpNode<BinaryOp>::BinaryOpNode(ArrayNode* a_ptr, ArrayNode* b_ptr)
        : ArrayOutputMixin(broadcast_shape(a_ptr->shape(), b_ptr->shape())),
          operands_({a_ptr, b_ptr}),
          values_info_(calculate_values_minmax<BinaryOp>(operands_[0], operands_[1]),
                       calculate_integral<BinaryOp>(operands_[0], operands_[1])),
          sizeinfo_(binaryop_calculate_sizeinfo(this, operands_[0], operands_[1])) {
    this->add_predecessor(a_ptr);
    this->add_predecessor(b_ptr);
}

template <class BinaryOp>
double const* BinaryOpNode<BinaryOp>::buff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->buff();
}

template <class BinaryOp>
std::span<const Update> BinaryOpNode<BinaryOp>::diff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->diff();
}

template <class BinaryOp>
void BinaryOpNode<BinaryOp>::commit(State& state) const {
    data_ptr<ArrayNodeStateData>(state)->commit();
}

template <class BinaryOp>
void BinaryOpNode<BinaryOp>::initialize_state(State& state) const {
    auto lhs_ptr = operands_[0];
    auto rhs_ptr = operands_[1];

    std::vector<double> values;

    if (std::ranges::equal(lhs_ptr->shape(state), rhs_ptr->shape(state))) {
        // This is the easy case - all we need to do is iterate over both as flat arrays
        values.reserve(lhs_ptr->size(state));

        auto it = lhs_ptr->begin(state);
        for (const double val : rhs_ptr->view(state)) {
            values.emplace_back(op(*it, val));  // order is important
            ++it;
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
        assert(false && "not yet implemented");
        unreachable();
    }

    emplace_data_ptr<ArrayNodeStateData>(state, std::move(values));
}

template <class BinaryOp>
bool BinaryOpNode<BinaryOp>::integral() const {
    return values_info_.integral;
}

template <class BinaryOp>
double BinaryOpNode<BinaryOp>::max() const {
    return this->values_info_.max;
}

template <class BinaryOp>
double BinaryOpNode<BinaryOp>::min() const {
    return this->values_info_.min;
}

template <class BinaryOp>
void BinaryOpNode<BinaryOp>::propagate(State& state) const {
    auto ptr = data_ptr<ArrayNodeStateData>(state);

    const Array* lhs_ptr = operands_[0];
    const Array* rhs_ptr = operands_[1];

    if (std::ranges::equal(lhs_ptr->shape(state), rhs_ptr->shape(state))) {
        // The easy case, just go through both predecessors making updates.

        std::span<const Update> lhs_diff = lhs_ptr->diff(state);
        std::span<const Update> rhs_diff = rhs_ptr->diff(state);

        // Handle the dynamic case by copying and deduplicating both diffs,
        // and then get a span pointing to the copies to use instead of
        // lhs_ptr/rhs_ptr->diff directly.
        std::vector<Update> lhs_diff_copy;
        std::vector<Update> rhs_diff_copy;
        if (lhs_ptr->dynamic()) {
            assert(rhs_ptr->dynamic());
            // Copy and then deduplicate both diffs
            lhs_diff_copy.assign(lhs_ptr->diff(state).begin(), lhs_ptr->diff(state).end());
            rhs_diff_copy.assign(rhs_ptr->diff(state).begin(), rhs_ptr->diff(state).end());
            deduplicate_diff(lhs_diff_copy);
            deduplicate_diff(rhs_diff_copy);
            lhs_diff = std::span<const Update>(lhs_diff.begin(), lhs_diff.end());
            rhs_diff = std::span<const Update>(rhs_diff.begin(), rhs_diff.end());
        }

        if (lhs_diff.size() && rhs_diff.size()) {
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
                return !up.removed() && !up.placed();
            };

            ptr->update(lhs_diff | std::views::transform(apply_op));
            // For the RHS, we have already dealt with growing/shrinking the array,
            // so we just ignore all updates that are placements/removals.
            ptr->update(rhs_diff | std::views::filter(is_standard_update) |
                        std::views::transform(apply_op));
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

        if (lhs_ptr->diff(state).size()) {
            // The lhs has changed, so in this case we're probably changing
            // everything, so just overwrite the state entirely.
            auto rhs_view = rhs_ptr->view(state);
            ptr->assign(rhs_view | std::views::transform(unary_func));
        } else {
            // The lhs did not change, so go through the changes on the rhs and apply them
            auto update_func = [&unary_func](Update update) {
                if (!update.removed()) update.value = unary_func(update.value);
                return update;
            };
            ptr->update(rhs_ptr->diff(state) | std::views::transform(update_func));
        }
    } else if (rhs_ptr->size() == 1) {
        // rhs is a single value being broadcast to the lhs array

        // create a unary version of our binary op
        const double rhs = rhs_ptr->view(state).front();
        auto unary_func = std::bind(op, std::placeholders::_1, rhs);

        if (rhs_ptr->diff(state).size()) {
            // The rhs has changed, so in this case we're probably changing
            // everything, so just overwrite the state entirely.
            auto lhs_view = lhs_ptr->view(state);
            ptr->assign(lhs_view | std::views::transform(unary_func));
        } else {
            // The rhs did not change, so go through the changes on the lhs and apply them
            auto update_func = [&unary_func](Update update) {
                if (!update.removed()) update.value = unary_func(update.value);
                return update;
            };
            ptr->update(lhs_ptr->diff(state) | std::views::transform(update_func));
        }
    } else {
        // this case is complicated we need to "stretch" dimensions into eachother
        assert(false && "not yet implemented");
        unreachable();
    }

    if (ptr->diff().size()) Node::propagate(state);
}

template <class BinaryOp>
void BinaryOpNode<BinaryOp>::revert(State& state) const {
    data_ptr<ArrayNodeStateData>(state)->revert();
}

template <class BinaryOp>
std::span<const ssize_t> BinaryOpNode<BinaryOp>::shape(const State& state) const {
    if (!this->dynamic()) return this->shape();

    const ssize_t lhs_size = operands_[0]->size(state);

    if (lhs_size == operands_[1]->size(state)) return operands_[0]->shape(state);

    return (lhs_size == 1) ? operands_[1]->shape(state) : operands_[0]->shape(state);
}

template <class BinaryOp>
ssize_t BinaryOpNode<BinaryOp>::size(const State& state) const {
    if (ssize_t size = this->size(); size >= 0) {
        return size;
    }

    const ssize_t lhs_size = operands_[0]->size(state);
    const ssize_t rhs_size = operands_[1]->size(state);

    if (lhs_size == rhs_size) return lhs_size;

    return (lhs_size == 1) ? rhs_size : lhs_size;
}

template <class BinaryOp>
ssize_t BinaryOpNode<BinaryOp>::size_diff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->size_diff();
}

SizeInfo binaryop_calculate_sizeinfo(const Array* node_ptr, const Array* lhs_ptr,
                                     const Array* rhs_ptr) {
    if (!node_ptr->dynamic()) return SizeInfo(node_ptr->size());

    if (lhs_ptr->dynamic() && rhs_ptr->dynamic()) {
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
    assert(false && "not implemeted");
    unreachable();
}

template <class BinaryOp>
SizeInfo BinaryOpNode<BinaryOp>::sizeinfo() const {
    return this->sizeinfo_;
}

// Uncommented are the tested specializations
template class BinaryOpNode<std::plus<double>>;
template class BinaryOpNode<std::minus<double>>;
template class BinaryOpNode<std::multiplies<double>>;
template class BinaryOpNode<std::divides<double>>;
template class BinaryOpNode<functional::modulus<double>>;
template class BinaryOpNode<std::equal_to<double>>;
template class BinaryOpNode<std::less_equal<double>>;
template class BinaryOpNode<std::logical_and<double>>;
template class BinaryOpNode<std::logical_or<double>>;
template class BinaryOpNode<functional::logical_xor<double>>;
template class BinaryOpNode<functional::max<double>>;
template class BinaryOpNode<functional::min<double>>;
template class BinaryOpNode<functional::safe_divides<double>>;

}  // namespace dwave::optimization

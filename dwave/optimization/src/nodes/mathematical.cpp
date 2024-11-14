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

#include "dwave-optimization/nodes/mathematical.hpp"

#include <ranges>

#include "_state.hpp"
#include "dwave-optimization/utils.hpp"

namespace dwave::optimization {

// BinaryOpNode ************************************************************

template <class BinaryOp>
BinaryOpNode<BinaryOp>::BinaryOpNode(ArrayNode* a_ptr, ArrayNode* b_ptr)
        : ArrayOutputMixin(broadcast_shape(a_ptr->shape(), b_ptr->shape())),
          operands_({a_ptr, b_ptr}) {
    const Array* lhs_ptr = operands_[0];
    const Array* rhs_ptr = operands_[1];

    // We support limited broadcasting - one side must be a scalar.
    // If one size is a scalar, we also support dynamic arrays.
    // Otherwise both arrays must be the same shape and not be dynamic
    if (lhs_ptr->size() == 1 || rhs_ptr->size() == 1) {
        // this is allowed
    } else if (lhs_ptr->dynamic() || rhs_ptr->dynamic()) {
        throw std::invalid_argument("cannot perform a binary op on two dynamic arrays");
    } else if (!std::ranges::equal(lhs_ptr->shape(), rhs_ptr->shape())) {
        throw std::invalid_argument("arrays must have the same shape or one must be a scalar");
    }

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
    int index = topological_index();
    assert(index >= 0 && "must be topologically sorted");
    assert(static_cast<int>(state.size()) > index && "unexpected state length");
    assert(state[index] == nullptr && "already initialized state");

    auto lhs_ptr = operands_[0];
    auto rhs_ptr = operands_[1];

    auto func = op();
    std::vector<double> values;

    if (std::ranges::equal(lhs_ptr->shape(state), rhs_ptr->shape(state))) {
        // This is the easy case - all we need to do is iterate over both as flat arrays
        values.reserve(lhs_ptr->size(state));

        auto it = lhs_ptr->begin(state);
        for (const double& val : rhs_ptr->view(state)) {
            values.emplace_back(func(*it, val));  // order is important
            ++it;
        }

    } else if (lhs_ptr->size() == 1) {
        values.reserve(rhs_ptr->size(state));

        const double& lhs = lhs_ptr->view(state).front();

        for (const double& val : rhs_ptr->view(state)) {
            values.emplace_back(func(lhs, val));
        }

    } else if (rhs_ptr->size() == 1) {
        values.reserve(lhs_ptr->size(state));

        const double& rhs = rhs_ptr->view(state).front();

        for (const double& val : lhs_ptr->view(state)) {
            values.emplace_back(func(val, rhs));
        }

    } else {
        // this case is complicated we need to "stretch" dimensions into each other
        assert(false && "not yet implemented");
        unreachable();
    }

    state[index] = std::make_unique<ArrayNodeStateData>(std::move(values));
}

template <class BinaryOp>
bool BinaryOpNode<BinaryOp>::integral() const {
    using result_type = typename std::invoke_result<BinaryOp, double&, double&>::type;

    if constexpr (std::is_integral<result_type>::value) {
        return true;
    }

    // The mathematical operations require a bit more fiddling.

    auto lhs_ptr = operands_[0];
    auto rhs_ptr = operands_[1];

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
double BinaryOpNode<BinaryOp>::max() const {
    using result_type = typename std::invoke_result<BinaryOp, double&, double&>::type;

    if constexpr (std::is_same<result_type, bool>::value) {
        return true;
    }

    // The mathematical operations require a bit more fiddling.

    auto lhs_ptr = operands_[0];
    auto rhs_ptr = operands_[1];

    // these can result in inf. If we update propagation/initialization to handle
    // that case we should update these as well.
    if constexpr (std::is_same<BinaryOp, functional::max<double>>::value ||
                  std::is_same<BinaryOp, functional::min<double>>::value ||
                  std::is_same<BinaryOp, std::plus<double>>::value) {
        return BinaryOp()(lhs_ptr->max(), rhs_ptr->max());
    }
    if constexpr (std::is_same<BinaryOp, std::minus<double>>::value) {
        return lhs_ptr->max() - rhs_ptr->min();
    }
    if constexpr (std::is_same<BinaryOp, functional::modulus<double>>::value) {
        // Upper bound is the greatest absolute value in the divisor (rhs)
        double pos_max = rhs_ptr->max();
        double neg_max = -(rhs_ptr->min());
        return neg_max > pos_max ? neg_max : pos_max;
    }
    if constexpr (std::is_same<BinaryOp, std::multiplies<double>>::value) {
        double lhs_low = lhs_ptr->min();
        double lhs_high = lhs_ptr->max();
        double rhs_low = rhs_ptr->min();
        double rhs_high = rhs_ptr->max();

        return std::max(
                {lhs_low * rhs_low, lhs_low * rhs_high, lhs_high * rhs_low, lhs_high * rhs_high});
    }

    assert(false && "not implemeted yet");
    unreachable();
}

template <class BinaryOp>
double BinaryOpNode<BinaryOp>::min() const {
    using result_type = typename std::invoke_result<BinaryOp, double&, double&>::type;

    if constexpr (std::is_same<result_type, bool>::value) {
        return false;
    }

    // The mathematical operations require a bit more fiddling.

    auto lhs_ptr = operands_[0];
    auto rhs_ptr = operands_[1];

    // these can result in inf. If we update propagation/initialization to handle
    // that case we should update these as well.
    if constexpr (std::is_same<BinaryOp, functional::max<double>>::value ||
                  std::is_same<BinaryOp, functional::min<double>>::value ||
                  std::is_same<BinaryOp, std::plus<double>>::value) {
        return BinaryOp()(lhs_ptr->min(), rhs_ptr->min());
    }
    if constexpr (std::is_same<BinaryOp, std::minus<double>>::value) {
        return lhs_ptr->min() - rhs_ptr->max();
    }
    if constexpr (std::is_same<BinaryOp, functional::modulus<double>>::value) {
        // Lower bound is the smallest negative absolute value
        double neg_min = rhs_ptr->min();
        double pos_min = -(rhs_ptr->max());
        return pos_min < neg_min ? pos_min : neg_min;
    }
    if constexpr (std::is_same<BinaryOp, std::multiplies<double>>::value) {
        double lhs_low = lhs_ptr->min();
        double lhs_high = lhs_ptr->max();
        double rhs_low = rhs_ptr->min();
        double rhs_high = rhs_ptr->max();

        return std::min(
                {lhs_low * rhs_low, lhs_low * rhs_high, lhs_high * rhs_low, lhs_high * rhs_high});
    }

    assert(false && "not implemeted yet");
    unreachable();
}

template <class BinaryOp>
void BinaryOpNode<BinaryOp>::propagate(State& state) const {
    auto ptr = data_ptr<ArrayNodeStateData>(state);

    const Array* lhs_ptr = operands_[0];
    const Array* rhs_ptr = operands_[1];

    auto func = op();
    auto& values = ptr->buffer;
    auto& changes = ptr->updates;

    if (std::ranges::equal(lhs_ptr->shape(state), rhs_ptr->shape(state))) {
        // The easy case, just go through both predecessors making updates.
        if (lhs_ptr->diff(state).size() && rhs_ptr->diff(state).size()) {
            // Both modified
            auto lit = lhs_ptr->begin(state);
            auto rit = rhs_ptr->begin(state);

            // go through both diffs, we may touch some of the indices twice
            // in which case we do a redundant recalculation, but we don't
            // save the diff
            for (const auto& [index, _, __] : lhs_ptr->diff(state)) {
                double old = values[index];
                values[index] = func(*(lit + index), *(rit + index));
                if (values[index] != old) {
                    changes.emplace_back(index, old, values[index]);
                }
            }
            for (const auto& [index, _, __] : rhs_ptr->diff(state)) {
                double old = values[index];
                values[index] = func(*(lit + index), *(rit + index));
                if (values[index] != old) {
                    changes.emplace_back(index, old, values[index]);
                }
            }
        } else if (lhs_ptr->diff(state).size()) {
            // LHS modified, but not RHS
            auto rit = rhs_ptr->begin(state);
            for (const auto& [index, _, value] : lhs_ptr->diff(state)) {
                double old = values[index];
                values[index] = func(value, *(rit + index));
                changes.emplace_back(index, old, values[index]);
            }
        } else if (rhs_ptr->diff(state).size()) {
            // RHS modified, but not LHS
            auto lit = lhs_ptr->begin(state);
            for (const auto& [index, _, value] : rhs_ptr->diff(state)) {
                double old = values[index];
                values[index] = func(*(lit + index), value);
                changes.emplace_back(index, old, values[index]);
            }
        }
    } else if (lhs_ptr->size() == 1) {
        // lhs is a single value being broadcast to the rhs array.

        // Create a unary version of our binary op.
        const double& lhs = lhs_ptr->view(state).front();
        auto unary_func = std::bind(func, lhs, std::placeholders::_1);

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
        const double& rhs = rhs_ptr->view(state).front();
        auto unary_func = std::bind(func, std::placeholders::_1, rhs);

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

    if (ptr->updates.size()) Node::propagate(state);
}

template <class BinaryOp>
void BinaryOpNode<BinaryOp>::revert(State& state) const {
    data_ptr<ArrayNodeStateData>(state)->revert();
}

template <class BinaryOp>
std::span<const ssize_t> BinaryOpNode<BinaryOp>::shape(const State& state) const {
    if (!this->dynamic()) return this->shape();

    const ssize_t lhs_size = operands_[0]->size(state);

    // we don't (yet) support other cases
    assert(lhs_size == 1 || operands_[1]->size(state) == 1);

    return (lhs_size == 1) ? operands_[1]->shape(state) : operands_[0]->shape(state);
}

template <class BinaryOp>
ssize_t BinaryOpNode<BinaryOp>::size(const State& state) const {
    if (ssize_t size = this->size(); size >= 0) {
        return size;
    }

    const ssize_t lhs_size = operands_[0]->size(state);
    const ssize_t rhs_size = operands_[1]->size(state);

    // we don't (yet) support other cases
    assert(lhs_size == 1 || rhs_size == 1);

    return (lhs_size == 1) ? rhs_size : lhs_size;
}

template <class BinaryOp>
ssize_t BinaryOpNode<BinaryOp>::size_diff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->size_diff();
}

template <class BinaryOp>
SizeInfo BinaryOpNode<BinaryOp>::sizeinfo() const {
    if (!dynamic()) return SizeInfo(size());

    const Array* lhs_ptr = operands_[0];
    const Array* rhs_ptr = operands_[1];

    if (lhs_ptr->dynamic() && rhs_ptr->dynamic()) {
        // not (yet) possible for both predecessors to be dynamic
        assert(false && "not implemeted");
        unreachable();
    } else if (lhs_ptr->dynamic()) {
        assert(rhs_ptr->size() == 1);
        return SizeInfo(lhs_ptr);
    } else if (rhs_ptr->dynamic()) {
        assert(lhs_ptr->size() == 1);
        return SizeInfo(rhs_ptr);
    }

    // not possible for us to be dynamic and none of our predecessors to be
    assert(false && "not implemeted");
    unreachable();
}

// Uncommented are the tested specializations
template class BinaryOpNode<std::plus<double>>;
template class BinaryOpNode<std::minus<double>>;
template class BinaryOpNode<std::multiplies<double>>;
// template class BinaryOpNode<std::divides<double>>;
template class BinaryOpNode<functional::modulus<double>>;
template class BinaryOpNode<std::equal_to<double>>;
// template class BinaryOpNode<std::not_equal_to<double>>;
// template class BinaryOpNode<std::greater<double>>;
// template class BinaryOpNode<std::less<double>>;
// template class BinaryOpNode<std::greater_equal<double>>;
template class BinaryOpNode<std::less_equal<double>>;
template class BinaryOpNode<std::logical_and<double>>;
template class BinaryOpNode<std::logical_or<double>>;
template class BinaryOpNode<functional::logical_xor<double>>;
template class BinaryOpNode<functional::max<double>>;
template class BinaryOpNode<functional::min<double>>;

// NaryOpNode *****************************************************************

template <class BinaryOp>
struct InverseOp {
    static bool constexpr exists() { return false; }

    double op [[noreturn]] (const double& x, const double& y) {
        assert(false && "op has no inverse");
        throw std::logic_error("op has no inverse");
    }
};

template <>
struct InverseOp<std::plus<double>> {
    static bool constexpr exists() { return true; }

    double op(const double& x, const double& y) { return x - y; }
};

template <>
struct InverseOp<std::multiplies<double>> {
    static bool constexpr exists() { return true; }

    double op(const double& x, const double& y) { return x / y; }
};

struct NaryOpNodeData : public ArrayNodeStateData {
    explicit NaryOpNodeData(std::vector<double> values, std::vector<ArrayIterator> iterators)
            : ArrayNodeStateData(std::move(values)), iterators(std::move(iterators)) {}

    // used to avoid reallocating memory for predecessor iterators every propagation
    std::vector<ArrayIterator> iterators;
};

template <class BinaryOp>
NaryOpNode<BinaryOp>::NaryOpNode(ArrayNode* node_ptr) : ArrayOutputMixin(node_ptr->shape()) {
    add_node(node_ptr);
}

// Enforce that the given span is nonempty and return the first element
ArrayNode* nonempty(std::span<ArrayNode*> node_ptrs) {
    if (node_ptrs.empty()) {
        throw std::invalid_argument("Must supply at least one predecessor");
    }
    return node_ptrs[0];
}

template <class BinaryOp>
NaryOpNode<BinaryOp>::NaryOpNode(std::span<ArrayNode*> node_ptrs)
        : ArrayOutputMixin(nonempty(node_ptrs)->shape()) {
    for (ArrayNode* ptr : node_ptrs) {
        add_node(ptr);
    }
}

template <class BinaryOp>
void NaryOpNode<BinaryOp>::add_node(ArrayNode* node_ptr) {
    if (this->topological_index() >= 0 && node_ptr->topological_index() >= 0 &&
        this->topological_index() < node_ptr->topological_index()) {
        throw std::logic_error("this operation would invalidate the topological ordering");
    }

    if (node_ptr->dynamic()) {
        throw std::invalid_argument("arrays must not be dynamic");
    }

    if (!std::ranges::equal(this->shape(), node_ptr->shape())) {
        throw std::invalid_argument("arrays must all be the same shape");
    }

    this->add_predecessor(node_ptr);
    operands_.emplace_back(node_ptr);
}

template <class BinaryOp>
double const* NaryOpNode<BinaryOp>::buff(const State& state) const {
    return data_ptr<NaryOpNodeData>(state)->buff();
}

template <class BinaryOp>
std::span<const Update> NaryOpNode<BinaryOp>::diff(const State& state) const {
    return data_ptr<NaryOpNodeData>(state)->diff();
}

template <class BinaryOp>
bool NaryOpNode<BinaryOp>::integral() const {
    using result_type = typename std::invoke_result<BinaryOp, double&, double&>::type;

    if constexpr (std::is_integral<result_type>::value) {
        return true;
    }
    if constexpr (std::is_same<BinaryOp, functional::max<double>>::value ||
                  std::is_same<BinaryOp, functional::min<double>>::value ||
                  std::is_same<BinaryOp, std::multiplies<double>>::value ||
                  std::is_same<BinaryOp, std::plus<double>>::value) {
        return std::ranges::all_of(operands_, [](const Array* ptr) { return ptr->integral(); });
    }

    assert(false && "not implemeted yet");
    unreachable();
}

template <class BinaryOp>
double NaryOpNode<BinaryOp>::max() const {
    using result_type = typename std::invoke_result<BinaryOp, double&, double&>::type;

    if constexpr (std::is_same<result_type, bool>::value) {
        return true;
    }

    // these can result in inf. If we update propagation/initialization to handle
    // that case we should update these as well.
    if constexpr (std::is_same<BinaryOp, std::plus<double>>::value) {
        return std::accumulate(operands_.begin(), operands_.end(), 0.0,
                               [](double lhs, const Array* rhs) { return lhs + rhs->max(); });
    }
    if constexpr (std::is_same<BinaryOp, functional::max<double>>::value ||
                  std::is_same<BinaryOp, functional::min<double>>::value) {
        assert(operands_.size() >= 1);  // checked by constructor
        auto cmp = op();
        double lhs = operands_[0]->max();
        for (const Array* rhs_ptr : operands_ | std::views::drop(1)) {
            lhs = cmp(lhs, rhs_ptr->max());
        }
        return lhs;
    }
    if constexpr (std::is_same<BinaryOp, std::multiplies<double>>::value) {
        // we need to be a bit careful with our signs
        assert(operands_.size() > 0);

        double lhs_low = operands_[0]->min();
        double lhs_high = operands_[0]->max();

        for (const Array* rhs_ptr : operands_ | std::views::drop(1)) {
            double rhs_low = rhs_ptr->min();
            double rhs_high = rhs_ptr->max();

            double tmp = std::min({lhs_low * rhs_low, lhs_low * rhs_high, lhs_high * rhs_low,
                                   lhs_high * rhs_high});

            lhs_high = std::max({lhs_low * rhs_low, lhs_low * rhs_high, lhs_high * rhs_low,
                                 lhs_high * rhs_high});

            lhs_low = tmp;
        }

        return lhs_high;
    }

    assert(false && "not implemeted yet");
    unreachable();
}

template <class BinaryOp>
double NaryOpNode<BinaryOp>::min() const {
    using result_type = typename std::invoke_result<BinaryOp, double&, double&>::type;

    if constexpr (std::is_same<result_type, bool>::value) {
        return false;
    }

    // these can result in inf. If we update propagation/initialization to handle
    // that case we should update these as well.
    if constexpr (std::is_same<BinaryOp, std::plus<double>>::value) {
        return std::accumulate(operands_.begin(), operands_.end(), 0.0,
                               [](double lhs, const Array* rhs) { return lhs + rhs->min(); });
    }
    if constexpr (std::is_same<BinaryOp, functional::max<double>>::value ||
                  std::is_same<BinaryOp, functional::min<double>>::value) {
        assert(operands_.size() >= 1);  // checked by constructor
        auto cmp = op();
        double lhs = operands_[0]->min();
        for (const Array* rhs_ptr : operands_ | std::views::drop(1)) {
            lhs = cmp(lhs, rhs_ptr->min());
        }
        return lhs;
    }
    if constexpr (std::is_same<BinaryOp, std::multiplies<double>>::value) {
        // we need to be a bit careful with our signs
        assert(operands_.size() > 0);

        double lhs_low = operands_[0]->min();
        double lhs_high = operands_[0]->max();

        for (const Array* rhs_ptr : operands_ | std::views::drop(1)) {
            double rhs_low = rhs_ptr->min();
            double rhs_high = rhs_ptr->max();

            double tmp = std::min({lhs_low * rhs_low, lhs_low * rhs_high, lhs_high * rhs_low,
                                   lhs_high * rhs_high});

            lhs_high = std::max({lhs_low * rhs_low, lhs_low * rhs_high, lhs_high * rhs_low,
                                 lhs_high * rhs_high});

            lhs_low = tmp;
        }

        return lhs_low;
    }

    assert(false && "not implemeted yet");
    unreachable();
}

template <class BinaryOp>
void NaryOpNode<BinaryOp>::commit(State& state) const {
    data_ptr<NaryOpNodeData>(state)->commit();
}

template <class BinaryOp>
void NaryOpNode<BinaryOp>::revert(State& state) const {
    data_ptr<NaryOpNodeData>(state)->revert();
}

template <class BinaryOp>
void NaryOpNode<BinaryOp>::initialize_state(State& state) const {
    int index = topological_index();
    assert(index >= 0 && "must be topologically sorted");
    assert(static_cast<int>(state.size()) > index && "unexpected state length");
    assert(state[index] == nullptr && "already initialized state");

    auto func = op();
    std::vector<double> values;

    values.reserve(size());

    std::vector<ArrayIterator> iterators;
    for (const Array* array_ptr : operands_) {
        iterators.push_back(array_ptr->begin(state));
    }

    auto& first_it = iterators[0];
    const auto& first_end = operands_[0]->end(state);

    while (first_it != first_end) {
        // get the value from the first node
        auto val = *first_it;
        ++first_it;

        // reduce the values from the rest of the nodes
        for (auto& it : iterators | std::views::drop(1)) {
            val = func(*it, val);
            ++it;
        }
        values.emplace_back(val);
    }

    state[index] = std::make_unique<NaryOpNodeData>(std::move(values), std::move(iterators));
}

template <class BinaryOp>
void NaryOpNode<BinaryOp>::propagate(State& state) const {
    auto node_data = data_ptr<NaryOpNodeData>(state);

    auto func = op();
    auto& values = node_data->buffer;
    auto& changes = node_data->updates;
    auto& iterators = node_data->iterators;

    std::vector<ssize_t> recompute_indices;

    if constexpr (!InverseOp<op>().exists()) {
        // have to recompute from all predecessors on any changed index
        for (const Array* input : operands_) {
            if (input->diff(state).size()) {
                for (const auto& [index, _, __] : input->diff(state)) {
                    recompute_indices.push_back(index);
                }
            }
        }

    } else {
        auto inv_func = InverseOp<op>();
        for (const Array* input : operands_) {
            if (input->diff(state).size()) {
                for (const auto& [index, old_val, new_val] : input->diff(state)) {
                    double new_reduced_val = values[index];
                    new_reduced_val = inv_func.op(new_reduced_val, old_val);
                    if (std::isnan(new_reduced_val) || std::isinf(new_reduced_val)) {
                        // calling the inverse has failed (such as divide by zero),
                        // need to fully recompute this index
                        recompute_indices.push_back(index);
                        continue;
                    }

                    double old_reduced = values[index];
                    values[index] = func(new_reduced_val, new_val);
                    changes.emplace_back(index, old_reduced, values[index]);
                }
            }
        }
    }

    // now fully recompute any needed indices
    if (recompute_indices.size()) {
        iterators.clear();
        for (const Array* node_ptr : operands_) {
            iterators.push_back(node_ptr->begin(state));
        }

        for (const auto& index : recompute_indices) {
            double old = values[index];
            double& val = values[index];
            val = *(iterators[0] + index);
            for (auto it : iterators | std::views::drop(1)) {
                val = func(*(it + index), val);
            }

            if (val != old) {
                changes.emplace_back(index, old, val);
            }
        }
    }
}

template class NaryOpNode<functional::max<double>>;
template class NaryOpNode<functional::min<double>>;
template class NaryOpNode<std::multiplies<double>>;
template class NaryOpNode<std::plus<double>>;

// PartialReduceNode *****************************************************************
std::vector<ssize_t> partial_reduce_shape(const std::span<const ssize_t> input_shape,
                                          const ssize_t axis) {
    std::vector<ssize_t> shape(input_shape.begin(), input_shape.end());
    shape.erase(shape.begin() + axis);
    return shape;
}

std::vector<ssize_t> as_contiguous_strides(const std::span<const ssize_t> shape) {
    ssize_t ndim = static_cast<ssize_t>(shape.size());

    assert(ndim > 0);
    std::vector<ssize_t> strides(ndim);
    // otherwise strides are a function of the shape
    strides[ndim - 1] = sizeof(double);
    for (auto i = ndim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

std::span<const ssize_t> nonempty(std::span<const ssize_t> span) {
    if (span.empty()){
        throw std::invalid_argument("Input span should not be empty");
    }
    return span;
}

/// TODO: support multiple axes
template <class BinaryOp>
PartialReduceNode<BinaryOp>::PartialReduceNode(ArrayNode* node_ptr, std::span<const ssize_t> axes,
                                               double init)
        : ArrayOutputMixin(partial_reduce_shape(node_ptr->shape(), nonempty(axes)[0])),
          init(init),
          array_ptr_(node_ptr),
          axes_(make_axes(axes)),
          parent_strides_c_(as_contiguous_strides(array_ptr_->shape())) {
    if (array_ptr_->dynamic()) {
        throw std::invalid_argument("cannot do a partial reduction on a dynamic array");
    } else if (array_ptr_->size() < 1) {
        throw std::invalid_argument("cannot do a partial reduction on an empty array");
    }

    // Validate axes
    /// TODO: support negative axes
    if (axes.size() != 1){
        throw std::invalid_argument("Partial reduction support only one axis");
    }

    assert(this->ndim() == array_ptr_->ndim() - 1);
    ssize_t axis = axes_[0];
    if (axis < 0 || axis >= array_ptr_->ndim()) {
        throw std::invalid_argument("Axes should be integers between 0 and n_dim - 1");
    }

    add_predecessor(node_ptr);
}

template <class BinaryOp>
PartialReduceNode<BinaryOp>::PartialReduceNode(ArrayNode* node_ptr,
                                               std::initializer_list<ssize_t> axes, double init)
        : PartialReduceNode(node_ptr, std::span(axes), init) {}

template <class BinaryOp>
PartialReduceNode<BinaryOp>::PartialReduceNode(ArrayNode* node_ptr, const ssize_t axis, double init)
        : PartialReduceNode(node_ptr, {axis}, init) {}

template <>
PartialReduceNode<std::plus<double>>::PartialReduceNode(ArrayNode* array_ptr, const ssize_t axis)
        : PartialReduceNode(array_ptr, axis, 0) {}

template <>
PartialReduceNode<std::plus<double>>::PartialReduceNode(ArrayNode* array_ptr, std::span<const ssize_t> axes)
        : PartialReduceNode(array_ptr, nonempty(axes)[0], 0) {
    if (axes.size() != 1){
        throw std::invalid_argument("Partial sum supports only one axis");
    }
}

template <class BinaryOp>
PartialReduceNode<BinaryOp>::PartialReduceNode(ArrayNode* array_ptr,
                                               std::initializer_list<ssize_t> axes)
        : PartialReduceNode(array_ptr, std::span(axes)) {}

template <class BinaryOp>
PartialReduceNode<BinaryOp>::PartialReduceNode(ArrayNode* array_ptr, const ssize_t axis)
        : PartialReduceNode(array_ptr, {axis}) {}

template <class BinaryOp>
std::span<const ssize_t> PartialReduceNode<BinaryOp>::axes() const {

    // TODO: support multiple axes
    return std::span(axes_.get(), 1);
}

template <class BinaryOp>
double const* PartialReduceNode<BinaryOp>::buff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->buff();
}

template <class BinaryOp>
void PartialReduceNode<BinaryOp>::commit(State& state) const {
    data_ptr<ArrayNodeStateData>(state)->commit();
}

template <class BinaryOp>
std::span<const Update> PartialReduceNode<BinaryOp>::diff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->diff();
}

template <class BinaryOp>
void PartialReduceNode<BinaryOp>::initialize_state(State& state) const {
    int index = topological_index();
    assert(index >= 0 && "must be topologically sorted");
    assert(static_cast<int>(state.size()) > index && "unexpected state length");
    assert(state[index] == nullptr && "already initialized state");

    std::vector<double> values(size(state));

    for (ssize_t i = 0, stop = size(state); i < stop; ++i) {
        assert(i < static_cast<ssize_t>(values.size()));
        values[i] = reduce(state, i);
    }

    state[index] = std::make_unique<ArrayNodeStateData>(std::move(values));
}

template <class BinaryOp>
bool PartialReduceNode<BinaryOp>::integral() const {
    using result_type = typename std::invoke_result<BinaryOp, double&, double&>::type;

    if constexpr (std::is_integral<result_type>::value) {
        return true;
    }

    if constexpr (std::is_same<BinaryOp, std::plus<double>>::value) {
        // the actual value of init doesn't matter, all of the above have no default
        // or an integer default init
        return array_ptr_->integral() && is_integer(init.value_or(0));
    }

    assert(false && "not implemented yet");
    unreachable();
}

template <class BinaryOp>
ssize_t PartialReduceNode<BinaryOp>::map_parent_index(const State& state,
                                                      ssize_t parent_flat_index) const {
    ssize_t axis = this->axes_[0];
    assert(axis >= 0 && axis < array_ptr_->size(state));
    assert(parent_flat_index >= 0 && parent_flat_index < array_ptr_->size(state));

    ssize_t index = 0;  // the linear index corresponding to the parent index being updated
    ssize_t current_parent_axis = 0;  // current parent axis visited
    ssize_t current_axis = 0;         // current axis

    for (auto stride : parent_strides_c_) {
        // calculate parent index on the current axis (not the linear one)
        ssize_t this_axis_index = parent_flat_index / (stride / array_ptr_->itemsize());

        // update the index now
        parent_flat_index = parent_flat_index % (stride / array_ptr_->itemsize());

        if (current_parent_axis == axis) {
            current_parent_axis++;
            continue;
        }

        // now do the calculation of the linear index of reduction
        index += this_axis_index * (this->strides()[current_axis] / this->itemsize());

        // increase the axis visited
        current_axis++;
        current_parent_axis++;
    }

    return index;
}

template <class BinaryOp>
double PartialReduceNode<BinaryOp>::max() const {
    using result_type = typename std::invoke_result<BinaryOp, double&, double&>::type;

    if constexpr (std::is_same<result_type, bool>::value) {
        return true;
    }

    ssize_t axis = axes_[0];
    if constexpr (std::is_same<BinaryOp, std::plus<double>>::value) {
        return this->init.value_or(0) + array_ptr_->shape()[axis] * array_ptr_->max();
    }

    assert(false && "not implemented yet");
    unreachable();
}

template <class BinaryOp>
double PartialReduceNode<BinaryOp>::min() const {
    using result_type = typename std::invoke_result<BinaryOp, double&, double&>::type;

    if constexpr (std::is_same<result_type, bool>::value) {
        return false;
    }

    ssize_t axis = axes_[0];
    if constexpr (std::is_same<BinaryOp, std::plus<double>>::value) {
        return this->init.value_or(0) + array_ptr_->shape()[axis] * array_ptr_->min();
    }

    assert(false && "not implemented yet");
    unreachable();
}

template <>
void PartialReduceNode<std::plus<double>>::propagate(State& state) const {
    auto ptr = data_ptr<ArrayNodeStateData>(state);

    auto& values = ptr->buffer;
    auto& changes = ptr->updates;

    for (const auto& [p_index, old, value] : array_ptr_->diff(state)) {
        const ssize_t index = map_parent_index(state, p_index);
        const double previous = values[index];
        values[index] += (value - old);
        changes.emplace_back(index, previous, values[index]);
    }

    if (ptr->updates.size()) Node::propagate(state);
}

template <class BinaryOp>
double PartialReduceNode<BinaryOp>::reduce(const State& state, ssize_t index) const {
    /// We wish to fill `index` of the partial reduction, given the state of the parent. We proceed
    /// as follows.
    /// 1. We unravel the index. This will give us e.g. the indices i, k for
    ///     R_ik = sum_j P_ijk
    /// 2. Then we know we have to iterate through the parent starting from j=0 to its full
    /// dimension. To do so, we first find the starting index in the parent.
    /// 3. We then create an iterator through the parent array along the axis.
    /// 4. We use the above iterators to perform the reduction.
    assert(index >= 0 && index < this->size(state));

    // 1. Unravel the index we are trying to fill
    std::vector<ssize_t> indices = unravel_index(this->strides(), index);
    assert(static_cast<ssize_t>(indices.size()) == this->ndim());

    // 2. Find the respective starting index in the parent
    ssize_t axis = axes_[0];
    ssize_t start_idx = 0;
    for (ssize_t ax = 0; ax < this->ndim(); ++ax) {
        if (ax >= axis) {
            start_idx += indices[ax] * array_ptr_->strides()[ax + 1] / array_ptr_->itemsize();
        } else {
            start_idx += indices[ax] * array_ptr_->strides()[ax] / array_ptr_->itemsize();
        }
    }
    assert(start_idx >= 0 && index < this->size(state));

    // 3. We create an iterator that starts from index just found and iterates through the axis we
    // are reducing over
    ArrayIterator begin =
            ArrayIterator(array_ptr_->buff(state) + start_idx, 1, &array_ptr_->shape()[axis],
                          &array_ptr_->strides()[axis]);

    ArrayIterator end = begin + array_ptr_->shape(state)[axis];

    // Get the initial value
    double init;
    if (this->init.has_value()) {
        init = this->init.value();
    } else {
        // if there is no init, we use the first value in the array as the init
        // we should only be here if the array is not dynamic and if it has at
        // least one entry
        assert(!array_ptr_->dynamic());
        assert(array_ptr_->size(state) >= 1);
        init = *begin;
        ;
        ++begin;
    }

    /// 4. Iterate through the parent on the correct axis.
    return std::reduce(begin, end, init, BinaryOp());
}

template <class BinaryOp>
void PartialReduceNode<BinaryOp>::revert(State& state) const {
    data_ptr<ArrayNodeStateData>(state)->revert();
}

template <class BinaryOp>
std::span<const ssize_t> PartialReduceNode<BinaryOp>::shape(const State& state) const {
    return this->shape();
}

template <class BinaryOp>
ssize_t PartialReduceNode<BinaryOp>::size(const State& state) const {
    return this->size();
}

template <class BinaryOp>
ssize_t PartialReduceNode<BinaryOp>::size_diff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->size_diff();
}

// Uncommented are the tested specializations
template class PartialReduceNode<std::plus<double>>;

// ReduceNode *****************************************************************

// Create a storage class for op-specific values.
template <class BinaryOp>
struct ExtraData {
    void commit() {}
    void revert() {}
};

template <>
struct ExtraData<std::logical_and<double>> {
    explicit ExtraData(ssize_t num_zero) : num_zero(num_zero) {}

    virtual ~ExtraData() = default;

    void commit() { old_num_zero = num_zero; }
    void revert() { num_zero = old_num_zero; }

    ssize_t num_zero;
    ssize_t old_num_zero = num_zero;
};

template<>
struct ExtraData<std::logical_or<double>> {
    explicit ExtraData(ssize_t num_nonzero) : num_nonzero(num_nonzero) {}

    virtual ~ExtraData() = default;

    void commit() { old_num_nonzero = num_nonzero; }
    void revert() { num_nonzero = old_num_nonzero; }

    ssize_t num_nonzero;  // num nonzero, following SciPy's naming
    ssize_t old_num_nonzero = num_nonzero;
};

template <>
struct ExtraData<std::multiplies<double>> {
    ExtraData(double nonzero, ssize_t num_zero) : nonzero(nonzero), num_zero(num_zero) {}

    virtual ~ExtraData() = default;

    void commit() {
        old_nonzero = nonzero;
        old_num_zero = num_zero;
    }
    void revert() {
        nonzero = old_nonzero;
        num_zero = old_num_zero;
    }

    // Track all the non-zero stuff. That is, if we changes all of the 0s
    // to 1s, this is the value we would see
    double nonzero;
    double old_nonzero = nonzero;

    // For multiplies we want to track the number of 0s, because when this number
    // is positive then the output is always 0 regardless of what changed.
    ssize_t num_zero;
    ssize_t old_num_zero = num_zero;
};

template <class BinaryOp>
struct ReduceNodeData : NodeStateData {
    template <class... Ts>
    explicit ReduceNodeData(double value, Ts... extra_args)
            : values(0, value, value), extra(extra_args...) {}

    double const* buff() const { return &values.value; }

    void commit() {
        values.old = values.value;
        extra.commit();
    }
    void revert() {
        values.value = values.old;
        extra.revert();
    }

    Update values;

    // op-specific storage.
    ExtraData<BinaryOp> extra;
};

template <class BinaryOp>
ReduceNode<BinaryOp>::ReduceNode(ArrayNode* node_ptr, double init)
        : init(init), array_ptr_(node_ptr) {
    add_predecessor(node_ptr);
}

template <>
ReduceNode<std::logical_and<double>>::ReduceNode(ArrayNode* array_ptr) : ReduceNode(array_ptr, 1) {}

template <>
ReduceNode<std::logical_or<double>>::ReduceNode(ArrayNode* array_ptr) : ReduceNode(array_ptr, 0) {}

template <>
ReduceNode<std::multiplies<double>>::ReduceNode(ArrayNode* array_ptr) : ReduceNode(array_ptr, 1) {}

template <>
ReduceNode<std::plus<double>>::ReduceNode(ArrayNode* array_ptr) : ReduceNode(array_ptr, 0) {}

template <class BinaryOp>
ReduceNode<BinaryOp>::ReduceNode(ArrayNode* array_ptr)
        : init(), array_ptr_(array_ptr) {
    if (array_ptr_->dynamic()) {
        throw std::invalid_argument(
                "cannot do a reduction on a dynamic array with an operation that has no identity "
                "without supplying an initial value");
    } else if (array_ptr_->size() < 1) {
        throw std::invalid_argument(
                "cannot do a reduction on an empty array with an operation that has no identity "
                "without supplying an initial value");
    }

    add_predecessor(array_ptr);
}

template <class BinaryOp>
void ReduceNode<BinaryOp>::commit(State& state) const {
    data_ptr<ReduceNodeData<op>>(state)->commit();
}

template <class BinaryOp>
double const* ReduceNode<BinaryOp>::buff(const State& state) const {
    return data_ptr<ReduceNodeData<op>>(state)->buff();
}

template <class BinaryOp>
std::span<const Update> ReduceNode<BinaryOp>::diff(const State& state) const {
    const Update& update = data_ptr<ReduceNodeData<op>>(state)->values;
    return std::span<const Update>(&update, static_cast<int>(update.old != update.value));
}

template <class BinaryOp>
void ReduceNode<BinaryOp>::initialize_state(State& state) const {
    int index = topological_index();
    assert(index >= 0 && "must be topologically sorted");
    assert(static_cast<int>(state.size()) > index && "unexpected state length");
    assert(state[index] == nullptr && "already initialized state");

    state[index] = std::make_unique<ReduceNodeData<op>>(reduce(state));
}

template <>
void ReduceNode<std::logical_and<double>>::initialize_state(State& state) const {
    int index = topological_index();
    assert(index >= 0 && "must be topologically sorted");
    assert(static_cast<int>(state.size()) > index && "unexpected state length");
    assert(state[index] == nullptr && "already initialized state");

    ssize_t num_zero = init.value_or(1) ? 0 : 1;
    for (const double& value : array_ptr_->view(state)) {
        num_zero += !value;
    }

    state[index] = std::make_unique<ReduceNodeData<op>>(!num_zero, num_zero);
}

template <>
void ReduceNode<std::logical_or<double>>::initialize_state(State& state) const {
    int index = topological_index();
    assert(index >= 0 && "must be topologically sorted");
    assert(static_cast<int>(state.size()) > index && "unexpected state length");
    assert(state[index] == nullptr && "already initialized state");

    ssize_t num_nonzero = init.value_or(1) ? 1 : 0;
    for (const double& value : array_ptr_->view(state)) {
        num_nonzero += static_cast<bool>(value);
    }

    state[index] = std::make_unique<ReduceNodeData<op>>(num_nonzero > 0, num_nonzero);
}

template <>
void ReduceNode<std::multiplies<double>>::initialize_state(State& state) const {
    int index = topological_index();
    assert(index >= 0 && "must be topologically sorted");
    assert(static_cast<int>(state.size()) > index && "unexpected state length");
    assert(state[index] == nullptr && "already initialized state");

    // there is an edge case here for init being 0, in that case the `nonzero`
    // component will always be 0, which is dumb but everything still works and
    // it's enough of an edge case that I don't think it makes sense to do
    // anything-performance wise.

    double nonzero = init.value_or(1);
    ssize_t num_zero = 0;

    for (const double& value : array_ptr_->view(state)) {
        if (value == 0) {
            num_zero += 1;
        } else {
            nonzero *= value;
        }
    }

    // finally get the value
    double value = num_zero > 0 ? 0 : nonzero;

    // and then create the state
    state[index] = std::make_unique<ReduceNodeData<op>>(value, nonzero, num_zero);
}

template <class BinaryOp>
bool ReduceNode<BinaryOp>::integral() const {
    using result_type = typename std::invoke_result<BinaryOp, double&, double&>::type;

    if constexpr (std::is_integral<result_type>::value) {
        return true;
    }

    if constexpr (std::is_same<BinaryOp, functional::max<double>>::value ||
                  std::is_same<BinaryOp, functional::min<double>>::value ||
                  std::is_same<BinaryOp, std::multiplies<double>>::value ||
                  std::is_same<BinaryOp, std::plus<double>>::value) {
        // the actual value of init doesn't matter, all of the above have no default
        // or an integer default init
        return array_ptr_->integral() && is_integer(init.value_or(0));
    }

    assert(false && "not implemeted yet");
    unreachable();
}

template <class BinaryOp>
double ReduceNode<BinaryOp>::max() const {
    using result_type = typename std::invoke_result<BinaryOp, double&, double&>::type;

    if constexpr (std::is_same<result_type, bool>::value) {
        return true;
    }

    // These can results in inf. If we fix that in initialization/propagation we
    // should also fix it here.
    if constexpr (std::is_same<BinaryOp, functional::max<double>>::value) {
        if (init.has_value()) return std::max(init.value(), array_ptr_->max());
        return array_ptr_->max();
    }
    if constexpr (std::is_same<BinaryOp, functional::min<double>>::value) {
        if (init.has_value()) return std::min(init.value(), array_ptr_->max());
        return array_ptr_->max();
    }
    if constexpr (std::is_same<BinaryOp, std::multiplies<double>>::value) {
        // this is complicated...

        // the dynamic case. For now let's just fall back to Array's default
        // implementation because this gets even more complicated
        if (array_ptr_->dynamic()) return Array::max();

        const double init = this->init.value_or(1);

        // the empty case, so we're always just init
        if (array_ptr_->size() == 0) return init;

        const double low = array_ptr_->min();
        const double high = array_ptr_->max();

        // exactly one element, handle the sign of init
        if (array_ptr_->size() == 1) return std::max(init * low, init * high);

        const ssize_t size = array_ptr_->size();

        // more than one element
        return std::max({
            init * std::pow(low, size),
            init * std::pow(high, size),
            init * low * std::pow(high, size - 1),
            init * high * std::pow(low, size - 1),
        });
    }
    if constexpr (std::is_same<BinaryOp, std::plus<double>>::value) {
        const double high = array_ptr_->max();
        const double init = this->init.value_or(0);

        // if high is 0, then it doesn't matter how many times we add it
        if (high == 0) return init;

        // if the array has a fixed size, then just multiply the largest value
        // by that size
        if (const ssize_t size = array_ptr_->size(); size >= 0) {
            return init + size * high;
        }

        // our predecessor array is dynamic. So there are a few more cases
        // we need to check

        // 100 is a magic number. It's how far back in the predecessor
        // chain to check to get good bounds on the size for the given array.
        // This will exit early if it converges.
        const SizeInfo sizeinfo = array_ptr_->sizeinfo().substitute(100);

        if (high > 0) {
            // if high is positive, then we're interested in the maxmum size
            // of the array

            // if the array is arbitrarily large, then just fall back to the
            // default max.
            if (!sizeinfo.max.has_value()) return Array::max();

            return init + sizeinfo.max.value() * high;
        } else {
            // if high is negative, then we're interested in the minimum size
            // of the array
            return init + sizeinfo.min.value_or(0) * high;
        }
    }

    assert(false && "not implemeted yet");
    unreachable();
}

template <class BinaryOp>
double ReduceNode<BinaryOp>::min() const {
    using result_type = typename std::invoke_result<BinaryOp, double&, double&>::type;

    if constexpr (std::is_same<result_type, bool>::value) {
        return false;
    }

    // These can results in inf. If we fix that in initialization/propagation we
    // should also fix it here.
    if constexpr (std::is_same<BinaryOp, functional::max<double>>::value) {
        if (init.has_value()) return std::max(init.value(), array_ptr_->min());
        return array_ptr_->min();
    }
    if constexpr (std::is_same<BinaryOp, functional::min<double>>::value) {
        if (init.has_value()) return std::min(init.value(), array_ptr_->min());
        return array_ptr_->min();
    }
    if constexpr (std::is_same<BinaryOp, std::multiplies<double>>::value) {
        // this is complicated...

        // the dynamic case. For now let's just fall back to Array's default
        // implementation because this gets even more complicated
        if (array_ptr_->dynamic()) return Array::min();

        const double init = this->init.value_or(1);

        // the empty case, so we're always just init
        if (array_ptr_->size() == 0) return init;

        const double low = array_ptr_->min();
        const double high = array_ptr_->max();

        // exactly one element, handle the sign of init
        if (array_ptr_->size() == 1) return std::min(init * low, init * high);

        const ssize_t size = array_ptr_->size();

        // more than one element
        return std::min({
            init * std::pow(low, size),
            init * std::pow(high, size),
            init * low * std::pow(high, size - 1),
            init * high * std::pow(low, size - 1),
        });
    }
    if constexpr (std::is_same<BinaryOp, std::plus<double>>::value) {
        const double low = array_ptr_->min();
        const double init = this->init.value_or(0);

        // if low is 0, then it doesn't matter how many times we add it
        if (low == 0) return init;

        // if the array has a fixed size, then just multiply the smallest value
        // by that size
        if (const ssize_t size = array_ptr_->size(); size >= 0) {
            return init + size * low;
        }

        // our predecessor array is dynamic. So there are a few more cases
        // we need to check

        // 100 is a magic number. It's how far back in the predecessor
        // chain to check to get good bounds on the size for the given array.
        // This will exit early if it converges.
        const SizeInfo sizeinfo = array_ptr_->sizeinfo().substitute(100);

        if (low < 0) {
            // if low is negative, then we're interested in the maximum size
            // of the array

            // if the array is arbitrarily large, then just fall back to the
            // default min.
            if (!sizeinfo.max.has_value()) return Array::min();

            return init + sizeinfo.max.value() * low;
        } else {
            // if low is positive, then we're interested in the minimum size
            // of the array
            return init + sizeinfo.min.value_or(0) * low;
        }
    }

    assert(false && "not implemeted yet");
    unreachable();
}

template <>
void ReduceNode<std::logical_and<double>>::propagate(State& state) const {
    auto ptr = data_ptr<ReduceNodeData<op>>(state);

    ssize_t& num_zero = ptr->extra.num_zero;

    // count the change in the num_zero
    for (const Update& update : array_ptr_->diff(state)) {
        const auto& [_, old, value] = update;

        if (update.placed()) {
            // added a value to the array
            num_zero += !value;
        } else if (update.removed()) {
            // removed a value from the array
            num_zero -= !old;
        } else if (!old && value) {
            // changed a 0 to a truthy value
            num_zero -= 1;
        } else if (old && !value) {
            // changed a truthy value to a 0
            num_zero += 1;
        } else {
            // otherwise we don't care because the net number of 0s hasn't changed
            assert(static_cast<bool>(old) == static_cast<bool>(value));
        }
    }

    assert(num_zero >= 0);  // should never go negative

    ptr->values.value = num_zero < 1;
    if (ptr->values.value != ptr->values.old) Node::propagate(state);
}

template <>
void ReduceNode<std::logical_or<double>>::propagate(State& state) const {
    auto ptr = data_ptr<ReduceNodeData<op>>(state);

    ssize_t& num_nonzero = ptr->extra.num_nonzero;

    // count the change in the num_nonzero
    for (const Update& update : array_ptr_->diff(state)) {
        const auto& [_, old, value] = update;

        if (update.placed()) {
            // added a value to the array
            num_nonzero += static_cast<bool>(value);
        } else if (update.removed()) {
            // removed a value from the array
            num_nonzero -= static_cast<bool>(old);
        } else if (!old && value) {
            // changed a 0 to a truthy value
            num_nonzero += 1;
        } else if (old && !value) {
            // changed a truthy value to a 0
            num_nonzero -= 1;
        } else {
            // otherwise we don't care because the net number of 0s hasn't changed
            assert(static_cast<bool>(old) == static_cast<bool>(value));
        }
    }

    assert(num_nonzero >= 0);  // should never go negative

    ptr->values.value = num_nonzero > 0;
    if (ptr->values.value != ptr->values.old) Node::propagate(state);
}

template <>
void ReduceNode<functional::max<double>>::propagate(State& state) const {
    auto ptr = data_ptr<ReduceNodeData<op>>(state);

    auto& value = ptr->values.value;

    bool reinitialize = false;
    for (const Update& update : array_ptr_->diff(state)) {
        if (update.removed()) {
            if (update.old >= value) {
                // uh oh, we may have just removed the known max. So we need to
                // recalculate everything from scratch
                reinitialize = true;
                break;
            }
        } else if (update.placed() || update.value > update.old) {
            // we added a new value or increased an existing once
            value = std::max(update.value, value);
        } else if (update.old == value && update.old != update.value) {
            // we potentially made the current max smaller
            assert(update.value < update.old);
            reinitialize = true;
            break;
        }
    }

    if (reinitialize) {
        value = reduce(state);
    }

    Node::propagate(state);
}

template <>
void ReduceNode<functional::min<double>>::propagate(State& state) const {
    auto ptr = data_ptr<ReduceNodeData<op>>(state);

    auto& value = ptr->values.value;

    bool reinitialize = false;
    for (const Update& update : array_ptr_->diff(state)) {
        if (update.removed()) {
            if (update.old <= value) {
                // uh oh, we may have just removed the known min. So we need to
                // recalculate everything from scratch
                reinitialize = true;
                break;
            }
        } else if (update.placed() || update.value < update.old) {
            // we added a new value, or decreased an existing one
            value = std::min(update.value, value);
        } else if (update.old == value && update.old != update.value) {
            // we potentially made the current min larger
            assert(update.value > update.old);
            reinitialize = true;
            break;
        }
    }

    if (reinitialize) {
        value = reduce(state);
    }

    Node::propagate(state);
}

template <>
void ReduceNode<std::multiplies<double>>::propagate(State& state) const {
    auto ptr = data_ptr<ReduceNodeData<op>>(state);

    auto& nonzero = ptr->extra.nonzero;
    auto& num_zero = ptr->extra.num_zero;

    assert(num_zero == 0 || ptr->values.value == 0);

    // count the change in the num_zero
    for (const Update& update : array_ptr_->diff(state)) {
        if (update.placed() && update.value == 0) {
            // added a zero
            num_zero += 1;
        } else if (update.placed()) {
            assert(update.value != 0);
            nonzero *= update.value;
        } else if (update.removed() && update.old == 0) {
            // removed a 0
            num_zero -= 1;
        } else if (update.removed()) {
            assert(update.old != 0);
            nonzero /= update.old;
        } else if (update.old == 0 && update.value == 0) {
            // changed a 0 to a 0, nothing to do
        } else if (update.old == 0) {
            // changed a 0 to something else
            nonzero *= update.value;
            num_zero -= 1;
        } else if (update.value == 0) {
            // changed something else to a 0
            nonzero /= update.old;
            num_zero += 1;
        } else {
            // something else to something else
            nonzero *= (update.value / update.old);
        }
    }

    ptr->values.value = num_zero > 0 ? 0 : nonzero;
    if (ptr->values.value != ptr->values.old) Node::propagate(state);
}

template <>
void ReduceNode<std::plus<double>>::propagate(State& state) const {
    auto ptr = data_ptr<ReduceNodeData<op>>(state);

    // todo: consider kahan summation

    auto& value = ptr->values.value;

    for (const Update& update : array_ptr_->diff(state)) {
        if (update.placed()) {
            value += update.value;
        } else if (update.removed()) {
            value -= update.old;
        } else {
            value += update.value - update.old;
        }
    }

    Node::propagate(state);
}

template <class BinaryOp>
double ReduceNode<BinaryOp>::reduce(const State& state) const {
    auto start = array_ptr_->begin(state);
    const auto end = array_ptr_->end(state);

    double init;

    if (this->init.has_value()) {
        init = this->init.value();
    } else {
        // if there is no init, we use the first value in the array as the init
        // we should only be here if the array is not dynamic and if it has at
        // least one entry
        assert(!array_ptr_->dynamic());
        assert(array_ptr_->size(state) >= 1);
        assert(start != end);

        init = *start;
        ++start;
    }

    return std::reduce(start, end, init, BinaryOp());
}

template <class BinaryOp>
void ReduceNode<BinaryOp>::revert(State& state) const {
    data_ptr<ReduceNodeData<op>>(state)->revert();
}

// Uncommented are the tested specializations
template class ReduceNode<functional::max<double>>;
template class ReduceNode<functional::min<double>>;
template class ReduceNode<std::logical_and<double>>;
template class ReduceNode<std::logical_or<double>>;
template class ReduceNode<std::multiplies<double>>;
template class ReduceNode<std::plus<double>>;

// UnaryOpNode *****************************************************************

template <class UnaryOp>
UnaryOpNode<UnaryOp>::UnaryOpNode(ArrayNode* node_ptr)
        : ArrayOutputMixin(node_ptr->shape()), array_ptr_(node_ptr) {
    add_predecessor(node_ptr);
}

template <class UnaryOp>
void UnaryOpNode<UnaryOp>::commit(State& state) const {
    data_ptr<ArrayNodeStateData>(state)->commit();
}

template <class UnaryOp>
double const* UnaryOpNode<UnaryOp>::buff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->buff();
}

template <class UnaryOp>
std::span<const Update> UnaryOpNode<UnaryOp>::diff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->diff();
}

template <class UnaryOp>
void UnaryOpNode<UnaryOp>::initialize_state(State& state) const {
    int index = topological_index();
    assert(index >= 0 && "must be topologically sorted");
    assert(static_cast<int>(state.size()) > index && "unexpected state length");
    assert(state[index] == nullptr && "already initialized state");

    auto func = op();
    std::vector<double> values;
    values.reserve(array_ptr_->size(state));
    for (const double& val : array_ptr_->view(state)) {
        values.emplace_back(func(val));
    }

    state[index] = std::make_unique<ArrayNodeStateData>(std::move(values));
}

template <>
bool UnaryOpNode<functional::abs<double>>::integral() const {
    return array_ptr_->integral();
}
template <>
bool UnaryOpNode<std::negate<double>>::integral() const {
    return array_ptr_->integral();
}
template <>
bool UnaryOpNode<functional::square<double>>::integral() const {
    return array_ptr_->integral();
}
template <class UnaryOp>
bool UnaryOpNode<UnaryOp>::integral() const {
    using result_type = typename std::invoke_result<UnaryOp, double&>::type;

    if constexpr (std::is_integral<result_type>::value) {
        return true;
    }

    return Array::integral();
}

template <>
double UnaryOpNode<functional::abs<double>>::max() const {
    const double max = array_ptr_->max();
    const double min = array_ptr_->min();
    assert(min <= max && "min > max");

    if (min >= 0 && max >= 0) {
        return max;
    } else if (min >= 0) {
        assert(false && "min > max");
        unreachable();
    } else if (max >= 0) {
        return std::max<double>(max, -min);
    } else {
        return -min;
    }
}
template <>
double UnaryOpNode<std::negate<double>>::max() const {
    return -(array_ptr_->min());
}
template <>
double UnaryOpNode<functional::square<double>>::max() const {
    const double max = array_ptr_->max();
    if (std::abs(max) >= std::sqrt(Array::max())) {
        // We could consider raising an error here but for now let's be
        // permissive.
        return Array::max();
    }
    return max * max;
}
template <class UnaryOp>
double UnaryOpNode<UnaryOp>::max() const {
    using result_type = typename std::invoke_result<UnaryOp, double&>::type;

    if constexpr (std::is_same<result_type, bool>::value) {
        return true;
    }

    return Array::max();
}

template <>
double UnaryOpNode<functional::abs<double>>::min() const {
    const double max = array_ptr_->max();
    const double min = array_ptr_->min();
    assert(min <= max && "min > max");

    if (min >= 0 && max >= 0) {
        return min;
    } else if (min >= 0) {
        assert(false && "min > max");
        unreachable();
    } else if (max >= 0) {
        return 0;
    } else {
        return -max;
    }
}
template <>
double UnaryOpNode<std::negate<double>>::min() const {
    return -(array_ptr_->max());
}
template <>
double UnaryOpNode<functional::square<double>>::min() const {
    const double min = array_ptr_->min();
    if (std::abs(min) >= std::sqrt(Array::max())) {
        // We could consider raising an error here but for now let's be
        // permissive.
        return Array::max();
    }
    return min * min;
}
template <class UnaryOp>
double UnaryOpNode<UnaryOp>::min() const {
    using result_type = typename std::invoke_result<UnaryOp, double&>::type;

    if constexpr (std::is_same<result_type, bool>::value) {
        return false;
    }

    return Array::min();
}

template <class UnaryOp>
void UnaryOpNode<UnaryOp>::propagate(State& state) const {
    auto func = op();
    auto node_data = data_ptr<ArrayNodeStateData>(state);

    for (const auto& update : array_ptr_->diff(state)) {
        const auto& [idx, _, value] = update;

        if (update.placed()) {
            assert(idx == static_cast<ssize_t>(node_data->buffer.size()));
            node_data->emplace_back(func(value));
        } else if (update.removed()) {
            assert(idx == static_cast<ssize_t>(node_data->buffer.size()) - 1);
            node_data->pop_back();
        } else {
            node_data->set(idx, func(value));
        }
    }

    if (node_data->updates.size()) Node::propagate(state);
}

template <class UnaryOp>
void UnaryOpNode<UnaryOp>::revert(State& state) const {
    data_ptr<ArrayNodeStateData>(state)->revert();
}

template <class UnaryOp>
std::span<const ssize_t> UnaryOpNode<UnaryOp>::shape(const State& state) const {
    return array_ptr_->shape(state);
}

template <class UnaryOp>
ssize_t UnaryOpNode<UnaryOp>::size(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->buffer.size();
}

template <class UnaryOp>
ssize_t UnaryOpNode<UnaryOp>::size_diff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->size_diff();
}

template class UnaryOpNode<functional::abs<double>>;
template class UnaryOpNode<functional::logical<double>>;
template class UnaryOpNode<functional::square<double>>;
template class UnaryOpNode<std::negate<double>>;
template class UnaryOpNode<std::logical_not<double>>;

}  // namespace dwave::optimization

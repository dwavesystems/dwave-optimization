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

#include "dwave-optimization/nodes/naryop.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <type_traits>

#include "_state.hpp"
#include "dwave-optimization/utils.hpp"

namespace dwave::optimization {

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
struct InverseOp<std::divides<double>> {
    static bool constexpr exists() { return true; }

    double op(const double& x, const double& y) { return x * y; }
};

template <>
struct InverseOp<std::multiplies<double>> {
    static bool constexpr exists() { return true; }

    double op(const double& x, const double& y) { return x / y; }
};

struct NaryOpNodeData : public ArrayNodeStateData {
    explicit NaryOpNodeData(std::vector<double> values,
                            std::vector<Array::const_iterator> iterators)
            : ArrayNodeStateData(std::move(values)), iterators(std::move(iterators)) {}

    // used to avoid reallocating memory for predecessor iterators every propagation
    std::vector<Array::const_iterator> iterators;
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
std::pair<double, double> NaryOpNode<BinaryOp>::minmax(
        optional_cache_type<std::pair<double, double>> cache) const {
    // If the output of the operation is boolean, then don't bother caching the result.
    using result_type = typename std::invoke_result<BinaryOp, double&, double&>::type;
    if constexpr (std::same_as<result_type, bool>) {
        return {false, true};
    }

    // Otherwise the min and max depend on the predecessors, so we want to cache

    // First check if we've already calculated it.
    if (cache.has_value()) {
        if (auto it = cache->get().find(this); it != cache->get().end()) {
            return it->second;
        }
    }

    auto op = BinaryOp();

    // these can result in inf. If we update propagation/initialization to handle
    // that case we should update these as well.
    if constexpr (std::same_as<BinaryOp, functional::max<double>> ||
                  std::same_as<BinaryOp, functional::min<double>> ||
                  std::same_as<BinaryOp, std::plus<double>>) {
        assert(operands_.size() >= 1);  // checked by constructor

        auto [low, high] = operands_[0]->minmax(cache);
        for (const Array* rhs_ptr : operands_ | std::views::drop(1)) {
            const auto [rhs_low, rhs_high] = rhs_ptr->minmax(cache);
            low = op(low, rhs_low);
            high = op(high, rhs_high);
        }

        assert(low <= high);
        return memoize(cache, std::make_pair(low, high));
    }
    if constexpr (std::same_as<BinaryOp, std::multiplies<double>>) {
        assert(operands_.size() >= 1);  // checked by constructor

        auto [low, high] = operands_[0]->minmax(cache);
        for (const Array* rhs_ptr : operands_ | std::views::drop(1)) {
            const auto [rhs_low, rhs_high] = rhs_ptr->minmax(cache);

            std::array<double, 4> combos{op(low, rhs_low), op(low, rhs_high), op(high, rhs_low),
                                         op(high, rhs_high)};

            low = std::ranges::min(combos);
            high = std::ranges::max(combos);
        }

        assert(low <= high);
        return memoize(cache, std::make_pair(low, high));
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
    std::vector<double> values;

    values.reserve(size());

    std::vector<Array::const_iterator> iterators;
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
            val = op(*it, val);
            ++it;
        }
        values.emplace_back(val);
    }

    emplace_data_ptr<NaryOpNodeData>(state, std::move(values), std::move(iterators));
}

template <class BinaryOp>
void NaryOpNode<BinaryOp>::propagate(State& state) const {
    auto node_data = data_ptr<NaryOpNodeData>(state);

    auto& iterators = node_data->iterators;

    std::vector<ssize_t> recompute_indices;

    if constexpr (!InverseOp<BinaryOp>().exists()) {
        // have to recompute from all predecessors on any changed index
        for (const Array* input : operands_) {
            if (input->diff(state).size()) {
                for (const auto& [index, _, __] : input->diff(state)) {
                    recompute_indices.push_back(index);
                }
            }
        }

    } else {
        auto inv_func = InverseOp<BinaryOp>();
        for (const Array* input : operands_) {
            if (input->diff(state).size()) {
                for (const auto& [index, old_val, new_val] : input->diff(state)) {
                    double new_reduced_val = node_data->get(index);
                    new_reduced_val = inv_func.op(new_reduced_val, old_val);
                    if (std::isnan(new_reduced_val) || std::isinf(new_reduced_val)) {
                        // calling the inverse has failed (such as divide by zero),
                        // need to fully recompute this index
                        recompute_indices.push_back(index);
                        continue;
                    }

                    node_data->set(index, op(new_reduced_val, new_val));
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
            double val = node_data->get(index);
            val = *(iterators[0] + index);
            for (auto it : iterators | std::views::drop(1)) {
                val = op(*(it + index), val);
            }

            node_data->set(index, val);
        }
    }
}

template class NaryOpNode<functional::max<double>>;
template class NaryOpNode<functional::min<double>>;
template class NaryOpNode<std::multiplies<double>>;
template class NaryOpNode<std::plus<double>>;

}  // namespace dwave::optimization

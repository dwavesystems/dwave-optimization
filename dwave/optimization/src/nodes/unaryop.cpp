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

#include "dwave-optimization/nodes/unaryop.hpp"

#include "_state.hpp"

namespace dwave::optimization {

template <class UnaryOp>
std::pair<double, double> calculate_values_minmax_(Array* const array_ptr) {
    // Do some checks to make sure the resulting domain/range will be valid
    if constexpr (std::is_same<UnaryOp, functional::square_root<double>>::value) {
        if (array_ptr->min() < 0) {
            throw std::invalid_argument("SquareRoot's predecessors cannot take a negative value");
        }
    } else if constexpr (std::is_same<UnaryOp, functional::log<double>>::value) {
        if (array_ptr->min() <= 0) {
            throw std::invalid_argument("Log's predecessors cannot take a negative or zero value");
        }
    }

    // If the output of the operation is boolean, then don't bother caching the result.
    using result_type = typename std::invoke_result<UnaryOp, double&>::type;
    if constexpr (std::same_as<result_type, bool>) {
        return {false, true};
    }

    // Likewise for sin/cos the minmax is -1/+1. We could tighten it if the domain
    // of our predecessor is smaller than 2pi, but let's keep it simple for now
    if constexpr (std::same_as<UnaryOp, functional::cos<double>> ||
                  std::same_as<UnaryOp, functional::sin<double>>) {
        return {-1, +1};
    }

    // Otherwise the min and max depend on the predecessor

    auto low = array_ptr->min();
    auto high = array_ptr->max();
    assert(low <= high);

    if constexpr (std::same_as<UnaryOp, functional::abs<double>>) {
        if (low >= 0 && high >= 0) {
            return std::make_pair(low, high);
        } else if (low >= 0) {
            assert(false && "min > max");
            unreachable();
        } else if (high >= 0) {
            return std::pair<double, double>(0.0, std::max<double>(-low, high));
        } else {
            return std::make_pair(-high, -low);
        }
    }
    if constexpr (std::same_as<UnaryOp, functional::exp<double>>) {
        return std::make_pair(std::exp(low), std::exp(high));
    }
    if constexpr (std::same_as<UnaryOp, functional::expit<double>>) {
        double expit_low = 1.0 / (1.0 + std::exp(-low));
        double expit_high = 1.0 / (1.0 + std::exp(-high));
        return std::make_pair(expit_low, expit_high);
    }
    if constexpr (std::same_as<UnaryOp, functional::log<double>>) {
        assert(low > 0);  // checked by constructor
        return std::make_pair(std::log(low), std::log(high));
    }
    if constexpr (std::same_as<UnaryOp, functional::rint<double>>) {
        return std::make_pair(std::rint(low), std::rint(high));
    }
    if constexpr (std::same_as<UnaryOp, functional::square<double>>) {
        const auto highest = std::numeric_limits<double>::max();
        return std::make_pair(std::min({low * low, high * high, highest}),
                              std::min(std::max({low * low, high * high}),
                                       highest));  // prevent inf
    }
    if constexpr (std::same_as<UnaryOp, functional::square_root<double>>) {
        assert(low >= 0);  // checked by constructor
        return std::make_pair(std::sqrt(low), std::sqrt(high));
    }
    if constexpr (std::same_as<UnaryOp, std::negate<double>>) {
        return std::make_pair(-high, -low);
    }

    assert(false && "not implemeted yet");
    unreachable();
}

template <class UnaryOp>
UnaryOpNode<UnaryOp>::UnaryOpNode(ArrayNode* node_ptr)
        : ArrayOutputMixin(node_ptr->shape()),
          array_ptr_(node_ptr),
          minmax_(calculate_values_minmax_<UnaryOp>(array_ptr_)) {
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
    std::vector<double> values;
    values.reserve(array_ptr_->size(state));
    for (const double val : array_ptr_->view(state)) {
        values.emplace_back(op(val));
    }

    emplace_data_ptr<ArrayNodeStateData>(state, std::move(values));
}

template <>
bool UnaryOpNode<functional::abs<double>>::integral() const {
    return array_ptr_->integral();
}
template <>
bool UnaryOpNode<functional::cos<double>>::integral() const {
    return false;
}
template <>
bool UnaryOpNode<functional::exp<double>>::integral() const {
    return false;
}
template <>
bool UnaryOpNode<functional::expit<double>>::integral() const {
    return false;
}
template <>
bool UnaryOpNode<functional::log<double>>::integral() const {
    return false;
}
template <>
bool UnaryOpNode<std::negate<double>>::integral() const {
    return array_ptr_->integral();
}
template <>
bool UnaryOpNode<functional::rint<double>>::integral() const {
    return true;
}
template <>
bool UnaryOpNode<functional::sin<double>>::integral() const {
    return false;
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

template <class UnaryOp>
double UnaryOpNode<UnaryOp>::min() const {
    return this->minmax_.first;
}

template <class UnaryOp>
double UnaryOpNode<UnaryOp>::max() const {
    return this->minmax_.second;
}

template <class UnaryOp>
void UnaryOpNode<UnaryOp>::propagate(State& state) const {
    auto node_data = data_ptr<ArrayNodeStateData>(state);

    for (const auto& update : array_ptr_->diff(state)) {
        const auto& [idx, _, value] = update;

        if (update.placed()) {
            assert(idx == static_cast<ssize_t>(node_data->size()));
            node_data->emplace_back(op(value));
        } else if (update.removed()) {
            assert(idx == static_cast<ssize_t>(node_data->size()) - 1);
            node_data->pop_back();
        } else {
            node_data->set(idx, op(value));
        }
    }

    if (node_data->diff().size()) Node::propagate(state);
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
    return data_ptr<ArrayNodeStateData>(state)->size();
}

template <class UnaryOp>
ssize_t UnaryOpNode<UnaryOp>::size_diff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->size_diff();
}

template <class UnaryOp>
SizeInfo UnaryOpNode<UnaryOp>::sizeinfo() const {
    if (dynamic()) return SizeInfo(array_ptr_);  // exactly the same as predecessor
    return SizeInfo(size());
}

template class UnaryOpNode<functional::abs<double>>;
template class UnaryOpNode<functional::cos<double>>;
template class UnaryOpNode<functional::exp<double>>;
template class UnaryOpNode<functional::expit<double>>;
template class UnaryOpNode<functional::log<double>>;
template class UnaryOpNode<functional::logical<double>>;
template class UnaryOpNode<functional::rint<double>>;
template class UnaryOpNode<functional::sin<double>>;
template class UnaryOpNode<functional::square<double>>;
template class UnaryOpNode<functional::square_root<double>>;
template class UnaryOpNode<std::negate<double>>;
template class UnaryOpNode<std::logical_not<double>>;

}  // namespace dwave::optimization

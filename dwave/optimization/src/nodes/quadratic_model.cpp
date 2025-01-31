// Copyright 2024 D-Wave Systems Inc.
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

#include "dwave-optimization/nodes/quadratic_model.hpp"

#include "dwave-optimization/utils.hpp"

namespace dwave::optimization {
using index_type = int;
using bias_type = double;
using size_type = ssize_t;

// On some architectures it will allow for 80 bits of precision.
using hi_res_t = long double;

struct QuadraticModelNodeData : public NodeStateData {
    explicit QuadraticModelNodeData(double value, std::vector<double>&& state,
                                    index_type num_variables)
            : value_old(value), value(value_old), update(0, value, value), previous_state_(state) {
        effective_changes_.reserve(num_variables);
    }

    double const* buff() const { return &value.value(); }

    std::span<const Update> diff() const {
        if (value_old != value) {
            // Calculating this in lazy fashion.
            update.index = 0;
            update.old = value_old.value();
            update.value = value.value();
            return std::span<const Update, 1>(&update, 1);
        } else {
            return {};
        }
    }

    void commit() {
        value_old = value;
        effective_changes_.clear();
    }
    void revert() {
        value = value_old;
        for (auto& change : effective_changes_) {
            previous_state_[change.first] = change.second;
        }
        effective_changes_.clear();
    }

    double_kahan value_old;
    double_kahan value;
    mutable Update update;

    // This will evolve to current state during propagation.
    std::vector<double> previous_state_;

    // Save the index and previous values to be able to revert.
    std::vector<std::pair<int, double>> effective_changes_;
};

QuadraticModel::QuadraticModel(index_type num_variables) : num_variables_(num_variables) {
    if (num_variables_ < 0) {
        throw std::domain_error("Number of variables cannot be negative");
    }

    linear_biases_.resize(num_variables_, 0);
    square_biases_.resize(num_variables_, 0);
    adj_.resize(num_variables_);
}

ssize_t QuadraticModel::num_variables() const noexcept { return num_variables_; }

ssize_t QuadraticModel::num_interactions() const noexcept {
    ssize_t num_interactions = 0;
    for (auto n : adj_) {
        num_interactions += n.num_smaller_neighbors;
    }
    return num_interactions;
}

void QuadraticModel::set_linear(index_type v, bias_type bias) {
    assert(v >= 0 && static_cast<size_type>(v) < num_variables());
    linear_biases_[v] = bias;
}

void QuadraticModel::add_linear(index_type v, bias_type bias) {
    assert(v >= 0 && static_cast<size_type>(v) < num_variables());
    linear_biases_[v] += bias;
}

bias_type QuadraticModel::get_linear(index_type v) const {
    assert(v >= 0 && static_cast<size_type>(v) < num_variables());
    return linear_biases_[v];
}

// Creates the bias if it doesn't already exist
bias_type& QuadraticModel::asymmetric_quadratic_ref(index_type u, index_type v) {
    assert(0 <= u && static_cast<size_type>(u) < num_variables());
    assert(0 <= v && static_cast<size_type>(v) < num_variables());

    auto& neighbors_u = adj_[u].neighbors;
    auto& qbiases_u = adj_[u].biases;
    auto it = std::lower_bound(neighbors_u.begin(), neighbors_u.end(), v);
    auto offset = std::distance(neighbors_u.begin(), it);
    if (it == neighbors_u.end() || *it != v) {
        if (v < u) adj_[u].num_smaller_neighbors++;
        neighbors_u.emplace(it, v);
        return *(qbiases_u.emplace(qbiases_u.begin() + offset, 0));
    } else {
        return *(qbiases_u.begin() + offset);
    }
}

void QuadraticModel::set_quadratic(index_type u, index_type v, bias_type bias) {
    assert(0 <= u && static_cast<size_type>(u) < num_variables());
    assert(0 <= v && static_cast<size_type>(v) < num_variables());

    if (u == v) {
        square_biases_[u] = bias;
    } else {
        asymmetric_quadratic_ref(u, v) = bias;
        asymmetric_quadratic_ref(v, u) = bias;
    }
}

void QuadraticModel::add_quadratic(index_type u, index_type v, bias_type bias) {
    assert(0 <= u && static_cast<size_type>(u) < num_variables());
    assert(0 <= v && static_cast<size_type>(v) < num_variables());

    if (u == v) {
        square_biases_[u] += bias;
    } else {
        asymmetric_quadratic_ref(u, v) += bias;
        asymmetric_quadratic_ref(v, u) += bias;
    }
}

bias_type QuadraticModel::get_quadratic(index_type v) const {
    assert(0 <= v && static_cast<size_type>(v) < num_variables());
    return square_biases_[v];
}

bias_type QuadraticModel::get_quadratic(index_type u, index_type v) const {
    assert(0 <= u && static_cast<size_type>(u) < num_variables());
    assert(0 <= v && static_cast<size_type>(v) < num_variables());

    if (u == v) {
        return get_quadratic(v);
    } else {
        auto& neighbors_u = adj_[u].neighbors;
        auto& qbiases_u = adj_[u].biases;
        auto it = std::lower_bound(neighbors_u.begin(), neighbors_u.end(), v);
        if (it == neighbors_u.end() || *it != v) {
            return 0;
        } else {
            auto offset = std::distance(neighbors_u.begin(), it);
            return *(qbiases_u.begin() + offset);
        }
    }
}

void QuadraticModel::add_linear(const bias_type* linear_ptr) {
    for (index_type i = 0; i < num_variables_; ++i) {
        add_linear(i, linear_ptr[i]);
    }
}

void QuadraticModel::get_linear(bias_type* linear) const {
    for (ssize_t i = 0; i < num_variables_; ++i) {
        linear[i] = linear_biases_[i];
    }
}

void QuadraticModel::add_squares(const bias_type* squares_ptr) {
    for (index_type i = 0; i < num_variables_; ++i) {
        square_biases_[i] += squares_ptr[i];
    }
}

void QuadraticModel::get_squares(bias_type* squares) const {
    for (ssize_t i = 0; i < num_variables_; ++i) {
        squares[i] = square_biases_[i];
    }
}

void QuadraticModel::add_quadratic(const size_type n_interactions, const index_type* row,
                                   const index_type* col, const bias_type* quad) {
    for (size_type i = 0; i < n_interactions; ++i) {
        auto irow = row[i];
        auto icol = col[i];
        auto iquad = quad[i];
        assert(irow < num_variables_);
        assert(icol < num_variables_);
        add_quadratic(irow, icol, iquad);
    }
}

void QuadraticModel::get_quadratic(index_type* row, index_type* col, bias_type* quad) const {
    size_type idx = 0;
    for (size_type i = 0; i < num_variables_; ++i) {
        for (size_type j = 0; j < static_cast<size_type>(adj_[i].neighbors.size()); ++j) {
            auto u = adj_[i].neighbors[j];
            if (i < u) {
                row[idx] = i;
                col[idx] = u;
                quad[idx] = adj_[i].biases[j];
                ++idx;
            }
        }
    }

    assert(idx == num_interactions());
}

bias_type QuadraticModel::get_effective_linear_bias(index_type u,
                                                    std::span<const double> state) const {
    assert(static_cast<size_type>(state.size()) == num_variables_);
    hi_res_t effective_linear_bias = linear_biases_[u];
    auto& biases = adj_[u].biases;
    auto& neighbors = adj_[u].neighbors;
    for (size_t j = 0; j < neighbors.size(); j++) {
        effective_linear_bias += state[neighbors[j]] * biases[j];
    }

    return effective_linear_bias;
}

bias_type QuadraticModel::compute_value(std::span<const double> state) const {
    assert(static_cast<size_type>(state.size()) == num_variables_);
    hi_res_t value = 0;
    for (auto i = 0; i < num_variables_; i++) {
        if (state[i]) {
            hi_res_t bias = linear_biases_[i] + (square_biases_[i] * state[i]);
            auto num_smaller_neighbors = adj_[i].num_smaller_neighbors;
            auto& biases = adj_[i].biases;
            auto& neighbors = adj_[i].neighbors;
            for (auto j = 0; j < num_smaller_neighbors; j++) {
                bias += state[neighbors[j]] * biases[j];
            }
            value += state[i] * bias;
        }
    }

    return value;
}

void QuadraticModel::shrink_to_fit() {
    for (auto& neighborhoods : adj_) {
        neighborhoods.neighbors.shrink_to_fit();
        neighborhoods.biases.shrink_to_fit();
    }
    linear_biases_.shrink_to_fit();
    square_biases_.shrink_to_fit();
}

QuadraticModelNode::QuadraticModelNode(ArrayNode* state_node_ptr, QuadraticModel&& quadratic_model)
        : quadratic_model_(quadratic_model) {
    if (!std::ranges::equal(state_node_ptr->shape(),
                            std::vector<ssize_t>{quadratic_model_.num_variables()})) {
        throw std::invalid_argument(
                "node array must be one dimensional of length same as QuadraticModelNode.shape[0]");
    }

    quadratic_model_.shrink_to_fit();
    Node::add_predecessor(state_node_ptr);
}

double const* QuadraticModelNode::buff(const State& state) const {
    return data_ptr<QuadraticModelNodeData>(state)->buff();
}

std::span<const Update> QuadraticModelNode::diff(const State& state) const {
    return data_ptr<QuadraticModelNodeData>(state)->diff();
}

void QuadraticModelNode::commit(State& state) const {
    data_ptr<QuadraticModelNodeData>(state)->commit();
}

void QuadraticModelNode::revert(State& state) const {
    data_ptr<QuadraticModelNodeData>(state)->revert();
}

void QuadraticModelNode::initialize_state(State& state) const {
    auto state_data = dynamic_cast<Array*>(predecessors()[0])->view(state);
    std::vector<double> state_copy(state_data.begin(), state_data.end());
    double value = quadratic_model_.compute_value(state_copy);
    emplace_data_ptr<QuadraticModelNodeData>(state, value, std::move(state_copy),
                                             quadratic_model_.num_variables());
}

void QuadraticModelNode::propagate(State& state) const {
    auto state_node_ptr = dynamic_cast<Array*>(predecessors()[0]);
    auto diff = state_node_ptr->diff(state);
    if (diff.size()) {
        auto node_data_ptr = data_ptr<QuadraticModelNodeData>(state);
        auto& previous_state = node_data_ptr->previous_state_;
        auto& effective_changes = node_data_ptr->effective_changes_;
        auto& value = node_data_ptr->value;

        if (state_node_ptr->contiguous()) {
            auto current_state =
                    std::span(state_node_ptr->buff(state), state_node_ptr->size(state));
            for (auto& update : diff) {
                auto index = update.index;
                auto neo = current_state[index];
                auto old = previous_state[index];

                // We do not need to process every update, only if there is a difference between the
                // evolved state and the evolving state.
                if (neo != old) {
                    effective_changes.emplace_back(std::make_pair(index, old));
                    value += (neo - old) *
                             (quadratic_model_.get_quadratic(index) * (neo + old) +
                              quadratic_model_.get_effective_linear_bias(index, previous_state));
                    previous_state[index] = neo;
                }
            }
        } else {
            auto current_state = state_node_ptr->view(state);
            for (auto& update : diff) {
                auto index = update.index;
                auto neo = current_state[index];
                auto old = previous_state[index];

                // We do not need to process every update, only if there is a difference between the
                // evolved state and the evolving state.
                if (neo != old) {
                    effective_changes.emplace_back(std::make_pair(index, old));
                    value += (neo - old) *
                             (quadratic_model_.get_quadratic(index) * (neo + old) +
                              quadratic_model_.get_effective_linear_bias(index, previous_state));
                    previous_state[index] = neo;
                }
            }
        }

        Node::propagate(state);
    }
}

QuadraticModel* QuadraticModelNode::get_quadratic_model() {
    QuadraticModel* qm_ptr = &quadratic_model_;
    return qm_ptr;
}

}  // namespace dwave::optimization

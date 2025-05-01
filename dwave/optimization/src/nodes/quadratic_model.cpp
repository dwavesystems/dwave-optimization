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

// Follows
// https://github.com/dwavesystems/dwave-networkx/blob/0.8.17/dwave_networkx/generators/zephyr.py
std::vector<std::tuple<int, int>> ZephyrLattice::edges() const {
    assert(m > 0 && "m must be positive");
    assert(t > 0 && "t must be positive");

    const int M = 2 * m + 1;  // confusing name, but following dnx

    std::vector<std::tuple<int, int>> edges;
    edges.reserve(num_edges());

    // Coordinates to linear index
    auto index = [&](int u, int w, int k, int j, int z) -> int {
        return (((u * M + w) * t + k) * 2 + j) * m + z;
    };

    // External edges
    for (int u : {0, 1}) {
        for (int w : std::views::iota(0, M)) {
            for (int k : std::views::iota(0, t)) {
                for (int j : {0, 1}) {
                    for (int z : std::views::iota(0, m - 1)) {
                        edges.emplace_back(index(u, w, k, j, z), index(u, w, k, j, z + 1));
                    }
                }
            }
        }
    }

    // Odd edges
    for (int u : {0, 1}) {
        for (int w : std::views::iota(0, M)) {
            for (int k : std::views::iota(0, t)) {
                for (int a : {0, 1}) {
                    for (int z : std::views::iota(a, m)) {
                        edges.emplace_back(index(u, w, k, 0, z), index(u, w, k, 1, z - a));
                    }
                }
            }
        }
    }

    // Internal edges
    for (int w : std::views::iota(0, m)) {
        for (int z : std::views::iota(0, m)) {
            for (int h : std::views::iota(0, t)) {
                for (int k : std::views::iota(0, t)) {
                    for (int i : {0, 1}) {
                        for (int j : {0, 1}) {
                            for (int a : {0, 1}) {
                                for (int b : {0, 1}) {
                                    edges.emplace_back(
                                            index(0, 2 * w + 1 + a * (2 * i - 1), k, j, z),
                                            index(1, 2 * z + 1 + b * (2 * j - 1), h, i, w));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Sort each edge lexicographically
    std::ranges::for_each(edges, [](auto& edge) {
        auto& u = std::get<0>(edge);
        auto& v = std::get<1>(edge);

        assert(u >= 0);  // no negative indices
        assert(v >= 0);  // no negative indices
        assert(u != v);  // no self-loops

        if (u > v) std::swap(u, v);
    });

    // Sort the whole thing lexicographically
    std::ranges::sort(edges);

    // Consider adding an assert for uniqueness
    assert(static_cast<int>(edges.size()) == num_edges());

    return edges;
}

template <Lattice T>
struct LatticeNode<T>::StateData : public ScalarOutputMixinStateData {
    StateData(double value, std::vector<double>&& x_state)
            : ScalarOutputMixinStateData(value), x_state(std::move(x_state)) {}

    void commit() {
        ScalarOutputMixinStateData::commit();
        x_diff.clear();  // clear any changes we tracked
    }

    void revert() {
        ScalarOutputMixinStateData::revert();

        // undo any changes to our x_state
        for (const auto& [v, old] : x_diff | std::views::reverse) {
            x_state[v] = old;
        }
        x_diff.clear();
    }

    // In order to calculate our energy diffs, we also need to save a copy
    // of x's state, which we then change during propagation to match x.
    std::vector<double> x_state;

    // x_state index, old value pairs
    std::vector<std::tuple<ssize_t, double>> x_diff;
};

template <Lattice T>
LatticeNode<T>::LatticeNode(ArrayNode* x_ptr, T lattice)
        : LatticeNode(
                  x_ptr, lattice, [](int u) { return 0; }, [](int u, int v) { return 0; }) {}

template <Lattice T>
LatticeNode<T>::LatticeNode(ArrayNode* x_ptr, T lattice, std::function<double(int)> linear,
                            std::function<double(int, int)> quadratic)
        : x_ptr_(x_ptr), lattice_(std::move(lattice)) {
    if (!std::ranges::equal(x_ptr_->shape(), std::vector<ssize_t>{lattice_.num_nodes()})) {
        throw std::invalid_argument(
                "x must be a 1D array with a length that matches the number of nodes in the "
                "lattice");
    }

    // Add the nodes (with their weights) to the adjacency
    adj_.resize(lattice_.num_nodes());
    for (ssize_t u = 0, N = adj_.size(); u < N; ++u) {
        adj_[u].bias = linear(u);
    }

    // Add the edges (with their weights) to the adjacency
    for (const auto& [u, v] : lattice_.edges()) {
        const double bias = quadratic(u, v);
        adj_[u].neighbors.emplace_back(v, bias);
        adj_[v].neighbors.emplace_back(u, bias);
    }
    // Make sure each neighborhood is sorted
    for (auto& Nu : adj_) {
        std::ranges::sort(Nu.neighbors);
    }

    add_predecessor(x_ptr_);
}

template <Lattice T>
double const* LatticeNode<T>::buff(const State& state) const {
    return this->template data_ptr<StateData>(state)->buff();
}

template <Lattice T>
void LatticeNode<T>::commit(State& state) const {
    this->template data_ptr<StateData>(state)->commit();
}

template <Lattice T>
std::span<const Update> LatticeNode<T>::diff(const State& state) const {
    return this->template data_ptr<StateData>(state)->diff();
}

template <Lattice T>
void LatticeNode<T>::initialize_state(State& state) const {
    // Get the state of x
    const auto view = x_ptr_->view(state);

    // Now just run through the graph doing the energy calculation
    double energy = 0;
    for (ssize_t u = 0, N = adj_.size(); u < N; ++u) {
        const auto& Nu = adj_[u];

        const auto u_val = view[u];

        energy += Nu.bias * u_val;

        for (const auto& [v, bias] : Nu.neighbors) {
            assert(u != v);     // self loops are not allowed
            if (v >= u) break;  // only traverse the lower triangle so we don't count each twice
            energy += bias * (u_val * view[v]);
        }
    }

    emplace_data_ptr<StateData>(state, energy, std::vector<double>(view.begin(), view.end()));
}

template <Lattice T>
double LatticeNode<T>::linear(int u) const noexcept {
    // if out of bounds, return 0
    if (u < 0 || static_cast<std::size_t>(u) >= adj_.size()) return 0;
    // else return the bias
    return adj_[u].bias;
}

template <Lattice T>
void LatticeNode<T>::propagate(State& state) const {
    auto ptr = data_ptr<StateData>(state);

    // get the current energy
    double& energy = ptr->update.value;
    std::vector<double>& x_state = ptr->x_state;
    std::vector<std::tuple<ssize_t, double>>& x_diff = ptr->x_diff;

    assert(x_diff.empty() && "x_diff not cleared between propagations");

    // Now go through the diff and calculate the changes
    for (const auto& [u, old_value, new_value] : x_ptr_->diff(state)) {
        const auto delta = new_value - old_value;

        // linear bias
        energy += adj_[u].bias * delta;

        // quadratic biases
        for (const auto& [v, bias] : adj_[u].neighbors) {
            assert(u != v);  // self loops are not allowed
            energy += bias * (delta * x_state[v]);
        }

        x_diff.emplace_back(u, x_state[u]);
        x_state[u] = new_value;
    }

    // we should have evolved such that our stored copy of x_state matches
    // our predecessors
    assert(std::ranges::equal(x_state, x_ptr_->view(state)));
}

template <Lattice T>
double LatticeNode<T>::quadratic(int u, int v) const noexcept {
    // If u or v is not in range, then the quadratic bias definitely doesn't
    // exist so we just return 0.
    if (u < 0 || static_cast<std::size_t>(u) >= adj_.size()) return 0;
    if (v < 0 || static_cast<std::size_t>(v) >= adj_.size()) return 0;

    // Otherwise we do a binary search through the neighborhood.
    auto it = std::ranges::lower_bound(adj_[u].neighbors, neighbor(v));

    // if v is not in u's neighborhood, then return 0
    if (it == adj_[u].neighbors.end()) return 0;
    if (it->v != v) return 0;

    // it is a neighbor, so we can return the bias
    return it->bias;
}

template <Lattice T>
void LatticeNode<T>::revert(State& state) const {
    this->template data_ptr<StateData>(state)->revert();
}

template class LatticeNode<ZephyrLattice>;

}  // namespace dwave::optimization

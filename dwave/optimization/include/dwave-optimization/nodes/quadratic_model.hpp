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

#pragma once

#include <algorithm>
#include <memory>
#include <ranges>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/graph.hpp"

namespace dwave::optimization {

class QuadraticModel {
    using index_type = int;
    using bias_type = double;
    using size_type = ssize_t;

    struct neighborhood {
        index_type num_smaller_neighbors;
        std::vector<index_type> neighbors;
        std::vector<bias_type> biases;
    };

 public:
    QuadraticModel(index_type num_variables);

    ssize_t num_variables() const noexcept;
    ssize_t num_interactions() const noexcept;
    void set_linear(index_type v, bias_type bias);
    void add_linear(index_type v, bias_type bias);
    bias_type get_linear(index_type v) const;

    void set_quadratic(index_type u, index_type v, bias_type bias);
    void add_quadratic(index_type u, index_type v, bias_type bias);
    bias_type get_quadratic(index_type u, index_type v) const;

    // Functions used to return
    void add_linear(const bias_type* linear_ptr);
    void get_linear(bias_type* linear_ptr) const;
    void add_squares(const bias_type* squares_ptr);
    void get_squares(bias_type* squares_ptr) const;
    void add_quadratic(const size_type n_interactions, const index_type* row, const index_type* col,
                       const bias_type* quad);
    void get_quadratic(index_type* row, index_type* col, bias_type* quad) const;

 private:
    void shrink_to_fit();
    bias_type get_quadratic(index_type v) const;
    bias_type& asymmetric_quadratic_ref(index_type u, index_type v);
    bias_type compute_value(std::span<const double> state) const;
    bias_type get_effective_linear_bias(index_type u, std::span<const double> state) const;

    const ssize_t num_variables_;
    std::vector<bias_type> linear_biases_;
    std::vector<bias_type> square_biases_;
    std::vector<neighborhood> adj_;

    friend class QuadraticModelNode;
};

class QuadraticModelNode : public ScalarOutputMixin<ArrayNode> {
 public:
    QuadraticModelNode(ArrayNode* state_node_ptr, QuadraticModel&& quadratic_model);

    double const* buff(const State& state) const override;
    std::span<const Update> diff(const State& state) const override;
    void commit(State& state) const override;
    void revert(State& state) const override;
    void initialize_state(State& state) const override;
    void propagate(State& state) const override;
    QuadraticModel* get_quadratic_model();

 private:
    QuadraticModel quadratic_model_;
};

/// Defines a Zephyr lattice.
/// See https://docs.dwavequantum.com/en/latest/concepts/index.html#term-Zephyr
struct ZephyrLattice {
    /// A single Zephyr cell.
    constexpr ZephyrLattice() noexcept : ZephyrLattice(1) {}

    /// A Zephyr lattice with grid parameter `m` and tile parameter `t`.
    explicit(false) constexpr ZephyrLattice(ssize_t m, ssize_t t = 4) : m(m), t(t) {
        if (m <= 0) throw std::invalid_argument("m must be positive");
        if (t <= 0) throw std::invalid_argument("t must be positive");
    }

    /// Two Zephyr graphs are equivalent if they have the same grid and tile
    /// parameters.
    constexpr friend bool operator==(const ZephyrLattice& lhs, const ZephyrLattice& rhs) noexcept {
        return lhs.m == rhs.m && lhs.t == rhs.t;
    }

    /// Return a vector of edges in the Zephyr lattice, sorted lexicographically.
    std::vector<std::tuple<int, int>> edges() const;

    /// The number of edges in a Zephyr lattice
    constexpr ssize_t num_edges() const {
        assert(m > 0 && "m must be positive");
        assert(t > 0 && "t must be positive");
        if (m == 1) return 2 * t * (8 * t + 3);
        return 2 * t * ((8 * t + 8) * m * m - 2 * m - 3);
    }

    /// The number of nodes in a Zephyr lattice
    constexpr ssize_t num_nodes() const {
        assert(m > 0 && "m must be positive");
        assert(t > 0 && "t must be positive");
        return 4 * t * m * (2 * m + 1);
    }

    /// The grid parameter.
    ssize_t m;

    /// The tile parameter.
    ssize_t t;
};

/// A node representing a quadratic model with linear and quadratic biases
/// structured according to the given Lattice.
template <class Lattice>  // todo: concept for Lattice?
class LatticeNode : public ScalarOutputMixin<ArrayNode> {
 public:
    LatticeNode() = delete;

    LatticeNode(ArrayNode* x_ptr, Lattice lattice)
            : LatticeNode(
                      x_ptr, lattice, [](int u) { return 0; }, [](int u, int v) { return 0; }) {}

    // `std::invocable` doesn't check the result type unfortunately, so the template is still
    // underspecified. But this is better than just `class`
    template <std::invocable<int> Linear, std::invocable<int, int> Quadratic>
    LatticeNode(ArrayNode* x_ptr, Lattice lattice, Linear&& linear, Quadratic&& quadratic)
            : x_ptr_(x_ptr), lattice_(std::move(lattice)) {
        if (!std::ranges::equal(x_ptr_->shape(), std::array{lattice_.num_nodes()})) {
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

    /// @copydoc Array::buff()
    double const* buff(const State& state) const final {
        return this->template data_ptr<StateData>(state)->buff();
    }

    /// @copydoc Node::commit()
    void commit(State& state) const final {
        this->template data_ptr<StateData>(state)->commit();
    }

    /// @copydoc Array::diff()
    std::span<const Update> diff(const State& state) const final {
        return this->template data_ptr<StateData>(state)->diff();
    }

    /// @copydoc Node::initialize_state()
    void initialize_state(State& state) const override {
        // Get the state of x
        const auto view = x_ptr_->view(state);

        // Now just run through the graph doing the energy calculation
        double energy = 0;
        for (ssize_t u = 0, N = adj_.size(); u < N; ++u) {
            const auto& Nu = adj_[u];

            const auto u_val = view[u];

            energy += Nu.bias * u_val;

            for (const auto& [v, bias] : Nu.neighbors) {
                assert(u != v);  // self loops are not allowed
                if (v >= u) break;  // only traverse the lower triangle so we don't count each twice
                energy += bias * (u_val * view[v]);
            }
        }

        emplace_data_ptr<StateData>(state, energy, std::vector<double>(view.begin(), view.end()));
    }

    /// Get the linear bias associated with `u`. Returns `0` if `u` is out-of-bounds.
    double linear(int u) const noexcept {
        // if out of bounds, return 0
        if (u < 0 || static_cast<std::size_t>(u) >= adj_.size()) return 0;
        // else return the bias
        return adj_[u].bias;
    }

    /// @copydoc Node::propagate()
    void propagate(State& state) const override {
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

    /// Get the linear bias associated with `u` and `v`.
    /// Returns `0` if `u` or `v` are out of bounds or if they have no interaction.
    double quadratic(int u, int v) const noexcept {
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

    /// @copydoc Node::revert()
    void revert(State& state) const final {
        this->template data_ptr<StateData>(state)->revert();
    }

 private:
    struct StateData : public ScalarOutputMixinStateData {
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


    ArrayNode* x_ptr_;

    Lattice lattice_;

    struct neighbor {
        neighbor() noexcept : neighbor(-1, 0) {}
        neighbor(int v) noexcept : neighbor(v, 0) {}
        neighbor(int v, double bias) noexcept : v(v), bias(bias) {}


        friend bool operator==(const neighbor& lhs, const neighbor& rhs) noexcept {
            return lhs.v == rhs.v;
        }
        friend bool operator!=(const neighbor& lhs, const neighbor& rhs) noexcept {
            return lhs.v != rhs.v;
        }
        friend std::weak_ordering operator<=>(const neighbor& lhs, const neighbor& rhs) noexcept {
            return lhs.v <=> rhs.v;
        }

        int v;

        double bias;
    };

    struct neighborhood {
        std::vector<neighbor> neighbors;
        double bias;
    };

    // store the graph in an adjacency format
    std::vector<neighborhood> adj_;
};

using ZephyrNode = LatticeNode<ZephyrLattice>;

}  // namespace dwave::optimization

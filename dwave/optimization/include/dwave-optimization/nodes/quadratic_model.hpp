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

/// The lattice types we currently support.
template <typename T>
concept Lattice = std::same_as<T, ZephyrLattice>;

/// A node representing a quadratic model with linear and quadratic biases
/// structured according to the given lattice.
template <Lattice LatticeType>
class LatticeNode : public ScalarOutputMixin<ArrayNode> {
 public:
    LatticeNode() = delete;

    /// Return a LatticeNode with all zero biases.
    LatticeNode(ArrayNode* x_ptr, LatticeType lattice);

    /// Construct a LatticeNode with the biases provided by the linear and quadratic functions.
    LatticeNode(ArrayNode* x_ptr, LatticeType lattice,
                std::function<double(int)> linear,           // function to get the linear biases
                std::function<double(int, int)> quadratic);  // function to get the quadratic biases

    /// @copydoc Array::buff()
    double const* buff(const State& state) const override;

    /// @copydoc Node::commit()
    void commit(State& state) const override;

    /// @copydoc Array::diff()
    std::span<const Update> diff(const State& state) const override;

    /// @copydoc Node::initialize_state()
    void initialize_state(State& state) const override;

    /// Return a reference to the lattice structure of the node.
    const LatticeType& lattice() const noexcept { return lattice_; }

    /// Get the linear bias associated with `u`. Returns `0` if `u` is out-of-bounds.
    double linear(int u) const noexcept;

    /// @copydoc Node::propagate()
    void propagate(State& state) const override;

    /// Get the linear bias associated with `u` and `v`.
    /// Returns `0` if `u` or `v` are out of bounds or if they have no interaction.
    double quadratic(int u, int v) const noexcept;

    /// @copydoc Node::revert()
    void revert(State& state) const override;

 private:
    struct StateData;

    ArrayNode* x_ptr_;

    LatticeType lattice_;

    struct neighbor {
        neighbor() noexcept : neighbor(-1, 0) {}
        neighbor(int v) noexcept : neighbor(v, 0) {}
        neighbor(int v, double bias) noexcept : v(v), bias(bias) {}

        // We want to be able to sort neighborhoods, so make neighbor weakly ordered
        friend bool operator==(const neighbor& lhs, const neighbor& rhs) noexcept {
            return lhs.v == rhs.v;
        }
        friend bool operator!=(const neighbor& lhs, const neighbor& rhs) noexcept {
            return lhs.v != rhs.v;
        }
        friend std::weak_ordering operator<=>(const neighbor& lhs, const neighbor& rhs) noexcept {
            return lhs.v <=> rhs.v;
        }

        int v;        // the variable index
        double bias;  // the quadratic bias
    };

    struct neighborhood {
        std::vector<neighbor> neighbors;
        double bias;  // the linear bias
    };

    // Store the graph in an adjacency format
    std::vector<neighborhood> adj_;
};

using ZephyrNode = LatticeNode<ZephyrLattice>;

}  // namespace dwave::optimization

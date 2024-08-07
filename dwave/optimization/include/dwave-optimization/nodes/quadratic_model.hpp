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

}  // namespace dwave::optimization

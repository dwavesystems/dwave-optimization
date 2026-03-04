// Copyright 2026 D-Wave Systems Inc.
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

#include <memory>
#include <numeric>
#include <sstream>

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_all.hpp"
#include "dwave-optimization/array.hpp"
#include "dwave-optimization/cp/core/cpvar.hpp"
#include "dwave-optimization/cp/core/interval_array.hpp"
#include "dwave-optimization/cp/state/copier.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/nodes.hpp"
#include "dwave-optimization/state.hpp"
#include "utils.hpp"

using Catch::Matchers::RangeEquals;

namespace dwave::optimization::cp {

TEST_CASE("IntegerNode -> CPVar") {
    dwave::optimization::Graph graph;

    // Add an integer node
    graph.emplace_node<dwave::optimization::IntegerNode>(10, 0, 2);

    // Lock the graph
    graph.topological_sort();

    // Construct the CP corresponding model
    CPModel model;

    // Start adding the
    std::vector<CPVar*> vars;
    for (const auto& n_uptr : graph.nodes()) {
        const ArrayNode* ptr = dynamic_cast<const ArrayNode*>(n_uptr.get());
        REQUIRE(ptr);
        vars.push_back(model.emplace_variable<CPVar>(model, ptr, ptr->topological_index()));
    }

    // Initialize an empty state?
    CPState state = model.initialize_state<Copier>();
    REQUIRE(state.get_propagators_state().size() == 0);
    REQUIRE(state.get_variables_state().size() == 1);

    auto* var = vars[0];
    var->initialize_state(state);
    REQUIRE(var->min_size(state.get_variables_state()) ==
            var->max_size(state.get_variables_state()));
    REQUIRE(var->min_size(state.get_variables_state()) == 10);

    for (ssize_t i = 0; i < 10; ++i) {
        REQUIRE(var->min(state.get_variables_state(), i) == 0);
        REQUIRE(var->max(state.get_variables_state(), i) == 2);
        REQUIRE(var->size(state.get_variables_state(), i) == 3);
        REQUIRE(var->is_active(state.get_variables_state(), i));
    }
}

TEST_CASE("ListNode -> CPVar") {
    dwave::optimization::Graph graph;

    // Add an list node
    graph.emplace_node<dwave::optimization::ListNode>(10, 2, 5);

    // Lock the graph
    graph.topological_sort();

    // Construct the CP corresponding model
    CPModel model;

    // Start adding the
    std::vector<CPVar*> vars;
    for (const auto& n_uptr : graph.nodes()) {
        const ArrayNode* ptr = dynamic_cast<const ArrayNode*>(n_uptr.get());
        REQUIRE(ptr);
        vars.push_back(model.emplace_variable<CPVar>(model, ptr, ptr->topological_index()));
    }

    // Initialize an empty state?
    CPState state = model.initialize_state<Copier>();
    REQUIRE(state.get_propagators_state().size() == 0);
    REQUIRE(state.get_variables_state().size() == 1);

    auto* var = vars[0];
    var->initialize_state(state);
    REQUIRE(var->min_size(state.get_variables_state()) == 2);
    REQUIRE(var->max_size(state.get_variables_state()) == 5);

    for (ssize_t i = 0; i < 5; ++i) {
        REQUIRE(var->min(state.get_variables_state(), i) == 0);
        REQUIRE(var->max(state.get_variables_state(), i) == 9);
        REQUIRE(var->size(state.get_variables_state(), i) == 10);
        if (i < 2) {
            REQUIRE(var->is_active(state.get_variables_state(), i));
        } else {
            REQUIRE_FALSE(var->is_active(state.get_variables_state(), i));
            REQUIRE(var->maybe_active(state.get_variables_state(), i));
        }
    }
}

}  // namespace dwave::optimization::cp

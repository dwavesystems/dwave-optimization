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

TEST_CASE("CP Variable") {
    dwave::optimization::Graph graph;
    CPModel model;

    // Add an integer node
    IntegerNode* i_ptr = graph.emplace_node<dwave::optimization::IntegerNode>(10, 0, 2);
    CPVar* var = model.emplace_variable<CPVar>(model, i_ptr, model.num_variables());
    CPState state = model.initialize_state<Copier>();
    var->initialize_state(state);

    for (ssize_t i = 0; i < 10; ++i) {
        REQUIRE(var->min(state.get_variables_state(), i) == 0);
        REQUIRE(var->max(state.get_variables_state(), i) == 2);
        REQUIRE(var->size(state.get_variables_state(), i) == 3);
    }
}

}  // namespace dwave::optimization::cp

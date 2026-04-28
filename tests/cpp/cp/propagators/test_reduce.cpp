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

#include "../utils.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_all.hpp"
#include "dwave-optimization/array.hpp"
#include "dwave-optimization/cp/core/cpvar.hpp"
#include "dwave-optimization/cp/core/index_transform.hpp"
#include "dwave-optimization/cp/core/interval_array.hpp"
#include "dwave-optimization/cp/propagators/reduce.hpp"
#include "dwave-optimization/cp/state/copier.hpp"
#include "dwave-optimization/nodes.hpp"
#include "dwave-optimization/state.hpp"

using Catch::Matchers::RangeEquals;

namespace dwave::optimization::cp {
TEST_CASE("SumPropagator") {
    GIVEN("A Graph and an integer node") {
        dwave::optimization::Graph graph;

        // Add an integer node
        IntegerNode* x = graph.emplace_node<dwave::optimization::IntegerNode>(3, 0, 5);
        auto y = graph.emplace_node<dwave::optimization::SumNode>(x);

        // Lock the graph
        graph.topological_sort();

        // Construct the CP corresponding model
        AND_GIVEN("A CP Model and fix-point engine") {
            CPModel model;
            CPEngine engine;

            // Add the variables to the model (manually)
            CPVar* cp_x = model.emplace_variable<CPVar>(model, x, x->topological_index());
            CPVar* cp_y = model.emplace_variable<CPVar>(model, y, y->topological_index());

            REQUIRE(cp_y->min_size() == 1);
            REQUIRE(cp_y->max_size() == 1);

            // Add the propagator for the add node
            Propagator* p_add =
                    model.emplace_propagator<SumPropagator>(model.num_propagators(), cp_x, cp_y);

            REQUIRE(cp_x->on_bounds.size() == 1);
            REQUIRE(cp_y->on_bounds.size() == 1);

            WHEN("We initialize a state") {
                CPState state = model.initialize_state<Copier>();
                CPVarsState& v_state = state.get_variables_state();
                CPPropagatorsState& p_state = state.get_propagators_state();

                REQUIRE(v_state.size() == 2);
                REQUIRE(p_state.size() == 1);

                cp_x->initialize_state(state);
                cp_y->initialize_state(state);
                p_add->initialize_state(state);

                THEN("The interval of y is correctly set") {
                    REQUIRE(cp_y->min(v_state, 0) == 0);
                    REQUIRE(cp_y->max(v_state, 0) == 15);
                    REQUIRE(cp_y->size(v_state, 0) == 16);
                }

                AND_WHEN("We restrict x[0] to [2, 5] and we propagate") {
                    cp_x->remove_below(v_state, 2, 0);
                    REQUIRE(cp_x->min(v_state, 0) == 2);
                    REQUIRE(cp_x->max(v_state, 0) == 5);
                    engine.fix_point(state);

                    THEN("The sum output variable is correctly set to [2, 15]") {
                        REQUIRE(cp_y->min(v_state, 0) == 2);
                        REQUIRE(cp_y->max(v_state, 0) == 15);
                    }
                }

                AND_WHEN("We restrict x[1] to [0, 1] and we propagate") {
                    cp_x->remove_above(v_state, 1, 1);
                    REQUIRE(cp_x->min(v_state, 1) == 0);
                    REQUIRE(cp_x->max(v_state, 1) == 1);

                    engine.fix_point(state);
                    THEN("The sum output variable is correctly set to [0, 11]") {
                        REQUIRE(cp_y->min(v_state, 0) == 0);
                        REQUIRE(cp_y->max(v_state, 0) == 11);
                    }
                }

                AND_WHEN("We restrict the sum to [3, 10]") {
                    cp_y->remove_below(v_state, 3, 0);
                    cp_y->remove_above(v_state, 10, 0);
                    REQUIRE(cp_y->max(v_state, 0) == 10);
                    REQUIRE(cp_y->min(v_state, 0) == 3);

                    AND_WHEN("We propagate") {
                        engine.fix_point(state);

                        THEN("The input variables gets unchanged]") {
                            for (int i = 0; i < 3; ++i) {
                                REQUIRE(cp_x->min(v_state, i) == 0);
                                REQUIRE(cp_x->max(v_state, i) == 5);
                            }
                        }
                    }
                }
            }
        }
    }
}
}  // namespace dwave::optimization::cp
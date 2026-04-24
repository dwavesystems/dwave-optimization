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
#include "dwave-optimization/cp/propagators/collections.hpp"
#include "dwave-optimization/cp/state/copier.hpp"
#include "dwave-optimization/nodes.hpp"
#include "dwave-optimization/state.hpp"

using Catch::Matchers::RangeEquals;

namespace dwave::optimization::cp {
TEST_CASE("All different (FWC) propagator") {
    GIVEN("A Graph and a static list node") {
        dwave::optimization::Graph graph;

        // Add an integer node
        ListNode* x = graph.emplace_node<ListNode>(5, 5, 5);

        // Lock the graph
        graph.topological_sort();

        // Construct the CP corresponding model
        AND_GIVEN("A CP Model and fix-point engine") {
            CPModel model;
            CPEngine engine;

            // Add the variables to the model (manually)
            CPVar* cp_x = model.emplace_variable<CPVar>(model, x, x->topological_index());
            REQUIRE(cp_x->min_size() == 5);
            REQUIRE(cp_x->max_size() == 5);

            // Add the propagator for the add node
            Propagator* p_add = model.emplace_propagator<AllDifferentFWCPropagator>(
                    model.num_propagators(), cp_x);

            REQUIRE(cp_x->on_bind.size() == 1);

            WHEN("We initialize a state") {
                CPState state = model.initialize_state<Copier>();
                CPVarsState& v_state = state.get_variables_state();
                CPPropagatorsState& p_state = state.get_propagators_state();

                REQUIRE(v_state.size() == 1);
                REQUIRE(p_state.size() == 1);

                cp_x->initialize_state(state, true);

                p_add->initialize_state(state);

                AND_WHEN("We assign x[0] to 2 and we propagate") {
                    cp_x->assign(v_state, 2, 0);
                    REQUIRE(cp_x->min(v_state, 0) == 2);
                    REQUIRE(cp_x->max(v_state, 0) == 2);
                    engine.fix_point(state);

                    THEN("The other variables don't contain 2") {
                        for (ssize_t i = 1; i < 5; ++i) {
                            REQUIRE_FALSE(cp_x->contains(v_state, 2, i));
                        }
                    }

                    AND_WHEN("We also assign x[1] to 3") {
                        cp_x->assign(v_state, 3, 1);
                        REQUIRE(cp_x->min(v_state, 1) == 3);
                        REQUIRE(cp_x->max(v_state, 1) == 3);
                        engine.fix_point(state);
                        THEN("The other variables don't contain 2") {
                            for (ssize_t i = 2; i < 5; ++i) {
                                REQUIRE_FALSE(cp_x->contains(v_state, 2, i));
                                REQUIRE_FALSE(cp_x->contains(v_state, 3, i));
                            }
                        }
                    }
                }
            }
        }
    }

    GIVEN("A Graph and a dynamic list node") {
        dwave::optimization::Graph graph;

        // Add an integer node
        ListNode* x = graph.emplace_node<ListNode>(5, 2, 5);

        // Lock the graph
        graph.topological_sort();

        // Construct the CP corresponding model
        AND_GIVEN("A CP Model and fix-point engine") {
            CPModel model;
            CPEngine engine;

            // Add the variables to the model (manually)
            CPVar* cp_x = model.emplace_variable<CPVar>(model, x, x->topological_index());
            REQUIRE(cp_x->min_size() == 2);
            REQUIRE(cp_x->max_size() == 5);

            // Add the propagator for the add node
            Propagator* p_add = model.emplace_propagator<AllDifferentFWCPropagator>(
                    model.num_propagators(), cp_x);

            REQUIRE(cp_x->on_bind.size() == 1);

            WHEN("We initialize a state") {
                CPState state = model.initialize_state<Copier>();
                CPVarsState& v_state = state.get_variables_state();
                CPPropagatorsState& p_state = state.get_propagators_state();

                REQUIRE(v_state.size() == 1);
                REQUIRE(p_state.size() == 1);

                cp_x->initialize_state(state, true);

                p_add->initialize_state(state);

                AND_WHEN("We assign x[0] to 2 and we propagate") {
                    cp_x->assign(v_state, 2, 0);
                    REQUIRE(cp_x->min(v_state, 0) == 2);
                    REQUIRE(cp_x->max(v_state, 0) == 2);
                    engine.fix_point(state);

                    THEN("The other variables don't contain 2") {
                        for (ssize_t i = 1; i < 5; ++i) {
                            REQUIRE_FALSE(cp_x->contains(v_state, 2, i));
                        }
                    }

                    AND_WHEN("We also assign x[3] to 3") {
                        cp_x->assign(v_state, 3, 3);
                        REQUIRE(cp_x->min(v_state, 3) == 3);
                        REQUIRE(cp_x->max(v_state, 3) == 3);
                        REQUIRE_FALSE(cp_x->is_active(v_state, 3));
                        REQUIRE(cp_x->maybe_active(v_state, 3));
                        engine.fix_point(state);
                        THEN("The other variables don't contain 2, but contain 3") {
                            for (ssize_t i = 0; i < 5; ++i) {
                                if (i != 0) {
                                    REQUIRE_FALSE(cp_x->contains(v_state, 2, i));
                                }
                                if (i == 3 or i == 0) continue;
                                std::cout << "i " << i << " constains 3? "
                                          << cp_x->contains(v_state, 3, i) << "\n";
                                REQUIRE(cp_x->contains(v_state, 3, i));
                            }
                        }
                    }
                }
            }
        }
    }
}
}  // namespace dwave::optimization::cp
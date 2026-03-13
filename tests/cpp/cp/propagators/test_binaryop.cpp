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
#include "dwave-optimization/cp/propagators/binaryop.hpp"
#include "dwave-optimization/cp/state/copier.hpp"
#include "dwave-optimization/nodes.hpp"
#include "dwave-optimization/state.hpp"

using Catch::Matchers::RangeEquals;

namespace dwave::optimization::cp {
TEST_CASE("AddPropagator") {
    GIVEN("A Graph and an integer node") {
        dwave::optimization::Graph graph;

        // Add an integer node
        IntegerNode* x = graph.emplace_node<dwave::optimization::IntegerNode>(10, 0, 5);
        IntegerNode* y = graph.emplace_node<dwave::optimization::IntegerNode>(10, -2, 3);
        auto z = graph.emplace_node<dwave::optimization::AddNode>(x, y);

        // Lock the graph
        graph.topological_sort();

        // Construct the CP corresponding model
        AND_GIVEN("A CP Model and fix-point engine") {
            CPModel model;
            CPEngine engine;

            // Add the variables to the model (manually)
            CPVar* cp_x = model.emplace_variable<CPVar>(model, x, x->topological_index());
            CPVar* cp_y = model.emplace_variable<CPVar>(model, y, y->topological_index());
            CPVar* cp_z = model.emplace_variable<CPVar>(model, z, z->topological_index());

            REQUIRE(cp_z->min_size() == 10);
            REQUIRE(cp_z->max_size() == 10);

            // Add the propagator for the add node
            Propagator* p_add = model.emplace_propagator<AddPropagator>(model.num_propagators(),
                                                                        cp_x, cp_y, cp_z);

            REQUIRE(cp_x->on_bounds.size() == 1);
            REQUIRE(cp_y->on_bounds.size() == 1);
            REQUIRE(cp_z->on_bounds.size() == 1);

            WHEN("We initialize a state") {
                CPState state = model.initialize_state<Copier>();
                CPVarsState& v_state = state.get_variables_state();
                CPPropagatorsState& p_state = state.get_propagators_state();

                REQUIRE(v_state.size() == 3);
                REQUIRE(p_state.size() == 1);

                cp_x->initialize_state(state);
                cp_y->initialize_state(state);
                cp_z->initialize_state(state);
                p_add->initialize_state(state);

                THEN("The interval of z is correctly set") {
                    for (int i = 0; i < 10; ++i) {
                        REQUIRE(cp_z->min(v_state, i) == -2);
                        REQUIRE(cp_z->max(v_state, i) == 8);
                        REQUIRE(cp_z->size(v_state, i) == 11);
                    }
                }

                AND_WHEN("We restrict x[0] to [2, 5] and we propagate") {
                    cp_x->remove_below(v_state, 2, 0);
                    REQUIRE(cp_x->min(v_state, 0) == 2);
                    REQUIRE(cp_x->max(v_state, 0) == 5);
                    engine.fix_point(state);

                    THEN("The sum output variable 0 is correctly set to [0, 8]") {
                        REQUIRE(cp_z->min(v_state, 0) == 0);
                        REQUIRE(cp_z->max(v_state, 0) == 8);
                    }
                }

                AND_WHEN("We restrict y[1] to [-2, 1] and we propagate") {
                    cp_y->remove_above(v_state, 1, 1);
                    REQUIRE(cp_y->min(v_state, 1) == -2);
                    REQUIRE(cp_y->max(v_state, 1) == 1);

                    engine.fix_point(state);
                    THEN("The sum output variable is correctly set to [-2, 6]") {
                        REQUIRE(cp_z->min(v_state, 1) == -2);
                        REQUIRE(cp_z->max(v_state, 1) == 6);
                    }
                }

                AND_WHEN("We restrict the sum[2] to [-1, 1]") {
                    cp_z->remove_below(v_state, -1, 2);
                    cp_z->remove_above(v_state, 1, 2);
                    REQUIRE(cp_z->max(v_state, 2) == 1);
                    REQUIRE(cp_z->min(v_state, 2) == -1);

                    AND_WHEN("We propagate") {
                        engine.fix_point(state);
                        THEN("The input variables gets changed accordingly to x[2] in [0, 3] and "
                             "y[2] in "
                             "[-2, 1]") {
                            REQUIRE(cp_x->min(v_state, 2) == 0);
                            REQUIRE(cp_x->max(v_state, 2) == 3);
                            REQUIRE(cp_y->min(v_state, 2) == -2);
                            REQUIRE(cp_y->max(v_state, 2) == 1);
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("LessEqualPropagator") {
    GIVEN("A Graph with two integer nodes x in [0, 2] and y in [3, 5] and their <= expression") {
        dwave::optimization::Graph graph;

        // Add an integer node
        IntegerNode* x = graph.emplace_node<dwave::optimization::IntegerNode>(3, 0, 2);
        IntegerNode* y = graph.emplace_node<dwave::optimization::IntegerNode>(3, 3, 5);
        auto z = graph.emplace_node<dwave::optimization::LessEqualNode>(x, y);

        // Lock the graph
        graph.topological_sort();

        // Construct the CP corresponding model
        AND_GIVEN("A CP Model and fix-point engine") {
            CPModel model;
            CPEngine engine;

            // Add the variables to the model (manually)
            CPVar* cp_x = model.emplace_variable<CPVar>(model, x, x->topological_index());
            CPVar* cp_y = model.emplace_variable<CPVar>(model, y, y->topological_index());
            CPVar* cp_z = model.emplace_variable<CPVar>(model, z, z->topological_index());

            REQUIRE(cp_z->min_size() == 3);
            REQUIRE(cp_z->max_size() == 3);

            // Add the propagator for the add node
            Propagator* p_add = model.emplace_propagator<LessEqualPropagator>(
                    model.num_propagators(), cp_x, cp_y, cp_z);

            REQUIRE(cp_x->on_bounds.size() == 1);
            REQUIRE(cp_y->on_bounds.size() == 1);
            REQUIRE(cp_z->on_bounds.size() == 1);

            WHEN("We initialize a state") {
                CPState state = model.initialize_state<Copier>();
                CPVarsState& v_state = state.get_variables_state();
                CPPropagatorsState& p_state = state.get_propagators_state();

                REQUIRE(v_state.size() == 3);
                REQUIRE(p_state.size() == 1);

                cp_x->initialize_state(state);
                cp_y->initialize_state(state);
                cp_z->initialize_state(state);
                p_add->initialize_state(state);

                THEN("The interval of z is correctly set") {
                    for (int i = 0; i < 3; ++i) {
                        REQUIRE(cp_z->min(v_state, i) == 0);
                        REQUIRE(cp_z->max(v_state, i) == 1);
                        REQUIRE(cp_z->size(v_state, i) == 2);
                    }
                }

                AND_WHEN("We restrict y[0] to [4, 5] and we propagate") {
                    cp_y->remove_below(v_state, 4, 0);
                    REQUIRE(cp_y->min(v_state, 0) == 4);
                    REQUIRE(cp_y->max(v_state, 0) == 5);
                    engine.fix_point(state);

                    THEN("The <= output variable 0 is left unchanged to [1, 1]") {
                        REQUIRE(cp_z->min(v_state, 0) == 1);
                        REQUIRE(cp_z->max(v_state, 0) == 1);
                    }
                }
            }
        }
    }

    GIVEN("A Graph with two integer nodes x in [0, 3] and y in [2, 5] and their <= expression") {
        dwave::optimization::Graph graph;

        // Add an integer node
        IntegerNode* x = graph.emplace_node<dwave::optimization::IntegerNode>(3, 0, 3);
        IntegerNode* y = graph.emplace_node<dwave::optimization::IntegerNode>(3, 2, 5);
        auto z = graph.emplace_node<dwave::optimization::LessEqualNode>(x, y);

        // Lock the graph
        graph.topological_sort();

        // Construct the CP corresponding model
        AND_GIVEN("A CP Model and fix-point engine") {
            CPModel model;
            CPEngine engine;

            // Add the variables to the model (manually)
            CPVar* cp_x = model.emplace_variable<CPVar>(model, x, x->topological_index());
            CPVar* cp_y = model.emplace_variable<CPVar>(model, y, y->topological_index());
            CPVar* cp_z = model.emplace_variable<CPVar>(model, z, z->topological_index());

            REQUIRE(cp_z->min_size() == 3);
            REQUIRE(cp_z->max_size() == 3);

            // Add the propagator for the add node
            Propagator* p_add = model.emplace_propagator<LessEqualPropagator>(
                    model.num_propagators(), cp_x, cp_y, cp_z);

            REQUIRE(cp_x->on_bounds.size() == 1);
            REQUIRE(cp_y->on_bounds.size() == 1);
            REQUIRE(cp_z->on_bounds.size() == 1);

            WHEN("We initialize a state") {
                CPState state = model.initialize_state<Copier>();
                CPVarsState& v_state = state.get_variables_state();
                CPPropagatorsState& p_state = state.get_propagators_state();

                REQUIRE(v_state.size() == 3);
                REQUIRE(p_state.size() == 1);

                cp_x->initialize_state(state);
                cp_y->initialize_state(state);
                cp_z->initialize_state(state);
                p_add->initialize_state(state);

                THEN("The interval of z is correctly set") {
                    for (int i = 0; i < 3; ++i) {
                        REQUIRE(cp_z->min(v_state, i) == 0);
                        REQUIRE(cp_z->max(v_state, i) == 1);
                        REQUIRE(cp_z->size(v_state, i) == 2);
                    }
                }

                AND_WHEN("We restrict y[0] to [4, 5] and we propagate") {
                    cp_y->remove_below(v_state, 4, 0);
                    REQUIRE(cp_y->min(v_state, 0) == 4);
                    REQUIRE(cp_y->max(v_state, 0) == 5);
                    engine.fix_point(state);

                    THEN("The <= output variable 0 is changed to [1, 1]") {
                        REQUIRE(cp_z->min(v_state, 0) == 1);
                        REQUIRE(cp_z->max(v_state, 0) == 1);
                    }
                }

                AND_WHEN("We restrict z[1] to [1, 1] and we propagate") {
                    cp_z->assign(v_state, 1, 1);
                    REQUIRE(cp_z->min(v_state, 1) == 1);
                    REQUIRE(cp_z->max(v_state, 1) == 1);
                    engine.fix_point(state);

                    THEN("Input nodes are left unchanged") {
                        REQUIRE(cp_x->min(v_state, 1) == 0);
                        REQUIRE(cp_x->max(v_state, 1) == 3);
                        REQUIRE(cp_y->min(v_state, 1) == 2);
                        REQUIRE(cp_y->max(v_state, 1) == 5);
                    }
                }

                AND_WHEN("We restrict z[2] to [0, 0] and we propagate") {
                    cp_z->assign(v_state, 0, 2);
                    REQUIRE(cp_z->min(v_state, 2) == 0);
                    REQUIRE(cp_z->max(v_state, 2) == 0);
                    engine.fix_point(state);

                    THEN("Input nodes are shrunk accordingly") {
                        REQUIRE(cp_x->min(v_state, 2) == 2);
                        REQUIRE(cp_x->max(v_state, 2) == 3);
                        REQUIRE(cp_y->min(v_state, 2) == 2);
                        REQUIRE(cp_y->max(v_state, 2) == 3);
                    }
                }
            }
        }
    }

    GIVEN("A Graph with two integer nodes x in [2, 5] and y in [0, 3] and their <= expression") {
        dwave::optimization::Graph graph;

        // Add an integer node
        IntegerNode* x = graph.emplace_node<dwave::optimization::IntegerNode>(3, 2, 5);
        IntegerNode* y = graph.emplace_node<dwave::optimization::IntegerNode>(3, 0, 3);
        auto z = graph.emplace_node<dwave::optimization::LessEqualNode>(x, y);

        // Lock the graph
        graph.topological_sort();

        // Construct the CP corresponding model
        AND_GIVEN("A CP Model and fix-point engine") {
            CPModel model;
            CPEngine engine;

            // Add the variables to the model (manually)
            CPVar* cp_x = model.emplace_variable<CPVar>(model, x, x->topological_index());
            CPVar* cp_y = model.emplace_variable<CPVar>(model, y, y->topological_index());
            CPVar* cp_z = model.emplace_variable<CPVar>(model, z, z->topological_index());

            REQUIRE(cp_z->min_size() == 3);
            REQUIRE(cp_z->max_size() == 3);

            // Add the propagator for the add node
            Propagator* p_add = model.emplace_propagator<LessEqualPropagator>(
                    model.num_propagators(), cp_x, cp_y, cp_z);

            REQUIRE(cp_x->on_bounds.size() == 1);
            REQUIRE(cp_y->on_bounds.size() == 1);
            REQUIRE(cp_z->on_bounds.size() == 1);

            WHEN("We initialize a state") {
                CPState state = model.initialize_state<Copier>();
                CPVarsState& v_state = state.get_variables_state();
                CPPropagatorsState& p_state = state.get_propagators_state();

                REQUIRE(v_state.size() == 3);
                REQUIRE(p_state.size() == 1);

                cp_x->initialize_state(state);
                cp_y->initialize_state(state);
                cp_z->initialize_state(state);
                p_add->initialize_state(state);

                THEN("The interval of z is correctly set") {
                    for (int i = 0; i < 3; ++i) {
                        REQUIRE(cp_z->min(v_state, i) == 0);
                        REQUIRE(cp_z->max(v_state, i) == 1);
                        REQUIRE(cp_z->size(v_state, i) == 2);
                    }
                }

                AND_WHEN("We restrict x[0] to [4, 5] and we propagate") {
                    cp_x->remove_below(v_state, 4, 0);
                    REQUIRE(cp_x->min(v_state, 0) == 4);
                    REQUIRE(cp_x->max(v_state, 0) == 5);
                    engine.fix_point(state);

                    THEN("The <= output variable 0 is changed to [0, 0]") {
                        REQUIRE(cp_z->min(v_state, 0) == 0);
                        REQUIRE(cp_z->max(v_state, 0) == 0);
                    }
                }

                AND_WHEN("We restrict z[1] to [1, 1] and we propagate") {
                    cp_z->assign(v_state, 1, 1);
                    REQUIRE(cp_z->min(v_state, 1) == 1);
                    REQUIRE(cp_z->max(v_state, 1) == 1);
                    engine.fix_point(state);

                    THEN("Input nodes are shrunk accordingly") {
                        REQUIRE(cp_x->min(v_state, 1) == 2);
                        REQUIRE(cp_x->max(v_state, 1) == 3);
                        REQUIRE(cp_y->min(v_state, 1) == 2);
                        REQUIRE(cp_y->max(v_state, 1) == 3);
                    }
                }

                AND_WHEN("We restrict z[2] to [0, 0] and we propagate") {
                    cp_z->assign(v_state, 0, 2);
                    REQUIRE(cp_z->min(v_state, 2) == 0);
                    REQUIRE(cp_z->max(v_state, 2) == 0);
                    engine.fix_point(state);

                    THEN("Input nodes are left unchanged") {
                        REQUIRE(cp_x->min(v_state, 2) == 2);
                        REQUIRE(cp_x->max(v_state, 2) == 5);
                        REQUIRE(cp_y->min(v_state, 2) == 0);
                        REQUIRE(cp_y->max(v_state, 2) == 3);
                    }
                }
            }
        }
    }
}

}  // namespace dwave::optimization::cp

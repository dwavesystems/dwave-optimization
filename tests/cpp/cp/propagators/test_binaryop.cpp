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

    GIVEN("A Graph with an integer node, constant node, and add node") {
        dwave::optimization::Graph graph;

        // Add an integer node
        IntegerNode* x = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, 0, 5);
        ConstantNode* c = graph.emplace_node<ConstantNode>(0.5);
        auto z = graph.emplace_node<AddNode>(x, c);

        // Lock the graph
        graph.topological_sort();

        // Construct the CP corresponding model
        AND_GIVEN("A CP Model and fix-point engine") {
            CPModel model;
            CPEngine engine;

            // Add the variables to the model (manually)
            CPVar* cp_x = model.emplace_variable<CPVar>(model, x, x->topological_index());
            CPVar* cp_c = model.emplace_variable<CPVar>(model, c, c->topological_index());
            CPVar* cp_z = model.emplace_variable<CPVar>(model, z, z->topological_index());

            REQUIRE(cp_z->min_size() == 1);
            REQUIRE(cp_z->max_size() == 1);

            // Add the propagator for the add node
            Propagator* p_add = model.emplace_propagator<AddPropagator>(model.num_propagators(),
                                                                        cp_x, cp_c, cp_z);

            REQUIRE(cp_x->on_bounds.size() == 1);
            REQUIRE(cp_c->on_bounds.size() == 1);
            REQUIRE(cp_z->on_bounds.size() == 1);

            WHEN("We initialize a state") {
                CPState state = model.initialize_state<Copier>();
                CPVarsState& v_state = state.get_variables_state();
                CPPropagatorsState& p_state = state.get_propagators_state();

                REQUIRE(v_state.size() == 3);
                REQUIRE(p_state.size() == 1);

                cp_x->initialize_state(state);
                cp_c->initialize_state(state);
                cp_z->initialize_state(state);
                p_add->initialize_state(state);

                THEN("The interval of z is correctly set") {
                    REQUIRE(cp_z->min(v_state, 0) == 0.5);
                    REQUIRE(cp_z->max(v_state, 0) == 5.5);
                    REQUIRE(cp_z->size(v_state, 0) == 5);
                }

                AND_WHEN("We restrict z to [2, 5] and we propagate") {
                    cp_z->remove_below(v_state, 2, 0);
                    cp_z->remove_above(v_state, 5, 0);
                    REQUIRE(cp_z->min(v_state, 0) == 2);
                    REQUIRE(cp_z->max(v_state, 0) == 5);
                    engine.fix_point(state);

                    THEN("x's bounds are now correctly set to [2, 4], and z's bounds are set to "
                         "[2.5, 4.5]") {
                        CHECK(cp_x->min(v_state, 0) == 2);
                        CHECK(cp_x->max(v_state, 0) == 4);

                        CHECK(cp_z->min(v_state, 0) == 2.5);
                        CHECK(cp_z->max(v_state, 0) == 4.5);
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

TEST_CASE("MultiplyPropagator") {
    GIVEN("A Graph and an integer node") {
        dwave::optimization::Graph graph;

        // Add an integer node
        IntegerNode* x = graph.emplace_node<dwave::optimization::IntegerNode>(10, 0, 5);
        IntegerNode* y = graph.emplace_node<dwave::optimization::IntegerNode>(10, -2, 3);
        auto z = graph.emplace_node<dwave::optimization::MultiplyNode>(x, y);

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

            // Add the propagator for the multiply node
            Propagator* p_mul = model.emplace_propagator<MultiplyPropagator>(
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
                p_mul->initialize_state(state);

                THEN("The interval of z is correctly set") {
                    for (int i = 0; i < 10; ++i) {
                        CHECK(cp_z->min(v_state, i) == -10);
                        CHECK(cp_z->max(v_state, i) == 15);
                        CHECK(cp_z->size(v_state, i) == 26);
                    }
                }

                AND_WHEN("We restrict x[0] to [2, 4] and we propagate") {
                    // x -> [2, 4]
                    // y -> [-2, 3]
                    // mul -> [-8, 12]
                    cp_x->remove_below(v_state, 2, 0);
                    cp_x->remove_above(v_state, 4, 0);
                    REQUIRE(cp_x->min(v_state, 0) == 2);
                    REQUIRE(cp_x->max(v_state, 0) == 4);
                    engine.fix_point(state);

                    THEN("The multiply output variable 0 is correctly set to [-8, 12]") {
                        CHECK(cp_z->min(v_state, 0) == -8);
                        CHECK(cp_z->max(v_state, 0) == 12);
                    }
                }

                AND_WHEN("We restrict y[1] to [-2, 1] and we propagate") {
                    // x -> [0, 5]
                    // y -> [-2, 1]
                    // mul -> [-10, 5]
                    cp_y->remove_above(v_state, 1, 1);
                    REQUIRE(cp_y->min(v_state, 1) == -2);
                    REQUIRE(cp_y->max(v_state, 1) == 1);
                    engine.fix_point(state);

                    THEN("The multiply output variable is correctly set to [-10, 5]") {
                        CHECK(cp_z->min(v_state, 1) == -10);
                        CHECK(cp_z->max(v_state, 1) == 5);
                    }
                }

                AND_WHEN("We restrict mul[2] to [-1, 1]") {
                    // x -> [0, 5]
                    // y -> [-2, 3]
                    // mul -> [-1, 1]
                    //
                    // Because x and y include 0 in their domains, the change to mul's
                    // domain should have no effect
                    cp_z->remove_below(v_state, -1, 2);
                    cp_z->remove_above(v_state, 1, 2);
                    REQUIRE(cp_z->min(v_state, 2) == -1);
                    REQUIRE(cp_z->max(v_state, 2) == 1);

                    AND_WHEN("We propagate") {
                        engine.fix_point(state);
                        THEN("The bounds have not changed") {
                            CHECK(cp_x->min(v_state, 2) == 0);
                            CHECK(cp_x->max(v_state, 2) == 5);
                            CHECK(cp_y->min(v_state, 2) == -2);
                            CHECK(cp_y->max(v_state, 2) == 3);
                        }
                    }
                }

                AND_WHEN("We restrict x[3] to [2, 4] and mul[3] to [-1, 2] and propagate") {
                    // x -> [2, 4]
                    // y -> [-2, 3]
                    // mul -> [-1, 2]
                    //
                    // y should then be restricted to [-0.5, 1] and then "rounded" to [0, 1].
                    // This should further restrict mul to [0, 2]

                    cp_x->remove_below(v_state, 2, 3);
                    cp_x->remove_above(v_state, 4, 3);
                    cp_z->remove_below(v_state, -1, 3);
                    cp_z->remove_above(v_state, 2, 3);
                    REQUIRE(cp_x->min(v_state, 3) == 2);
                    REQUIRE(cp_x->max(v_state, 3) == 4);
                    REQUIRE(cp_z->min(v_state, 3) == -1);
                    REQUIRE(cp_z->max(v_state, 3) == 2);

                    engine.fix_point(state);

                    THEN("Check x has the same bounds, y has been restricted to [0, 1], and z to "
                         "[0, 2]") {
                        CHECK(cp_x->min(v_state, 3) == 2);
                        CHECK(cp_x->max(v_state, 3) == 4);
                        CHECK(cp_y->min(v_state, 3) == 0);
                        CHECK(cp_y->max(v_state, 3) == 1);
                        CHECK(cp_z->min(v_state, 3) == 0);
                        CHECK(cp_z->max(v_state, 3) == 2);
                    }
                }
            }
        }
    }
}

TEST_CASE("DividePropagator") {
    GIVEN("A Graph and an integer node") {
        dwave::optimization::Graph graph;

        // Add an integer node
        IntegerNode* x = graph.emplace_node<dwave::optimization::IntegerNode>(10, -2, 5);
        IntegerNode* y = graph.emplace_node<dwave::optimization::IntegerNode>(10, 1, 4);
        auto z = graph.emplace_node<dwave::optimization::DivideNode>(x, y);

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

            // Add the propagator for the divide node
            Propagator* p_div = model.emplace_propagator<DividePropagator>(model.num_propagators(),
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
                p_div->initialize_state(state);

                THEN("The interval of z is correctly set") {
                    for (int i = 0; i < 10; ++i) {
                        CHECK(cp_z->min(v_state, i) == -2);
                        CHECK(cp_z->max(v_state, i) == 5);
                        CHECK(cp_z->size(v_state, i) == 7);
                    }
                }

                AND_WHEN("We restrict x[0] to [2, 4] and we propagate") {
                    // x -> [2, 4]
                    // y -> [1, 4]
                    // div -> [0.5, 4]
                    cp_x->remove_below(v_state, 2, 0);
                    cp_x->remove_above(v_state, 4, 0);
                    REQUIRE(cp_x->min(v_state, 0) == 2);
                    REQUIRE(cp_x->max(v_state, 0) == 4);
                    engine.fix_point(state);

                    THEN("The divide output variable 0's domain is correctly set to [0.5, 4]") {
                        CHECK(cp_z->min(v_state, 0) == 0.5);
                        CHECK(cp_z->max(v_state, 0) == 4);
                    }
                }

                AND_WHEN("We restrict y[1] to [2, 3] and we propagate") {
                    // x -> [-2, 5]
                    // y -> [2, 3]
                    // div -> [-1, 2.5]
                    cp_y->remove_below(v_state, 2, 1);
                    cp_y->remove_above(v_state, 3, 1);
                    REQUIRE(cp_y->min(v_state, 1) == 2);
                    REQUIRE(cp_y->max(v_state, 1) == 3);
                    engine.fix_point(state);

                    THEN("The divide output variable 1's domain is correctly set to [-10, 5]") {
                        CHECK(cp_z->min(v_state, 1) == -1);
                        CHECK(cp_z->max(v_state, 1) == 2.5);
                    }
                }

                AND_WHEN("We restrict div[2] to [-1, 1]") {
                    // x -> [-2, 5]
                    // y -> [1, 4]
                    // div -> [-1, 1]
                    //
                    // x should be restricted to [-2, 4]
                    // y should remain at [1, 4]
                    cp_z->remove_below(v_state, -1, 2);
                    cp_z->remove_above(v_state, 1, 2);
                    REQUIRE(cp_z->min(v_state, 2) == -1);
                    REQUIRE(cp_z->max(v_state, 2) == 1);

                    AND_WHEN("We propagate") {
                        engine.fix_point(state);
                        THEN("The bounds of x and y are correct") {
                            CHECK(cp_x->min(v_state, 2) == -2);
                            CHECK(cp_x->max(v_state, 2) == 4);
                            CHECK(cp_y->min(v_state, 2) == 1);
                            CHECK(cp_y->max(v_state, 2) == 4);
                        }
                    }
                }

                AND_WHEN("We restrict x[3] to [2, 4] and div[3] to [-1, 2] and propagate") {
                    // x -> [2, 4]
                    // y -> [1, 4]
                    // div -> [-1, 0.5]
                    //
                    // Forward prop should make div's minimum 0.5 (fixing it to 0.5).
                    // This should then restrict x's maximum to 2 (fixing it to 2),
                    // and restrict y's minimum to 4 (fixing it to 4).

                    cp_x->remove_below(v_state, 2, 3);
                    cp_x->remove_above(v_state, 4, 3);
                    cp_z->remove_below(v_state, -1, 3);
                    cp_z->remove_above(v_state, 0.5, 3);
                    REQUIRE(cp_x->min(v_state, 3) == 2);
                    REQUIRE(cp_x->max(v_state, 3) == 4);
                    REQUIRE(cp_z->min(v_state, 3) == -1);
                    REQUIRE(cp_z->max(v_state, 3) == 0.5);

                    engine.fix_point(state);

                    THEN("Check that x has been fixed to 2, y to 4 and div to 0.5") {
                        CHECK(cp_x->min(v_state, 3) == 2);
                        CHECK(cp_x->max(v_state, 3) == 2);
                        CHECK(cp_y->min(v_state, 3) == 4);
                        CHECK(cp_y->max(v_state, 3) == 4);
                        CHECK(cp_z->min(v_state, 3) == 0.5);
                        CHECK(cp_z->max(v_state, 3) == 0.5);
                    }
                }

                AND_WHEN("We restrict x and y to be positive and div to be negative") {
                    // x -> [2, 4]
                    // y -> [1, 4]
                    // div -> [-1, -0.5]

                    cp_x->remove_below(v_state, 2, 3);
                    cp_x->remove_above(v_state, 4, 3);
                    cp_z->remove_below(v_state, -1, 3);
                    cp_z->remove_above(v_state, -0.5, 3);
                    REQUIRE(cp_x->min(v_state, 3) == 2);
                    REQUIRE(cp_x->max(v_state, 3) == 4);
                    REQUIRE(cp_z->min(v_state, 3) == -1);
                    REQUIRE(cp_z->max(v_state, 3) == -0.5);

                    THEN("Fix point will find an inconsistency") {
                        CHECK(engine.fix_point(state) == CPStatus::Inconsistency);
                    }
                }
            }
        }
    }
}

}  // namespace dwave::optimization::cp

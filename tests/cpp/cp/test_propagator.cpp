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
#include "dwave-optimization/cp/core/index_transform.hpp"
#include "dwave-optimization/cp/core/interval_array.hpp"
#include "dwave-optimization/cp/propagators/identity_propagator.hpp"
#include "dwave-optimization/cp/propagators/indexing_propagators.hpp"
#include "dwave-optimization/cp/state/copier.hpp"
#include "dwave-optimization/nodes.hpp"
#include "dwave-optimization/state.hpp"
#include "utils.hpp"

using Catch::Matchers::RangeEquals;

namespace dwave::optimization::cp {
TEST_CASE("ElementWiseIdentityPropagator") {
    GIVEN("A Graph and an integer node") {
        dwave::optimization::Graph graph;

        // Add an integer node
        graph.emplace_node<dwave::optimization::IntegerNode>(10, 0, 5);

        // Lock the graph
        graph.topological_sort();

        // Construct the CP corresponding model
        AND_GIVEN("The CP Model") {
            CPModel model;

            // Add the variabbles to the model
            std::vector<CPVar*> vars;
            for (const auto& n_uptr : graph.nodes()) {
                const ArrayNode* ptr = dynamic_cast<const ArrayNode*>(n_uptr.get());
                REQUIRE(ptr);
                vars.push_back(model.emplace_variable<CPVar>(model, ptr, ptr->topological_index()));
            }
            CPVar* var = vars[0];
            Propagator* p = model.emplace_propagator<ElementWiseIdentityPropagator>(
                    model.num_propagators(), var);

            // build the advisor for the propagator p aimed to the variable var
            Advisor advisor(p, 0, std::make_unique<ElementWiseTransform>());
            var->propagate_on_domain_change(std::move(advisor));
            REQUIRE(var->on_domain.size() == 1);
            WHEN("We initialize a state") {
                CPState state = model.initialize_state<Copier>();
                CPVarsState& s_state = state.get_variables_state();
                CPPropagatorsState& p_state = state.get_propagators_state();

                REQUIRE(s_state.size() == 1);
                REQUIRE(p_state.size() == 1);

                var->initialize_state(state);
                p->initialize_state(state);

                AND_WHEN("We alter the domain of one variable") {
                    CPStatus status = var->assign(s_state, 3, 2);
                    REQUIRE(status == CPStatus::OK);
                    THEN("We see that the propagator is triggered to run on the same index") {
                        REQUIRE(p_state[0]->scheduled());
                        REQUIRE(p->num_indices_to_process(p_state) == 1);

                        for (int i = 0; i < 10; ++i) {
                            REQUIRE(p->active(p_state, i));
                            if (i == 2) {
                                REQUIRE(p->scheduled(p_state, i));
                            } else {
                                REQUIRE_FALSE(p->scheduled(p_state, i));
                            }
                        }
                    }
                }

                AND_WHEN("We alter the domain of a variable twice") {
                    CPStatus status = var->remove_above(s_state, 4, 0);
                    REQUIRE(status == CPStatus::OK);
                    status = var->remove_below(s_state, 2, 0);
                    REQUIRE(status == CPStatus::OK);

                    THEN("We see that the propagator is triggered to run on the same index only "
                         "once") {
                        REQUIRE(p_state[0]->scheduled());
                        REQUIRE(p->num_indices_to_process(p_state) == 1);
                        REQUIRE(p->scheduled(p_state, 0));
                        for (int i = 1; i < 10; ++i) {
                            REQUIRE_FALSE(p->scheduled(p_state, i));
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("BasicIndexingPropagator") {
    using namespace dwave::optimization;

    GIVEN("A dwopt graph with basic indexing") {
        Graph graph;
        auto i = graph.emplace_node<IntegerNode>(5, -3, 4);
        auto b = graph.emplace_node<BasicIndexingNode>(i, Slice(1, 4));

        // Lock the graph
        graph.topological_sort();

        // Construct the CP corresponding model
        AND_GIVEN("The CP Model") {
            CPModel model;

            // Add the variabbles to the model
            std::vector<CPVar*> vars;
            for (const auto& n_uptr : graph.nodes()) {
                const ArrayNode* ptr = dynamic_cast<const ArrayNode*>(n_uptr.get());
                REQUIRE(ptr);
                vars.push_back(model.emplace_variable<CPVar>(model, ptr, ptr->topological_index()));
            }

            Propagator* p = model.emplace_propagator<BasicIndexingPropagator>(
                    model.num_propagators(), vars[0], vars[1]);

            // build the advisor for the propagator p aimed to the variable for i
            Advisor advisor_i(p, 0, std::make_unique<BasicIndexingForwardTransform>(i, b));
            vars[0]->propagate_on_domain_change(std::move(advisor_i));

            // build the advisor for the propagator p aimed to the variable for b
            Advisor advisor_b(p, 1, std::make_unique<ElementWiseTransform>());
            vars[1]->propagate_on_domain_change(std::move(advisor_b));

            REQUIRE(vars[0]->on_domain.size() == 1);
            REQUIRE(vars[1]->on_domain.size() == 1);

            WHEN("We initialize the a state") {
                CPState state = model.initialize_state<Copier>();
                CPVarsState& s_state = state.get_variables_state();
                CPPropagatorsState& p_state = state.get_propagators_state();

                REQUIRE(s_state.size() == 2);
                REQUIRE(p_state.size() == 1);

                vars[0]->initialize_state(state);
                vars[1]->initialize_state(state);
                p->initialize_state(state);

                AND_WHEN("We alter the domain of the integer variable inside the slice") {
                    CPStatus status = vars[0]->assign(s_state, -2, 0);
                    REQUIRE(status == CPStatus::OK);
                    THEN("We see that the propagator is not triggered") {
                        // Not clear to me if it should be scheduled or not
                        // CHECK(not p_state[0]->scheduled());
                        CHECK(p_state[0]->indices_to_process().size() == 0);
                    }
                }

                AND_WHEN("We alter the domain of the integer variable inside the slice") {
                    CPStatus status = vars[0]->assign(s_state, -2, 3);
                    REQUIRE(status == CPStatus::OK);
                    THEN("We see that the propagator is triggered to run on the same index") {
                        REQUIRE(p_state[0]->scheduled());
                        REQUIRE(p_state[0]->indices_to_process().size() == 1);

                        CHECK(p_state[0]->scheduled(2));
                    }
                }
            }
        }
    }
}

}  // namespace dwave::optimization::cp

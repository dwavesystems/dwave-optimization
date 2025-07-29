// Copyright 2025 D-Wave
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

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "dwave-optimization/nodes/collections.hpp"
#include "dwave-optimization/nodes/constants.hpp"
#include "dwave-optimization/nodes/numbers.hpp"
#include "dwave-optimization/nodes/statistics.hpp"
#include "dwave-optimization/nodes/testing.hpp"

namespace dwave::optimization {

TEST_CASE("MeanNode") {
    auto graph = Graph();

    GIVEN("A non-empty constant node and a mean node") {
        auto a_ptr = graph.emplace_node<ConstantNode>(std::vector{-4.5, 3.2, 1.0, -2.7});
        auto mean_ptr = graph.emplace_node<MeanNode>(a_ptr);
        graph.emplace_node<ArrayValidationNode>(mean_ptr);

        WHEN("We initialize a state") {
            auto state = graph.initialize_state();

            THEN("The initial mean is correct") { CHECK(mean_ptr->view(state)[0] == -0.75); }
        }
    }
    GIVEN("An empty integer node with upper/lower bounds above zero and a mean node") {
        auto i_ptr = graph.emplace_node<IntegerNode>(0, 1, 10);
        auto mean_ptr = graph.emplace_node<MeanNode>(i_ptr);
        graph.emplace_node<ArrayValidationNode>(mean_ptr);

        WHEN("We initialize a state") {
            auto state = graph.initialize_state();

            // empty integer node and mean should be default value of 0.0
            THEN("The initial mean is correct") { CHECK(mean_ptr->view(state)[0] == 0.0); }
        }
    }
    GIVEN("A non-empty integer node and a mean node") {
        auto i_ptr = graph.emplace_node<IntegerNode>(3);
        auto mean_ptr = graph.emplace_node<MeanNode>(i_ptr);
        graph.emplace_node<ArrayValidationNode>(mean_ptr);

        WHEN("We initialize a state") {
            auto state = graph.empty_state();
            i_ptr->initialize_state(state, {6.0, 5.0, 1.0});
            graph.initialize_state(state);

            THEN("The initial mean is correct") { CHECK(mean_ptr->view(state)[0] == 4.0); }

            AND_WHEN("We make some changes to the integer node and propagate") {
                i_ptr->set_value(state, 1, 0.0);
                i_ptr->set_value(state, 2, 3.0);
                // i_ptr should now be [6.0, 0.0, 3.0]
                graph.propagate(state);

                THEN("The mean is correct") { CHECK(mean_ptr->view(state)[0] == 3.0); }
            }
            AND_WHEN("We make some changes to the integer node and propagate") {
                i_ptr->set_value(state, 1, 0.0);
                i_ptr->set_value(state, 2, 3.0);
                // i_ptr should now be [6.0, 0.0, 3.0]
                graph.propagate(state);

                THEN("The mean is correct") { CHECK(mean_ptr->view(state)[0] == 3.0); }
            }
        }
    }
    GIVEN("A dynamic set node and a mean node") {
        auto set_ptr = graph.emplace_node<SetNode>(9);
        auto mean_ptr = graph.emplace_node<MeanNode>(set_ptr);
        graph.emplace_node<ArrayValidationNode>(mean_ptr);

        WHEN("We initialize a state") {
            auto state = graph.initialize_state();

            THEN("The initial mean is correct") {
                // set node is empty
                CHECK(mean_ptr->view(state)[0] == 0.0);
            }

            AND_WHEN("We grow the set node and propagate") {
                set_ptr->assign(state, std::vector<double>{5, 8, 2, 6});
                graph.propagate(state);

                THEN("The mean is correct") { CHECK(mean_ptr->view(state)[0] == 5.25); }

                AND_WHEN("We commit, shrink the set node, and propagate") {
                    graph.commit(state);
                    set_ptr->shrink(state);
                    set_ptr->shrink(state);
                    // set should be [5, 8]
                    graph.propagate(state);

                    THEN("The mean is correct") { CHECK(mean_ptr->view(state)[0] == 6.5); }

                    AND_WHEN("We commit, shrink the set node to nothing, and propagate") {
                        graph.commit(state);
                        set_ptr->shrink(state);
                        set_ptr->shrink(state);
                        // set should empty
                        graph.propagate(state);

                        THEN("The mean is correct") {
                            // set node is empty
                            CHECK(mean_ptr->view(state)[0] == 0.0);
                        }
                    }
                }
            }
        }
    }
    GIVEN("A dynamic array with (min,max) = (1,5) and a mean node") {
        auto dyn_ptr = graph.emplace_node<DynamicArrayTestingNode>(
                std::initializer_list<ssize_t>{-1}, 1, 5, false);
        auto mean_ptr = graph.emplace_node<MeanNode>(dyn_ptr);
        graph.emplace_node<ArrayValidationNode>(mean_ptr);

        WHEN("We initialize a state") {
            auto state = graph.initialize_state();

            THEN("The initial mean is correct") {
                // set node is empty
                CHECK(mean_ptr->view(state)[0] == 0.0);
            }
        }
    }
    GIVEN("A dynamic array with (min,max) = (-10,-2) and a mean node") {
        auto dyn_ptr = graph.emplace_node<DynamicArrayTestingNode>(
                std::initializer_list<ssize_t>{-1}, -10, -2, false);
        auto mean_ptr = graph.emplace_node<MeanNode>(dyn_ptr);
        graph.emplace_node<ArrayValidationNode>(mean_ptr);

        WHEN("We initialize a state") {
            auto state = graph.initialize_state();

            THEN("The initial mean is correct") {
                // set node is empty
                CHECK(mean_ptr->view(state)[0] == 0.0);
            }
        }
    }
}
}  // namespace dwave::optimization

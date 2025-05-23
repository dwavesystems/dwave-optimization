// Copyright 2024 D-Wave
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
#include <dwave-optimization/graph.hpp>
#include <dwave-optimization/nodes/collections.hpp>
#include <dwave-optimization/nodes/constants.hpp>
#include <dwave-optimization/nodes/flow.hpp>
#include <dwave-optimization/nodes/inputs.hpp>
#include <dwave-optimization/nodes/mathematical.hpp>
#include <dwave-optimization/nodes/numbers.hpp>
#include <dwave-optimization/nodes/testing.hpp>

namespace dwave::optimization {

TEST_CASE("InputNode") {
    auto graph = Graph();

    // Ensure error thrown on bad limits
    CHECK_THROWS(graph.emplace_node<InputNode>(std::vector<ssize_t>{1}, 0, -0.5, false));

    GIVEN("An input node starting with state copied from a vector") {
        auto ptr = graph.emplace_node<InputNode>(std::vector<ssize_t>{4}, 10, 50, true);
        auto val = graph.emplace_node<ArrayValidationNode>(ptr);

        THEN("Graph.num_inputs() is correct") { CHECK(graph.num_inputs() == 1); }

        THEN("It copies the values into a 1d array") {
            CHECK(ptr->ndim() == 1);
            CHECK(ptr->size() == 4);
            CHECK(std::ranges::equal(ptr->shape(), std::vector{4}));
            CHECK(std::ranges::equal(ptr->strides(), std::vector{sizeof(double)}));
        }

        THEN("min/max/integral are set from arguments") {
            CHECK(ptr->min() == 10);
            CHECK(ptr->max() == 50);
            CHECK(ptr->integral());
        }

        THEN("initializing the graph state (without initializing the InputNode) throws an error") {
            CHECK_THROWS(graph.initialize_state());
        }

        AND_GIVEN("An initialized state") {
            std::vector<double> values = {30, 10, 40, 20};
            auto state = graph.empty_state();
            ptr->initialize_state(state, std::span{values});
            graph.initialize_state(state);

            THEN("The state defaults to the values from the vector") {
                CHECK(std::ranges::equal(ptr->view(state), values));
            }

            AND_WHEN("We assign new values and propagate") {
                std::vector<double> new_values = {20, 10, 49, 50};
                ptr->assign(state, new_values);

                ptr->propagate(state);
                val->propagate(state);

                THEN("The InputNode has the new values") {
                    CHECK(std::ranges::equal(ptr->view(state), new_values));
                }

                THEN("We can commit") {
                    ptr->commit(state);
                    val->commit(state);
                }

                THEN("We can revert") {
                    ptr->revert(state);
                    val->revert(state);
                }
            }

            AND_WHEN("We assign invalid values we get an exception") {
                std::vector<double> new_values = {20, 10, 49, 51};
                CHECK_THROWS(ptr->assign(state, new_values));

                new_values = {9, 9, 9, 9};
                CHECK_THROWS(ptr->assign(state, new_values));

                new_values = {9.99, 50.01, 25, 25};
                CHECK_THROWS(ptr->assign(state, new_values));

                new_values = {20, 20, 20};
                CHECK_THROWS(ptr->assign(state, new_values));

                new_values = {20, 20, 20, 20, 20};
                CHECK_THROWS(ptr->assign(state, new_values));
            }
        }

        AND_GIVEN("Another node, and another input") {
            graph.emplace_node<IntegerNode>();
            graph.emplace_node<InputNode>(std::vector<ssize_t>{1}, 10, 50, false);

            THEN("Graph.num_inputs() is correct") { CHECK(graph.num_inputs() == 2); }
        }
    }
}

}  // namespace dwave::optimization

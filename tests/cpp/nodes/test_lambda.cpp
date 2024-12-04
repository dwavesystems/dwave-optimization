// Copyright 2024 D-Wave Inc.
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
#include <dwave-optimization/nodes/lambda.hpp>
#include <dwave-optimization/nodes/mathematical.hpp>
#include <dwave-optimization/nodes/numbers.hpp>
#include <dwave-optimization/nodes/testing.hpp>

namespace dwave::optimization {

TEST_CASE("InputNode") {
    auto graph = Graph();

    GIVEN("An input node starting with state copied from a vector") {
        auto ptr = graph.emplace_node<InputNode>(std::vector<ssize_t>{4}, 10, 50, true);
        auto val = graph.emplace_node<ArrayValidationNode>(ptr);

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
    }
}

TEST_CASE("NaryReduceNode") {
    auto graph = Graph();

    GIVEN("A vector<Node*> of constants and an expression") {
        std::vector<double> i = {0, 1, 2, 2};
        std::vector<double> j = {1, 2, 4, 3};

        auto args = std::vector<ArrayNode*>{graph.emplace_node<ConstantNode>(i),
                                            graph.emplace_node<ConstantNode>(j)};

        // x0 * x1 + x2
        auto expression = Graph();
        std::vector<InputNode*> inputs = {expression.emplace_node<InputNode>(),
                                          expression.emplace_node<InputNode>(),
                                          expression.emplace_node<InputNode>()};
        auto output_ptr = expression.emplace_node<AddNode>(
                expression.emplace_node<MultiplyNode>(inputs[0], inputs[1]), inputs[2]);
        expression.topological_sort();

        THEN("We can create an accumulate node") {
            std::vector<double> initial_values({1, 2, 3});
            auto reduce_ptr = graph.emplace_node<NaryReduceNode>(std::move(expression), inputs,
                                                                 output_ptr, initial_values, args);

            AND_WHEN("We initialize a state") {
                auto state = graph.initialize_state();

                THEN("The state is correct") {
                    CHECK(std::ranges::equal(reduce_ptr->view(state), std::vector{5, 7, 15, 21}));
                }
            }
        }
    }

    GIVEN("Two integer nodes and a more complicated expression") {
        auto i_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{5}, -10, 10);
        auto j_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{5}, -10, 10);

        // (x0 + 1) * x1 - x2 + 5
        auto expression = Graph();
        std::vector<InputNode*> inputs = {expression.emplace_node<InputNode>(),
                                          expression.emplace_node<InputNode>(),
                                          expression.emplace_node<InputNode>()};
        auto output_ptr = expression.emplace_node<AddNode>(
                expression.emplace_node<SubtractNode>(
                        expression.emplace_node<MultiplyNode>(
                                expression.emplace_node<AddNode>(
                                        inputs[0], expression.emplace_node<ConstantNode>(1)),
                                inputs[1]),
                        inputs[2]),
                expression.emplace_node<ConstantNode>(5));
        expression.topological_sort();

        THEN("We can create a lambda node") {
            std::vector<ArrayNode*> args({i_ptr, j_ptr});

            std::vector<double> initial_values({1, 2, 3});
            auto reduce_ptr = graph.emplace_node<NaryReduceNode>(std::move(expression), inputs,
                                                                 output_ptr, initial_values, args);

            auto validation_ptr = graph.emplace_node<ArrayValidationNode>(reduce_ptr);

            AND_WHEN("We initialize a state") {
                auto state = graph.initialize_state();

                THEN("The state is correct") {
                    CHECK(std::ranges::equal(reduce_ptr->view(state),
                                             std::vector{-1, 6, -1, 6, -1}));
                }

                AND_WHEN("We mutate the integers and propagate") {
                    i_ptr->set_value(state, 4, 4);
                    j_ptr->set_value(state, 4, 5);

                    i_ptr->set_value(state, 0, 3);
                    j_ptr->set_value(state, 0, -1);

                    j_ptr->set_value(state, 3, 7);

                    i_ptr->propagate(state);  // [3, 0, 0, 0, 4]
                    j_ptr->propagate(state);  // [-1, 0, 0, 7, 5]
                    reduce_ptr->propagate(state);
                    validation_ptr->propagate(state);

                    THEN("The state is correct") {
                        CHECK(std::ranges::equal(reduce_ptr->view(state),
                                                 std::vector{-5, 10, -5, 17, 13}));
                    }
                }
            }
        }
    }

    GIVEN("Three integer nodes and an expression") {
        auto i_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{5}, 0, 100);
        auto j_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{5}, 0, 100);

        // max(x0 + x2, x1)
        auto expression = Graph();
        std::vector<InputNode*> inputs = {expression.emplace_node<InputNode>(),
                                          expression.emplace_node<InputNode>(),
                                          expression.emplace_node<InputNode>()};
        auto output_ptr = expression.emplace_node<MaximumNode>(
                expression.emplace_node<AddNode>(inputs[0], inputs[2]), inputs[1]);
        expression.topological_sort();

        THEN("We can create a lambda node with basic functions and logic control") {
            std::vector<ArrayNode*> args({i_ptr, j_ptr});

            std::vector<double> initial_values({0, 0, 0});
            auto reduce_ptr = graph.emplace_node<NaryReduceNode>(std::move(expression), inputs,
                                                                 output_ptr, initial_values, args);

            auto validation_ptr = graph.emplace_node<ArrayValidationNode>(reduce_ptr);

            AND_WHEN("We initialize a state") {
                auto state = graph.empty_state();
                i_ptr->initialize_state(state, {0, 1, 2, 3, 4});
                j_ptr->initialize_state(state, {10, 10, 20, 30, 32});

                graph.initialize_state(state);

                THEN("The state is correct") {
                    CHECK(std::ranges::equal(reduce_ptr->view(state),
                                             std::vector{10, 11, 20, 30, 34}));
                }

                AND_WHEN("We mutate the integers and propagate") {
                    i_ptr->set_value(state, 4, 5);

                    j_ptr->set_value(state, 1, 15);
                    j_ptr->set_value(state, 2, 15);

                    i_ptr->propagate(state);  // [0, 1, 2, 3, 5]
                    j_ptr->propagate(state);  // [10, 15, 15, 30, 32]
                    reduce_ptr->propagate(state);
                    validation_ptr->propagate(state);

                    THEN("The state is correct") {
                        CHECK(std::ranges::equal(reduce_ptr->view(state),
                                                 std::vector{10, 15, 17, 30, 35}));
                    }
                }
            }
        }
    }

    GIVEN("A constant node") {
        std::vector<double> i = {0, 1, 2, 2};

        std::vector<double> initial_values({1});
        auto args = std::vector<ArrayNode*>{graph.emplace_node<ConstantNode>(i)};

        THEN("We can't create a NaryReduceNode with an expression with decision variables") {
            auto expression = Graph();
            std::vector<InputNode*> inputs = {
                    expression.emplace_node<InputNode>(),
            };
            auto output_ptr = expression.emplace_node<AddNode>(
                    inputs[0], expression.emplace_node<IntegerNode>());
            expression.topological_sort();

            CHECK_THROWS(graph.emplace_node<NaryReduceNode>(std::move(expression), inputs,
                                                            output_ptr, initial_values, args));
        }

        THEN("We can't create a NaryReduceNode with non-scalar nodes") {
            auto expression = Graph();
            std::vector<InputNode*> inputs = {
                    expression.emplace_node<InputNode>(std::vector<ssize_t>{2}, 0, 1, false),
            };
            auto output_ptr = expression.emplace_node<AddNode>(
                    inputs[0],
                    expression.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{2}));
            expression.topological_sort();

            CHECK_THROWS(graph.emplace_node<NaryReduceNode>(std::move(expression), inputs,
                                                            output_ptr, initial_values, args));
        }

        // TODO: num initial values
        // TODO: num args
        // TODO: unsupported nodes
    }
}

}  // namespace dwave::optimization

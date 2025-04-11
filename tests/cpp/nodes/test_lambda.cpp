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
#include <dwave-optimization/nodes/inputs.hpp>
#include <dwave-optimization/nodes/lambda.hpp>
#include <dwave-optimization/nodes/mathematical.hpp>
#include <dwave-optimization/nodes/numbers.hpp>
#include <dwave-optimization/nodes/testing.hpp>

namespace dwave::optimization {

TEST_CASE("NaryReduceNode") {
    auto graph = Graph();

    GIVEN("A vector<Node*> of constants and an expression") {
        std::vector<double> i = {0, 1, 2, 2};
        std::vector<double> j = {1, 2, 4, 3};

        auto args = std::vector<ArrayNode*>{graph.emplace_node<ConstantNode>(i),
                                            graph.emplace_node<ConstantNode>(j)};

        // x0 * x1 + x2
        auto expression = Graph();
        std::vector<InputNode*> inputs = {expression.emplace_node<InputNode>(InputNode::unbounded_scalar()),
                                          expression.emplace_node<InputNode>(InputNode::unbounded_scalar()),
                                          expression.emplace_node<InputNode>(InputNode::unbounded_scalar())};
        auto output_ptr = expression.emplace_node<AddNode>(
                expression.emplace_node<MultiplyNode>(inputs[1], inputs[2]), inputs[0]);
        expression.set_objective(output_ptr);
        expression.topological_sort();

        std::cout << output_ptr->min() << " - " << output_ptr->max() << "\n";

        THEN("We can create an accumulate node") {
            auto reduce_ptr = graph.emplace_node<NaryReduceNode>(std::move(expression), args, 5);

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
        std::vector<InputNode*> inputs = {expression.emplace_node<InputNode>(InputNode::unbounded_scalar()),
                                          expression.emplace_node<InputNode>(InputNode::unbounded_scalar()),
                                          expression.emplace_node<InputNode>(InputNode::unbounded_scalar())};
        auto output_ptr = expression.emplace_node<AddNode>(
                expression.emplace_node<SubtractNode>(
                        expression.emplace_node<MultiplyNode>(
                                expression.emplace_node<AddNode>(
                                        inputs[1], expression.emplace_node<ConstantNode>(1)),
                                inputs[2]),
                        inputs[0]),
                expression.emplace_node<ConstantNode>(5));
        expression.set_objective(output_ptr);
        expression.topological_sort();

        THEN("We can create a lambda node") {
            std::vector<ArrayNode*> args({i_ptr, j_ptr});

            auto reduce_ptr = graph.emplace_node<NaryReduceNode>(std::move(expression), args, 6);

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
        std::vector<InputNode*> inputs = {expression.emplace_node<InputNode>(InputNode::unbounded_scalar()),
                                          expression.emplace_node<InputNode>(InputNode::unbounded_scalar()),
                                          expression.emplace_node<InputNode>(InputNode::unbounded_scalar())};
        auto output_ptr = expression.emplace_node<MaximumNode>(
                expression.emplace_node<AddNode>(inputs[1], inputs[0]), inputs[2]);
        expression.set_objective(output_ptr);
        expression.topological_sort();

        THEN("We can create a lambda node with basic functions and logic control") {
            std::vector<ArrayNode*> args({i_ptr, j_ptr});

            auto reduce_ptr = graph.emplace_node<NaryReduceNode>(std::move(expression), args, 0);

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

    GIVEN("A non-integral constant node") {
        std::vector<double> i = {0, 1, 2, 2.5};

        auto args = std::vector<ArrayNode*>{graph.emplace_node<ConstantNode>(i)};

        THEN("We can't create a NaryReduceNode with an expression with decision variables") {
            auto expression = Graph();
            std::vector<InputNode*> inputs = {
                    expression.emplace_node<InputNode>(InputNode::unbounded_scalar()),
                    expression.emplace_node<InputNode>(InputNode::unbounded_scalar()),
            };
            expression.set_objective(expression.emplace_node<AddNode>(
                    inputs[0], expression.emplace_node<IntegerNode>()));
            expression.topological_sort();

            CHECK_THROWS(graph.emplace_node<NaryReduceNode>(std::move(expression), args, 0));
        }

        THEN("We can't create a NaryReduceNode with non-scalar nodes") {
            auto expression = Graph();
            std::vector<InputNode*> inputs = {
                    expression.emplace_node<InputNode>(std::vector<ssize_t>{2}, 0, 10, false),
                    expression.emplace_node<InputNode>(std::vector<ssize_t>{2}, 0, 10, false),
            };
            expression.set_objective(expression.emplace_node<MinNode>(
                    expression.emplace_node<AddNode>(inputs[0], inputs[1])));
            expression.topological_sort();

            CHECK_THROWS(graph.emplace_node<NaryReduceNode>(std::move(expression), args, 0));
        }

        THEN("We can't create a NaryReduceNode with an expression that has inputs with a smaller "
             "domain") {
            auto expression = Graph();
            std::vector<InputNode*> inputs = {
                    expression.emplace_node<InputNode>(std::vector<ssize_t>{}, 0, 1, false),
                    expression.emplace_node<InputNode>(std::vector<ssize_t>{}, 0, 1, false),
            };
            expression.set_objective(expression.emplace_node<AddNode>(inputs[0], inputs[1]));
            expression.topological_sort();
            CHECK_THROWS(graph.emplace_node<NaryReduceNode>(std::move(expression), args, 0));
        }

        THEN("We can't create a NaryReduceNode with an expression that has inputs with an integral "
             "domain") {
            auto expression = Graph();
            std::vector<InputNode*> inputs = {
                    expression.emplace_node<InputNode>(std::vector<ssize_t>{}, 0, 10, true),
                    expression.emplace_node<InputNode>(std::vector<ssize_t>{}, 0, 10, true),
            };
            expression.set_objective(expression.emplace_node<AddNode>(inputs[0], inputs[1]));
            expression.topological_sort();
            CHECK_THROWS(graph.emplace_node<NaryReduceNode>(std::move(expression), args, 0));
        }
    }

    GIVEN("An integral constant node") {
        std::vector<double> i = {0, 1, 2};

        auto args = std::vector<ArrayNode*>{graph.emplace_node<ConstantNode>(i)};

        THEN("We can't create a NaryReduceNode with an expression that has inputs with an integral "
             "domain, and output that is non-integral") {
            auto expression = Graph();
            std::vector<InputNode*> inputs = {
                    expression.emplace_node<InputNode>(std::vector<ssize_t>{}, 0, 10, true),
                    expression.emplace_node<InputNode>(std::vector<ssize_t>{}, 0, 10, true),
            };
            expression.set_objective(expression.emplace_node<AddNode>(
                    inputs[0], expression.emplace_node<MultiplyNode>(
                                       expression.emplace_node<ConstantNode>(0.5), inputs[1])));
            expression.topological_sort();
            CHECK_THROWS(graph.emplace_node<NaryReduceNode>(std::move(expression), args, 0));
        }
    }
}

}  // namespace dwave::optimization

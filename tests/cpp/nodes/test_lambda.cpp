// Copyright 2025 D-Wave Inc.
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
#include <dwave-optimization/nodes/numbers.hpp>
#include <dwave-optimization/nodes/reduce.hpp>
#include <dwave-optimization/nodes/testing.hpp>

#include "catch2/matchers/catch_matchers_all.hpp"

namespace dwave::optimization {

using Catch::Matchers::RangeEquals;

TEST_CASE("AccumulateZipNode") {
    auto graph = Graph();

    GIVEN("An zero arg expression") {
        auto expression = Graph();
        expression.set_objective(expression.emplace_node<ConstantNode>(std::vector{1.0}));
        expression.topological_sort();

        THEN("We get an exception trying to use it with an AccumulateZipNode") {
            CHECK_THROWS_AS(
                    AccumulateZipNode(std::move(expression), std::vector<ArrayNode*>{}, 0.0),
                    std::invalid_argument);
        }
    }

    GIVEN("Two constant vector nodes and an expression") {
        std::vector<double> i = {0, 1, 2, 2};
        std::vector<double> j = {1, 2, 4, 3};

        auto args = std::vector<ArrayNode*>{graph.emplace_node<ConstantNode>(i),
                                            graph.emplace_node<ConstantNode>(j)};

        // x0 * x1 + x2
        auto expression = Graph();
        std::vector<InputNode*> inputs = {
                expression.emplace_node<InputNode>(InputNode::unbounded_scalar()),
                expression.emplace_node<InputNode>(InputNode::unbounded_scalar()),
                expression.emplace_node<InputNode>(InputNode::unbounded_scalar())};
        auto output_ptr = expression.emplace_node<AddNode>(
                expression.emplace_node<MultiplyNode>(inputs[1], inputs[2]), inputs[0]);
        expression.set_objective(output_ptr);
        expression.topological_sort();

        THEN("We can create a accumulate node") {
            auto accumulate_ptr = graph.emplace_node<AccumulateZipNode>(std::move(expression), args, 5.0);

            AND_WHEN("We initialize a state") {
                auto state = graph.initialize_state();

                THEN("The state is correct") {
                    CHECK(std::ranges::equal(accumulate_ptr->view(state), std::vector{5, 7, 15, 21}));
                }
            }
        }

        AND_GIVEN("A scalar node") {
            IntegerNode* initial =
                    graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{}, -10, 10);

            THEN("We can create a accumulate node with a non-constant initial value") {
                auto accumulate_ptr =
                        graph.emplace_node<AccumulateZipNode>(std::move(expression), args, initial);

                AND_WHEN("We initialize a state") {
                    auto state = graph.empty_state();
                    initial->initialize_state(state, {5.0});
                    graph.initialize_state(state);

                    THEN("The state is correct") {
                        CHECK(std::ranges::equal(accumulate_ptr->view(state),
                                                 std::vector{5, 7, 15, 21}));
                    }

                    AND_WHEN("We mutate the initial value and propagate") {
                        initial->set_value(state, 0, -3.0);
                        graph.propagate(state);

                        THEN("The state is correct") {
                            CHECK(std::ranges::equal(accumulate_ptr->view(state),
                                                     std::vector{-3.0, -1.0, 7.0, 13.0}));
                        }
                    }
                }
            }
        }
    }

    GIVEN("Two integer nodes and a more complicated expression") {
        auto i_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{5}, -10, 10);
        auto j_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{5}, -10, 10);

        // (x0 + 1) * x1 - x2 + 5
        auto expression = Graph();
        std::vector<InputNode*> inputs = {
                expression.emplace_node<InputNode>(InputNode::unbounded_scalar()),
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

            auto accumulate_ptr = graph.emplace_node<AccumulateZipNode>(std::move(expression), args, 6.0);

            auto validation_ptr = graph.emplace_node<ArrayValidationNode>(accumulate_ptr);

            AND_WHEN("We initialize a state") {
                auto state = graph.initialize_state();

                THEN("The state is correct") {
                    CHECK(std::ranges::equal(accumulate_ptr->view(state),
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
                    accumulate_ptr->propagate(state);
                    validation_ptr->propagate(state);

                    THEN("The state is correct") {
                        CHECK(std::ranges::equal(accumulate_ptr->view(state),
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
        std::vector<InputNode*> inputs = {
                expression.emplace_node<InputNode>(InputNode::unbounded_scalar()),
                expression.emplace_node<InputNode>(InputNode::unbounded_scalar()),
                expression.emplace_node<InputNode>(InputNode::unbounded_scalar())};
        auto output_ptr = expression.emplace_node<MaximumNode>(
                expression.emplace_node<AddNode>(inputs[1], inputs[0]), inputs[2]);
        expression.set_objective(output_ptr);
        expression.topological_sort();

        THEN("We can create a lambda node with basic functions and logic control") {
            std::vector<ArrayNode*> args({i_ptr, j_ptr});

            auto accumulate_ptr = graph.emplace_node<AccumulateZipNode>(std::move(expression), args, 0.0);

            auto validation_ptr = graph.emplace_node<ArrayValidationNode>(accumulate_ptr);

            AND_WHEN("We initialize a state") {
                auto state = graph.empty_state();
                i_ptr->initialize_state(state, {0, 1, 2, 3, 4});
                j_ptr->initialize_state(state, {10, 10, 20, 30, 32});

                graph.initialize_state(state);

                THEN("The state is correct") {
                    CHECK(std::ranges::equal(accumulate_ptr->view(state),
                                             std::vector{10, 11, 20, 30, 34}));
                }

                AND_WHEN("We mutate the integers and propagate") {
                    i_ptr->set_value(state, 4, 5);

                    j_ptr->set_value(state, 1, 15);
                    j_ptr->set_value(state, 2, 15);

                    i_ptr->propagate(state);  // [0, 1, 2, 3, 5]
                    j_ptr->propagate(state);  // [10, 15, 15, 30, 32]
                    accumulate_ptr->propagate(state);
                    validation_ptr->propagate(state);

                    THEN("The state is correct") {
                        CHECK(std::ranges::equal(accumulate_ptr->view(state),
                                                 std::vector{10, 15, 17, 30, 35}));
                    }
                }
            }
        }
    }

    GIVEN("A non-integral constant node") {
        std::vector<double> i = {0, 1, 2, 2.5};

        auto args = std::vector<ArrayNode*>{graph.emplace_node<ConstantNode>(i)};

        THEN("We can't create a AccumulateZipNode with an expression with decision variables") {
            auto expression = Graph();
            std::vector<InputNode*> inputs = {
                    expression.emplace_node<InputNode>(InputNode::unbounded_scalar()),
                    expression.emplace_node<InputNode>(InputNode::unbounded_scalar()),
            };
            expression.set_objective(expression.emplace_node<AddNode>(
                    inputs[0], expression.emplace_node<IntegerNode>()));
            expression.topological_sort();

            CHECK_THROWS(graph.emplace_node<AccumulateZipNode>(std::move(expression), args, 0.0));
        }

        THEN("We can't create a AccumulateZipNode with non-scalar nodes") {
            auto expression = Graph();
            std::vector<InputNode*> inputs = {
                    expression.emplace_node<InputNode>(std::vector<ssize_t>{2}, 0, 10, false),
                    expression.emplace_node<InputNode>(std::vector<ssize_t>{2}, 0, 10, false),
            };
            expression.set_objective(expression.emplace_node<MinNode>(
                    expression.emplace_node<AddNode>(inputs[0], inputs[1])));
            expression.topological_sort();

            CHECK_THROWS(graph.emplace_node<AccumulateZipNode>(std::move(expression), args, 0.0));
        }

        THEN("We can't create a AccumulateZipNode with an expression that has inputs with a smaller "
             "domain") {
            auto expression = Graph();
            std::vector<InputNode*> inputs = {
                    expression.emplace_node<InputNode>(std::vector<ssize_t>{}, 0, 1, false),
                    expression.emplace_node<InputNode>(std::vector<ssize_t>{}, 0, 1, false),
            };
            expression.set_objective(expression.emplace_node<AddNode>(inputs[0], inputs[1]));
            expression.topological_sort();
            CHECK_THROWS(graph.emplace_node<AccumulateZipNode>(std::move(expression), args, 0.0));
        }

        THEN("We can't create a AccumulateZipNode with an expression that has inputs with an integral "
             "domain") {
            auto expression = Graph();
            std::vector<InputNode*> inputs = {
                    expression.emplace_node<InputNode>(std::vector<ssize_t>{}, 0, 10, true),
                    expression.emplace_node<InputNode>(std::vector<ssize_t>{}, 0, 10, true),
            };
            expression.set_objective(expression.emplace_node<AddNode>(inputs[0], inputs[1]));
            expression.topological_sort();
            CHECK_THROWS(graph.emplace_node<AccumulateZipNode>(std::move(expression), args, 0.0));
        }

        THEN("We can't create a AccumulateZipNode with an initial value larger than allowed by the "
             "first input") {
            auto expression = std::make_shared<Graph>();
            std::vector<InputNode*> inputs = {
                    expression->emplace_node<InputNode>(std::vector<ssize_t>{}, -10, 10, false),
                    expression->emplace_node<InputNode>(std::vector<ssize_t>{}, -10, 10, false),
            };
            expression->set_objective(inputs[0]);
            expression->topological_sort();

            CHECK_THROWS(graph.emplace_node<AccumulateZipNode>(expression, args, -11.0));
            CHECK_THROWS(graph.emplace_node<AccumulateZipNode>(expression, args, +11.0));
            CHECK_THROWS(graph.emplace_node<AccumulateZipNode>(
                    expression, args,
                    graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -11, 10)));
            CHECK_THROWS(graph.emplace_node<AccumulateZipNode>(
                    expression, args,
                    graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -10, 11)));
        }
    }

    GIVEN("An integral constant node") {
        std::vector<double> i = {0, 1, 2};

        auto args = std::vector<ArrayNode*>{graph.emplace_node<ConstantNode>(i)};

        THEN("We can't create a AccumulateZipNode with an expression that has inputs with an integral "
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
            CHECK_THROWS(graph.emplace_node<AccumulateZipNode>(std::move(expression), args, 0.0));
        }
    }

    GIVEN("A set node and an expression") {
        auto set_ptr = graph.emplace_node<SetNode>(10);

        auto expression = Graph();
        std::vector<InputNode*> inputs = {
                expression.emplace_node<InputNode>(InputNode::unbounded_scalar()),
                expression.emplace_node<InputNode>(InputNode::unbounded_scalar()),
        };
        expression.set_objective(expression.emplace_node<AddNode>(inputs[0], inputs[1]));
        expression.topological_sort();

        THEN("We create and initialize a AccumulateZipNode") {
            auto accumulatezip_ptr = graph.emplace_node<AccumulateZipNode>(
                    std::move(expression), std::vector<ArrayNode*>{set_ptr}, 0.0);

            auto state = graph.initialize_state();

            AND_WHEN("We modify the set variable and propagate") {
                set_ptr->assign(state, {1, 3, 8});
                graph.propose(state, {set_ptr});

                THEN("The AccumulateZipNode has the correct size and state") {
                    REQUIRE(accumulatezip_ptr->size(state) == 3);
                    CHECK_THAT(accumulatezip_ptr->view(state), RangeEquals({1, 4, 12}));
                }

                AND_WHEN("We modify the set variable, propagate and revert") {
                    set_ptr->assign(state, {3, 6});
                    REQUIRE_THAT(set_ptr->view(state), RangeEquals({3, 6}));

                    set_ptr->propagate(state);
                    accumulatezip_ptr->propagate(state);

                    REQUIRE(accumulatezip_ptr->size(state) == 2);
                    REQUIRE_THAT(accumulatezip_ptr->view(state), RangeEquals({3, 9}));

                    set_ptr->revert(state);
                    accumulatezip_ptr->revert(state);

                    THEN("The AccumulateZipNode has the correct size and state") {
                        REQUIRE(accumulatezip_ptr->size(state) == 3);
                        CHECK_THAT(accumulatezip_ptr->view(state), RangeEquals({1, 4, 12}));
                    }
                }
            }
        }
    }
}

}  // namespace dwave::optimization

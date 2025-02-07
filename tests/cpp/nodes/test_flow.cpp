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
#include <catch2/matchers/catch_matchers_all.hpp>
#include <dwave-optimization/graph.hpp>
#include <dwave-optimization/nodes/collections.hpp>
#include <dwave-optimization/nodes/flow.hpp>
#include <dwave-optimization/nodes/numbers.hpp>
#include <dwave-optimization/nodes/testing.hpp>

using Catch::Matchers::RangeEquals;

namespace dwave::optimization {

TEST_CASE("WhereNode") {
    auto graph = Graph();

    GIVEN("Three integer scalars, `condition`,`x`,`y`") {
        auto condition_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, 0, 5);
        auto x_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -10, 10);
        auto y_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, 0, 3);

        auto where_ptr = graph.emplace_node<WhereNode>(condition_ptr, x_ptr, y_ptr);

        THEN("The `where` node has the shape we expect") {
            CHECK(where_ptr->ndim() == 0);
            CHECK(where_ptr->size() == 1);
        }

        THEN("The `where` node has the min/max etc that we expect") {
            CHECK(where_ptr->max() == 10);
            CHECK(where_ptr->min() == -10);
            CHECK(where_ptr->integral());
        }

        WHEN("We default-initialize") {
            auto state = graph.initialize_state();

            THEN("where has the state we expect") {
                double condition = condition_ptr->view(state)[0];
                double x = x_ptr->view(state)[0];
                double y = y_ptr->view(state)[0];
                double where = where_ptr->view(state)[0];

                CHECK((condition ? x : y) == where);
            }
        }

        WHEN("We set the initial value to select x") {
            auto state = graph.empty_state();
            condition_ptr->initialize_state(state, {1});
            x_ptr->initialize_state(state, {3});
            y_ptr->initialize_state(state, {2});
            graph.initialize_state(state);

            THEN("`where` has the state we expect") {
                double condition = condition_ptr->view(state)[0];
                double x = x_ptr->view(state)[0];
                double y = y_ptr->view(state)[0];
                double where = where_ptr->view(state)[0];

                CHECK((condition ? x : y) == where);
            }
        }

        WHEN("We set the initial value to select y") {
            auto state = graph.empty_state();
            condition_ptr->initialize_state(state, {0});
            x_ptr->initialize_state(state, {3});
            y_ptr->initialize_state(state, {2});
            graph.initialize_state(state);

            THEN("`where` has the state we expect") {
                double condition = condition_ptr->view(state)[0];
                double x = x_ptr->view(state)[0];
                double y = y_ptr->view(state)[0];
                double where = where_ptr->view(state)[0];

                CHECK((condition ? x : y) == where);
            }

            AND_WHEN("We update `condition`") {
                condition_ptr->set_value(state, 0, 1);  // 0->1
                graph.propagate(state, graph.descendants(state, {condition_ptr}));

                THEN("where has the state we expect") {
                    double condition = condition_ptr->view(state)[0];
                    double x = x_ptr->view(state)[0];
                    double y = y_ptr->view(state)[0];
                    double where = where_ptr->view(state)[0];

                    CHECK((condition ? x : y) == where);
                    CHECK(where == 3);
                }

                AND_WHEN("We commit") {
                    graph.commit(state, graph.descendants(state, {condition_ptr}));

                    THEN("where has the state we expect") {
                        double condition = condition_ptr->view(state)[0];
                        double x = x_ptr->view(state)[0];
                        double y = y_ptr->view(state)[0];
                        double where = where_ptr->view(state)[0];

                        CHECK((condition ? x : y) == where);
                        CHECK(where == 3);
                    }
                }

                AND_WHEN("We revert") {
                    graph.revert(state, graph.descendants(state, {condition_ptr}));

                    THEN("where has the state we expect") {
                        double condition = condition_ptr->view(state)[0];
                        double x = x_ptr->view(state)[0];
                        double y = y_ptr->view(state)[0];
                        double where = where_ptr->view(state)[0];

                        CHECK((condition ? x : y) == where);
                        CHECK(where == 2);
                    }
                }
            }
        }
    }

    GIVEN("Two dynamic nodes and a 1d array to switch between them") {
        auto condition_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{1}, 0, 5);
        auto x_ptr = graph.emplace_node<SetNode>(5);
        auto y_ptr = graph.emplace_node<SetNode>(3);
        auto where_ptr = graph.emplace_node<WhereNode>(condition_ptr, x_ptr, y_ptr);

        // this node will be testing consistency/shape/etc
        graph.emplace_node<ArrayValidationNode>(where_ptr);

        // initialize the state to an interesting starting point
        auto state = graph.empty_state();
        condition_ptr->initialize_state(state, {0});
        x_ptr->initialize_state(state, {0});
        y_ptr->initialize_state(state, {0, 1});
        graph.initialize_state(state);

        WHEN("We set the condition value to 1 and then propagate and accept") {
            condition_ptr->set_value(state, 0, 1);
            graph.propose(state, {condition_ptr});

            THEN("The `where` output is x") {
                CHECK(std::ranges::equal(where_ptr->view(state), x_ptr->view(state)));
            }

            AND_WHEN("We set the condition value to 0 and then propagate and accept") {
                condition_ptr->set_value(state, 0, 0);
                graph.propose(state, {condition_ptr});

                THEN("The `where` output is y") {
                    CHECK(std::ranges::equal(where_ptr->view(state), y_ptr->view(state)));
                }
            }

            AND_WHEN("We change x's value") {
                x_ptr->grow(state);
                graph.propose(state, {x_ptr});

                THEN("The `where` output is the updated x") {
                    CHECK(std::ranges::equal(where_ptr->view(state), x_ptr->view(state)));
                }

                x_ptr->exchange(state, 0, 1);
                x_ptr->shrink(state);
                x_ptr->shrink(state);
                x_ptr->grow(state);
                graph.propose(state, {x_ptr});

                THEN("The `where` output is the updated x") {
                    CHECK(std::ranges::equal(where_ptr->view(state), x_ptr->view(state)));
                }
            }

            AND_WHEN("We change y's value") {
                y_ptr->grow(state);
                graph.propose(state, {y_ptr});

                THEN("The `where` output is x") {
                    CHECK(std::ranges::equal(where_ptr->view(state), x_ptr->view(state)));
                }
            }
        }
    }

    GIVEN("Three array nodes and a where node") {
        auto condition_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{2, 2}, 0, 5);
        auto x_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{2, 2}, 0, 5);
        auto y_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{2, 2}, 0, 5);
        auto where_ptr = graph.emplace_node<WhereNode>(condition_ptr, x_ptr, y_ptr);

        // this node will be testing consistency/shape/etc
        graph.emplace_node<ArrayValidationNode>(where_ptr);

        // initialize the state to an interesting starting point
        auto state = graph.empty_state();
        condition_ptr->initialize_state(state, {0, 1, 0, 2});
        x_ptr->initialize_state(state, {1, 2, 3, 4});
        y_ptr->initialize_state(state, {1, 1, 1, 1});
        graph.initialize_state(state);

        THEN("`where` has the expected output") {
            CHECK_THAT(where_ptr->view(state), RangeEquals({1, 2, 1, 4}));
        }

        WHEN("We update `condition`") {
            condition_ptr->set_value(state, 0, 4);
            condition_ptr->set_value(state, 3, 0);
            graph.propose(state, {condition_ptr});

            THEN("`where` has the expected output") {
                CHECK_THAT(where_ptr->view(state), RangeEquals({1, 2, 1, 1}));
            }
        }

        WHEN("We do a bunch of updates") {
            condition_ptr->set_value(state, 0, 4);
            x_ptr->set_value(state, 0, 4);
            y_ptr->set_value(state, 0, 3);
            x_ptr->set_value(state, 1, 3);
            x_ptr->set_value(state, 1, 3);  // deliberate duplicate
            y_ptr->set_value(state, 1, 2);
            y_ptr->set_value(state, 2, 2);
            graph.propose(state, {condition_ptr, x_ptr, y_ptr});

            THEN("`where` has the expected output") {
                // condition = [4, 1, 0, 2]
                // x = [4, 3, 3, 4]
                // y = [3, 2, 2, 1]
                CHECK_THAT(where_ptr->view(state), RangeEquals({4, 3, 2, 4}));
            }
        }
    }
}

}  // namespace dwave::optimization

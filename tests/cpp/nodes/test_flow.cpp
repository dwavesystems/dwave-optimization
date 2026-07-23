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

#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/nodes/collections.hpp"
#include "dwave-optimization/nodes/flow.hpp"
#include "dwave-optimization/nodes/indexing.hpp"
#include "dwave-optimization/nodes/numbers.hpp"
#include "dwave-optimization/nodes/testing.hpp"

using Catch::Matchers::RangeEquals;

namespace dwave::optimization {

TEST_CASE("ExtractNode") {
    auto graph = Graph();

    GIVEN("Two integer scalars, `condition` and `x`") {
        auto condition_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, 0, 5);
        auto arr_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -10, 10);

        auto extract_ptr = graph.emplace_node<ExtractNode>(condition_ptr, arr_ptr);

        THEN("The `extract` node has the shape we expect") {
            CHECK(extract_ptr->ndim() == 1);
            CHECK(extract_ptr->dynamic());
        }

        THEN("The `extract` node has the min/max etc that we expect") {
            CHECK(extract_ptr->max() == 10);
            CHECK(extract_ptr->min() == -10);
            CHECK(extract_ptr->integral());
        }

        WHEN("We default-initialize") {
            auto state = graph.initialize_state();

            THEN("extract has the size we expect") {
                double condition = condition_ptr->view(state)[0];

                CHECK(condition == extract_ptr->size(state));
            }
        }

        WHEN("We set the initial value to include the x value") {
            auto state = graph.empty_state();
            condition_ptr->initialize_state(state, {1});
            arr_ptr->initialize_state(state, {3});
            graph.initialize_state(state);

            THEN("`extract` has the state we expect") {
                REQUIRE(extract_ptr->size(state) == 1);
                REQUIRE(extract_ptr->shape(state)[0] == 1);
                CHECK(extract_ptr->view(state)[0] == 3);
            }
        }

        WHEN("We set the initial value to skip the x value") {
            auto state = graph.empty_state();
            condition_ptr->initialize_state(state, {0});
            arr_ptr->initialize_state(state, {3});
            graph.initialize_state(state);

            THEN("`extract` has the state we expect") {
                CHECK(extract_ptr->size(state) == 0);
                CHECK(extract_ptr->shape(state)[0] == 0);
            }

            AND_WHEN("We update `condition`") {
                condition_ptr->set_value(state, 0, 1);  // 0->1
                graph.propagate(state, graph.descendants(state, {condition_ptr}));

                THEN("extract has the state we expect") {
                    REQUIRE(extract_ptr->size(state) == 1);
                    REQUIRE(extract_ptr->shape(state)[0] == 1);
                    CHECK(extract_ptr->view(state)[0] == 3);
                }

                AND_WHEN("We commit") {
                    graph.commit(state, graph.descendants(state, {condition_ptr}));

                    THEN("extract has the state we expect") {
                        REQUIRE(extract_ptr->size(state) == 1);
                        REQUIRE(extract_ptr->shape(state)[0] == 1);
                        CHECK(extract_ptr->view(state)[0] == 3);
                    }
                }

                AND_WHEN("We revert") {
                    graph.revert(state, graph.descendants(state, {condition_ptr}));

                    THEN("extract has the state we expect") {
                        CHECK(extract_ptr->size(state) == 0);
                        CHECK(extract_ptr->shape(state)[0] == 0);
                    }
                }
            }
        }
    }

    GIVEN("Two arrays of same size but different shape") {
        auto condition_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{2, 4}, 0, 5);
        auto arr_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{2, 2, 2}, -10, 10);

        THEN("We can construct an `extract` with them") {
            auto extract_ptr = graph.emplace_node<ExtractNode>(condition_ptr, arr_ptr);

            AND_WHEN("We initialize the state") {
                auto state = graph.empty_state();
                condition_ptr->initialize_state(state, {0, 1, 1, 0, 0, 1, 1, 0});
                arr_ptr->initialize_state(state, {1, 2, 3, 4, 5, 6, 7, 8});
                graph.initialize_state(state);

                THEN("The `extract` output is correct") {
                    CHECK(
                        std::ranges::equal(
                            extract_ptr->view(state), std::vector<double>{2, 3, 6, 7}
                        )
                    );
                }
            }
        }
    }

    GIVEN("Two same-sized dynamic nodes") {
        auto dyn_ptr = graph.emplace_node<DynamicArrayTestingNode>(
            std::initializer_list<ssize_t>{-1, 2}, -1, 5, false
        );
        auto condition_ptr = graph.emplace_node<BasicIndexingNode>(dyn_ptr, Slice(), 0);
        auto arr_ptr = graph.emplace_node<BasicIndexingNode>(dyn_ptr, Slice(), 1);

        auto extract_ptr = graph.emplace_node<ExtractNode>(condition_ptr, arr_ptr);

        // this node will be testing consistency/shape/etc
        graph.emplace_node<ArrayValidationNode>(extract_ptr);

        // initialize the state to an interesting starting point
        auto state = graph.empty_state();
        // condition = [0, 0, 0]
        // arr = [1, 2, 3]
        dyn_ptr->initialize_state(state, {0, 1, 0, 2, 0, 3});
        graph.initialize_state(state);

        REQUIRE(std::ranges::equal(condition_ptr->view(state), std::vector<double>{0, 0, 0}));
        REQUIRE(std::ranges::equal(arr_ptr->view(state), std::vector<double>{1, 2, 3}));

        THEN("The output is empty") { CHECK(extract_ptr->size(state) == 0); }

        WHEN("We set some of the condition to true") {
            // condition = [2, 0, 0.5]
            dyn_ptr->set(state, 0, 2);  // both truthy
            dyn_ptr->set(state, 4, 0.5);
            graph.propose(state, {dyn_ptr});

            THEN("The `extract` output is correct") {
                CHECK(std::ranges::equal(extract_ptr->view(state), std::vector<double>{1, 3}));
            }

            AND_WHEN("We change some of the array's values") {
                // arr = [1, -0.5, -0.6]
                dyn_ptr->set(state, 3, -0.5);
                dyn_ptr->set(state, 5, -0.6);
                graph.propose(state, {dyn_ptr});

                THEN("The `extract` output is correct") {
                    CHECK(
                        std::ranges::equal(extract_ptr->view(state), std::vector<double>{1, -0.6})
                    );
                }
            }

            AND_WHEN("We grow, shrink, and change the values of condition") {
                // condition = [2, 0, 0.5, 1, 1]
                // arr = [1, 2, 3, 3.5, 3.6]
                dyn_ptr->grow(state, {1, 3.5, 1, 3.6});
                // condition = [0, 0, 0.5, 1, 1]
                dyn_ptr->set(state, 0, 0);
                // condition = [0, 0, 0.5, 1]
                dyn_ptr->shrink(state);
                graph.propose(state, {dyn_ptr});

                THEN("The `extract` output is correct") {
                    CHECK(
                        std::ranges::equal(extract_ptr->view(state), std::vector<double>{3, 3.5})
                    );
                }
            }
        }
    }

    SECTION("equality") {
        auto* c0_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, 0, 5);
        auto* x0_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -10, 10);

        auto* c1_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, 0, 5);
        auto* x1_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -10, 10);

        Node* a_ptr = graph.emplace_node<ExtractNode>(c0_ptr, x0_ptr);
        Node* b_ptr = graph.emplace_node<ExtractNode>(c0_ptr, x0_ptr);
        Node* c_ptr = graph.emplace_node<ExtractNode>(c1_ptr, x0_ptr);
        Node* d_ptr = graph.emplace_node<ExtractNode>(c0_ptr, x1_ptr);

        CHECK(a_ptr->equal_to(*a_ptr));
        CHECK(a_ptr->equal_to(*b_ptr));
        CHECK(not a_ptr->equal_to(*x0_ptr));
        CHECK(not a_ptr->equal_to(*c_ptr));
        CHECK(not a_ptr->equal_to(*d_ptr));
    }

    SECTION("predecessor replacement") {
        auto* c0_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, 0, 5);
        auto* x0_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -10, 10);

        auto* c1_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, 0, 5);
        auto* x1_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -10, 10);

        auto* extract_ptr = graph.emplace_node<ExtractNode>(c0_ptr, x0_ptr);

        c1_ptr->take_successors(*c0_ptr);
        x1_ptr->take_successors(*x0_ptr);

        CHECK_THAT(extract_ptr->predecessors(), RangeEquals({c1_ptr, x1_ptr}));

        auto state = graph.empty_state();
        c0_ptr->initialize_state(state, {0});
        x0_ptr->initialize_state(state, {1});
        c1_ptr->initialize_state(state, {1});
        x1_ptr->initialize_state(state, {5});
        graph.initialize_state(state);

        CHECK_THAT(extract_ptr->view(state), RangeEquals({5}));
    }
}

TEST_CASE("ArgWhereNode") {
    auto graph = Graph();

    GIVEN("A scalar integer") {
        auto scalar_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -5, 5);

        THEN("We cannot construct a ArgWhereNode from it") {
            CHECK_THROWS_AS(graph.emplace_node<ArgWhereNode>(scalar_ptr), std::invalid_argument);
        }
    }

    GIVEN("A 1d integer array and its ArgWhereNode") {
        auto arr_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{5}, -5, 5);
        auto nz_ptr = graph.emplace_node<ArgWhereNode>(arr_ptr);

        THEN("The ArgWhereNode has the shape/properties we expect") {
            CHECK(nz_ptr->ndim() == 2);
            CHECK(nz_ptr->dynamic());
            CHECK(std::ranges::equal(nz_ptr->shape(), std::vector<ssize_t>{-1, 1}));
            CHECK(nz_ptr->integral());
            CHECK(nz_ptr->min() == 0);
            CHECK(nz_ptr->max() == 4);  // largest index into an array of length 5
        }

        THEN("The ArgWhereNode has the sizeinfo we expect") {
            auto sizeinfo = nz_ptr->sizeinfo();
            CHECK(sizeinfo.min == 0);
            CHECK(sizeinfo.max == 5);  // ndim (1) * size (5)
        }

        WHEN("We initialize a state with some non-zero values") {
            auto state = graph.empty_state();
            arr_ptr->initialize_state(state, {0, 3, 0, -2, 1});
            graph.initialize_state(state);

            THEN("The output is the transpose of the non-zero indices") {
                CHECK(std::ranges::equal(nz_ptr->shape(state), std::vector<ssize_t>{3, 1}));
                CHECK(nz_ptr->size(state) == 3);
                CHECK(std::ranges::equal(nz_ptr->view(state), std::vector<double>{1, 3, 4}));
            }

            AND_WHEN("We change some values and propagate") {
                arr_ptr->set_value(state, 0, 4);  // becomes non-zero
                arr_ptr->set_value(state, 3, 0);  // becomes zero
                graph.propagate(state, graph.descendants(state, {arr_ptr}));

                THEN("The output is updated") {
                    CHECK(std::ranges::equal(nz_ptr->view(state), std::vector<double>{0, 1, 4}));
                }

                AND_WHEN("We commit") {
                    graph.commit(state, graph.descendants(state, {arr_ptr}));

                    THEN("The output is retained") {
                        CHECK(std::ranges::equal(nz_ptr->view(state), std::vector<double>{0, 1, 4}));
                    }
                }

                AND_WHEN("We revert") {
                    graph.revert(state, graph.descendants(state, {arr_ptr}));

                    THEN("The output returns to its original value") {
                        CHECK(std::ranges::equal(nz_ptr->view(state), std::vector<double>{1, 3, 4}));
                        CHECK(std::ranges::equal(nz_ptr->shape(state), std::vector<ssize_t>{3, 1}));
                    }
                }
            }
        }
    }

    GIVEN("A 2d integer array and its ArgWhereNode") {
        auto arr_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{2, 3}, -5, 5);
        auto nz_ptr = graph.emplace_node<ArgWhereNode>(arr_ptr);

        THEN("The ArgWhereNode has the shape/properties we expect") {
            CHECK(std::ranges::equal(nz_ptr->shape(), std::vector<ssize_t>{-1, 2}));
            CHECK(nz_ptr->min() == 0);
            CHECK(nz_ptr->max() == 2);  // largest index is into the length-3 axis
        }

        WHEN("We initialize a state") {
            auto state = graph.empty_state();
            // [[0, 5, 0],
            //  [0, 0, 2]]
            arr_ptr->initialize_state(state, {0, 5, 0, 0, 0, 2});
            graph.initialize_state(state);

            THEN("The output holds the (row, col) index of each non-zero") {
                CHECK(std::ranges::equal(nz_ptr->shape(state), std::vector<ssize_t>{2, 2}));
                // (0, 1) and (1, 2)
                CHECK(std::ranges::equal(nz_ptr->view(state), std::vector<double>{0, 1, 1, 2}));
            }
        }
    }

    GIVEN("A 1d dynamic array and its ArgWhereNode, tracked for consistency") {
        auto dyn_ptr = graph.emplace_node<DynamicArrayTestingNode>(
            std::initializer_list<ssize_t>{-1}, -5, 5, true
        );
        auto nz_ptr = graph.emplace_node<ArgWhereNode>(dyn_ptr);
        graph.emplace_node<ArrayValidationNode>(nz_ptr);

        auto state = graph.empty_state();
        dyn_ptr->initialize_state(state, {0, 3, 0, 2});
        graph.initialize_state(state);

        THEN("The output starts correct") {
            CHECK(std::ranges::equal(nz_ptr->view(state), std::vector<double>{1, 3}));
        }

        WHEN("We flip a value, grow, and shrink") {
            dyn_ptr->set(state, 0, 5);       // index 0 becomes non-zero
            dyn_ptr->grow(state, {0, 7});    // append indices 4 (zero) and 5 (non-zero)
            dyn_ptr->shrink(state);          // drop index 5
            graph.propose(state, {dyn_ptr});

            THEN("The output is correct") {
                // array is now [5, 3, 0, 2, 0], non-zero at 0, 1, 3
                CHECK(std::ranges::equal(nz_ptr->view(state), std::vector<double>{0, 1, 3}));
            }
        }

        WHEN("We make changes but reject them") {
            dyn_ptr->set(state, 1, 0);
            dyn_ptr->grow(state, {9});
            graph.propose(state, {dyn_ptr}, [](const Graph&, State&) { return false; });

            THEN("The output is unchanged") {
                CHECK(std::ranges::equal(nz_ptr->view(state), std::vector<double>{1, 3}));
            }
        }
    }

    GIVEN("A 2d dynamic array and its ArgWhereNode, tracked for consistency") {
        auto dyn_ptr = graph.emplace_node<DynamicArrayTestingNode>(
            std::initializer_list<ssize_t>{-1, 2}, -5, 5, true
        );
        auto nz_ptr = graph.emplace_node<ArgWhereNode>(dyn_ptr);
        graph.emplace_node<ArrayValidationNode>(nz_ptr);

        auto state = graph.empty_state();
        // [[0, 0], [1, 0]]
        dyn_ptr->initialize_state(state, {0, 0, 1, 0});
        graph.initialize_state(state);

        THEN("The output starts correct") {
            CHECK(std::ranges::equal(nz_ptr->shape(state), std::vector<ssize_t>{1, 2}));
            CHECK(std::ranges::equal(nz_ptr->view(state), std::vector<double>{1, 0}));
        }

        WHEN("We grow a row with a non-zero and propagate") {
            dyn_ptr->grow(state, {0, 3});  // append row [0, 3]
            graph.propose(state, {dyn_ptr});

            THEN("The new (row, col) index appears") {
                // non-zero at (1, 0) and (2, 1)
                CHECK(std::ranges::equal(nz_ptr->view(state), std::vector<double>{1, 0, 2, 1}));
            }
        }
    }

    SECTION("equality") {
        auto* a0_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{3}, -5, 5);
        auto* a1_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{3}, -5, 5);

        Node* a_ptr = graph.emplace_node<ArgWhereNode>(a0_ptr);
        Node* b_ptr = graph.emplace_node<ArgWhereNode>(a0_ptr);
        Node* c_ptr = graph.emplace_node<ArgWhereNode>(a1_ptr);

        CHECK(a_ptr->equal_to(*a_ptr));
        CHECK(a_ptr->equal_to(*b_ptr));
        CHECK(not a_ptr->equal_to(*c_ptr));
        CHECK(not a_ptr->equal_to(*a0_ptr));
    }

    SECTION("predecessor replacement") {
        auto* a0_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{3}, -5, 5);
        auto* a1_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{3}, -5, 5);

        auto* nz_ptr = graph.emplace_node<ArgWhereNode>(a0_ptr);

        a1_ptr->take_successors(*a0_ptr);

        CHECK_THAT(nz_ptr->predecessors(), RangeEquals({a1_ptr}));

        auto state = graph.empty_state();
        a0_ptr->initialize_state(state, {0, 0, 0});
        a1_ptr->initialize_state(state, {1, 0, 2});
        graph.initialize_state(state);

        CHECK_THAT(nz_ptr->view(state), RangeEquals({0, 2}));
    }
}

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

    GIVEN("Three dynamic nodes and a where node") {
        auto dyn_ptr =
            graph.emplace_node<DynamicArrayTestingNode>(std::initializer_list<ssize_t>{-1, 3});

        auto condition_ptr = graph.emplace_node<BasicIndexingNode>(dyn_ptr, Slice(), 0);
        auto x_ptr = graph.emplace_node<BasicIndexingNode>(dyn_ptr, Slice(), 1);
        auto y_ptr = graph.emplace_node<BasicIndexingNode>(dyn_ptr, Slice(), 2);
        auto where_ptr = graph.emplace_node<WhereNode>(condition_ptr, x_ptr, y_ptr);
        graph.emplace_node<ArrayValidationNode>(where_ptr);

        // this node will be testing consistency/shape/etc
        graph.emplace_node<ArrayValidationNode>(where_ptr);

        // initialize the state to an interesting starting point
        auto state = graph.empty_state();
        dyn_ptr->initialize_state(
            state,
            {
                // condition, x, y
                0,
                1,
                1,
                1,
                2,
                1,
                0,
                3,
                1,
                2,
                4,
                1,
            }
        );
        graph.initialize_state(state);

        THEN("`where` has the expected sizeinfo") {
            CHECK(where_ptr->sizeinfo().array_ptr == dyn_ptr);
            CHECK(where_ptr->sizeinfo().multiplier == fraction(1, 3));
        }

        THEN("`where` has the expected output") {
            CHECK_THAT(where_ptr->view(state), RangeEquals({1, 2, 1, 4}));
        }

        WHEN("We update `condition`") {
            // condition_ptr->set_value(state, 0, 4);
            // condition_ptr->set_value(state, 3, 0);
            dyn_ptr->set(state, 0 * 3, 4);
            dyn_ptr->set(state, 3 * 3, 0);
            graph.propose(state, {dyn_ptr});

            THEN("`where` has the expected output") {
                CHECK_THAT(where_ptr->view(state), RangeEquals({1, 2, 1, 1}));
            }
        }

        WHEN("We do a bunch of updates") {
            // condition_ptr->set_value(state, 0, 4);
            dyn_ptr->set(state, 0 * 3 + 0, 4);
            // x_ptr->set_value(state, 0, 4);
            dyn_ptr->set(state, 0 * 3 + 1, 4);
            // y_ptr->set_value(state, 0, 3);
            dyn_ptr->set(state, 0 * 3 + 2, 3);
            // x_ptr->set_value(state, 1, 3);
            dyn_ptr->set(state, 1 * 3 + 1, 3);
            // x_ptr->set_value(state, 1, 3);  // deliberate duplicate
            dyn_ptr->set(state, 1 * 3 + 1, 3);
            // y_ptr->set_value(state, 1, 2);
            dyn_ptr->set(state, 1 * 3 + 2, 2);
            // y_ptr->set_value(state, 2, 2);
            dyn_ptr->set(state, 2 * 3 + 2, 2);

            graph.propose(state, {dyn_ptr});

            THEN("`where` has the expected output") {
                // condition = [4, 1, 0, 2]
                // x = [4, 3, 3, 4]
                // y = [3, 2, 2, 1]
                CHECK_THAT(where_ptr->view(state), RangeEquals({4, 3, 2, 4}));
            }
        }

        WHEN("We grow and shrink the arrays") {
            dyn_ptr->grow(state, {0, -1, 5});
            // dyn_ptr->grow(state, {1, -2, 6});
            // dyn_ptr->shrink(state);

            graph.propose(state, {dyn_ptr});

            THEN("`where` has the expected output") {
                // condition = [0, 1, 0, 2, 0]
                // x = [1, 2, 3, 4, -1]
                // y = [1, 1, 1, 1, 5]
                CHECK_THAT(where_ptr->view(state), RangeEquals({1, 2, 1, 4, 5}));
            }
        }

        WHEN("We shrink and grow the arrays") {
            dyn_ptr->shrink(state);
            dyn_ptr->grow(state, {0, -1, 5});
            // set condition[0] = 1
            dyn_ptr->set(state, 0, 1);
            dyn_ptr->grow(state, {1, -2, 6});
            dyn_ptr->shrink(state);
            // set y[2] = -7
            dyn_ptr->set(state, 2 * 3 + 2, -7);
            dyn_ptr->shrink(state);

            graph.propose(state, {dyn_ptr});

            THEN("`where` has the expected output") {
                // condition = [1, 1, 0]
                // x = [1, 2, 3]
                // y = [1, 1, 1]
                CHECK_THAT(where_ptr->view(state), RangeEquals({1, 2, -7}));
            }
        }
    }

    GIVEN("Three differently sized dynamic nodes") {
        auto condition_ptr = graph.emplace_node<SetNode>(10);
        auto x_ptr = graph.emplace_node<SetNode>(10);
        auto y_ptr = graph.emplace_node<SetNode>(10);

        THEN("We cannot create a where node with them") {
            CHECK_THROWS_AS(WhereNode(condition_ptr, x_ptr, y_ptr), std::invalid_argument);
        }
    }

    SECTION("equality") {
        auto* c0_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, 0, 5);
        auto* x0_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -10, 10);
        auto* y0_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, 0, 10);

        auto* c1_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, 0, 5);
        auto* x1_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -10, 10);
        auto* y1_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, 0, 10);

        Node* a_ptr = graph.emplace_node<WhereNode>(c0_ptr, x0_ptr, y0_ptr);
        Node* b_ptr = graph.emplace_node<WhereNode>(c0_ptr, x0_ptr, y0_ptr);
        Node* c_ptr = graph.emplace_node<WhereNode>(c1_ptr, x0_ptr, y0_ptr);
        Node* d_ptr = graph.emplace_node<WhereNode>(c0_ptr, x1_ptr, y0_ptr);
        Node* e_ptr = graph.emplace_node<WhereNode>(c0_ptr, x0_ptr, y1_ptr);

        CHECK(a_ptr->equal_to(*a_ptr));
        CHECK(a_ptr->equal_to(*b_ptr));
        CHECK(not a_ptr->equal_to(*x0_ptr));
        CHECK(not a_ptr->equal_to(*c_ptr));
        CHECK(not a_ptr->equal_to(*d_ptr));
        CHECK(not a_ptr->equal_to(*e_ptr));
    }

    SECTION("predecessor replacement") {
        auto* c0_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, 0, 5);
        auto* x0_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -10, 10);
        auto* y0_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, 0, 10);

        auto* c1_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, 0, 5);
        auto* x1_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -10, 10);
        auto* y1_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, 0, 10);

        auto* where_ptr = graph.emplace_node<WhereNode>(c0_ptr, x0_ptr, y0_ptr);

        c1_ptr->take_successors(*c0_ptr);
        x1_ptr->take_successors(*x0_ptr);
        y1_ptr->take_successors(*y0_ptr);

        CHECK_THAT(where_ptr->predecessors(), RangeEquals({c1_ptr, x1_ptr, y1_ptr}));

        auto state = graph.empty_state();
        c0_ptr->initialize_state(state, {0});
        x0_ptr->initialize_state(state, {1});
        y0_ptr->initialize_state(state, {2});
        c1_ptr->initialize_state(state, {1});
        x1_ptr->initialize_state(state, {3});
        y1_ptr->initialize_state(state, {4});
        graph.initialize_state(state);

        CHECK_THAT(where_ptr->view(state), RangeEquals({3}));
    }
}

}  // namespace dwave::optimization

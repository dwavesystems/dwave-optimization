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

using Catch::Matchers::RangeEquals;

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
            std::initializer_list<ssize_t>{-1}, 1, 5, false
        );
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
            std::initializer_list<ssize_t>{-1}, -10, -2, false
        );
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

    SECTION("equality") {
        auto graph = Graph();

        auto* c0_ptr = graph.emplace_node<ConstantNode>(std::vector{0, 1, 2, 3});
        auto* c1_ptr = graph.emplace_node<ConstantNode>(std::vector{4, 5, 6, 7});

        Node* a_ptr = graph.emplace_node<MeanNode>(c0_ptr);
        Node* b_ptr = graph.emplace_node<MeanNode>(c0_ptr);
        Node* c_ptr = graph.emplace_node<MeanNode>(c1_ptr);

        CHECK(a_ptr->equal_to(*a_ptr));
        CHECK(a_ptr->equal_to(*b_ptr));
        CHECK(not a_ptr->equal_to(*c0_ptr));
        CHECK(not a_ptr->equal_to(*c_ptr));
    }

    SECTION("predecessor replacement") {
        auto graph = Graph();

        auto* c0_ptr = graph.emplace_node<ConstantNode>(std::vector{0, 1, 2, 3});
        auto* c1_ptr = graph.emplace_node<ConstantNode>(std::vector{7, 6, 5, 4});

        auto* mean_ptr = graph.emplace_node<MeanNode>(c0_ptr);

        c1_ptr->take_successors(*c0_ptr);

        CHECK_THAT(mean_ptr->predecessors(), RangeEquals({c1_ptr}));

        auto state = graph.initialize_state();
        CHECK_THAT(mean_ptr->view(state), RangeEquals({5.5}));
    }
}

TEST_CASE("BinCount") {
    auto graph = Graph();

    GIVEN("An empty integer node with upper/lower bounds above zero and a c node") {
        auto i_ptr = graph.emplace_node<IntegerNode>(0, 0, 10);
        auto bincount_ptr = graph.emplace_node<BinCountNode>(i_ptr);
        graph.emplace_node<ArrayValidationNode>(bincount_ptr);

        WHEN("We initialize a state") {
            auto state = graph.initialize_state();

            // bincount should return empty for an empty input
            THEN("The initial bincount is correct") {
                CHECK(bincount_ptr->view(state).size() == 0);
            }
        }
    }

    GIVEN("An non-empty integer node with upper/lower bounds above zero and a bincount node") {
        auto i_ptr = graph.emplace_node<IntegerNode>(7, 0, 10);
        auto bincount_ptr = graph.emplace_node<BinCountNode>(i_ptr);
        graph.emplace_node<ArrayValidationNode>(bincount_ptr);

        WHEN("We initialize a state") {
            auto state = graph.empty_state();
            i_ptr->initialize_state(state, {0.0, 0.0, 0.0, 2.0, 3.0, 3.0, 5.0});
            graph.initialize_state(state);

            // Size of bincount array should be 6
            THEN("The initial bincount is correct") {
                CHECK(bincount_ptr->view(state).size() == 6);
                std::vector<double> bincount_test = {3.0, 0.0, 1.0, 2.0, 0.0, 1.0};

                int test_size = bincount_test.size();
                for (int i = 0; i < test_size; ++i) {
                    CHECK(bincount_ptr->view(state)[i] == bincount_test[i]);
                }
            }

            AND_WHEN("We make changes to integer node and propagate") {
                i_ptr->set_value(state, 1, 1.0);
                // i_ptr should be [0.0, 1.0, 0.0, 2.0, 3.0, 3.0, 5.0]
                graph.propagate(state);

                THEN("The bincount state is correct") {
                    CHECK(bincount_ptr->view(state)[0] == 2.0);
                    CHECK(bincount_ptr->view(state)[1] == 1.0);
                }

                AND_WHEN("We commit state, make changes to integer node, and propagate") {
                    graph.commit(state);
                    i_ptr->set_value(state, 0, 7.0);
                    // i_ptr should now be [7.0, 1.0, 0.0, 2.0, 3.0, 3.0, 5.0]
                    graph.propagate(state);

                    THEN("The softmax state is correct") {
                        CHECK(bincount_ptr->view(state)[0] == 1.0);
                        CHECK(bincount_ptr->view(state)[7] == 1.0);
                    }
                    AND_WHEN("We revert to get the previous state") {
                        graph.revert(state);
                        // i_ptr should now be [0.0, 1.0, 0.0, 2.0, 3.0, 3.0, 5.0]

                        THEN("The softmax state is correct") {
                            CHECK(bincount_ptr->size(state) == 6);
                            CHECK(bincount_ptr->view(state)[0] == 2.0);
                            CHECK(bincount_ptr->view(state)[1] == 1.0);
                        }
                    }
                }
            }
        }
    }

    GIVEN("An non-empty constant node with integer values") {
        auto* i_ptr =
            graph.emplace_node<ConstantNode>(std::vector{0.0, 0.0, 0.0, 2.0, 3.0, 3.0, 5.0});
        auto bincount_ptr = graph.emplace_node<BinCountNode>(i_ptr);
        graph.emplace_node<ArrayValidationNode>(bincount_ptr);

        WHEN("We initialize a state") {
            auto state = graph.empty_state();
            graph.initialize_state(state);

            // Size of bincount array should be 6
            THEN("The initial bincount is correct") {
                CHECK(bincount_ptr->size(state) == 6);
                CHECK(bincount_ptr->size() == -1);
                std::vector<double> bincount_test = {3.0, 0.0, 1.0, 2.0, 0.0, 1.0};
                int test_size = bincount_test.size();
                for (int i = 0; i < test_size; ++i) {
                    CHECK(bincount_ptr->view(state)[i] == bincount_test[i]);
                }
            }
        }
    }

    GIVEN("A dynamic array node and a bincount node") {
        auto* dyn_ptr = graph.emplace_node<DynamicArrayTestingNode>(
            std::initializer_list<ssize_t>{-1}, 0.0, 10.0, true
        );
        auto* bincount_ptr = graph.emplace_node<BinCountNode>(dyn_ptr);

        graph.emplace_node<ArrayValidationNode>(bincount_ptr);

        WHEN("We initialize a state") {
            auto state = graph.empty_state();
            dyn_ptr->initialize_state(state, {3.0, 3.0, 8.0});
            // array should be [3.0, 3.0, 8.0]
            graph.initialize_state(state);

            THEN("The initial bincount state and size is correct") {
                CHECK(bincount_ptr->view(state)[0] == 0);
                CHECK(bincount_ptr->view(state)[3] == 2);
                CHECK(bincount_ptr->view(state)[8] == 1);
                CHECK(bincount_ptr->size() == -1);
                CHECK(bincount_ptr->size(state) == 9);
            }

            AND_WHEN("We change array values, shrink and grow array, and then propagate") {
                dyn_ptr->set(state, 0, 9.0);
                dyn_ptr->shrink(state);
                dyn_ptr->grow(state, {1.0, 9.0});
                // array should be [9.0, 3.0, 1.0, 9.0]
                graph.propagate(state);

                THEN("The softmax state and size is correct") {
                    CHECK(bincount_ptr->view(state)[1] == 1);
                    CHECK(bincount_ptr->view(state)[3] == 1);
                    CHECK(bincount_ptr->view(state)[8] == 0);
                    CHECK(bincount_ptr->view(state)[9] == 2);
                    CHECK(bincount_ptr->size(state) == 10);
                }

                AND_WHEN("We revert") {
                    graph.revert(state, graph.descendants(state, {dyn_ptr}));
                    // array should be [3.0, 3.0, 8.0]

                    THEN("The softmax state and size are correct") {
                        CHECK(bincount_ptr->view(state)[1] == 0);
                        CHECK(bincount_ptr->view(state)[3] == 2);
                        CHECK(bincount_ptr->view(state)[8] == 1);
                        CHECK(bincount_ptr->size(state) == 9);
                    }
                }
            }
        }

        WHEN("We initialize a state with a non-integer value") {
            auto state = graph.empty_state();
            dyn_ptr->initialize_state(state, {1.5, 3.0, 8.0});

            THEN("We should throw an error") {
                CHECK_THROWS_WITH(graph.initialize_state(state), "array values must be integer");
            }
        }

        WHEN("We initialize a state with a negative value") {
            auto state = graph.empty_state();
            dyn_ptr->initialize_state(state, {-1.0, 3.0, 8.0});

            THEN("We should throw an error") {
                CHECK_THROWS_WITH(
                    graph.initialize_state(state), "array values must be non-negative"
                );
            }
        }

        WHEN("We initialize a state with a value larger than allowed") {
            auto state = graph.empty_state();
            dyn_ptr->initialize_state(state, {1.0, 3.0, 3000000000.0});

            THEN("We should throw an error") {
                CHECK_THROWS_WITH(graph.initialize_state(state), "array values too large");
            }
        }
    }

    SECTION("equality") {
        auto graph = Graph();

        auto* c0_ptr = graph.emplace_node<ConstantNode>(std::vector{0, 1, 2, 3});
        auto* c1_ptr = graph.emplace_node<ConstantNode>(std::vector{4, 5, 6, 7});

        Node* a_ptr = graph.emplace_node<BinCountNode>(c0_ptr);
        Node* b_ptr = graph.emplace_node<BinCountNode>(c0_ptr);
        Node* c_ptr = graph.emplace_node<BinCountNode>(c1_ptr);

        CHECK(a_ptr->equal_to(*a_ptr));
        CHECK(a_ptr->equal_to(*b_ptr));
        CHECK(not a_ptr->equal_to(*c0_ptr));
        CHECK(not a_ptr->equal_to(*c_ptr));
    }

    SECTION("predecessor replacement") {
        auto graph = Graph();

        auto* c0_ptr = graph.emplace_node<ConstantNode>(std::vector{0, 0, 0, 1});
        auto* c1_ptr = graph.emplace_node<ConstantNode>(std::vector{0, 1, 1, 3});

        auto* bincount_ptr = graph.emplace_node<BinCountNode>(c0_ptr);

        c1_ptr->take_successors(*c0_ptr);

        CHECK_THAT(bincount_ptr->predecessors(), RangeEquals({c1_ptr}));

        auto state = graph.initialize_state();
        CHECK(bincount_ptr->view(state)[0] == 1.0);
    }
}

}  // namespace dwave::optimization

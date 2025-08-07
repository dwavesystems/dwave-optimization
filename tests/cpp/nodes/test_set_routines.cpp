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

#include <catch2/benchmark/catch_benchmark_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "dwave-optimization/nodes/collections.hpp"
#include "dwave-optimization/nodes/constants.hpp"
#include "dwave-optimization/nodes/numbers.hpp"
#include "dwave-optimization/nodes/set_routines.hpp"
#include "dwave-optimization/nodes/testing.hpp"

using Catch::Matchers::RangeEquals;

namespace dwave::optimization {

TEST_CASE("IsInNode") {
    auto graph = Graph();

    GIVEN("Two constant nodes and an isin node") {
        auto c1_ptr = graph.emplace_node<ConstantNode>(std::vector{1.1, 5.0, 3.0, 2.0});
        auto c2_ptr = graph.emplace_node<ConstantNode>(std::vector{1.1, 2.2});

        auto isin_ptr = graph.emplace_node<IsInNode>(c2_ptr, c1_ptr);

        graph.emplace_node<ArrayValidationNode>(isin_ptr);

        CHECK(isin_ptr->min() == 0.0);
        CHECK(isin_ptr->max() == 1.0);

        WHEN("We initialize a state") {
            auto state = graph.initialize_state();

            THEN("The initial isin state is correct") {
                CHECK_THAT(isin_ptr->view(state), RangeEquals({1.0, 0.0}));
            }
        }
    }

    GIVEN("Two integer nodes and an isin node") {
        auto i1_ptr = graph.emplace_node<IntegerNode>(6);
        auto i2_ptr = graph.emplace_node<IntegerNode>(3);

        auto isin_ptr = graph.emplace_node<IsInNode>(i2_ptr, i1_ptr);

        graph.emplace_node<ArrayValidationNode>(isin_ptr);

        WHEN("We initialize a state") {
            auto state = graph.empty_state();
            i1_ptr->initialize_state(state, {5, 1, 7, 9, 1, 1});
            i2_ptr->initialize_state(state, {2, 1, 9});
            graph.initialize_state(state);

            THEN("The initial isin state is correct") {
                CHECK_THAT(isin_ptr->view(state), RangeEquals({0.0, 1.0, 1.0}));
            }

            AND_WHEN("We make some changes to test_elements integer node and propagate") {
                i1_ptr->set_value(state, 1, 2);
                i1_ptr->set_value(state, 3, 4);
                // i1_ptr should now be [5, 2, 7, 4, 1, 1]

                graph.propagate(state);

                THEN("The isin state is correct") {
                    CHECK_THAT(isin_ptr->view(state), RangeEquals({1.0, 1.0, 0.0}));
                }
                AND_WHEN("We commit, make changes to both integer nodes, and propagate") {
                    graph.commit(state);
                    i1_ptr->set_value(state, 4, 2);
                    i1_ptr->set_value(state, 5, 10);
                    // i1_ptr should now be [5, 2, 7, 4, 2, 10]
                    i2_ptr->set_value(state, 0, 3);
                    i2_ptr->set_value(state, 2, 10);
                    // i2_ptr should now be [3, 1, 10]

                    graph.propagate(state);

                    THEN("The isin state is correct") {
                        CHECK_THAT(isin_ptr->view(state), RangeEquals({0.0, 0.0, 1.0}));
                    }
                }
            }
        }
        WHEN("We initialize a state") {
            auto state = graph.empty_state();
            i1_ptr->initialize_state(state, {5, 1, 7, 9, 1, 1});
            i2_ptr->initialize_state(state, {0, 1, 0});
            graph.initialize_state(state);

            CHECK_THAT(isin_ptr->view(state), RangeEquals({0.0, 1.0, 0.0}));
            THEN("We make some changes to both integer node, and revert") {
                graph.commit(state);
                i1_ptr->set_value(state, 0, 2);
                i1_ptr->set_value(state, 4, 4);
                // i1_ptr should now be [2, 1, 7, 9, 4, 1]
                i2_ptr->set_value(state, 0, 4);
                i2_ptr->set_value(state, 1, 10);
                i2_ptr->set_value(state, 2, 1);
                // i2_ptr should now be [4, 10, 1]

                graph.revert(state);

                THEN("The isin state is correct") {
                    CHECK_THAT(isin_ptr->view(state), RangeEquals({0.0, 1.0, 0.0}));
                }
            }
        }
    }

    GIVEN("Two dynamic set nodes and an isin node") {
        auto set1_ptr = graph.emplace_node<SetNode>(6);
        auto set2_ptr = graph.emplace_node<SetNode>(10);
        auto isin_ptr = graph.emplace_node<IsInNode>(set2_ptr, set1_ptr);
        graph.emplace_node<ArrayValidationNode>(isin_ptr);

        CHECK(isin_ptr->size() == -1);

        WHEN("We initialize a state") {
            auto state = graph.initialize_state();

            THEN("The isin's state is empty like the set2's") {
                CHECK(isin_ptr->size(state) == 0);
                REQUIRE(isin_ptr->ndim() == 1);
                CHECK(isin_ptr->shape(state)[0] == 0);
            }

            AND_WHEN("We grow the set nodes and propagate") {
                set1_ptr->assign(state, std::vector<double>{2, 4, 5});
                set2_ptr->assign(state, std::vector<double>{3, 1, 6, 7});
                graph.propagate(state);

                THEN("The isin's state is correct") {
                    CHECK_THAT(isin_ptr->view(state), RangeEquals({0.0, 0.0, 0.0, 0.0}));
                }

                AND_WHEN("We commit, shrink and change the set nodes, and propagate") {
                    graph.commit(state);

                    set1_ptr->shrink(state);
                    // set1 should be [2, 4]
                    set2_ptr->grow(state);
                    set2_ptr->grow(state);
                    // set2 should be [3, 1, 6, 7, 0, 2]

                    graph.propagate(state);

                    THEN("The isin's state is correct") {
                        CHECK(isin_ptr->size(state) == 6);
                        CHECK_THAT(isin_ptr->view(state),
                                   RangeEquals({0.0, 0.0, 0.0, 0.0, 0.0, 1.0}));
                    }
                }
            }
        }
    }
}
}  // namespace dwave::optimization

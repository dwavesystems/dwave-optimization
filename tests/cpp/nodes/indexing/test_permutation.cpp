// Copyright 2026 D-Wave
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
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "catch2/matchers/catch_matchers.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/nodes/collections.hpp"
#include "dwave-optimization/nodes/constants.hpp"
#include "dwave-optimization/nodes/indexing.hpp"
#include "dwave-optimization/nodes/testing.hpp"

using Catch::Matchers::RangeEquals;

namespace dwave::optimization {

TEST_CASE("PermutationNode") {
    auto graph = Graph();

    GIVEN("A 2d NxN matrix and an order array") {
        std::vector<double> values = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        auto arr_ptr =
            graph.emplace_node<ConstantNode>(values, std::initializer_list<ssize_t>{3, 3});

        std::vector<double> order = {2, 1, 0};
        auto i_ptr = graph.emplace_node<ConstantNode>(order);

        WHEN("We permute the matrix by the order") {
            auto out_ptr = graph.emplace_node<PermutationNode>(arr_ptr, i_ptr);

            THEN("We get the shape we expect") {
                CHECK(out_ptr->size() == 9);
                CHECK_THAT(out_ptr->shape(), RangeEquals({3, 3}));
            }

            THEN("We see the predecessors we expect") {
                CHECK(
                    std::ranges::equal(
                        out_ptr->predecessors(), std::vector<Node*>{arr_ptr, i_ptr}
                    )
                );
            }

            THEN("We see the min/max/integral we expect") {
                CHECK(out_ptr->min() == 0);
                CHECK(out_ptr->max() == 8);
                CHECK(out_ptr->integral());
            }

            AND_WHEN("We create a state") {
                auto state = graph.initialize_state();

                THEN("We can read out the state of the nodes") {
                    CHECK(std::ranges::equal(arr_ptr->view(state), values));
                    CHECK_THAT(i_ptr->view(state), RangeEquals({2, 1, 0}));
                    CHECK_THAT(out_ptr->view(state), RangeEquals({8, 7, 6, 5, 4, 3, 2, 1, 0}));
                }
            }
        }
    }

    GIVEN("A 2d NxN matrix and an order array of size smaller than first dimension") {
        std::vector<double> values = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        auto arr_ptr =
            graph.emplace_node<ConstantNode>(values, std::initializer_list<ssize_t>{3, 3});

        std::vector<double> order = {0, 1};
        auto i_ptr = graph.emplace_node<ConstantNode>(order);

        WHEN("We permute the matrix by the order") {
            auto out_ptr = graph.emplace_node<PermutationNode>(arr_ptr, i_ptr);

            THEN("We get the shape we expect") {
                CHECK(out_ptr->size() == 4);
                CHECK_THAT(out_ptr->shape(), RangeEquals({2, 2}));
            }

            THEN("We see the predecessors we expect") {
                CHECK(
                    std::ranges::equal(
                        out_ptr->predecessors(), std::vector<Node*>{arr_ptr, i_ptr}
                    )
                );
            }

            THEN("We see the min/max/integral we expect") {
                CHECK(out_ptr->min() == 0);
                CHECK(out_ptr->max() == 8);
                CHECK(out_ptr->integral());
            }

            AND_WHEN("We create a state") {
                auto state = graph.initialize_state();

                THEN("We can read out the state of the nodes") {
                    CHECK(std::ranges::equal(arr_ptr->view(state), values));
                    CHECK_THAT(i_ptr->view(state), RangeEquals({0, 1}));
                    CHECK_THAT(out_ptr->view(state), RangeEquals({0, 1, 3, 4}));
                }
            }
        }
    }

    GIVEN("A 2d NxN matrix and a list of size smaller than first dimension") {
        std::vector<double> values = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        auto arr_ptr =
            graph.emplace_node<ConstantNode>(values, std::initializer_list<ssize_t>{3, 3});

        auto list_ptr = graph.emplace_node<ListNode>(3, 2, 2);

        WHEN("We permute the matrix by the order") {
            auto out_ptr = graph.emplace_node<PermutationNode>(arr_ptr, list_ptr);

            THEN("We get the shape we expect") {
                CHECK(out_ptr->size() == 4);
                CHECK_THAT(out_ptr->shape(), RangeEquals({2, 2}));
            }

            AND_WHEN("We create a state") {
                auto state = graph.empty_state();
                std::vector<double> order{0, 1};
                list_ptr->initialize_state(state, order);
                graph.initialize_state(state);

                THEN("We can read out the state of the nodes") {
                    CHECK(std::ranges::equal(arr_ptr->view(state), values));
                    CHECK_THAT(list_ptr->view(state), RangeEquals({0, 1}));
                    CHECK_THAT(out_ptr->view(state), RangeEquals({0, 1, 3, 4}));
                }

                AND_WHEN("We mutate the list") {
                    list_ptr->exchange(state, 0, 1);
                    graph.propagate(state, graph.descendants(state, {list_ptr}));

                    THEN("The state is updated") {
                        CHECK_THAT(out_ptr->view(state), RangeEquals({4, 3, 1, 0}));
                    }

                    AND_WHEN("We commit") {
                        graph.commit(state, graph.descendants(state, {list_ptr}));

                        THEN("The values stick, and the diff is cleared") {
                            CHECK_THAT(out_ptr->view(state), RangeEquals({4, 3, 1, 0}));

                            CHECK(out_ptr->size_diff(state) == 0);
                            CHECK(out_ptr->diff(state).size() == 0);
                        }
                    }

                    AND_WHEN("We revert") {
                        graph.revert(state, graph.descendants(state, {list_ptr}));

                        THEN("We're back to where we started") {
                            CHECK_THAT(list_ptr->view(state), RangeEquals({0, 1}));
                            CHECK_THAT(out_ptr->view(state), RangeEquals({0, 1, 3, 4}));

                            CHECK(out_ptr->size_diff(state) == 0);
                            CHECK(out_ptr->diff(state).size() == 0);
                        }
                    }
                }
            }
        }
    }

    SECTION("equality") {
        std::vector<double> values = {0, 1, 2, 3};
        auto* arr0_ptr = graph.emplace_node<ConstantNode>(values, std::array<ssize_t, 2>{2, 2});
        auto* arr1_ptr = graph.emplace_node<ConstantNode>(values, std::array<ssize_t, 2>{2, 2});

        auto* order0_ptr = graph.emplace_node<ListNode>(2);
        auto* order1_ptr = graph.emplace_node<ListNode>(2);

        Node* a_ptr = graph.emplace_node<PermutationNode>(arr0_ptr, order0_ptr);
        Node* b_ptr = graph.emplace_node<PermutationNode>(arr0_ptr, order0_ptr);
        Node* c_ptr = graph.emplace_node<PermutationNode>(arr1_ptr, order0_ptr);
        Node* d_ptr = graph.emplace_node<PermutationNode>(arr0_ptr, order1_ptr);

        CHECK(a_ptr->equal_to(*a_ptr));
        CHECK(a_ptr->equal_to(*b_ptr));
        CHECK(not a_ptr->equal_to(*arr0_ptr));
        CHECK(not a_ptr->equal_to(*c_ptr));
        CHECK(not a_ptr->equal_to(*d_ptr));
    }

    SECTION("predecessor replacement") {
        std::vector<double> values0 = {0, 1, 2, 3};
        auto* arr0_ptr = graph.emplace_node<ConstantNode>(values0, std::array<ssize_t, 2>{2, 2});
        std::vector<double> values1 = {5, 6, 7, 8};
        auto* arr1_ptr = graph.emplace_node<ConstantNode>(values1, std::array<ssize_t, 2>{2, 2});

        auto* order0_ptr = graph.emplace_node<ListNode>(2);
        auto* order1_ptr = graph.emplace_node<ListNode>(2);

        auto* permutation_ptr = graph.emplace_node<PermutationNode>(arr0_ptr, order0_ptr);

        arr1_ptr->take_successors(*arr0_ptr);
        CHECK_THAT(
            permutation_ptr->predecessors(), RangeEquals(std::array<Node*, 2>{arr1_ptr, order0_ptr})
        );
        order1_ptr->take_successors(*order0_ptr);
        CHECK_THAT(
            permutation_ptr->predecessors(), RangeEquals(std::array<Node*, 2>{arr1_ptr, order1_ptr})
        );

        auto state = graph.empty_state();
        order0_ptr->initialize_state(state, {0, 1});
        order1_ptr->initialize_state(state, {1, 0});
        graph.initialize_state(state);

        CHECK_THAT(permutation_ptr->view(state), RangeEquals({8, 7, 6, 5}));
    }
}

}  // namespace dwave::optimization

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

#include <ranges>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "dwave-optimization/nodes/collections.hpp"
#include "dwave-optimization/nodes/constants.hpp"
#include "dwave-optimization/nodes/indexing.hpp"
#include "dwave-optimization/nodes/manipulation.hpp"
#include "dwave-optimization/nodes/mathematical.hpp"
#include "dwave-optimization/nodes/numbers.hpp"
#include "dwave-optimization/nodes/testing.hpp"

namespace dwave::optimization {

TEST_CASE("ConcatenateNode") {
    GIVEN("Two constant nodes with 8 elements each") {
        auto a = ConstantNode(std::vector{1, 2, 3, 4, 5, 6, 7, 8});
        auto b = ConstantNode(std::vector{9, 10, 11, 12, 13, 14, 15, 16});

        WHEN("Reshaped to (2,2,2) and concatenated on axis 1") {
            auto ra = ReshapeNode(&a, std::vector<ssize_t>({2, 2, 2}));
            auto rb = ReshapeNode(&b, std::vector<ssize_t>({2, 2, 2}));

            auto c = ConcatenateNode(std::vector<ArrayNode*>{&ra, &rb}, 1);

            THEN("The concatenated node has shape (2,4,2) and ndim 3") {
                CHECK(std::ranges::equal(c.shape(), std::vector{2, 4, 2}));
                CHECK(c.ndim() == 3);
            }
        }
    }

    GIVEN("Two constant nodes with shape (2,2,2,1)") {
        auto graph = Graph();

        auto a_ptr = graph.emplace_node<ConstantNode>(std::vector<double>{1, 2, 3, 4, 5, 6, 7, 8});
        auto b_ptr = graph.emplace_node<ConstantNode>(
                std::vector<double>{9, 10, 11, 12, 13, 14, 15, 16});

        auto ra_ptr = graph.emplace_node<ReshapeNode>(a_ptr, std::vector<ssize_t>{2, 2, 2, 1});
        auto rb_ptr = graph.emplace_node<ReshapeNode>(b_ptr, std::vector<ssize_t>{2, 2, 2, 1});

        std::vector<ArrayNode*> v{ra_ptr, rb_ptr};

        WHEN("Concatenated on axis 0") {
            auto c_ptr = graph.emplace_node<ConcatenateNode>(v, 0);

            THEN("The concatenated node has shape (4,2,2,1)") {
                CHECK(std::ranges::equal(c_ptr->shape(), std::vector{4, 2, 2, 1}));
            }

            AND_WHEN("The graph is initialized") {
                auto state = graph.initialize_state();
                auto expected =
                        std::vector<ssize_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
                THEN("Buffer is initialized correctly") {
                    CHECK(std::ranges::equal(c_ptr->view(state), expected));
                }
            }
        }

        WHEN("Concatenated on axis 1") {
            auto c_ptr = graph.emplace_node<ConcatenateNode>(v, 1);

            THEN("The concatenated node has shape (2,4,2,1)") {
                CHECK(std::ranges::equal(c_ptr->shape(), std::vector{2, 4, 2, 1}));
            }

            AND_WHEN("The graph is initialized") {
                auto state = graph.initialize_state();
                auto expected =
                        std::vector<ssize_t>{1, 2, 3, 4, 9, 10, 11, 12, 5, 6, 7, 8, 13, 14, 15, 16};
                THEN("Buffer is initialized correctly") {
                    CHECK(std::ranges::equal(c_ptr->view(state), expected));
                }
            }
        }

        WHEN("Concatenated on axis 2") {
            auto c_ptr = graph.emplace_node<ConcatenateNode>(v, 2);

            THEN("The concatenated node has shape (2,2,4,1)") {
                CHECK(std::ranges::equal(c_ptr->shape(), std::vector{2, 2, 4, 1}));
            }

            AND_WHEN("The graph is initialized") {
                auto state = graph.initialize_state();
                auto expected =
                        std::vector<ssize_t>{1, 2, 9, 10, 3, 4, 11, 12, 5, 6, 13, 14, 7, 8, 15, 16};
                THEN("Buffer is initialized correctly") {
                    CHECK(std::ranges::equal(c_ptr->view(state), expected));
                }
            }
        }
    }

    GIVEN("Two arrays with shapes (2,2,2) and (3,2,2) that are concatenated on axis 0") {
        auto graph = Graph();
        auto a_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{2, 2, 2}, 0, 100);
        auto b_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{3, 2, 2}, 0, 100);
        auto c_ptr = graph.emplace_node<ConcatenateNode>(std::vector<ArrayNode*>{a_ptr, b_ptr}, 0);

        WHEN("The graph is initialized and the arrays are given new values") {
            auto state = graph.initialize_state();

            // Ranges 1-8, 9-20
            for (auto i : std::ranges::iota_view(0, 8)) a_ptr->set_value(state, i, i + 1);
            for (auto i : std::ranges::iota_view(0, 12)) b_ptr->set_value(state, i, i + 9);

            graph.propose(state, {a_ptr, b_ptr}, [](const Graph&, State&) { return true; });
            THEN("ConcatenateNode values are propagated correctly") {
                CHECK(std::ranges::equal(c_ptr->view(state), std::ranges::iota_view{1, 21}));
            }
        }
    }

    GIVEN("Two arrays with shapes (2,2,2) and (2,3,2) that are concatenated on axis 1") {
        auto graph = Graph();
        auto a_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{2, 2, 2}, 0, 100);
        auto b_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{2, 3, 2}, 0, 100);
        auto c_ptr = graph.emplace_node<ConcatenateNode>(std::vector<ArrayNode*>{a_ptr, b_ptr}, 1);

        WHEN("The graph is initialized and the arrays are given new values") {
            auto state = graph.initialize_state();

            // Ranges 1-8, 9-20
            for (auto i : std::ranges::iota_view(0, 8)) a_ptr->set_value(state, i, i + 1);
            for (auto i : std::ranges::iota_view(0, 12)) b_ptr->set_value(state, i, i + 9);

            graph.propose(state, {a_ptr, b_ptr}, [](const Graph&, State&) { return true; });
            std::vector<ssize_t> expected = {1, 2, 3, 4, 9,  10, 11, 12, 13, 14,
                                             5, 6, 7, 8, 15, 16, 17, 18, 19, 20};
            THEN("ConcatenateNode values are propagated correctly") {
                CHECK(std::ranges::equal(c_ptr->view(state), expected));
            }
        }
    }

    GIVEN("Two arrays with shapes (2,2,2) and (2,2,3) that are concatenated on axis 2") {
        auto graph = Graph();
        auto a_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{2, 2, 2}, 0, 100);
        auto b_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{2, 2, 3}, 0, 100);
        auto c_ptr = graph.emplace_node<ConcatenateNode>(std::vector<ArrayNode*>{a_ptr, b_ptr}, 2);

        WHEN("The graph is initialized and the arrays are given new values") {
            auto state = graph.initialize_state();

            // Ranges 1-8, 9-20
            for (auto i : std::ranges::iota_view(0, 8)) a_ptr->set_value(state, i, i + 1);
            for (auto i : std::ranges::iota_view(0, 12)) b_ptr->set_value(state, i, i + 9);

            graph.propose(state, {a_ptr, b_ptr}, [](const Graph&, State&) { return true; });
            std::vector<ssize_t> expected = {1, 2, 9,  10, 11, 3, 4, 12, 13, 14,
                                             5, 6, 15, 16, 17, 7, 8, 18, 19, 20};
            THEN("ConcatenateNode values are propagated correctly") {
                CHECK(std::ranges::equal(c_ptr->view(state), expected));
            }
        }
    }

    GIVEN("Three arrays with 12, 18 and 24 elements") {
        auto graph = Graph();

        auto a = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{12}, 0, 100);
        auto b = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{18}, 0, 100);
        auto c = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{24}, 0, 100);

        WHEN("The arrays are reshaped, concatenated on axis 0, and assigned new values") {
            auto a_r = graph.emplace_node<ReshapeNode>(a, std::vector<ssize_t>{2, 3, 2, 1});
            auto b_r = graph.emplace_node<ReshapeNode>(b, std::vector<ssize_t>{3, 3, 2, 1});
            auto c_r = graph.emplace_node<ReshapeNode>(c, std::vector<ssize_t>{4, 3, 2, 1});
            auto abc_ptr =
                    graph.emplace_node<ConcatenateNode>(std::vector<ArrayNode*>{a_r, b_r, c_r}, 0);

            auto state = graph.initialize_state();

            // Values 1-12, 13-30, 31-54
            for (auto i : std::ranges::iota_view(0, 12)) a->set_value(state, i, i + 1);
            for (auto i : std::ranges::iota_view(0, 18)) b->set_value(state, i, i + 13);
            for (auto i : std::ranges::iota_view(0, 24)) c->set_value(state, i, i + 31);

            graph.propose(state, {a, b, c}, [](const Graph&, State&) { return true; });
            auto expected = std::ranges::iota_view(1, 55);

            THEN("ConcatenatedNode values are propagated correctly") {
                CHECK(std::ranges::equal(abc_ptr->view(state), expected));
            }
        }

        WHEN("The arrays are reshaped, concatenated on axis 1, and assigned new values") {
            auto a_r = graph.emplace_node<ReshapeNode>(a, std::vector<ssize_t>{3, 2, 2, 1});
            auto b_r = graph.emplace_node<ReshapeNode>(b, std::vector<ssize_t>{3, 3, 2, 1});
            auto c_r = graph.emplace_node<ReshapeNode>(c, std::vector<ssize_t>{3, 4, 2, 1});
            auto abc_ptr =
                    graph.emplace_node<ConcatenateNode>(std::vector<ArrayNode*>{a_r, b_r, c_r}, 1);

            auto state = graph.initialize_state();

            // Values 1-12, 13-30, 31-54
            for (auto i : std::ranges::iota_view(0, 12)) a->set_value(state, i, i + 1);
            for (auto i : std::ranges::iota_view(0, 18)) b->set_value(state, i, i + 13);
            for (auto i : std::ranges::iota_view(0, 24)) c->set_value(state, i, i + 31);

            graph.propose(state, {a, b, c}, [](const Graph&, State&) { return true; });
            std::vector<double> expected{1,  2,  3,  4,  13, 14, 15, 16, 17, 18, 31, 32, 33, 34,
                                         35, 36, 37, 38, 5,  6,  7,  8,  19, 20, 21, 22, 23, 24,
                                         39, 40, 41, 42, 43, 44, 45, 46, 9,  10, 11, 12, 25, 26,
                                         27, 28, 29, 30, 47, 48, 49, 50, 51, 52, 53, 54};

            THEN("ConcatenatedNode values are propagated correctly") {
                CHECK(std::ranges::equal(abc_ptr->view(state), expected));
            }
        }

        WHEN("The arrays are reshaped, concatenated on axis 2, and assigned new values") {
            auto a_r = graph.emplace_node<ReshapeNode>(a, std::vector<ssize_t>{3, 2, 2, 1});
            auto b_r = graph.emplace_node<ReshapeNode>(b, std::vector<ssize_t>{3, 2, 3, 1});
            auto c_r = graph.emplace_node<ReshapeNode>(c, std::vector<ssize_t>{3, 2, 4, 1});
            auto abc_ptr =
                    graph.emplace_node<ConcatenateNode>(std::vector<ArrayNode*>{a_r, b_r, c_r}, 2);

            auto state = graph.initialize_state();

            // Values 1-12, 13-30, 31-54
            for (auto i : std::ranges::iota_view(0, 12)) a->set_value(state, i, i + 1);
            for (auto i : std::ranges::iota_view(0, 18)) b->set_value(state, i, i + 13);
            for (auto i : std::ranges::iota_view(0, 24)) c->set_value(state, i, i + 31);

            graph.propose(state, {a, b, c}, [](const Graph&, State&) { return true; });
            std::vector<double> expected{1,  2,  13, 14, 15, 31, 32, 33, 34, 3,  4,  16, 17, 18,
                                         35, 36, 37, 38, 5,  6,  19, 20, 21, 39, 40, 41, 42, 7,
                                         8,  22, 23, 24, 43, 44, 45, 46, 9,  10, 25, 26, 27, 47,
                                         48, 49, 50, 11, 12, 28, 29, 30, 51, 52, 53, 54};

            THEN("ConcatenatedNode values are propagated correctly") {
                CHECK(std::ranges::equal(abc_ptr->view(state), expected));
            }
        }

        WHEN("The arrays are reshaped, concatenated on axis 3, and assigned new values") {
            auto a_r = graph.emplace_node<ReshapeNode>(a, std::vector<ssize_t>{3, 2, 1, 2});
            auto b_r = graph.emplace_node<ReshapeNode>(b, std::vector<ssize_t>{3, 2, 1, 3});
            auto c_r = graph.emplace_node<ReshapeNode>(c, std::vector<ssize_t>{3, 2, 1, 4});
            auto abc_ptr =
                    graph.emplace_node<ConcatenateNode>(std::vector<ArrayNode*>{a_r, b_r, c_r}, 3);

            auto state = graph.initialize_state();

            // Values 1-12, 13-30, 31-54
            for (auto i : std::ranges::iota_view(0, 12)) a->set_value(state, i, i + 1);
            for (auto i : std::ranges::iota_view(0, 18)) b->set_value(state, i, i + 13);
            for (auto i : std::ranges::iota_view(0, 24)) c->set_value(state, i, i + 31);

            graph.propose(state, {a, b, c}, [](const Graph&, State&) { return true; });
            std::vector<double> expected{1,  2,  13, 14, 15, 31, 32, 33, 34, 3,  4,  16, 17, 18,
                                         35, 36, 37, 38, 5,  6,  19, 20, 21, 39, 40, 41, 42, 7,
                                         8,  22, 23, 24, 43, 44, 45, 46, 9,  10, 25, 26, 27, 47,
                                         48, 49, 50, 11, 12, 28, 29, 30, 51, 52, 53, 54};

            THEN("ConcatenatedNode values are propagated correctly") {
                CHECK(std::ranges::equal(abc_ptr->view(state), expected));
            }
        }
    }
}

TEST_CASE("PutNode") {
    SECTION("a = [0, 1, 2, 3, 4], ind = [0, 2], v = [-44, -55], b = PutNode(a, ind, v)") {
        auto graph = Graph();

        auto a_ptr = graph.emplace_node<ConstantNode>(std::vector{0, 1, 2, 3, 4});
        auto ind_ptr = graph.emplace_node<ConstantNode>(std::vector{0, 2});
        auto v_ptr = graph.emplace_node<ConstantNode>(std::vector{-44, -55});

        auto put_ptr = graph.emplace_node<PutNode>(a_ptr, ind_ptr, v_ptr);

        graph.emplace_node<ArrayValidationNode>(put_ptr);

        auto state = graph.initialize_state();

        CHECK(std::ranges::equal(put_ptr->view(state), std::vector{-44, 1, -55, 3, 4}));
    }

    GIVEN("A 3x3 array of integers, and a set indexer, and a same-length array of values") {
        auto graph = Graph();

        auto a_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{3, 3});
        auto ind_ptr = graph.emplace_node<SetNode>(9);

        auto c_ptr =
                graph.emplace_node<ConstantNode>(std::vector{-10, -1, -2, -3, -4, -5, -6, -7, -8});
        auto v_ptr = graph.emplace_node<AdvancedIndexingNode>(c_ptr, ind_ptr);

        auto put_ptr = graph.emplace_node<PutNode>(a_ptr, ind_ptr, v_ptr);

        graph.emplace_node<ArrayValidationNode>(put_ptr);

        auto state = graph.empty_state();
        a_ptr->initialize_state(state, {0, 1, 2, 3, 4, 5, 6, 7, 8});
        ind_ptr->initialize_state(state, {1, 3});
        graph.initialize_state(state);

        CHECK(std::ranges::equal(put_ptr->view(state), std::vector{0, -1, 2, -3, 4, 5, 6, 7, 8}));
        CHECK(std::ranges::equal(put_ptr->mask(state), std::vector{0, 1, 0, 1, 0, 0, 0, 0, 0}));

        WHEN("The base array is updated") {
            a_ptr->set_value(state, 2, 4);  // not overwritten by put
            a_ptr->set_value(state, 3, 6);  // overwritten by put
            graph.propagate(state, graph.descendants(state, {a_ptr}));

            THEN("Commiting updates the output correctly") {
                graph.commit(state, graph.descendants(state, {a_ptr}));
                CHECK(std::ranges::equal(put_ptr->view(state),
                                         std::vector{0, -1, 4, -3, 4, 5, 6, 7, 8}));
                CHECK(std::ranges::equal(put_ptr->mask(state),
                                         std::vector{0, 1, 0, 1, 0, 0, 0, 0, 0}));
            }

            THEN("Reverting results in no change") {
                graph.revert(state, graph.descendants(state, {ind_ptr}));
                CHECK(std::ranges::equal(put_ptr->view(state),
                                         std::vector{0, -1, 2, -3, 4, 5, 6, 7, 8}));
                CHECK(std::ranges::equal(put_ptr->mask(state),
                                         std::vector{0, 1, 0, 1, 0, 0, 0, 0, 0}));
            }
        }

        WHEN("The indices are updated") {
            ind_ptr->grow(state);
            ind_ptr->exchange(state, 0, 3);
            graph.propagate(state, graph.descendants(state, {ind_ptr}));

            THEN("Commiting updates the output correctly") {
                graph.commit(state, graph.descendants(state, {ind_ptr}));
                // implementation details but required for the final check
                CHECK(std::ranges::equal(ind_ptr->view(state), std::vector{2, 3, 0}));
                CHECK(std::ranges::equal(v_ptr->view(state), std::vector{-2, -3, -10}));

                CHECK(std::ranges::equal(put_ptr->view(state),
                                         std::vector{-10, 1, -2, -3, 4, 5, 6, 7, 8}));
                CHECK(std::ranges::equal(put_ptr->mask(state),
                                         std::vector{1, 0, 1, 1, 0, 0, 0, 0, 0}));
            }

            THEN("Reverting results in no change") {
                graph.revert(state, graph.descendants(state, {ind_ptr}));
                CHECK(std::ranges::equal(put_ptr->view(state),
                                         std::vector{0, -1, 2, -3, 4, 5, 6, 7, 8}));
                CHECK(std::ranges::equal(put_ptr->mask(state),
                                         std::vector{0, 1, 0, 1, 0, 0, 0, 0, 0}));
            }
        }

        WHEN("An index is removed") {
            ind_ptr->shrink(state);
            graph.propagate(state, graph.descendants(state, {ind_ptr}));

            THEN("Commiting updates the output correctly") {
                graph.commit(state, graph.descendants(state, {ind_ptr}));
                // implementation details but required for the final check
                CHECK(std::ranges::equal(ind_ptr->view(state), std::vector{1}));
                CHECK(std::ranges::equal(v_ptr->view(state), std::vector{-1}));

                CHECK(std::ranges::equal(put_ptr->view(state),
                                         std::vector{0, -1, 2, 3, 4, 5, 6, 7, 8}));
                CHECK(std::ranges::equal(put_ptr->mask(state),
                                         std::vector{0, 1, 0, 0, 0, 0, 0, 0, 0}));
            }

            THEN("Reverting results in no change") {
                graph.revert(state, graph.descendants(state, {ind_ptr}));
                CHECK(std::ranges::equal(put_ptr->view(state),
                                         std::vector{0, -1, 2, -3, 4, 5, 6, 7, 8}));
                CHECK(std::ranges::equal(put_ptr->mask(state),
                                         std::vector{0, 1, 0, 1, 0, 0, 0, 0, 0}));
            }
        }

        WHEN("An index is removed and then added again") {
            ind_ptr->shrink(state);
            ind_ptr->grow(state);
            graph.propagate(state, graph.descendants(state, {ind_ptr}));

            THEN("Commiting updates the output correctly") {
                graph.commit(state, graph.descendants(state, {ind_ptr}));
                // implementation details but required for the final check
                CHECK(std::ranges::equal(ind_ptr->view(state), std::vector{1, 3}));
                CHECK(std::ranges::equal(v_ptr->view(state), std::vector{-1, -3}));

                CHECK(std::ranges::equal(put_ptr->view(state),
                                         std::vector{0, -1, 2, -3, 4, 5, 6, 7, 8}));
                CHECK(std::ranges::equal(put_ptr->mask(state),
                                         std::vector{0, 1, 0, 1, 0, 0, 0, 0, 0}));
            }

            THEN("Reverting results in no change") {
                graph.revert(state, graph.descendants(state, {ind_ptr}));
                CHECK(std::ranges::equal(put_ptr->view(state),
                                         std::vector{0, -1, 2, -3, 4, 5, 6, 7, 8}));
                CHECK(std::ranges::equal(put_ptr->mask(state),
                                         std::vector{0, 1, 0, 1, 0, 0, 0, 0, 0}));
            }
        }

        WHEN("An index is added and then removed again") {
            ind_ptr->grow(state);
            ind_ptr->shrink(state);
            graph.propagate(state, graph.descendants(state, {ind_ptr}));

            THEN("Commiting updates the output correctly") {
                graph.commit(state, graph.descendants(state, {ind_ptr}));

                // implementation details but required for the final check
                CHECK(std::ranges::equal(ind_ptr->view(state), std::vector{1, 3}));
                CHECK(std::ranges::equal(v_ptr->view(state), std::vector{-1, -3}));

                CHECK(std::ranges::equal(put_ptr->view(state),
                                         std::vector{0, -1, 2, -3, 4, 5, 6, 7, 8}));
                CHECK(std::ranges::equal(put_ptr->mask(state),
                                         std::vector{0, 1, 0, 1, 0, 0, 0, 0, 0}));
            }

            THEN("Reverting results in no change") {
                graph.revert(state, graph.descendants(state, {ind_ptr}));
                CHECK(std::ranges::equal(put_ptr->view(state),
                                         std::vector{0, -1, 2, -3, 4, 5, 6, 7, 8}));

                CHECK(std::ranges::equal(put_ptr->mask(state),
                                         std::vector{0, 1, 0, 1, 0, 0, 0, 0, 0}));
            }
        }
    }

    GIVEN("A 2x3 array of integers, 3 indices with duplicates, and 3 values") {
        auto graph = Graph();

        auto a_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{2, 3});
        auto ind_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{3}, 0, 5);
        auto v_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{3});

        auto put_ptr = graph.emplace_node<PutNode>(a_ptr, ind_ptr, v_ptr);

        graph.emplace_node<ArrayValidationNode>(put_ptr);

        auto state = graph.empty_state();
        a_ptr->initialize_state(state, {0, 1, 2, 3, 4, 5});
        ind_ptr->initialize_state(state, {1, 3, 1});
        v_ptr->initialize_state(state, {10, 20, 30});
        graph.initialize_state(state);

        // duplicate takes the "most recent", i.e. second, value
        CHECK(std::ranges::equal(put_ptr->view(state), std::vector{0, 30, 2, 20, 4, 5}));
        CHECK(std::ranges::equal(put_ptr->mask(state), std::vector{0, 2, 0, 1, 0, 0}));

        WHEN("One of the duplicate values is updated") {
            v_ptr->set_value(state, 0, 40);
            graph.propagate(state, graph.descendants(state, {v_ptr}));

            THEN("Commiting keeps the change") {
                graph.commit(state, graph.descendants(state, {v_ptr}));
                CHECK(std::ranges::equal(put_ptr->view(state), std::vector{0, 40, 2, 20, 4, 5}));
                CHECK(std::ranges::equal(put_ptr->mask(state), std::vector{0, 2, 0, 1, 0, 0}));
            }

            THEN("Reverting results in no change") {
                graph.revert(state, graph.descendants(state, {v_ptr}));
                CHECK(std::ranges::equal(put_ptr->view(state), std::vector{0, 30, 2, 20, 4, 5}));
                CHECK(std::ranges::equal(put_ptr->mask(state), std::vector{0, 2, 0, 1, 0, 0}));
            }
        }
    }

    GIVEN("A length 6 array of integers, and two integer arrays for indexing and values") {
        auto graph = Graph();

        auto a_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{6});
        auto ind_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{2}, 0, 5);
        auto val_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{2});
        auto mask_ptr = graph.emplace_node<DynamicArrayTestingNode>(
                std::initializer_list<ssize_t>{-1}, 0, 1, true);

        auto put_ptr = graph.emplace_node<PutNode>(
                a_ptr, graph.emplace_node<AdvancedIndexingNode>(ind_ptr, mask_ptr),
                graph.emplace_node<AdvancedIndexingNode>(val_ptr, mask_ptr));

        graph.emplace_node<ArrayValidationNode>(put_ptr);

        auto state = graph.empty_state();
        a_ptr->initialize_state(state, {0, 1, 2, 3, 4, 5});
        graph.initialize_state(state);

        WHEN("The indices contain two duplicates with different corresponding values") {
            ind_ptr->set_value(state, 0, 0);
            ind_ptr->set_value(state, 1, 0);

            val_ptr->set_value(state, 0, 10);
            val_ptr->set_value(state, 1, 11);

            mask_ptr->grow(state, {0});
            mask_ptr->grow(state, {1});

            // This is now effectively put([0, 1, 2, 3, 4, 5], [0, 0], [10, 11])

            graph.propagate(state, graph.descendants(state, {val_ptr, ind_ptr, mask_ptr}));
            graph.commit(state, graph.descendants(state, {val_ptr, ind_ptr, mask_ptr}));

            THEN("State is correct") {
                CHECK(std::ranges::equal(put_ptr->view(state), std::vector{11, 1, 2, 3, 4, 5}));

                AND_WHEN("We shrink the mask, shrinking both effective indices and values") {
                    mask_ptr->shrink(state);

                    // This is now effectively put([0, 1, 2, 3, 4, 5], [0], [10])

                    graph.propagate(state, graph.descendants(state, {val_ptr, ind_ptr, mask_ptr}));
                    graph.commit(state, graph.descendants(state, {val_ptr, ind_ptr, mask_ptr}));

                    THEN("First index goes to the other provided value") {
                        CHECK(std::ranges::equal(put_ptr->view(state),
                                                 std::vector{10, 1, 2, 3, 4, 5}));
                    }
                }
            }
        }
    }
}

TEST_CASE("ReshapeNode") {
    GIVEN("A 1d array encoding range(12)") {
        auto A = ConstantNode(std::vector{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});

        WHEN("It is reshaped to a (12,) array") {
            auto B = ReshapeNode(&A, {12});

            THEN("It has the shape/size/etc we expect") {
                CHECK(B.ndim() == 1);
                CHECK(std::ranges::equal(B.shape(), std::vector{12}));
            }
        }
    }

    GIVEN("Two paths through a DAG calculating the same thing") {
        auto graph = Graph();

        auto x_ptr = graph.emplace_node<BinaryNode>(std::vector<ssize_t>{9});
        auto A_ptr = graph.emplace_node<ConstantNode>(std::vector{1, 2, 3, 4, 5, 6, 7, 8, 9});

        // lhs path, (x * A).sum()
        auto lhs_ptr = graph.emplace_node<SumNode>(graph.emplace_node<MultiplyNode>(x_ptr, A_ptr));

        // rhs path, (x.reshape(3, 3) * A.reshape(3, 3)).sum()
        auto rhs_ptr = graph.emplace_node<SumNode>(graph.emplace_node<MultiplyNode>(
                graph.emplace_node<ReshapeNode>(x_ptr, std::vector<ssize_t>{3, 3}),
                graph.emplace_node<ReshapeNode>(A_ptr, std::vector<ssize_t>{3, 3})));

        graph.emplace_node<ArrayValidationNode>(lhs_ptr);
        graph.emplace_node<ArrayValidationNode>(rhs_ptr);

        WHEN("We do random moves") {
            auto state = graph.initialize_state();

            auto rng = std::default_random_engine(42);
            std::uniform_int_distribution<int> num_moves(1, 5);
            std::uniform_int_distribution<int> coin(0, 1);
            std::uniform_int_distribution<int> variable(0, 8);

            for (int i = 0; i < 100; ++i) {
                // do a random number of flips of random variables
                for (int j = 0, n = num_moves(rng); j < n; ++j) {
                    x_ptr->flip(state, variable(rng));
                }

                graph.propose(state, {x_ptr}, [&](const Graph&, State&) { return coin(rng); });

                CHECK(std::ranges::equal(lhs_ptr->view(state), rhs_ptr->view(state)));
            }
        }
    }
}

TEST_CASE("SizeNode") {
    GIVEN("A 0d node") {
        auto C = ConstantNode(5);

        WHEN("We create a node accessing the length") {
            auto len = SizeNode(&C);

            THEN("The SizeNode is a scalar output") {
                CHECK(len.size() == 1);
                CHECK(len.ndim() == 0);
                CHECK(!len.dynamic());
            }

            THEN("The output is a integer and we already know the min/max") {
                CHECK(len.integral());
                CHECK(len.min() == C.size());
                CHECK(len.max() == C.size());
            }
        }
    }

    GIVEN("A 1D node with a fixed size") {
        auto C = ConstantNode(std::vector{0, 1, 2});

        WHEN("We create a node accessing the length") {
            auto len = SizeNode(&C);

            THEN("The SizeNode is a scalar output") {
                CHECK(len.size() == 1);
                CHECK(len.ndim() == 0);
                CHECK(!len.dynamic());
            }

            THEN("The output is a integer and we already know the min/max") {
                CHECK(len.integral());
                CHECK(len.min() == C.size());
                CHECK(len.max() == C.size());
            }
        }
    }

    GIVEN("A dynamic node") {
        auto S = SetNode(5, 2, 4);

        WHEN("We create a node accessing the length") {
            auto len = SizeNode(&S);

            THEN("The SizeNode is a scalar output") {
                CHECK(len.size() == 1);
                CHECK(len.ndim() == 0);
                CHECK(!len.dynamic());
            }

            THEN("The output is a integer and we already know the min/max") {
                CHECK(len.integral());
                CHECK(len.min() == 2);
                CHECK(len.max() == 4);
            }
        }

        AND_WHEN("We create a node indirectly accessing the length") {
            auto T = BasicIndexingNode(&S, Slice(0, -1));  // also dynamic, but indirect
            auto len = SizeNode(&T);

            THEN("The SizeNode is a scalar output") {
                CHECK(len.size() == 1);
                CHECK(len.ndim() == 0);
                CHECK(!len.dynamic());
            }

            THEN("The output is a integer and we already know the min/max") {
                CHECK(len.integral());

                // these should always be true
                CHECK(len.min() >= 0);
                CHECK(len.max() <= std::numeric_limits<ssize_t>::max());

                // these are an implementation detail and could change in the future
                CHECK(len.min() == 0);
                CHECK(len.max() == std::numeric_limits<ssize_t>::max());
            }
        }
    }

    GIVEN("A graph with a SetNode and a corresponding SizeNode, and a state") {
        auto graph = Graph();

        auto s_ptr = graph.emplace_node<SetNode>(5);
        auto len_ptr = graph.emplace_node<SizeNode>(s_ptr);

        auto state = graph.empty_state();
        s_ptr->initialize_state(state, {});  // default to empty
        graph.initialize_state(state);

        THEN("The state of the SizeNode is the same as the size of the SetNode") {
            CHECK(*(len_ptr->buff(state)) == s_ptr->size(state));
        }

        WHEN("We update the set and propagate") {
            s_ptr->grow(state);
            s_ptr->propagate(state);
            len_ptr->propagate(state);

            THEN("The state of the SizeNode is the same as the size of the SetNode") {
                CHECK(*(len_ptr->buff(state)) == s_ptr->size(state));
            }

            AND_WHEN("We commit") {
                s_ptr->commit(state);
                len_ptr->commit(state);

                THEN("The state of the SizeNode is the same as the size of the SetNode") {
                    CHECK(*(len_ptr->buff(state)) == s_ptr->size(state));
                }
            }

            AND_WHEN("We revert") {
                s_ptr->revert(state);
                len_ptr->revert(state);

                THEN("The state of the SizeNode is the same as the size of the SetNode") {
                    CHECK(*(len_ptr->buff(state)) == s_ptr->size(state));
                }
            }
        }
    }
}

TEST_CASE("StackNode") {
    GIVEN("Three constant nodes with 10 elements each reshaped to (2,5)") {
        auto a = ConstantNode(std::vector{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        auto b = ConstantNode(std::vector{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
        auto c = ConstantNode(std::vector{21, 22, 23, 24, 25, 26, 27, 28, 29, 30});
        auto ra = ReshapeNode(&a, std::vector<ssize_t>({2, 5}));
        auto rb = ReshapeNode(&b, std::vector<ssize_t>({2, 5}));
        auto rc = ReshapeNode(&c, std::vector<ssize_t>({2, 5}));

        WHEN("Stacked on axis 0") {
            auto c = StackNode(std::vector<ArrayNode*>{&ra, &rb, &rc}, 0);

            THEN("The stacked node has shape (2,2,5) and ndim 3") {
                CHECK(std::ranges::equal(c.shape(), std::vector{3, 2, 5}));
                CHECK(c.ndim() == 3);
            }
        }
        WHEN("Stacked on axis 1") {
            auto c = StackNode(std::vector<ArrayNode*>{&ra, &rb, &rc}, 1);

            THEN("The stacked node has shape (2,2,5) and ndim 3") {
                CHECK(std::ranges::equal(c.shape(), std::vector{2, 3, 5}));
                CHECK(c.ndim() == 3);
            }
        }
        WHEN("Stacked on axis 2") {
            auto c = StackNode(std::vector<ArrayNode*>{&ra, &rb, &rc}, 2);

            THEN("The stacked node has shape (2,5,2) and ndim 3") {
                CHECK(std::ranges::equal(c.shape(), std::vector{2, 5, 3}));
                CHECK(c.ndim() == 3);
            }
        }
    }

    GIVEN("Two constant nodes reshaped to (2,1,2,1)") {
        auto graph = Graph();

        auto a_ptr = graph.emplace_node<ConstantNode>(std::vector<double>{1, 2, 3, 4});
        auto b_ptr = graph.emplace_node<ConstantNode>(std::vector<double>{5, 6, 7, 8});

        auto ra_ptr = graph.emplace_node<ReshapeNode>(a_ptr, std::vector<ssize_t>{2, 1, 2, 1});
        auto rb_ptr = graph.emplace_node<ReshapeNode>(b_ptr, std::vector<ssize_t>{2, 1, 2, 1});

        std::vector<ArrayNode*> v{ra_ptr, rb_ptr};

        WHEN("Stacked on axis 0") {
            auto c_ptr = graph.emplace_node<StackNode>(v, 0);

            THEN("The stacked node has shape (2,2,1,2,1) and ndim 5") {
                CHECK(std::ranges::equal(c_ptr->shape(), std::vector{2, 2, 1, 2, 1}));
                CHECK(c_ptr->ndim() == 5);
            }

            AND_WHEN("The graph is initialized") {
                auto state = graph.initialize_state();
                auto expected = std::vector<ssize_t>{1, 2, 3, 4, 5, 6, 7, 8};
                THEN("Buffer is initialized correctly") {
                    CHECK(std::ranges::equal(c_ptr->view(state), expected));
                }
            }
        }

        WHEN("Stacked on axis 1") {
            auto c_ptr = graph.emplace_node<StackNode>(v, 1);

            THEN("The concatenated node has shape (2,2,1,2,1) and ndim 5") {
                CHECK(std::ranges::equal(c_ptr->shape(), std::vector{2, 2, 1, 2, 1}));
                CHECK(c_ptr->ndim() == 5);
            }

            AND_WHEN("The graph is initialized") {
                auto state = graph.initialize_state();
                auto expected = std::vector<ssize_t>{1, 2, 5, 6, 3, 4, 7, 8};
                THEN("Buffer is initialized correctly") {
                    CHECK(std::ranges::equal(c_ptr->view(state), expected));
                }
            }
        }

        WHEN("Stacked on axis 2") {
            auto c_ptr = graph.emplace_node<StackNode>(v, 2);

            THEN("The concatenated node has shape (2,1,2,2,1) and ndim 5") {
                CHECK(std::ranges::equal(c_ptr->shape(), std::vector{2, 1, 2, 2, 1}));
                CHECK(c_ptr->ndim() == 5);
            }

            AND_WHEN("The graph is initialized") {
                auto state = graph.initialize_state();
                auto expected = std::vector<ssize_t>{1, 2, 5, 6, 3, 4, 7, 8};
                THEN("Buffer is initialized correctly") {
                    CHECK(std::ranges::equal(c_ptr->view(state), expected));
                }
            }
        }

        WHEN("Stacked on axis 3") {
            auto c_ptr = graph.emplace_node<StackNode>(v, 3);

            THEN("The concatenated node has shape (2,1,2,2,1) and ndim 5") {
                CHECK(std::ranges::equal(c_ptr->shape(), std::vector{2, 1, 2, 2, 1}));
                CHECK(c_ptr->ndim() == 5);
            }

            AND_WHEN("The graph is initialized") {
                auto state = graph.initialize_state();
                auto expected = std::vector<ssize_t>{1, 5, 2, 6, 3, 7, 4, 8};
                THEN("Buffer is initialized correctly") {
                    CHECK(std::ranges::equal(c_ptr->view(state), expected));
                }
            }
        }

        WHEN("Stacked on axis 4") {
            auto c_ptr = graph.emplace_node<StackNode>(v, 4);

            THEN("The concatenated node has shape (2,1,2,1,2) and ndim 5") {
                CHECK(std::ranges::equal(c_ptr->shape(), std::vector{2, 1, 2, 1, 2}));
                CHECK(c_ptr->ndim() == 5);
            }

            AND_WHEN("The graph is initialized") {
                auto state = graph.initialize_state();
                auto expected = std::vector<ssize_t>{1, 5, 2, 6, 3, 7, 4, 8};
                THEN("Buffer is initialized correctly") {
                    CHECK(std::ranges::equal(c_ptr->view(state), expected));
                }
            }
        }
    }

    GIVEN("Three arrays with shapes (2,2)") {
        auto graph = Graph();
        auto a_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{2, 2}, 0, 100);
        auto b_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{2, 2}, 0, 100);
        auto c_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{2, 2}, 0, 100);

        WHEN("Stacked on axis 0 and the arrays are updated") {
            auto abc_ptr = graph.emplace_node<StackNode>(std::vector<ArrayNode*>{a_ptr, b_ptr, c_ptr}, 0);
            auto state = graph.initialize_state();

            THEN("shape and ndim are (2,2,3) and 3") {
                CHECK(std::ranges::equal(abc_ptr->shape(), std::vector{3, 2, 2}));
                CHECK(abc_ptr->ndim() == 3);
            }

            // Values 1-4, 5-8, 9-12
            for (auto i : std::ranges::iota_view(0, 4)) a_ptr->set_value(state, i, i + 1);
            for (auto i : std::ranges::iota_view(0, 4)) b_ptr->set_value(state, i, i + 5);
            for (auto i : std::ranges::iota_view(0, 4)) c_ptr->set_value(state, i, i + 9);

            graph.propose(state, {a_ptr, b_ptr, c_ptr}, [](const Graph&, State&) { return true; });
            THEN("StackNode values are propagated correctly") {
                CHECK(std::ranges::equal(abc_ptr->view(state), std::ranges::iota_view{1, 13}));
            }
        }

        WHEN("Stacked on axis 1 and the arrays are updated") {
            auto abc_ptr = graph.emplace_node<StackNode>(std::vector<ArrayNode*>{a_ptr, b_ptr, c_ptr}, 1);
            auto state = graph.initialize_state();

            THEN("shape and ndim are (2,2,3) and 3") {
                CHECK(std::ranges::equal(abc_ptr->shape(), std::vector{2, 3, 2}));
                CHECK(abc_ptr->ndim() == 3);
            }

            // Values 1-4, 5-8, 9-12
            for (auto i : std::ranges::iota_view(0, 4)) a_ptr->set_value(state, i, i + 1);
            for (auto i : std::ranges::iota_view(0, 4)) b_ptr->set_value(state, i, i + 5);
            for (auto i : std::ranges::iota_view(0, 4)) c_ptr->set_value(state, i, i + 9);

            graph.propose(state, {a_ptr, b_ptr, c_ptr}, [](const Graph&, State&) { return true; });
            THEN("StackNode values are propagated correctly") {
                std::vector<ssize_t> expected = { 1, 2, 5, 6, 9, 10, 3, 4, 7, 8, 11, 12 };
                CHECK(std::ranges::equal(abc_ptr->view(state), expected));
            }
        }

        WHEN("Stacked on axis 2 and the arrays are updated") {
            auto abc_ptr = graph.emplace_node<StackNode>(std::vector<ArrayNode*>{a_ptr, b_ptr, c_ptr}, 2);
            auto state = graph.initialize_state();

            THEN("shape and ndim are (2,2,3) and 3") {
                CHECK(std::ranges::equal(abc_ptr->shape(), std::vector{2, 2, 3}));
                CHECK(abc_ptr->ndim() == 3);
            }

            // Values 1-4, 5-8, 9-12
            for (auto i : std::ranges::iota_view(0, 4)) a_ptr->set_value(state, i, i + 1);
            for (auto i : std::ranges::iota_view(0, 4)) b_ptr->set_value(state, i, i + 5);
            for (auto i : std::ranges::iota_view(0, 4)) c_ptr->set_value(state, i, i + 9);

            graph.propose(state, {a_ptr, b_ptr, c_ptr}, [](const Graph&, State&) { return true; });
            THEN("StackNode values are propagated correctly") {
                std::vector<ssize_t> expected = { 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12 };
                CHECK(std::ranges::equal(abc_ptr->view(state), expected));
            }
        }
    }

    GIVEN("Integral constant nodes with many elements") {
        auto a = ConstantNode(std::vector{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        auto b = ConstantNode(std::vector{11, 12, 13, 14, 15, 16, 17, 18, 19, 20});

        WHEN("When stacked") {
            auto s = StackNode(std::vector<ArrayNode*>{&a,&b}, 0);

            THEN("The stack node is integral and we know the min and max") {
                CHECK(s.integral());
                CHECK(s.min() == 1);
                CHECK(s.max() == 20);
            }
        }
    }

    GIVEN("Two integral scalars") {
        auto a = ConstantNode(std::vector{-5});
        auto b = ConstantNode(std::vector{7});

        WHEN("When stacked") {
            auto s = StackNode(std::vector<ArrayNode*>{&a,&b}, 0);

            THEN("The stack node is integral and we know the min and max") {
                CHECK(s.integral());
                CHECK(s.min() == -5);
                CHECK(s.max() == 7);
            }
        }
    }

    GIVEN("One integral scalar that is stacked") {
        auto a = ConstantNode(std::vector{9});
        auto s = StackNode(std::vector<ArrayNode*>{&a}, 0);

        THEN("The stack node is integral and we know the min and max") {
            CHECK(s.integral());
            CHECK(s.min() == 9);
            CHECK(s.max() == 9);
        }
    }

    GIVEN("A non-integral scalar that is stacked") {
        auto a = ConstantNode(std::vector{5.5});
        auto s = StackNode(std::vector<ArrayNode*>{&a}, 0);

        THEN("THe stack node is not integral and we know the min and max") {
            CHECK(s.integral() == false);
            CHECK(s.min() == 5.5);
            CHECK(s.max() == 5.5);
        }
    }
}
}  // namespace dwave::optimization

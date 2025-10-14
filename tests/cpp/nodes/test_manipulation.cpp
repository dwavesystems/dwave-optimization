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
#include <stdexcept>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_range_equals.hpp"
#include "dwave-optimization/nodes/binaryop.hpp"
#include "dwave-optimization/nodes/collections.hpp"
#include "dwave-optimization/nodes/constants.hpp"
#include "dwave-optimization/nodes/indexing.hpp"
#include "dwave-optimization/nodes/inputs.hpp"
#include "dwave-optimization/nodes/manipulation.hpp"
#include "dwave-optimization/nodes/numbers.hpp"
#include "dwave-optimization/nodes/reduce.hpp"
#include "dwave-optimization/nodes/testing.hpp"

using Catch::Matchers::RangeEquals;

namespace dwave::optimization {

TEST_CASE("BroadcastToNode") {
    auto graph = Graph();

    SECTION("broadcast [0, 1, 2] to a (2,3) array") {
        auto c = graph.emplace_node<ConstantNode>(std::vector{0, 1, 2});
        auto b = graph.emplace_node<BroadcastToNode>(c, std::vector<ssize_t>{2, 3});
        graph.emplace_node<ArrayValidationNode>(b);
        CHECK_THAT(b->shape(), RangeEquals({2, 3}));
        CHECK_THAT(b->strides(), RangeEquals({0, 8}));

        auto state = graph.initialize_state();
        CHECK_THAT(b->view(state), RangeEquals({0, 1, 2, 0, 1, 2}));
    }

    SECTION("broadcast (3,) array to a (2,3) array") {
        auto i = graph.emplace_node<InputNode>(std::vector<ssize_t>{3});
        auto b = graph.emplace_node<BroadcastToNode>(i, std::vector<ssize_t>{2, 3});
        graph.emplace_node<ArrayValidationNode>(b);

        CHECK_THAT(b->shape(), RangeEquals({2, 3}));
        CHECK_THAT(b->strides(), RangeEquals({0, 8}));

        auto state = graph.empty_state();
        i->initialize_state(state, {0, 1, 2});
        graph.initialize_state(state);
        CHECK_THAT(b->view(state), RangeEquals({0, 1, 2, 0, 1, 2}));

        // Do an update followed by a propagation
        i->assign(state, {1, 0, 2});
        graph.propagate(state);

        CHECK_THAT(b->view(state), RangeEquals({1, 0, 2, 1, 0, 2}));

        AND_WHEN("We commit") {
            graph.commit(state);
            CHECK_THAT(b->view(state), RangeEquals({1, 0, 2, 1, 0, 2}));
        }
        AND_WHEN("We revert") {
            graph.revert(state);
            CHECK_THAT(b->view(state), RangeEquals({0, 1, 2, 0, 1, 2}));
        }
    }

    SECTION("broadcast (-1,1) array to a (-1,2) array") {
        auto s = graph.emplace_node<SetNode>(100);
        auto rs = graph.emplace_node<ReshapeNode>(s, std::vector<ssize_t>{-1, 1});
        CHECK(rs->ndim() == 2);
        auto b = graph.emplace_node<BroadcastToNode>(rs, std::vector<ssize_t>{-1, 2});
        graph.emplace_node<ArrayValidationNode>(b);

        CHECK_THAT(b->shape(), RangeEquals({-1, 2}));

        auto state = graph.empty_state();
        s->initialize_state(state, {0, 1, 2});
        graph.initialize_state(state);

        CHECK_THAT(b->shape(state), RangeEquals({3, 2}));
        CHECK_THAT(b->view(state), RangeEquals({0, 0, 1, 1, 2, 2}));

        WHEN("We grow the set") {
            s->assign(state, {0, 2, 1, 3, 5});
            graph.propagate(state);

            CHECK_THAT(b->shape(state), RangeEquals({5, 2}));
            CHECK_THAT(b->view(state), RangeEquals({0, 0, 2, 2, 1, 1, 3, 3, 5, 5}));

            AND_WHEN("We commit") {
                graph.commit(state);

                CHECK_THAT(b->shape(state), RangeEquals({5, 2}));
                CHECK_THAT(b->view(state), RangeEquals({0, 0, 2, 2, 1, 1, 3, 3, 5, 5}));
            }

            AND_WHEN("We revert") {
                graph.revert(state);

                CHECK_THAT(b->shape(state), RangeEquals({3, 2}));
                CHECK_THAT(b->view(state), RangeEquals({0, 0, 1, 1, 2, 2}));
            }
        }

        WHEN("We shrink the set") {
            s->assign(state, {0, 5});
            graph.propagate(state);

            CHECK_THAT(b->shape(state), RangeEquals({2, 2}));
            CHECK_THAT(b->view(state), RangeEquals({0, 0, 5, 5}));

            AND_WHEN("We commit") {
                graph.commit(state);

                CHECK_THAT(b->shape(state), RangeEquals({2, 2}));
                CHECK_THAT(b->view(state), RangeEquals({0, 0, 5, 5}));
            }

            AND_WHEN("We revert") {
                graph.revert(state);

                CHECK_THAT(b->shape(state), RangeEquals({3, 2}));
                CHECK_THAT(b->view(state), RangeEquals({0, 0, 1, 1, 2, 2}));
            }
        }
    }

    SECTION("broadcast [[0], [1], [2]] to a (3,2) array") {
        auto c = graph.emplace_node<ConstantNode>(std::vector{0, 1, 2}, std::vector<ssize_t>{3, 1});
        auto b = graph.emplace_node<BroadcastToNode>(c, std::vector<ssize_t>{3, 2});
        graph.emplace_node<ArrayValidationNode>(b);
        CHECK_THAT(b->shape(), RangeEquals({3, 2}));
        CHECK_THAT(b->strides(), RangeEquals({8, 0}));

        auto state = graph.initialize_state();
        CHECK_THAT(b->view(state), RangeEquals({0, 0, 1, 1, 2, 2}));
    }

    SECTION("broadcast [[0], [1], [2]] to a (2,3,2) array") {
        auto c = graph.emplace_node<ConstantNode>(std::vector{0, 1, 2}, std::vector<ssize_t>{3, 1});
        auto b = graph.emplace_node<BroadcastToNode>(c, std::vector<ssize_t>{2, 3, 2});
        graph.emplace_node<ArrayValidationNode>(b);
        CHECK_THAT(b->shape(), RangeEquals({2, 3, 2}));
        CHECK_THAT(b->strides(), RangeEquals({0, 8, 0}));

        auto state = graph.initialize_state();
        CHECK_THAT(b->view(state), RangeEquals({0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2}));
    }

    SECTION("broadcast [[0, 1], [2, 3], [4, 5]] to a (2,3,2) array") {
        auto c = graph.emplace_node<ConstantNode>(std::vector{0, 1, 2, 3, 4, 5},
                                                  std::vector<ssize_t>{3, 2});
        auto b = graph.emplace_node<BroadcastToNode>(c, std::vector<ssize_t>{2, 3, 2});
        graph.emplace_node<ArrayValidationNode>(b);
        CHECK_THAT(b->shape(), RangeEquals({2, 3, 2}));
        CHECK_THAT(b->strides(), RangeEquals({0, 16, 8}));

        auto state = graph.initialize_state();
        CHECK_THAT(b->view(state), RangeEquals({0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5}));
    }

    SECTION("broadcast (3,2) array to a (2,3,2) array") {
        auto a = graph.emplace_node<InputNode>(std::vector<ssize_t>{3, 2});
        auto b = graph.emplace_node<BroadcastToNode>(a, std::vector<ssize_t>{2, 3, 2});
        graph.emplace_node<ArrayValidationNode>(b);
        CHECK_THAT(b->shape(), RangeEquals({2, 3, 2}));
        CHECK_THAT(b->strides(), RangeEquals({0, 16, 8}));

        auto state = graph.empty_state();
        a->initialize_state(state, {0, 1, 2, 3, 4, 5});
        graph.initialize_state(state);
        CHECK_THAT(b->view(state), RangeEquals({0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5}));

        // Do an update followed by a propagation
        a->assign(state, {1, 0, 2, 3, 6, 5});
        graph.propagate(state);
        CHECK_THAT(b->view(state), RangeEquals({1, 0, 2, 3, 6, 5, 1, 0, 2, 3, 6, 5}));
    }

    SECTION("broadcast (3,2) array to a (3,2,3,2) array") {
        auto a = graph.emplace_node<InputNode>(std::vector<ssize_t>{3, 2});
        auto b = graph.emplace_node<BroadcastToNode>(a, std::vector<ssize_t>{3, 2, 3, 2});
        graph.emplace_node<ArrayValidationNode>(b);
        CHECK_THAT(b->shape(), RangeEquals({3, 2, 3, 2}));
        CHECK_THAT(b->strides(), RangeEquals({0, 0, 16, 8}));

        auto state = graph.empty_state();
        a->initialize_state(state, {0, 1, 2, 3, 4, 5});
        graph.initialize_state(state);
        CHECK_THAT(b->view(state),
                   RangeEquals({0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
                                0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5}));

        // Do an update followed by a propagation
        a->assign(state, {1, 0, 2, 3, 6, 5});
        graph.propagate(state);
        CHECK_THAT(b->view(state),
                   RangeEquals({1, 0, 2, 3, 6, 5, 1, 0, 2, 3, 6, 5, 1, 0, 2, 3, 6, 5,
                                1, 0, 2, 3, 6, 5, 1, 0, 2, 3, 6, 5, 1, 0, 2, 3, 6, 5}));
    }

    GIVEN("arr = broadcast_to(set(10).reshape(-1, 1, 1), (-1, 2, 3))") {
        auto graph = Graph();
        auto set_ptr = graph.emplace_node<SetNode>(10);
        auto reshape_ptr =
                graph.emplace_node<ReshapeNode>(set_ptr, std::array<ssize_t, 3>{-1, 1, 1});
        auto arr_ptr =
                graph.emplace_node<BroadcastToNode>(reshape_ptr, std::array<ssize_t, 3>{-1, 2, 3});
        graph.emplace_node<ArrayValidationNode>(arr_ptr);

        AND_GIVEN("set = {1, 2, 3]") {
            auto state = graph.empty_state();
            set_ptr->initialize_state(state, {1, 2, 3});
            graph.initialize_state(state);

            CHECK_THAT(arr_ptr->view(state),
                       RangeEquals({1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3}));

            WHEN("We change the set to equal {1, 2, 4, 3}") {
                set_ptr->assign(state, {1, 2, 4, 3});
                graph.propagate(state);

                CHECK_THAT(arr_ptr->view(state), RangeEquals({1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                                                              4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3}));
            }
        }
    }
}

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
                CHECK_THAT(c_ptr->shape(), RangeEquals({4, 2, 2, 1}));
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
                CHECK_THAT(c_ptr->shape(), RangeEquals({2, 4, 2, 1}));
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
                CHECK_THAT(c_ptr->shape(), RangeEquals({2, 2, 4, 1}));
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

    GIVEN("Two flat integer arrays: a = [0, 1, 2], b = [3, 4, 5], c = concatenate(a, b)") {
        auto graph = Graph();
        auto a_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{3}, 0, 100);
        auto b_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{3}, 0, 100);
        auto c_ptr = graph.emplace_node<ConcatenateNode>(std::vector<ArrayNode*>{a_ptr, b_ptr}, 0);

        graph.emplace_node<ArrayValidationNode>(c_ptr);

        auto state = graph.empty_state();
        a_ptr->initialize_state(state, {0, 1, 2});
        b_ptr->initialize_state(state, {3, 4, 5});
        graph.initialize_state(state);

        CHECK_THAT(c_ptr->view(state), RangeEquals({0, 1, 2, 3, 4, 5}));

        WHEN("b is updated to [3, 40, 5]") {
            b_ptr->set_value(state, 1, 40);
            graph.propagate(state, graph.descendants(state, {b_ptr}));

            CHECK_THAT(c_ptr->view(state), RangeEquals({0, 1, 2, 3, 40, 5}));
            CHECK_THAT(c_ptr->diff(state), RangeEquals({Update(4, 4, 40)}));
        }
    }

    GIVEN("Two flat integer arrays: a = [0, 1, 2], b = [3, 4, 5], c = concatenate(a.copy(), b)") {
        auto graph = Graph();
        auto a_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{3}, 0, 50);
        auto b_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{3}, -10, 100);
        auto a_copy_ptr = graph.emplace_node<CopyNode>(a_ptr);
        auto c_ptr =
                graph.emplace_node<ConcatenateNode>(std::vector<ArrayNode*>{a_copy_ptr, b_ptr}, 0);

        THEN("We can get the min(), max() and integral() as expected") {
            CHECK(c_ptr->integral());
            CHECK(c_ptr->min() == -10);
            CHECK(c_ptr->max() == 100);
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
                CHECK_THAT(c_ptr->view(state), RangeEquals(std::ranges::iota_view{1, 21}));
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

    GIVEN("Two constant nodes of integrals") {
        auto a = ConstantNode(std::vector{-5, 10, 3});
        auto b = ConstantNode(std::vector{1, 10, -6});

        WHEN("both are concatenated") {
            auto c = ConcatenateNode(std::vector<ArrayNode*>{&a, &b}, 0);

            THEN("the concatenated node is integral and we know the min/max") {
                CHECK(c.integral());
                CHECK(c.min() == -6);
                CHECK(c.max() == 10);
            }
        }

        WHEN("only one is concatenated") {
            auto c = ConcatenateNode(std::vector<ArrayNode*>{&a}, 0);

            THEN("the concatenated node is integral and we know the min/max") {
                CHECK(c.integral());
                CHECK(c.min() == -5);
                CHECK(c.max() == 10);
            }
        }
    }

    GIVEN("Constant nodes that contains at least one non-integral") {
        auto a = ConstantNode(std::vector{1.0, 2.2, 3.0});
        auto b = ConstantNode(std::vector{1, 2, 3});

        WHEN("Concatenated") {
            auto c = ConcatenateNode(std::vector<ArrayNode*>{&a, &b}, 0);

            THEN("The concatenated node is not integral") { CHECK(c.integral() == false); }
        }
    }
}

TEST_CASE("CopyNode") {
    GIVEN("x = IntegerNode({6}, 0, 10); y = x[::2]; c = CopyNode(y)") {
        auto graph = Graph();

        auto x_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{6}, 0, 10);
        auto y_ptr = graph.emplace_node<BasicIndexingNode>(x_ptr, Slice(0, 10, 2));
        auto c_ptr = graph.emplace_node<CopyNode>(y_ptr);

        graph.emplace_node<ArrayValidationNode>(c_ptr);

        auto state = graph.empty_state();
        x_ptr->initialize_state(state, {0, 1, 2, 3, 4, 5});
        graph.initialize_state(state);

        THEN("c has the same shape as y and the same values") {
            CHECK(std::ranges::equal(c_ptr->shape(state), y_ptr->shape(state)));
            CHECK(std::ranges::equal(c_ptr->view(state), y_ptr->view(state)));

            CHECK(!y_ptr->contiguous());
            CHECK(c_ptr->contiguous());

            CHECK(c_ptr->max() == y_ptr->max());
            CHECK(c_ptr->min() == y_ptr->min());
            CHECK(c_ptr->integral() == y_ptr->integral());
        }

        WHEN("We mutate the state of x") {
            x_ptr->set_value(state, 2, 10);
            graph.propagate(state, graph.descendants(state, {x_ptr}));

            THEN("c has the same shape as y and the same values") {
                CHECK(std::ranges::equal(c_ptr->view(state), y_ptr->view(state)));
            }

            AND_WHEN("we commit") {
                graph.commit(state, graph.descendants(state, {x_ptr}));

                THEN("c has the same shape as y and the same values") {
                    CHECK(std::ranges::equal(c_ptr->view(state), y_ptr->view(state)));
                }
            }

            AND_WHEN("we revert") {
                graph.revert(state, graph.descendants(state, {x_ptr}));

                THEN("c has the same shape as y and the same values") {
                    CHECK(std::ranges::equal(c_ptr->view(state), y_ptr->view(state)));
                }
            }
        }
    }

    GIVEN("x = SetNode(6); y = x[::2]; c = CopyNode(y)") {
        auto graph = Graph();

        auto x_ptr = graph.emplace_node<SetNode>(6);
        auto y_ptr = graph.emplace_node<BasicIndexingNode>(x_ptr, Slice(0, 10, 2));
        auto c_ptr = graph.emplace_node<CopyNode>(y_ptr);

        graph.emplace_node<ArrayValidationNode>(c_ptr);

        auto state = graph.empty_state();
        x_ptr->initialize_state(state, {0, 1, 2});
        graph.initialize_state(state);

        THEN("c has the same shape as y and the same values") {
            CHECK(std::ranges::equal(c_ptr->shape(state), y_ptr->shape(state)));
            CHECK(std::ranges::equal(c_ptr->view(state), y_ptr->view(state)));

            CHECK(!y_ptr->contiguous());
            CHECK(c_ptr->contiguous());

            CHECK(c_ptr->max() == y_ptr->max());
            CHECK(c_ptr->min() == y_ptr->min());
            CHECK(c_ptr->integral() == y_ptr->integral());
        }

        WHEN("We mutate the state of x") {
            x_ptr->grow(state);
            graph.propagate(state, graph.descendants(state, {x_ptr}));

            THEN("c has the same shape as y and the same values") {
                CHECK(std::ranges::equal(c_ptr->view(state), y_ptr->view(state)));
            }

            AND_WHEN("we commit") {
                graph.commit(state, graph.descendants(state, {x_ptr}));

                THEN("c has the same shape as y and the same values") {
                    CHECK(std::ranges::equal(c_ptr->view(state), y_ptr->view(state)));
                }
            }

            AND_WHEN("we revert") {
                graph.revert(state, graph.descendants(state, {x_ptr}));

                THEN("c has the same shape as y and the same values") {
                    CHECK(std::ranges::equal(c_ptr->view(state), y_ptr->view(state)));
                }
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

        CHECK_THAT(put_ptr->view(state), RangeEquals({-44, 1, -55, 3, 4}));
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

        CHECK_THAT(put_ptr->view(state), RangeEquals({0, -1, 2, -3, 4, 5, 6, 7, 8}));
        CHECK_THAT(put_ptr->mask(state), RangeEquals({0, 1, 0, 1, 0, 0, 0, 0, 0}));

        WHEN("The base array is updated") {
            a_ptr->set_value(state, 2, 4);  // not overwritten by put
            a_ptr->set_value(state, 3, 6);  // overwritten by put
            graph.propagate(state, graph.descendants(state, {a_ptr}));

            THEN("Commiting updates the output correctly") {
                graph.commit(state, graph.descendants(state, {a_ptr}));
                CHECK_THAT(put_ptr->view(state), RangeEquals({0, -1, 4, -3, 4, 5, 6, 7, 8}));
                CHECK_THAT(put_ptr->mask(state), RangeEquals({0, 1, 0, 1, 0, 0, 0, 0, 0}));
            }

            THEN("Reverting results in no change") {
                graph.revert(state, graph.descendants(state, {ind_ptr}));
                CHECK_THAT(put_ptr->view(state), RangeEquals({0, -1, 2, -3, 4, 5, 6, 7, 8}));
                CHECK_THAT(put_ptr->mask(state), RangeEquals({0, 1, 0, 1, 0, 0, 0, 0, 0}));
            }
        }

        WHEN("The indices are updated") {
            ind_ptr->grow(state);
            ind_ptr->exchange(state, 0, 3);
            graph.propagate(state, graph.descendants(state, {ind_ptr}));

            THEN("Commiting updates the output correctly") {
                graph.commit(state, graph.descendants(state, {ind_ptr}));
                // implementation details but required for the final check
                CHECK_THAT(ind_ptr->view(state), RangeEquals({2, 3, 0}));
                CHECK_THAT(v_ptr->view(state), RangeEquals({-2, -3, -10}));

                CHECK_THAT(put_ptr->view(state), RangeEquals({-10, 1, -2, -3, 4, 5, 6, 7, 8}));
                CHECK_THAT(put_ptr->mask(state), RangeEquals({1, 0, 1, 1, 0, 0, 0, 0, 0}));
            }

            THEN("Reverting results in no change") {
                graph.revert(state, graph.descendants(state, {ind_ptr}));
                CHECK_THAT(put_ptr->view(state), RangeEquals({0, -1, 2, -3, 4, 5, 6, 7, 8}));
                CHECK_THAT(put_ptr->mask(state), RangeEquals({0, 1, 0, 1, 0, 0, 0, 0, 0}));
            }
        }

        WHEN("An index is removed") {
            ind_ptr->shrink(state);
            graph.propagate(state, graph.descendants(state, {ind_ptr}));

            THEN("Commiting updates the output correctly") {
                graph.commit(state, graph.descendants(state, {ind_ptr}));
                // implementation details but required for the final check
                CHECK_THAT(ind_ptr->view(state), RangeEquals({1}));
                CHECK_THAT(v_ptr->view(state), RangeEquals({-1}));

                CHECK_THAT(put_ptr->view(state), RangeEquals({0, -1, 2, 3, 4, 5, 6, 7, 8}));
                CHECK_THAT(put_ptr->mask(state), RangeEquals({0, 1, 0, 0, 0, 0, 0, 0, 0}));
            }

            THEN("Reverting results in no change") {
                graph.revert(state, graph.descendants(state, {ind_ptr}));
                CHECK_THAT(put_ptr->view(state), RangeEquals({0, -1, 2, -3, 4, 5, 6, 7, 8}));
                CHECK_THAT(put_ptr->mask(state), RangeEquals({0, 1, 0, 1, 0, 0, 0, 0, 0}));
            }
        }

        WHEN("An index is removed and then added again") {
            ind_ptr->shrink(state);
            ind_ptr->grow(state);
            graph.propagate(state, graph.descendants(state, {ind_ptr}));

            THEN("Commiting updates the output correctly") {
                graph.commit(state, graph.descendants(state, {ind_ptr}));
                // implementation details but required for the final check
                CHECK_THAT(ind_ptr->view(state), RangeEquals({1, 3}));
                CHECK_THAT(v_ptr->view(state), RangeEquals({-1, -3}));

                CHECK_THAT(put_ptr->view(state), RangeEquals({0, -1, 2, -3, 4, 5, 6, 7, 8}));
                CHECK_THAT(put_ptr->mask(state), RangeEquals({0, 1, 0, 1, 0, 0, 0, 0, 0}));
            }

            THEN("Reverting results in no change") {
                graph.revert(state, graph.descendants(state, {ind_ptr}));
                CHECK_THAT(put_ptr->view(state), RangeEquals({0, -1, 2, -3, 4, 5, 6, 7, 8}));
                CHECK_THAT(put_ptr->mask(state), RangeEquals({0, 1, 0, 1, 0, 0, 0, 0, 0}));
            }
        }

        WHEN("An index is added and then removed again") {
            ind_ptr->grow(state);
            ind_ptr->shrink(state);
            graph.propagate(state, graph.descendants(state, {ind_ptr}));

            THEN("Commiting updates the output correctly") {
                graph.commit(state, graph.descendants(state, {ind_ptr}));

                // implementation details but required for the final check
                CHECK_THAT(ind_ptr->view(state), RangeEquals({1, 3}));
                CHECK_THAT(v_ptr->view(state), RangeEquals({-1, -3}));

                CHECK_THAT(put_ptr->view(state), RangeEquals({0, -1, 2, -3, 4, 5, 6, 7, 8}));
                CHECK_THAT(put_ptr->mask(state), RangeEquals({0, 1, 0, 1, 0, 0, 0, 0, 0}));
            }

            THEN("Reverting results in no change") {
                graph.revert(state, graph.descendants(state, {ind_ptr}));
                CHECK_THAT(put_ptr->view(state), RangeEquals({0, -1, 2, -3, 4, 5, 6, 7, 8}));
                CHECK_THAT(put_ptr->mask(state), RangeEquals({0, 1, 0, 1, 0, 0, 0, 0, 0}));
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
        CHECK_THAT(put_ptr->view(state), RangeEquals({0, 30, 2, 20, 4, 5}));
        CHECK_THAT(put_ptr->mask(state), RangeEquals({0, 2, 0, 1, 0, 0}));

        WHEN("One of the duplicate values is updated") {
            v_ptr->set_value(state, 0, 40);
            graph.propagate(state, graph.descendants(state, {v_ptr}));

            THEN("Commiting keeps the change") {
                graph.commit(state, graph.descendants(state, {v_ptr}));
                CHECK_THAT(put_ptr->view(state), RangeEquals({0, 40, 2, 20, 4, 5}));
                CHECK_THAT(put_ptr->mask(state), RangeEquals({0, 2, 0, 1, 0, 0}));
            }

            THEN("Reverting results in no change") {
                graph.revert(state, graph.descendants(state, {v_ptr}));
                CHECK_THAT(put_ptr->view(state), RangeEquals({0, 30, 2, 20, 4, 5}));
                CHECK_THAT(put_ptr->mask(state), RangeEquals({0, 2, 0, 1, 0, 0}));
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
                CHECK_THAT(put_ptr->view(state), RangeEquals({11, 1, 2, 3, 4, 5}));

                AND_WHEN("We shrink the mask, shrinking both effective indices and values") {
                    mask_ptr->shrink(state);

                    // This is now effectively put([0, 1, 2, 3, 4, 5], [0], [10])

                    graph.propagate(state, graph.descendants(state, {val_ptr, ind_ptr, mask_ptr}));
                    graph.commit(state, graph.descendants(state, {val_ptr, ind_ptr, mask_ptr}));

                    THEN("First index goes to the other provided value") {
                        CHECK_THAT(put_ptr->view(state), RangeEquals({10, 1, 2, 3, 4, 5}));
                    }
                }
            }
        }
    }

    GIVEN("a = [0, 0, 0, 0], indices = integer(2), values = [2, 2]") {
        auto graph = Graph();

        auto a_ptr = graph.emplace_node<ConstantNode>(std::vector{0, 0, 0, 0});
        auto indices_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{2}, 0, 3);
        auto values_ptr = graph.emplace_node<ConstantNode>(std::vector{2, 2});

        auto put_ptr = graph.emplace_node<PutNode>(a_ptr, indices_ptr, values_ptr);

        graph.emplace_node<ArrayValidationNode>(put_ptr);

        auto state = graph.empty_state();
        indices_ptr->initialize_state(state, {0, 1});
        graph.initialize_state(state);

        CHECK_THAT(put_ptr->view(state), RangeEquals({2, 2, 0, 0}));

        WHEN("We change indices and then revert and commit a few times") {
            indices_ptr->set_value(state, 1, 0);
            graph.propagate(state, graph.descendants(state, {indices_ptr}));
            CHECK_THAT(put_ptr->view(state), RangeEquals({2, 0, 0, 0}));
            CHECK_THAT(put_ptr->mask(state), RangeEquals({2, 0, 0, 0}));

            graph.revert(state, graph.descendants(state, {indices_ptr}));
            CHECK_THAT(put_ptr->view(state), RangeEquals({2, 2, 0, 0}));
            CHECK_THAT(put_ptr->mask(state), RangeEquals({1, 1, 0, 0}));

            indices_ptr->set_value(state, 1, 0);
            graph.propagate(state, graph.descendants(state, {indices_ptr}));
            CHECK_THAT(put_ptr->view(state), RangeEquals({2, 0, 0, 0}));
            CHECK_THAT(put_ptr->mask(state), RangeEquals({2, 0, 0, 0}));

            graph.revert(state, graph.descendants(state, {indices_ptr}));
            CHECK_THAT(put_ptr->view(state), RangeEquals({2, 2, 0, 0}));
            CHECK_THAT(put_ptr->mask(state), RangeEquals({1, 1, 0, 0}));

            indices_ptr->set_value(state, 1, 0);
            graph.propagate(state, graph.descendants(state, {indices_ptr}));
            CHECK_THAT(put_ptr->view(state), RangeEquals({2, 0, 0, 0}));
            CHECK_THAT(put_ptr->mask(state), RangeEquals({2, 0, 0, 0}));

            graph.commit(state, graph.descendants(state, {indices_ptr}));
            CHECK_THAT(put_ptr->view(state), RangeEquals({2, 0, 0, 0}));
            CHECK_THAT(put_ptr->mask(state), RangeEquals({2, 0, 0, 0}));
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

                CHECK(B.max() == A.max());
                CHECK(B.min() == A.min());
                CHECK(B.integral() == A.integral());
            }
        }

        WHEN("It is reshaped without specifying the size of axis 0") {
            auto B = ReshapeNode(&A, {-1});

            THEN("It has the shape/size/etc we expect") {
                CHECK(B.ndim() == 1);
                CHECK(std::ranges::equal(B.shape(), std::vector{12}));

                CHECK(B.max() == A.max());
                CHECK(B.min() == A.min());
                CHECK(B.integral() == A.integral());
            }
        }

        WHEN("We reshape it into a 3x4 array explicitly") {
            auto B = ReshapeNode(&A, {3, 4});

            THEN("It has the shape/size/etc we expect") {
                CHECK(B.ndim() == 2);
                CHECK(std::ranges::equal(B.shape(), std::vector{3, 4}));

                CHECK(B.max() == A.max());
                CHECK(B.min() == A.min());
                CHECK(B.integral() == A.integral());
            }
        }

        WHEN("We reshape it into a 3x4 array implicitly") {
            auto B = ReshapeNode(&A, {3, -1});

            THEN("It has the shape/size/etc we expect") {
                CHECK(B.ndim() == 2);
                CHECK(std::ranges::equal(B.shape(), std::vector{3, 4}));

                CHECK(B.max() == A.max());
                CHECK(B.min() == A.min());
                CHECK(B.integral() == A.integral());
            }
        }

        WHEN("We try to use more than one undefined axis") {
            CHECK_THROWS_AS(ReshapeNode(&A, {2, -1, -1}), std::invalid_argument);
            CHECK_THROWS_AS(ReshapeNode(&A, {-1, 2, -1}), std::invalid_argument);
            CHECK_THROWS_AS(ReshapeNode(&A, {12, -1, -1}), std::invalid_argument);
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

    GIVEN("A set(10) reshaped to shape (-1,1)") {
        auto graph = Graph();

        auto set_ptr = graph.emplace_node<SetNode>(10);
        auto reshape_ptr = graph.emplace_node<ReshapeNode>(set_ptr, std::array{-1, 1});
        graph.emplace_node<ArrayValidationNode>(reshape_ptr);

        CHECK_THAT(reshape_ptr->shape(), RangeEquals({-1, 1}));

        CHECK(reshape_ptr->sizeinfo() == SizeInfo(set_ptr, 0, 10));

        auto state = graph.empty_state();
        set_ptr->initialize_state(state, {0, 1, 2, 3});  // size 4
        graph.initialize_state(state);
        CHECK_THAT(reshape_ptr->shape(state), RangeEquals({4, 1}));

        set_ptr->grow(state);
        graph.propagate(state);
        CHECK_THAT(reshape_ptr->shape(state), RangeEquals({5, 1}));

        AND_WHEN("We commit") {
            graph.commit(state);
            CHECK_THAT(reshape_ptr->shape(state), RangeEquals({5, 1}));
        }
        AND_WHEN("We revert") {
            graph.revert(state);
            CHECK_THAT(reshape_ptr->shape(state), RangeEquals({4, 1}));
        }
    }

    // todo: once we support BroadcastToNode we can test some other more interesting reshapes

    GIVEN("A set(10) that we try to do several invalid reshapes on") {
        auto graph = Graph();
        auto set_ptr = graph.emplace_node<SetNode>(10);

        CHECK_THROWS_AS(graph.emplace_node<ReshapeNode>(set_ptr, std::array{-1, 2}),
                        std::invalid_argument);

        CHECK_THROWS_AS(graph.emplace_node<ReshapeNode>(set_ptr, std::array{-1, 1, -1, 1}),
                        std::invalid_argument);

        CHECK(graph.num_nodes() == 1);  // no side effects
    }
}

TEST_CASE("ResizeNode") {
    GIVEN("A set(10) resized to size 5") {
        auto graph = Graph();

        auto set_ptr = graph.emplace_node<SetNode>(10);
        auto resize_ptr = graph.emplace_node<ResizeNode>(set_ptr, std::array{5}, 15.1);
        graph.emplace_node<ArrayValidationNode>(resize_ptr);

        CHECK(resize_ptr->min() == 0);
        CHECK(resize_ptr->max() == 15.1);  // the fill value
        CHECK(!resize_ptr->integral());

        WHEN("The set is smaller than the desired size") {
            auto state = graph.empty_state();
            set_ptr->initialize_state(state, {0, 1, 2});
            graph.initialize_state(state);

            CHECK_THAT(resize_ptr->view(state), RangeEquals({0., 1., 2., 15.1, 15.1}));

            AND_WHEN("We grow the set past the size of our resize") {
                set_ptr->assign(state, {0, 1, 2, 3, 4, 5, 6});  // larger than our window
                graph.propagate(state);

                CHECK_THAT(resize_ptr->view(state), RangeEquals({0, 1, 2, 3, 4}));
            }
        }

        WHEN("The set is larger than the desired size") {
            auto state = graph.empty_state();
            set_ptr->initialize_state(state, {0, 1, 2, 3, 4, 5, 6});
            graph.initialize_state(state);

            CHECK_THAT(resize_ptr->view(state), RangeEquals({0, 1, 2, 3, 4}));

            AND_WHEN("We shrink the set down to less than our size") {
                set_ptr->assign(state, {0, 1});  // smaller than our window
                graph.propagate(state);

                CHECK_THAT(resize_ptr->view(state), RangeEquals({0., 1., 15.1, 15.1, 15.1}));
            }
        }
    }

    GIVEN("A set(10, 5, 10) resized to size 5") {
        auto graph = Graph();

        auto set_ptr = graph.emplace_node<SetNode>(10, 5, 10);
        auto resize_ptr = graph.emplace_node<ResizeNode>(set_ptr, std::array{5}, 15.1);
        graph.emplace_node<ArrayValidationNode>(resize_ptr);

        CHECK(resize_ptr->min() == 0);
        CHECK(resize_ptr->max() == 9);  // doesn't use the fill value
        CHECK(resize_ptr->integral());  // doesn't use the fill value
    }

    GIVEN("A constant(range(5)) resized to size 2x2") {
        auto graph = Graph();

        auto c_ptr = graph.emplace_node<ConstantNode>(std::array{0, 1, 2, 3, 4});
        auto resize_ptr = graph.emplace_node<ResizeNode>(c_ptr, std::array{2, 2}, -1);
        graph.emplace_node<ArrayValidationNode>(resize_ptr);

        CHECK(resize_ptr->min() == 0);  // doesn't use the fill value
        CHECK(resize_ptr->max() == 4);  // doesn't use the fill value,
                                        // also not smart enough to know it's a constant
        CHECK(resize_ptr->integral());  // doesn't use the fill value

        CHECK_THAT(resize_ptr->shape(), RangeEquals({2, 2}));

        auto state = graph.initialize_state();
        CHECK_THAT(resize_ptr->view(state), RangeEquals({0, 1, 2, 3}));
    }

    GIVEN("A constant(range(5)) resized to size 3x2") {
        auto graph = Graph();

        auto c_ptr = graph.emplace_node<ConstantNode>(std::array{0, 1, 2, 3, 4});
        auto resize_ptr = graph.emplace_node<ResizeNode>(c_ptr, std::array{3, 2}, -1);
        graph.emplace_node<ArrayValidationNode>(resize_ptr);

        CHECK(resize_ptr->min() == -1);  // always has a fill value
        CHECK(resize_ptr->max() == 4);
        CHECK_THAT(resize_ptr->shape(), RangeEquals({3, 2}));

        auto state = graph.initialize_state();
        CHECK_THAT(resize_ptr->view(state), RangeEquals({0, 1, 2, 3, 4, -1}));
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

                // These should be one less than the min/max of the original set node
                CHECK(len.min() == 1.0);
                CHECK(len.max() == 3.0);
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

TEST_CASE("TransposeNode") {
    auto graph = Graph();

    GIVEN("A dynamic vector and a transpose node") {
        auto vec_ptr =
                graph.emplace_node<DynamicArrayTestingNode>(std::initializer_list<ssize_t>{-1});
        auto transpose_ptr = graph.emplace_node<TransposeNode>(vec_ptr);

        THEN("The basic attributes of the transpose node are correct") {
            CHECK_THAT(transpose_ptr->strides(), RangeEquals({8}));
            CHECK_THAT(transpose_ptr->shape(), RangeEquals({-1}));
            CHECK(transpose_ptr->min() == vec_ptr->min());
            CHECK(transpose_ptr->max() == vec_ptr->max());
            CHECK(transpose_ptr->integral() == vec_ptr->integral());
            CHECK(transpose_ptr->contiguous() == vec_ptr->contiguous());
        }

        WHEN("We initialize an empty state") {
            auto state = graph.empty_state();
            vec_ptr->initialize_state(state, {});
            graph.initialize_state(state);

            THEN("The initial transpose is correct") { CHECK(transpose_ptr->size(state) == 0); }
        }

        WHEN("We initialize a state") {
            auto state = graph.empty_state();
            vec_ptr->initialize_state(state, {0, 1});
            graph.initialize_state(state);

            THEN("The initial transpose is correct") {
                CHECK_THAT(transpose_ptr->view(state), RangeEquals({0, 1}));
            }

            AND_WHEN("We make some changes to the integer node and propagate") {
                vec_ptr->grow(state, std::vector<double>{3});
                // vec_ptr should now be {0, 1, 3}

                graph.propagate(state);

                THEN("The transpose is correct") {
                    CHECK_THAT(transpose_ptr->view(state), RangeEquals({0, 1, 3}));
                    CHECK_THAT(transpose_ptr->diff(state), RangeEquals({Update(2, NAN, 3)}));
                    CHECK(transpose_ptr->size_diff(state) == 1);
                }

                AND_WHEN("We commit, make changes to integer node, and propagate") {
                    graph.commit(state);
                    vec_ptr->shrink(state);
                    vec_ptr->shrink(state);
                    // vec_ptr should now be {0}

                    graph.propagate(state);

                    THEN("The transpose is correct") {
                        CHECK_THAT(transpose_ptr->view(state), RangeEquals({0}));
                        CHECK_THAT(transpose_ptr->diff(state),
                                   RangeEquals({Update(2, 3, NAN), Update(1, 1, NAN)}));
                        CHECK(transpose_ptr->size_diff(state) == -2);
                    }

                    AND_WHEN("We revert") {
                        graph.revert(state);

                        THEN("The transpose is correct") {
                            CHECK_THAT(transpose_ptr->view(state), RangeEquals({0, 1, 3}));
                        }
                    }
                }
            }
        }
    }

    GIVEN("A dynamic (>=2)-D array and a transpose node") {
        auto arr_ptr = graph.emplace_node<DynamicArrayTestingNode>(
                std::initializer_list<ssize_t>{-1, 3, 5, 4});
        REQUIRE_THROWS(graph.emplace_node<TransposeNode>(arr_ptr));
    }

    GIVEN("An integer vector and a transpose node") {
        auto i_ptr = graph.emplace_node<IntegerNode>(5);
        auto transpose_ptr = graph.emplace_node<TransposeNode>(i_ptr);

        THEN("The basic attributes of the transpose node are correct") {
            CHECK_THAT(transpose_ptr->strides(), RangeEquals({8}));
            CHECK_THAT(transpose_ptr->shape(), RangeEquals({5}));
            CHECK(transpose_ptr->min() == i_ptr->min());
            CHECK(transpose_ptr->max() == i_ptr->max());
            CHECK(transpose_ptr->integral() == i_ptr->integral());
            CHECK(transpose_ptr->contiguous() == i_ptr->contiguous());
        }

        WHEN("We initialize a state") {
            auto state = graph.empty_state();
            i_ptr->initialize_state(state, {0, 1, 2, 3, 4});
            graph.initialize_state(state);

            THEN("The initial transpose state is correct") {
                CHECK_THAT(transpose_ptr->view(state), RangeEquals({0, 1, 2, 3, 4}));
            }

            AND_WHEN("We make some changes to the integer node and propagate") {
                i_ptr->set_value(state, 1, 6);
                i_ptr->set_value(state, 4, 7);
                // i_ptr should now be {0, 6, 2, 3, 7}

                graph.propagate(state);

                THEN("The transpose is correct") {
                    CHECK_THAT(transpose_ptr->view(state), RangeEquals({0, 6, 2, 3, 7}));
                    CHECK_THAT(transpose_ptr->diff(state),
                               RangeEquals({Update(1, 1, 6), Update(4, 4, 7)}));
                }

                AND_WHEN("We commit, make changes to integer node, and propagate") {
                    graph.commit(state);
                    i_ptr->set_value(state, 0, 8);
                    // i_ptr should now be {8, 6, 2, 3, 7}

                    graph.propagate(state);

                    THEN("The transpose is correct") {
                        CHECK_THAT(transpose_ptr->view(state), RangeEquals({8, 6, 2, 3, 7}));
                        CHECK_THAT(transpose_ptr->diff(state), RangeEquals({Update(0, 0, 8)}));
                    }

                    AND_WHEN("We revert") {
                        graph.revert(state);

                        THEN("The transpose is correct") {
                            CHECK_THAT(transpose_ptr->view(state), RangeEquals({0, 6, 2, 3, 7}));
                        }
                    }
                }
            }
        }
    }

    GIVEN("An integer 2-D array and a transpose node") {
        auto array_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{2, 3});
        auto transpose_ptr = graph.emplace_node<TransposeNode>(array_ptr);

        THEN("The basic attributes of the transpose node are correct") {
            CHECK_THAT(transpose_ptr->strides(), RangeEquals({8, 24}));
            CHECK_THAT(transpose_ptr->shape(), RangeEquals({3, 2}));
            CHECK(transpose_ptr->min() == array_ptr->min());
            CHECK(transpose_ptr->max() == array_ptr->max());
            CHECK(transpose_ptr->integral() == array_ptr->integral());
            CHECK(!transpose_ptr->contiguous());
        }

        WHEN("We initialize a state") {
            auto state = graph.empty_state();
            array_ptr->initialize_state(state, {0, 1, 2, 3, 4, 5});
            // | 0 1 2 |T    | 0 3 |
            // | 3 4 5 |  -> | 1 4 |
            //               | 2 5 |
            graph.initialize_state(state);

            THEN("The initial transpose state is correct") {
                CHECK_THAT(transpose_ptr->view(state), RangeEquals({0, 3, 1, 4, 2, 5}));
            }

            AND_THEN("We make some changes to the array node and propagate") {
                array_ptr->set_value(state, 1, 6);
                array_ptr->set_value(state, 4, 7);
                array_ptr->set_value(state, 5, 8);
                // | 0 6 2 |T    | 0 3 |
                // | 3 7 8 |  -> | 6 7 |
                //               | 2 8 |
                graph.propagate(state);

                THEN("The transpose state is correct") {
                    CHECK_THAT(transpose_ptr->view(state), RangeEquals({0, 3, 6, 7, 2, 8}));
                    CHECK_THAT(transpose_ptr->diff(state),
                               RangeEquals({Update(2, 1, 6), Update(3, 4, 7), Update(5, 5, 8)}));
                }

                AND_WHEN("We commit, make changes to integer node, and propagate") {
                    graph.commit(state);
                    array_ptr->set_value(state, 0, 9);
                    array_ptr->set_value(state, 2, 10);
                    // | 9 6 2 |T    | 9 3 |
                    // | 3 7 8 |  -> | 6 7 |
                    //               | 10 8 |

                    graph.propagate(state);

                    THEN("The transpose is correct") {
                        CHECK_THAT(transpose_ptr->view(state), RangeEquals({9, 3, 6, 7, 10, 8}));
                        CHECK_THAT(transpose_ptr->diff(state),
                                   RangeEquals({Update(0, 0, 9), Update(4, 2, 10)}));
                    }

                    AND_WHEN("We revert") {
                        graph.revert(state);

                        THEN("The transpose is correct") {
                            CHECK_THAT(transpose_ptr->view(state), RangeEquals({0, 3, 6, 7, 2, 8}));
                        }
                    }
                }
            }
        }
    }

    GIVEN("An integer 2-D array and TWO transpose node") {
        auto array_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{2, 3});
        auto transpose_ptr_1 = graph.emplace_node<TransposeNode>(array_ptr);
        auto transpose_ptr_2 = graph.emplace_node<TransposeNode>(transpose_ptr_1);

        THEN("The basic attributes of the second transpose node are correct") {
            CHECK_THAT(transpose_ptr_2->strides(), RangeEquals({24, 8}));
            CHECK_THAT(transpose_ptr_2->shape(), RangeEquals({2, 3}));
            CHECK(transpose_ptr_2->contiguous());
        }

        WHEN("We initialize a state") {
            auto state = graph.empty_state();
            array_ptr->initialize_state(state, {0, 1, 2, 3, 4, 5});
            // | 0 1 2 |
            // | 3 4 5 |
            graph.initialize_state(state);

            THEN("The initial second transpose state is correct") {
                CHECK_THAT(transpose_ptr_2->view(state), RangeEquals({0, 1, 2, 3, 4, 5}));
            }

            AND_THEN("We make some changes to the array node and propagate") {
                array_ptr->set_value(state, 1, 6);
                array_ptr->set_value(state, 4, 7);
                array_ptr->set_value(state, 5, 8);
                // | 0 6 2 |
                // | 3 7 8 |

                graph.propagate(state);

                THEN("The second transpose state is correct") {
                    CHECK_THAT(transpose_ptr_2->view(state), RangeEquals({0, 6, 2, 3, 7, 8}));
                    CHECK_THAT(transpose_ptr_2->diff(state),
                               RangeEquals({Update(1, 1, 6), Update(4, 4, 7), Update(5, 5, 8)}));
                }

                AND_WHEN("We commit, make changes to integer node, and propagate") {
                    graph.commit(state);
                    array_ptr->set_value(state, 0, 9);
                    array_ptr->set_value(state, 2, 10);
                    // | 9 6 10 |
                    // | 3 7 8  |

                    graph.propagate(state);

                    THEN("The second transpose is correct") {
                        CHECK_THAT(transpose_ptr_2->view(state), RangeEquals({9, 6, 10, 3, 7, 8}));
                        CHECK_THAT(transpose_ptr_2->diff(state),
                                   RangeEquals({Update(0, 0, 9), Update(2, 2, 10)}));
                    }

                    AND_WHEN("We revert") {
                        graph.revert(state);

                        THEN("The second transpose is correct") {
                            CHECK_THAT(transpose_ptr_2->view(state),
                                       RangeEquals({0, 6, 2, 3, 7, 8}));
                        }
                    }
                }
            }
        }
    }

    GIVEN("An integer 4-D array and a transpose node") {
        auto array_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{2, 2, 2, 3});
        auto transpose_ptr = graph.emplace_node<TransposeNode>(array_ptr);
        // Python code to obtain the outputs because I was unwilling to do this by hand..
        // >>> import numpy as np
        // >>> a = np.arange(24).reshape(2, 2, 2, 3)

        THEN("The basic attributes of the transpose node are correct") {
            // >>> a.transpose().strides
            // (8, 24, 48, 96)
            CHECK_THAT(transpose_ptr->strides(), RangeEquals({8, 24, 48, 96}));
            CHECK_THAT(transpose_ptr->shape(), RangeEquals({3, 2, 2, 2}));
            CHECK(transpose_ptr->min() == array_ptr->min());
            CHECK(transpose_ptr->max() == array_ptr->max());
            CHECK(transpose_ptr->integral() == array_ptr->integral());
            CHECK(!transpose_ptr->contiguous());
        }

        WHEN("We initialize a state") {
            auto state = graph.empty_state();
            array_ptr->initialize_state(state, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
            graph.initialize_state(state);

            // >>> a.transpose().flatten()
            // array([ 0, 12,  6, 18,  3, 15,  9, 21,  1, 13,  7, 19,  4, 16, 10, 22,  2,
            // 14,  8, 20,  5, 17, 11, 23])
            CHECK_THAT(transpose_ptr->view(state),
                       RangeEquals({0, 12, 6,  18, 3, 15, 9, 21, 1, 13, 7,  19,
                                    4, 16, 10, 22, 2, 14, 8, 20, 5, 17, 11, 23}));

            AND_THEN("We make some changes to the array node and propagate") {
                array_ptr->set_value(state, 1, 24);
                array_ptr->set_value(state, 7, 25);
                array_ptr->set_value(state, 10, 26);
                array_ptr->set_value(state, 16, 27);
                array_ptr->set_value(state, 22, 28);
                graph.propagate(state);
                // >>> a = a.flatten()
                // >>> a[1] = 24
                // >>> a[7] = 25
                // >>> a[10] = 26
                // >>> a[16] = 27
                // >>> a[22] = 28
                // >>> a = a.reshape(2,2,2,3)

                THEN("The transpose state is correct") {
                    // >>> a.transpose().flatten()
                    // array([ 0, 12,  6, 18,  3, 15,  9, 21, 24, 13, 25, 19,  4, 27, 26, 28,  2,
                    // 14,  8, 20,  5, 17, 11, 23])
                    CHECK_THAT(transpose_ptr->view(state),
                               RangeEquals({0, 12, 6,  18, 3, 15, 9, 21, 24, 13, 25, 19,
                                            4, 27, 26, 28, 2, 14, 8, 20, 5,  17, 11, 23}));
                    // >>> np.where(a.transpose().flatten() == 24)
                    // (array([8]),)
                    // >>> np.where(a.transpose().flatten() == 25)
                    // (array([10]),)
                    // >>> np.where(a.transpose().flatten() == 26)
                    // (array([14]),)
                    // >>> np.where(a.transpose().flatten() == 27)
                    // (array([13]),)
                    // >>> np.where(a.transpose().flatten() == 28)
                    // (array([15]),)
                    CHECK_THAT(transpose_ptr->diff(state),
                               RangeEquals({Update(8, 1, 24), Update(10, 7, 25), Update(14, 10, 26),
                                            Update(13, 16, 27), Update(15, 22, 28)}));
                }
            }
        }
    }
}

}  // namespace dwave::optimization

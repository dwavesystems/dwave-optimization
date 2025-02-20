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

#include "dwave-optimization/nodes/creation.hpp"
#include "dwave-optimization/nodes/numbers.hpp"
#include "dwave-optimization/nodes/testing.hpp"

using Catch::Matchers::RangeEquals;

namespace dwave::optimization {

TEST_CASE("ARangeNode") {
    GIVEN("arange = ARangeNode()") {
        auto arange = ARangeNode();
        CHECK_THAT(arange.shape(), RangeEquals({0}));
        CHECK_THAT(arange.predecessors(), RangeEquals(std::vector<Node*>{}));

        CHECK(arange.sizeinfo() == SizeInfo(0));
        CHECK(arange.minmax() == std::pair<double, double>(0, 0));
    }

    GIVEN("arange = ARangeNode(5)") {
        auto graph = Graph();
        auto arange_ptr = graph.emplace_node<ARangeNode>(5);
        CHECK_THAT(arange_ptr->shape(), RangeEquals({5}));
        CHECK_THAT(arange_ptr->predecessors(), RangeEquals(std::vector<Node*>{}));

        auto state = graph.initialize_state();
        CHECK_THAT(arange_ptr->view(state), RangeEquals({0, 1, 2, 3, 4}));

        CHECK(arange_ptr->sizeinfo() == SizeInfo(5));
        CHECK(arange_ptr->minmax() == std::pair<double, double>(0, 4));
    }

    GIVEN("arange = ARangeNode(-2, -5, -1)") {
        auto graph = Graph();
        auto arange_ptr = graph.emplace_node<ARangeNode>(-2, -5, -1);
        CHECK_THAT(arange_ptr->shape(), RangeEquals({3}));
        CHECK_THAT(arange_ptr->predecessors(), RangeEquals(std::vector<Node*>{}));

        auto state = graph.initialize_state();
        CHECK_THAT(arange_ptr->view(state), RangeEquals({-2, -3, -4}));

        CHECK(arange_ptr->sizeinfo() == SizeInfo(3));
        CHECK(arange_ptr->minmax() == std::pair<double, double>(-4, -2));
    }

    GIVEN("i = IntegerNode({}, -5, 5), arange = ARangeNode(i)") {
        // arange(-5) => []
        // arange(5) => [0, 1, 2, 3, 4]
        auto graph = Graph();
        auto i_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{}, -5, 5);
        auto arange_ptr = graph.emplace_node<ARangeNode>(i_ptr);

        graph.emplace_node<ArrayValidationNode>(arange_ptr);

        CHECK_THAT(arange_ptr->shape(), RangeEquals({-1}));
        CHECK_THAT(arange_ptr->predecessors(), RangeEquals(std::vector<Node*>{i_ptr}));
        CHECK(arange_ptr->sizeinfo() == SizeInfo(arange_ptr, 0, 5));
        CHECK(arange_ptr->minmax() == std::pair<double, double>(0, 4));

        auto state = graph.empty_state();
        i_ptr->initialize_state(state, {3});
        graph.initialize_state(state);
        CHECK_THAT(arange_ptr->shape(state), RangeEquals({3}));
        CHECK_THAT(arange_ptr->view(state), RangeEquals({0, 1, 2}));

        WHEN("We change i to be a larger value") {
            i_ptr->set_value(state, 0, 5);
            i_ptr->set_value(state, 0, 4);
            graph.propagate(state, graph.descendants(state, {i_ptr}));

            CHECK_THAT(arange_ptr->shape(state), RangeEquals({4}));
            CHECK_THAT(arange_ptr->view(state), RangeEquals({0, 1, 2, 3}));

            AND_WHEN("We commit") {
                graph.commit(state, graph.descendants(state, {i_ptr}));

                CHECK_THAT(arange_ptr->shape(state), RangeEquals({4}));
                CHECK_THAT(arange_ptr->view(state), RangeEquals({0, 1, 2, 3}));
            }

            AND_WHEN("We revert") {
                graph.revert(state, graph.descendants(state, {i_ptr}));

                CHECK_THAT(arange_ptr->shape(state), RangeEquals({3}));
                CHECK_THAT(arange_ptr->view(state), RangeEquals({0, 1, 2}));
            }
        }

        WHEN("We change i to be a smaller value") {
            i_ptr->set_value(state, 0, 5);
            i_ptr->set_value(state, 0, 2);
            graph.propagate(state, graph.descendants(state, {i_ptr}));

            CHECK_THAT(arange_ptr->shape(state), RangeEquals({2}));
            CHECK_THAT(arange_ptr->view(state), RangeEquals({0, 1}));

            AND_WHEN("We commit") {
                graph.commit(state, graph.descendants(state, {i_ptr}));

                CHECK_THAT(arange_ptr->shape(state), RangeEquals({2}));
                CHECK_THAT(arange_ptr->view(state), RangeEquals({0, 1}));
            }

            AND_WHEN("We revert") {
                graph.revert(state, graph.descendants(state, {i_ptr}));

                CHECK_THAT(arange_ptr->shape(state), RangeEquals({3}));
                CHECK_THAT(arange_ptr->view(state), RangeEquals({0, 1, 2}));
            }
        }
    }

    GIVEN("i = IntegerNode({}, -7, 5), arange = ARangeNode(0, i, -1)") {
        // arange(0, -7, -1) => [0, -1, -2, -3, -4, -5, -6]
        // arange(0, 5, -1) => []
        auto graph = Graph();
        auto i_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{}, -7, 5);
        auto arange_ptr = graph.emplace_node<ARangeNode>(0, i_ptr, -1);

        graph.emplace_node<ArrayValidationNode>(arange_ptr);

        CHECK_THAT(arange_ptr->shape(), RangeEquals({-1}));
        CHECK_THAT(arange_ptr->predecessors(), RangeEquals(std::vector<Node*>{i_ptr}));

        CHECK(arange_ptr->sizeinfo() == SizeInfo(arange_ptr, 0, 7));
        CHECK(arange_ptr->minmax() == std::pair<double, double>(-6, 0));

        auto state = graph.empty_state();
        i_ptr->initialize_state(state, {-5});
        graph.initialize_state(state);
        CHECK_THAT(arange_ptr->shape(state), RangeEquals({5}));
        CHECK_THAT(arange_ptr->view(state), RangeEquals({0, -1, -2, -3, -4}));

        WHEN("We change i to be a larger value") {
            i_ptr->set_value(state, 0, -3);
            graph.propagate(state, graph.descendants(state, {i_ptr}));

            CHECK_THAT(arange_ptr->shape(state), RangeEquals({3}));
            CHECK_THAT(arange_ptr->view(state), RangeEquals({0, -1, -2}));

            AND_WHEN("We commit") {
                graph.commit(state, graph.descendants(state, {i_ptr}));

                CHECK_THAT(arange_ptr->shape(state), RangeEquals({3}));
                CHECK_THAT(arange_ptr->view(state), RangeEquals({0, -1, -2}));
            }

            AND_WHEN("We revert") {
                graph.revert(state, graph.descendants(state, {i_ptr}));

                CHECK_THAT(arange_ptr->shape(state), RangeEquals({5}));
                CHECK_THAT(arange_ptr->view(state), RangeEquals({0, -1, -2, -3, -4}));
            }
        }

        WHEN("We change i to be a smaller value") {
            i_ptr->set_value(state, 0, -7);
            graph.propagate(state, graph.descendants(state, {i_ptr}));

            CHECK_THAT(arange_ptr->shape(state), RangeEquals({7}));
            CHECK_THAT(arange_ptr->view(state), RangeEquals({0, -1, -2, -3, -4, -5, -6}));

            AND_WHEN("We commit") {
                graph.commit(state, graph.descendants(state, {i_ptr}));

                CHECK_THAT(arange_ptr->shape(state), RangeEquals({7}));
                CHECK_THAT(arange_ptr->view(state), RangeEquals({0, -1, -2, -3, -4, -5, -6}));
            }

            AND_WHEN("We revert") {
                graph.revert(state, graph.descendants(state, {i_ptr}));

                CHECK_THAT(arange_ptr->shape(state), RangeEquals({5}));
                CHECK_THAT(arange_ptr->view(state), RangeEquals({0, -1, -2, -3, -4}));
            }
        }

        WHEN("We change i to be a positive value") {
            i_ptr->set_value(state, 0, 5);
            graph.propagate(state, graph.descendants(state, {i_ptr}));

            CHECK_THAT(arange_ptr->shape(state), RangeEquals({0}));
            CHECK_THAT(arange_ptr->view(state), RangeEquals(std::vector<double>{}));

            AND_WHEN("We commit") {
                graph.commit(state, graph.descendants(state, {i_ptr}));

                CHECK_THAT(arange_ptr->shape(state), RangeEquals({0}));
                CHECK_THAT(arange_ptr->view(state), RangeEquals(std::vector<double>{}));
            }

            AND_WHEN("We revert") {
                graph.revert(state, graph.descendants(state, {i_ptr}));

                CHECK_THAT(arange_ptr->shape(state), RangeEquals({5}));
                CHECK_THAT(arange_ptr->view(state), RangeEquals({0, -1, -2, -3, -4}));
            }
        }
    }

    GIVEN("i = IntegerNode({}, -5, 5), arange = ARangeNode(i, 5)") {
        // arange(-5, 5) => [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
        // arange(5, 5) => []
        auto graph = Graph();
        auto i_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{}, -5, 5);
        auto arange_ptr = graph.emplace_node<ARangeNode>(i_ptr, 5);

        graph.emplace_node<ArrayValidationNode>(arange_ptr);

        CHECK_THAT(arange_ptr->shape(), RangeEquals({-1}));
        CHECK_THAT(arange_ptr->predecessors(), RangeEquals(std::vector<Node*>{i_ptr}));

        CHECK(arange_ptr->sizeinfo() == SizeInfo(arange_ptr, 0, 10));
        CHECK(arange_ptr->minmax() == std::pair<double, double>(-5, 4));

        auto state = graph.empty_state();
        i_ptr->initialize_state(state, {3});
        graph.initialize_state(state);
        CHECK_THAT(arange_ptr->shape(state), RangeEquals({2}));
        CHECK_THAT(arange_ptr->view(state), RangeEquals({3, 4}));

        WHEN("We change i to be a larger value") {
            i_ptr->set_value(state, 0, 5);
            i_ptr->set_value(state, 0, 4);
            graph.propagate(state, graph.descendants(state, {i_ptr}));

            CHECK_THAT(arange_ptr->shape(state), RangeEquals({1}));
            CHECK_THAT(arange_ptr->view(state), RangeEquals({4}));

            AND_WHEN("We commit") {
                graph.commit(state, graph.descendants(state, {i_ptr}));

                CHECK_THAT(arange_ptr->shape(state), RangeEquals({1}));
                CHECK_THAT(arange_ptr->view(state), RangeEquals({4}));
            }

            AND_WHEN("We revert") {
                graph.revert(state, graph.descendants(state, {i_ptr}));

                CHECK_THAT(arange_ptr->shape(state), RangeEquals({2}));
                CHECK_THAT(arange_ptr->view(state), RangeEquals({3, 4}));
            }
        }

        WHEN("We change i to be a smaller value") {
            i_ptr->set_value(state, 0, 2);
            graph.propagate(state, graph.descendants(state, {i_ptr}));

            CHECK_THAT(arange_ptr->shape(state), RangeEquals({3}));
            CHECK_THAT(arange_ptr->view(state), RangeEquals({2, 3, 4}));

            AND_WHEN("We commit") {
                graph.commit(state, graph.descendants(state, {i_ptr}));

                CHECK_THAT(arange_ptr->shape(state), RangeEquals({3}));
                CHECK_THAT(arange_ptr->view(state), RangeEquals({2, 3, 4}));
            }

            AND_WHEN("We revert") {
                graph.revert(state, graph.descendants(state, {i_ptr}));

                CHECK_THAT(arange_ptr->shape(state), RangeEquals({2}));
                CHECK_THAT(arange_ptr->view(state), RangeEquals({3, 4}));
            }
        }
    }

    GIVEN("stop = IntegerNode(-5, 5), step = IntegerNode(1, 2), arange=ARangeNode(0, i, j)") {
        // arange(0, -5, 1) => []
        // arange(0, 5, 1) => [0, 1, 2, 3, 4]
        // arange(0, -5, 2) => []
        // arange(0, 5, 2) => [0, 2, 4]
        auto graph = Graph();
        auto stop_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{}, -5, 5);
        auto step_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{}, 1, 2);
        auto arange_ptr = graph.emplace_node<ARangeNode>(0, stop_ptr, step_ptr);

        graph.emplace_node<ArrayValidationNode>(arange_ptr);

        CHECK_THAT(arange_ptr->shape(), RangeEquals({-1}));
        CHECK_THAT(arange_ptr->predecessors(), RangeEquals(std::vector<Node*>{stop_ptr, step_ptr}));

        CHECK(arange_ptr->sizeinfo() == SizeInfo(arange_ptr, 0, 5));
        CHECK(arange_ptr->minmax() == std::pair<double, double>(0, 4));

        auto state = graph.empty_state();
        stop_ptr->initialize_state(state, {5});
        step_ptr->initialize_state(state, {1});
        graph.initialize_state(state);
        CHECK_THAT(arange_ptr->shape(state), RangeEquals({5}));
        CHECK_THAT(arange_ptr->view(state), RangeEquals({0, 1, 2, 3, 4}));

        WHEN("We change step to be a larger value") {
            step_ptr->set_value(state, 0, 2);
            graph.propagate(state, graph.descendants(state, {step_ptr}));

            CHECK_THAT(arange_ptr->shape(state), RangeEquals({3}));
            CHECK_THAT(arange_ptr->view(state), RangeEquals({0, 2, 4}));

            AND_WHEN("We commit") {
                graph.commit(state, graph.descendants(state, {step_ptr}));

                CHECK_THAT(arange_ptr->shape(state), RangeEquals({3}));
                CHECK_THAT(arange_ptr->view(state), RangeEquals({0, 2, 4}));
            }

            AND_WHEN("We revert") {
                graph.revert(state, graph.descendants(state, {step_ptr}));

                CHECK_THAT(arange_ptr->shape(state), RangeEquals({5}));
                CHECK_THAT(arange_ptr->view(state), RangeEquals({0, 1, 2, 3, 4}));
            }
        }
    }

    GIVEN("start in {0, 1}, arange = ARangeNode(start, 15, 7)") {
        // arange(0, 15, 7) => [0, 7, 14]
        // arange(1, 15, 7) => [1, 8]
        auto graph = Graph();
        auto start_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{}, 0, 1);

        auto arange_ptr = graph.emplace_node<ARangeNode>(start_ptr, 15, 7);

        THEN("The min/max are correct") {
            const auto [low, high] = arange_ptr->minmax();
            CHECK(low == 0);
            CHECK(high == 14);
        }
    }

    GIVEN("start in {0, 1}, arange = ARangeNode(start, 15, 10)") {
        // arange(0, 15, 10) => [1, 11]
        // arange(1, 15, 10) => [0, 10]

        auto graph = Graph();
        auto start_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{}, 0, 1);

        auto arange_ptr = graph.emplace_node<ARangeNode>(start_ptr, 15, 10);

        THEN("The min/max are correct") {
            const auto [low, high] = arange_ptr->minmax();
            CHECK(low == 0);
            CHECK(high == 11);
        }
    }

    GIVEN("step in {3, 14}, arange = ARangeNode(0, 15, step)") {
        // arange(0, 15, 3) => [0, 3, 6, 9, 12]
        // arange(0, 15, 14) => [0, 14]

        auto graph = Graph();
        auto step_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{}, 3, 14);

        auto arange_ptr = graph.emplace_node<ARangeNode>(0, 15, step_ptr);

        THEN("The min/max are correct") {
            const auto [low, high] = arange_ptr->minmax();
            CHECK(low == 0);
            CHECK(high == 14);
        }
    }

    GIVEN("start in {0, -1}, arange = ARangeNode(start, -15, -7)") {
        // arange(0, -15, -7) => [0, -7, -14]
        // arange(-1, -15, -7) => [-1, -8]
        auto graph = Graph();
        auto start_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{}, -1, 0);

        auto arange_ptr = graph.emplace_node<ARangeNode>(start_ptr, -15, -7);

        THEN("The min/max are correct") {
            const auto [low, high] = arange_ptr->minmax();
            CHECK(low == -14);
            CHECK(high == 0);
        }
    }
}

}  // namespace dwave::optimization

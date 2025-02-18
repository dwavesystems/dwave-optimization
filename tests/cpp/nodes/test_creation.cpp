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
    }

    GIVEN("arange = ARangeNode(5)") {
        auto graph = Graph();
        auto arange_ptr = graph.emplace_node<ARangeNode>(5);
        CHECK_THAT(arange_ptr->shape(), RangeEquals({5}));
        CHECK_THAT(arange_ptr->predecessors(), RangeEquals(std::vector<Node*>{}));

        auto state = graph.initialize_state();
        CHECK_THAT(arange_ptr->view(state), RangeEquals({0, 1, 2, 3, 4}));
    }

    GIVEN("i = IntegerNode({}, -5, 5), arange = ARangeNode(i)") {
        auto graph = Graph();
        auto i_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{}, -5, 5);
        auto arange_ptr = graph.emplace_node<ARangeNode>(i_ptr);

        graph.emplace_node<ArrayValidationNode>(arange_ptr);

        CHECK_THAT(arange_ptr->shape(), RangeEquals({-1}));
        CHECK_THAT(arange_ptr->predecessors(), RangeEquals(std::vector<Node*>{i_ptr}));

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
        auto graph = Graph();
        auto i_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{}, -7, 5);
        auto arange_ptr = graph.emplace_node<ARangeNode>(0, i_ptr, -1);

        graph.emplace_node<ArrayValidationNode>(arange_ptr);

        CHECK_THAT(arange_ptr->shape(), RangeEquals({-1}));
        CHECK_THAT(arange_ptr->predecessors(), RangeEquals(std::vector<Node*>{i_ptr}));

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
        auto graph = Graph();
        auto i_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{}, -5, 5);
        auto arange_ptr = graph.emplace_node<ARangeNode>(i_ptr, 5);

        graph.emplace_node<ArrayValidationNode>(arange_ptr);

        CHECK_THAT(arange_ptr->shape(), RangeEquals({-1}));
        CHECK_THAT(arange_ptr->predecessors(), RangeEquals(std::vector<Node*>{i_ptr}));

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
        auto graph = Graph();
        auto stop_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{}, -5, 5);
        auto step_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{}, 1, 2);
        auto arange_ptr = graph.emplace_node<ARangeNode>(0, stop_ptr, step_ptr);

        graph.emplace_node<ArrayValidationNode>(arange_ptr);

        CHECK_THAT(arange_ptr->shape(), RangeEquals({-1}));
        CHECK_THAT(arange_ptr->predecessors(), RangeEquals(std::vector<Node*>{stop_ptr, step_ptr}));

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
}

}  // namespace dwave::optimization

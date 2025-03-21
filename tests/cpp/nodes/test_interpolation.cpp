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

#include "catch2/catch_test_macros.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/nodes/constants.hpp"
#include "dwave-optimization/nodes/interpolation.hpp"
#include "dwave-optimization/nodes/numbers.hpp"

namespace dwave::optimization {

TEST_CASE("BSpline") {

    GIVEN("A simple BSplineNode"){
        auto graph = Graph();
        auto x = ConstantNode(std::vector<double>{2.5});

        int k = 2;
        std::vector<double> t = {0, 1, 2, 3, 4, 5, 6};
        std::vector<double> c = {-1, 2, 0, -1};

        WHEN("We create a BSplineNode"){
            auto bspline = BSplineNode(&x, k, t, c);

            THEN("The BSplineNode is a scalar output") {
                    CHECK(bspline.size() == 1);
                    CHECK(bspline.ndim() == 1);
                    CHECK(!bspline.dynamic());
            }

            THEN("The output is not a integer, we know the min/max and can get the bspline constants") {
                    CHECK(!bspline.integral());
                    CHECK(bspline.min() == -1);
                    CHECK(bspline.max() == 2);
                    CHECK(bspline.k() == k);
                    CHECK(bspline.t() == t);
                    CHECK(bspline.c() == c);
            }
        }
    }

    GIVEN("A graph with a ConstantNode and a corresponding BSplineNode"){
        auto graph = Graph();

        std::vector<double> x = {2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0};
        int k = 2;
        std::vector<double> t = {0, 1, 2, 3, 4, 5, 6};
        std::vector<double> c = {-1, 2, 0, -1};

        auto array_ptr = graph.emplace_node<ConstantNode>(x);
        auto bspline_ptr = graph.emplace_node<BSplineNode>(array_ptr, k, t,  c);

        auto state = graph.empty_state();
        array_ptr->initialize_state(state);
        graph.initialize_state(state);

        THEN("The state of the BSplineNode is as expected" ) {
            std::vector<double> expected = {0.5, 1.09375, 1.375, 1.34375, 1.0, 0.53125, 0.125, -0.21875, -0.5};
            CHECK(std::ranges::equal(bspline_ptr->view(state), expected));
        }
    }

    GIVEN("A graph with a IntegerNode and a corresponding BSplineNode"){
        auto graph = Graph();

        int k = 2;
        std::vector<double> t = {0, 1, 2, 3, 4, 5, 6};
        std::vector<double> c = {-1, 2, 0, -1};

        auto x_ptr = graph.emplace_node<IntegerNode>(std::span<const ssize_t>{}, 2, 4);
        auto bspline_ptr = graph.emplace_node<BSplineNode>(x_ptr, k, t,  c);

        auto state = graph.empty_state();
        x_ptr->initialize_state(state, {2});
        graph.initialize_state(state);

        THEN("The state of the BSplineNode is as expected" ) {
            CHECK(bspline_ptr->view(state)[0] == 0.5);

            AND_WHEN("We change the integer's state and propagate") {
                x_ptr->set_value(state, 0, 3);
                x_ptr->propagate(state);
                bspline_ptr->propagate(state);

                THEN("The output is what we expect") {
                    CHECK(bspline_ptr->view(state)[0] == 1);
                    CHECK(bspline_ptr->diff(state).size() == 1);
                }

            AND_WHEN("We revert") {
                    x_ptr->revert(state);
                    bspline_ptr->revert(state);

                    THEN("The value reverts to the previous") {
                        CHECK(bspline_ptr->view(state)[0] == 0.5);
                    }
                    THEN("The diffs are cleared") { CHECK(bspline_ptr->diff(state).size() == 0); }
                }
            }
        }
    }
}

}  // namespace dwave::optimization

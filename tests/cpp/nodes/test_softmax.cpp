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

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include "dwave-optimization/nodes/constants.hpp"
#include "dwave-optimization/nodes/numbers.hpp"
#include "dwave-optimization/nodes/softmax.hpp"
#include "dwave-optimization/nodes/testing.hpp"

namespace dwave::optimization {

using Catch::Matchers::WithinRel;

TEST_CASE("SoftMaxNode") {
    auto graph = Graph();

    GIVEN("A constant node and a softmax node") {
        auto a_ptr = graph.emplace_node<ConstantNode>(std::vector{-1.0, 0.0, 3.0, 5.0});
        auto softmax_ptr = graph.emplace_node<SoftMaxNode>(a_ptr);
        graph.emplace_node<ArrayValidationNode>(softmax_ptr);

        WHEN("We initialize a state") {
            auto state = graph.initialize_state();

            THEN("The initial softmax state and size is correct") {
                CHECK_THAT(softmax_ptr->view(state)[0], WithinRel(0.00216569646006, 1e-9));
                CHECK_THAT(softmax_ptr->view(state)[1], WithinRel(0.00588697333334, 1e-9));
                CHECK_THAT(softmax_ptr->view(state)[2], WithinRel(0.11824302025266, 1e-9));
                CHECK_THAT(softmax_ptr->view(state)[3], WithinRel(0.87370430995393, 1e-9));
                CHECK(softmax_ptr->size() == 4);
            }
        }
    }

    GIVEN("An integer node and a softmax node") {
        auto i_ptr = graph.emplace_node<IntegerNode>(2);
        auto softmax_ptr = graph.emplace_node<SoftMaxNode>(i_ptr);
        graph.emplace_node<ArrayValidationNode>(softmax_ptr);

        WHEN("We initialize a state") {
            auto state = graph.empty_state();
            i_ptr->initialize_state(state, {2.0, 5.0});
            graph.initialize_state(state);

            THEN("The initial softmax state and size is correct") {
                CHECK_THAT(softmax_ptr->view(state)[0], WithinRel(0.04742587317757, 1e-9));
                CHECK_THAT(softmax_ptr->view(state)[1], WithinRel(0.95257412682243, 1e-9));
                CHECK(softmax_ptr->size() == 2);
            }
            AND_WHEN("We make changes to integer node and propagate") {
                i_ptr->set_value(state, 1, 1.0);
                // i_ptr should be [2.0, 1.0]
                graph.propagate(state);

                THEN("The softmax state is correct") {
                    CHECK_THAT(softmax_ptr->view(state)[0], WithinRel(0.73105857863, 1e-9));
                    CHECK_THAT(softmax_ptr->view(state)[1], WithinRel(0.26894142137, 1e-9));
                }
                AND_WHEN("We commit state, make changes to integer node, and propagate") {
                    graph.commit(state);
                    i_ptr->set_value(state, 0, 4.0);
                    // i_ptr should now be [4.0, 1.0]
                    graph.propagate(state);

                    THEN("The softmax state is correct") {
                        CHECK_THAT(softmax_ptr->view(state)[0], WithinRel(0.95257412682243, 1e-9));
                        CHECK_THAT(softmax_ptr->view(state)[1], WithinRel(0.04742587317757, 1e-9));
                    }
                }
            }
        }
        WHEN("We initialize a state") {
            auto state = graph.empty_state();
            i_ptr->initialize_state(state, {2.0, 5.0});
            graph.initialize_state(state);

            THEN("The initial softmax state and size is correct") {
                CHECK_THAT(softmax_ptr->view(state)[0], WithinRel(0.04742587317757, 1e-9));
                CHECK_THAT(softmax_ptr->view(state)[1], WithinRel(0.95257412682243, 1e-9));
                CHECK(softmax_ptr->size() == 2);
            }
            AND_WHEN(
                    "We commit, make changes to integer node that result in the same "
                    "denominator and propagate") {
                graph.commit(state);
                i_ptr->set_value(state, 0, 5.0);
                i_ptr->set_value(state, 1, 2.0);
                // i_ptr should be [5.0, 2.0]
                graph.propagate(state);

                THEN("The softmax state is correct") {
                    CHECK_THAT(softmax_ptr->view(state)[0], WithinRel(0.95257412682243, 1e-9));
                    CHECK_THAT(softmax_ptr->view(state)[1], WithinRel(0.04742587317757, 1e-9));
                }
            }
        }
    }
    GIVEN("A dynamic array node and a softmax node") {
        auto dyn_ptr = graph.emplace_node<DynamicArrayTestingNode>(
                std::initializer_list<ssize_t>{-1}, -10.0, 10.0, false);
        auto softmax_ptr = graph.emplace_node<SoftMaxNode>(dyn_ptr);

        graph.emplace_node<ArrayValidationNode>(softmax_ptr);

        WHEN("We initialize a state") {
            auto state = graph.empty_state();
            dyn_ptr->initialize_state(state, {3.0, -3.0, 1.5});
            // array should be [3.0, -3.0, 1.5]
            graph.initialize_state(state);

            THEN("The initial softmax state and size is correct") {
                CHECK_THAT(softmax_ptr->view(state)[0], WithinRel(0.81592095973169, 1e-9));
                CHECK_THAT(softmax_ptr->view(state)[1], WithinRel(0.00202246585492, 1e-9));
                CHECK_THAT(softmax_ptr->view(state)[2], WithinRel(0.18205657441339, 1e-9));
                CHECK(softmax_ptr->size() == -1);
                CHECK(softmax_ptr->size(state) == 3);
            }
            AND_WHEN("We change array values, shrink and grow array, and then propagate") {
                dyn_ptr->set(state, 0, 2.1);
                dyn_ptr->shrink(state);
                dyn_ptr->grow(state, {1.0, 2.1});
                // array should be [2.1, -3.0, 1.0, 2.1]
                graph.propagate(state);

                THEN("The softmax state and size is correct") {
                    CHECK_THAT(softmax_ptr->view(state)[0], WithinRel(0.42753901403052, 1e-9));
                    CHECK_THAT(softmax_ptr->view(state)[1], WithinRel(0.00260659701541, 1e-9));
                    CHECK_THAT(softmax_ptr->view(state)[2], WithinRel(0.14231537492355, 1e-9));
                    CHECK_THAT(softmax_ptr->view(state)[3], WithinRel(0.42753901403052, 1e-9));
                    CHECK(softmax_ptr->size(state) == 4);
                }
                AND_WHEN("We revert") {
                    graph.revert(state, graph.descendants(state, {dyn_ptr}));
                    // array should be [3.0, -3.0, 1.5]

                    THEN("The softmax state and size are correct") {
                        CHECK_THAT(softmax_ptr->view(state)[0], WithinRel(0.81592095973169, 1e-9));
                        CHECK_THAT(softmax_ptr->view(state)[1], WithinRel(0.00202246585492, 1e-9));
                        CHECK_THAT(softmax_ptr->view(state)[2], WithinRel(0.18205657441339, 1e-9));
                        CHECK(softmax_ptr->size(state) == 3);
                    }
                    AND_WHEN("We make changes to array values") {
                        dyn_ptr->set(state, 0, 1.01);
                        dyn_ptr->set(state, 1, -2.0);
                        // array should be [1.01, -2.0, 1.5]
                        graph.propagate(state);

                        THEN("The softmax state is correct") {
                            CHECK_THAT(softmax_ptr->view(state)[0],
                                       WithinRel(0.37291059609381, 1e-9));
                            CHECK_THAT(softmax_ptr->view(state)[1],
                                       WithinRel(0.01838138930903, 1e-9));
                            CHECK_THAT(softmax_ptr->view(state)[2],
                                       WithinRel(0.60870801459716, 1e-9));
                        }
                    }
                }
            }
        }
        WHEN("We initialize a state") {
            auto state = graph.empty_state();
            dyn_ptr->initialize_state(state, {-2.0, 0.9, 1.5});
            graph.initialize_state(state);

            AND_WHEN("We shrink array and then propagate") {
                dyn_ptr->shrink(state);
                // array should be [-2.0, 0.9]
                graph.propagate(state);

                THEN("The softmax state and size are correct") {
                    CHECK_THAT(softmax_ptr->view(state)[0], WithinRel(0.05215356307842, 1e-9));
                    CHECK_THAT(softmax_ptr->view(state)[1], WithinRel(0.94784643692158, 1e-9));
                    CHECK(softmax_ptr->size(state) == 2);
                }
                AND_WHEN("We commit state, grow array, and then propagate") {
                    graph.commit(state);
                    dyn_ptr->grow(state, {1.7, -3.33});
                    // array should be [-2.0, 0.9, 1.7, -3.33]
                    graph.propagate(state);

                    THEN("The softmax state and size are correct") {
                        CHECK_THAT(softmax_ptr->view(state)[0], WithinRel(0.01669841397218, 1e-9));
                        CHECK_THAT(softmax_ptr->view(state)[1], WithinRel(0.30347940296945, 1e-9));
                        CHECK_THAT(softmax_ptr->view(state)[2], WithinRel(0.67540583226297, 1e-9));
                        CHECK_THAT(softmax_ptr->view(state)[3], WithinRel(0.00441635079541, 1e-9));
                        CHECK(softmax_ptr->size(state) == 4);
                    }
                    AND_WHEN(
                            "We commit state, repeatedly change the same index in array, and then "
                            "propagate") {
                        graph.commit(state);
                        dyn_ptr->set(state, 0, 1.1);
                        dyn_ptr->set(state, 0, 1.2);
                        dyn_ptr->set(state, 0, 1.3);
                        dyn_ptr->set(state, 0, 1.4);
                        dyn_ptr->set(state, 0, 1.5);
                        // array should be [1.5, 0.9, 1.7, -3.33]
                        graph.propagate(state);

                        THEN("The softmax state and size are correct") {
                            CHECK_THAT(softmax_ptr->view(state)[0],
                                       WithinRel(0.35994516970087, 1e-9));
                            CHECK_THAT(softmax_ptr->view(state)[1],
                                       WithinRel(0.19754209748767, 1e-9));
                            CHECK_THAT(softmax_ptr->view(state)[2],
                                       WithinRel(0.43963802305907, 1e-9));
                            CHECK_THAT(softmax_ptr->view(state)[3],
                                       WithinRel(0.00287470975239, 1e-9));
                            CHECK(softmax_ptr->size(state) == 4);
                        }
                    }
                }
            }
        }
    }
}
}  // namespace dwave::optimization

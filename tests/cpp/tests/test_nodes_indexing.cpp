// Copyright 2023 D-Wave Inc.
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

#include "catch2/catch.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/nodes/collections.hpp"
#include "dwave-optimization/nodes/constants.hpp"
#include "dwave-optimization/nodes/indexing.hpp"
#include "dwave-optimization/nodes/mathematical.hpp"
#include "dwave-optimization/nodes/numbers.hpp"
#include "dwave-optimization/nodes/testing.hpp"
#include "utils.hpp"

namespace dwave::optimization {

TEST_CASE("AdvancedIndexingNode") {
    auto graph = Graph();

    GIVEN("A 2d NxN matrix with two const 1d index arrays") {
        std::vector<double> values = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        auto arr_ptr =
                graph.emplace_node<ConstantNode>(values, std::initializer_list<ssize_t>{3, 3});

        std::vector<double> i = {0, 1, 2};
        auto i_ptr = graph.emplace_node<ConstantNode>(i);

        std::vector<double> j = {1, 2, 0};
        auto j_ptr = graph.emplace_node<ConstantNode>(j);

        WHEN("We access the matrix by the indices") {
            auto out_ptr = graph.emplace_node<AdvancedIndexingNode>(arr_ptr, i_ptr, j_ptr);

            THEN("We get the shape we expect") {
                CHECK(out_ptr->size() == 3);
                CHECK(std::ranges::equal(out_ptr->shape(), std::vector{3}));

                CHECK(array_shape_equal(i_ptr, out_ptr));
                CHECK(array_shape_equal(j_ptr, out_ptr));
            }

            THEN("We see the predecessors we expect") {
                CHECK(std::ranges::equal(out_ptr->predecessors(),
                                         std::vector<Node*>{arr_ptr, i_ptr, j_ptr}));
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
                    CHECK(std::ranges::equal(i_ptr->view(state), std::vector{0, 1, 2}));
                    CHECK(std::ranges::equal(j_ptr->view(state), std::vector{1, 2, 0}));
                    CHECK(std::ranges::equal(out_ptr->view(state), std::vector{1, 5, 6}));
                }
            }
        }
    }

    GIVEN("A 1d length 5 array accessed by a SetNode(5)") {
        std::vector<double> values{4, 3, 2, 1, 0};
        auto A_ptr = graph.emplace_node<ConstantNode>(values);
        auto s_ptr = graph.emplace_node<SetNode>(5);
        auto B_ptr = graph.emplace_node<AdvancedIndexingNode>(A_ptr, s_ptr);

        THEN("The resulting array has the size/shape we expect") {
            CHECK(B_ptr->dynamic());
            CHECK(std::ranges::equal(B_ptr->shape(), std::vector{-1}));
            CHECK(B_ptr->size() == Array::DYNAMIC_SIZE);
            CHECK(std::ranges::equal(B_ptr->strides(), std::vector{sizeof(double)}));
            CHECK(B_ptr->ndim() == 1);

            CHECK(array_shape_equal(B_ptr, s_ptr));
        }

        THEN("We see the min/max/integral we expect") {
            CHECK(B_ptr->min() == 0);
            CHECK(B_ptr->max() == 4);
            CHECK(B_ptr->integral());
        }

        WHEN("We default-initialize the state") {
            auto state = graph.initialize_state();

            THEN("We have the state we expect") {
                CHECK(std::ranges::equal(A_ptr->view(state), values));
                CHECK(std::ranges::equal(s_ptr->view(state), std::vector<double>{}));
                CHECK(std::ranges::equal(B_ptr->view(state), std::vector<double>{}));
            }

            THEN("The resulting array has the same size as the SetNode") {
                CHECK(std::ranges::equal(B_ptr->shape(state), s_ptr->shape(state)));
                CHECK(B_ptr->size(state) == s_ptr->size(state));
            }

            AND_WHEN("We grow the SetNode once") {
                s_ptr->grow(state);
                s_ptr->propagate(state);
                B_ptr->propagate(state);  // so any changes are incorporated

                THEN("The state is updated and the updates are signalled") {
                    CHECK(std::ranges::equal(B_ptr->shape(state), s_ptr->shape(state)));
                    CHECK(B_ptr->size(state) == s_ptr->size(state));

                    CHECK(std::ranges::equal(B_ptr->view(state), std::vector{4}));

                    verify_array_diff({}, {4}, B_ptr->diff(state));
                }

                AND_WHEN("We commit") {
                    s_ptr->commit(state);
                    B_ptr->commit(state);

                    THEN("The values stick, and the diff is cleared") {
                        CHECK(std::ranges::equal(B_ptr->shape(state), s_ptr->shape(state)));
                        CHECK(B_ptr->size(state) == s_ptr->size(state));
                        CHECK(std::ranges::equal(B_ptr->view(state), std::vector{4}));

                        CHECK(B_ptr->size_diff(state) == 0);
                        CHECK(B_ptr->diff(state).size() == 0);
                    }
                }

                AND_WHEN("We revert") {
                    s_ptr->revert(state);
                    B_ptr->revert(state);

                    THEN("We're back to where we started") {
                        CHECK(std::ranges::equal(B_ptr->shape(state), s_ptr->shape(state)));
                        CHECK(B_ptr->size(state) == s_ptr->size(state));
                        CHECK(std::ranges::equal(B_ptr->view(state), std::vector<double>{}));

                        CHECK(B_ptr->size_diff(state) == 0);
                        CHECK(B_ptr->diff(state).size() == 0);
                    }
                }
            }

            AND_WHEN("We grow the SetNode twice") {
                s_ptr->grow(state);
                s_ptr->grow(state);
                s_ptr->propagate(state);
                B_ptr->propagate(state);  // so any changes are incorporated

                THEN("The state is updated and the updates are signalled") {
                    CHECK(std::ranges::equal(B_ptr->shape(state), s_ptr->shape(state)));
                    CHECK(B_ptr->size(state) == s_ptr->size(state));

                    CHECK(std::ranges::equal(B_ptr->view(state), std::vector{4, 3}));

                    CHECK(B_ptr->size_diff(state) == 2);  // grew by two

                    verify_array_diff({}, {4, 3}, B_ptr->diff(state));
                }

                AND_WHEN("We commit") {
                    s_ptr->commit(state);
                    B_ptr->commit(state);

                    THEN("The values stick, and the diff is cleared") {
                        CHECK(std::ranges::equal(B_ptr->shape(state), s_ptr->shape(state)));
                        CHECK(B_ptr->size(state) == s_ptr->size(state));
                        CHECK(std::ranges::equal(B_ptr->view(state), std::vector{4, 3}));

                        CHECK(B_ptr->size_diff(state) == 0);
                        CHECK(B_ptr->diff(state).size() == 0);
                    }

                    AND_WHEN("We shrink the SetNode once") {
                        s_ptr->shrink(state);
                        s_ptr->propagate(state);
                        B_ptr->propagate(state);  // so any changes are incorporated

                        THEN("The state is updated and the updates are signalled") {
                            CHECK(std::ranges::equal(B_ptr->shape(state), s_ptr->shape(state)));
                            CHECK(B_ptr->size(state) == s_ptr->size(state));

                            CHECK(std::ranges::equal(B_ptr->view(state), std::vector{4}));

                            CHECK(B_ptr->size_diff(state) == -1);  // shrank by one

                            verify_array_diff({4, 3}, {4}, B_ptr->diff(state));
                        }

                        AND_WHEN("We commit") {
                            s_ptr->commit(state);
                            B_ptr->commit(state);

                            THEN("The values stick, and the diff is cleared") {
                                CHECK(std::ranges::equal(B_ptr->shape(state), s_ptr->shape(state)));
                                CHECK(B_ptr->size(state) == s_ptr->size(state));
                                CHECK(std::ranges::equal(B_ptr->view(state), std::vector{4}));

                                CHECK(B_ptr->size_diff(state) == 0);
                                CHECK(B_ptr->diff(state).size() == 0);
                            }
                        }

                        AND_WHEN("We revert") {
                            s_ptr->revert(state);
                            B_ptr->revert(state);

                            THEN("We're back to where we started") {
                                CHECK(std::ranges::equal(B_ptr->shape(state), s_ptr->shape(state)));
                                CHECK(B_ptr->size(state) == s_ptr->size(state));
                                CHECK(std::ranges::equal(B_ptr->view(state),
                                                         std::vector<double>{4, 3}));

                                CHECK(B_ptr->size_diff(state) == 0);
                                CHECK(B_ptr->diff(state).size() == 0);
                            }
                        }
                    }
                }

                AND_WHEN("We revert") {
                    s_ptr->revert(state);
                    B_ptr->revert(state);

                    THEN("We're back to where we started") {
                        CHECK(std::ranges::equal(B_ptr->shape(state), s_ptr->shape(state)));
                        CHECK(B_ptr->size(state) == s_ptr->size(state));
                        CHECK(std::ranges::equal(B_ptr->view(state), std::vector<double>{}));

                        CHECK(B_ptr->size_diff(state) == 0);
                        CHECK(B_ptr->diff(state).size() == 0);
                    }
                }
            }
        }
    }

    GIVEN("A 2d 3x3 matrix accessed by two List(3) nodes") {
        std::vector<double> values{0, 1, 2, 3, 4, 5, 6, 7, 8};
        auto A_ptr = graph.emplace_node<ConstantNode>(values, std::initializer_list<ssize_t>{3, 3});

        auto i_ptr = graph.emplace_node<ListNode>(3);
        auto j_ptr = graph.emplace_node<ListNode>(3);

        auto B_ptr = graph.emplace_node<AdvancedIndexingNode>(A_ptr, i_ptr, j_ptr);

        THEN("Then the resulting matrix has the size/shape we expect") {
            CHECK(std::ranges::equal(B_ptr->shape(), std::vector{3}));
            CHECK(B_ptr->size() == 3);
            CHECK(B_ptr->ndim() == 1);

            CHECK(array_shape_equal(B_ptr, i_ptr));
            CHECK(array_shape_equal(B_ptr, j_ptr));
        }

        THEN("We see the min/max/integral we expect") {
            CHECK(B_ptr->min() == 0);
            CHECK(B_ptr->max() == 8);
            CHECK(B_ptr->integral());
        }

        WHEN("We default-initialize the state") {
            auto state = graph.initialize_state();

            THEN("We have the state we expect") {
                CHECK(std::ranges::equal(A_ptr->view(state), values));
                CHECK(std::ranges::equal(i_ptr->view(state), std::vector{0, 1, 2}));
                CHECK(std::ranges::equal(j_ptr->view(state), std::vector{0, 1, 2}));
                CHECK(std::ranges::equal(B_ptr->view(state), std::vector{0, 4, 8}));
            }
        }

        WHEN("We explicitly initialize the state") {
            auto state = graph.empty_state();
            i_ptr->initialize_state(state, {0, 2, 1});
            j_ptr->initialize_state(state, {2, 1, 0});
            graph.initialize_state(state);

            THEN("We have the state we expect") {
                CHECK(std::ranges::equal(A_ptr->view(state), values));
                CHECK(std::ranges::equal(i_ptr->view(state), std::vector{0, 2, 1}));
                CHECK(std::ranges::equal(j_ptr->view(state), std::vector{2, 1, 0}));
                CHECK(std::ranges::equal(B_ptr->view(state), std::vector{2, 7, 3}));
            }

            AND_WHEN("We mutate one of the decision variables and then propagate") {
                i_ptr->exchange(state, 1, 2);  // [0, 2, 1] -> [0, 1, 2]

                i_ptr->propagate(state);
                B_ptr->propagate(state);

                THEN("We have the state we expect") {
                    CHECK(std::ranges::equal(A_ptr->view(state), values));
                    CHECK(std::ranges::equal(i_ptr->view(state), std::vector{0, 1, 2}));
                    CHECK(std::ranges::equal(j_ptr->view(state), std::vector{2, 1, 0}));
                    CHECK(std::ranges::equal(B_ptr->view(state), std::vector{2, 4, 6}));

                    verify_array_diff({2, 7, 3}, {2, 4, 6}, B_ptr->diff(state));
                }

                AND_WHEN("We commit") {
                    i_ptr->commit(state);
                    B_ptr->commit(state);

                    THEN("We have the updated state still") {
                        CHECK(std::ranges::equal(A_ptr->view(state), values));
                        CHECK(std::ranges::equal(i_ptr->view(state), std::vector{0, 1, 2}));
                        CHECK(std::ranges::equal(j_ptr->view(state), std::vector{2, 1, 0}));
                        CHECK(std::ranges::equal(B_ptr->view(state), std::vector{2, 4, 6}));
                    }
                }

                AND_WHEN("We revert") {
                    i_ptr->revert(state);
                    B_ptr->revert(state);

                    THEN("We have the original state") {
                        CHECK(std::ranges::equal(A_ptr->view(state), values));
                        CHECK(std::ranges::equal(i_ptr->view(state), std::vector{0, 2, 1}));
                        CHECK(std::ranges::equal(j_ptr->view(state), std::vector{2, 1, 0}));
                        CHECK(std::ranges::equal(B_ptr->view(state), std::vector{2, 7, 3}));
                    }
                }
            }
        }
    }

    GIVEN("A 2d 3x3 matrix accessed by two dynamic nodes") {
        std::vector<double> values{0, 1, 2, 3, 4, 5, 6, 7, 8};
        auto A_ptr = graph.emplace_node<ConstantNode>(values, std::initializer_list<ssize_t>{3, 3});

        auto dyn_ptr = graph.emplace_node<DynamicArrayTestingNode>(
                std::initializer_list<ssize_t>{-1, 2}, 0, 2, true);

        auto i_ptr = graph.emplace_node<BasicIndexingNode>(dyn_ptr, Slice(), 0);
        auto j_ptr = graph.emplace_node<BasicIndexingNode>(dyn_ptr, Slice(), 1);

        auto B_ptr = graph.emplace_node<AdvancedIndexingNode>(A_ptr, i_ptr, j_ptr);

        THEN("Then the resulting matrix has the size/shape we expect") {
            CHECK(B_ptr->dynamic());
            CHECK(B_ptr->ndim() == 1);

            CHECK(array_shape_equal(B_ptr, i_ptr));
            CHECK(array_shape_equal(B_ptr, j_ptr));
        }

        WHEN("We explicitly initialize an empty state") {
            auto state = graph.empty_state();
            dyn_ptr->initialize_state(state);
            graph.initialize_state(state);

            THEN("We have the state we expect") {
                CHECK(B_ptr->dynamic());
                CHECK(B_ptr->size(state) == 0);
                CHECK(std::ranges::equal(B_ptr->view(state), std::vector<double>()));
            }

            AND_WHEN("We grow both of the decision variables and then propagate") {
                // i_ptr->grow(state);  // [] -> [0]
                // j_ptr->grow(state);  // [] -> [0]
                dyn_ptr->grow(state, {0, 0});

                dyn_ptr->propagate(state);
                i_ptr->propagate(state);
                j_ptr->propagate(state);
                B_ptr->propagate(state);

                CHECK(std::ranges::equal(B_ptr->view(state), std::vector{0}));
                verify_array_diff({}, {0}, B_ptr->diff(state));
            }
        }

        WHEN("We explicitly initialize a state") {
            auto state = graph.empty_state();
            // i_ptr = [0, 2]
            // j_ptr = [2, 1]
            dyn_ptr->initialize_state(state, {0, 2, 2, 1});
            graph.initialize_state(state);

            THEN("We have the state we expect") {
                CHECK(dyn_ptr->size(state) == 4);
                CHECK(std::ranges::equal(dyn_ptr->shape(state), std::vector{2, 2}));

                CHECK(i_ptr->size(state) == 2);
                CHECK(std::ranges::equal(i_ptr->shape(state), std::vector{2}));
                CHECK(std::ranges::equal(i_ptr->view(state), std::vector{0, 2}));

                CHECK(j_ptr->size(state) == 2);
                CHECK(std::ranges::equal(j_ptr->shape(state), std::vector{2}));
                CHECK(std::ranges::equal(j_ptr->view(state), std::vector{2, 1}));

                CHECK(std::ranges::equal(B_ptr->view(state), std::vector{2, 7}));
            }

            AND_WHEN("We grow the decision variable and then propagate") {
                // i_ptr [0, 2] -> [0, 2, 1]
                // j_ptr [2, 1] -> [2, 1, 0]
                dyn_ptr->grow(state, {1, 0});

                dyn_ptr->propagate(state);
                i_ptr->propagate(state);
                j_ptr->propagate(state);
                B_ptr->propagate(state);

                CHECK(std::ranges::equal(dyn_ptr->view(state), std::vector{0, 2, 2, 1, 1, 0}));
                CHECK(std::ranges::equal(i_ptr->view(state), std::vector{0, 2, 1}));
                CHECK(std::ranges::equal(j_ptr->view(state), std::vector{2, 1, 0}));
                CHECK(std::ranges::equal(B_ptr->view(state), std::vector{2, 7, 3}));
                verify_array_diff({2, 7}, {2, 7, 3}, B_ptr->diff(state));

                AND_WHEN("We revert") {
                    dyn_ptr->revert(state);
                    i_ptr->revert(state);
                    j_ptr->revert(state);
                    B_ptr->revert(state);

                    CHECK(std::ranges::equal(dyn_ptr->view(state), std::vector{0, 2, 2, 1}));
                    CHECK(std::ranges::equal(i_ptr->view(state), std::vector{0, 2}));
                    CHECK(std::ranges::equal(j_ptr->view(state), std::vector{2, 1}));
                    CHECK(std::ranges::equal(B_ptr->view(state), std::vector{2, 7}));
                    CHECK(B_ptr->diff(state).size() == 0);
                }
            }

            AND_WHEN("We shrink both of the decision variables and then propagate") {
                // i_ptr [0, 2] -> [0]
                // j_ptr [2, 1] -> [2]
                dyn_ptr->shrink(state);

                dyn_ptr->propagate(state);
                i_ptr->propagate(state);
                j_ptr->propagate(state);
                B_ptr->propagate(state);

                CHECK(std::ranges::equal(i_ptr->view(state), std::vector{0}));
                CHECK(std::ranges::equal(j_ptr->view(state), std::vector{2}));
                CHECK(std::ranges::equal(B_ptr->view(state), std::vector{2}));
                verify_array_diff({2, 7}, {2}, B_ptr->diff(state));

                AND_WHEN("We revert") {
                    dyn_ptr->revert(state);
                    i_ptr->revert(state);
                    j_ptr->revert(state);
                    B_ptr->revert(state);

                    CHECK(std::ranges::equal(B_ptr->view(state), std::vector{2, 7}));
                    CHECK(B_ptr->diff(state).size() == 0);
                }
            }

            AND_WHEN("We change and shrink") {
                // i_ptr [0, 2] -> [0]
                // j_ptr [2, 1] -> [1]
                dyn_ptr->set(state, 1, 1);
                dyn_ptr->shrink(state);

                dyn_ptr->propagate(state);
                i_ptr->propagate(state);
                j_ptr->propagate(state);
                B_ptr->propagate(state);

                CHECK(std::ranges::equal(i_ptr->view(state), std::vector{0}));
                CHECK(std::ranges::equal(j_ptr->view(state), std::vector{1}));
                CHECK(std::ranges::equal(B_ptr->view(state), std::vector{1}));
                verify_array_diff({2, 7}, {1}, B_ptr->diff(state));

                AND_WHEN("We revert") {
                    dyn_ptr->revert(state);
                    i_ptr->revert(state);
                    j_ptr->revert(state);
                    B_ptr->revert(state);

                    CHECK(std::ranges::equal(B_ptr->view(state), std::vector{2, 7}));
                    CHECK(B_ptr->diff(state).size() == 0);
                }
            }

            AND_WHEN("We grow, update, shrink, then propagate") {
                // i_ptr [0, 2] -> [0, 2, 1] -> [0, 2]
                // j_ptr [2, 1] -> [2, 1, 0] -> [0, 1, 2] -> [2, 1, 0] -> [2, 1]
                dyn_ptr->grow(state, {1, 0});
                dyn_ptr->set(state, 1, 0);
                dyn_ptr->set(state, 5, 2);
                dyn_ptr->set(state, 1, 2);
                dyn_ptr->set(state, 5, 0);
                dyn_ptr->shrink(state);

                dyn_ptr->propagate(state);
                i_ptr->propagate(state);
                j_ptr->propagate(state);
                B_ptr->propagate(state);

                CHECK(std::ranges::equal(i_ptr->view(state), std::vector{0, 2}));
                CHECK(std::ranges::equal(j_ptr->view(state), std::vector{2, 1}));
                CHECK(std::ranges::equal(B_ptr->view(state), std::vector{2, 7}));
                verify_array_diff({2, 7}, {2, 7}, B_ptr->diff(state));

                AND_WHEN("We revert") {
                    dyn_ptr->revert(state);
                    i_ptr->revert(state);
                    j_ptr->revert(state);
                    B_ptr->revert(state);

                    CHECK(std::ranges::equal(B_ptr->view(state), std::vector{2, 7}));
                    CHECK(B_ptr->diff(state).size() == 0);
                }
            }
        }
    }

    GIVEN("A 3d 2x3x5 matrix with two const 1d index arrays") {
        std::vector<double> values(30);
        std::iota(values.begin(), values.end(), 0);
        auto arr_ptr =
                graph.emplace_node<ConstantNode>(values, std::initializer_list<ssize_t>{2, 3, 5});

        std::vector<double> i = {1, 0};
        auto i_ptr = graph.emplace_node<ConstantNode>(i);

        std::vector<double> j = {1, 1};
        auto j_ptr = graph.emplace_node<ConstantNode>(j);

        WHEN("We access the matrix by (i, j, :)") {
            auto adv = graph.emplace_node<AdvancedIndexingNode>(arr_ptr, i_ptr, j_ptr, Slice());

            THEN("We get the shape we expect") {
                CHECK(adv->size() == 10);
                CHECK(std::ranges::equal(adv->shape(), std::vector{2, 5}));
            }

            THEN("We see the predecessors we expect") {
                CHECK(std::ranges::equal(adv->predecessors(),
                                         std::vector<Node*>{arr_ptr, i_ptr, j_ptr}));
            }

            AND_WHEN("We create a state") {
                auto state = graph.initialize_state();

                THEN("We can read out the state of the nodes") {
                    CHECK(std::ranges::equal(arr_ptr->view(state), values));
                    CHECK(std::ranges::equal(adv->view(state),
                                             std::vector{20, 21, 22, 23, 24, 5, 6, 7, 8, 9}));
                }
            }
        }

        WHEN("We access the matrix by (i, :, j)") {
            auto adv = graph.emplace_node<AdvancedIndexingNode>(arr_ptr, i_ptr, Slice(), j_ptr);

            THEN("We get the shape we expect") {
                CHECK(adv->size() == 6);
                CHECK(std::ranges::equal(adv->shape(), std::vector{2, 3}));
            }

            THEN("We see the predecessors we expect") {
                CHECK(std::ranges::equal(adv->predecessors(),
                                         std::vector<Node*>{arr_ptr, i_ptr, j_ptr}));
            }

            AND_WHEN("We create a state") {
                auto state = graph.initialize_state();

                THEN("We can read out the state of the nodes") {
                    CHECK(std::ranges::equal(arr_ptr->view(state), values));
                    CHECK(std::ranges::equal(adv->view(state), std::vector{16, 21, 26, 1, 6, 11}));
                }
            }
        }
    }

    GIVEN("A 4d 2x3x5x4 matrix with three const 1d index arrays") {
        std::vector<double> values(2 * 3 * 5 * 4);
        std::iota(values.begin(), values.end(), 0);
        auto arr_ptr = graph.emplace_node<ConstantNode>(values,
                                                        std::initializer_list<ssize_t>{2, 3, 5, 4});

        std::vector<double> i = {1, 0};
        auto i_ptr = graph.emplace_node<ConstantNode>(i);

        std::vector<double> j = {1, 1};
        auto j_ptr = graph.emplace_node<ConstantNode>(j);

        std::vector<double> k = {1, 2};
        auto k_ptr = graph.emplace_node<ConstantNode>(k);

        WHEN("We access the matrix by (i, j, k, :)") {
            auto adv =
                    graph.emplace_node<AdvancedIndexingNode>(arr_ptr, i_ptr, j_ptr, k_ptr, Slice());

            THEN("We get the shape we expect") {
                CHECK(adv->size() == 8);
                CHECK(std::ranges::equal(adv->shape(), std::vector{2, 4}));
            }

            AND_WHEN("We create a state") {
                auto state = graph.initialize_state();

                THEN("We can read out the state of the nodes") {
                    CHECK(std::ranges::equal(arr_ptr->view(state), values));
                    CHECK(std::ranges::equal(adv->view(state),
                                             std::vector{84, 85, 86, 87, 28, 29, 30, 31}));
                }
            }
        }

        WHEN("We access the matrix by (i, j, :, k)") {
            auto adv =
                    graph.emplace_node<AdvancedIndexingNode>(arr_ptr, i_ptr, j_ptr, Slice(), k_ptr);

            THEN("We get the shape we expect") {
                CHECK(adv->size() == 10);
                CHECK(std::ranges::equal(adv->shape(), std::vector{2, 5}));
            }

            AND_WHEN("We create a state") {
                auto state = graph.initialize_state();

                THEN("We can read out the state of the nodes") {
                    CHECK(std::ranges::equal(arr_ptr->view(state), values));
                    CHECK(std::ranges::equal(adv->view(state),
                                             std::vector{81, 85, 89, 93, 97, 22, 26, 30, 34, 38}));
                }
            }
        }

        WHEN("We access the matrix by (i, :, j, k)") {
            auto adv =
                    graph.emplace_node<AdvancedIndexingNode>(arr_ptr, i_ptr, Slice(), j_ptr, k_ptr);

            THEN("We get the shape we expect") {
                CHECK(adv->size() == 6);
                CHECK(std::ranges::equal(adv->shape(), std::vector{2, 3}));
            }

            AND_WHEN("We create a state") {
                auto state = graph.initialize_state();

                THEN("We can read out the state of the nodes") {
                    CHECK(std::ranges::equal(arr_ptr->view(state), values));
                    CHECK(std::ranges::equal(adv->view(state),
                                             std::vector{65, 85, 105, 6, 26, 46}));
                }
            }
        }

        WHEN("We access the matrix by (i, :, :, k)") {
            auto adv = graph.emplace_node<AdvancedIndexingNode>(arr_ptr, i_ptr, Slice(), Slice(),
                                                                k_ptr);

            THEN("We get the shape we expect") {
                CHECK(adv->size() == 30);
                CHECK(std::ranges::equal(adv->shape(), std::vector{2, 3, 5}));
            }

            AND_WHEN("We create a state") {
                auto state = graph.initialize_state();

                THEN("We can read out the state of the nodes") {
                    CHECK(std::ranges::equal(arr_ptr->view(state), values));
                    CHECK(std::ranges::equal(
                            adv->view(state),
                            std::vector{61,  65,  69,  73,  77,  81, 85, 89, 93, 97,
                                        101, 105, 109, 113, 117, 2,  6,  10, 14, 18,
                                        22,  26,  30,  34,  38,  42, 46, 50, 54, 58}));
                }
            }
        }

        WHEN("We access the matrix by (i, :, k, :)") {
            auto adv = graph.emplace_node<AdvancedIndexingNode>(arr_ptr, i_ptr, Slice(), k_ptr,
                                                                Slice());

            THEN("We get the shape we expect") {
                CHECK(adv->size() == 24);
                CHECK(std::ranges::equal(adv->shape(), std::vector{2, 3, 4}));
            }

            AND_WHEN("We create a state") {
                auto state = graph.initialize_state();

                THEN("We can read out the state of the nodes") {
                    CHECK(std::ranges::equal(arr_ptr->view(state), values));
                    CHECK(std::ranges::equal(
                            adv->view(state),
                            std::vector{64, 65, 66, 67, 84, 85, 86, 87, 104, 105, 106, 107,
                                        8,  9,  10, 11, 28, 29, 30, 31, 48,  49,  50,  51}));
                }
            }
        }

        WHEN("We access the matrix by (:, i, k, :)") {
            auto adv = graph.emplace_node<AdvancedIndexingNode>(arr_ptr, Slice(), i_ptr, k_ptr,
                                                                Slice());

            THEN("We get the shape we expect") {
                CHECK(adv->size() == 16);
                CHECK(std::ranges::equal(adv->shape(), std::vector{2, 2, 4}));
            }

            AND_WHEN("We create a state") {
                auto state = graph.initialize_state();

                THEN("We can read out the state of the nodes") {
                    CHECK(std::ranges::equal(arr_ptr->view(state), values));
                    CHECK(std::ranges::equal(adv->view(state),
                                             std::vector{24, 25, 26, 27, 8, 9, 10, 11, 84, 85, 86,
                                                         87, 68, 69, 70, 71}));
                }
            }
        }
    }

    GIVEN("A 4d 2x3x5x4 matrix with 3 dynamic indexing arrays") {
        std::vector<double> values(2 * 3 * 5 * 4);
        std::iota(values.begin(), values.end(), 0);
        auto arr_ptr = graph.emplace_node<ConstantNode>(values,
                                                        std::initializer_list<ssize_t>{2, 3, 5, 4});
        auto dyn_ptr = graph.emplace_node<DynamicArrayTestingNode>(
                std::initializer_list<ssize_t>{-1, 3}, 0, 1, true);

        auto i_ptr = graph.emplace_node<BasicIndexingNode>(dyn_ptr, Slice(), 0);
        auto j_ptr = graph.emplace_node<BasicIndexingNode>(dyn_ptr, Slice(), 1);
        auto k_ptr = graph.emplace_node<BasicIndexingNode>(dyn_ptr, Slice(), 2);

        WHEN("We access the matrix by (i, j, k, :)") {
            auto adv =
                    graph.emplace_node<AdvancedIndexingNode>(arr_ptr, i_ptr, j_ptr, k_ptr, Slice());

            THEN("We get the shape we expect") {
                CHECK(adv->dynamic());
                CHECK(std::ranges::equal(adv->shape(), std::vector{-1, 4}));
            }

            AND_WHEN("We create a state") {
                auto state = graph.initialize_state();

                THEN("The state starts empty") {
                    CHECK(std::ranges::equal(adv->view(state), std::vector<double>{}));
                }

                AND_WHEN("We grow the indexing nodes and propagate") {
                    // i_ptr -> {0, 1}
                    // j_ptr -> {1, 2}
                    // k_ptr -> {4, 4}
                    dyn_ptr->grow(state, {0, 1, 4, 1, 2, 4});

                    dyn_ptr->propagate(state);
                    i_ptr->propagate(state);
                    j_ptr->propagate(state);
                    k_ptr->propagate(state);
                    adv->propagate(state);

                    THEN("The state has the expected values and the diff is correct") {
                        CHECK(adv->size(state) == 8);
                        CHECK(std::ranges::equal(adv->view(state),
                                                 std::vector{36, 37, 38, 39, 116, 117, 118, 119}));
                        verify_array_diff({}, {36, 37, 38, 39, 116, 117, 118, 119},
                                          adv->diff(state));
                    }

                    AND_WHEN("We shrink the indexing nodes and propagate") {
                        dyn_ptr->commit(state);
                        i_ptr->commit(state);
                        j_ptr->commit(state);
                        k_ptr->commit(state);
                        adv->commit(state);

                        dyn_ptr->shrink(state);

                        dyn_ptr->propagate(state);
                        i_ptr->propagate(state);
                        j_ptr->propagate(state);
                        k_ptr->propagate(state);
                        adv->propagate(state);

                        THEN("The state has the expected values and the diff is correct") {
                            CHECK(adv->size(state) == 4);
                            CHECK(std::ranges::equal(adv->view(state),
                                                     std::vector{36, 37, 38, 39}));
                            verify_array_diff({36, 37, 38, 39, 116, 117, 118, 119},
                                              {36, 37, 38, 39}, adv->diff(state));
                        }

                        AND_WHEN("We revert") {
                            dyn_ptr->revert(state);
                            i_ptr->revert(state);
                            j_ptr->revert(state);
                            k_ptr->revert(state);
                            adv->revert(state);

                            THEN("The state has returned to the original") {
                                CHECK(adv->size(state) == 8);
                                CHECK(std::ranges::equal(
                                        adv->view(state),
                                        std::vector{36, 37, 38, 39, 116, 117, 118, 119}));
                                CHECK(adv->diff(state).size() == 0);
                            }
                        }
                    }
                }
            }
        }

        WHEN("We access the matrix by (i, :, j, k)") {
            auto adv =
                    graph.emplace_node<AdvancedIndexingNode>(arr_ptr, i_ptr, Slice(), j_ptr, k_ptr);

            THEN("We get the shape we expect") {
                CHECK(adv->dynamic());
                CHECK(std::ranges::equal(adv->shape(), std::vector{-1, 3}));
            }

            AND_WHEN("We create a state") {
                auto state = graph.initialize_state();

                THEN("The state starts empty") {
                    CHECK(std::ranges::equal(adv->view(state), std::vector<double>{}));
                }

                AND_WHEN("We grow the indexing nodes and propagate") {
                    // i_ptr -> {0, 1}
                    // j_ptr -> {1, 2}
                    // k_ptr -> {3, 3}
                    dyn_ptr->grow(state, {0, 1, 3, 1, 2, 3});

                    dyn_ptr->propagate(state);
                    i_ptr->propagate(state);
                    j_ptr->propagate(state);
                    k_ptr->propagate(state);
                    adv->propagate(state);

                    THEN("The state has the expected values and the diff is correct") {
                        CHECK(adv->size(state) == 6);
                        CHECK(std::ranges::equal(adv->view(state),
                                                 std::vector{7, 27, 47, 71, 91, 111}));
                        verify_array_diff({}, {7, 27, 47, 71, 91, 111}, adv->diff(state));
                    }

                    AND_WHEN("We shrink the indexing nodes and propagate") {
                        dyn_ptr->commit(state);
                        i_ptr->commit(state);
                        j_ptr->commit(state);
                        k_ptr->commit(state);
                        adv->commit(state);

                        dyn_ptr->shrink(state);

                        dyn_ptr->propagate(state);
                        i_ptr->propagate(state);
                        j_ptr->propagate(state);
                        k_ptr->propagate(state);
                        adv->propagate(state);

                        THEN("The state has the expected values and the diff is correct") {
                            CHECK(adv->size(state) == 3);
                            CHECK(std::ranges::equal(adv->view(state), std::vector{7, 27, 47}));
                            verify_array_diff({7, 27, 47, 71, 91, 111}, {7, 27, 47},
                                              adv->diff(state));
                        }

                        AND_WHEN("We revert") {
                            dyn_ptr->revert(state);
                            i_ptr->revert(state);
                            j_ptr->revert(state);
                            k_ptr->revert(state);
                            adv->revert(state);

                            THEN("The state has returned to the original") {
                                CHECK(adv->size(state) == 6);
                                CHECK(std::ranges::equal(adv->view(state),
                                                         std::vector{7, 27, 47, 71, 91, 111}));
                                CHECK(adv->diff(state).size() == 0);
                            }
                        }
                    }
                }
            }
        }
    }

    GIVEN("A dynamic 4d Nx3x5x4 matrix with 3 dynamic indexing arrays") {
        auto arr_ptr = graph.emplace_node<DynamicArrayTestingNode>(
                std::initializer_list<ssize_t>{-1, 3, 5, 4},  //
                -180, 180, true,  // takes values in [-180, 180] and will always be integers
                120, 1200);  // will always have at least two rows in the first dimension, up to 20

        auto i_range_ptr =
                graph.emplace_node<ConstantNode>(std::vector{0, 1});  // will index axis 1
        auto j_range_ptr = graph.emplace_node<ConstantNode>(std::vector{0, 1, 2, 3, 4});  // axis 3
        auto k_range_ptr = graph.emplace_node<ConstantNode>(std::vector{0, 1, 2, 3});     // axis 4

        auto dyn_ptr = graph.emplace_node<ListNode>(2, 0, 2);  // subsets of range(2)

        auto i_ptr = graph.emplace_node<AdvancedIndexingNode>(i_range_ptr, dyn_ptr);
        auto j_ptr = graph.emplace_node<AdvancedIndexingNode>(j_range_ptr, dyn_ptr);
        auto k_ptr = graph.emplace_node<AdvancedIndexingNode>(k_range_ptr, dyn_ptr);

        // toss a few validation nodes on for safey
        graph.emplace_node<ArrayValidationNode>(i_ptr);
        graph.emplace_node<ArrayValidationNode>(j_ptr);
        graph.emplace_node<ArrayValidationNode>(k_ptr);

        WHEN("We access the matrix by (i, :, j, k)") {
            auto adv_ptr =
                    graph.emplace_node<AdvancedIndexingNode>(arr_ptr, i_ptr, Slice(), j_ptr, k_ptr);

            graph.emplace_node<ArrayValidationNode>(adv_ptr);

            THEN("We get the shape we expect") {
                CHECK(adv_ptr->dynamic());
                CHECK(std::ranges::equal(adv_ptr->shape(), std::vector{-1, 3}));
            }

            AND_WHEN("We create a state") {
                auto state = graph.empty_state();

                // as promised, but 120 values into the array initially
                std::vector<double> values(2 * 3 * 5 * 4);
                std::iota(values.begin(), values.end(), 0);
                arr_ptr->initialize_state(state, values);

                // dyn array starts empty
                dyn_ptr->initialize_state(state, std::vector<double>{});

                graph.initialize_state(state);

                AND_WHEN("We mutate the main array, grow the indexing nodes and propagate") {
                    // double the size of the array
                    arr_ptr->grow(state, values);
                    dyn_ptr->grow(state);

                    // Change should be visible
                    arr_ptr->set(state, 20, -1);

                    // Not present in the ranges indexed
                    arr_ptr->set(state, 41, -2);

                    graph.propagate(state, graph.descendants(state, {arr_ptr, dyn_ptr}));

                    AND_WHEN("We commit") {
                        graph.commit(state, graph.descendants(state, {arr_ptr, dyn_ptr}));

                        THEN("The output is as expected") {
                            CHECK(std::ranges::equal(adv_ptr->view(state), std::vector{0, -1, 40}));
                            // ArrayValidationNode checks most of the consistency etc
                        }
                    }

                    AND_WHEN("We revert") {
                        graph.revert(state, graph.descendants(state, {arr_ptr, dyn_ptr}));

                        THEN("The output is as expected") {
                            CHECK(std::ranges::equal(adv_ptr->view(state), std::vector<double>{}));
                            // ArrayValidationNode checks most of the consistency etc
                        }
                    }
                }
            }
        }

        WHEN("We access the matrix by (i, :, j, ;)") {
            auto adv_ptr = graph.emplace_node<AdvancedIndexingNode>(arr_ptr, i_ptr, Slice(), j_ptr,
                                                                    Slice());

            graph.emplace_node<ArrayValidationNode>(adv_ptr);

            THEN("We get the shape we expect") {
                CHECK(adv_ptr->dynamic());
                CHECK(std::ranges::equal(adv_ptr->shape(), std::vector{-1, 3, 4}));
            }

            AND_WHEN("We create a state") {
                auto state = graph.empty_state();

                // as promised, but 120 values into the array initially
                std::vector<double> values(2 * 3 * 5 * 4);
                std::iota(values.begin(), values.end(), 0);
                arr_ptr->initialize_state(state, values);

                // dyn array starts empty
                dyn_ptr->initialize_state(state, std::vector<double>{});

                graph.initialize_state(state);

                AND_WHEN("We mutate the main array, grow the indexing nodes and propagate") {
                    // double the size of the array
                    arr_ptr->grow(state, values);
                    dyn_ptr->grow(state);

                    // These changes should be visible
                    arr_ptr->set(state, 21, -1);
                    arr_ptr->set(state, 42, -2);

                    // Not present in the ranges indexed
                    arr_ptr->set(state, 8, -4);

                    graph.propagate(state, graph.descendants(state, {arr_ptr, dyn_ptr}));

                    AND_WHEN("We commit") {
                        graph.commit(state, graph.descendants(state, {arr_ptr, dyn_ptr}));

                        THEN("The output is as expected") {
                            CHECK(std::ranges::equal(
                                    adv_ptr->view(state),
                                    std::vector{0, 1, 2, 3, 20, -1, 22, 23, 40, 41, -2, 43}));
                            // ArrayValidationNode checks most of the consistency etc
                        }
                    }

                    AND_WHEN("We revert") {
                        graph.revert(state, graph.descendants(state, {arr_ptr, dyn_ptr}));

                        THEN("The output is as expected") {
                            CHECK(std::ranges::equal(adv_ptr->view(state), std::vector<double>{}));
                            // ArrayValidationNode checks most of the consistency etc
                        }
                    }
                }
            }
        }

        THEN("We get an exception when accessing the matrix by (:, i, :, j)") {
            CHECK_THROWS(graph.emplace_node<AdvancedIndexingNode>(arr_ptr, Slice(), i_ptr, Slice(),
                                                                  j_ptr));
        }

        THEN("We get an exception when accessing the matrix by (:, i, j, :)") {
            CHECK_THROWS(graph.emplace_node<AdvancedIndexingNode>(arr_ptr, Slice(), i_ptr, j_ptr,
                                                                  Slice()));
        }
    }

    GIVEN("A non-constant and non-dynamic 1d array and a dynamic indexing array") {
        auto arr_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{10});
        auto i_ptr = graph.emplace_node<DynamicArrayTestingNode>(std::initializer_list<ssize_t>{-1},
                                                                 0, 8, true);

        WHEN("We access the array by i") {
            auto adv = graph.emplace_node<AdvancedIndexingNode>(arr_ptr, i_ptr);

            THEN("We get the shape we expect") {
                CHECK(adv->dynamic());
                CHECK(std::ranges::equal(adv->shape(), std::vector{-1}));
            }

            AND_WHEN("We create a state") {
                auto state = graph.empty_state();
                std::vector<double> values(10);
                std::iota(values.begin(), values.end(), 0);
                arr_ptr->initialize_state(state, values);
                graph.initialize_state(state);

                THEN("The state starts empty") {
                    CHECK(std::ranges::equal(adv->view(state), std::vector<double>{}));
                }

                AND_WHEN("We grow the indexing nodes and propagate") {
                    // i_ptr -> {3, 5, 7, 2}
                    i_ptr->grow(state, {3, 5, 7, 2});

                    arr_ptr->propagate(state);
                    i_ptr->propagate(state);
                    adv->propagate(state);

                    THEN("The state has the expected values and the diff is correct") {
                        CHECK(adv->size(state) == 4);
                        CHECK(std::ranges::equal(adv->view(state), std::vector{3, 5, 7, 2}));
                        verify_array_diff({}, {3, 5, 7, 2}, adv->diff(state));
                    }

                    AND_WHEN("We mutate the main array") {
                        arr_ptr->commit(state);
                        i_ptr->commit(state);
                        adv->commit(state);

                        arr_ptr->set_value(state, 3, 103);
                        arr_ptr->set_value(state, 8, 108);
                        arr_ptr->set_value(state, 2, 102);
                        arr_ptr->set_value(state, 5, 105);
                        arr_ptr->set_value(state, 9, 109);

                        arr_ptr->propagate(state);
                        i_ptr->propagate(state);
                        adv->propagate(state);

                        THEN("The state has the expected values and the diff is correct") {
                            CHECK(adv->size(state) == 4);
                            CHECK(std::ranges::equal(adv->view(state),
                                                     std::vector{103, 105, 7, 102}));
                            verify_array_diff({3, 5, 7, 2}, {103, 105, 7, 102}, adv->diff(state));
                        }

                        AND_WHEN("We revert") {
                            arr_ptr->revert(state);
                            i_ptr->revert(state);
                            adv->revert(state);

                            THEN("The state has the expected values and the diff is empty") {
                                CHECK(adv->size(state) == 4);
                                CHECK(std::ranges::equal(adv->view(state),
                                                         std::vector{3, 5, 7, 2}));
                                CHECK(adv->diff(state).size() == 0);
                            }
                        }
                    }
                }
            }
        }
    }

    GIVEN("A dynamic 4d Nx3x5x4 matrix with 3 non-constant and non-dynamic indexing arrays") {
        auto arr_ptr = graph.emplace_node<DynamicArrayTestingNode>(
                std::initializer_list<ssize_t>{-1, 3, 5, 4});

        auto i_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{3}, 0, 2);
        auto j_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{3}, 0, 4);
        auto k_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{3}, 0, 3);

        WHEN("We access the matrix by (:, i, j, k)") {
            auto adv =
                    graph.emplace_node<AdvancedIndexingNode>(arr_ptr, Slice(), i_ptr, j_ptr, k_ptr);

            THEN("We get the shape we expect") {
                CHECK(adv->dynamic());
                CHECK(std::ranges::equal(adv->shape(), std::vector{-1, 3}));
            }

            AND_WHEN("We create a state") {
                auto state = graph.empty_state();
                i_ptr->initialize_state(state, {1, 0, 2});
                j_ptr->initialize_state(state, {1, 2, 1});
                k_ptr->initialize_state(state, {0, 0, 2});
                graph.initialize_state(state);

                THEN("The state starts empty") {
                    CHECK(std::ranges::equal(adv->view(state), std::vector<double>{}));
                }

                AND_WHEN("We grow the main array and propagate") {
                    std::vector<double> values(2 * 3 * 5 * 4);
                    std::iota(values.begin(), values.end(), 0);
                    arr_ptr->grow(state, values);

                    arr_ptr->propagate(state);
                    adv->propagate(state);

                    THEN("The state has the expected values and the diff is correct") {
                        CHECK(adv->size(state) == 6);
                        CHECK(std::ranges::equal(adv->view(state),
                                                 std::vector{24, 8, 46, 84, 68, 106}));
                        verify_array_diff({}, {24, 8, 46, 84, 68, 106}, adv->diff(state));
                    }

                    AND_WHEN("We mutate the main array") {
                        arr_ptr->commit(state);
                        adv->commit(state);

                        REQUIRE(adv->diff(state).size() == 0);

                        arr_ptr->set(state, 84, -1);
                        arr_ptr->set(state, 68, -2);
                        arr_ptr->set(state, 2, -3);
                        arr_ptr->set(state, 8, -4);

                        arr_ptr->propagate(state);
                        adv->propagate(state);

                        THEN("The state has the expected values and the diff is correct") {
                            CHECK(adv->size(state) == 6);
                            CHECK(std::ranges::equal(adv->view(state),
                                                     std::vector{24, -4, 46, -1, -2, 106}));
                            verify_array_diff({24, 8, 46, 84, 68, 106}, {24, -4, 46, -1, -2, 106},
                                              adv->diff(state));
                        }
                    }

                    AND_WHEN("We mutate the indexing arrays") {
                        arr_ptr->commit(state);
                        adv->commit(state);

                        REQUIRE(adv->diff(state).size() == 0);

                        arr_ptr->set(state, 84, -1);
                        arr_ptr->set(state, 68, -2);
                        arr_ptr->set(state, 2, -3);
                        arr_ptr->set(state, 8, -4);

                        arr_ptr->propagate(state);
                        adv->propagate(state);

                        THEN("The state has the expected values and the diff is correct") {
                            CHECK(adv->size(state) == 6);
                            CHECK(std::ranges::equal(adv->view(state),
                                                     std::vector{24, -4, 46, -1, -2, 106}));
                            verify_array_diff({24, 8, 46, 84, 68, 106}, {24, -4, 46, -1, -2, 106},
                                              adv->diff(state));
                        }
                    }
                }
            }
        }

        WHEN("We access the matrix by (:, i, j, :)") {
            auto adv = graph.emplace_node<AdvancedIndexingNode>(arr_ptr, Slice(), i_ptr, j_ptr,
                                                                Slice());
            auto val = graph.emplace_node<ArrayValidationNode>(adv);

            THEN("We get the shape we expect") {
                CHECK(adv->dynamic());
                CHECK(std::ranges::equal(adv->shape(), std::vector{-1, 3, 4}));
            }

            AND_WHEN("We create a state") {
                auto state = graph.empty_state();
                i_ptr->initialize_state(state, {1, 0, 2});
                j_ptr->initialize_state(state, {1, 2, 1});
                graph.initialize_state(state);

                THEN("The state starts empty") {
                    CHECK(std::ranges::equal(adv->view(state), std::vector<double>{}));
                }

                AND_WHEN("We grow the main array and propagate") {
                    std::vector<double> values(2 * 3 * 5 * 4);
                    std::iota(values.begin(), values.end(), 0);
                    arr_ptr->grow(state, values);

                    arr_ptr->propagate(state);
                    adv->propagate(state);
                    val->propagate(state);

                    std::vector<double> expected({24, 25, 26, 27, 8,   9,   10,  11,
                                                  44, 45, 46, 47, 84,  85,  86,  87,
                                                  68, 69, 70, 71, 104, 105, 106, 107});

                    THEN("The state has the expected values and the diff is correct") {
                        CHECK(adv->size(state) == 2 * 3 * 4);
                        CHECK(std::ranges::equal(adv->shape(state), std::vector{2, 3, 4}));
                        CHECK(std::ranges::equal(adv->view(state), expected));
                        verify_array_diff({}, expected, adv->diff(state));
                    }

                    arr_ptr->commit(state);
                    adv->commit(state);
                    val->commit(state);

                    AND_WHEN("We mutate the main array") {
                        REQUIRE(adv->diff(state).size() == 0);

                        std::vector<double> new_expected = expected;

                        arr_ptr->set(state, 84, -1);
                        new_expected[12] = -1;

                        arr_ptr->set(state, 68, -2);
                        new_expected[16] = -2;

                        // Outside of the indexed range
                        arr_ptr->set(state, 2, -3);

                        arr_ptr->set(state, 11, -4);
                        new_expected[7] = -4;

                        arr_ptr->propagate(state);
                        adv->propagate(state);
                        val->propagate(state);

                        THEN("The state has the expected values and the diff is correct") {
                            CHECK(adv->size(state) == 2 * 3 * 4);
                            CHECK(std::ranges::equal(adv->shape(state), std::vector{2, 3, 4}));
                            CHECK(std::ranges::equal(adv->view(state), new_expected));
                            verify_array_diff(expected, new_expected, adv->diff(state));
                        }
                    }

                    AND_WHEN("We mutate the indices") {
                        i_ptr->set_value(state, 2, 1);  // 1, 0, 1
                        j_ptr->set_value(state, 1, 1);  // 1, 1, 1

                        std::vector<double> new_expected({24, 25, 26, 27, 4,  5,  6,  7,
                                                          24, 25, 26, 27, 84, 85, 86, 87,
                                                          64, 65, 66, 67, 84, 85, 86, 87});

                        i_ptr->propagate(state);
                        j_ptr->propagate(state);
                        adv->propagate(state);
                        val->propagate(state);

                        THEN("The state has the expected values and the diff is correct") {
                            CHECK(adv->size(state) == 2 * 3 * 4);
                            CHECK(std::ranges::equal(adv->shape(state), std::vector{2, 3, 4}));
                            CHECK(std::ranges::equal(adv->view(state), new_expected));
                            verify_array_diff(expected, new_expected, adv->diff(state));
                        }

                        AND_WHEN("We revert") {
                            i_ptr->revert(state);
                            j_ptr->revert(state);
                            adv->revert(state);
                            THEN("We get back the original state") {
                                // Everything should be checked by the ArrayValidationNode here
                                val->revert(state);
                                CHECK(std::ranges::equal(adv->view(state), expected));
                                CHECK(adv->diff(state).size() == 0);
                            }
                        }
                    }

                    AND_WHEN("We shrink and grow the array and mutate the indices") {
                        arr_ptr->shrink(state);
                        arr_ptr->shrink(state);

                        std::vector<double> values(2 * 3 * 5 * 4);
                        std::iota(values.begin(), values.end(), 1000);
                        arr_ptr->grow(state, values);

                        i_ptr->set_value(state, 2, 1);  // 1, 0, 1
                        j_ptr->set_value(state, 1, 1);  // 1, 1, 1

                        std::vector<double> new_expected({1024, 1025, 1026, 1027, 1004, 1005,
                                                          1006, 1007, 1024, 1025, 1026, 1027,
                                                          1084, 1085, 1086, 1087, 1064, 1065,
                                                          1066, 1067, 1084, 1085, 1086, 1087});

                        i_ptr->propagate(state);
                        j_ptr->propagate(state);
                        adv->propagate(state);
                        val->propagate(state);

                        THEN("The state has the expected values and the diff is correct") {
                            CHECK(adv->size(state) == 2 * 3 * 4);
                            CHECK(std::ranges::equal(adv->shape(state), std::vector{2, 3, 4}));
                            CHECK(std::ranges::equal(adv->view(state), new_expected));
                            verify_array_diff(expected, new_expected, adv->diff(state));
                        }

                        AND_WHEN("We revert") {
                            i_ptr->revert(state);
                            j_ptr->revert(state);
                            adv->revert(state);
                            THEN("We get back the original state") {
                                // Everything should be checked by the ArrayValidationNode here
                                val->revert(state);
                                CHECK(std::ranges::equal(adv->view(state), expected));
                                CHECK(adv->diff(state).size() == 0);
                            }
                        }
                    }
                }
            }
        }

        THEN("We get an exception when accessing the matrix by (:, i, :, j)") {
            CHECK_THROWS(graph.emplace_node<AdvancedIndexingNode>(arr_ptr, Slice(), i_ptr, Slice(),
                                                                  j_ptr));
        }
    }

    GIVEN("A static-sized 4d 2x3x5x4 matrix with 3 non-constant scalar indices") {
        auto arr_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{2, 3, 5, 4},
                                                       -1000, 1000);

        auto i_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{}, 0, 1);
        auto j_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{}, 0, 2);
        auto k_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{}, 0, 3);

        WHEN("We access the matrix by (i, j, :, :)") {
            auto adv = graph.emplace_node<AdvancedIndexingNode>(arr_ptr, i_ptr, j_ptr, Slice(),
                                                                Slice());

            THEN("We get the shape we expect") {
                CHECK(!adv->dynamic());
                CHECK(std::ranges::equal(adv->shape(), std::vector{5, 4}));
            }

            AND_WHEN("We create a state") {
                std::vector<double> values(2 * 3 * 5 * 4);
                std::iota(values.begin(), values.end(), 0);

                auto state = graph.empty_state();
                arr_ptr->initialize_state(state, values);
                i_ptr->initialize_state(state, {1});
                j_ptr->initialize_state(state, {1});
                graph.initialize_state(state);

                std::vector<double> expected_initial_state({80, 81, 82, 83, 84, 85, 86,
                                                            87, 88, 89, 90, 91, 92, 93,
                                                            94, 95, 96, 97, 98, 99});

                THEN("The state starts with the expected values") {
                    CHECK(std::ranges::equal(adv->view(state), expected_initial_state));
                }

                AND_WHEN("We mutate i and j") {
                    i_ptr->set_value(state, 0, 0);
                    j_ptr->set_value(state, 0, 2);

                    i_ptr->propagate(state);
                    j_ptr->propagate(state);
                    adv->propagate(state);

                    THEN("The state has the expected values and the diff is correct") {
                        CHECK(adv->size(state) == 20);

                        std::vector<double> new_state({40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                                                       50, 51, 52, 53, 54, 55, 56, 57, 58, 59});
                        CHECK(std::ranges::equal(adv->view(state), new_state));
                        verify_array_diff(expected_initial_state, new_state, adv->diff(state));
                    }
                }

                AND_WHEN("We mutate the main array") {
                    arr_ptr->set_value(state, 80, 80);
                    arr_ptr->set_value(state, 81, -81);

                    arr_ptr->set_value(state, 79, 79);
                    arr_ptr->set_value(state, 78, -78);

                    arr_ptr->propagate(state);
                    adv->propagate(state);

                    THEN("The state has the expected values and the diff is correct") {
                        CHECK(adv->size(state) == 20);

                        std::vector<double> new_state({80, -81, 82, 83, 84, 85, 86, 87, 88, 89,
                                                       90, 91,  92, 93, 94, 95, 96, 97, 98, 99});
                        CHECK(std::ranges::equal(adv->view(state), new_state));
                        verify_array_diff(expected_initial_state, new_state, adv->diff(state));
                    }
                }
            }
        }

        WHEN("We access the matrix by (:, i, j, :)") {
            auto adv = graph.emplace_node<AdvancedIndexingNode>(arr_ptr, Slice(), i_ptr, j_ptr,
                                                                Slice());

            THEN("We get the shape we expect") {
                CHECK(!adv->dynamic());
                CHECK(std::ranges::equal(adv->shape(), std::vector{2, 4}));
            }

            AND_WHEN("We create a state") {
                std::vector<double> values(2 * 3 * 5 * 4);
                std::iota(values.begin(), values.end(), 0);

                auto state = graph.empty_state();
                arr_ptr->initialize_state(state, values);
                i_ptr->initialize_state(state, {1});
                j_ptr->initialize_state(state, {1});
                graph.initialize_state(state);

                std::vector<double> expected_initial_state({24, 25, 26, 27, 84, 85, 86, 87});

                THEN("The state starts with the expected values") {
                    CHECK(std::ranges::equal(adv->view(state), expected_initial_state));
                }

                AND_WHEN("We mutate i and j") {
                    i_ptr->set_value(state, 0, 0);
                    j_ptr->set_value(state, 0, 2);

                    i_ptr->propagate(state);
                    j_ptr->propagate(state);
                    adv->propagate(state);

                    THEN("The state has the expected values and the diff is correct") {
                        CHECK(adv->size(state) == 8);

                        std::vector<double> new_state({8, 9, 10, 11, 68, 69, 70, 71});
                        CHECK(std::ranges::equal(adv->view(state), new_state));
                        verify_array_diff(expected_initial_state, new_state, adv->diff(state));
                    }
                }

                AND_WHEN("We mutate the main array") {
                    arr_ptr->set_value(state, 24, 24);
                    arr_ptr->set_value(state, 25, -25);

                    arr_ptr->set_value(state, 23, 23);
                    arr_ptr->set_value(state, 22, -22);

                    arr_ptr->propagate(state);
                    adv->propagate(state);

                    THEN("The state has the expected values and the diff is correct") {
                        CHECK(adv->size(state) == 8);

                        std::vector<double> new_state({24, -25, 26, 27, 84, 85, 86, 87});
                        CHECK(std::ranges::equal(adv->view(state), new_state));
                        verify_array_diff(expected_initial_state, new_state, adv->diff(state));
                    }
                }
            }
        }

        WHEN("We access the matrix by (i, j, :, k)") {
            auto adv =
                    graph.emplace_node<AdvancedIndexingNode>(arr_ptr, i_ptr, j_ptr, Slice(), k_ptr);

            THEN("We get the shape we expect") {
                CHECK(!adv->dynamic());
                CHECK(std::ranges::equal(adv->shape(), std::vector{5}));
            }

            AND_WHEN("We create a state") {
                std::vector<double> values(2 * 3 * 5 * 4);
                std::iota(values.begin(), values.end(), 0);

                auto state = graph.empty_state();
                arr_ptr->initialize_state(state, values);
                i_ptr->initialize_state(state, {1});
                j_ptr->initialize_state(state, {1});
                k_ptr->initialize_state(state, {3});
                graph.initialize_state(state);

                std::vector<double> expected_initial_state({83, 87, 91, 95, 99});

                THEN("The state starts with the expected values") {
                    CHECK(std::ranges::equal(adv->view(state), expected_initial_state));
                }

                AND_WHEN("We mutate i and j") {
                    i_ptr->set_value(state, 0, 0);
                    j_ptr->set_value(state, 0, 2);
                    k_ptr->set_value(state, 0, 1);

                    i_ptr->propagate(state);
                    j_ptr->propagate(state);
                    adv->propagate(state);

                    THEN("The state has the expected values and the diff is correct") {
                        CHECK(adv->size(state) == 5);

                        std::vector<double> new_state({41, 45, 49, 53, 57});
                        CHECK(std::ranges::equal(adv->view(state), new_state));
                        verify_array_diff(expected_initial_state, new_state, adv->diff(state));
                    }
                }

                AND_WHEN("We mutate the main array") {
                    arr_ptr->set_value(state, 83, 83);
                    arr_ptr->set_value(state, 87, -87);

                    arr_ptr->set_value(state, 82, 82);
                    arr_ptr->set_value(state, 81, -81);

                    arr_ptr->propagate(state);
                    adv->propagate(state);

                    THEN("The state has the expected values and the diff is correct") {
                        CHECK(adv->size(state) == 5);

                        std::vector<double> new_state({83, -87, 91, 95, 99});
                        CHECK(std::ranges::equal(adv->view(state), new_state));
                        verify_array_diff(expected_initial_state, new_state, adv->diff(state));
                    }
                }
            }
        }
    }

    GIVEN("A static-sized 4d 2x3x5x4 matrix with a 2d indexing array") {
        auto arr_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{2, 3, 5, 4},
                                                       -1000, 1000);

        auto i_ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{2, 3}, 0, 1);

        THEN("We are prevented from doing an indexing operation") {
            CHECK_THROWS(graph.emplace_node<AdvancedIndexingNode>(arr_ptr, Slice(), i_ptr, Slice(),
                                                                  Slice()));
        }
    }
}

TEST_CASE("BasicIndexingNode") {
    SECTION("SizeInfo") {
        SECTION("SetNode(10)[:]") {
            auto set = SetNode(10);

            auto slice = BasicIndexingNode(&set, Slice());

            auto sizeinfo = slice.sizeinfo();
            CHECK(sizeinfo.array_ptr == &set);
            CHECK(sizeinfo.multiplier == 1);
            CHECK(sizeinfo.offset == 0);
            CHECK(!sizeinfo.min.has_value());
            CHECK(!sizeinfo.max.has_value());

            // now substitute so we can see what we did to the array's size
            sizeinfo = sizeinfo.substitute();
            CHECK(sizeinfo.array_ptr == &set);
            CHECK(sizeinfo.multiplier == 1);
            CHECK(sizeinfo.offset == 0);
            CHECK(sizeinfo.min == 0);
            CHECK(sizeinfo.max == 10);
        }

        SECTION("SetNode(10)[::2]") {
            auto set = SetNode(10);

            auto slice = BasicIndexingNode(&set, Slice(std::nullopt, std::nullopt, 2));

            auto sizeinfo = slice.sizeinfo();
            CHECK(sizeinfo.array_ptr == &set);
            CHECK(sizeinfo.multiplier == fraction(1, 2));
            CHECK(sizeinfo.offset == 0);
            CHECK(!sizeinfo.min.has_value());
            CHECK(!sizeinfo.max.has_value());

            // now substitute so we can see what we did to the array's size
            sizeinfo = sizeinfo.substitute();
            CHECK(sizeinfo.array_ptr == &set);
            CHECK(sizeinfo.multiplier == fraction(1, 2));
            CHECK(sizeinfo.offset == 0);
            CHECK(sizeinfo.min == 0);
            CHECK(sizeinfo.max == 5);
        }

        SECTION("SetNode(10)[1:]") {
            auto set = SetNode(10);

            auto slice = BasicIndexingNode(&set, Slice(1, std::nullopt));

            auto sizeinfo = slice.sizeinfo();
            CHECK(sizeinfo.array_ptr == &set);
            CHECK(sizeinfo.multiplier == 1);
            CHECK(sizeinfo.offset == -1);
            CHECK(!sizeinfo.min.has_value());
            CHECK(!sizeinfo.max.has_value());

            // now substitute so we can see what we did to the array's size
            sizeinfo = sizeinfo.substitute();
            CHECK(sizeinfo.array_ptr == &set);
            CHECK(sizeinfo.multiplier == 1);
            CHECK(sizeinfo.offset == -1);
            CHECK(sizeinfo.min == 0);
            CHECK(sizeinfo.max == 9);
        }

        SECTION("SetNode(10)[11:]") {
            // dev note: this is an interesting case because clearly we have
            // enough information at construction time to determine that the
            // resulting array will have a fixed size of exactly 0. We should
            // think about having the constructor check and potentially raise
            // an error or set itself to not be dynamic.
            auto set = SetNode(10);

            auto slice = BasicIndexingNode(&set, Slice(11, std::nullopt));

            auto sizeinfo = slice.sizeinfo();
            CHECK(sizeinfo.array_ptr == &set);
            CHECK(sizeinfo.multiplier == 1);
            CHECK(sizeinfo.offset == -11);
            CHECK(!sizeinfo.min.has_value());
            CHECK(!sizeinfo.max.has_value());

            // now substitute so we can see what we did to the array's size
            // in this case, it's always 0!
            sizeinfo = sizeinfo.substitute();
            CHECK(sizeinfo == 0);
        }

        SECTION("SetNode(10)[:1]") {
            auto set = SetNode(10);

            auto slice = BasicIndexingNode(&set, Slice(1));

            auto sizeinfo = slice.sizeinfo();
            CHECK(sizeinfo.array_ptr == &set);
            CHECK(sizeinfo.multiplier == 1);
            CHECK(sizeinfo.offset == 0);
            CHECK(!sizeinfo.min.has_value());
            CHECK(sizeinfo.max == 1);

            // now substitute so we can see what we did to the array's size
            sizeinfo = sizeinfo.substitute();
            CHECK(sizeinfo.array_ptr == &set);
            CHECK(sizeinfo.multiplier == 1);
            CHECK(sizeinfo.offset == 0);
            CHECK(sizeinfo.min == 0);
            CHECK(sizeinfo.max == 1);
        }

        SECTION("SetNode(10)[:-1]") {
            auto set = SetNode(10);

            auto slice = BasicIndexingNode(&set, Slice(0, -1));

            auto sizeinfo = slice.sizeinfo();
            CHECK(sizeinfo.array_ptr == &set);
            CHECK(sizeinfo.multiplier == 1);
            CHECK(sizeinfo.offset == -1);
            CHECK(!sizeinfo.min.has_value());
            CHECK(!sizeinfo.max.has_value());

            // now substitute so we can see what we did to the array's size
            sizeinfo = sizeinfo.substitute();
            CHECK(sizeinfo.array_ptr == &set);
            CHECK(sizeinfo.multiplier == 1);
            CHECK(sizeinfo.offset == -1);
            CHECK(sizeinfo.min == 0);
            CHECK(sizeinfo.max == 9);
        }

        SECTION("SetNode(10)[-2:]") {
            auto set = SetNode(10);

            auto slice = BasicIndexingNode(&set, Slice(-2, std::nullopt));

            auto sizeinfo = slice.sizeinfo();
            CHECK(sizeinfo.array_ptr == &set);
            CHECK(sizeinfo.multiplier == 1);
            CHECK(sizeinfo.offset == 0);
            CHECK(!sizeinfo.min.has_value());
            CHECK(sizeinfo.max == 2);

            // now substitute so we can see what we did to the array's size
            sizeinfo = sizeinfo.substitute();
            CHECK(sizeinfo.array_ptr == &set);
            CHECK(sizeinfo.multiplier == 1);
            CHECK(sizeinfo.offset == 0);
            CHECK(sizeinfo.min == 0);
            CHECK(sizeinfo.max == 2);
        }

        SECTION("SetNode(10)[5:4]") {
            auto set = SetNode(10);

            auto slice = BasicIndexingNode(&set, Slice(5, 4));

            auto sizeinfo = slice.sizeinfo();
            CHECK(sizeinfo.array_ptr == &set);
            CHECK(sizeinfo.multiplier == 1);
            CHECK(sizeinfo.offset == -5);
            CHECK(!sizeinfo.min.has_value());
            CHECK(sizeinfo.max == 4);

            // now substitute so we can see what we did to the array's size
            sizeinfo = sizeinfo.substitute();
            CHECK(sizeinfo.array_ptr == &set);
            CHECK(sizeinfo.multiplier == 0);
            CHECK(sizeinfo.offset == 0);
            CHECK(sizeinfo.min == 0);
            CHECK(sizeinfo.max == 0);
        }
    }

    auto graph = Graph();

    GIVEN("A 1d ConstantNode") {
        std::vector<double> values = {30, 10, 40, 20};
        auto array_ptr = graph.emplace_node<ConstantNode>(values);

        WHEN("We do integer basic indexing") {
            auto ptr = graph.emplace_node<BasicIndexingNode>(array_ptr, 3);

            THEN("The resulting node has the shape we expect") {
                CHECK(ptr->ndim() == 0);
                CHECK(ptr->size() == 1);
                CHECK(std::ranges::equal(ptr->shape(), std::vector<ssize_t>()));
                CHECK(std::ranges::equal(ptr->strides(), std::vector<ssize_t>()));

                CHECK(ptr->contiguous());
            }

            THEN("The resulting node has the predecessors we expect") {
                CHECK(std::ranges::equal(ptr->predecessors(), std::vector<Node*>{array_ptr}));
                CHECK(std::ranges::equal(array_ptr->successors(), std::vector<Node*>{ptr}));
            }

            AND_WHEN("We create and then read the state") {
                auto state = graph.initialize_state();

                THEN("It has the values and shape we expect") {
                    CHECK(ptr->size(state) == 1);
                    CHECK(std::ranges::equal(ptr->shape(state), std::vector<ssize_t>()));
                    CHECK(std::ranges::equal(ptr->view(state), std::vector{20}));
                }
            }
        }

        WHEN("We do integer negative indexing") {
            auto ptr = graph.emplace_node<BasicIndexingNode>(array_ptr, -2);

            THEN("The resulting node has the shape we expect") {
                CHECK(ptr->ndim() == 0);
                CHECK(ptr->size() == 1);
                CHECK(std::ranges::equal(ptr->shape(), std::vector<ssize_t>()));
                CHECK(std::ranges::equal(ptr->strides(), std::vector<ssize_t>()));

                CHECK(ptr->contiguous());
            }

            AND_WHEN("We create and then read the state") {
                auto state = graph.initialize_state();

                THEN("It has the values and shape we expect") {
                    CHECK(ptr->size(state) == 1);
                    CHECK(std::ranges::equal(ptr->shape(state), std::vector<ssize_t>()));
                    CHECK(std::ranges::equal(ptr->view(state), std::vector{40}));
                }
            }
        }

        WHEN("We do indexing by an explicit Slice of length 1") {
            auto ptr = graph.emplace_node<BasicIndexingNode>(array_ptr, Slice(1, 2));

            THEN("The resulting node has the shape we expect") {
                CHECK(ptr->ndim() == 1);
                CHECK(ptr->size() == 1);
                CHECK(std::ranges::equal(ptr->shape(), std::vector{1}));
                CHECK(std::ranges::equal(ptr->strides(), std::vector{sizeof(double)}));

                CHECK(ptr->contiguous());
            }

            AND_WHEN("We create and then read the state") {
                auto state = graph.initialize_state();

                THEN("It has the values and shape we expect") {
                    CHECK(ptr->size(state) == 1);
                    CHECK(std::ranges::equal(ptr->shape(state), std::vector{1}));
                    CHECK(std::ranges::equal(ptr->view(state), std::vector{10}));
                }
            }
        }

        WHEN("We do indexing by an explicit Slice of length 2 using negative indexing") {
            auto ptr = graph.emplace_node<BasicIndexingNode>(array_ptr, Slice(1, -1));

            THEN("The resulting node has the shape we expect") {
                CHECK(ptr->ndim() == 1);
                CHECK(ptr->size() == 2);
                CHECK(std::ranges::equal(ptr->shape(), std::vector{2}));
                CHECK(std::ranges::equal(ptr->strides(), std::vector{sizeof(double)}));

                CHECK(ptr->contiguous());
            }

            AND_WHEN("We create and then read the state") {
                auto state = graph.initialize_state();

                THEN("It has the values and shape we expect") {
                    CHECK(ptr->size(state) == 2);
                    CHECK(std::ranges::equal(ptr->shape(state), std::vector{2}));
                    CHECK(std::ranges::equal(ptr->view(state), std::vector{10, 40}));
                }
            }
        }
    }

    GIVEN("A 2d ConstantNode") {
        std::vector<double> values = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        auto array_ptr =
                graph.emplace_node<ConstantNode>(values, std::initializer_list<ssize_t>{3, 3});

        WHEN("We do integer basic indexing") {
            auto ptr = graph.emplace_node<BasicIndexingNode>(array_ptr, 2, 0);

            THEN("The resulting node has the shape we expect") {
                CHECK(ptr->ndim() == 0);
                CHECK(ptr->size() == 1);
                CHECK(std::ranges::equal(ptr->shape(), std::vector<ssize_t>()));
                CHECK(std::ranges::equal(ptr->strides(), std::vector<ssize_t>()));

                CHECK(ptr->contiguous());
            }

            AND_WHEN("We create and then read the state") {
                auto state = graph.initialize_state();

                THEN("It has the values and shape we expect") {
                    CHECK(ptr->size(state) == 1);
                    CHECK(std::ranges::equal(ptr->shape(state), std::vector<int>{}));
                    CHECK(std::ranges::equal(ptr->view(state), std::vector{7}));
                }
            }
        }

        WHEN("We do indexing by an explicit Slice of length 1") {
            auto ptr = graph.emplace_node<BasicIndexingNode>(array_ptr, Slice(1, 2), Slice(0, 1));

            THEN("The resulting node has the shape we expect") {
                CHECK(ptr->ndim() == 2);
                CHECK(ptr->size() == 1);
                CHECK(std::ranges::equal(ptr->shape(), std::vector{1, 1}));
                CHECK(std::ranges::equal(ptr->strides(),
                                         std::vector{3 * sizeof(double), sizeof(double)}));

                CHECK(ptr->contiguous());
            }

            AND_WHEN("We create and then read the state") {
                auto state = graph.initialize_state();

                THEN("It has the values and shape we expect") {
                    CHECK(ptr->size(state) == 1);
                    CHECK(std::ranges::equal(ptr->shape(state), std::vector{1, 1}));
                    CHECK(std::ranges::equal(ptr->view(state), std::vector{4}));
                }
            }
        }

        WHEN("We do indexing by an explicit Slices of length 2 using negative indexing") {
            auto ptr = graph.emplace_node<BasicIndexingNode>(array_ptr, Slice(0, -1), Slice(-3, 2));

            THEN("The resulting node has the shape we expect") {
                CHECK(ptr->ndim() == 2);
                CHECK(ptr->size() == 4);
                CHECK(std::ranges::equal(ptr->shape(), std::vector{2, 2}));
                CHECK(std::ranges::equal(ptr->strides(),
                                         std::vector{3 * sizeof(double), sizeof(double)}));

                CHECK(!ptr->contiguous());
            }

            AND_WHEN("We create and then read the state") {
                auto state = graph.initialize_state();

                THEN("It has the values and shape we expect") {
                    CHECK(ptr->size(state) == 4);
                    CHECK(std::ranges::equal(ptr->shape(state), std::vector{2, 2}));
                    CHECK(std::ranges::equal(ptr->view(state), std::vector{1, 2, 4, 5}));
                }
            }
        }

        WHEN("We do indexing by a mix of slices and indices (slice, index)") {
            auto ptr = graph.emplace_node<BasicIndexingNode>(array_ptr, Slice(0, -1), 1);

            THEN("The resulting node has the shape we expect") {
                CHECK(ptr->ndim() == 1);
                CHECK(ptr->size() == 2);
                CHECK(std::ranges::equal(ptr->shape(), std::vector{2}));
                CHECK(std::ranges::equal(ptr->strides(), std::vector{3 * sizeof(double)}));

                CHECK(!ptr->contiguous());
            }

            AND_WHEN("We create and then read the state") {
                auto state = graph.initialize_state();

                THEN("It has the values and shape we expect") {
                    CHECK(ptr->size(state) == 2);
                    CHECK(std::ranges::equal(ptr->shape(state), std::vector{2}));
                    CHECK(std::ranges::equal(ptr->view(state), std::vector{2, 5}));
                }
            }
        }

        WHEN("We do indexing by a mix of slices and negative indices (slice, index)") {
            auto ptr = graph.emplace_node<BasicIndexingNode>(array_ptr, Slice(0, 3), -2);

            THEN("The resulting node has the shape we expect") {
                CHECK(ptr->ndim() == 1);
                CHECK(ptr->size() == 3);
                CHECK(std::ranges::equal(ptr->shape(), std::vector{3}));
                CHECK(std::ranges::equal(ptr->strides(), std::vector{3 * sizeof(double)}));

                CHECK(!ptr->contiguous());
            }

            AND_WHEN("We create and then read the state") {
                auto state = graph.initialize_state();

                THEN("It has the values and shape we expect") {
                    CHECK(ptr->size(state) == 3);
                    CHECK(std::ranges::equal(ptr->shape(state), std::vector{3}));
                    CHECK(std::ranges::equal(ptr->view(state), std::vector{2, 5, 8}));
                }
            }
        }

        WHEN("We do indexing by a mix of slices and indices (index, slice)") {
            auto ptr = graph.emplace_node<BasicIndexingNode>(array_ptr, 1, Slice(1, 3));

            THEN("The resulting node has the shape we expect") {
                CHECK(ptr->ndim() == 1);
                CHECK(ptr->size() == 2);
                CHECK(std::ranges::equal(ptr->shape(), std::vector{2}));
                CHECK(std::ranges::equal(ptr->strides(), std::vector{sizeof(double)}));

                CHECK(ptr->contiguous());
            }

            AND_WHEN("We create and then read the state") {
                auto state = graph.initialize_state();

                THEN("It has the values and shape we expect") {
                    CHECK(ptr->size(state) == 2);
                    CHECK(std::ranges::equal(ptr->shape(state), std::vector{2}));
                    CHECK(std::ranges::equal(ptr->view(state), std::vector{5, 6}));
                }
            }
        }
    }

    GIVEN("A 3d ConstantNode") {
        std::vector<double> values = {1, 2, 3, 4, 5, 6, 7, 8};
        auto array_ptr =
                graph.emplace_node<ConstantNode>(values, std::initializer_list<ssize_t>{2, 2, 2});

        WHEN("We do integer basic indexing") {
            auto ptr = graph.emplace_node<BasicIndexingNode>(array_ptr, 1, 0, 1);

            THEN("The resulting node has the shape we expect") {
                CHECK(ptr->ndim() == 0);
                CHECK(ptr->size() == 1);
                CHECK(std::ranges::equal(ptr->shape(), std::vector<ssize_t>()));
                CHECK(std::ranges::equal(ptr->strides(), std::vector<ssize_t>()));
            }

            THEN("We get the min/max/integral we expect") {
                CHECK(ptr->min() == 1);
                CHECK(ptr->max() == 8);
                CHECK(ptr->integral());
            }

            AND_WHEN("We create and then read the state") {
                auto state = graph.initialize_state();

                THEN("It has the values and shape we expect") {
                    CHECK(ptr->size(state) == 1);
                    CHECK(std::ranges::equal(ptr->shape(state), ptr->shape()));
                    CHECK(std::ranges::equal(ptr->view(state), std::vector{6}));
                }
            }
        }

        WHEN("We do indexing by a mix of slices and indices (slice, index, slice)") {
            auto ptr =
                    graph.emplace_node<BasicIndexingNode>(array_ptr, Slice(0, -1), 1, Slice(0, 2));

            THEN("The resulting node has the shape we expect") {
                CHECK(ptr->ndim() == 2);
                CHECK(ptr->size() == 2);
                CHECK(std::ranges::equal(ptr->shape(), std::vector{1, 2}));
                CHECK(std::ranges::equal(ptr->strides(),
                                         std::vector{4 * sizeof(double), sizeof(double)}));
            }

            AND_WHEN("We create and then read the state") {
                auto state = graph.initialize_state();

                THEN("It has the values and shape we expect") {
                    CHECK(ptr->size(state) == 2);
                    CHECK(std::ranges::equal(ptr->shape(state), ptr->shape()));
                    CHECK(std::ranges::equal(ptr->view(state), std::vector{3, 4}));
                }
            }
        }
    }

    GIVEN("x = List(n), that is x is fixed length") {
        constexpr int num_items = 5;

        auto x_ptr = graph.emplace_node<ListNode>(num_items);

        WHEN("y = x[:]") {
            auto y_ptr = graph.emplace_node<BasicIndexingNode>(x_ptr, Slice());

            THEN("y's shape is independent of state") {
                CHECK(std::ranges::equal(y_ptr->shape(), std::vector{num_items}));
                CHECK(y_ptr->size() == num_items);
                CHECK(y_ptr->ndim() == 1);
            }

            THEN("y == x") {
                auto state = graph.initialize_state();
                CHECK(std::ranges::equal(x_ptr->view(state), y_ptr->view(state)));
            }
        }
    }

    WHEN("We access a constant fixed-length 1d array by a positive index") {
        const std::vector<double> values{0, 10, 20, 30, 40};
        auto arr_ptr = graph.emplace_node<ConstantNode>(values);

        auto a_ptr = graph.emplace_node<BasicIndexingNode>(arr_ptr, 3);

        THEN("The resulting BasicIndexingNode has the shape we expect") {
            CHECK(a_ptr->size() == 1);
            CHECK(a_ptr->ndim() == 0);
            CHECK(a_ptr->shape().size() == 0);
        }

        AND_WHEN("We inspect the state") {
            auto state = graph.initialize_state();

            THEN("We get the value we expect") {
                CHECK(a_ptr->size(state) == 1);
                CHECK(a_ptr->shape(state).size() == 0);
                CHECK(std::ranges::equal(a_ptr->view(state), std::vector{30}));
            }
        }
    }

    WHEN("We access a constant fixed-length 1d array by a negative index") {
        const std::vector<double> values{0, 10, 20, 30, 40};
        auto arr_ptr = graph.emplace_node<ConstantNode>(values);

        auto a_ptr = graph.emplace_node<BasicIndexingNode>(arr_ptr, -1);

        THEN("The resulting BasicIndexingNode has the shape we expect") {
            CHECK(a_ptr->size() == 1);
            CHECK(a_ptr->ndim() == 0);
            CHECK(a_ptr->shape().size() == 0);
        }

        AND_WHEN("We inspect the state") {
            auto state = graph.initialize_state();

            THEN("We get the value we expect") {
                CHECK(a_ptr->size(state) == 1);
                CHECK(a_ptr->shape(state).size() == 0);
                CHECK(std::ranges::equal(a_ptr->view(state), std::vector{40}));
            }
        }
    }

    WHEN("We access a fixed-length 1d array by a slice specified with negative indices") {
        auto x_ptr = graph.emplace_node<ListNode>(5);

        auto y_ptr = graph.emplace_node<BasicIndexingNode>(x_ptr, Slice(-1));  // x[:-1]

        THEN("The resulting BasicIndexingNode has the shape we expect") {
            CHECK(y_ptr->size() == x_ptr->size() - 1);
            CHECK(y_ptr->ndim() == 1);
            CHECK(y_ptr->shape().size() == 1);
            CHECK(y_ptr->shape()[0] == y_ptr->size());
        }

        AND_WHEN("We inspect the state") {
            auto state = graph.initialize_state();

            THEN("We get the value we expect") {
                CHECK(y_ptr->size(state) == x_ptr->size() - 1);
                CHECK(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == y_ptr->size());

                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{0, 1, 2, 3}));
            }
        }
    }

    WHEN("We access a fixed-length 1d array by a slice") {
        auto x_ptr = graph.emplace_node<ListNode>(5);

        auto y_ptr = graph.emplace_node<BasicIndexingNode>(x_ptr, Slice(1, std::nullopt));  // x[1:]

        THEN("The resulting BasicIndexingNode has the shape we expect") {
            CHECK(y_ptr->size() == x_ptr->size() - 1);
            CHECK(y_ptr->ndim() == 1);
            CHECK(y_ptr->shape().size() == 1);
            CHECK(y_ptr->shape()[0] == y_ptr->size());
        }

        AND_WHEN("We inspect the state") {
            auto state = graph.initialize_state();

            THEN("We get the value we expect") {
                CHECK(y_ptr->size(state) == x_ptr->size() - 1);
                CHECK(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == y_ptr->size());

                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{1, 2, 3, 4}));
            }
        }
    }

    GIVEN("x = List(5); y = x[1::2]") {
        auto x_ptr = graph.emplace_node<ListNode>(5);
        auto y_ptr = graph.emplace_node<BasicIndexingNode>(x_ptr, Slice(1, std::nullopt, 2));

        THEN("y has the shape we expect") {
            CHECK(y_ptr->size() == 2);
            CHECK(y_ptr->ndim() == 1);
        }

        auto state = graph.initialize_state();

        THEN("We get the default states we expect") {
            CHECK(std::ranges::equal(x_ptr->view(state), std::vector{0, 1, 2, 3, 4}));
            CHECK(std::ranges::equal(y_ptr->view(state), std::vector{1, 3}));
        }

        WHEN("We do propagation") {
            x_ptr->exchange(state, 1, 2);

            x_ptr->propagate(state);
            y_ptr->propagate(state);

            THEN("The states are as expected") {
                CHECK(std::ranges::equal(x_ptr->view(state), std::vector{0, 2, 1, 3, 4}));
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{2, 3}));

                verify_array_diff({1, 3}, {2, 3}, y_ptr->diff(state));
            }
        }
    }

    GIVEN("x = Binary((3, 3)); y = x[::2, 1:]") {
        auto x_ptr = graph.emplace_node<BinaryNode>(std::vector<ssize_t>{3, 3});
        auto y_ptr = graph.emplace_node<BasicIndexingNode>(
                x_ptr, Slice(std::nullopt, std::nullopt, 2), Slice(1, std::nullopt));

        THEN("y has the shape and strides we expect") {
            CHECK(std::ranges::equal(y_ptr->shape(), std::vector{2, 2}));

            CHECK(std::ranges::equal(y_ptr->strides(), std::vector{48, 8}));
        }

        auto state = graph.initialize_state();

        THEN("We get the default states we expect") {
            CHECK(std::ranges::equal(x_ptr->view(state), std::vector<ssize_t>(9)));
            CHECK(std::ranges::equal(y_ptr->view(state), std::vector<ssize_t>(4)));
        }

        WHEN("We do propagation") {
            x_ptr->flip(state, 0);
            x_ptr->flip(state, 1);

            x_ptr->propagate(state);
            y_ptr->propagate(state);

            THEN("The states are as expected") {
                CHECK(std::ranges::equal(x_ptr->view(state),
                                         std::vector{1, 1, 0, 0, 0, 0, 0, 0, 0}));
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{1, 0, 0, 0}));

                verify_array_diff({0, 0, 0, 0}, {1, 0, 0, 0}, y_ptr->diff(state));
            }
        }
    }

    WHEN("We access a dynamic length 1d array by a slice to the end") {
        auto x_ptr = graph.emplace_node<SetNode>(5);
        auto y_ptr = graph.emplace_node<BasicIndexingNode>(x_ptr, Slice(1, std::nullopt));  // x[1:]

        THEN("The resulting BasicIndexingNode has the shape we expect") {
            CHECK(y_ptr->size() == Array::DYNAMIC_SIZE);
            CHECK(y_ptr->ndim() == 1);
            CHECK(y_ptr->shape().size() == 1);
            CHECK(y_ptr->shape()[0] == y_ptr->size());
        }

        AND_WHEN("We inspect the state") {
            auto state = graph.initialize_state();

            THEN("We get the value we expect") {
                CHECK(y_ptr->size(state) == 0);
                CHECK(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == y_ptr->size(state));

                CHECK(std::ranges::equal(y_ptr->view(state), std::vector<double>()));
            }

            AND_WHEN("We change the shape of the dynamic array") {
                x_ptr->grow(state);

                x_ptr->propagate(state);
                y_ptr->propagate(state);

                CHECK(y_ptr->size(state) == x_ptr->size(state) - 1);
                CHECK(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == y_ptr->size(state));
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector<double>()));

                x_ptr->grow(state);
                x_ptr->grow(state);

                x_ptr->propagate(state);
                y_ptr->propagate(state);

                CHECK(y_ptr->size(state) == x_ptr->size(state) - 1);
                CHECK(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == y_ptr->size(state));
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{1, 2}));
            }
        }
    }

    WHEN("We access a dynamic length 1d array by a slice from the beginning with negative end") {
        auto x_ptr = graph.emplace_node<SetNode>(5);
        auto y_ptr = graph.emplace_node<BasicIndexingNode>(x_ptr, Slice(0, -2));  // x[:-2]

        THEN("The resulting BasicIndexingNode has the shape we expect") {
            CHECK(y_ptr->size() == Array::DYNAMIC_SIZE);
            CHECK(y_ptr->ndim() == 1);
            CHECK(y_ptr->shape().size() == 1);
            CHECK(y_ptr->shape()[0] == y_ptr->size());
        }

        AND_WHEN("We inspect the state") {
            auto state = graph.initialize_state();

            THEN("We get the value we expect") {
                CHECK(y_ptr->size(state) == 0);
                CHECK(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == y_ptr->size(state));

                CHECK(std::ranges::equal(y_ptr->view(state), std::vector<double>()));
            }

            AND_WHEN("We grow the dynamic array below the range") {
                x_ptr->grow(state);
                x_ptr->grow(state);

                x_ptr->propagate(state);
                y_ptr->propagate(state);

                // Still should have 0 size
                CHECK(y_ptr->size(state) == 0);
                CHECK(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == y_ptr->size(state));
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector<double>()));
                CHECK(y_ptr->diff(state).size() == 0);
            }

            AND_WHEN("We grow the dynamic array up to the range") {
                x_ptr->grow(state);
                x_ptr->grow(state);
                x_ptr->grow(state);

                x_ptr->propagate(state);
                y_ptr->propagate(state);

                CHECK(y_ptr->size(state) == 1);
                REQUIRE(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == y_ptr->size(state));
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{0}));

                verify_array_diff({}, {0}, y_ptr->diff(state));

                x_ptr->commit(state);
                y_ptr->commit(state);

                x_ptr->grow(state);

                x_ptr->propagate(state);
                y_ptr->propagate(state);

                verify_array_diff({0}, {0, 1}, y_ptr->diff(state));

                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{0, 1}));
            }

            AND_WHEN("We grow the dynamic array past the range") {
                for (int i = 0; i < 4; i++) x_ptr->grow(state);

                x_ptr->propagate(state);
                y_ptr->propagate(state);

                CHECK(y_ptr->size(state) == 2);
                REQUIRE(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == y_ptr->size(state));
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{0, 1}));

                verify_array_diff({}, {0, 1}, y_ptr->diff(state));

                x_ptr->commit(state);
                y_ptr->commit(state);

                x_ptr->grow(state);

                x_ptr->propagate(state);
                y_ptr->propagate(state);

                verify_array_diff({0, 1}, {0, 1, 2}, y_ptr->diff(state));
            }

            AND_WHEN("We shrink the dynamic array below the range") {
                for (int i = 0; i < 4; i++) x_ptr->grow(state);

                x_ptr->propagate(state);
                y_ptr->propagate(state);
                x_ptr->commit(state);
                y_ptr->commit(state);

                // These should have already been tested
                REQUIRE(std::ranges::equal(y_ptr->view(state), std::vector{0, 1}));
                REQUIRE(y_ptr->diff(state).size() == 0);

                // Now shrink
                x_ptr->shrink(state);
                x_ptr->propagate(state);
                y_ptr->propagate(state);
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{0}));
                verify_array_diff({0, 1}, {0}, y_ptr->diff(state));

                x_ptr->commit(state);
                y_ptr->commit(state);

                // Shrink again
                x_ptr->shrink(state);
                x_ptr->propagate(state);
                y_ptr->propagate(state);
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector<double>()));
                verify_array_diff({0}, {}, y_ptr->diff(state));

                x_ptr->commit(state);
                y_ptr->commit(state);

                // Shrink again, should be no change
                x_ptr->shrink(state);
                x_ptr->propagate(state);
                y_ptr->propagate(state);
                x_ptr->commit(state);
                y_ptr->commit(state);
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector<double>()));
                CHECK(y_ptr->diff(state).size() == 0);
            }

            AND_WHEN(
                    "We shrink the dynamic array below the range, and then update a value above "
                    "that range") {
                for (int i = 0; i < 4; i++) x_ptr->grow(state);

                x_ptr->propagate(state);
                y_ptr->propagate(state);
                x_ptr->commit(state);
                y_ptr->commit(state);

                // These should have already been tested
                REQUIRE(std::ranges::equal(y_ptr->view(state), std::vector{0, 1}));
                REQUIRE(y_ptr->diff(state).size() == 0);

                // Now shrink
                x_ptr->grow(state);            // [0, 1, 2, 3, 4][:-2] = [0, 1, 2]
                x_ptr->exchange(state, 2, 4);  // [0, 1, 4, 3, 2][:-2] = [0, 1, 4]
                x_ptr->shrink(state);          // [0, 1, 4, 3][:-2] = [0, 1]
                x_ptr->propagate(state);
                y_ptr->propagate(state);

                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{0, 1}));
                verify_array_diff({0, 1}, {0, 1}, y_ptr->diff(state));
            }

            AND_WHEN("We shrink and grow the dynamic array") {
                for (int i = 0; i < 3; i++) {
                    x_ptr->grow(state);
                    x_ptr->grow(state);

                    x_ptr->shrink(state);
                }

                x_ptr->propagate(state);
                y_ptr->propagate(state);

                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{0}));
                verify_array_diff({}, {0}, y_ptr->diff(state));

                x_ptr->commit(state);
                y_ptr->commit(state);

                x_ptr->grow(state);
                x_ptr->exchange(state, 1, 2);
                x_ptr->shrink(state);

                x_ptr->propagate(state);
                y_ptr->propagate(state);

                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{0}));
                verify_array_diff({0}, {0}, y_ptr->diff(state));
            }

            AND_WHEN("We change the shape of the dynamic array, and then revert") {
                x_ptr->grow(state);
                x_ptr->grow(state);
                x_ptr->grow(state);

                x_ptr->revert(state);
                y_ptr->revert(state);

                CHECK(y_ptr->size(state) == 0);
                REQUIRE(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == y_ptr->size(state));
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector<double>()));
                CHECK(y_ptr->diff(state).size() == 0);
            }
        }
    }

    WHEN("We access a dynamic length 1d array by a slice with a negative start") {
        auto x_ptr = graph.emplace_node<SetNode>(5);
        auto y_ptr =
                graph.emplace_node<BasicIndexingNode>(x_ptr, Slice(-2, std::nullopt));  // x[-2:]

        THEN("The resulting BasicIndexingNode has the shape we expect") {
            CHECK(y_ptr->size() == Array::DYNAMIC_SIZE);
            CHECK(y_ptr->ndim() == 1);
            CHECK(y_ptr->shape().size() == 1);
            CHECK(y_ptr->shape()[0] == y_ptr->size());
        }

        AND_WHEN("We inspect the state") {
            auto state = graph.initialize_state();

            THEN("We get the value we expect") {
                CHECK(y_ptr->size(state) == 0);
                CHECK(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == y_ptr->size(state));

                CHECK(std::ranges::equal(y_ptr->view(state), std::vector<double>()));
            }

            AND_WHEN("We grow the dynamic array below the range") {
                x_ptr->grow(state);

                x_ptr->propagate(state);
                y_ptr->propagate(state);

                CHECK(y_ptr->size(state) == 1);
                CHECK(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == 1);
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{0}));
                verify_array_diff({}, {0}, y_ptr->diff(state));
            }

            AND_WHEN("We grow the dynamic array up to the range") {
                x_ptr->grow(state);
                x_ptr->grow(state);

                x_ptr->propagate(state);
                y_ptr->propagate(state);

                CHECK(y_ptr->size(state) == 2);
                REQUIRE(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == 2);
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{0, 1}));

                verify_array_diff({}, {0, 1}, y_ptr->diff(state));

                x_ptr->commit(state);
                y_ptr->commit(state);

                x_ptr->grow(state);

                x_ptr->propagate(state);
                y_ptr->propagate(state);

                verify_array_diff({0, 1}, {1, 2}, y_ptr->diff(state));

                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{1, 2}));
            }

            AND_WHEN("We grow the dynamic array past the range") {
                for (int i = 0; i < 4; i++) x_ptr->grow(state);

                x_ptr->propagate(state);
                y_ptr->propagate(state);

                CHECK(y_ptr->size(state) == 2);
                REQUIRE(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == y_ptr->size(state));
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{2, 3}));

                verify_array_diff({}, {2, 3}, y_ptr->diff(state));

                x_ptr->commit(state);
                y_ptr->commit(state);

                x_ptr->grow(state);

                x_ptr->propagate(state);
                y_ptr->propagate(state);

                verify_array_diff({2, 3}, {3, 4}, y_ptr->diff(state));
            }

            AND_WHEN("We shrink the dynamic array below the range") {
                for (int i = 0; i < 4; i++) x_ptr->grow(state);

                x_ptr->propagate(state);  // [0, 1, 2, 3]
                y_ptr->propagate(state);  // [2, 3]
                x_ptr->commit(state);
                y_ptr->commit(state);

                // These should have already been tested
                REQUIRE(std::ranges::equal(y_ptr->view(state), std::vector{2, 3}));
                REQUIRE(y_ptr->diff(state).size() == 0);

                // Now shrink
                x_ptr->shrink(state);
                x_ptr->propagate(state);
                y_ptr->propagate(state);
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{1, 2}));
                verify_array_diff({2, 3}, {1, 2}, y_ptr->diff(state));

                x_ptr->commit(state);
                y_ptr->commit(state);

                // Shrink twice, should now be only length one
                x_ptr->shrink(state);
                x_ptr->shrink(state);
                x_ptr->propagate(state);
                y_ptr->propagate(state);
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{0}));
                verify_array_diff({1, 2}, {0}, y_ptr->diff(state));

                x_ptr->commit(state);
                y_ptr->commit(state);

                // Shrink again, should now be length zero
                x_ptr->shrink(state);
                x_ptr->propagate(state);
                y_ptr->propagate(state);
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector<double>()));
                verify_array_diff({0}, {}, y_ptr->diff(state));
            }

            AND_WHEN("We shrink and grow the dynamic array") {
                for (int i = 0; i < 4; i++) {
                    x_ptr->grow(state);
                    x_ptr->grow(state);

                    x_ptr->shrink(state);
                }

                x_ptr->propagate(state);  // [0, 1, 2, 3]
                y_ptr->propagate(state);  // [2, 3]

                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{2, 3}));
                verify_array_diff({}, {2, 3}, y_ptr->diff(state));
            }

            AND_WHEN("We change the shape of the dynamic array, and then revert") {
                x_ptr->grow(state);
                x_ptr->grow(state);
                x_ptr->grow(state);

                x_ptr->revert(state);
                y_ptr->revert(state);

                CHECK(y_ptr->size(state) == 0);
                REQUIRE(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == y_ptr->size(state));
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector<double>()));
                CHECK(y_ptr->diff(state).size() == 0);
            }
        }
    }

    WHEN("We use the DynamicArrayTestingNode to do some fuzzing with BasicIndexingNode on 1D "
         "dynamic arrays") {
        auto slice = GENERATE(Slice(0, std::nullopt), Slice(0, 1), Slice(0, 2), Slice(0, 10),
                              Slice(1, std::nullopt), Slice(5, std::nullopt), Slice(0, -1),
                              Slice(0, -5), Slice(1, -5), Slice(3, -5), Slice(-1, std::nullopt),
                              Slice(-5, std::nullopt), Slice(-5, -2));

        auto x_ptr =
                graph.emplace_node<DynamicArrayTestingNode>(std::initializer_list<ssize_t>{-1});
        auto y_ptr = graph.emplace_node<BasicIndexingNode>(x_ptr, slice);
        auto validation = graph.emplace_node<ArrayValidationNode>(y_ptr);

        AND_WHEN("We do random moves and accept") {
            auto state = graph.initialize_state();
            auto rng = RngAdaptor(std::mt19937(42));

            for (int i = 0; i < 1000; ++i) {
                x_ptr->random_moves(state, rng, 20);

                x_ptr->propagate(state);
                y_ptr->propagate(state);
                validation->propagate(state);

                x_ptr->commit(state);
                y_ptr->commit(state);
                validation->commit(state);
            }
        }
    }

    // TODO slicing on multidimensional dynamic array, when we have one to test...
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

        WHEN("We do random moves") {
            auto rng = RngAdaptor(std::mt19937(42));

            auto state = graph.initialize_state();

            std::uniform_int_distribution<int> num_moves(1, 5);
            std::uniform_int_distribution<int> coin(0, 1);

            for (auto i = 0; i < 100; ++i) {
                // do a random number of default moves
                for (auto j = 0, n = num_moves(rng); j < n; ++j) {
                    x_ptr->default_move(state, rng);
                }

                graph.propose(state, {x_ptr},
                              [&rng, &coin](const Graph&, State&) { return coin(rng); });

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

}  // namespace dwave::optimization

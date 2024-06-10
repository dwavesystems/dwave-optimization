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
#include "dwave-optimization/nodes/mathematical.hpp"
#include "dwave-optimization/nodes/numbers.hpp"

namespace dwave::optimization {

TEMPLATE_TEST_CASE("BinaryOpNode", "", std::equal_to<double>, std::less_equal<double>,
                   std::plus<double>, std::minus<double>, std::multiplies<double>,
                   functional::max<double>, functional::min<double>, std::logical_and<double>,
                   std::logical_or<double>) {
    auto graph = Graph();

    auto func = TestType();

    GIVEN("Two scalar constants operated on") {
        auto a_ptr = graph.emplace_node<ConstantNode>(5);
        auto b_ptr = graph.emplace_node<ConstantNode>(6);

        auto p_ptr = graph.emplace_node<BinaryOpNode<TestType>>(a_ptr, b_ptr);

        THEN("The shape is also a scalar") {
            CHECK(p_ptr->ndim() == 0);
            CHECK(p_ptr->size() == 1);
        }

        WHEN("We make a state") {
            auto state = graph.initialize_state();

            THEN("The output has the value and shape we expect") {
                CHECK(p_ptr->size(state) == 1);
                CHECK(p_ptr->shape(state).size() == 0);
                CHECK(p_ptr->view(state)[0] == func(5, 6));
            }
        }
    }

    GIVEN("Two constant arrays operated on") {
        std::vector<double> values_a = {0, 1, 2, 3};
        std::vector<double> values_b = {3, 2, 1, 0};

        auto a_ptr = graph.emplace_node<ConstantNode>(values_a);
        auto b_ptr = graph.emplace_node<ConstantNode>(values_b);

        auto p_ptr = graph.emplace_node<BinaryOpNode<TestType>>(a_ptr, b_ptr);

        THEN("The shape is also a 1D array") {
            CHECK(p_ptr->ndim() == 1);
            CHECK(p_ptr->size() == 4);
        }

        WHEN("We make a state") {
            auto state = graph.initialize_state();

            THEN("The output has the value and shape we expect") {
                CHECK(p_ptr->size(state) == 4);
                CHECK(p_ptr->shape(state).size() == 1);

                for (int i = 0; i < p_ptr->size(state); ++i) {
                    CHECK(p_ptr->view(state)[i] == func(values_a[i], values_b[i]));
                }
            }
        }
    }

    GIVEN("One list and one const array operated on") {
        std::vector<double> values_a = {3, 2, 1, 0};
        // list value                  [0, 1, 2, 3]
        auto a_ptr = graph.emplace_node<ConstantNode>(values_a);
        auto b_ptr = graph.emplace_node<ListNode>(4);

        auto p_ptr = graph.emplace_node<BinaryOpNode<TestType>>(a_ptr, b_ptr);

        THEN("The shape is also a 1D array") {
            CHECK(p_ptr->ndim() == 1);
            CHECK(p_ptr->size() == 4);
        }

        WHEN("We make a state") {
            auto state = graph.initialize_state();

            THEN("The product has the value and shape we expect") {
                CHECK(p_ptr->size(state) == 4);
                CHECK(p_ptr->shape(state).size() == 1);

                for (int i = 0; i < p_ptr->size(state); ++i) {
                    CHECK(p_ptr->view(state)[i] == func(values_a[i], b_ptr->view(state)[i]));
                }
            }

            AND_WHEN("We modify the list state and propagate") {
                b_ptr->exchange(state, 1, 2);
                b_ptr->propagate(state);
                p_ptr->propagate(state);

                THEN("The output has the values we expect") {
                    CHECK(p_ptr->size(state) == 4);
                    CHECK(p_ptr->shape(state).size() == 1);

                    for (int i = 0; i < p_ptr->size(state); ++i) {
                        CHECK(p_ptr->view(state)[i] == func(values_a[i], b_ptr->view(state)[i]));
                    }
                }
            }
        }
    }

    GIVEN("Two lists operated on") {
        auto a_ptr = graph.emplace_node<ListNode>(4);
        auto b_ptr = graph.emplace_node<ListNode>(4);

        auto p_ptr = graph.emplace_node<BinaryOpNode<TestType>>(a_ptr, b_ptr);

        THEN("The shape is also a 1D array") {
            CHECK(p_ptr->ndim() == 1);
            CHECK(p_ptr->size() == 4);
        }

        WHEN("We make a state") {
            auto state = graph.initialize_state();

            THEN("The product has the value and shape we expect") {
                CHECK(p_ptr->size(state) == 4);
                CHECK(p_ptr->shape(state).size() == 1);

                for (int i = 0; i < p_ptr->size(state); ++i) {
                    CHECK(p_ptr->view(state)[i] ==
                          func(a_ptr->view(state)[i], b_ptr->view(state)[i]));
                }
            }

            AND_WHEN("We modify the 2 lists with single and overlapping changes") {
                a_ptr->exchange(state, 0, 1);
                b_ptr->exchange(state, 1, 2);

                a_ptr->propagate(state);
                b_ptr->propagate(state);
                p_ptr->propagate(state);

                THEN("The output has the values we expect") {
                    CHECK(p_ptr->size(state) == 4);
                    CHECK(p_ptr->shape(state).size() == 1);

                    for (int i = 0; i < p_ptr->size(state); ++i) {
                        CHECK(p_ptr->view(state)[i] ==
                              func(a_ptr->view(state)[i], b_ptr->view(state)[i]));
                    }
                }
            }
        }
    }
}

TEST_CASE("LessEqualNode") {
    auto graph = Graph();

    GIVEN("A less equal node operating on two continuous arrays") {
        auto lhs_ptr = graph.emplace_node<ConstantNode>(std::vector{0.5, 1.5, 2.5, 3.5, 4.5});
        auto rhs_ptr = graph.emplace_node<ConstantNode>(std::vector{4.5, 3.5, 2.5, 1.5, 0.5});
        auto le_ptr = graph.emplace_node<LessEqualNode>(lhs_ptr, rhs_ptr);

        THEN("The LessEqualNode is binary-valued") {
            CHECK(le_ptr->integral());
            CHECK(le_ptr->min() == 0);
            CHECK(le_ptr->max() == 1);
        }
    }

    GIVEN("A less equal node operating on broad-casted arrays") {
        auto lhs_ptr = graph.emplace_node<ConstantNode>(std::vector{0.5, 1.5, 2.5, 3.5, 4.5});
        auto rhs_ptr = graph.emplace_node<ConstantNode>(std::vector{2.5});
        auto le_ptr = graph.emplace_node<LessEqualNode>(lhs_ptr, rhs_ptr);

        THEN("The LessEqualNode is binary-valued") {
            CHECK(le_ptr->integral());
            CHECK(le_ptr->min() == 0);
            CHECK(le_ptr->max() == 1);
        }

        THEN("The LessEqualNode has correct shape") {
            CHECK(le_ptr->shape().size() == 1);
            CHECK(le_ptr->shape()[0] == 5);
        }
    }
}

TEST_CASE("MaxNode and MinNode") {
    auto graph = Graph();

    GIVEN("A list node with a min and max node over it") {
        auto list_ptr = graph.emplace_node<ListNode>(5, 0, 5);

        // chose init values out of range - we test that init works correctly
        // in the ReduceNode tests
        auto max_ptr = graph.emplace_node<MaxNode>(list_ptr, -1);
        auto min_ptr = graph.emplace_node<MinNode>(list_ptr, 6);

        AND_GIVEN("An initial state of [ 1 2 3 | 0 4 ]") {
            auto state = graph.empty_state();
            list_ptr->initialize_state(state, {1, 2, 3, 0, 4});
            list_ptr->shrink(state);
            list_ptr->shrink(state);
            list_ptr->commit(state);
            graph.initialize_state(state);

            THEN("The min and max are as expected") {
                CHECK(max_ptr->view(state)[0] == 3);
                CHECK(min_ptr->view(state)[0] == 1);
            }

            WHEN("We grow once to [ 1 2 3 0 | 4 ]") {
                list_ptr->grow(state);
                list_ptr->propagate(state);
                max_ptr->propagate(state);
                min_ptr->propagate(state);

                THEN("The min and max are as expected") {
                    CHECK(max_ptr->view(state)[0] == 3);
                    CHECK(min_ptr->view(state)[0] == 0);
                }
            }

            WHEN("We swap and then grow once to [ 1 2 3 4 | 0 ]") {
                list_ptr->exchange(state, 3, 4);
                list_ptr->grow(state);
                list_ptr->propagate(state);
                max_ptr->propagate(state);
                min_ptr->propagate(state);

                THEN("The min and max are as expected") {
                    CHECK(max_ptr->view(state)[0] == 4);
                    CHECK(min_ptr->view(state)[0] == 1);
                }
            }

            WHEN("We shrink once to [ 1 2 | 3 0 4 ]") {
                list_ptr->shrink(state);
                list_ptr->propagate(state);
                max_ptr->propagate(state);
                min_ptr->propagate(state);

                THEN("The min and max are as expected") {
                    CHECK(max_ptr->view(state)[0] == 2);
                    CHECK(min_ptr->view(state)[0] == 1);
                }
            }

            WHEN("We swap out the min value for something smaller [ 0 2 3 | 1 4 ]") {
                list_ptr->exchange(state, 0, 3);
                list_ptr->propagate(state);
                max_ptr->propagate(state);
                min_ptr->propagate(state);

                THEN("The min and max are as expected") {
                    CHECK(max_ptr->view(state)[0] == 3);
                    CHECK(min_ptr->view(state)[0] == 0);
                }
            }

            WHEN("We swap out the min value for something larger [ 4 2 3 | 0 1 ]") {
                list_ptr->exchange(state, 0, 4);
                list_ptr->propagate(state);
                max_ptr->propagate(state);
                min_ptr->propagate(state);

                THEN("The min and max are as expected") {
                    CHECK(max_ptr->view(state)[0] == 4);
                    CHECK(min_ptr->view(state)[0] == 2);
                }
            }

            WHEN("We swap out the max value for something larger [ 1 2 4 | 0 3 ]") {
                list_ptr->exchange(state, 2, 4);
                list_ptr->propagate(state);
                max_ptr->propagate(state);
                min_ptr->propagate(state);

                THEN("The min and max are as expected") {
                    CHECK(max_ptr->view(state)[0] == 4);
                    CHECK(min_ptr->view(state)[0] == 1);
                }
            }

            WHEN("We swap out the max value for something smaller [ 1 2 0 | 3 4 ]") {
                list_ptr->exchange(state, 2, 3);
                list_ptr->propagate(state);
                max_ptr->propagate(state);
                min_ptr->propagate(state);

                THEN("The min and max are as expected") {
                    CHECK(max_ptr->view(state)[0] == 2);
                    CHECK(min_ptr->view(state)[0] == 0);
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("NaryOpNode", "", functional::max<double>, functional::min<double>,
                   std::plus<double>, std::multiplies<double>) {
    auto graph = Graph();

    auto func = TestType();

    GIVEN("Several scalar constants func'ed") {
        auto a_ptr = graph.emplace_node<ConstantNode>(5);
        auto b_ptr = graph.emplace_node<ConstantNode>(6);
        auto c_ptr = graph.emplace_node<ConstantNode>(7);
        auto d_ptr = graph.emplace_node<ConstantNode>(8);

        auto p_ptr = graph.emplace_node<NaryOpNode<TestType>>(a_ptr);
        p_ptr->add_node(b_ptr);
        p_ptr->add_node(c_ptr);
        p_ptr->add_node(d_ptr);

        THEN("The shape is also a scalar") {
            CHECK(p_ptr->ndim() == 0);
            CHECK(p_ptr->size() == 1);
        }

        WHEN("We make a state") {
            auto state = graph.initialize_state();

            THEN("The product has the value and shape we expect") {
                CHECK(p_ptr->size(state) == 1);
                CHECK(p_ptr->shape(state).size() == 0);
                CHECK(p_ptr->view(state)[0] == func(5, func(6, func(7, 8))));
            }
        }
    }

    GIVEN("Two constants array func'ed") {
        std::vector<double> values_a = {0, 1, 2, 3};
        std::vector<double> values_b = {3, 2, 1, 0};

        auto a_ptr = graph.emplace_node<ConstantNode>(values_a);
        auto b_ptr = graph.emplace_node<ConstantNode>(values_b);

        auto p_ptr = graph.emplace_node<NaryOpNode<TestType>>(a_ptr);
        p_ptr->add_node(b_ptr);

        THEN("The shape is also a 1D array") {
            CHECK(p_ptr->ndim() == 1);
            CHECK(p_ptr->size() == 4);
        }

        WHEN("We make a state") {
            auto state = graph.initialize_state();

            THEN("The product has the value and shape we expect") {
                CHECK(p_ptr->size(state) == 4);
                CHECK(p_ptr->shape(state).size() == 1);

                for (int i = 0; i < p_ptr->size(state); ++i) {
                    CHECK(p_ptr->view(state)[i] == func(values_a[i], values_b[i]));
                }
            }
        }
    }
    GIVEN("Three lists func'ed") {
        auto a_ptr = graph.emplace_node<ListNode>(4);
        auto b_ptr = graph.emplace_node<ListNode>(4);
        auto c_ptr = graph.emplace_node<ListNode>(4);

        auto p_ptr = graph.emplace_node<NaryOpNode<TestType>>(a_ptr);
        p_ptr->add_node(b_ptr);
        p_ptr->add_node(c_ptr);

        THEN("The shape is also a 1D array") {
            CHECK(p_ptr->ndim() == 1);
            CHECK(p_ptr->size() == 4);
        }

        WHEN("We make a state") {
            auto state = graph.initialize_state();

            THEN("The product has the value and shape we expect") {
                CHECK(p_ptr->size(state) == 4);
                CHECK(p_ptr->shape(state).size() == 1);

                for (int i = 0; i < p_ptr->size(state); ++i) {
                    CHECK(p_ptr->view(state)[i] ==
                          func(a_ptr->view(state)[i],
                               func(b_ptr->view(state)[i], c_ptr->view(state)[i])));
                }
            }

            AND_WHEN("We modify the 2 lists with single and overlapping changes") {
                a_ptr->exchange(state, 0, 1);
                b_ptr->exchange(state, 1, 2);

                a_ptr->propagate(state);
                b_ptr->propagate(state);
                p_ptr->propagate(state);

                THEN("The product has the values we expect") {
                    for (int i = 0; i < p_ptr->size(state); ++i) {
                        CHECK(p_ptr->view(state)[i] ==
                              func(a_ptr->view(state)[i],
                                   func(b_ptr->view(state)[i], c_ptr->view(state)[i])));
                    }
                }
            }
        }
    }
}

TEST_CASE("ProdNode") {
    auto graph = Graph();

    GIVEN("Given a list node with a prod over it") {
        auto list_ptr = graph.emplace_node<ListNode>(5, 0, 5);
        auto prod_ptr = graph.emplace_node<ProdNode>(list_ptr);

        AND_GIVEN("An initial state of [ 1 2 3 | 0 4 ]") {
            auto state = graph.empty_state();
            list_ptr->initialize_state(state, {1, 2, 3, 0, 4});
            list_ptr->shrink(state);
            list_ptr->shrink(state);
            list_ptr->commit(state);
            graph.initialize_state(state);

            THEN("prod([ 1 2 3 ]) == 6") { CHECK(prod_ptr->view(state)[0] == 6); }

            WHEN("We add a 0 by growing once [ 1 2 3 0 | 4 ]") {
                list_ptr->grow(state);
                list_ptr->propagate(state);
                prod_ptr->propagate(state);

                THEN("prod([ 1 2 3 0 ]) == 0") { CHECK(prod_ptr->view(state)[0] == 0); }

                list_ptr->commit(state);
                prod_ptr->commit(state);

                AND_WHEN("We then shrink again to [ 1 2 3 ]") {
                    list_ptr->shrink(state);
                    list_ptr->propagate(state);
                    prod_ptr->propagate(state);

                    THEN("prod([ 1 2 3 ]) == 6") { CHECK(prod_ptr->view(state)[0] == 6); }
                }
            }

            WHEN("We add a 4 by swapping then growing once [ 1 2 3 4 | 0 ]") {
                list_ptr->exchange(state, 3, 4);
                list_ptr->grow(state);
                list_ptr->propagate(state);
                prod_ptr->propagate(state);

                THEN("prod([ 1 2 3 0 ]) == 24") { CHECK(prod_ptr->view(state)[0] == 24); }

                list_ptr->commit(state);
                prod_ptr->commit(state);

                AND_WHEN("We then shrink again to [ 1 2 3 ]") {
                    list_ptr->shrink(state);
                    list_ptr->propagate(state);
                    prod_ptr->propagate(state);

                    THEN("prod([ 1 2 3 ]) == 6") { CHECK(prod_ptr->view(state)[0] == 6); }
                }
            }

            WHEN("We add a 0 by swapping [ 1 0 3 | 2 4 ]") {
                list_ptr->exchange(state, 1, 3);
                list_ptr->propagate(state);
                prod_ptr->propagate(state);

                THEN("prod([ 1 0 3 ]) == 0") { CHECK(prod_ptr->view(state)[0] == 0); }

                AND_WHEN("We then revert") {
                    list_ptr->revert(state);
                    prod_ptr->revert(state);

                    THEN("prod([ 1 2 3 ]) == 6") { CHECK(prod_ptr->view(state)[0] == 6); }

                    AND_WHEN("we do an null propagation") {
                        list_ptr->propagate(state);
                        prod_ptr->propagate(state);

                        THEN("prod([ 1 2 3 ]) == 6") { CHECK(prod_ptr->view(state)[0] == 6); }
                    }
                }

                list_ptr->commit(state);
                prod_ptr->commit(state);

                AND_WHEN("We then swap back to [ 1 2 3 | 0 4 ]") {
                    list_ptr->exchange(state, 1, 3);
                    list_ptr->propagate(state);
                    prod_ptr->propagate(state);

                    THEN("prod([ 1 2 3 ]) == 6") { CHECK(prod_ptr->view(state)[0] == 6); }
                }
            }

            WHEN("We swap some values [ 3 2 1 | 0 4 ]") {
                list_ptr->exchange(state, 0, 2);
                list_ptr->propagate(state);
                prod_ptr->propagate(state);

                THEN("prod([ 3 2 1 ]) == 6") { CHECK(prod_ptr->view(state)[0] == 6); }

                list_ptr->commit(state);
                prod_ptr->commit(state);

                AND_WHEN("We then swap back to [ 1 2 3 | 0 4 ]") {
                    list_ptr->exchange(state, 0, 2);
                    list_ptr->propagate(state);
                    prod_ptr->propagate(state);

                    THEN("prod([ 1 2 3 ]) == 6") { CHECK(prod_ptr->view(state)[0] == 6); }
                }
            }
        }
    }

    GIVEN("Given a list node with a prod over it with an initial value of 0") {
        auto list_ptr = graph.emplace_node<ListNode>(5, 0, 5);
        auto prod_ptr = graph.emplace_node<ProdNode>(list_ptr, 0);

        AND_GIVEN("An initial state of [ 1 2 3 | 0 4 ]") {
            auto state = graph.empty_state();
            list_ptr->initialize_state(state, {1, 2, 3, 0, 4});
            list_ptr->shrink(state);
            list_ptr->shrink(state);
            list_ptr->commit(state);
            graph.initialize_state(state);

            THEN("prod([ 1 2 3 ], init=0) == 0") { CHECK(prod_ptr->view(state)[0] == 0); }

            WHEN("We add a 0 by growing once [ 1 2 3 0 | 4 ]") {
                list_ptr->grow(state);
                list_ptr->propagate(state);
                prod_ptr->propagate(state);

                THEN("prod([ 1 2 3 0 ], init=0) == 0") { CHECK(prod_ptr->view(state)[0] == 0); }

                list_ptr->commit(state);
                prod_ptr->commit(state);

                AND_WHEN("We then shrink again to [ 1 2 3 ]") {
                    list_ptr->shrink(state);
                    list_ptr->propagate(state);
                    prod_ptr->propagate(state);

                    THEN("prod([ 1 2 3 ], init=0) == 0") { CHECK(prod_ptr->view(state)[0] == 0); }
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("ReduceNode", "",
                   functional::max<double>,
                   functional::min<double>,
                   std::logical_and<double>,
                   std::multiplies<double>,
                   std::plus<double>) {
    auto graph = Graph();

    auto func = TestType();

    GIVEN("A scalar constant reduced") {
        auto a_ptr = graph.emplace_node<ConstantNode>(5);
        auto r_ptr = graph.emplace_node<ReduceNode<TestType>>(a_ptr);

        THEN("The output shape is scalar") {
            CHECK(r_ptr->ndim() == 0);
            CHECK(r_ptr->size() == 1);
        }

        WHEN("We make a state") {
            auto state = graph.initialize_state();

            THEN("The reduction has the value and shape we expect") {
                // this might not be true for all reduction operations
                CHECK(r_ptr->size(state) == 1);
                CHECK(r_ptr->shape(state).size() == 0);

                if (r_ptr->logical()) {
                    CHECK(r_ptr->view(state)[0] == static_cast<bool>(a_ptr->view(state)[0]));
                } else {
                    CHECK(r_ptr->view(state)[0] == a_ptr->view(state)[0]);
                }
            }
        }
    }

    GIVEN("An array reduced without an explicit init") {
        std::vector<double> values_a = {1, 2, 3, 4};
        auto a_ptr = graph.emplace_node<ConstantNode>(values_a);
        auto r_ptr = graph.emplace_node<ReduceNode<TestType>>(a_ptr);

        THEN("The output shape is scalar") {
            CHECK(r_ptr->ndim() == 0);
            CHECK(r_ptr->size() == 1);
        }

        WHEN("We make a state") {
            auto state = graph.initialize_state();

            THEN("The reduction has the value and shape we expect") {
                CHECK(r_ptr->ndim() == 0);
                CHECK(r_ptr->size(state) == 1);

                // write this in such a way as to not need an init value
                double lhs = values_a.at(0);
                for (std::size_t i = 1; i < values_a.size(); ++i) {
                    lhs = func(lhs, values_a.at(i));
                }
                CHECK(r_ptr->view(state)[0] == lhs);
            }
        }
    }

    GIVEN("An array reduced with an explicit init value") {
        double init = 17;
        std::vector<double> values = {1, 2, 3, 4};
        auto a_ptr = graph.emplace_node<ConstantNode>(values);
        auto r_ptr = graph.emplace_node<ReduceNode<TestType>>(a_ptr, init);

        THEN("The output shape is scalar") {
            CHECK(r_ptr->ndim() == 0);
            CHECK(r_ptr->size() == 1);
        }

        WHEN("We make a state") {
            auto state = graph.initialize_state();

            THEN("The reduction has the value and shape we expect") {
                CHECK(r_ptr->ndim() == 0);
                CHECK(r_ptr->size(state) == 1);
                CHECK(r_ptr->view(state)[0] ==
                      std::reduce(values.begin(), values.end(), init, func));
            }
        }
    }

    GIVEN("A set reduced with an explicit init value") {
        double init = 17;
        auto a_ptr = graph.emplace_node<SetNode>(4);
        auto r_ptr = graph.emplace_node<ReduceNode<TestType>>(a_ptr, init);

        THEN("The output shape is scalar") {
            CHECK(r_ptr->ndim() == 0);
            CHECK(r_ptr->size() == 1);
        }

        WHEN("We create an empty set state") {
            auto state = graph.empty_state();
            a_ptr->initialize_state(state, std::vector<double>{});
            graph.initialize_state(state);

            THEN("The reduction has the value and shape we expect") {
                CHECK(r_ptr->ndim() == 0);
                CHECK(r_ptr->size(state) == 1);
                if (r_ptr->logical()) {
                    CHECK(r_ptr->view(state)[0] == static_cast<bool>(init));
                } else {
                    CHECK(r_ptr->view(state)[0] == init);
                }
            }

            AND_WHEN("We grow the set by one") {
                a_ptr->grow(state);
                a_ptr->propagate(state);
                r_ptr->propagate(state);
                a_ptr->commit(state);
                r_ptr->commit(state);

                THEN("The reduction has the value and shape we expect") {
                    CHECK(r_ptr->ndim() == 0);
                    CHECK(r_ptr->size(state) == 1);
                    CHECK(r_ptr->view(state)[0] == func(init, a_ptr->view(state)[0]));
                }

                AND_WHEN("We shrink the set by one") {
                    a_ptr->shrink(state);
                    a_ptr->propagate(state);
                    r_ptr->propagate(state);
                    a_ptr->commit(state);
                    r_ptr->commit(state);

                    THEN("The reduction has the value and shape we expect") {
                        CHECK(r_ptr->ndim() == 0);
                        CHECK(r_ptr->size(state) == 1);
                        if (r_ptr->logical()) {
                            CHECK(r_ptr->view(state)[0] == static_cast<bool>(init));
                        } else {
                            CHECK(r_ptr->view(state)[0] == init);
                        }
                    }
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("ReduceNode - no default init", "", functional::max<double>,
                   functional::min<double>) {
    auto graph = Graph();

    GIVEN("A dynamic node") {
        auto a_ptr = graph.emplace_node<SetNode>(4);

        THEN("We cannot construct a reduce node without an initial value") {
            CHECK_THROWS(graph.emplace_node<ReduceNode<TestType>>(a_ptr));
            CHECK(a_ptr->successors().size() == 0);  // no side effects
        }
    }
}

TEMPLATE_TEST_CASE("ReduceNode - valid default init", "", std::plus<double>) {
    auto graph = Graph();

    auto func = TestType();

    GIVEN("A set reduced") {
        auto a_ptr = graph.emplace_node<SetNode>(4);
        auto r_ptr = graph.emplace_node<ReduceNode<TestType>>(a_ptr);

        THEN("The output shape is scalar") {
            CHECK(r_ptr->ndim() == 0);
            CHECK(r_ptr->size() == 1);
        }

        WHEN("We make a state - defaulting the set to populated") {
            auto state = graph.empty_state();
            a_ptr->initialize_state(state, std::vector{0.0, 1.0, 2.0});
            graph.initialize_state(state);

            THEN("The reduction has the value and shape we expect") {
                CHECK(r_ptr->ndim() == 0);
                CHECK(r_ptr->size(state) == 1);

                // write this in such a way as to not need an init value
                double lhs = a_ptr->view(state)[0];
                for (ssize_t i = 1; i < a_ptr->size(state); ++i) {
                    lhs = func(lhs, a_ptr->view(state)[i]);
                }
                CHECK(r_ptr->view(state)[0] == lhs);
            }

            AND_WHEN("We shrink the set") {
                REQUIRE(r_ptr->ndim() == 0);
                REQUIRE(r_ptr->size() == 1);

                double old_value = r_ptr->view(state)[0];

                a_ptr->shrink(state);
                a_ptr->propagate(state);
                r_ptr->propagate(state);

                THEN("The reduction has the value and shape we expect") {
                    CHECK(r_ptr->ndim() == 0);
                    CHECK(r_ptr->size(state) == 1);

                    // write this in such a way as to not need an init value
                    double lhs = a_ptr->view(state)[0];
                    for (ssize_t i = 1; i < a_ptr->size(state); ++i) {
                        lhs = func(lhs, a_ptr->view(state)[i]);
                    }
                    CHECK(r_ptr->view(state)[0] == lhs);

                    AND_WHEN("We commit") {
                        a_ptr->commit(state);
                        r_ptr->commit(state);

                        THEN("The value is maintained") { CHECK(r_ptr->view(state)[0] == lhs); }
                    }

                    AND_WHEN("We revert") {
                        a_ptr->revert(state);
                        r_ptr->revert(state);

                        THEN("The value is reverted") { CHECK(r_ptr->view(state)[0] == old_value); }
                    }
                }
            }

            AND_WHEN("We grow the set") {
                REQUIRE(r_ptr->ndim() == 0);
                REQUIRE(r_ptr->size(state) == 1);

                double old_value = r_ptr->view(state)[0];

                a_ptr->grow(state);
                a_ptr->propagate(state);
                r_ptr->propagate(state);

                THEN("The reduction has the value and shape we expect") {
                    CHECK(r_ptr->ndim() == 0);
                    CHECK(r_ptr->size(state) == 1);

                    // write this in such a way as to not need an init value
                    double lhs = a_ptr->view(state)[0];
                    for (ssize_t i = 1; i < a_ptr->size(state); ++i) {
                        lhs = func(lhs, a_ptr->view(state)[i]);
                    }
                    CHECK(r_ptr->view(state)[0] == lhs);

                    AND_WHEN("We commit") {
                        a_ptr->commit(state);
                        r_ptr->commit(state);

                        THEN("The value is maintained") { CHECK(r_ptr->view(state)[0] == lhs); }
                    }

                    AND_WHEN("We revert") {
                        a_ptr->revert(state);
                        r_ptr->revert(state);

                        THEN("The value is reverted") { CHECK(r_ptr->view(state)[0] == old_value); }
                    }
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("UnaryOpNode", "", functional::abs<double>, std::negate<double>) {
    auto graph = Graph();

    auto func = TestType();

    GIVEN("A constant scalar input") {
        auto a_ptr = graph.emplace_node<ConstantNode>(-5);

        auto p_ptr = graph.emplace_node<UnaryOpNode<TestType>>(a_ptr);

        THEN("The shape is also a scalar") {
            CHECK(p_ptr->ndim() == 0);
            CHECK(p_ptr->size() == 1);
        }

        WHEN("We make a state") {
            auto state = graph.initialize_state();

            THEN("The output has the value and shape we expect") {
                CHECK(p_ptr->size(state) == 1);
                CHECK(p_ptr->shape(state).size() == 0);
                CHECK(p_ptr->view(state)[0] == func(-5));
            }
        }
    }

    GIVEN("A 0-d integer decision input") {
        auto a_ptr = graph.emplace_node<IntegerNode>(std::span<const ssize_t>{}, -100,
                                                     100);  // Scalar output

        auto p_ptr = graph.emplace_node<UnaryOpNode<TestType>>(a_ptr);

        WHEN("We make a state") {
            auto state = graph.empty_state();
            a_ptr->initialize_state(state, {-5});
            graph.initialize_state(state);

            THEN("The output has the value and shape we expect") {
                CHECK(p_ptr->size(state) == 1);
                CHECK(p_ptr->shape(state).size() == 0);
                CHECK(p_ptr->view(state)[0] == func(-5));
            }

            AND_WHEN("We change the integer's state and propagate") {
                a_ptr->set_value(state, 0, 17);
                a_ptr->propagate(state);
                p_ptr->propagate(state);

                THEN("The output is what we expect") { CHECK(p_ptr->view(state)[0] == func(17)); }

                AND_WHEN("We commit") {
                    a_ptr->commit(state);
                    p_ptr->commit(state);

                    THEN("The value hasn't changed") { CHECK(p_ptr->view(state)[0] == func(17)); }
                    THEN("The diffs are cleared") { CHECK(p_ptr->diff(state).size() == 0); }
                }

                AND_WHEN("We revert") {
                    a_ptr->revert(state);
                    p_ptr->revert(state);

                    THEN("The value reverts to the previous") {
                        CHECK(p_ptr->view(state)[0] == func(-5));
                    }
                    THEN("The diffs are cleared") { CHECK(p_ptr->diff(state).size() == 0); }
                }
            }
        }
    }

    GIVEN("A 3-d integer decision input") {
        auto a_ptr = graph.emplace_node<IntegerNode>(std::span<const ssize_t>({2, 3, 2}), -100,
                                                     100);  // Scalar output

        auto p_ptr = graph.emplace_node<UnaryOpNode<TestType>>(a_ptr);

        WHEN("We make a state") {
            auto state = graph.empty_state();
            a_ptr->initialize_state(state, {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6});
            graph.initialize_state(state);

            THEN("The output has the value and shape we expect") {
                CHECK(p_ptr->size(state) == 12);
                CHECK(p_ptr->shape(state).size() == 3);
                CHECK(p_ptr->view(state)[0] == func(-5));
                CHECK(p_ptr->view(state)[4] == func(-1));
            }

            AND_WHEN("We change the integer's state and propagate") {
                a_ptr->set_value(state, 0, 17);
                a_ptr->propagate(state);
                p_ptr->propagate(state);

                THEN("The output is what we expect") { CHECK(p_ptr->view(state)[0] == func(17)); }

                AND_WHEN("We commit") {
                    a_ptr->commit(state);
                    p_ptr->commit(state);

                    THEN("The value hasn't changed") { CHECK(p_ptr->view(state)[0] == func(17)); }
                    THEN("The diffs are cleared") { CHECK(p_ptr->diff(state).size() == 0); }
                }

                AND_WHEN("We revert") {
                    a_ptr->revert(state);
                    p_ptr->revert(state);

                    THEN("The value reverts to the previous") {
                        CHECK(p_ptr->view(state)[0] == func(-5));
                    }
                    THEN("The diffs are cleared") { CHECK(p_ptr->diff(state).size() == 0); }
                }
            }
        }
    }
}

}  // namespace dwave::optimization

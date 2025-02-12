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

#include "catch2/catch_template_test_macros.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "catch2/matchers/catch_matchers_all.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/nodes/collections.hpp"
#include "dwave-optimization/nodes/constants.hpp"
#include "dwave-optimization/nodes/indexing.hpp"
#include "dwave-optimization/nodes/mathematical.hpp"
#include "dwave-optimization/nodes/numbers.hpp"
#include "dwave-optimization/nodes/testing.hpp"

using Catch::Matchers::RangeEquals;

namespace dwave::optimization {

TEMPLATE_TEST_CASE("PartialReduceNode", "",
        std::multiplies<double>,
        std::plus<double>) {
    GIVEN("A 1D array of 5 integers, and a reduction over axis 0 and an explicit initial value") {
        const double init = GENERATE(-1, 0, 1);

        auto graph = Graph();
        auto x_ptr = graph.emplace_node<IntegerNode>(5, 0, 10);  // 5 integers in [0, 10]
        auto r_ptr = graph.emplace_node<PartialReduceNode<TestType>>(x_ptr, 0, init);
        graph.emplace_node<ArrayValidationNode>(r_ptr);

        // this is equivalent to a reduction, so the output is a scalar
        CHECK(r_ptr->ndim() == 0);

        auto values = std::vector{1, 2, 3, 4, 5};

        auto state = graph.empty_state();
        x_ptr->initialize_state(state, values);
        graph.initialize_state(state);

        // the output is consistent with a "by hand" reduction over x
        auto value = std::reduce(x_ptr->begin(state), x_ptr->end(state), init, TestType());
        CHECK(r_ptr->view(state).front() == value);

        WHEN("We make changes to x") {
            // a few redundant changes
            x_ptr->set_value(state, 0, 10);
            x_ptr->set_value(state, 0, 5);
            x_ptr->set_value(state, 0, 10);

            // and one single
            x_ptr->set_value(state, 4, 4);

            graph.propagate(state, graph.descendants(state, {x_ptr}));

            // the output is consistent with a "by hand" reduction over x
            auto value = std::reduce(x_ptr->begin(state), x_ptr->end(state), init, TestType());
            CHECK(r_ptr->view(state).front() == value);

            AND_WHEN("We commit") {
                graph.commit(state, graph.descendants(state, {x_ptr}));

                // the output is consistent with a "by hand" reduction over x
                auto value = std::reduce(x_ptr->begin(state), x_ptr->end(state), init, TestType());
                CHECK(r_ptr->view(state).front() == value);
            }

            AND_WHEN("We revert") {
                graph.revert(state, graph.descendants(state, {x_ptr}));

                // the output is consistent with a "by hand" reduction over x
                auto value = std::reduce(x_ptr->begin(state), x_ptr->end(state), init, TestType());
                CHECK(r_ptr->view(state).front() == value);
            }
        }
    }
}

TEST_CASE("PartialReduceNode - PartialProdNode") {
    auto graph = Graph();
    GIVEN("A 3D array with shape (2, 2, 2) and partially reduce over the axes") {
        std::vector<double> values = {0, 1, 2, 3, 4, 5, 6, 7};
        auto ptr =
                graph.emplace_node<ConstantNode>(values, std::initializer_list<ssize_t>{2, 2, 2});
        auto r_ptr_0 = graph.emplace_node<PartialProdNode>(ptr, 0);
        auto r_ptr_1 = graph.emplace_node<PartialProdNode>(ptr, 1);
        auto r_ptr_2 = graph.emplace_node<PartialProdNode>(ptr, 2);

        graph.emplace_node<ArrayValidationNode>(r_ptr_0);
        graph.emplace_node<ArrayValidationNode>(r_ptr_1);
        graph.emplace_node<ArrayValidationNode>(r_ptr_2);

        CHECK(r_ptr_0->ndim() == 2);
        CHECK(r_ptr_1->ndim() == 2);
        CHECK(r_ptr_2->ndim() == 2);

        WHEN("We make a state") {
            auto state = graph.initialize_state();

            THEN("The partial reduction has the values and shapes we expect") {
                CHECK(r_ptr_0->ndim() == 2);
                CHECK(r_ptr_0->size(state) == 4);
                CHECK(r_ptr_0->shape(state).size() == 2);

                CHECK(r_ptr_1->ndim() == 2);
                CHECK(r_ptr_1->size(state) == 4);
                CHECK(r_ptr_1->shape(state).size() == 2);

                CHECK(r_ptr_2->ndim() == 2);
                CHECK(r_ptr_2->size(state) == 4);
                CHECK(r_ptr_2->shape(state).size() == 2);

                /// Check with
                /// A = np.arange(8).reshape((2, 2, 2))
                /// np.prod(A, axis=0)
                CHECK_THAT(r_ptr_0->view(state), RangeEquals({0, 5, 12, 21}));
                /// np.prod(A, axis=1)
                CHECK_THAT(r_ptr_1->view(state), RangeEquals({0, 3, 24, 35}));
                /// np.prod(A, axis=2)
                CHECK_THAT(r_ptr_2->view(state), RangeEquals({0, 6, 20, 42}));
            }
        }
    }
}

TEST_CASE("PartialReduceNode - PartialSumNode") {
    auto graph = Graph();
    GIVEN("A 3D array with shape (2, 2, 2) and partially reduce over the axes ") {
        std::vector<double> values = {0, 1, 2, 3, 4, 5, 6, 7};
        auto ptr =
                graph.emplace_node<ConstantNode>(values, std::initializer_list<ssize_t>{2, 2, 2});
        auto r_ptr_0 = graph.emplace_node<PartialReduceNode<std::plus<double>>>(ptr, 0);
        auto r_ptr_1 = graph.emplace_node<PartialReduceNode<std::plus<double>>>(ptr, 1);
        auto r_ptr_2 = graph.emplace_node<PartialReduceNode<std::plus<double>>>(ptr, 2);

        THEN("The dimensions of the partial reductions are correct") {
            CHECK(r_ptr_0->ndim() == 2);
            CHECK(r_ptr_1->ndim() == 2);
            CHECK(r_ptr_2->ndim() == 2);
        }

        WHEN("We make a state") {
            auto state = graph.initialize_state();

            THEN("The partial reduction has the values and shapes we expect") {
                CHECK(r_ptr_0->ndim() == 2);
                CHECK(r_ptr_0->size(state) == 4);
                CHECK(r_ptr_0->shape(state).size() == 2);

                CHECK(r_ptr_1->ndim() == 2);
                CHECK(r_ptr_1->size(state) == 4);
                CHECK(r_ptr_1->shape(state).size() == 2);

                CHECK(r_ptr_2->ndim() == 2);
                CHECK(r_ptr_2->size(state) == 4);
                CHECK(r_ptr_2->shape(state).size() == 2);

                /// Check with
                /// A = np.arange(8).reshape((2, 2, 2))
                /// np.sum(A, axis=0)
                CHECK_THAT(r_ptr_0->view(state), RangeEquals({4, 6, 8, 10}));
                /// np.sum(A, axis=1)
                CHECK_THAT(r_ptr_1->view(state), RangeEquals({2, 4, 10, 12}));
                /// np.sum(A, axis=2)
                CHECK_THAT(r_ptr_2->view(state), RangeEquals({1, 5, 9, 13}));
            }
        }
    }

    GIVEN("A 3D binary array with shape (2, 3, 2) and partially reduce over the axes") {
        auto ptr = graph.emplace_node<BinaryNode>(std::initializer_list<ssize_t>{2, 3, 2});
        auto r_ptr_0 = graph.emplace_node<PartialReduceNode<std::plus<double>>>(ptr, 0);
        auto r_ptr_1 = graph.emplace_node<PartialReduceNode<std::plus<double>>>(ptr, 1);
        auto r_ptr_2 = graph.emplace_node<PartialReduceNode<std::plus<double>>>(ptr, 2);

        THEN("The dimensions of the partial reductions are correct") {
            CHECK(r_ptr_0->ndim() == 2);
            CHECK(r_ptr_1->ndim() == 2);
            CHECK(r_ptr_2->ndim() == 2);
        }

        WHEN("We make a state") {
            auto state = graph.initialize_state();

            THEN("The partial reduction has the values and shapes we expect") {
                CHECK(r_ptr_0->ndim() == 2);
                CHECK(r_ptr_0->size(state) == 6);
                CHECK(r_ptr_0->shape(state).size() == 2);

                CHECK(r_ptr_1->ndim() == 2);
                CHECK(r_ptr_1->size(state) == 4);
                CHECK(r_ptr_1->shape(state).size() == 2);

                CHECK(r_ptr_2->ndim() == 2);
                CHECK(r_ptr_2->size(state) == 6);
                CHECK(r_ptr_2->shape(state).size() == 2);

                CHECK(std::ranges::equal(r_ptr_0->view(state), std::vector<double>(6, 0)));
                CHECK(std::ranges::equal(r_ptr_1->view(state), std::vector<double>(4, 0)));
                CHECK(std::ranges::equal(r_ptr_2->view(state), std::vector<double>(6, 0)));

                AND_WHEN("We update a variable") {
                    ptr->flip(state, 4);

                    // manually propagate
                    ptr->propagate(state);
                    r_ptr_0->propagate(state);
                    r_ptr_1->propagate(state);
                    r_ptr_2->propagate(state);

                    THEN("The partial reductions are updated correctly") {
                        CHECK_THAT(r_ptr_0->view(state), RangeEquals({0, 0, 0, 0, 1, 0}));
                        CHECK_THAT(r_ptr_1->view(state), RangeEquals({1, 0, 0, 0}));
                        CHECK_THAT(r_ptr_2->view(state), RangeEquals({0, 0, 1, 0, 0, 0}));
                    }

                    AND_WHEN("We commit") {
                        ptr->commit(state);
                        r_ptr_0->commit(state);
                        r_ptr_1->commit(state);
                        r_ptr_2->commit(state);

                        THEN("The values are maintained") {
                            CHECK_THAT(r_ptr_0->view(state), RangeEquals({0, 0, 0, 0, 1, 0}));
                            CHECK_THAT(r_ptr_1->view(state), RangeEquals({1, 0, 0, 0}));
                            CHECK_THAT(r_ptr_2->view(state), RangeEquals({0, 0, 1, 0, 0, 0}));
                        }
                    }

                    AND_WHEN("We revert") {
                        ptr->revert(state);
                        r_ptr_0->revert(state);
                        r_ptr_1->revert(state);
                        r_ptr_2->revert(state);

                        THEN("The values are reverted") {
                            CHECK(std::ranges::equal(r_ptr_0->view(state),
                                                     std::vector<double>(6, 0)));
                            CHECK(std::ranges::equal(r_ptr_1->view(state),
                                                     std::vector<double>(4, 0)));
                            CHECK(std::ranges::equal(r_ptr_2->view(state),
                                                     std::vector<double>(6, 0)));
                        }
                    }
                }
            }
        }
    }

    GIVEN("A 3D array, index it by slice, int and slice and we take partial traces") {
        std::vector<double> values = {0, 1, 2, 3, 4, 5, 6, 7};
        auto array_ptr =
                graph.emplace_node<ConstantNode>(values, std::initializer_list<ssize_t>{2, 2, 2});
        auto ptr = graph.emplace_node<BasicIndexingNode>(array_ptr, Slice(0, 2), 1, Slice(0, 2));

        // then there are only 2 possible reductions
        auto r_ptr_0 = graph.emplace_node<PartialReduceNode<std::plus<double>>>(ptr, 0);
        auto r_ptr_1 = graph.emplace_node<PartialReduceNode<std::plus<double>>>(ptr, 1);

        THEN("The dimensions of the partial reductions are correct") {
            CHECK(ptr->ndim() == 2);
            CHECK(r_ptr_0->ndim() == 1);
            CHECK(r_ptr_1->ndim() == 1);
        }

        WHEN("We make a state") {
            auto state = graph.initialize_state();

            THEN("The partial reduction has the values and shapes we expect") {
                CHECK(r_ptr_0->ndim() == 1);
                CHECK(r_ptr_0->size(state) == 2);
                CHECK(r_ptr_0->shape(state).size() == 1);

                CHECK(r_ptr_1->ndim() == 1);
                CHECK(r_ptr_1->size(state) == 2);
                CHECK(r_ptr_1->shape(state).size() == 1);

                /// Check with
                /// A = np.arange(8).reshape((2, 2, 2))
                /// B = A[:, 1, :]
                /// np.sum(B, axis=0)
                CHECK_THAT(r_ptr_0->view(state), RangeEquals({8, 10}));
                /// np.sum(B, axis=1)
                CHECK_THAT(r_ptr_1->view(state), RangeEquals({5, 13}));
            }
        }
    }
}

TEMPLATE_TEST_CASE("ReduceNode", "", functional::max<double>, functional::min<double>,
                   std::logical_and<double>, std::logical_or<double>, std::multiplies<double>,
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

        THEN("The constant is the operand") {
            REQUIRE(r_ptr->operands().size() == r_ptr->predecessors().size());
            CHECK(static_cast<Array*>(a_ptr) == r_ptr->operands()[0]);
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

        THEN("The constant is the operand") {
            CHECK(r_ptr->operands().size() == r_ptr->predecessors().size());
            CHECK(static_cast<const Array*>(a_ptr) == r_ptr->operands()[0]);
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

        THEN("The constant is the operand") {
            CHECK(r_ptr->operands().size() == r_ptr->predecessors().size());
            CHECK(static_cast<const Array*>(a_ptr) == r_ptr->operands()[0]);
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

        THEN("The set is the operand") {
            CHECK(r_ptr->operands().size() == r_ptr->predecessors().size());
            CHECK(static_cast<const Array*>(a_ptr) == r_ptr->operands()[0]);
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

TEST_CASE("ReduceNode - AllNode/AnyNode") {
    auto graph = Graph();

    GIVEN("x = BinaryNode({5}), y = x.any()") {
        auto x_ptr = graph.emplace_node<BinaryNode>(std::vector<ssize_t>{5});
        auto y_ptr = graph.emplace_node<AllNode>(x_ptr);
        auto z_ptr = graph.emplace_node<AnyNode>(x_ptr);

        graph.emplace_node<ArrayValidationNode>(y_ptr);
        graph.emplace_node<ArrayValidationNode>(z_ptr);

        THEN("y,z are logical and scalar") {
            CHECK(y_ptr->logical());
            CHECK(y_ptr->ndim() == 0);
            CHECK(z_ptr->logical());
            CHECK(z_ptr->ndim() == 0);
        }

        WHEN("x == [0, 0, 0, 0, 0]") {
            auto state = graph.empty_state();
            x_ptr->initialize_state(state, {0, 0, 0, 0, 0});
            graph.initialize_state(state);

            THEN("y is false and z is false") {
                CHECK_THAT(y_ptr->view(state), RangeEquals({false}));
                CHECK_THAT(z_ptr->view(state), RangeEquals({false}));
            }

            AND_WHEN("x is updated to [0, 0, 1, 0, 0]") {
                x_ptr->set_value(state, 2, 1);
                graph.propagate(state, {x_ptr, y_ptr, z_ptr});

                THEN("y is false and z is true") {
                    CHECK_THAT(y_ptr->view(state), RangeEquals({false}));
                    CHECK_THAT(z_ptr->view(state), RangeEquals({true}));
                }

                graph.commit(state, {x_ptr, y_ptr, z_ptr});

                AND_WHEN("x is updated to [0, 0, 0, 0, 0]") {
                    x_ptr->set_value(state, 2, 0);
                    graph.propagate(state, {x_ptr, y_ptr, z_ptr});

                    THEN("y is false and z is false") {
                        CHECK_THAT(y_ptr->view(state), RangeEquals({false}));
                        CHECK_THAT(z_ptr->view(state), RangeEquals({false}));
                    }
                }
            }
        }

        WHEN("x == [0, 0, 1, 0, 0]") {
            auto state = graph.empty_state();
            x_ptr->initialize_state(state, {0, 0, 1, 0, 0});
            graph.initialize_state(state);

            THEN("y is false and z is true") {
                CHECK_THAT(y_ptr->view(state), RangeEquals({false}));
                CHECK_THAT(z_ptr->view(state), RangeEquals({true}));
            }
        }
    }

    GIVEN("x = [], y = x.all(), z = x.any()") {
        auto x_ptr = graph.emplace_node<BinaryNode>(std::vector<ssize_t>{0});
        auto y_ptr = graph.emplace_node<AllNode>(x_ptr);
        auto z_ptr = graph.emplace_node<AnyNode>(x_ptr);

        graph.emplace_node<ArrayValidationNode>(y_ptr);
        graph.emplace_node<ArrayValidationNode>(z_ptr);

        THEN("y,z are logical and scalar") {
            CHECK(y_ptr->logical());
            CHECK(y_ptr->ndim() == 0);
            CHECK(z_ptr->logical());
            CHECK(z_ptr->ndim() == 0);
        }

        auto state = graph.initialize_state();

        THEN("y and not z") {
            CHECK_THAT(y_ptr->view(state), RangeEquals({true}));
            CHECK_THAT(z_ptr->view(state), RangeEquals({false}));
        }
    }
}

TEST_CASE("ReduceNode - MaxNode/MinNode") {
    auto graph = Graph();

    GIVEN("A list node with a min and max node over it") {
        auto list_ptr = graph.emplace_node<ListNode>(5, 0, 5);

        // chose init values out of range - we test that init works correctly
        // in the ReduceNode tests
        auto max_ptr = graph.emplace_node<MaxNode>(list_ptr, -1);
        auto min_ptr = graph.emplace_node<MinNode>(list_ptr, 6);

        graph.emplace_node<ArrayValidationNode>(max_ptr);
        graph.emplace_node<ArrayValidationNode>(min_ptr);

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

    GIVEN("A dynamic node") {
        auto a_ptr = graph.emplace_node<SetNode>(4);

        THEN("We cannot construct a reduce node without an initial value") {
            CHECK_THROWS(graph.emplace_node<MaxNode>(a_ptr));
            CHECK_THROWS(graph.emplace_node<MinNode>(a_ptr));
            CHECK(a_ptr->successors().size() == 0);  // no side effects
        }
    }
}

TEST_CASE("ReduceNode - MaxNode") {
    auto graph = Graph();

    GIVEN("x = IntegerNode(3, -5, 2), y = x.max()") {
        auto x_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{3}, -5, 2);
        auto y_ptr = graph.emplace_node<MaxNode>(x_ptr);

        THEN("y's min/max/integral are as expected") {
            CHECK(y_ptr->min() == -5);
            CHECK(y_ptr->max() == 2);
            CHECK(y_ptr->integral());

            // check that the cache is populated with minmax
            Array::cache_type<std::pair<double, double>> cache;
            y_ptr->minmax(cache);
            // the output of a node depends on the inputs, so it shows
            // up in cache
            CHECK(cache.contains(y_ptr));
            // mutating the cache should also mutate the output
            cache[y_ptr].first = -1000;
            CHECK(y_ptr->minmax(cache).first == -1000);
            CHECK(y_ptr->minmax().first == -5);  // ignores the cache
        }
    }

    GIVEN("x = IntegerNode(3, -5, 2), y = x.max(init=-.5)") {
        auto x_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{3}, -5, 2);
        auto y_ptr = graph.emplace_node<MaxNode>(x_ptr, -.5);

        THEN("y's min/max/integral are as expected") {
            CHECK(y_ptr->min() == -.5);
            CHECK(y_ptr->max() == 2);
            CHECK(!y_ptr->integral());
        }
    }
}

TEST_CASE("ReduceNode - MinNode") {
    auto graph = Graph();

    GIVEN("x = IntegerNode(3, -5, 2), y = x.min()") {
        auto x_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{3}, -5, 2);
        auto y_ptr = graph.emplace_node<MinNode>(x_ptr);

        THEN("y's min/max/integral are as expected") {
            CHECK(y_ptr->min() == -5);
            CHECK(y_ptr->max() == 2);
            CHECK(y_ptr->integral());
        }
    }

    GIVEN("x = IntegerNode(3, -5, 2), y = x.min(init=-.5)") {
        auto x_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{3}, -5, 2);
        auto y_ptr = graph.emplace_node<MinNode>(x_ptr, -.5);

        THEN("y's min/max/integral are as expected") {
            CHECK(y_ptr->min() == -5);
            CHECK(y_ptr->max() == -.5);
            CHECK(!y_ptr->integral());
        }
    }
}

TEST_CASE("ReduceNode - ProdNode") {
    auto graph = Graph();

    GIVEN("x = IntegerNode(3, -5, 2), y = x.prod()") {
        auto x_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{3}, -5, 2);
        auto y_ptr = graph.emplace_node<ProdNode>(x_ptr);

        THEN("y's min/max/integral are as expected") {
            CHECK(y_ptr->min() == -5 * -5 * -5);
            CHECK(y_ptr->max() == -5 * -5 * 2);
            CHECK(y_ptr->integral());
        }
    }

    GIVEN("x = IntegerNode(3, -5, 2), y = x.prod(init=-.5)") {
        auto x_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{3}, -5, 2);
        auto y_ptr = graph.emplace_node<ProdNode>(x_ptr, -.5);

        THEN("y's min/max/integral are as expected") {
            CHECK(y_ptr->min() == -5 * -5 * 2 * -.5);
            CHECK(y_ptr->max() == -5 * -5 * -5 * -.5);
            CHECK(!y_ptr->integral());
        }
    }

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

TEST_CASE("ReduceNode - SumNode") {
    auto graph = Graph();

    GIVEN("x = IntegerNode(3, -5, 2), y = x.sum()") {
        auto x_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{3}, -5, 2);
        auto y_ptr = graph.emplace_node<SumNode>(x_ptr);

        THEN("y's min/max/integral are as expected") {
            CHECK(y_ptr->min() == -5 + -5 + -5);
            CHECK(y_ptr->max() == 2 + 2 + 2);
            CHECK(y_ptr->integral());
        }
    }

    GIVEN("x = IntegerNode(3, -5, 2), y = x.sum(init=-.5)") {
        auto x_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{3}, -5, 2);
        auto y_ptr = graph.emplace_node<SumNode>(x_ptr, -.5);

        THEN("y's min/max/integral are as expected") {
            CHECK(y_ptr->min() == -5 + -5 + -5 + -.5);
            CHECK(y_ptr->max() == 2 + 2 + 2 + -.5);
            CHECK(!y_ptr->integral());
        }
    }

    GIVEN("a = [0, 3, 2], x = SetNode(3), y = a[x].sum()") {
        auto a_ptr = graph.emplace_node<ConstantNode>(std::vector{0, 3, 2});
        auto x_ptr = graph.emplace_node<SetNode>(3);
        auto ax_ptr = graph.emplace_node<AdvancedIndexingNode>(a_ptr, x_ptr);
        auto y_ptr = graph.emplace_node<SumNode>(ax_ptr);

        THEN("y's min/max/integral are as expected") {
            CHECK(y_ptr->min() == 0);
            CHECK(y_ptr->max() == 9);
            CHECK(y_ptr->integral());
        }
    }

    GIVEN("a = [1, 3, 2], x = SetNode(3), y = a[x].sum()") {
        auto a_ptr = graph.emplace_node<ConstantNode>(std::vector{1, 3, 2});
        auto x_ptr = graph.emplace_node<SetNode>(3);
        auto ax_ptr = graph.emplace_node<AdvancedIndexingNode>(a_ptr, x_ptr);
        auto y_ptr = graph.emplace_node<SumNode>(ax_ptr);

        THEN("y's min/max/integral are as expected") {
            CHECK(y_ptr->min() == 0);  // can be 0 because x can be empty
            CHECK(y_ptr->max() == 9);
            CHECK(y_ptr->integral());
        }
    }

    GIVEN("a = [2, 3, 2], x = SetNode(3, 2, 3), y = a[x].sum()") {
        auto a_ptr = graph.emplace_node<ConstantNode>(std::vector{2, 3, 2});
        auto x_ptr = graph.emplace_node<SetNode>(3, 2, 3);
        auto ax_ptr = graph.emplace_node<AdvancedIndexingNode>(a_ptr, x_ptr);
        auto y_ptr = graph.emplace_node<SumNode>(ax_ptr);

        THEN("y's min/max/integral are as expected") {
            CHECK(y_ptr->min() == 4);  // Set has at least 2 elements
            CHECK(y_ptr->max() == 9);
            CHECK(y_ptr->integral());
        }
    }

    GIVEN("a = [1, -3, 2], x = SetNode(3), y = a[x].sum()") {
        auto a_ptr = graph.emplace_node<ConstantNode>(std::vector{1, -3, 2});
        auto x_ptr = graph.emplace_node<SetNode>(3);
        auto ax_ptr = graph.emplace_node<AdvancedIndexingNode>(a_ptr, x_ptr);
        auto y_ptr = graph.emplace_node<SumNode>(ax_ptr);

        THEN("y's min/max/integral are as expected") {
            CHECK(y_ptr->min() == -9);  // set has at most 3 elements
            CHECK(y_ptr->max() == 6);
            CHECK(y_ptr->integral());
        }
    }

    GIVEN("a = [-1, -3, -2], x = SetNode(3), y = a[x].sum()") {
        auto a_ptr = graph.emplace_node<ConstantNode>(std::vector{-1, -3, -2});
        auto x_ptr = graph.emplace_node<SetNode>(3);
        auto ax_ptr = graph.emplace_node<AdvancedIndexingNode>(a_ptr, x_ptr);
        auto y_ptr = graph.emplace_node<SumNode>(ax_ptr);

        THEN("y's min/max/integral are as expected") {
            CHECK(y_ptr->min() == -9);
            CHECK(y_ptr->max() == 0);
            CHECK(y_ptr->integral());
        }
    }

    GIVEN("A set reduced") {
        auto a_ptr = graph.emplace_node<SetNode>(4);
        auto r_ptr = graph.emplace_node<SumNode>(a_ptr);

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
                    lhs = lhs + a_ptr->view(state)[i];
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
                        lhs = lhs + a_ptr->view(state)[i];
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
                        lhs = lhs + a_ptr->view(state)[i];
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

}  // namespace dwave::optimization

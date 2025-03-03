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

// NOTE: divides test is disabled because the template-tests have invalid denominators.
TEMPLATE_TEST_CASE("BinaryOpNode", "",
                   // std::divides<double>,
                   std::equal_to<double>, std::less_equal<double>, std::plus<double>,
                   std::minus<double>, functional::modulus<double>, std::multiplies<double>,
                   functional::max<double>, functional::min<double>, std::logical_and<double>,
                   std::logical_or<double>, functional::logical_xor<double>) {
    auto graph = Graph();

    auto func = TestType();

    GIVEN("Two scalar constants operated on") {
        auto a_ptr = graph.emplace_node<ConstantNode>(5);
        auto b_ptr = graph.emplace_node<ConstantNode>(6);

        auto p_ptr = graph.emplace_node<BinaryOpNode<TestType>>(a_ptr, b_ptr);

        graph.emplace_node<ArrayValidationNode>(p_ptr);

        THEN("a and b are the operands") {
            REQUIRE(std::ranges::equal(p_ptr->operands(), std::vector<Array*>{a_ptr, b_ptr}));

            // we can cast to a non-const ptr if we're not const
            CHECK(static_cast<Array*>(p_ptr->operands()[0]) == static_cast<Array*>(a_ptr));
        }

        THEN("The state is deterministic") {
            CHECK(p_ptr->deterministic_state());
        }

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

        graph.emplace_node<ArrayValidationNode>(p_ptr);

        THEN("The shape is also a 1D array") {
            CHECK(p_ptr->ndim() == 1);
            CHECK(p_ptr->size() == 4);
        }

        THEN("a and b are the operands") {
            CHECK(std::ranges::equal(p_ptr->operands(), std::vector<Array*>{a_ptr, b_ptr}));
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

        graph.emplace_node<ArrayValidationNode>(p_ptr);

        THEN("The shape is also a 1D array") {
            CHECK(p_ptr->ndim() == 1);
            CHECK(p_ptr->size() == 4);
        }

        THEN("a and b are the operands") {
            CHECK(std::ranges::equal(p_ptr->operands(), std::vector<Array*>{a_ptr, b_ptr}));
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

        graph.emplace_node<ArrayValidationNode>(p_ptr);

        THEN("The shape is also a 1D array") {
            CHECK(p_ptr->ndim() == 1);
            CHECK(p_ptr->size() == 4);
        }

        THEN("a and b are the operands") {
            CHECK(std::ranges::equal(p_ptr->operands(), std::vector<Array*>{a_ptr, b_ptr}));
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

TEST_CASE("BinaryOpNode - LessEqualNode") {
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

    GIVEN("x = Integer(), y = List(5), le = x <= y, ge = y <= x") {
        auto x_ptr = graph.emplace_node<IntegerNode>();
        auto y_ptr = graph.emplace_node<ListNode>(5);
        auto le_ptr = graph.emplace_node<LessEqualNode>(x_ptr, y_ptr);
        auto ge_ptr = graph.emplace_node<LessEqualNode>(y_ptr, x_ptr);

        THEN("We have the shape we expect") {
            CHECK_THAT(le_ptr->shape(), RangeEquals({5}));
            CHECK_THAT(ge_ptr->shape(), RangeEquals({5}));
        }

        // let's also toss an ArrayValidationNode on there to do most of the
        // testing for us
        graph.emplace_node<ArrayValidationNode>(le_ptr);
        graph.emplace_node<ArrayValidationNode>(ge_ptr);

        WHEN("We initialize x = 3, y = [0, 1, 2, 3, 4]") {
            auto state = graph.empty_state();
            x_ptr->initialize_state(state, {3});
            y_ptr->initialize_state(state, {0, 1, 2, 3, 4});
            graph.initialize_state(state);

            THEN("le == x <= y == [false, false, false, true, true]") {
                CHECK_THAT(le_ptr->view(state), RangeEquals({0, 0, 0, 1, 1}));
            }

            THEN("ge == y <= x == [true, true, true, true, false]") {
                CHECK_THAT(ge_ptr->view(state), RangeEquals({1, 1, 1, 1, 0}));
            }

            AND_WHEN("We then set x = 2") {
                x_ptr->set_value(state, 0, 2);
                graph.propagate(state, graph.descendants(state, {x_ptr}));

                THEN("le == x <= y == [false, false, true, true, true]") {
                    CHECK_THAT(le_ptr->view(state), RangeEquals({0, 0, 1, 1, 1}));
                }

                THEN("ge == y <= x == [true, true, true, false, false]") {
                    CHECK_THAT(ge_ptr->view(state), RangeEquals({1, 1, 1, 0, 0}));
                }

                AND_WHEN("We commit") {
                    graph.commit(state, graph.descendants(state, {x_ptr}));

                    THEN("le == x <= y == [false, false, true, true, true]") {
                        CHECK_THAT(le_ptr->view(state), RangeEquals({0, 0, 1, 1, 1}));
                    }

                    THEN("ge == y <= x == [true, true, true, false, false]") {
                        CHECK_THAT(ge_ptr->view(state), RangeEquals({1, 1, 1, 0, 0}));
                    }
                }

                AND_WHEN("We revert") {
                    graph.revert(state, graph.descendants(state, {x_ptr}));

                    THEN("le == x <= y == [false, false, false, true, true]") {
                        CHECK_THAT(le_ptr->view(state), RangeEquals({0, 0, 0, 1, 1}));
                    }

                    THEN("ge == y <= x == [true, true, true, true, false]") {
                        CHECK_THAT(ge_ptr->view(state), RangeEquals({1, 1, 1, 1, 0}));
                    }
                }
            }
        }
    }

    GIVEN("x = Integer(), y = List(5, 2, 4), le = x <= y, ge = y <= x") {
        auto x_ptr = graph.emplace_node<IntegerNode>();
        auto y_ptr = graph.emplace_node<ListNode>(5, 2, 4);
        auto le_ptr = graph.emplace_node<LessEqualNode>(x_ptr, y_ptr);
        auto ge_ptr = graph.emplace_node<LessEqualNode>(y_ptr, x_ptr);

        THEN("We have the shape we expect") {
            CHECK_THAT(le_ptr->shape(), RangeEquals({-1}));
            CHECK_THAT(ge_ptr->shape(), RangeEquals({-1}));

            // derives its size from the dynamic node
            CHECK(le_ptr->sizeinfo() == SizeInfo(y_ptr));
            CHECK(ge_ptr->sizeinfo() == SizeInfo(y_ptr));
        }

        // let's also toss an ArrayValidationNode on there to do most of the
        // testing for us
        graph.emplace_node<ArrayValidationNode>(le_ptr);
        graph.emplace_node<ArrayValidationNode>(ge_ptr);

        auto state = graph.initialize_state();

        WHEN("We initialize x = 3, y = [4, 0, 3]") {
            auto state = graph.empty_state();
            x_ptr->initialize_state(state, {3});
            y_ptr->initialize_state(state, {4, 0, 3});
            graph.initialize_state(state);

            THEN("le == x <= y == [true, false, true]") {
                CHECK_THAT(le_ptr->view(state), RangeEquals({1, 0, 1}));
            }

            THEN("ge == y <= x == [false, true, true]") {
                CHECK_THAT(ge_ptr->view(state), RangeEquals({0, 1, 1}));
            }

            AND_WHEN("We mutate y to [4, 3, 0, 1]") {
                y_ptr->shrink(state);
                y_ptr->exchange(state, 1, 2);
                y_ptr->grow(state);
                y_ptr->grow(state);
                graph.propagate(state, graph.descendants(state, {y_ptr}));

                // the 1 is actually an implementation detail but let's make that
                // assumption for the purpose of these tests
                CHECK_THAT(y_ptr->view(state), RangeEquals({4, 3, 0, 1}));

                THEN("le == x <= y == [true, true, false, false]") {
                    CHECK_THAT(le_ptr->view(state), RangeEquals({1, 1, 0, 0}));
                }

                THEN("ge == y <= x == [false, true, true, true]") {
                    CHECK_THAT(ge_ptr->view(state), RangeEquals({0, 1, 1, 1}));
                }

                AND_WHEN("We commit") {
                    graph.commit(state, graph.descendants(state, {y_ptr}));

                    THEN("le == x <= y == [true, true, false, false]") {
                        CHECK_THAT(le_ptr->view(state), RangeEquals({1, 1, 0, 0}));
                    }

                    THEN("ge == y <= x == [false, true, true, true]") {
                        CHECK_THAT(ge_ptr->view(state), RangeEquals({0, 1, 1, 1}));
                    }
                }

                AND_WHEN("We revert") {
                    graph.revert(state, graph.descendants(state, {y_ptr}));

                    THEN("le == x <= y == [true, false, true]") {
                        CHECK_THAT(le_ptr->view(state), RangeEquals({1, 0, 1}));
                    }

                    THEN("ge == y <= x == [false, true, true]") {
                        CHECK_THAT(ge_ptr->view(state), RangeEquals({0, 1, 1}));
                    }
                }
            }
        }
    }
}

TEST_CASE("BinaryOpNode - MultiplyNode") {
    auto graph = Graph();

    GIVEN("x = IntegerNode(-5, 5), a = 3, y = x * a") {
        auto x_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -5, 5);
        auto a_ptr = graph.emplace_node<ConstantNode>(3);

        auto y_ptr = graph.emplace_node<MultiplyNode>(x_ptr, a_ptr);

        THEN("y's max/min/integral are as expected") {
            CHECK(y_ptr->max() == 15);
            CHECK(y_ptr->min() == -15);
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
            CHECK(y_ptr->minmax().first == -15);  // ignores the cache
        }
    }

    GIVEN("x = IntegerNode(-5, 5), a = -3, y = x * a") {
        auto x_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -5, 5);
        auto a_ptr = graph.emplace_node<ConstantNode>(-3);

        auto y_ptr = graph.emplace_node<MultiplyNode>(x_ptr, a_ptr);

        THEN("y's max/min/integral are as expected") {
            CHECK(y_ptr->max() == 15);
            CHECK(y_ptr->min() == -15);
            CHECK(y_ptr->integral());
        }
    }
}

TEST_CASE("BinaryOpNode - DivideNode") {
    auto graph = Graph();

    GIVEN("x = IntegerNode(-5, 5), a = 3, y = x / a") {
        auto x_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -5, 5);
        auto a_ptr = graph.emplace_node<ConstantNode>(3);

        auto y_ptr = graph.emplace_node<DivideNode>(x_ptr, a_ptr);

        THEN("y's max/min/integral are as expected") {
            CHECK(std::abs(y_ptr->max() - 5.0 / 3.0) < 10e-16);
            CHECK(std::abs(y_ptr->min() - -5.0 / 3.0) < 10e-16);
            CHECK_FALSE(y_ptr->integral());
        }
    }

    GIVEN("a = 0, x = IntegerNode(1, 5), y = a / x") {
        auto a_ptr = graph.emplace_node<ConstantNode>(0);
        auto x_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, 1, 5);

        auto y_ptr = graph.emplace_node<DivideNode>(a_ptr, x_ptr);

        THEN("y's max/min/integral are as expected") {
            CHECK(y_ptr->max() == 0);
            CHECK(y_ptr->min() == 0);
            CHECK_FALSE(y_ptr->integral());
        }
    }

    GIVEN("x = IntegerNode(-5, 5), a = -3, y = x / a") {
        auto x_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -5, 5);
        auto a_ptr = graph.emplace_node<ConstantNode>(-3);

        auto y_ptr = graph.emplace_node<DivideNode>(x_ptr, a_ptr);

        THEN("y's max/min/integral are as expected") {
            CHECK(y_ptr->max() == -5.0 / -3.0);
            CHECK(y_ptr->min() == 5.0 / -3.0);
            CHECK_FALSE(y_ptr->integral());
        }
    }
    GIVEN("x = IntegerNode(-5, 5), a = 0, y = x / a") {
        auto x_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -5, 5);
        auto a_ptr = graph.emplace_node<ConstantNode>(0);

        THEN("Check division-by-zero") {
            CHECK_THROWS(graph.emplace_node<DivideNode>(x_ptr, a_ptr));
        }
    }
    GIVEN("x = IntegerNode(-5, 5), a = IntegerNode(-5, 0), y = x / a") {
        auto x_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -5, 5);
        auto a_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -5, 0);

        THEN("Check division-by-zero") {
            CHECK_THROWS(graph.emplace_node<DivideNode>(x_ptr, a_ptr));
        }
    }
    GIVEN("x = IntegerNode(-5, 5), a = IntegerNode(0, 5), y = x / a") {
        auto x_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -5, 5);
        auto a_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, 0, 5);

        THEN("Check division-by-zero") {
            CHECK_THROWS(graph.emplace_node<DivideNode>(x_ptr, a_ptr));
        }
    }
}

TEST_CASE("BinaryOpNode - SubtractNode") {
    auto graph = Graph();

    GIVEN("x = IntegerNode(-5, 5), a = 3, y = x - a") {
        auto x_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -5, 5);
        auto a_ptr = graph.emplace_node<ConstantNode>(3);

        auto y_ptr = graph.emplace_node<SubtractNode>(x_ptr, a_ptr);

        THEN("y's max/min/integral are as expected") {
            CHECK(y_ptr->max() == 2);
            CHECK(y_ptr->min() == -8);
            CHECK(y_ptr->integral());
        }
    }

    GIVEN("x = IntegerNode(-5, 5), a = -3, y = x - a") {
        auto x_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -5, 5);
        auto a_ptr = graph.emplace_node<ConstantNode>(-3);

        auto y_ptr = graph.emplace_node<SubtractNode>(x_ptr, a_ptr);

        THEN("y's max/min/integral are as expected") {
            CHECK(y_ptr->max() == 8);
            CHECK(y_ptr->min() == -2);
            CHECK(y_ptr->integral());
        }
    }

    GIVEN("x = IntegerNode(-5, 5), a = 3.5, y = x - a") {
        auto x_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -5, 5);
        auto a_ptr = graph.emplace_node<ConstantNode>(3.5);

        auto y_ptr = graph.emplace_node<SubtractNode>(x_ptr, a_ptr);

        THEN("y's max/min/integral are as expected") {
            CHECK(y_ptr->max() == 1.5);
            CHECK(y_ptr->min() == -8.5);
            CHECK(!y_ptr->integral());
        }
    }

    GIVEN("x = IntegerNode(-5, 5), y = IntegerNode(-3, 4), z = x - y") {
        auto x_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -5, 5);
        auto y_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -3, 4);

        auto z_ptr = graph.emplace_node<SubtractNode>(x_ptr, y_ptr);

        THEN("z's max/min/integral are as expected") {
            CHECK(z_ptr->max() == 8);
            CHECK(z_ptr->min() == -9);
            CHECK(z_ptr->integral());
        }
    }
}

}  // namespace dwave::optimization

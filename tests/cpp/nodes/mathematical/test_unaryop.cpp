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

#include "catch2/catch_test_macros.hpp"
#include "catch2/catch_template_test_macros.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/nodes/collections.hpp"
#include "dwave-optimization/nodes/constants.hpp"
#include "dwave-optimization/nodes/indexing.hpp"
#include "dwave-optimization/nodes/mathematical.hpp"
#include "dwave-optimization/nodes/numbers.hpp"
#include "dwave-optimization/nodes/testing.hpp"

namespace dwave::optimization {

TEMPLATE_TEST_CASE("UnaryOpNode", "", functional::abs<double>, functional::logical<double>,
                   functional::square<double>, std::negate<double>, std::logical_not<double>) {
    auto graph = Graph();

    auto func = TestType();

    GIVEN("A constant scalar input") {
        auto a_ptr = graph.emplace_node<ConstantNode>(-5);

        auto p_ptr = graph.emplace_node<UnaryOpNode<TestType>>(a_ptr);

        THEN("The shape is also a scalar") {
            CHECK(p_ptr->ndim() == 0);
            CHECK(p_ptr->size() == 1);
        }

        THEN("The constant is the operand") {
            REQUIRE(p_ptr->operands().size() == p_ptr->predecessors().size());
            CHECK(static_cast<Array*>(a_ptr) == p_ptr->operands()[0]);
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

    GIVEN("A dynamic array input") {
        auto a_ptr = graph.emplace_node<ListNode>(5, 0, 5);

        auto p_ptr = graph.emplace_node<UnaryOpNode<TestType>>(a_ptr);

        graph.emplace_node<ArrayValidationNode>(p_ptr);

        THEN("The shape is also dynamic") {
            CHECK(p_ptr->dynamic());
            CHECK(p_ptr->ndim() == 1);
        }

        WHEN("We initialize the input node to be empty") {
            auto state = graph.empty_state();
            a_ptr->initialize_state(state, {});
            graph.initialize_state(state);

            THEN("The output has the value and shape we expect") {
                CHECK(p_ptr->size(state) == 0);
                CHECK(std::ranges::equal(p_ptr->view(state), std::vector<double>{}));
            }

            AND_WHEN("We grow the array and then propagate") {
                a_ptr->grow(state);
                a_ptr->grow(state);
                a_ptr->exchange(state, 0, 1);
                a_ptr->shrink(state);

                a_ptr->propagate(state);
                p_ptr->propagate(state);

                THEN("The output is what we expect") {
                    REQUIRE(p_ptr->size(state) == 1);
                    double val = func(a_ptr->view(state).front());
                    CHECK(std::ranges::equal(p_ptr->view(state), std::vector{val}));
                }

                AND_WHEN("We commit") {
                    a_ptr->commit(state);
                    p_ptr->commit(state);

                    THEN("The output is what we expect") {
                        REQUIRE(p_ptr->size(state) == 1);
                        double val = func(a_ptr->view(state).front());
                        CHECK(std::ranges::equal(p_ptr->view(state), std::vector{val}));
                    }
                }

                AND_WHEN("We revert") {
                    a_ptr->revert(state);
                    p_ptr->revert(state);

                    THEN("The output is what we expect") {
                        CHECK(p_ptr->size(state) == 0);
                        CHECK(std::ranges::equal(p_ptr->view(state), std::vector<double>{}));
                    }
                }
            }
        }
    }

    GIVEN("A 0-d integer decision input") {
        auto a_ptr = graph.emplace_node<IntegerNode>(std::span<const ssize_t>{}, -100,
                                                     100);  // Scalar output

        auto p_ptr = graph.emplace_node<UnaryOpNode<TestType>>(a_ptr);

        THEN("The integer is the operand") {
            CHECK(p_ptr->operands().size() == p_ptr->predecessors().size());
            CHECK(static_cast<const Array*>(a_ptr) == p_ptr->operands()[0]);
        }

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

        THEN("The integers node is the operand") {
            CHECK(p_ptr->operands().size() == p_ptr->predecessors().size());
            CHECK(static_cast<const Array*>(a_ptr) == p_ptr->operands()[0]);
        }

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

TEST_CASE("UnaryOpNode - AbsoluteNode") {
    auto graph = Graph();

    GIVEN("An integer variable with domain [-3, 2]") {
        auto i_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -3, 2);
        auto abs_ptr = graph.emplace_node<AbsoluteNode>(i_ptr);

        THEN("It has the min/max we expect") {
            CHECK(abs_ptr->min() == 0);
            CHECK(abs_ptr->max() == 3);
        }

        THEN("abs(i) is integral") { CHECK(abs_ptr->integral()); }
    }

    GIVEN("An integer variable with domain [-2, 4]") {
        auto i_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -2, 4);
        auto abs_ptr = graph.emplace_node<AbsoluteNode>(i_ptr);

        THEN("It has the min/max we expect") {
            CHECK(abs_ptr->min() == 0);
            CHECK(abs_ptr->max() == 4);
        }

        THEN("abs(i) is integral") { CHECK(abs_ptr->integral()); }
    }
}

TEST_CASE("UnaryOpNode - LogicalNode") {
    auto graph = Graph();
    GIVEN("A constant of mixed doubles and a negation of it") {
        auto c_ptr = graph.emplace_node<ConstantNode>(std::vector{-2., -1., 0., 1., 2., -.5, .5});
        auto logical_ptr = graph.emplace_node<LogicalNode>(c_ptr);

        THEN("NotNode is logical") {
            CHECK(logical_ptr->integral());
            CHECK(logical_ptr->max() == 1);
            CHECK(logical_ptr->min() == 0);
            CHECK(logical_ptr->logical());
        }

        THEN("The negation has the state we expect") {
            auto state = graph.initialize_state();
            CHECK(std::ranges::equal(logical_ptr->view(state),
                                     std::vector{true, true, false, true, true, true, true}));
        }
    }
}

TEST_CASE("UnaryOpNode - NegativeNode") {
    auto graph = Graph();
    GIVEN("An integer array and an asymmetric domain") {
        auto i_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{5}, -3, 8);
        auto ni_ptr = graph.emplace_node<NegativeNode>(i_ptr);

        THEN("Negative has the min/max we expect") {
            CHECK(i_ptr->min() == -3);
            CHECK(i_ptr->max() == 8);
            CHECK(ni_ptr->min() == -8);
            CHECK(ni_ptr->max() == 3);
        }

        THEN("Negative is also integral") { CHECK(ni_ptr->integral()); }
    }
}

TEST_CASE("UnaryOpNode - NotNode") {
    auto graph = Graph();
    GIVEN("A constant of mixed doubles and a negation of it") {
        auto c_ptr = graph.emplace_node<ConstantNode>(std::vector{-2., -1., 0., 1., 2., -.5, .5});
        auto nc_ptr = graph.emplace_node<NotNode>(c_ptr);

        THEN("NotNode is logical") {
            CHECK(nc_ptr->integral());
            CHECK(nc_ptr->max() == 1);
            CHECK(nc_ptr->min() == 0);
            CHECK(nc_ptr->logical());
        }

        THEN("The negation has the state we expect") {
            auto state = graph.initialize_state();
            CHECK(std::ranges::equal(nc_ptr->view(state),
                                     std::vector{false, false, true, false, false, false, false}));
        }
    }
}

TEST_CASE("UnaryOpNode - SquareNode") {
    auto graph = Graph();
    GIVEN("An integer with max domain") {
        auto i_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{});
        auto square_ptr = graph.emplace_node<SquareNode>(i_ptr);

        THEN("The min/max are expected") {
            CHECK(square_ptr->min() == 0);
            // we might consider capping this differently for integer types in the future
            CHECK(square_ptr->max() ==
                  static_cast<std::size_t>(2000000000) * static_cast<std::size_t>(2000000000));
        }
    }
}

}  // namespace dwave::optimization

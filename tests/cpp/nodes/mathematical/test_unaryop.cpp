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

// NOTE: square_root and log should also be included but the templated tests need to be updated first.
TEMPLATE_TEST_CASE("UnaryOpNode", "", functional::abs<double>, functional::expit<double>,
                   functional::logical<double>, functional::rint<double>, functional::square<double>,
                   std::negate<double>, std::logical_not<double>) {
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
                CHECK_THAT(p_ptr->view(state), RangeEquals(std::vector<double>{}));
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
                    CHECK_THAT(p_ptr->view(state), RangeEquals({val}));
                }

                AND_WHEN("We commit") {
                    a_ptr->commit(state);
                    p_ptr->commit(state);

                    THEN("The output is what we expect") {
                        REQUIRE(p_ptr->size(state) == 1);
                        double val = func(a_ptr->view(state).front());
                        CHECK_THAT(p_ptr->view(state), RangeEquals({val}));
                    }
                }

                AND_WHEN("We revert") {
                    a_ptr->revert(state);
                    p_ptr->revert(state);

                    THEN("The output is what we expect") {
                        CHECK(p_ptr->size(state) == 0);
                        CHECK_THAT(p_ptr->view(state), RangeEquals(std::vector<double>{}));
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

        THEN("It has the min/max/integrality we expect") {
            CHECK(abs_ptr->min() == 0);
            CHECK(abs_ptr->max() == 3);
            CHECK(abs_ptr->integral());

            // check that the cache is populated with minmax
            Array::cache_type<std::pair<double, double>> cache;
            abs_ptr->minmax(cache);
            // the output of a node depends on the inputs, so it shows
            // up in cache
            CHECK(cache.contains(abs_ptr));
            // mutating the cache should also mutate the output
            cache[abs_ptr].first = -1000;
            CHECK(abs_ptr->minmax(cache).first == -1000);
            CHECK(abs_ptr->minmax().first == 0);  // ignores the cache
        }
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

TEST_CASE("UnaryOpNode - ExpitNode") {
    auto graph = Graph();
    GIVEN("An arbitrary number") {
        double c = 3.0;
        auto c_ptr = graph.emplace_node<ConstantNode>(c);
        auto expit_ptr = graph.emplace_node<ExpitNode>(c_ptr);
        auto state = graph.initialize_state();
        CHECK(expit_ptr->min() == 1.0 / (1.0 + std::exp(-c)));
        CHECK(expit_ptr->max() == 1.0 / (1.0 + std::exp(-c)));
    }

    GIVEN("A negative number") {
        double c = -4.0;
        auto c_ptr = graph.emplace_node<ConstantNode>(c);
        auto expit_ptr = graph.emplace_node<ExpitNode>(c_ptr);
        auto state = graph.initialize_state();
        CHECK(expit_ptr->min() == 1.0 / (1.0 + std::exp(-c)));
        CHECK(expit_ptr->max() == 1.0 / (1.0 + std::exp(-c)));
    }

    GIVEN("A constant 1d array of doubles") {
        auto c_ptr = graph.emplace_node<ConstantNode>(std::vector{-6.0, -0.3, 0.0, 1.2, 5.6});
        auto expit_ptr = graph.emplace_node<ExpitNode>(c_ptr);

        THEN("The min/max are expected") {
            THEN("expit(x) is not integral") { CHECK_FALSE(expit_ptr->integral()); }
            CHECK(expit_ptr->min() == 1.0 / (1.0 + std::exp(-(-6.0))));
            CHECK(expit_ptr->max() == 1.0 / (1.0 + std::exp(-5.6)));
        }
    }

    GIVEN("An integer with max domain") {
        auto i_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{});
        auto expit_ptr = graph.emplace_node<ExpitNode>(i_ptr);
        graph.emplace_node<ArrayValidationNode>(expit_ptr);

        THEN("The min/max are expected") {
            CHECK(expit_ptr->min() == 0.5);
            CHECK(expit_ptr->max() == 1);
        }
    }
}

TEST_CASE("UnaryOpNode - LogNode") {
    auto graph = Graph();
    GIVEN("A constant 1d array of doubles") {
        auto c_ptr = graph.emplace_node<ConstantNode>(std::vector{6.0, 0.3, 1.0, 1.2, 5.6});
        auto log_ptr = graph.emplace_node<LogNode>(c_ptr);

        THEN("The min/max are expected") {
            THEN("log(x) is not integral") { CHECK_FALSE(log_ptr->integral()); }
            CHECK(log_ptr->min() == std::log(0.3));
            CHECK(log_ptr->max() == std::log(6.0));
        }
    }
    GIVEN("An arbitrary number") {
        double c = 10.0;
        auto c_ptr = graph.emplace_node<ConstantNode>(c);
        auto log_ptr = graph.emplace_node<LogNode>(c_ptr);
        auto state = graph.initialize_state();
        CHECK(log_ptr->min() == std::log(c));
        CHECK(log_ptr->max() == std::log(c));
    }
    GIVEN("A negative number") {
        double c = -10.0;
        auto c_ptr = graph.emplace_node<ConstantNode>(c);
        CHECK_THROWS(graph.emplace_node<LogNode>(c_ptr));
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
            CHECK_THAT(logical_ptr->view(state),
                       RangeEquals({true, true, false, true, true, true, true}));
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
            CHECK_THAT(nc_ptr->view(state),
                       RangeEquals({false, false, true, false, false, false, false}));
        }
    }
}

TEST_CASE("UnaryOpNode - RintNode") {
    auto graph = Graph();
    GIVEN("An arbitrary number") {
        double c = 10.3;
        auto c_ptr = graph.emplace_node<ConstantNode>(c);
        auto rint_ptr = graph.emplace_node<RintNode>(c_ptr);
        auto state = graph.initialize_state();
        CHECK(rint_ptr->min() == std::rint(c));
        CHECK(rint_ptr->max() == std::rint(c));
    }

    GIVEN("A negative number") {
        double c = -10.5;
        auto c_ptr = graph.emplace_node<ConstantNode>(c);
        auto rint_ptr = graph.emplace_node<RintNode>(c_ptr);
        auto state = graph.initialize_state();
        CHECK(rint_ptr->min() == std::rint(c));
        CHECK(rint_ptr->max() == std::rint(c));
    }

    GIVEN("A constant 1d array of doubles") {
        auto c_ptr = graph.emplace_node<ConstantNode>(std::vector{-3.8, 0.3, 1.2, 5.6});
        auto rint_ptr = graph.emplace_node<RintNode>(c_ptr);

        THEN("The min/max are expected") {
            CHECK(rint_ptr->integral());
            CHECK(rint_ptr->min() == -4);
            CHECK(rint_ptr->max() == 6);
        }
    }

    WHEN("We access a constant 1d array using integers from a RintNode") {
        double c0 = 0.3;
        auto c0_ptr = graph.emplace_node<ConstantNode>(c0);
        auto rint0_ptr = graph.emplace_node<RintNode>(c0_ptr);

        double c3 = 2.8;
        auto c3_ptr = graph.emplace_node<ConstantNode>(c3);
        auto rint3_ptr = graph.emplace_node<RintNode>(c3_ptr);;


        auto arr_ptr = graph.emplace_node<ConstantNode>(std::vector{0, 10, 20, 30});

        auto a0_ptr = graph.emplace_node<AdvancedIndexingNode>(arr_ptr, rint0_ptr);
        auto a3_ptr = graph.emplace_node<AdvancedIndexingNode>(arr_ptr, rint3_ptr);

        auto state = graph.initialize_state();

        THEN("We get the value we expect") {
            CHECK_THAT(a0_ptr->view(state), RangeEquals({0}));
            CHECK_THAT(a3_ptr->view(state), RangeEquals({30}));
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

TEST_CASE("UnaryOpNode - SquareRootNode") {
    auto graph = Graph();
    GIVEN("An integer with max domain") {
        auto i_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{});
        auto square_root_ptr = graph.emplace_node<SquareRootNode>(i_ptr);
        graph.emplace_node<ArrayValidationNode>(square_root_ptr);

        THEN("The min/max are expected") {
            CHECK(square_root_ptr->min() == 0);
            // we might consider capping this differently for integer types in the future
            CHECK(square_root_ptr->max() == std::sqrt(static_cast<std::size_t>(2000000000)));
        }
        THEN("sqrt(i) is not integral") { CHECK_FALSE(square_root_ptr->integral()); }
    }
    GIVEN("An arbitrary number") {
        double c = 10.0;
        auto c_ptr = graph.emplace_node<ConstantNode>(c);
        auto square_root_ptr = graph.emplace_node<SquareRootNode>(c_ptr);
        auto state = graph.initialize_state();
        CHECK(square_root_ptr->min() == std::sqrt(c));
        CHECK(square_root_ptr->max() == std::sqrt(c));
    }
    GIVEN("A negative number") {
        double c = -10.0;
        auto c_ptr = graph.emplace_node<ConstantNode>(c);
        CHECK_THROWS(graph.emplace_node<SquareRootNode>(c_ptr));
    }
}

}  // namespace dwave::optimization

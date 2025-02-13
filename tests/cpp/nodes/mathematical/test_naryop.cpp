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
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/nodes/collections.hpp"
#include "dwave-optimization/nodes/constants.hpp"
#include "dwave-optimization/nodes/indexing.hpp"
#include "dwave-optimization/nodes/mathematical.hpp"
#include "dwave-optimization/nodes/numbers.hpp"
#include "dwave-optimization/nodes/testing.hpp"

namespace dwave::optimization {

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

        graph.emplace_node<ArrayValidationNode>(p_ptr);

        THEN("The shape is also a scalar") {
            CHECK(p_ptr->ndim() == 0);
            CHECK(p_ptr->size() == 1);
        }

        THEN("min/max/integral are defined") {
            // these are really just smoke tests
            CHECK(p_ptr->min() <= p_ptr->max());
            CHECK(p_ptr->integral() == p_ptr->integral());
        }

        THEN("The operands are all available via operands()") {
            REQUIRE(std::ranges::equal(p_ptr->operands(),
                                       std::vector<Array*>{a_ptr, b_ptr, c_ptr, d_ptr}));

            // we can cast to a non-const ptr if we're not const
            CHECK(static_cast<Array*>(p_ptr->operands()[0]) == static_cast<Array*>(a_ptr));
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

        graph.emplace_node<ArrayValidationNode>(p_ptr);

        THEN("The shape is also a 1D array") {
            CHECK(p_ptr->ndim() == 1);
            CHECK(p_ptr->size() == 4);
        }

        THEN("The operands are all available via operands()") {
            CHECK(std::ranges::equal(p_ptr->operands(), std::vector<Array*>{a_ptr, b_ptr}));
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

        graph.emplace_node<ArrayValidationNode>(p_ptr);

        THEN("The shape is also a 1D array") {
            CHECK(p_ptr->ndim() == 1);
            CHECK(p_ptr->size() == 4);
        }

        THEN("The operands are all available via operands()") {
            CHECK(std::ranges::equal(p_ptr->operands(), std::vector<Array*>{a_ptr, b_ptr, c_ptr}));
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

    GIVEN("A vector<Node*> of constants") {
        auto nodes = std::vector<ArrayNode*>{
                graph.emplace_node<ConstantNode>(5), graph.emplace_node<ConstantNode>(6),
                graph.emplace_node<ConstantNode>(7), graph.emplace_node<ConstantNode>(8)};

        WHEN("We construct a NaryOpNode from it") {
            auto p_ptr = graph.emplace_node<NaryOpNode<TestType>>(nodes);

            graph.emplace_node<ArrayValidationNode>(p_ptr);

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
    }
}

TEST_CASE("NaryOpNode - NaryMaximumNode") {
    auto graph = Graph();

    GIVEN("a = Integer(-5, 5), b = IntegerNode(2, 4), c = IntegerNode(-7, -3), x = a * b * c") {
        auto a_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -5, 5);
        auto b_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, 2, 4);
        auto c_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -7, -3);

        auto x_ptr = graph.emplace_node<NaryMaximumNode>(a_ptr, b_ptr, c_ptr);

        THEN("x's max/min/integral are as expected") {
            CHECK(x_ptr->max() == 5);
            CHECK(x_ptr->min() == 2);
            CHECK(x_ptr->integral());

            // check that the cache is populated with minmax
            Array::cache_type<std::pair<double, double>> cache;
            x_ptr->minmax(cache);
            // the output of a node depends on the inputs, so it shows
            // up in cache
            CHECK(cache.contains(x_ptr));
            // mutating the cache should also mutate the output
            cache[x_ptr].first = -1000;
            CHECK(x_ptr->minmax(cache).first == -1000);
            CHECK(x_ptr->minmax().first == 2);  // ignores the cache
        }
    }
}

TEST_CASE("NaryOpNode - NaryMinimumNode") {
    auto graph = Graph();

    GIVEN("a = Integer(-5, 5), b = IntegerNode(2, 4), c = IntegerNode(-7, -3), x = a * b * c") {
        auto a_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -5, 5);
        auto b_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, 2, 4);
        auto c_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -7, -3);

        auto x_ptr = graph.emplace_node<NaryMinimumNode>(a_ptr, b_ptr, c_ptr);

        THEN("x's max/min/integral are as expected") {
            CHECK(x_ptr->max() == -3);
            CHECK(x_ptr->min() == -7);
            CHECK(x_ptr->integral());
        }
    }
}

TEST_CASE("NaryOpNode - NaryMultiplyNode") {
    auto graph = Graph();

    GIVEN("a = Integer(-5, 5), b = IntegerNode(2, 4), c = IntegerNode(-7, -3), x = a * b * c") {
        auto a_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -5, 5);
        auto b_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, 2, 4);
        auto c_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -7, -3);

        auto x_ptr = graph.emplace_node<NaryMultiplyNode>(a_ptr, b_ptr, c_ptr);

        THEN("x's max/min/integral are as expected") {
            CHECK(x_ptr->max() == 140);
            CHECK(x_ptr->min() == -140);
            CHECK(x_ptr->integral());
        }
    }
}

TEST_CASE("NaryOpNode - NaryAddNode") {
    auto graph = Graph();

    GIVEN("a = Integer(-5, 5), b = IntegerNode(2, 4), c = IntegerNode(-7, -3), x = a + b + c") {
        auto a_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -5, 5);
        auto b_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, 2, 4);
        auto c_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -7, -3);

        auto x_ptr = graph.emplace_node<NaryAddNode>(a_ptr, b_ptr, c_ptr);

        THEN("x's max/min/integral are as expected") {
            CHECK(x_ptr->max() == 5 + 4 - 3);
            CHECK(x_ptr->min() == -5 + 2 + -7);
            CHECK(x_ptr->integral());
        }
    }

    GIVEN("a = Integer(-5, 5), b = IntegerNode(2, 4), c = .5, x = a + b + c") {
        auto a_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, -5, 5);
        auto b_ptr = graph.emplace_node<IntegerNode>(std::vector<ssize_t>{}, 2, 4);
        auto c_ptr = graph.emplace_node<ConstantNode>(.5);

        auto x_ptr = graph.emplace_node<NaryAddNode>(a_ptr, b_ptr, c_ptr);

        THEN("x's max/min/integral are as expected") {
            CHECK(x_ptr->max() == 5 + 4 + .5);
            CHECK(x_ptr->min() == -5 + 2 + .5);
            CHECK(!x_ptr->integral());
        }
    }
}

}  // namespace dwave::optimization

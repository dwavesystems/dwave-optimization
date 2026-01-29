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

#include <initializer_list>
#include <optional>

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers.hpp"
#include "catch2/matchers/catch_matchers_all.hpp"
#include "catch2/matchers/catch_matchers_range_equals.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/nodes/numbers.hpp"

using Catch::Matchers::RangeEquals;

namespace dwave::optimization {

TEST_CASE("BoundAxisInfo") {
    GIVEN("BoundAxisInfo(axis = 0, operators = {}, bounds = {1.0})") {
        REQUIRE_THROWS_WITH(
                NumberNode::BoundAxisInfo(0, std::vector<NumberNode::BoundAxisOperator>{},
                                          std::vector<double>{1.0}),
                "Bad axis-wise bounds for axis: 0, `operators` and `bounds` must each have "
                "non-zero size.");
    }

    GIVEN("BoundAxisInfo(axis = 0, operators = {<=}, bounds = {})") {
        REQUIRE_THROWS_WITH(
                NumberNode::BoundAxisInfo(0,
                                          std::vector<NumberNode::BoundAxisOperator>{
                                                  NumberNode::NumberNode::LessEqual},
                                          std::vector<double>{}),
                "Bad axis-wise bounds for axis: 0, `operators` and `bounds` must each have "
                "non-zero size.");
    }

    GIVEN("BoundAxisInfo(axis = 1, operators = {<=, ==, ==}, bounds = {2.0, 1.0})") {
        REQUIRE_THROWS_WITH(
                NumberNode::BoundAxisInfo(
                        1,
                        std::vector<NumberNode::BoundAxisOperator>{
                                NumberNode::LessEqual, NumberNode::Equal, NumberNode::Equal},
                        std::vector<double>{2.0, 1.0}),
                "Bad axis-wise bounds for axis: 1, `operators` and `bounds` should have same size "
                "if neither has size 1.");
    }

    GIVEN("BoundAxisInfo(axis = 2, operators = {==}, bounds = {1.0})") {
        NumberNode::BoundAxisInfo bound_axis(
                2, std::vector<NumberNode::BoundAxisOperator>{NumberNode::Equal},
                std::vector<double>{1.0});
        THEN("The bound axis info is correct") {
            CHECK(bound_axis.axis == 2);
            CHECK_THAT(bound_axis.operators, RangeEquals({NumberNode::Equal}));
            CHECK_THAT(bound_axis.bounds, RangeEquals({1.0}));
        }
    }

    GIVEN("BoundAxisInfo(axis = 2, operators = {==, <=, >=}, bounds = {1.0, 2.0, 3.0})") {
        NumberNode::BoundAxisInfo bound_axis(
                2,
                std::vector<NumberNode::BoundAxisOperator>{NumberNode::Equal, NumberNode::LessEqual,
                                                           NumberNode::GreaterEqual},
                std::vector<double>{1.0, 2.0, 3.0});
        THEN("The bound axis info is correct") {
            CHECK(bound_axis.axis == 2);
            CHECK_THAT(bound_axis.operators, RangeEquals({NumberNode::Equal, NumberNode::LessEqual,
                                                          NumberNode::GreaterEqual}));
            CHECK_THAT(bound_axis.bounds, RangeEquals({1.0, 2.0, 3.0}));
        }
    }
}

TEST_CASE("BinaryNode") {
    auto graph = Graph();

    GIVEN("Default BinaryNode") {
        auto bnode_ptr = graph.emplace_node<BinaryNode>();

        THEN("The function to check valid binary works") {
            CHECK(bnode_ptr->max() == 1.0);
            CHECK(bnode_ptr->min() == 0.0);
            CHECK(bnode_ptr->size() == 1);
            CHECK(bnode_ptr->ndim() == 0);
            CHECK(bnode_ptr->lower_bound() == 0.0);
            CHECK(bnode_ptr->upper_bound() == 1.0);
            CHECK(bnode_ptr->lower_bound(0) == 0.0);
            CHECK(bnode_ptr->upper_bound(1) == 1.0);
        }
    }

    GIVEN("A Binary Node representing an 1d array of 10 elements") {
        auto ptr = graph.emplace_node<BinaryNode>(std::initializer_list<ssize_t>{10});

        THEN("The state is not deterministic") { CHECK(!ptr->deterministic_state()); }

        THEN("The shape is fixed") {
            CHECK(ptr->ndim() == 1);
            CHECK(ptr->size() == 10);
            CHECK_THAT(ptr->shape(), RangeEquals({10}));
            CHECK_THAT(ptr->strides(), RangeEquals({sizeof(double)}));
        }

        THEN("The default bounds are set properly") {
            CHECK(ptr->lower_bound() == 0.0);
            CHECK(ptr->upper_bound() == 1.0);
            for (int i = 0; i < 10; ++i) {
                CHECK(ptr->lower_bound(i) == 0);
                CHECK(ptr->upper_bound(i) == 1);
            }
        }

        WHEN("We create a state using the default value") {
            auto state = graph.initialize_state();

            THEN("All elements are zero") {
                CHECK(ptr->size(state) == 10);
                CHECK_THAT(ptr->shape(state), RangeEquals({10}));
                CHECK_THAT(ptr->view(state), RangeEquals({0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
            }
        }

        WHEN("We create a state using a random number generator") {
            auto state = graph.empty_state();
            auto rng = std::default_random_engine(42);
            ptr->initialize_state(state, rng);
            graph.initialize_state(state);
            auto state_view = ptr->view(state);

            THEN("Then all elements are binary valued") {
                CHECK(std::find_if(state_view.begin(), state_view.end(),
                                   [](int i) { return (i != 0 && i != 1); }) == state_view.end());
            }
        }

        WHEN("We create a state using a specified values") {
            auto state = graph.empty_state();
            std::vector<double> vec_d{0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
            ptr->initialize_state(state, vec_d);
            graph.initialize_state(state);

            THEN("The passed in values are not modified") {
                CHECK_THAT(vec_d, RangeEquals({0, 1, 0, 1, 0, 1, 0, 1, 0, 1}));
            }

            THEN("We can read the state") { CHECK_THAT(ptr->view(state), RangeEquals(vec_d)); }

            WHEN("We flip the states") {
                for (int i = 0, stop = ptr->size(); i < stop; ++i) {
                    // Note, index-wise bounds are all [0,1]
                    ptr->flip(state, i);
                    vec_d[i] = !vec_d[i];
                }

                THEN("Elments are flipped properly") {
                    CHECK_THAT(ptr->view(state), RangeEquals(vec_d));
                    CHECK(static_cast<ssize_t>(ptr->diff(state).size()) == ptr->size());
                }
            }

            WHEN("We set all the elements") {
                int set_count_ground = 0;
                for (int i = 0, stop = ptr->size(); i < stop; ++i) {
                    // Note, index-wise bounds are all [0,1]
                    ptr->set(state, i);
                    set_count_ground += !vec_d[i];
                }

                THEN("Elments are set properly") {
                    CHECK(std::ranges::all_of(ptr->view(state), [](int i) { return i == 1; }));
                }

                THEN("The number of elements set equals the number of initially unset elements") {
                    CHECK(static_cast<int>(ptr->diff(state).size()) == set_count_ground);
                }
            }

            WHEN("We unset all the elements") {
                int unset_count_ground = 0;
                for (int i = 0, stop = ptr->size(); i < stop; ++i) {
                    // Note, index-wise bounds are all [0,1]
                    ptr->unset(state, i);
                    unset_count_ground += vec_d[i];
                }

                THEN("Elments are unset properly") {
                    CHECK(std::ranges::all_of(ptr->view(state), [](int i) { return i == 0; }));
                }

                THEN("The number of elements unset equals the number of initially set elements") {
                    CHECK(static_cast<int>(ptr->diff(state).size()) == unset_count_ground);
                }
            }

            WHEN("We exchange the elements") {
                int exchange_count_ground = 0;
                for (int i = 0, stop = ptr->size() - 1; i < stop; ++i) {
                    // Note, index-wise bounds are all [0,1]
                    ptr->exchange(state, i, i + 1);
                    std::swap(vec_d[i], vec_d[i + 1]);
                    exchange_count_ground += (vec_d[i] != vec_d[i + 1]);
                }

                THEN("Elments are exchanged properly") {
                    CHECK(std::ranges::equal(ptr->view(state), vec_d));
                }

                THEN("The number of effective exchanges are correct") {
                    // Two updates per exchange
                    CHECK(static_cast<int>(ptr->diff(state).size()) == 2 * exchange_count_ground);
                }
            }
        }
    }

    GIVEN("A Binary Node representing an 2d array of 5x5 elements") {
        auto ptr = graph.emplace_node<BinaryNode>(std::initializer_list<ssize_t>{5, 5});

        THEN("The shape is fixed") {
            CHECK(ptr->ndim() == 2);
            CHECK(ptr->size() == 25);
            CHECK_THAT(ptr->shape(), RangeEquals({5, 5}));
            CHECK_THAT(ptr->strides(), RangeEquals({5 * sizeof(double), sizeof(double)}));
        }

        THEN("The default bounds are set properly") {
            CHECK(ptr->lower_bound() == 0.0);
            CHECK(ptr->upper_bound() == 1.0);
            for (int i = 0; i < 25; ++i) {
                CHECK(ptr->lower_bound(i) == 0);
                CHECK(ptr->upper_bound(i) == 1);
            }
        }

        WHEN("We create a state using the default value") {
            auto state = graph.initialize_state();

            THEN("All elements are zero") {
                CHECK(ptr->size(state) == 25);
                CHECK_THAT(ptr->shape(state), RangeEquals({5, 5}));
                CHECK_THAT(ptr->view(state), RangeEquals({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
            }
        }

        WHEN("We create a state using a random number generator") {
            auto state = graph.empty_state();
            auto rng = std::default_random_engine(42);
            ptr->initialize_state(state, rng);
            graph.initialize_state(state);
            auto state_view = ptr->view(state);

            THEN("Then all elements are binary valued") {
                CHECK(std::find_if(state_view.begin(), state_view.end(),
                                   [](int i) { return (i != 0 && i != 1); }) == state_view.end());
            }
        }

        WHEN("We create a state using a specified values") {
            auto state = graph.empty_state();
            std::vector<double> vec_d{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                                      1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
            ptr->initialize_state(state, vec_d);
            graph.initialize_state(state);

            THEN("The passed in values are not modified") {
                CHECK_THAT(vec_d, RangeEquals({0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                                               1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0}));
            }

            THEN("We can read the state") { CHECK(std::ranges::equal(ptr->view(state), vec_d)); }

            WHEN("We flip the states") {
                for (int i = 0, stop = ptr->size(); i < stop; ++i) {
                    // Note, index-wise bounds are all [0,1]
                    ptr->flip(state, i);
                    vec_d[i] = !vec_d[i];
                }

                THEN("Elments are flipped properly") {
                    CHECK(std::ranges::equal(ptr->view(state), vec_d));
                }

                THEN("The number of elements set equals the number of initially unset elements") {
                    CHECK(static_cast<ssize_t>(ptr->diff(state).size()) == ptr->size());
                }
            }

            WHEN("We set all the elements") {
                int set_count_ground = 0;
                for (int i = 0, stop = ptr->size(); i < stop; ++i) {
                    // Note, index-wise bounds are all [0,1]
                    ptr->set(state, i);
                    set_count_ground += !vec_d[i];
                }

                THEN("Elments are set properly") {
                    CHECK(std::ranges::all_of(ptr->view(state), [](int i) { return i == 1; }));
                }

                THEN("The number of elements set equals the number of initially unset elements") {
                    CHECK(static_cast<int>(ptr->diff(state).size()) == set_count_ground);
                }
            }

            WHEN("We unset all the elements") {
                int unset_count_ground = 0;
                for (int i = 0, stop = ptr->size(); i < stop; ++i) {
                    // Note, index-wise bounds are all [0,1]
                    ptr->unset(state, i);
                    unset_count_ground += vec_d[i];
                }

                THEN("Elments are unset properly") {
                    CHECK(std::ranges::all_of(ptr->view(state), [](int i) { return i == 0; }));
                }

                THEN("The number of elements unset equals the number of initially set elements") {
                    CHECK(static_cast<int>(ptr->diff(state).size()) == unset_count_ground);
                }
            }

            WHEN("We exchange the elements") {
                int exchange_count_ground = 0;
                for (int i = 0, stop = ptr->size() - 1; i < stop; ++i) {
                    // Note, index-wise bounds are all [0,1]
                    ptr->exchange(state, i, i + 1);
                    std::swap(vec_d[i], vec_d[i + 1]);
                    exchange_count_ground += (vec_d[i] != vec_d[i + 1]);
                }

                THEN("Elments are exchanged properly") {
                    CHECK(std::ranges::equal(ptr->view(state), vec_d));
                }

                THEN("The number of effective exchanges are correct") {
                    // Two updates per exchange
                    CHECK(static_cast<int>(ptr->diff(state).size()) == 2 * exchange_count_ground);
                }
            }
        }
    }

    GIVEN("Binary node with index-wise bounds") {
        auto bnode_ptr = graph.emplace_node<BinaryNode>(3, std::vector<double>{-1, 0, 1},
                                                        std::vector<double>{2, 1, 1});

        THEN("The shape, max, min, and bounds are correct") {
            CHECK(bnode_ptr->size() == 3);
            CHECK(bnode_ptr->ndim() == 1);

            CHECK(bnode_ptr->max() == 1.0);
            CHECK(bnode_ptr->min() == 0.0);

            REQUIRE_THROWS(bnode_ptr->lower_bound());
            REQUIRE_THROWS(bnode_ptr->upper_bound());
            CHECK(bnode_ptr->lower_bound(0) == 0.0);
            CHECK(bnode_ptr->lower_bound(1) == 0.0);
            CHECK(bnode_ptr->lower_bound(2) == 1.0);
            CHECK(bnode_ptr->upper_bound(0) == 1.0);
            CHECK(bnode_ptr->upper_bound(1) == 1.0);
            CHECK(bnode_ptr->upper_bound(2) == 1.0);
        }

        AND_WHEN("We set the state at one of the indices") {
            auto state = graph.initialize_state();

            THEN("The initialized values are correct") {
                CHECK_THAT(bnode_ptr->view(state), RangeEquals({0, 0, 1}));
            }

            // Note, index-wise bounds are [[0,1], [0,1], [1,1]]
            bnode_ptr->set_value(state, 0, 1.0);

            THEN("The value is correct") {
                CHECK(bnode_ptr->diff(state).size() == 1);
                CHECK(bnode_ptr->get_value(state, 0) == 1.0);
            }
        }

        AND_WHEN("We set the state at the indices using set()") {
            auto state = graph.initialize_state();
            // Note, index-wise bounds are [[0,1], [0,1], [1,1]]
            bnode_ptr->set(state, 1);

            THEN("The values at index 0 and 1 are correct") {
                CHECK(bnode_ptr->get_value(state, 0) == 0.0);  // Default value
                CHECK(bnode_ptr->get_value(state, 1) == 1.0);  // Set value
            }

            AND_WHEN("We commit the state") {
                graph.commit(state);

                THEN("We can perform a flip()") {
                    // Note, index-wise bounds are [[0,1], [0,1], [1,1]]
                    CHECK(bnode_ptr->get_value(state, 1) == 1.0);
                    bnode_ptr->flip(state, 1);
                    CHECK(bnode_ptr->get_value(state, 1) == 0.0);
                }

                THEN("We can perform an unset()") {
                    // Note, index-wise bounds are [[0,1], [0,1], [1,1]]
                    CHECK(bnode_ptr->get_value(state, 1) == 1.0);
                    bnode_ptr->unset(state, 1);
                    CHECK(bnode_ptr->get_value(state, 1) == 0.0);
                }

                THEN("We can perform a set()") {
                    // Note, index-wise bounds are [[0,1], [0,1], [1,1]]
                    CHECK(bnode_ptr->get_value(state, 0) == 0.0);
                    bnode_ptr->set(state, 0);
                    CHECK(bnode_ptr->get_value(state, 0) == 1.0);
                }

                THEN("We can perform an exchange()") {
                    // Note, index-wise bounds are [[0,1], [0,1], [1,1]]
                    CHECK(bnode_ptr->get_value(state, 0) == 0.0);
                    CHECK(bnode_ptr->get_value(state, 1) == 1.0);
                    bnode_ptr->exchange(state, 0, 1);
                    CHECK(bnode_ptr->get_value(state, 0) == 1.0);
                    CHECK(bnode_ptr->get_value(state, 1) == 0.0);
                }
            }
        }

        AND_WHEN("We set the state at indices using clip()") {
            auto state = graph.initialize_state();

            bnode_ptr->clip_and_set_value(state, 0, -2);
            bnode_ptr->clip_and_set_value(state, 1, 0);
            bnode_ptr->clip_and_set_value(state, 2, 2);

            THEN("Clip set the values correctly") {
                // Note, index-wise bounds are [[0,1], [0,1], [1,1]]
                CHECK(bnode_ptr->get_value(state, 0) == 0);
                CHECK(bnode_ptr->get_value(state, 1) == 0);
                CHECK(bnode_ptr->get_value(state, 2) == 1);
            }
        }
    }

    GIVEN("Binary node with index-wise upper bound and general lower bound") {
        auto bnode_ptr = graph.emplace_node<BinaryNode>(2, -2.0, std::vector<double>{0.0, 1.1});

        THEN("The max, min, and bounds are correct") {
            CHECK(bnode_ptr->max() == 1.0);
            CHECK(bnode_ptr->min() == 0.0);

            CHECK(bnode_ptr->lower_bound() == 0.0);
            REQUIRE_THROWS(bnode_ptr->upper_bound());
            CHECK(bnode_ptr->lower_bound(0) == 0.0);
            CHECK(bnode_ptr->lower_bound(1) == 0.0);
            CHECK(bnode_ptr->upper_bound(0) == 0.0);
            CHECK(bnode_ptr->upper_bound(1) == 1.0);
        }
    }

    GIVEN("Binary node with index-wise lower bound and general upper bound") {
        auto bnode_ptr = graph.emplace_node<BinaryNode>(2, std::vector<double>{-1.0, 1.0}, 100.0);

        THEN("The max, min, and bounds are correct") {
            CHECK(bnode_ptr->max() == 1.0);
            CHECK(bnode_ptr->min() == 0.0);

            REQUIRE_THROWS(bnode_ptr->lower_bound());
            CHECK(bnode_ptr->upper_bound() == 1.0);
            CHECK(bnode_ptr->lower_bound(0) == 0.0);
            CHECK(bnode_ptr->lower_bound(1) == 1.0);
            CHECK(bnode_ptr->upper_bound(0) == 1.0);
            CHECK(bnode_ptr->upper_bound(1) == 1.0);
        }
    }

    GIVEN("Binary node with invalid index-wise lower bounds at index 0") {
        REQUIRE_THROWS(graph.emplace_node<BinaryNode>(2, std::vector<double>{2, 0},
                                                      std::vector<double>{1, 1}));
    }

    GIVEN("Binary node with invalid index-wise upper bounds at index 1") {
        REQUIRE_THROWS(graph.emplace_node<BinaryNode>(2, std::vector<double>{0, 0},
                                                      std::vector<double>{1, -1}));
    }

    GIVEN("Invalid dynamically sized BinaryNode") {
        REQUIRE_THROWS_WITH(graph.emplace_node<BinaryNode>(std::initializer_list<ssize_t>{-1, 2}),
                            "Number array cannot have dynamic size.");
    }

    GIVEN("(2x3)-BinaryNode with axis-wise bounds on the invalid axis -1") {
        NumberNode::BoundAxisInfo bound_axis{
                -1, std::vector<NumberNode::BoundAxisOperator>{NumberNode::Equal},
                std::vector<double>{1.0}};
        REQUIRE_THROWS_WITH(
                graph.emplace_node<dwave::optimization::BinaryNode>(
                        std::initializer_list<ssize_t>{2, 3}, std::nullopt, std::nullopt,
                        std::vector<NumberNode::BoundAxisInfo>{bound_axis}),
                "Invalid bound axis: -1. Note, negative indexing is not supported for "
                "axis-wise bounds.");
    }

    GIVEN("(2x3)-BinaryNode with axis-wise bounds on the invalid axis 2") {
        NumberNode::BoundAxisInfo bound_axis{
                2, std::vector<NumberNode::BoundAxisOperator>{NumberNode::Equal},
                std::vector<double>{1.0}};
        REQUIRE_THROWS_WITH(
                graph.emplace_node<dwave::optimization::BinaryNode>(
                        std::initializer_list<ssize_t>{2, 3}, std::nullopt, std::nullopt,
                        std::vector<NumberNode::BoundAxisInfo>{bound_axis}),
                "Invalid bound axis: 2. Note, negative indexing is not supported for "
                "axis-wise bounds.");
    }

    GIVEN("(2x3)-BinaryNode with axis-wise bounds on axis: 1 with too many operators.") {
        NumberNode::BoundAxisInfo bound_axis{
                1,
                std::vector<NumberNode::BoundAxisOperator>{NumberNode::LessEqual, NumberNode::Equal,
                                                           NumberNode::Equal, NumberNode::Equal},
                std::vector<double>{1.0}};
        REQUIRE_THROWS_WITH(
                graph.emplace_node<dwave::optimization::BinaryNode>(
                        std::initializer_list<ssize_t>{2, 3}, std::nullopt, std::nullopt,
                        std::vector<NumberNode::BoundAxisInfo>{bound_axis}),
                "Invalid number of axis-wise operators along axis: 1 given axis size: 3");
    }

    GIVEN("(2x3)-BinaryNode with axis-wise bounds on axis: 1 with too few operators.") {
        NumberNode::BoundAxisInfo bound_axis{1,
                                             std::vector<NumberNode::BoundAxisOperator>{
                                                     NumberNode::LessEqual, NumberNode::Equal},
                                             std::vector<double>{1.0}};
        REQUIRE_THROWS_WITH(
                graph.emplace_node<dwave::optimization::BinaryNode>(
                        std::initializer_list<ssize_t>{2, 3}, std::nullopt, std::nullopt,
                        std::vector<NumberNode::BoundAxisInfo>{bound_axis}),
                "Invalid number of axis-wise operators along axis: 1 given axis size: 3");
    }

    GIVEN("(2x3)-BinaryNode with axis-wise bounds on axis: 1 with too many bounds.") {
        NumberNode::BoundAxisInfo bound_axis{
                1, std::vector<NumberNode::BoundAxisOperator>{NumberNode::LessEqual},
                std::vector<double>{1.0, 2.0, 3.0, 4.0}};
        REQUIRE_THROWS_WITH(
                graph.emplace_node<dwave::optimization::BinaryNode>(
                        std::initializer_list<ssize_t>{2, 3}, std::nullopt, std::nullopt,
                        std::vector<NumberNode::BoundAxisInfo>{bound_axis}),
                "Invalid number of axis-wise bounds along axis: 1 given axis size: 3");
    }

    GIVEN("(2x3)-BinaryNode with axis-wise bounds on axis: 1 with too few bounds.") {
        NumberNode::BoundAxisInfo bound_axis{
                1, std::vector<NumberNode::BoundAxisOperator>{NumberNode::LessEqual},
                std::vector<double>{1.0, 2.0}};
        REQUIRE_THROWS_WITH(
                graph.emplace_node<dwave::optimization::BinaryNode>(
                        std::initializer_list<ssize_t>{2, 3}, std::nullopt, std::nullopt,
                        std::vector<NumberNode::BoundAxisInfo>{bound_axis}),
                "Invalid number of axis-wise bounds along axis: 1 given axis size: 3");
    }

    GIVEN("(2x3)-BinaryNode with duplicate axis-wise bounds on axis: 1") {
        NumberNode::BoundAxisInfo bound_axis{
                1, std::vector<NumberNode::BoundAxisOperator>{NumberNode::Equal},
                std::vector<double>{1.0}};
        REQUIRE_THROWS_WITH(
                graph.emplace_node<dwave::optimization::BinaryNode>(
                        std::initializer_list<ssize_t>{2, 3}, std::nullopt, std::nullopt,
                        std::vector<NumberNode::BoundAxisInfo>{bound_axis, bound_axis}),
                "Cannot define multiple axis-wise bounds for a single axis.");
    }

    GIVEN("(2x3)-BinaryNode with axis-wise bounds on axes: 0 and 1") {
        NumberNode::BoundAxisInfo bound_axis_0{
                0, std::vector<NumberNode::BoundAxisOperator>{NumberNode::LessEqual},
                std::vector<double>{1.0}};
        NumberNode::BoundAxisInfo bound_axis_1{
                1, std::vector<NumberNode::BoundAxisOperator>{NumberNode::LessEqual},
                std::vector<double>{1.0}};
        REQUIRE_THROWS_WITH(
                graph.emplace_node<dwave::optimization::BinaryNode>(
                        std::initializer_list<ssize_t>{2, 3}, std::nullopt, std::nullopt,
                        std::vector<NumberNode::BoundAxisInfo>{bound_axis_0, bound_axis_1}),
                "Axis-wise bounds are supported for at most one axis.");
    }

    GIVEN("(2x3x4)-IntegerNode with non-integral axis-wise bounds") {
        NumberNode::BoundAxisInfo bound_axis{
                1, std::vector<NumberNode::BoundAxisOperator>{NumberNode::Equal},
                std::vector<double>{0.1}};
        REQUIRE_THROWS_WITH(
                graph.emplace_node<dwave::optimization::BinaryNode>(
                        std::initializer_list<ssize_t>{2, 3}, std::nullopt, std::nullopt,
                        std::vector<NumberNode::BoundAxisInfo>{bound_axis}),
                "Axis wise bounds for integral number arrays must be intregral.");
    }

    GIVEN("(3x2x2)-BinaryNode with infeasible axis-wise bound on axis: 0") {
        auto graph = Graph();

        NumberNode::BoundAxisInfo bound_axis{
                0,
                std::vector<NumberNode::BoundAxisOperator>{NumberNode::Equal, NumberNode::LessEqual,
                                                           NumberNode::GreaterEqual},
                std::vector<double>{5.0, 2.0, 3.0}};

        // Each hyperslice along axis 0 has size 4. There is no feasible
        // assignment to the values in slice 0 (along axis 0) that results in a
        // sum equal to 5.
        graph.emplace_node<dwave::optimization::BinaryNode>(
                std::initializer_list<ssize_t>{3, 2, 2}, std::nullopt, std::nullopt,
                std::vector<NumberNode::BoundAxisInfo>{bound_axis});

        WHEN("We create a state by initialize_state()") {
            REQUIRE_THROWS_WITH(graph.initialize_state(), "Infeasible axis-wise bounds.");
        }
    }

    GIVEN("(3x2x2)-BinaryNode with infeasible axis-wise bound on axis: 1") {
        auto graph = Graph();

        NumberNode::BoundAxisInfo bound_axis{1,
                                             std::vector<NumberNode::BoundAxisOperator>{
                                                     NumberNode::Equal, NumberNode::GreaterEqual},
                                             std::vector<double>{5.0, 7.0}};

        graph.emplace_node<dwave::optimization::BinaryNode>(
                std::initializer_list<ssize_t>{3, 2, 2}, std::nullopt, std::nullopt,
                std::vector<NumberNode::BoundAxisInfo>{bound_axis});

        WHEN("We create a state by initialize_state()") {
            // Each hyperslice along axis 1 has size 6. There is no feasible
            // assignment to the values in slice 1 (along axis 1) that results in a
            // sum greater than or equal to 7.
            REQUIRE_THROWS_WITH(graph.initialize_state(), "Infeasible axis-wise bounds.");
        }
    }

    GIVEN("(3x2x2)-BinaryNode with infeasible axis-wise bound on axis: 2") {
        auto graph = Graph();

        NumberNode::BoundAxisInfo bound_axis{2,
                                             std::vector<NumberNode::BoundAxisOperator>{
                                                     NumberNode::Equal, NumberNode::LessEqual},
                                             std::vector<double>{5.0, -1.0}};

        graph.emplace_node<dwave::optimization::BinaryNode>(
                std::initializer_list<ssize_t>{3, 2, 2}, std::nullopt, std::nullopt,
                std::vector<NumberNode::BoundAxisInfo>{bound_axis});

        WHEN("We create a state by initialize_state()") {
            // Each hyperslice along axis 2 has size 6. There is no feasible
            // assignment to the values in slice 1 (along axis 2) that results in a
            // sum less than or equal to -1.
            REQUIRE_THROWS_WITH(graph.initialize_state(), "Infeasible axis-wise bounds.");
        }
    }

    GIVEN("(3x2x2)-BinaryNode with feasible axis-wise bound on axis: 0") {
        auto graph = Graph();

        NumberNode::BoundAxisInfo bound_axis{
                0,
                std::vector<NumberNode::BoundAxisOperator>{NumberNode::Equal, NumberNode::LessEqual,
                                                           NumberNode::GreaterEqual},
                std::vector<double>{1.0, 2.0, 3.0}};

        auto bnode_ptr = graph.emplace_node<dwave::optimization::BinaryNode>(
                std::initializer_list<ssize_t>{3, 2, 2}, std::nullopt, std::nullopt,
                std::vector<NumberNode::BoundAxisInfo>{bound_axis});

        THEN("Axis wise bound is correct") {
            CHECK(bnode_ptr->axis_wise_bounds().size() == 1);
            NumberNode::BoundAxisInfo bnode_bound_axis = bnode_ptr->axis_wise_bounds()[0];
            CHECK(bound_axis.axis == bnode_bound_axis.axis);
            CHECK_THAT(bound_axis.operators, RangeEquals(bnode_bound_axis.operators));
            CHECK_THAT(bound_axis.bounds, RangeEquals(bnode_bound_axis.bounds));
        }

        WHEN("We create a state by initialize_state()") {
            auto state = graph.initialize_state();
            graph.initialize_state(state);
            // import numpy as np
            // a = np.asarray([i for i in range(3*2*2)]).reshape(3, 2, 2)
            // print(a[0, :, :].flatten())
            // ... [0 1 2 3]
            // print(a[1, :, :].flatten())
            // ... [4 5 6 7]
            // print(a[2, :, :].flatten())
            // ... [ 8  9 10 11]
            std::vector<double> expected_init{1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0};
            // Cannonically least state that satisfies bounds
            // slice 0  slice 1 slice 2
            //  1, 0     0, 0    1, 1
            //  0, 0     0, 0    1, 0

            auto bound_axis_sums = bnode_ptr->bound_axis_sums(state);

            THEN("The bound axis sums and state are correct") {
                CHECK(bnode_ptr->bound_axis_sums(state).size() == 1);
                CHECK(bnode_ptr->bound_axis_sums(state).data()[0].size() == 3);
                CHECK_THAT(bnode_ptr->bound_axis_sums(state)[0], RangeEquals({1, 0, 3}));
                CHECK_THAT(bnode_ptr->view(state), RangeEquals(expected_init));
            }
        }
    }

    GIVEN("(3x2x2)-BinaryNode with feasible axis-wise bound on axis: 1") {
        auto graph = Graph();

        NumberNode::BoundAxisInfo bound_axis{
                1,
                std::vector<NumberNode::BoundAxisOperator>{NumberNode::LessEqual,
                                                           NumberNode::GreaterEqual},
                std::vector<double>{1.0, 5.0}};

        auto bnode_ptr = graph.emplace_node<dwave::optimization::BinaryNode>(
                std::initializer_list<ssize_t>{3, 2, 2}, std::nullopt, std::nullopt,
                std::vector<NumberNode::BoundAxisInfo>{bound_axis});

        THEN("Axis wise bound is correct") {
            CHECK(bnode_ptr->axis_wise_bounds().size() == 1);
            NumberNode::BoundAxisInfo bnode_bound_axis = bnode_ptr->axis_wise_bounds()[0];
            CHECK(bound_axis.axis == bnode_bound_axis.axis);
            CHECK_THAT(bound_axis.operators, RangeEquals(bnode_bound_axis.operators));
            CHECK_THAT(bound_axis.bounds, RangeEquals(bnode_bound_axis.bounds));
        }

        WHEN("We create a state by initialize_state()") {
            auto state = graph.initialize_state();
            graph.initialize_state(state);
            // import numpy as np
            // a = np.asarray([i for i in range(3*2*2)]).reshape(3, 2, 2)
            // print(a[:, 0, :].flatten())
            // ... [0 1 4 5 8 9]
            // print(a[:, 1, :].flatten())
            // ... [ 2  3  6  7 10 11]
            std::vector<double> expected_init{0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0};
            // Cannonically least state that satisfies bounds
            // slice 0  slice 1
            //  0, 0     1, 1
            //  0, 0     1, 1
            //  0, 0     1, 0

            auto bound_axis_sums = bnode_ptr->bound_axis_sums(state);

            THEN("The bound axis sums and state are correct") {
                CHECK(bnode_ptr->bound_axis_sums(state).size() == 1);
                CHECK(bnode_ptr->bound_axis_sums(state).data()[0].size() == 2);
                CHECK_THAT(bnode_ptr->bound_axis_sums(state)[0], RangeEquals({0, 5}));
                CHECK_THAT(bnode_ptr->view(state), RangeEquals(expected_init));
            }
        }
    }

    GIVEN("(3x2x2)-BinaryNode with feasible axis-wise bound on axis: 2") {
        auto graph = Graph();

        NumberNode::BoundAxisInfo bound_axis{2,
                                             std::vector<NumberNode::BoundAxisOperator>{
                                                     NumberNode::Equal, NumberNode::GreaterEqual},
                                             std::vector<double>{3.0, 6.0}};

        auto bnode_ptr = graph.emplace_node<dwave::optimization::BinaryNode>(
                std::initializer_list<ssize_t>{3, 2, 2}, std::nullopt, std::nullopt,
                std::vector<NumberNode::BoundAxisInfo>{bound_axis});

        THEN("Axis wise bound is correct") {
            CHECK(bnode_ptr->axis_wise_bounds().size() == 1);
            NumberNode::BoundAxisInfo bnode_bound_axis = bnode_ptr->axis_wise_bounds()[0];
            CHECK(bound_axis.axis == bnode_bound_axis.axis);
            CHECK_THAT(bound_axis.operators, RangeEquals(bnode_bound_axis.operators));
            CHECK_THAT(bound_axis.bounds, RangeEquals(bnode_bound_axis.bounds));
        }

        WHEN("We create a state by initialize_state()") {
            auto state = graph.initialize_state();
            graph.initialize_state(state);
            // import numpy as np
            // a = np.asarray([i for i in range(3*2*2)]).reshape(3, 2, 2)
            // print(a[:, :, 0].flatten())
            // ... [ 0  2  4  6  8 10]
            // print(a[:, :, 1].flatten())
            // ... [ 1  3  5  7  9 11]
            std::vector<double> expected_init{1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1};
            // Cannonically least state that satisfies bounds
            // slice 0  slice 1
            //  1, 1     1, 1
            //  1, 0     1, 1
            //  0, 0     1, 1

            auto bound_axis_sums = bnode_ptr->bound_axis_sums(state);

            THEN("The bound axis sums and state are correct") {
                CHECK(bnode_ptr->bound_axis_sums(state).size() == 1);
                CHECK(bnode_ptr->bound_axis_sums(state).data()[0].size() == 2);
                CHECK_THAT(bnode_ptr->bound_axis_sums(state)[0], RangeEquals({3, 6}));
                CHECK_THAT(bnode_ptr->view(state), RangeEquals(expected_init));
            }
        }
    }

    GIVEN("(3x2x2)-BinaryNode with an axis-wise bound on axis: 0") {
        auto graph = Graph();

        NumberNode::BoundAxisInfo bound_axis{
                0,
                std::vector<NumberNode::BoundAxisOperator>{NumberNode::Equal, NumberNode::LessEqual,
                                                           NumberNode::GreaterEqual},
                std::vector<double>{1.0, 2.0, 3.0}};

        auto bnode_ptr = graph.emplace_node<dwave::optimization::BinaryNode>(
                std::initializer_list<ssize_t>{3, 2, 2}, std::nullopt, std::nullopt,
                std::vector<NumberNode::BoundAxisInfo>{bound_axis});

        THEN("Axis wise bound is correct") {
            CHECK(bnode_ptr->axis_wise_bounds().size() == 1);
            NumberNode::BoundAxisInfo bnode_bound_axis = bnode_ptr->axis_wise_bounds()[0];
            CHECK(bound_axis.axis == bnode_bound_axis.axis);
            CHECK_THAT(bound_axis.operators, RangeEquals(bnode_bound_axis.operators));
            CHECK_THAT(bound_axis.bounds, RangeEquals(bnode_bound_axis.bounds));
        }

        WHEN("We initialize three invalid states") {
            auto state = graph.empty_state();
            // This state violates the 0th hyperslice along axis 0
            std::vector<double> init_values{1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
            // import numpy as np
            // a = np.asarray([1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
            // a = a.reshape(3, 2, 2)
            // a.sum(axis=(1, 2))
            // >>> array([2, 2, 4])
            CHECK_THROWS_WITH(bnode_ptr->initialize_state(state, init_values),
                              "Initialized values do not satisfy axis-wise bounds.");

            state = graph.empty_state();
            // This state violates the 1st hyperslice along axis 0
            init_values = {0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1};
            // import numpy as np
            // a = np.asarray([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
            // a = a.reshape(3, 2, 2)
            // a.sum(axis=(1, 2))
            // >>> array([1, 3, 4])
            CHECK_THROWS_WITH(bnode_ptr->initialize_state(state, init_values),
                              "Initialized values do not satisfy axis-wise bounds.");

            state = graph.empty_state();
            // This state violates the 2nd hyperslice along axis 0
            init_values = {0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0};
            // import numpy as np
            // a = np.asarray([0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0])
            // a = a.reshape(3, 2, 2)
            // a.sum(axis=(1, 2))
            // >>> array([1, 2, 2])
            CHECK_THROWS_WITH(bnode_ptr->initialize_state(state, init_values),
                              "Initialized values do not satisfy axis-wise bounds.");
        }

        WHEN("We initialize a valid state") {
            auto state = graph.empty_state();
            std::vector<double> init_values{0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
            bnode_ptr->initialize_state(state, init_values);
            graph.initialize_state(state);

            auto bound_axis_sums = bnode_ptr->bound_axis_sums(state);

            THEN("The bound axis sums and state are correct") {
                // **Python Code 1**
                // import numpy as np
                // a = np.asarray([0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
                // a = a.reshape(3, 2, 2)
                // a.sum(axis=(1, 2))
                CHECK(bnode_ptr->bound_axis_sums(state).size() == 1);
                CHECK(bnode_ptr->bound_axis_sums(state).data()[0].size() == 3);
                CHECK_THAT(bnode_ptr->bound_axis_sums(state)[0], RangeEquals({1, 2, 4}));
                CHECK_THAT(bnode_ptr->view(state), RangeEquals(init_values));
            }

            THEN("We exchange() some values") {
                bnode_ptr->exchange(state, 0, 3);  // Does nothing.
                bnode_ptr->exchange(state, 1, 6);  // Does nothing.
                bnode_ptr->exchange(state, 1, 3);
                std::swap(init_values[0], init_values[3]);
                std::swap(init_values[1], init_values[6]);
                std::swap(init_values[1], init_values[3]);
                // state is now: [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1]

                THEN("The bound axis sums and state updated correctly") {
                    // Cont. w/ Python code at **Python Code 1**
                    // a[np.unravel_index(1, a.shape)] = 0
                    // a[np.unravel_index(3, a.shape)] = 1
                    // a.sum(axis=(1, 2))
                    CHECK_THAT(bnode_ptr->bound_axis_sums(state)[0], RangeEquals({1, 2, 4}));
                    CHECK(bnode_ptr->diff(state).size() == 2);  // 2 updates per exchange
                    CHECK_THAT(bnode_ptr->view(state), RangeEquals(init_values));
                }

                AND_WHEN("We revert") {
                    graph.revert(state);

                    THEN("The bound axis sums reverted correctly") {
                        CHECK_THAT(bnode_ptr->bound_axis_sums(state)[0], RangeEquals({1, 2, 4}));
                        CHECK(bnode_ptr->diff(state).size() == 0);
                    }
                }
            }

            THEN("We clip_and_set_value() some values") {
                bnode_ptr->clip_and_set_value(state, 5, -1);  // Does nothing.
                bnode_ptr->clip_and_set_value(state, 7, -1);
                bnode_ptr->clip_and_set_value(state, 9, 1);  // Does nothing.
                bnode_ptr->clip_and_set_value(state, 11, 0);
                bnode_ptr->clip_and_set_value(state, 11, 1);
                bnode_ptr->clip_and_set_value(state, 10, 0);
                init_values[5] = 0;
                init_values[7] = 0;
                init_values[9] = 1;
                init_values[11] = 1;
                init_values[10] = 0;
                // state is now: [0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1]

                THEN("The bound axis sums and state updated correctly") {
                    // Cont. w/ Python code at **Python Code 1**
                    // a[np.unravel_index(5, a.shape)] = 0
                    // a[np.unravel_index(7, a.shape)] = 0
                    // a[np.unravel_index(9, a.shape)] = 1
                    // a[np.unravel_index(11, a.shape)] = 1
                    // a[np.unravel_index(10, a.shape)] = 0
                    // a.sum(axis=(1, 2))
                    CHECK_THAT(bnode_ptr->bound_axis_sums(state)[0], RangeEquals({1, 1, 3}));
                    CHECK(bnode_ptr->diff(state).size() == 4);
                    CHECK_THAT(bnode_ptr->view(state), RangeEquals(init_values));
                }

                AND_WHEN("We revert") {
                    graph.revert(state);

                    THEN("The bound axis sums reverted correctly") {
                        CHECK_THAT(bnode_ptr->bound_axis_sums(state)[0], RangeEquals({1, 2, 4}));
                        CHECK(bnode_ptr->diff(state).size() == 0);
                    }
                }
            }

            THEN("We set_value() some values") {
                bnode_ptr->set_value(state, 0, 0);  // Does nothing.
                bnode_ptr->set_value(state, 6, 0);
                bnode_ptr->set_value(state, 7, 0);
                bnode_ptr->set_value(state, 4, 1);
                bnode_ptr->set_value(state, 10, 1);  // Does nothing.
                bnode_ptr->set_value(state, 11, 0);
                init_values[0] = 0;
                init_values[6] = 0;
                init_values[7] = 0;
                init_values[4] = 1;
                init_values[10] = 1;
                init_values[11] = 0;
                // state is now: [0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0]

                THEN("The bound axis sums and state updated correctly") {
                    // Cont. w/ Python code at **Python Code 1**
                    // a[np.unravel_index(0, a.shape)] = 0
                    // a[np.unravel_index(6, a.shape)] = 0
                    // a[np.unravel_index(7, a.shape)] = 0
                    // a[np.unravel_index(4, a.shape)] = 1
                    // a[np.unravel_index(10, a.shape)] = 1
                    // a[np.unravel_index(11, a.shape)] = 0
                    // a.sum(axis=(1, 2))
                    CHECK_THAT(bnode_ptr->bound_axis_sums(state)[0], RangeEquals({1, 1, 3}));
                    CHECK(bnode_ptr->diff(state).size() == 4);
                    CHECK_THAT(bnode_ptr->view(state), RangeEquals(init_values));
                }

                AND_WHEN("We revert") {
                    graph.revert(state);

                    THEN("The bound axis sums reverted correctly") {
                        CHECK_THAT(bnode_ptr->bound_axis_sums(state)[0], RangeEquals({1, 2, 4}));
                        CHECK(bnode_ptr->diff(state).size() == 0);
                    }
                }
            }

            THEN("We flip() some values") {
                bnode_ptr->flip(state, 6);   // 1 -> 0
                bnode_ptr->flip(state, 4);   // 0 -> 1
                bnode_ptr->flip(state, 11);  // 1 -> 0
                init_values[6] = !init_values[6];
                init_values[4] = !init_values[4];
                init_values[11] = !init_values[11];
                // state is now: [0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0]

                THEN("The bound axis sums and state updated correctly") {
                    // Cont. w/ Python code at **Python Code 1**
                    // a[np.unravel_index(6, a.shape)] = 0
                    // a[np.unravel_index(4, a.shape)] = 1
                    // a[np.unravel_index(11, a.shape)] = 0
                    // a.sum(axis=(1, 2))
                    CHECK_THAT(bnode_ptr->bound_axis_sums(state)[0], RangeEquals({1, 2, 3}));
                    CHECK(bnode_ptr->diff(state).size() == 3);
                    CHECK_THAT(bnode_ptr->view(state), RangeEquals(init_values));
                }

                AND_WHEN("We revert") {
                    graph.revert(state);

                    THEN("The bound axis sums reverted correctly") {
                        CHECK_THAT(bnode_ptr->bound_axis_sums(state)[0], RangeEquals({1, 2, 4}));
                        CHECK(bnode_ptr->diff(state).size() == 0);
                    }
                }
            }

            THEN("We unset() some values") {
                bnode_ptr->unset(state, 0);  // Does nothing.
                bnode_ptr->unset(state, 6);
                bnode_ptr->unset(state, 11);
                init_values[0] = 0;
                init_values[6] = 0;
                init_values[11] = 0;
                // state is now: [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]

                THEN("The bound axis sums and state updated correctly") {
                    // Cont. w/ Python code at **Python Code 1**
                    // a[np.unravel_index(0, a.shape)] = 0
                    // a[np.unravel_index(6, a.shape)] = 0
                    // a[np.unravel_index(11, a.shape)] = 0
                    // a.sum(axis=(1, 2))
                    CHECK_THAT(bnode_ptr->bound_axis_sums(state)[0], RangeEquals({1, 1, 3}));
                    CHECK(bnode_ptr->diff(state).size() == 2);
                    CHECK_THAT(bnode_ptr->view(state), RangeEquals(init_values));
                }

                AND_WHEN("We commit and set() some values") {
                    graph.commit(state);

                    bnode_ptr->set(state, 10);  // Does nothing.
                    bnode_ptr->set(state, 11);
                    init_values[10] = 1;
                    init_values[11] = 1;
                    // state is now: [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

                    THEN("The bound axis sums updated correctly") {
                        CHECK_THAT(bnode_ptr->bound_axis_sums(state)[0], RangeEquals({1, 1, 4}));
                        CHECK(bnode_ptr->diff(state).size() == 1);
                        CHECK_THAT(bnode_ptr->view(state), RangeEquals(init_values));
                    }

                    AND_WHEN("We revert") {
                        graph.revert(state);

                        THEN("The bound axis sums reverted correctly") {
                            CHECK_THAT(bnode_ptr->bound_axis_sums(state)[0],
                                       RangeEquals({1, 1, 3}));
                            CHECK(bnode_ptr->diff(state).size() == 0);
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("IntegerNode") {
    auto graph = Graph();

    GIVEN("Default IntegerNode") {
        auto inode_ptr = graph.emplace_node<IntegerNode>();

        THEN("The function to check valid integers works") {
            CHECK(inode_ptr->max() == IntegerNode::default_upper_bound);
            CHECK(inode_ptr->min() == IntegerNode::default_lower_bound);
            CHECK(inode_ptr->size() == 1);
            CHECK(inode_ptr->ndim() == 0);
            CHECK(inode_ptr->lower_bound(0) == IntegerNode::default_lower_bound);
            CHECK(inode_ptr->upper_bound(0) == IntegerNode::default_upper_bound);
            CHECK(inode_ptr->lower_bound() == IntegerNode::default_lower_bound);
            CHECK(inode_ptr->upper_bound() == IntegerNode::default_upper_bound);
        }
    }

    GIVEN("Double precision numbers, which may fall outside integer range or are not integral") {
        IntegerNode inode({1});

        THEN("The state is not deterministic") { CHECK(!inode.deterministic_state()); }

        THEN("The function to check valid integers works") {
            CHECK(inode.max() == 2000000000);
            CHECK(inode.min() == 0);
            CHECK(inode.is_valid(0, inode.min() - 1) == false);
            CHECK(inode.is_valid(0, inode.max() + 1) == false);
            CHECK(inode.is_valid(0, 10.5) == false);
            CHECK(inode.is_valid(0, inode.min()) == true);
            CHECK(inode.is_valid(0, inode.max()) == true);
            CHECK(inode.lower_bound(0) == IntegerNode::default_lower_bound);
            CHECK(inode.upper_bound(0) == IntegerNode::default_upper_bound);
            CHECK(inode.lower_bound() == IntegerNode::default_lower_bound);
            CHECK(inode.upper_bound() == IntegerNode::default_upper_bound);
        }
    }

    GIVEN("Integer node with custom range, the range works as expected") {
        IntegerNode inode({1}, -5, 10);

        THEN("The function to check valid integers works") {
            CHECK(inode.max() == 10);
            CHECK(inode.min() == -5);
            CHECK(inode.is_valid(0, inode.min() - 1) == false);
            CHECK(inode.is_valid(0, inode.max() + 1) == false);
            CHECK(inode.is_valid(0, 5.5) == false);
            CHECK(inode.is_valid(0, inode.min()) == true);
            CHECK(inode.is_valid(0, inode.max()) == true);
            CHECK(inode.is_valid(0, 5) == true);
            CHECK(inode.lower_bound(0) == -5);
            CHECK(inode.upper_bound(0) == 10);
            CHECK(inode.lower_bound() == -5);
            CHECK(inode.upper_bound() == 10);
        }
    }

    GIVEN("Integer with an upper bound specified, but no lower bound") {
        IntegerNode inode({1}, std::nullopt, 10);

        THEN("The bounds are correct") {
            CHECK(inode.lower_bound(0) == IntegerNode::default_lower_bound);
            CHECK(inode.upper_bound(0) == 10);
            CHECK(inode.lower_bound() == IntegerNode::default_lower_bound);
            CHECK(inode.upper_bound() == 10);
        }
    }

    GIVEN("Integer with a lower bound specified, but no upper bound provided") {
        IntegerNode inode1({1}, 5);

        THEN("The bounds are correct") {
            CHECK(inode1.lower_bound(0) == 5);
            CHECK(inode1.upper_bound(0) == IntegerNode::default_upper_bound);
            CHECK(inode1.lower_bound() == 5);
            CHECK(inode1.upper_bound() == IntegerNode::default_upper_bound);
        }
    }

    GIVEN("Integer with a lower bound specified, but with unspecified upper bound provided") {
        IntegerNode inode1({1}, 5, std::nullopt);

        THEN("The bounds are correct") {
            CHECK(inode1.lower_bound(0) == 5);
            CHECK(inode1.upper_bound(0) == IntegerNode::default_upper_bound);
            CHECK(inode1.lower_bound() == 5);
            CHECK(inode1.upper_bound() == IntegerNode::default_upper_bound);
        }
    }

    GIVEN("Integer node with index-wise bounds") {
        auto inode_ptr = graph.emplace_node<IntegerNode>(3, std::vector<double>{-1, 3, 5},
                                                         std::vector<double>{1, 7, 7});

        THEN("The shape, max, min, and bounds are correct") {
            CHECK(inode_ptr->size() == 3);
            CHECK(inode_ptr->ndim() == 1);

            CHECK(inode_ptr->max() == 7.0);
            CHECK(inode_ptr->min() == -1.0);

            CHECK(inode_ptr->lower_bound(0) == -1.0);
            CHECK(inode_ptr->lower_bound(1) == 3.0);
            CHECK(inode_ptr->lower_bound(2) == 5.0);
            CHECK(inode_ptr->upper_bound(0) == 1.0);
            CHECK(inode_ptr->upper_bound(1) == 7.0);
            CHECK(inode_ptr->upper_bound(2) == 7.0);
            REQUIRE_THROWS(inode_ptr->lower_bound());
            REQUIRE_THROWS(inode_ptr->upper_bound());
        }

        AND_WHEN("We set the state at one of the indices") {
            auto state = graph.initialize_state();

            // Note, index-wise bounds are [[-1,1], [3,7], [5,7]]
            inode_ptr->set_value(state, 2, 6.0);

            THEN("The value is correct") {
                CHECK(inode_ptr->diff(state).size() == 1);
                CHECK(inode_ptr->get_value(state, 2) == 6.0);
            }
        }

        AND_WHEN("We set the state at the indices using clip") {
            auto state = graph.initialize_state();

            inode_ptr->clip_and_set_value(state, 0, -2);
            inode_ptr->clip_and_set_value(state, 1, 5);
            inode_ptr->clip_and_set_value(state, 2, 9);

            THEN("Clip set the values correctly") {
                // Note, index-wise bounds are [[-1,1], [3,7], [5,7]]
                CHECK(inode_ptr->get_value(state, 0) == -1);
                CHECK(inode_ptr->get_value(state, 1) == 5);
                CHECK(inode_ptr->get_value(state, 2) == 7);
            }

            AND_THEN("We commit the state") {
                graph.commit(state);

                THEN("We can exchange() within bounds") {
                    // Note, index-wise bounds are [[-1,1], [3,7], [5,7]]
                    CHECK(inode_ptr->get_value(state, 1) == 5);
                    CHECK(inode_ptr->get_value(state, 2) == 7);
                    inode_ptr->exchange(state, 1, 2);
                    CHECK(inode_ptr->get_value(state, 1) == 7);
                    CHECK(inode_ptr->get_value(state, 2) == 5);
                }
            }
        }
    }

    GIVEN("Integer node with index-wise upper bound and general integer lower bound") {
        auto inode_ptr = graph.emplace_node<IntegerNode>(2, 10, std::vector<double>{20, 10});

        THEN("The max, min, and bounds are correct") {
            CHECK(inode_ptr->max() == 20.0);
            CHECK(inode_ptr->min() == 10.0);

            CHECK(inode_ptr->lower_bound(0) == 10.0);
            CHECK(inode_ptr->lower_bound(1) == 10.0);
            CHECK(inode_ptr->upper_bound(0) == 20.0);
            CHECK(inode_ptr->upper_bound(1) == 10.0);
            CHECK(inode_ptr->lower_bound() == 10.0);
            REQUIRE_THROWS(inode_ptr->upper_bound());
        }
    }

    GIVEN("Integer node with invalid index-wise bounds at index 0") {
        REQUIRE_THROWS(graph.emplace_node<IntegerNode>(2, std::vector<double>{19, 12},
                                                       std::vector<double>{20, 11}));
    }

    GIVEN("An Integer Node representing an 1d array of 10 elements with lower bound -10") {
        auto ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{10}, -10);

        THEN("The shape is fixed") {
            CHECK(ptr->ndim() == 1);
            CHECK(ptr->size() == 10);
            CHECK_THAT(ptr->shape(), RangeEquals({10}));
            CHECK_THAT(ptr->strides(), RangeEquals({sizeof(double)}));
        }

        THEN("The bounds are correct") {
            CHECK(ptr->lower_bound() == -10.0);
            CHECK(ptr->upper_bound() == IntegerNode::default_upper_bound);
            for (int i = 0; i < 10; ++i) {
                CHECK(ptr->lower_bound(i) == -10);
                CHECK(ptr->upper_bound(i) == IntegerNode::default_upper_bound);
            }
        }

        WHEN("We create a state using the default value") {
            auto state = graph.initialize_state();

            THEN("All elements are zero") {
                CHECK(ptr->size(state) == 10);
                CHECK_THAT(ptr->shape(state), RangeEquals({10}));
                CHECK_THAT(ptr->view(state), RangeEquals({0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
            }
        }

        WHEN("We create a state using a random number generator") {
            auto state = graph.empty_state();
            auto rng = std::default_random_engine(42);
            ptr->initialize_state(state, rng);
            graph.initialize_state(state);
            auto state_view = ptr->view(state);

            THEN("Then all elements are integral and within range") {
                bool found_invalid = false;
                for (ssize_t i = 0, stop = state_view.size(); i < stop; ++i) {
                    if (!ptr->is_valid(i, state_view[i])) {
                        found_invalid = true;
                        break;
                    }
                }
                CHECK(!found_invalid);
            }
        }

        WHEN("We create a state using a specified values") {
            auto state = graph.empty_state();
            std::vector<double> vec_d{-4, -4, -2, -2, 0, 0, 2, 2, 4, 4};
            ptr->initialize_state(state, vec_d);
            graph.initialize_state(state);

            THEN("The passed in values are not modified") {
                CHECK_THAT(vec_d, RangeEquals({-4, -4, -2, -2, 0, 0, 2, 2, 4, 4}));
            }

            THEN("We can read the state") { CHECK(std::ranges::equal(ptr->view(state), vec_d)); }

            WHEN("We set half the elements to 9 and leave the other half unchange") {
                auto initial_state = vec_d;
                std::vector<double> new_values;
                auto view = ptr->view(state);
                for (int i = 0, stop = ptr->size(); i < stop; ++i) {
                    double new_val = (i % 2) ? 9 : view[i];
                    // Note, index-wise bounds are all [-10, 2000000000]
                    ptr->set_value(state, i, new_val);
                    new_values.push_back(new_val);
                }

                THEN("Elments are set properly") {
                    CHECK(std::ranges::equal(ptr->view(state), new_values));
                }

                THEN("The number of elements set is correct") {
                    CHECK(static_cast<ssize_t>(ptr->diff(state).size()) == ptr->size() / 2);
                }

                WHEN("We revert") {
                    ptr->revert(state);

                    THEN("We get the initial state back") {
                        CHECK(std::ranges::equal(ptr->view(state), initial_state));
                    }
                }
            }

            WHEN("We exchange the elements") {
                auto initial_state = vec_d;
                int exchange_count_ground = 0;
                for (int i = 0, stop = ptr->size() - 1; i < stop; ++i) {
                    // Note, index-wise bounds are all [-10, 2000000000]
                    ptr->exchange(state, i, i + 1);
                    std::swap(vec_d[i], vec_d[i + 1]);
                    exchange_count_ground += (vec_d[i] != vec_d[i + 1]);
                }

                THEN("Elments are exchanged properly") {
                    CHECK(std::ranges::equal(ptr->view(state), vec_d));
                }

                THEN("The number of effective exchanges are correct") {
                    // Two updates per exchange
                    CHECK(static_cast<int>(ptr->diff(state).size()) == 2 * exchange_count_ground);
                }

                WHEN("We revert") {
                    ptr->revert(state);

                    THEN("We get the initial state back") {
                        CHECK(std::ranges::equal(ptr->view(state), initial_state));
                    }
                }
            }
        }
    }

    GIVEN("Invalid dynamically sized IntegerNode") {
        REQUIRE_THROWS_WITH(graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{-1, 3}),
                            "Number array cannot have dynamic size.");
    }

    GIVEN("(2x3)-IntegerNode with axis-wise bounds on the invalid axis -2") {
        NumberNode::BoundAxisInfo bound_axis{
                -2, std::vector<NumberNode::BoundAxisOperator>{NumberNode::Equal},
                std::vector<double>{20.0}};
        REQUIRE_THROWS_WITH(
                graph.emplace_node<dwave::optimization::IntegerNode>(
                        std::initializer_list<ssize_t>{2, 3}, std::nullopt, std::nullopt,
                        std::vector<NumberNode::BoundAxisInfo>{bound_axis}),
                "Invalid bound axis: -2. Note, negative indexing is not supported for "
                "axis-wise bounds.");
    }

    GIVEN("(2x3x4)-IntegerNode with axis-wise bounds on the invalid axis 3") {
        NumberNode::BoundAxisInfo bound_axis{
                3, std::vector<NumberNode::BoundAxisOperator>{NumberNode::Equal},
                std::vector<double>{10.0}};
        REQUIRE_THROWS_WITH(
                graph.emplace_node<dwave::optimization::IntegerNode>(
                        std::initializer_list<ssize_t>{2, 3, 4}, std::nullopt, std::nullopt,
                        std::vector<NumberNode::BoundAxisInfo>{bound_axis}),
                "Invalid bound axis: 3. Note, negative indexing is not supported for "
                "axis-wise bounds.");
    }

    GIVEN("(2x3x4)-IntegerNode with axis-wise bounds on axis: 1 with too many operators.") {
        NumberNode::BoundAxisInfo bound_axis{
                1,
                std::vector<NumberNode::BoundAxisOperator>{NumberNode::LessEqual, NumberNode::Equal,
                                                           NumberNode::Equal, NumberNode::Equal},
                std::vector<double>{-10.0}};
        REQUIRE_THROWS_WITH(
                graph.emplace_node<dwave::optimization::IntegerNode>(
                        std::initializer_list<ssize_t>{2, 3, 4}, std::nullopt, std::nullopt,
                        std::vector<NumberNode::BoundAxisInfo>{bound_axis}),
                "Invalid number of axis-wise operators along axis: 1 given axis size: 3");
    }

    GIVEN("(2x3x4)-IntegerNode with axis-wise bounds on axis: 1 with too few operators.") {
        NumberNode::BoundAxisInfo bound_axis{1,
                                             std::vector<NumberNode::BoundAxisOperator>{
                                                     NumberNode::LessEqual, NumberNode::Equal},
                                             std::vector<double>{-11.0}};
        REQUIRE_THROWS_WITH(
                graph.emplace_node<dwave::optimization::IntegerNode>(
                        std::initializer_list<ssize_t>{2, 3, 4}, std::nullopt, std::nullopt,
                        std::vector<NumberNode::BoundAxisInfo>{bound_axis}),
                "Invalid number of axis-wise operators along axis: 1 given axis size: 3");
    }

    GIVEN("(2x3x4)-IntegerNode with axis-wise bounds on axis: 1 with too many bounds.") {
        NumberNode::BoundAxisInfo bound_axis{
                1, std::vector<NumberNode::BoundAxisOperator>{NumberNode::LessEqual},
                std::vector<double>{-10.0, 20.0, 30.0, 40.0}};
        REQUIRE_THROWS_WITH(
                graph.emplace_node<dwave::optimization::IntegerNode>(
                        std::initializer_list<ssize_t>{2, 3, 4}, std::nullopt, std::nullopt,
                        std::vector<NumberNode::BoundAxisInfo>{bound_axis}),
                "Invalid number of axis-wise bounds along axis: 1 given axis size: 3");
    }

    GIVEN("(2x3x4)-IntegerNode with axis-wise bounds on axis: 1 with too few bounds.") {
        NumberNode::BoundAxisInfo bound_axis{
                1, std::vector<NumberNode::BoundAxisOperator>{NumberNode::LessEqual},
                std::vector<double>{111.0, -223.0}};
        REQUIRE_THROWS_WITH(
                graph.emplace_node<dwave::optimization::IntegerNode>(
                        std::initializer_list<ssize_t>{2, 3, 4}, std::nullopt, std::nullopt,
                        std::vector<NumberNode::BoundAxisInfo>{bound_axis}),
                "Invalid number of axis-wise bounds along axis: 1 given axis size: 3");
    }

    GIVEN("(2x3x4)-IntegerNode with duplicate axis-wise bounds on axis: 1") {
        NumberNode::BoundAxisInfo bound_axis{
                1, std::vector<NumberNode::BoundAxisOperator>{NumberNode::Equal},
                std::vector<double>{100.0}};
        REQUIRE_THROWS_WITH(
                graph.emplace_node<dwave::optimization::IntegerNode>(
                        std::initializer_list<ssize_t>{2, 3, 4}, std::nullopt, std::nullopt,
                        std::vector<NumberNode::BoundAxisInfo>{bound_axis, bound_axis}),
                "Cannot define multiple axis-wise bounds for a single axis.");
    }

    GIVEN("(2x3x4)-IntegerNode with axis-wise bounds on axes: 0 and 1") {
        NumberNode::BoundAxisInfo bound_axis_0{
                0, std::vector<NumberNode::BoundAxisOperator>{NumberNode::LessEqual},
                std::vector<double>{11.0}};
        NumberNode::BoundAxisInfo bound_axis_1{
                1, std::vector<NumberNode::BoundAxisOperator>{NumberNode::LessEqual},
                std::vector<double>{12.0}};
        REQUIRE_THROWS_WITH(
                graph.emplace_node<dwave::optimization::IntegerNode>(
                        std::initializer_list<ssize_t>{2, 3, 4}, std::nullopt, std::nullopt,
                        std::vector<NumberNode::BoundAxisInfo>{bound_axis_0, bound_axis_1}),
                "Axis-wise bounds are supported for at most one axis.");
    }

    GIVEN("(2x3x4)-IntegerNode with non-integral axis-wise bounds") {
        NumberNode::BoundAxisInfo bound_axis{
                2, std::vector<NumberNode::BoundAxisOperator>{NumberNode::LessEqual},
                std::vector<double>{11.0, 12.0001, 0.0, 0.0}};
        REQUIRE_THROWS_WITH(
                graph.emplace_node<dwave::optimization::IntegerNode>(
                        std::initializer_list<ssize_t>{2, 3, 4}, std::nullopt, std::nullopt,
                        std::vector<NumberNode::BoundAxisInfo>{bound_axis}),
                "Axis wise bounds for integral number arrays must be intregral.");
    }

    GIVEN("(2x3x2)-IntegerNode with infeasible axis-wise bound on axis: 0") {
        auto graph = Graph();

        NumberNode::BoundAxisInfo bound_axis{0,
                                             std::vector<NumberNode::BoundAxisOperator>{
                                                     NumberNode::Equal, NumberNode::LessEqual},
                                             std::vector<double>{5.0, -31.0}};

        graph.emplace_node<dwave::optimization::IntegerNode>(
                std::initializer_list<ssize_t>{2, 3, 2}, -5, 8,
                std::vector<NumberNode::BoundAxisInfo>{bound_axis});

        WHEN("We create a state by initialize_state()") {
            // Each hyperslice along axis 0 has size 6. There is no feasible
            // assignment to the values in slice 1 (along axis 0) that results in a
            // sum less than or equal to -5*6-1 = -31.
            REQUIRE_THROWS_WITH(graph.initialize_state(), "Infeasible axis-wise bounds.");
        }
    }

    GIVEN("(2x3x2)-IntegerNode with infeasible axis-wise bound on axis: 1") {
        auto graph = Graph();

        NumberNode::BoundAxisInfo bound_axis{
                1,
                std::vector<NumberNode::BoundAxisOperator>{NumberNode::GreaterEqual,
                                                           NumberNode::Equal, NumberNode::Equal},
                std::vector<double>{33.0, 0.0, 0.0}};

        graph.emplace_node<dwave::optimization::IntegerNode>(
                std::initializer_list<ssize_t>{2, 3, 2}, -5, 8,
                std::vector<NumberNode::BoundAxisInfo>{bound_axis});

        WHEN("We create a state by initialize_state()") {
            // Each hyperslice along axis 1 has size 4. There is no feasible
            // assignment to the values in slice 0 (along axis 1) that results in a
            // sum greater than or equal to 4*8+1 = 33.
            REQUIRE_THROWS_WITH(graph.initialize_state(), "Infeasible axis-wise bounds.");
        }
    }

    GIVEN("(2x3x2)-IntegerNode with infeasible axis-wise bound on axis: 2") {
        auto graph = Graph();

        NumberNode::BoundAxisInfo bound_axis{2,
                                             std::vector<NumberNode::BoundAxisOperator>{
                                                     NumberNode::GreaterEqual, NumberNode::Equal},
                                             std::vector<double>{-1.0, 49.0}};

        graph.emplace_node<dwave::optimization::IntegerNode>(
                std::initializer_list<ssize_t>{2, 3, 2}, -5, 8,
                std::vector<NumberNode::BoundAxisInfo>{bound_axis});

        WHEN("We create a state by initialize_state()") {
            // Each hyperslice along axis 2 has size 6. There is no feasible
            // assignment to the values in slice 1 (along axis 2) that results in a
            // sum or equal to 6*8+1 = 49
            REQUIRE_THROWS_WITH(graph.initialize_state(), "Infeasible axis-wise bounds.");
        }
    }

    GIVEN("(2x3x2)-IntegerNode with feasible axis-wise bound on axis: 0") {
        auto graph = Graph();

        NumberNode::BoundAxisInfo bound_axis{0,
                                             std::vector<NumberNode::BoundAxisOperator>{
                                                     NumberNode::Equal, NumberNode::GreaterEqual},
                                             std::vector<double>{-21.0, 9.0}};

        auto bnode_ptr = graph.emplace_node<dwave::optimization::IntegerNode>(
                std::initializer_list<ssize_t>{2, 3, 2}, -5, 8,
                std::vector<NumberNode::BoundAxisInfo>{bound_axis});

        THEN("Axis wise bound is correct") {
            CHECK(bnode_ptr->axis_wise_bounds().size() == 1);
            NumberNode::BoundAxisInfo bnode_bound_axis = bnode_ptr->axis_wise_bounds()[0];
            CHECK(bound_axis.axis == bnode_bound_axis.axis);
            CHECK_THAT(bound_axis.operators, RangeEquals(bnode_bound_axis.operators));
            CHECK_THAT(bound_axis.bounds, RangeEquals(bnode_bound_axis.bounds));
        }

        WHEN("We create a state by initialize_state()") {
            auto state = graph.initialize_state();
            graph.initialize_state(state);
            // import numpy as np
            // a = np.asarray([i for i in range(2*3*2)]).reshape(2, 3, 2)
            // print(a[0, :, :].flatten())
            // ... [0 1 2 3 4 5]
            // print(a[1, :, :].flatten())
            // ... [ 6  7  8  9 10 11]
            //
            // initialize_state() will start with
            // [-5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5]
            // repair slice 0
            // [4, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5]
            // repair slice 1
            // [4, -5, -5, -5, -5, -5, 8, 8, 8, -5, -5, -5]
            std::vector<double> expected_init{4, -5, -5, -5, -5, -5, 8, 8, 8, -5, -5, -5};
            auto bound_axis_sums = bnode_ptr->bound_axis_sums(state);

            THEN("The bound axis sums and state are correct") {
                CHECK(bnode_ptr->bound_axis_sums(state).size() == 1);
                CHECK(bnode_ptr->bound_axis_sums(state).data()[0].size() == 2);
                CHECK_THAT(bnode_ptr->bound_axis_sums(state)[0], RangeEquals({-21.0, 9.0}));
                CHECK_THAT(bnode_ptr->view(state), RangeEquals(expected_init));
            }
        }
    }

    GIVEN("(2x3x2)-IntegerNode with feasible axis-wise bound on axis: 1") {
        auto graph = Graph();

        NumberNode::BoundAxisInfo bound_axis{
                1,
                std::vector<NumberNode::BoundAxisOperator>{
                        NumberNode::Equal, NumberNode::GreaterEqual, NumberNode::LessEqual},
                std::vector<double>{0.0, -2.0, 0.0}};

        auto bnode_ptr = graph.emplace_node<dwave::optimization::IntegerNode>(
                std::initializer_list<ssize_t>{2, 3, 2}, -5, 8,
                std::vector<NumberNode::BoundAxisInfo>{bound_axis});

        THEN("Axis wise bound is correct") {
            CHECK(bnode_ptr->axis_wise_bounds().size() == 1);
            NumberNode::BoundAxisInfo bnode_bound_axis = bnode_ptr->axis_wise_bounds()[0];
            CHECK(bound_axis.axis == bnode_bound_axis.axis);
            CHECK_THAT(bound_axis.operators, RangeEquals(bnode_bound_axis.operators));
            CHECK_THAT(bound_axis.bounds, RangeEquals(bnode_bound_axis.bounds));
        }

        WHEN("We create a state by initialize_state()") {
            auto state = graph.initialize_state();
            graph.initialize_state(state);
            // import numpy as np
            // a = np.asarray([i for i in range(2*3*2)]).reshape(2, 3, 2)
            // print(a[:, 0, :].flatten())
            // ... [0 1 6 7]
            // print(a[:, 1, :].flatten())
            // ... [2 3 8 9]
            // print(a[:, 2, :].flatten())
            // ... [ 4  5 10 11]
            //
            // initialize_state() will start with
            // [-5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5]
            // repair slice 0 w/ [8, 2, -5, -5]
            // [8, 2, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5]
            // repair slice 1 w/ [8, 0, -5, -5]
            // [8, 2, 8, 0, -5, -5, -5, -5, -5, -5, -5, -5]
            // no need to repair slice 2
            std::vector<double> expected_init{8, 2, 8, 0, -5, -5, -5, -5, -5, -5, -5, -5};
            auto bound_axis_sums = bnode_ptr->bound_axis_sums(state);

            THEN("The bound axis sums and state are correct") {
                CHECK(bnode_ptr->bound_axis_sums(state).size() == 1);
                CHECK(bnode_ptr->bound_axis_sums(state).data()[0].size() == 3);
                CHECK_THAT(bnode_ptr->bound_axis_sums(state)[0], RangeEquals({0.0, -2.0, -20.0}));
                CHECK_THAT(bnode_ptr->view(state), RangeEquals(expected_init));
            }
        }
    }

    GIVEN("(2x3x2)-IntegerNode with feasible axis-wise bound on axis: 2") {
        auto graph = Graph();

        NumberNode::BoundAxisInfo bound_axis{2,
                                             std::vector<NumberNode::BoundAxisOperator>{
                                                     NumberNode::Equal, NumberNode::GreaterEqual},
                                             std::vector<double>{23.0, 14.0}};

        auto bnode_ptr = graph.emplace_node<dwave::optimization::IntegerNode>(
                std::initializer_list<ssize_t>{2, 3, 2}, -5, 8,
                std::vector<NumberNode::BoundAxisInfo>{bound_axis});

        THEN("Axis wise bound is correct") {
            CHECK(bnode_ptr->axis_wise_bounds().size() == 1);
            NumberNode::BoundAxisInfo bnode_bound_axis = bnode_ptr->axis_wise_bounds()[0];
            CHECK(bound_axis.axis == bnode_bound_axis.axis);
            CHECK_THAT(bound_axis.operators, RangeEquals(bnode_bound_axis.operators));
            CHECK_THAT(bound_axis.bounds, RangeEquals(bnode_bound_axis.bounds));
        }

        WHEN("We create a state by initialize_state()") {
            auto state = graph.initialize_state();
            graph.initialize_state(state);
            // import numpy as np
            // a = np.asarray([i for i in range(2*3*2)]).reshape(2, 3, 2)
            // print(a[:, :, 0].flatten())
            // ... [ 0  2  4  6  8 10]
            // print(a[:, :, 0].flatten())
            // ... [ 1  3  5  7  9 11]
            //
            // initialize_state() will start with
            // [-5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5]
            // repair slice 0 w/ [8, 8, 8, 8, -4, -5]
            // [8, -5, 8, -5, 8, -5, 8, -5, -4, -5, -5, -5]
            // repair slice 0 w/ [8, 8, 8, 0, -5, -5]
            // [8, 8, 8, 8, 8, 8, 8, 0, -4, -5, -5, -5]
            std::vector<double> expected_init{8, 8, 8, 8, 8, 8, 8, 0, -4, -5, -5, -5};
            auto bound_axis_sums = bnode_ptr->bound_axis_sums(state);

            THEN("The bound axis sums and state are correct") {
                CHECK(bnode_ptr->bound_axis_sums(state).size() == 1);
                CHECK(bnode_ptr->bound_axis_sums(state).data()[0].size() == 2);
                CHECK_THAT(bnode_ptr->bound_axis_sums(state)[0], RangeEquals({23.0, 14.0}));
                CHECK_THAT(bnode_ptr->view(state), RangeEquals(expected_init));
            }
        }
    }

    GIVEN("(2x3x2)-IntegerNode with index-wise bounds and an axis-wise bound on axis: 1") {
        auto graph = Graph();

        NumberNode::BoundAxisInfo bound_axis{
                1,
                std::vector<NumberNode::BoundAxisOperator>{NumberNode::Equal, NumberNode::LessEqual,
                                                           NumberNode::GreaterEqual},
                std::vector<double>{11.0, 2.0, 5.0}};

        auto inode_ptr = graph.emplace_node<dwave::optimization::IntegerNode>(
                std::initializer_list<ssize_t>{2, 3, 2}, -5, 8,
                std::vector<NumberNode::BoundAxisInfo>{bound_axis});

        THEN("Axis wise bound is correct") {
            CHECK(inode_ptr->axis_wise_bounds().size() == 1);
            const NumberNode::BoundAxisInfo inode_bound_axis_ptr =
                    inode_ptr->axis_wise_bounds().data()[0];
            CHECK(bound_axis.axis == inode_bound_axis_ptr.axis);
            CHECK_THAT(bound_axis.operators, RangeEquals(inode_bound_axis_ptr.operators));
            CHECK_THAT(bound_axis.bounds, RangeEquals(inode_bound_axis_ptr.bounds));
        }

        WHEN("We initialize three invalid states") {
            auto state = graph.empty_state();
            // This state violates the 0th hyperslice along axis 1
            std::vector<double> init_values{5, 6, 0, 0, 3, 1, 4, 0, 2, 0, 0, 3};
            // import numpy as np
            // a = np.asarray([5, 6, 0, 0, 3, 1, 4, 0, 2, 0, 0, 3])
            // a = a.reshape(2, 3, 2)
            // a.sum(axis=(0, 2))
            // >>> array([15, 2, 7])
            CHECK_THROWS_WITH(inode_ptr->initialize_state(state, init_values),
                              "Initialized values do not satisfy axis-wise bounds.");

            state = graph.empty_state();
            // This state violates the 1st hyperslice along axis 1
            init_values = {5, 2, 0, 0, 3, 1, 4, 0, 2, 1, 0, 3};
            // import numpy as np
            // a = np.asarray([5, 2, 0, 0, 3, 1, 4, 0, 2, 1, 0, 3])
            // a = a.reshape(2, 3, 2)
            // a.sum(axis=(0, 2))
            // >>> array([11, 3, 7])
            CHECK_THROWS_WITH(inode_ptr->initialize_state(state, init_values),
                              "Initialized values do not satisfy axis-wise bounds.");

            state = graph.empty_state();
            // This state violates the 2nd hyperslice along axis 1
            init_values = {5, 2, 0, 0, 3, 1, 4, 0, 1, 0, 0, 0};
            // import numpy as np
            // a = np.asarray([5, 2, 0, 0, 3, 1, 4, 0, 1, 0, 0, 0])
            // a = a.reshape(2, 3, 2)
            // a.sum(axis=(0, 2))
            // >>> array([11, 1, 4])
            CHECK_THROWS_WITH(inode_ptr->initialize_state(state, init_values),
                              "Initialized values do not satisfy axis-wise bounds.");
        }

        WHEN("We initialize a valid state") {
            auto state = graph.empty_state();
            std::vector<double> init_values{5, 2, 0, 0, 3, 1, 4, 0, 2, 0, 0, 3};
            inode_ptr->initialize_state(state, init_values);
            graph.initialize_state(state);

            auto bound_axis_sums = inode_ptr->bound_axis_sums(state);

            THEN("The bound axis sums and state are correct") {
                // **Python Code 2**
                // import numpy as np
                // a = np.asarray([5, 2, 0, 0, 3, 1, 4, 0, 2, 0, 0, 3])
                // a = a.reshape(2, 3, 2)
                // a.sum(axis=(0, 2))
                // >>> array([11, 2, 7])
                CHECK(inode_ptr->bound_axis_sums(state).size() == 1);
                CHECK(inode_ptr->bound_axis_sums(state).data()[0].size() == 3);
                CHECK_THAT(inode_ptr->bound_axis_sums(state)[0], RangeEquals({11, 2, 7}));
                CHECK_THAT(inode_ptr->view(state), RangeEquals(init_values));
            }

            THEN("We exchange() some values") {
                inode_ptr->exchange(state, 2, 3);  // Does nothing.
                inode_ptr->exchange(state, 1, 8);  // Does nothing.
                inode_ptr->exchange(state, 8, 10);
                inode_ptr->exchange(state, 0, 1);
                std::swap(init_values[2], init_values[3]);
                std::swap(init_values[1], init_values[8]);
                std::swap(init_values[8], init_values[10]);
                std::swap(init_values[0], init_values[1]);
                // state is now: [2, 5, 0, 0, 3, 1, 4, 0, 0, 0, 2, 3]

                THEN("The bound axis sums and state updated correctly") {
                    // Cont. w/ Python code at **Python Code 2**
                    // a[np.unravel_index(8, a.shape)] = 0
                    // a[np.unravel_index(10, a.shape)] = 2
                    // a[np.unravel_index(0, a.shape)] = 2
                    // a[np.unravel_index(1, a.shape)] = 5
                    // a.sum(axis=(0, 2))
                    CHECK_THAT(inode_ptr->bound_axis_sums(state)[0], RangeEquals({11, 0, 9}));
                    CHECK(inode_ptr->diff(state).size() == 4);  // 2 updates per exchange
                    CHECK_THAT(inode_ptr->view(state), RangeEquals(init_values));
                }

                AND_WHEN("We revert") {
                    graph.revert(state);

                    THEN("The bound axis sums reverted correctly") {
                        CHECK_THAT(inode_ptr->bound_axis_sums(state)[0], RangeEquals({11, 2, 7}));
                        CHECK(inode_ptr->diff(state).size() == 0);
                    }
                }
            }

            THEN("We clip_and_set_value() some values") {
                inode_ptr->clip_and_set_value(state, 0, 5);  // Does nothing.
                inode_ptr->clip_and_set_value(state, 8, -300);
                inode_ptr->clip_and_set_value(state, 10, 100);
                init_values[8] = -5;
                init_values[10] = 8;
                // state is now: [5,  2,  0,  0,  3,  1,  4,  0, -5,  0,  8,  3]

                THEN("The bound axis sums and state updated correctly") {
                    // Cont. w/ Python code at **Python Code 2**
                    // a[np.unravel_index(8, a.shape)] = -5
                    // a[np.unravel_index(10, a.shape)] = 8
                    // a.sum(axis=(0, 2))
                    CHECK_THAT(inode_ptr->bound_axis_sums(state)[0], RangeEquals({11, -5, 15}));
                    CHECK(inode_ptr->diff(state).size() == 2);
                    CHECK_THAT(inode_ptr->view(state), RangeEquals(init_values));
                }

                AND_WHEN("We revert") {
                    graph.revert(state);

                    THEN("The bound axis sums reverted correctly") {
                        CHECK_THAT(inode_ptr->bound_axis_sums(state)[0], RangeEquals({11, 2, 7}));
                        CHECK(inode_ptr->diff(state).size() == 0);
                    }
                }
            }

            THEN("We set_value() some values") {
                inode_ptr->set_value(state, 0, 5);  // Does nothing.
                inode_ptr->set_value(state, 8, 0);
                inode_ptr->set_value(state, 9, 1);
                inode_ptr->set_value(state, 10, 5);
                inode_ptr->set_value(state, 11, 0);
                init_values[0] = 5;
                init_values[8] = 0;
                init_values[9] = 1;
                init_values[10] = 5;
                init_values[11] = 0;
                // state is now: [5, 2, 0, 0, 3, 1, 4, 0, 0, 1, 5, 0]

                THEN("The bound axis sums and state updated correctly") {
                    // Cont. w/ Python code at **Python Code 2**
                    // a[np.unravel_index(0, a.shape)] = 5
                    // a[np.unravel_index(8, a.shape)] = 0
                    // a[np.unravel_index(9, a.shape)] = 1
                    // a[np.unravel_index(10, a.shape)] = 5
                    // a[np.unravel_index(11, a.shape)] = 0
                    // a.sum(axis=(0, 2))
                    CHECK_THAT(inode_ptr->bound_axis_sums(state)[0], RangeEquals({11, 1, 9}));
                    CHECK(inode_ptr->diff(state).size() == 4);
                    CHECK_THAT(inode_ptr->view(state), RangeEquals(init_values));
                }

                AND_WHEN("We revert") {
                    graph.revert(state);

                    THEN("The bound axis sums reverted correctly") {
                        CHECK_THAT(bound_axis_sums[0], RangeEquals({11, 2, 7}));
                        CHECK(inode_ptr->diff(state).size() == 0);
                    }
                }
            }
        }
    }
}

}  // namespace dwave::optimization

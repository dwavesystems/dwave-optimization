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

#include <optional>

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers.hpp"
#include "catch2/matchers/catch_matchers_all.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/nodes/numbers.hpp"

using Catch::Matchers::RangeEquals;

namespace dwave::optimization {

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
                    CHECK(ptr->flip(state, i));
                    vec_d[i] = !vec_d[i];
                }

                THEN("Elments are flipped properly") {
                    CHECK_THAT(ptr->view(state), RangeEquals(vec_d));
                }
            }

            WHEN("We set all the elements") {
                auto set_count = 0;
                auto set_count_ground = 0;
                for (int i = 0, stop = ptr->size(); i < stop; ++i) {
                    set_count += ptr->set(state, i);
                    set_count_ground += !vec_d[i];
                }

                THEN("Elments are set properly") {
                    CHECK(std::ranges::all_of(ptr->view(state), [](int i) { return i == 1; }));
                }

                THEN("The number of elements set equals the number of initially unset elements") {
                    CHECK(set_count == set_count_ground);
                }
            }

            WHEN("We unset all the elements") {
                auto unset_count = 0;
                auto unset_count_ground = 0;
                for (int i = 0, stop = ptr->size(); i < stop; ++i) {
                    unset_count += ptr->unset(state, i);
                    unset_count_ground += vec_d[i];
                }

                THEN("Elments are unset properly") {
                    CHECK(std::ranges::all_of(ptr->view(state), [](int i) { return i == 0; }));
                }

                THEN("The number of elements unset equals the number of initially set elements") {
                    CHECK(unset_count == unset_count_ground);
                }
            }

            WHEN("We exchange the elements") {
                auto exchange_count = 0;
                auto exchange_count_ground = 0;
                for (int i = 0, stop = ptr->size() - 1; i < stop; ++i) {
                    exchange_count += ptr->exchange(state, i, i + 1);
                    std::swap(vec_d[i], vec_d[i + 1]);
                    exchange_count_ground += (vec_d[i] != vec_d[i + 1]);
                }

                THEN("Elments are exchanged properly") {
                    CHECK(std::ranges::equal(ptr->view(state), vec_d));
                }

                THEN("The number of effective exchanges are correct") {
                    CHECK(exchange_count == exchange_count_ground);
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
                    CHECK(ptr->flip(state, i));
                    vec_d[i] = !vec_d[i];
                }

                THEN("Elments are flipped properly") {
                    CHECK(std::ranges::equal(ptr->view(state), vec_d));
                }
            }

            WHEN("We set all the elements") {
                auto set_count = 0;
                auto set_count_ground = 0;
                for (int i = 0, stop = ptr->size(); i < stop; ++i) {
                    set_count += ptr->set(state, i);
                    set_count_ground += !vec_d[i];
                }

                THEN("Elments are set properly") {
                    CHECK(std::ranges::all_of(ptr->view(state), [](int i) { return i == 1; }));
                }

                THEN("The number of elements set equals the number of initially unset elements") {
                    CHECK(set_count == set_count_ground);
                }
            }

            WHEN("We unset all the elements") {
                auto unset_count = 0;
                auto unset_count_ground = 0;
                for (int i = 0, stop = ptr->size(); i < stop; ++i) {
                    unset_count += ptr->unset(state, i);
                    unset_count_ground += vec_d[i];
                }

                THEN("Elments are unset properly") {
                    CHECK(std::ranges::all_of(ptr->view(state), [](int i) { return i == 0; }));
                }

                THEN("The number of elements unset equals the number of initially set elements") {
                    CHECK(unset_count == unset_count_ground);
                }
            }

            WHEN("We exchange the elements") {
                auto exchange_count = 0;
                auto exchange_count_ground = 0;
                for (int i = 0, stop = ptr->size() - 1; i < stop; ++i) {
                    exchange_count += ptr->exchange(state, i, i + 1);
                    std::swap(vec_d[i], vec_d[i + 1]);
                    exchange_count_ground += (vec_d[i] != vec_d[i + 1]);
                }

                THEN("Elments are exchanged properly") {
                    CHECK(std::ranges::equal(ptr->view(state), vec_d));
                }

                THEN("The number of effective exchanges are correct") {
                    CHECK(exchange_count == exchange_count_ground);
                }
            }
        }
    }

    GIVEN("Binary node with index-wise bounds") {
        auto bnode_ptr = graph.emplace_node<dwave::optimization::BinaryNode>(
                3, std::vector<double>{-1, 0, 1}, std::vector<double>{2, 1, 1});

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

            THEN("An exception is raised if it's out of bounds") {
                REQUIRE_THROWS(bnode_ptr->set_value(state, 0, 1.9));
            }

            THEN("The value is within bounds") { CHECK(bnode_ptr->is_valid(0, 1.0)); }

            CHECK(bnode_ptr->set_value(state, 0, 1.0));

            THEN("The value is correct") { CHECK(bnode_ptr->get_value(state, 0) == 1.0); }

            AND_WHEN("We commit the state") {
                graph.commit(state);

                THEN("Cannot flip() index outside of bounds") {
                    CHECK(bnode_ptr->flip(state, 2) == false);
                }

                THEN("Cannot exchange() index outside of bounds") {
                    CHECK(bnode_ptr->exchange(state, 0, 2) == false);
                }

                THEN("Cannot unset() index outside of bounds") {
                    CHECK(bnode_ptr->unset(state, 2) == false);
                }
            }
        }

        AND_WHEN("We set the state at the indices using set()") {
            auto state = graph.initialize_state();
            THEN("The value is within bounds") { CHECK(bnode_ptr->is_valid(0, 0.0)); }
            THEN("The value is within bounds") { CHECK(bnode_ptr->is_valid(1, 1.0)); }

            CHECK(bnode_ptr->set(state, 1) == true);

            THEN("The value at index 0 is correct") {
                CHECK(bnode_ptr->get_value(state, 0) == 0.0);
            }
            THEN("The value at index 1 is correct") {
                CHECK(bnode_ptr->get_value(state, 1) == 1.0);
            }

            AND_WHEN("We commit the state") {
                graph.commit(state);

                THEN("We can perform a flip()") {
                    CHECK(bnode_ptr->flip(state, 1) == true);
                    CHECK(bnode_ptr->get_value(state, 1) == 0.0);
                }

                THEN("We can perform an unset()") {
                    CHECK(bnode_ptr->unset(state, 1) == true);
                    CHECK(bnode_ptr->get_value(state, 1) == 0.0);
                }

                THEN("We can perform an unset()") {
                    CHECK(bnode_ptr->set(state, 0) == true);
                    CHECK(bnode_ptr->get_value(state, 0) == 1.0);
                }

                THEN("We can perform an exchange()") {
                    CHECK(bnode_ptr->exchange(state, 0, 1) == true);
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
                CHECK(bnode_ptr->get_value(state, 0) == 0);
                CHECK(bnode_ptr->get_value(state, 1) == 0);
                CHECK(bnode_ptr->get_value(state, 2) == 1);
            }
        }
    }

    GIVEN("Binary node with index-wise upper bound and general lower bound") {
        auto bnode_ptr = graph.emplace_node<dwave::optimization::BinaryNode>(
                2, -2.0, std::vector<double>{0.0, 1.1});

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
        auto bnode_ptr = graph.emplace_node<dwave::optimization::BinaryNode>(
                2, std::vector<double>{-1.0, 1.0}, 100.0);

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
        REQUIRE_THROWS(graph.emplace_node<dwave::optimization::BinaryNode>(
                2, std::vector<double>{2, 0}, std::vector<double>{1, 1}));
    }

    GIVEN("Binary node with invalid index-wise upper bounds at index 1") {
        REQUIRE_THROWS(graph.emplace_node<dwave::optimization::BinaryNode>(
                2, std::vector<double>{0, 0}, std::vector<double>{1, -1}));
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
        auto inode_ptr = graph.emplace_node<dwave::optimization::IntegerNode>(
                3, std::vector<double>{-1, 3, 5}, std::vector<double>{1, 7, 7});

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
            THEN("An exception is raised if it's out of bounds") {
                REQUIRE_THROWS(inode_ptr->set_value(state, 2, -1.0));
            }

            THEN("The value is within bounds") { CHECK(inode_ptr->is_valid(2, 6.0)); }

            CHECK(inode_ptr->set_value(state, 2, 6.0) == true);

            THEN("The value is correct") { CHECK(inode_ptr->get_value(state, 2) == 6.0); }
        }

        AND_WHEN("We set the state at the indices using clip") {
            auto state = graph.initialize_state();

            inode_ptr->clip_and_set_value(state, 0, -2);
            inode_ptr->clip_and_set_value(state, 1, 5);
            inode_ptr->clip_and_set_value(state, 2, 9);

            THEN("Clip set the values correctly") {
                CHECK(inode_ptr->get_value(state, 0) == -1);
                CHECK(inode_ptr->get_value(state, 1) == 5);
                CHECK(inode_ptr->get_value(state, 2) == 7);
            }

            AND_THEN("We commit the state") {
                graph.commit(state);

                THEN("We cannot exchange() outside of bounds") {
                    CHECK(inode_ptr->exchange(state, 0, 2) == false);
                }

                THEN("We can exchange() within bounds") {
                    CHECK(inode_ptr->exchange(state, 1, 2) == true);
                    CHECK(inode_ptr->get_value(state, 1) == 7);
                    CHECK(inode_ptr->get_value(state, 2) == 5);
                }
            }
        }
    }

    GIVEN("Integer node with index-wise upper bound and general integer lower bound") {
        auto inode_ptr = graph.emplace_node<dwave::optimization::IntegerNode>(
                2, 10, std::vector<double>{20, 10});

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
        REQUIRE_THROWS(graph.emplace_node<dwave::optimization::IntegerNode>(
                2, std::vector<double>{19, 12}, std::vector<double>{20, 11}));
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

            WHEN("We set all the elements to different values") {
                auto initial_state = vec_d;
                auto set_count = 0;
                std::vector<double> new_values;
                auto view = ptr->view(state);
                for (int i = 0, stop = ptr->size(); i < stop; ++i) {
                    double new_val = (i % 2) ? 9 : view[i];
                    set_count += ptr->set_value(state, i, new_val);
                    new_values.push_back(new_val);
                }

                THEN("Elments are set properly") {
                    CHECK(std::ranges::equal(ptr->view(state), new_values));
                }

                THEN("The number of elements set is correct") {
                    CHECK(set_count == ptr->size() / 2);
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
                auto exchange_count = 0;
                auto exchange_count_ground = 0;
                for (int i = 0, stop = ptr->size() - 1; i < stop; ++i) {
                    exchange_count += ptr->exchange(state, i, i + 1);
                    std::swap(vec_d[i], vec_d[i + 1]);
                    exchange_count_ground += (vec_d[i] != vec_d[i + 1]);
                }

                THEN("Elments are exchanged properly") {
                    CHECK(std::ranges::equal(ptr->view(state), vec_d));
                }

                THEN("The number of effective exchanges are correct") {
                    CHECK(exchange_count == exchange_count_ground);
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
}

}  // namespace dwave::optimization

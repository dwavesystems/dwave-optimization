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

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_all.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/nodes/numbers.hpp"

using Catch::Matchers::RangeEquals;

namespace dwave::optimization {

TEST_CASE("BinaryNode") {
    auto graph = Graph();

    GIVEN("A Binary Node representing an 1d array of 10 elements") {
        auto ptr = graph.emplace_node<BinaryNode>(std::initializer_list<ssize_t>{10});

        THEN("The state is not deterministic") {
            CHECK(!ptr->deterministic_state());
        }

        THEN("The shape is fixed") {
            CHECK(ptr->ndim() == 1);
            CHECK(ptr->size() == 10);
            CHECK_THAT(ptr->shape(), RangeEquals({10}));
            CHECK_THAT(ptr->strides(), RangeEquals({sizeof(double)}));
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
                for (int i = 0; i < ptr->size(); i++) {
                    ptr->flip(state, i);
                    vec_d[i] = !vec_d[i];
                }

                THEN("Elments are flipped properly") {
                    CHECK_THAT(ptr->view(state), RangeEquals(vec_d));
                }
            }

            WHEN("We set all the elements") {
                auto set_count = 0;
                auto set_count_ground = 0;
                for (int i = 0; i < ptr->size(); i++) {
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
                for (int i = 0; i < ptr->size(); i++) {
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
                for (int i = 0; i < ptr->size() - 1; i++) {
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
                for (int i = 0; i < ptr->size(); i++) {
                    ptr->flip(state, i);
                    vec_d[i] = !vec_d[i];
                }

                THEN("Elments are flipped properly") {
                    CHECK(std::ranges::equal(ptr->view(state), vec_d));
                }
            }

            WHEN("We set all the elements") {
                auto set_count = 0;
                auto set_count_ground = 0;
                for (int i = 0; i < ptr->size(); i++) {
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
                for (int i = 0; i < ptr->size(); i++) {
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
                for (int i = 0; i < ptr->size() - 1; i++) {
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
}

TEST_CASE("IntegerNode") {
    auto graph = Graph();

    GIVEN("Double precision numbers, which may fall outside integer range or are not integral") {
        IntegerNode inode({1});

        THEN("The state is not deterministic") {
            CHECK(!inode.deterministic_state());
        }

        THEN("The function to check valid integers works") {
            CHECK(inode.max() == 2000000000);
            CHECK(inode.min() == 0);
            CHECK(inode.is_valid(inode.min() - 1) == false);
            CHECK(inode.is_valid(inode.max() + 1) == false);
            CHECK(inode.is_valid(10.5) == false);
            CHECK(inode.is_valid(inode.min()) == true);
            CHECK(inode.is_valid(inode.max()) == true);
            CHECK(inode.is_valid(10) == true);
        }
    }

    GIVEN("Integer node with custom range, the range works as expected") {
        IntegerNode inode({1}, -5, 10);

        THEN("The function to check valid integers works") {
            CHECK(inode.max() == 10);
            CHECK(inode.min() == -5);
            CHECK(inode.is_valid(inode.min() - 1) == false);
            CHECK(inode.is_valid(inode.max() + 1) == false);
            CHECK(inode.is_valid(5.5) == false);
            CHECK(inode.is_valid(inode.min()) == true);
            CHECK(inode.is_valid(inode.max()) == true);
            CHECK(inode.is_valid(5) == true);
        }
    }

    GIVEN("Integer with an upper bound specified, but no lower bound") {
        IntegerNode inode({1}, {}, 10);

        THEN("The lower bound takes the default we expect") {
            CHECK(inode.lower_bound() == IntegerNode::default_lower_bound);
            CHECK(inode.upper_bound() == 10);
        }
    }

    GIVEN("Integer with a lower bound specified, but no upper bound provided") {
        IntegerNode inode1({1}, 5);

        THEN("The lower bound takes the default we expect") {
            CHECK(inode1.lower_bound() == 5);
            CHECK(inode1.upper_bound() == IntegerNode::default_upper_bound);
        }
    }

    GIVEN("Integer with a lower bound specified, but with unspecified upper bound provided") {
        IntegerNode inode1({1}, 5, {});

        THEN("The lower bound takes the default we expect") {
            CHECK(inode1.lower_bound() == 5);
            CHECK(inode1.upper_bound() == IntegerNode::default_upper_bound);
        }
    }

    GIVEN("An Integer Node representing an 1d array of 10 elements with lower bound -10") {
        auto ptr = graph.emplace_node<IntegerNode>(std::initializer_list<ssize_t>{10}, -10);

        THEN("The shape is fixed") {
            CHECK(ptr->ndim() == 1);
            CHECK(ptr->size() == 10);
            CHECK_THAT(ptr->shape(), RangeEquals({10}));
            CHECK_THAT(ptr->strides(), RangeEquals({sizeof(double)}));
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
                CHECK(std::find_if(state_view.begin(), state_view.end(), [&](double i) {
                          return !ptr->is_valid(i);
                      }) == state_view.end());
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
                for (int i = 0; i < ptr->size(); i++) {
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
                for (int i = 0; i < ptr->size() - 1; i++) {
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

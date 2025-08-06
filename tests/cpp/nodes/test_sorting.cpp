// Copyright 2025 D-Wave
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

#include <catch2/benchmark/catch_benchmark_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "dwave-optimization/nodes/collections.hpp"
#include "dwave-optimization/nodes/constants.hpp"
#include "dwave-optimization/nodes/numbers.hpp"
#include "dwave-optimization/nodes/sorting.hpp"
#include "dwave-optimization/nodes/testing.hpp"

using Catch::Matchers::RangeEquals;

namespace dwave::optimization {

TEST_CASE("ArgSortNode") {
    auto graph = Graph();

    GIVEN("An integer constant node and an argsort node") {
        auto a_ptr = graph.emplace_node<ConstantNode>(std::vector{1.0, 5.0, 3.0, 2.0});

        auto argsort_ptr = graph.emplace_node<ArgSortNode>(a_ptr);

        graph.emplace_node<ArrayValidationNode>(argsort_ptr);

        CHECK(argsort_ptr->min() == 0.0);
        CHECK(argsort_ptr->max() == 3.0);

        WHEN("We initialize a state") {
            auto state = graph.initialize_state();

            THEN("The initial argsort state is correct") {
                CHECK_THAT(argsort_ptr->view(state), RangeEquals({0, 3, 2, 1}));
            }
        }
    }

    GIVEN("An integer node and an argsort node") {
        auto i_ptr = graph.emplace_node<IntegerNode>(5);

        auto argsort_ptr = graph.emplace_node<ArgSortNode>(i_ptr);

        graph.emplace_node<ArrayValidationNode>(argsort_ptr);

        WHEN("We initialize a state") {
            auto state = graph.empty_state();
            i_ptr->initialize_state(state, {1.0, 5.0, 3.0, 2.0, 7.0});
            graph.initialize_state(state);

            THEN("The initial argsort state is correct") {
                CHECK_THAT(argsort_ptr->view(state), RangeEquals({0, 3, 2, 1, 4}));
            }

            AND_WHEN("We make some changes to the integer node and propagate") {
                i_ptr->set_value(state, 1, 0.0);
                i_ptr->set_value(state, 2, 4.0);
                // i_ptr should now be [1.0, 0.0, 4.0, 2.0, 7.0]

                graph.propagate(state);

                THEN("The argsort state is correct") {
                    CHECK_THAT(argsort_ptr->view(state), RangeEquals({1, 0, 3, 2, 4}));
                }
            }
        }

        WHEN("We initialize a state where the integer node has multiple duplicate values") {
            auto state = graph.empty_state();
            i_ptr->initialize_state(state, {1.0, 5.0, 1.0, 2.0, 5.0});
            graph.initialize_state(state);

            THEN("The argsort returns the stable ordering") {
                CHECK_THAT(argsort_ptr->view(state), RangeEquals({0, 2, 3, 1, 4}));
            }

            AND_WHEN("We make some changes and propagate") {
                i_ptr->set_value(state, 3, 1.0);
                i_ptr->set_value(state, 1, 0.0);
                i_ptr->set_value(state, 4, 0.0);
                // i_ptr should now be [1.0, 0.0, 1.0, 1.0, 0.0]

                graph.propagate(state);

                THEN("The argsort returns the stable ordering") {
                    CHECK_THAT(argsort_ptr->view(state), RangeEquals({1, 4, 0, 2, 3}));
                }
            }
        }
    }

    GIVEN("A dynamic set node") {
        auto set_ptr = graph.emplace_node<SetNode>(10);
        auto argsort_ptr = graph.emplace_node<ArgSortNode>(set_ptr);
        graph.emplace_node<ArrayValidationNode>(argsort_ptr);

        CHECK(argsort_ptr->sizeinfo().array_ptr == set_ptr);

        WHEN("We initialize a state") {
            auto state = graph.initialize_state();

            THEN("The argsort's state is empty like the set's") {
                CHECK(argsort_ptr->size(state) == 0);
                REQUIRE(argsort_ptr->ndim() == 1);
                CHECK(argsort_ptr->shape(state)[0] == 0);
            }

            AND_WHEN("We grow the set node and propagate") {
                set_ptr->assign(state, std::vector<double>{5, 9, 2, 6});
                graph.propagate(state);

                THEN("The argsort's state is correct") {
                    CHECK_THAT(argsort_ptr->view(state), RangeEquals({2, 0, 3, 1}));
                }

                AND_WHEN("We commit, shrink the set node a propagate") {
                    graph.commit(state);

                    set_ptr->shrink(state);
                    set_ptr->shrink(state);
                    // set should be [5, 9]
                    graph.propagate(state);

                    THEN("The argsort's state is correct") {
                        CHECK_THAT(argsort_ptr->view(state), RangeEquals({0, 1}));
                    }
                }

                AND_WHEN("We revert and propagate again") {
                    graph.revert(state);

                    set_ptr->assign(state, std::vector<double>{4, 8, 7, 2});
                    graph.propagate(state);

                    THEN("The argsort's state is correct") {
                        CHECK_THAT(argsort_ptr->view(state), RangeEquals({3, 0, 2, 1}));
                    }
                }
            }
        }
    }
}

}  // namespace dwave::optimization

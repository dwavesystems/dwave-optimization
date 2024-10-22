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

#include <ranges>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "dwave-optimization/nodes/collections.hpp"
#include "dwave-optimization/nodes/constants.hpp"
#include "dwave-optimization/nodes/indexing.hpp"
#include "dwave-optimization/nodes/manipulation.hpp"
#include "dwave-optimization/nodes/mathematical.hpp"
#include "dwave-optimization/nodes/numbers.hpp"

namespace dwave::optimization {

TEST_CASE("ReshapeNode") {
    GIVEN("A 1d array encoding range(12)") {
        auto A = ConstantNode(std::vector{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});

        WHEN("It is reshaped to a (12,) array") {
            auto B = ReshapeNode(&A, {12});

            THEN("It has the shape/size/etc we expect") {
                CHECK(B.ndim() == 1);
                CHECK(std::ranges::equal(B.shape(), std::vector{12}));
            }
        }
    }

    GIVEN("Two paths through a DAG calculating the same thing") {
        auto graph = Graph();

        auto x_ptr = graph.emplace_node<BinaryNode>(std::vector<ssize_t>{9});
        auto A_ptr = graph.emplace_node<ConstantNode>(std::vector{1, 2, 3, 4, 5, 6, 7, 8, 9});

        // lhs path, (x * A).sum()
        auto lhs_ptr = graph.emplace_node<SumNode>(graph.emplace_node<MultiplyNode>(x_ptr, A_ptr));

        // rhs path, (x.reshape(3, 3) * A.reshape(3, 3)).sum()
        auto rhs_ptr = graph.emplace_node<SumNode>(graph.emplace_node<MultiplyNode>(
                graph.emplace_node<ReshapeNode>(x_ptr, std::vector<ssize_t>{3, 3}),
                graph.emplace_node<ReshapeNode>(A_ptr, std::vector<ssize_t>{3, 3})));

        WHEN("We do random moves") {
            auto rng = RngAdaptor(std::mt19937(42));

            auto state = graph.initialize_state();

            std::uniform_int_distribution<int> num_moves(1, 5);
            std::uniform_int_distribution<int> coin(0, 1);

            for (auto i = 0; i < 100; ++i) {
                // do a random number of default moves
                for (auto j = 0, n = num_moves(rng); j < n; ++j) {
                    x_ptr->default_move(state, rng);
                }

                graph.propose(state, {x_ptr},
                              [&rng, &coin](const Graph&, State&) { return coin(rng); });

                CHECK(std::ranges::equal(lhs_ptr->view(state), rhs_ptr->view(state)));
            }
        }
    }
}

TEST_CASE("SizeNode") {
    GIVEN("A 0d node") {
        auto C = ConstantNode(5);

        WHEN("We create a node accessing the length") {
            auto len = SizeNode(&C);

            THEN("The SizeNode is a scalar output") {
                CHECK(len.size() == 1);
                CHECK(len.ndim() == 0);
                CHECK(!len.dynamic());
            }

            THEN("The output is a integer and we already know the min/max") {
                CHECK(len.integral());
                CHECK(len.min() == C.size());
                CHECK(len.max() == C.size());
            }
        }
    }

    GIVEN("A 1D node with a fixed size") {
        auto C = ConstantNode(std::vector{0, 1, 2});

        WHEN("We create a node accessing the length") {
            auto len = SizeNode(&C);

            THEN("The SizeNode is a scalar output") {
                CHECK(len.size() == 1);
                CHECK(len.ndim() == 0);
                CHECK(!len.dynamic());
            }

            THEN("The output is a integer and we already know the min/max") {
                CHECK(len.integral());
                CHECK(len.min() == C.size());
                CHECK(len.max() == C.size());
            }
        }
    }

    GIVEN("A dynamic node") {
        auto S = SetNode(5, 2, 4);

        WHEN("We create a node accessing the length") {
            auto len = SizeNode(&S);

            THEN("The SizeNode is a scalar output") {
                CHECK(len.size() == 1);
                CHECK(len.ndim() == 0);
                CHECK(!len.dynamic());
            }

            THEN("The output is a integer and we already know the min/max") {
                CHECK(len.integral());
                CHECK(len.min() == 2);
                CHECK(len.max() == 4);
            }
        }

        AND_WHEN("We create a node indirectly accessing the length") {
            auto T = BasicIndexingNode(&S, Slice(0, -1));  // also dynamic, but indirect
            auto len = SizeNode(&T);

            THEN("The SizeNode is a scalar output") {
                CHECK(len.size() == 1);
                CHECK(len.ndim() == 0);
                CHECK(!len.dynamic());
            }

            THEN("The output is a integer and we already know the min/max") {
                CHECK(len.integral());

                // these should always be true
                CHECK(len.min() >= 0);
                CHECK(len.max() <= std::numeric_limits<ssize_t>::max());

                // these are an implementation detail and could change in the future
                CHECK(len.min() == 0);
                CHECK(len.max() == std::numeric_limits<ssize_t>::max());
            }
        }
    }

    GIVEN("A graph with a SetNode and a corresponding SizeNode, and a state") {
        auto graph = Graph();

        auto s_ptr = graph.emplace_node<SetNode>(5);
        auto len_ptr = graph.emplace_node<SizeNode>(s_ptr);

        auto state = graph.empty_state();
        s_ptr->initialize_state(state, {});  // default to empty
        graph.initialize_state(state);

        THEN("The state of the SizeNode is the same as the size of the SetNode") {
            CHECK(*(len_ptr->buff(state)) == s_ptr->size(state));
        }

        WHEN("We update the set and propagate") {
            s_ptr->grow(state);
            s_ptr->propagate(state);
            len_ptr->propagate(state);

            THEN("The state of the SizeNode is the same as the size of the SetNode") {
                CHECK(*(len_ptr->buff(state)) == s_ptr->size(state));
            }

            AND_WHEN("We commit") {
                s_ptr->commit(state);
                len_ptr->commit(state);

                THEN("The state of the SizeNode is the same as the size of the SetNode") {
                    CHECK(*(len_ptr->buff(state)) == s_ptr->size(state));
                }
            }

            AND_WHEN("We revert") {
                s_ptr->revert(state);
                len_ptr->revert(state);

                THEN("The state of the SizeNode is the same as the size of the SetNode") {
                    CHECK(*(len_ptr->buff(state)) == s_ptr->size(state));
                }
            }
        }
    }
}

}  // namespace dwave::optimization

// Copyright 2023 D-Wave Systems Inc.
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

#include "catch2/catch.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/nodes.hpp"

namespace dwave::optimization {

TEST_CASE("Test RngAdaptor") {
    GIVEN("A Mersenne Twister engine and an RngAdaptor constructed from one") {
        std::mt19937 mt_rng(42);
        RngAdaptor ra_rng(std::mt19937(42));

        THEN("The behave the same") {
            std::uniform_int_distribution<> d6(1, 6);
            for (int i = 0; i < 10; ++i) {
                CHECK(d6(ra_rng) == d6(mt_rng));
            }
        }
    }
}

TEST_CASE("Topological Sort", "[topological_sort]") {
    auto graph = Graph();

    WHEN("We add a mix of unconnected decision and non-decision variables") {
        auto x0_ptr = graph.emplace_node<ListNode>(5);
        auto x1_ptr = graph.emplace_node<ListNode>(5);
        auto c0_ptr = graph.emplace_node<ConstantNode>();
        auto x2_ptr = graph.emplace_node<ListNode>(5);
        auto c1_ptr = graph.emplace_node<ConstantNode>();
        auto c2_ptr = graph.emplace_node<ConstantNode>();
        auto x3_ptr = graph.emplace_node<ListNode>(5);

        THEN("The decisions are proactively topologically sorted") {
            CHECK(x0_ptr->topological_index() == 0);
            CHECK(x1_ptr->topological_index() == 1);
            CHECK(x2_ptr->topological_index() == 2);
            CHECK(x3_ptr->topological_index() == 3);
        }
        THEN("The non-decisions are unsorted") {
            CHECK(c0_ptr->topological_index() < 0);
            CHECK(c1_ptr->topological_index() < 0);
            CHECK(c2_ptr->topological_index() < 0);
        }

        AND_WHEN("we topologically sort the model") {
            graph.topological_sort();

            THEN("The decisions keep their topologically indices") {
                CHECK(x0_ptr->topological_index() == 0);
                CHECK(x1_ptr->topological_index() == 1);
                CHECK(x2_ptr->topological_index() == 2);
                CHECK(x3_ptr->topological_index() == 3);
            }
            THEN("The non-decisions are sorted") {
                CHECK(c0_ptr->topological_index() >= 4);
                CHECK(c1_ptr->topological_index() >= 4);
                CHECK(c2_ptr->topological_index() >= 4);
                CHECK(c0_ptr->topological_index() < 7);
                CHECK(c1_ptr->topological_index() < 7);
                CHECK(c2_ptr->topological_index() < 7);
                CHECK(c0_ptr->topological_index() != c1_ptr->topological_index());
                CHECK(c0_ptr->topological_index() != c2_ptr->topological_index());
                CHECK(c1_ptr->topological_index() != c2_ptr->topological_index());
            }

            AND_WHEN("We reset the topological sort") {
                graph.reset_topological_sort();

                THEN("The decisions keep their topologically indices") {
                    CHECK(x0_ptr->topological_index() == 0);
                    CHECK(x1_ptr->topological_index() == 1);
                    CHECK(x2_ptr->topological_index() == 2);
                    CHECK(x3_ptr->topological_index() == 3);
                }
                THEN("The non-decisions are unsorted") {
                    CHECK(c0_ptr->topological_index() < 0);
                    CHECK(c1_ptr->topological_index() < 0);
                    CHECK(c2_ptr->topological_index() < 0);
                }
            }
        }

        AND_WHEN("We reset the topological sort before sorting") {
            graph.reset_topological_sort();

            THEN("The decisions keep their topologically indices") {
                CHECK(x0_ptr->topological_index() == 0);
                CHECK(x1_ptr->topological_index() == 1);
                CHECK(x2_ptr->topological_index() == 2);
                CHECK(x3_ptr->topological_index() == 3);
            }
            THEN("The non-decisions are unsorted") {
                CHECK(c0_ptr->topological_index() < 0);
                CHECK(c1_ptr->topological_index() < 0);
                CHECK(c2_ptr->topological_index() < 0);
            }
        }
    }

    WHEN("We add a disjoint lists node and successors") {
        auto disjoint_lists = graph.emplace_node<DisjointListsNode>(10, 3);

        auto disjoint_list0 = graph.emplace_node<DisjointListNode>(disjoint_lists);
        auto disjoint_list1 = graph.emplace_node<DisjointListNode>(disjoint_lists);
        auto disjoint_list2 = graph.emplace_node<DisjointListNode>(disjoint_lists);

        AND_WHEN("We topologically sort the model") {
            graph.topological_sort();

            THEN("The order remains stable wrt successor order") {
                CHECK(disjoint_list0->topological_index() < disjoint_list1->topological_index());
                CHECK(disjoint_list1->topological_index() < disjoint_list2->topological_index());
            }
        }

        AND_WHEN("We add some more successors, and topologically sort the model") {
            graph.emplace_node<SumNode>(disjoint_list1);
            graph.emplace_node<SumNode>(disjoint_list0);

            graph.topological_sort();

            THEN("The order remains stable wrt successor order") {
                CHECK(disjoint_list0->topological_index() < disjoint_list1->topological_index());
                CHECK(disjoint_list1->topological_index() < disjoint_list2->topological_index());
            }
        }
    }

    WHEN("We add a disjoint bitsets node and successors") {
        auto disjoint_bitsets = graph.emplace_node<DisjointBitSetsNode>(10, 3);

        auto disjoint_bitset0 = graph.emplace_node<DisjointBitSetNode>(disjoint_bitsets);
        auto disjoint_bitset1 = graph.emplace_node<DisjointBitSetNode>(disjoint_bitsets);
        auto disjoint_bitset2 = graph.emplace_node<DisjointBitSetNode>(disjoint_bitsets);

        AND_WHEN("We topologically sort the model") {
            graph.topological_sort();

            THEN("The order remains stable wrt successor order") {
                CHECK(disjoint_bitset0->topological_index() < disjoint_bitset1->topological_index());
                CHECK(disjoint_bitset1->topological_index() < disjoint_bitset2->topological_index());
            }
        }

        AND_WHEN("We add some more successors, and topologically sort the model") {
            graph.emplace_node<SumNode>(disjoint_bitset1);
            graph.emplace_node<SumNode>(disjoint_bitset0);

            graph.topological_sort();

            THEN("The order remains stable wrt successor order") {
                CHECK(disjoint_bitset0->topological_index() < disjoint_bitset1->topological_index());
                CHECK(disjoint_bitset1->topological_index() < disjoint_bitset2->topological_index());
            }
        }
    }

    // todo: test the actual sort function. This is tested implicitly by
    // a lot of the examples but once we have more node types we should
    // test edge cases.
}

TEST_CASE("Graph::objective()") {
    GIVEN("A graph") {
        auto graph = Graph();

        THEN("The objective defaults to nullptr") {
            CHECK(graph.objective() == nullptr);
            CHECK(static_cast<const Graph&>(graph).objective() == nullptr);
        }

        WHEN("We add an objective") {
            ArrayNode* x_ptr = graph.emplace_node<BinaryNode>(std::vector<ssize_t>{});
            graph.set_objective(x_ptr);

            THEN("The objective is set") {
                CHECK(graph.objective() == x_ptr);
                CHECK(static_cast<const Graph&>(graph).objective() == x_ptr);
            }
        }
    }
}

}  // namespace dwave::optimization

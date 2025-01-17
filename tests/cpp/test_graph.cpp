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

#include "catch2/catch_test_macros.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/nodes.hpp"

namespace dwave::optimization {

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
                CHECK(disjoint_bitset0->topological_index() <
                      disjoint_bitset1->topological_index());
                CHECK(disjoint_bitset1->topological_index() <
                      disjoint_bitset2->topological_index());
            }
        }

        AND_WHEN("We add some more successors, and topologically sort the model") {
            graph.emplace_node<SumNode>(disjoint_bitset1);
            graph.emplace_node<SumNode>(disjoint_bitset0);

            graph.topological_sort();

            THEN("The order remains stable wrt successor order") {
                CHECK(disjoint_bitset0->topological_index() <
                      disjoint_bitset1->topological_index());
                CHECK(disjoint_bitset1->topological_index() <
                      disjoint_bitset2->topological_index());
            }
        }
    }

    // todo: test the actual sort function. This is tested implicitly by
    // a lot of the examples but once we have more node types we should
    // test edge cases.
}

TEST_CASE("Graph constructors, assignment operators, and swapping") {
    static_assert(std::is_nothrow_default_constructible<Graph>::value);
    static_assert(!std::is_copy_constructible<Graph>::value);
    static_assert(!std::is_copy_assignable<Graph>::value);
    static_assert(std::is_nothrow_move_constructible<Graph>::value);
    static_assert(std::is_nothrow_move_assignable<Graph>::value);
    static_assert(std::is_swappable<Graph>::value);

    GIVEN("A graph with a few nodes in it") {
        auto graph = Graph();
        auto a_ptr = graph.emplace_node<ConstantNode>(1.5);
        auto b_ptr = graph.emplace_node<ConstantNode>(2);
        auto c_ptr = graph.emplace_node<AddNode>(a_ptr, b_ptr);
        graph.set_objective(c_ptr);

        WHEN("We construct another graph using the move constructor") {
            auto other = Graph(std::move(graph));

            THEN("The nodes have all be moved over") {
                // these tests are far from complete, but we're using default
                // constructors/operators so this is intended to be a sanity
                // check
                CHECK(other.nodes().size() == 3);
                CHECK(graph.nodes().size() == 0);
            }
        }

        WHEN("We construct another graph using the move assignment operator") {
            auto other = Graph();
            other = std::move(graph);

            THEN("The nodes have all be moved over") {
                // these tests are far from complete, but we're using default
                // constructors/operators so this is intended to be a sanity
                // check
                CHECK(other.nodes().size() == 3);
                CHECK(graph.nodes().size() == 0);
            }
        }

        AND_GIVEN("A different graph with at least one node") {
            auto other = Graph();
            auto d_ptr = other.emplace_node<BinaryNode>(std::initializer_list<ssize_t>{1});
            other.add_constraint(d_ptr);

            WHEN("We swap the graphs") {
                std::swap(graph, other);

                THEN("Their content have been swapped as expected") {
                    // these tests are far from complete, but we're using default
                    // constructors/operators so this is intended to be a sanity
                    // check
                    CHECK(other.nodes().size() == 3);
                    CHECK(other.constraints().size() == 0);

                    CHECK(graph.nodes().size() == 1);
                    CHECK(graph.constraints().size() == 1);
                }
            }
        }
    }
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

TEST_CASE("Graph::remove_unused_nodes()") {
    GIVEN("A single integer variable") {
        auto graph = Graph();

        auto i_ptr = graph.emplace_node<IntegerNode>();

        THEN("remove_unused_nodes() does nothing") {
            ssize_t num_removed = graph.remove_unused_nodes();
            CHECK(num_removed == 0);
            CHECK(graph.num_nodes() == 1);
            CHECK(i_ptr->topological_index() == 0);
        }

        WHEN("We add a successor that's not used in a constraint") {
            graph.emplace_node<AbsoluteNode>(i_ptr);

            THEN("remove_unused_nodes() removes it") {
                ssize_t num_removed = graph.remove_unused_nodes();
                CHECK(num_removed == 1);
                CHECK(graph.num_nodes() == 1);
                CHECK(i_ptr->topological_index() == 0);
                CHECK(i_ptr->successors().size() == 0);
            }
        }

        WHEN("We add two successors that are not used in a constraint") {
            graph.emplace_node<LogicalNode>(graph.emplace_node<AbsoluteNode>(i_ptr));

            THEN("remove_unused_nodes() removes them") {
                ssize_t num_removed = graph.remove_unused_nodes();
                CHECK(num_removed == 2);
                CHECK(graph.num_nodes() == 1);
                CHECK(i_ptr->topological_index() == 0);
                CHECK(i_ptr->successors().size() == 0);
            }
        }

        WHEN("We add two successors and use them in a constraint") {
            graph.set_objective(
                    graph.emplace_node<LogicalNode>(graph.emplace_node<AbsoluteNode>(i_ptr)));

            THEN("remove_unused_nodes() doesn't remove them") {
                ssize_t num_removed = graph.remove_unused_nodes();
                CHECK(num_removed == 0);
                CHECK(graph.num_nodes() == 3);
                CHECK(i_ptr->topological_index() == 0);
                CHECK(i_ptr->successors().size() == 1);
            }
        }

        WHEN("We add a mix of nodes that are used in constraints and not used anywhere") {
            // i -> a -> c -> d
            // i -> b />    \> e -> objective

            auto a_ptr = graph.emplace_node<AbsoluteNode>(i_ptr);
            auto b_ptr = graph.emplace_node<AbsoluteNode>(i_ptr);
            auto c_ptr = graph.emplace_node<AddNode>(a_ptr, b_ptr);
            auto d_ptr = graph.emplace_node<LogicalNode>(c_ptr);
            auto e_ptr = graph.emplace_node<LogicalNode>(c_ptr);

            // give d a listener
            auto d_expired = d_ptr->expired_ptr();

            graph.set_objective(e_ptr);

            THEN("remove_unused_nodes(ignore_listeners=True) removes only the ones we want") {
                ssize_t num_removed = graph.remove_unused_nodes(true);
                CHECK(num_removed == 1);
                CHECK(*d_expired);  // it was d that was removed
            }

            THEN("remove_unused_nodes(ignore_listeners=False) removes none") {
                ssize_t num_removed = graph.remove_unused_nodes();
                CHECK(num_removed == 0);
                CHECK(!*d_expired);  // d wasn't removed because it has a listener
            }
        }
    }
}

}  // namespace dwave::optimization

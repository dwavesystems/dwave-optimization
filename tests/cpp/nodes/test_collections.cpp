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

#include <set>
#include <unordered_set>

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_all.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/nodes/collections.hpp"
#include "dwave-optimization/nodes/testing.hpp"

using Catch::Matchers::RangeEquals;

namespace dwave::optimization {

TEST_CASE("DisjointBitSetsNode") {
    auto graph = Graph();

    GIVEN("A disjoint-bit-set node representing 5 elements partitioned into 3 sets") {
        const ssize_t primary_set_size = 5;
        const ssize_t num_disjoint_sets = 3;
        auto ptr = graph.emplace_node<DisjointBitSetsNode>(primary_set_size, num_disjoint_sets);

        THEN("We already know a lot about the size etc") {
            CHECK(ptr->primary_set_size() == 5);
            CHECK(ptr->num_disjoint_sets() == 3);
        }

        THEN("The state is not deterministic") {
            CHECK(!ptr->deterministic_state());
        }

        WHEN("We add three array output successors") {
            std::vector<DisjointBitSetNode*> sets;
            for (int i = 0; i < 3; ++i) {
                sets.push_back(graph.emplace_node<DisjointBitSetNode>(ptr));

                graph.emplace_node<ArrayValidationNode>(sets.at(i));
            }

            THEN("We shouldn't be able to add any more successors") {
                CHECK_THROWS(graph.emplace_node<DisjointBitSetNode>(ptr));
            }

            THEN("The size should be static and equal to # of elements") {
                for (int i = 0; i < 3; ++i) {
                    CHECK(sets[i]->size() == 5);
                    CHECK(std::ranges::equal(sets[i]->shape(), std::vector{5}));
                }
            }

            AND_WHEN("We default-initialize the state") {
                auto state = graph.initialize_state();

                THEN("The output nodes contain the whole set") {
                    std::vector<int> bit_set(5, 0);

                    for (auto const& set : sets) {
                        REQUIRE(set->size(state) == 5);
                        for (ssize_t i = 0; i < set->size(state); ++i) {
                            REQUIRE((set->view(state)[i] == 0 || set->view(state)[i] == 1));
                            bit_set[i] += set->view(state)[i];
                        }
                    }

                    CHECK_THAT(bit_set, RangeEquals({1, 1, 1, 1, 1}));
                }

                AND_WHEN("We copy the state") {
                    state[ptr->topological_index()] = state[ptr->topological_index()]->copy();

                    THEN("The output nodes contain the whole set") {
                        std::vector<int> bit_set(5, 0);

                        for (auto const& set : sets) {
                            REQUIRE(set->size(state) == 5);
                            for (ssize_t i = 0; i < set->size(state); ++i) {
                                REQUIRE((set->view(state)[i] == 0 || set->view(state)[i] == 1));
                                bit_set[i] += set->view(state)[i];
                            }
                        }

                        CHECK_THAT(bit_set, RangeEquals({1, 1, 1, 1, 1}));
                    }
                }

                AND_WHEN("We then mutate the node and propagate") {
                    ptr->swap_between_sets(state, 0, 1, 2);  // {0 1 3 4} {2} {}
                    ptr->swap_between_sets(state, 0, 2, 4);  // {0 1 3} {2} {4}
                    ptr->swap_between_sets(state, 0, 1, 0);  // {1 3} {0 2} {4}
                    ptr->swap_between_sets(state, 1, 2, 2);  // {1 3} {0} {2 4}

                    graph.propagate(state, graph.descendants(state, {ptr}));

                    THEN("The node's state reflects the relevant changes") {
                        CHECK(std::ranges::equal(sets[0]->view(state), std::vector{0, 1, 0, 1, 0}));
                        CHECK(std::ranges::equal(sets[1]->view(state), std::vector{1, 0, 0, 0, 0}));
                        CHECK(std::ranges::equal(sets[2]->view(state), std::vector{0, 0, 1, 0, 1}));
                    }

                    THEN("The successor nodes' size_diffs are correct") {
                        CHECK(sets[0]->size_diff(state) == 0);
                        CHECK(sets[1]->size_diff(state) == 0);
                        CHECK(sets[2]->size_diff(state) == 0);
                    }

                    AND_WHEN("We commit") {
                        graph.commit(state, graph.descendants(state, {ptr}));

                        THEN("The changes persist") {
                            CHECK(std::ranges::equal(sets[0]->view(state),
                                                     std::vector{0, 1, 0, 1, 0}));
                            CHECK(std::ranges::equal(sets[1]->view(state),
                                                     std::vector{1, 0, 0, 0, 0}));
                            CHECK(std::ranges::equal(sets[2]->view(state),
                                                     std::vector{0, 0, 1, 0, 1}));
                        }
                    }

                    AND_WHEN("We revert") {
                        graph.revert(state, graph.descendants(state, {ptr}));

                        THEN("The changes are undone") {
                            CHECK(std::ranges::equal(sets[0]->view(state),
                                                     std::vector{1, 1, 1, 1, 1}));
                            CHECK(sets[0]->diff(state).size() == 0);
                            CHECK(std::ranges::equal(sets[1]->view(state),
                                                     std::vector{0, 0, 0, 0, 0}));
                            CHECK(sets[1]->diff(state).size() == 0);
                            CHECK(std::ranges::equal(sets[2]->view(state),
                                                     std::vector{0, 0, 0, 0, 0}));
                            CHECK(sets[2]->diff(state).size() == 0);
                        }
                    }
                }
            }

            AND_WHEN("We initialize the state with given partitions") {
                auto state = graph.empty_state();
                ptr->initialize_state(state, {{0, 0, 0, 0, 1}, {1, 0, 1, 0, 0}, {0, 1, 0, 1, 0}});
                graph.initialize_state(state);

                THEN("We can read out the expected state from the disjoint set nodes") {
                    CHECK(std::ranges::equal(sets[0]->view(state), std::vector{0, 0, 0, 0, 1}));
                    CHECK(std::ranges::equal(sets[1]->view(state), std::vector{1, 0, 1, 0, 0}));
                    CHECK(std::ranges::equal(sets[2]->view(state), std::vector{0, 1, 0, 1, 0}));
                }
            }

            AND_WHEN("We initialize an empty state") {
                auto state = graph.empty_state();

                THEN("We get an error when trying to initialize invalid partitions") {
                    CHECK_THROWS(ptr->initialize_state(
                            state, {{0, 0, 0, 0, 2}, {1, 0, 1, 0, 0}, {0, 1, 0, 1, 0}}));
                    CHECK_THROWS(ptr->initialize_state(
                            state, {{0, 0, 0, 0, 0}, {1, 0, 1, 0, 0}, {0, 1, 0, 1, 0}}));
                    CHECK_THROWS(ptr->initialize_state(
                            state, {{0, 0, 0, 1, 1}, {1, 0, 1, 0, 0}, {0, 1, 0, 1, 0}}));
                }
            }
        }
    }
}

TEST_CASE("DisjointListsNode") {
    auto graph = Graph();

    GIVEN("A disjoint-list node representing 5 elements partitioned into 3 lists") {
        const ssize_t primary_set_size = 5;
        const ssize_t num_disjoint_lists = 3;
        auto ptr = graph.emplace_node<DisjointListsNode>(primary_set_size, num_disjoint_lists);

        THEN("The state is not deterministic") {
            CHECK(!ptr->deterministic_state());
        }

        THEN("We already know a lot about the size etc") {
            CHECK(ptr->primary_set_size() == 5);
            CHECK(ptr->num_disjoint_lists() == 3);
        }

        WHEN("We add three array output successors") {
            std::vector<DisjointListNode*> lists;
            for (int i = 0; i < 3; ++i) {
                lists.push_back(graph.emplace_node<DisjointListNode>(ptr));

                graph.emplace_node<ArrayValidationNode>(lists.at(i));
            }

            THEN("We shouldn't be able to add any more successors") {
                CHECK_THROWS(graph.emplace_node<DisjointListNode>(ptr));
            }

            THEN("The shape of the successors are as expected") {
                for (const DisjointListNode* ptr : lists) {
                    CHECK(ptr->dynamic());

                    auto sizeinfo = ptr->sizeinfo();
                    CHECK(sizeinfo.array_ptr == ptr);
                    CHECK(sizeinfo.multiplier == 1);
                    CHECK(sizeinfo.offset == 0);
                    CHECK(sizeinfo.min == 0);
                    CHECK(sizeinfo.max == primary_set_size);
                }
            }

            std::vector<ArrayValidationNode*> validation_nodes;
            for (const auto& list : lists) {
                validation_nodes.push_back(graph.emplace_node<ArrayValidationNode>(list));
            }

            AND_WHEN("We default-initialize the state") {
                auto state = graph.initialize_state();

                THEN("The output nodes contain the whole set") {
                    std::set<double> set;
                    int num_elements = 0;

                    for (auto const& list : lists) {
                        for (auto const& el : list->view(state)) {
                            set.insert(el);
                        }
                        num_elements += list->view(state).size();
                        CHECK_THAT(list->shape(state), RangeEquals({list->view(state).size()}));
                    }

                    CHECK(num_elements == 5);
                    CHECK(set == std::set<double>{0, 1, 2, 3, 4});
                }

                THEN("The node's state defaults to range(5)") {
                    CHECK(std::ranges::equal(lists[0]->view(state), std::vector{0, 1, 2, 3, 4}));
                }

                AND_WHEN("We then mutate the node and propagate") {
                    ptr->swap_in_list(state, 0, 1, 3);    // [0 3 2 1 4] [] []
                    ptr->rotate_in_list(state, 0, 4, 1);  // [0 2 1 4 3] [] []
                    ptr->rotate_in_list(state, 0, 4, 1);  // [0 1 4 3 2] [] []
                    ptr->pop_to_list(state, 0, 1, 2, 0);  // [0 4 3 2] [] [1]
                    ptr->pop_to_list(state, 0, 2, 2, 1);  // [0 4 2] [] [1 3]
                    ptr->swap_in_list(state, 2, 1, 0);    // [0 4 2] [] [3 1]
                    graph.propagate(state, graph.descendants(state, {ptr}));

                    THEN("The node's state reflects the relevant changes") {
                        CHECK(std::ranges::equal(lists[0]->view(state), std::vector{0, 4, 2}));
                        CHECK(lists[1]->shape(state)[0] == 0);
                        CHECK(std::ranges::equal(lists[2]->view(state), std::vector{3, 1}));
                    }

                    THEN("The successor nodes' size_diffs are correct") {
                        CHECK(lists[0]->size_diff(state) == -2);
                        CHECK(lists[1]->size_diff(state) == 0);
                        CHECK(lists[2]->size_diff(state) == 2);
                    }

                    AND_WHEN("We commit") {
                        graph.commit(state, graph.descendants(state, {ptr}));

                        THEN("The changes persist") {
                            CHECK(std::ranges::equal(lists[0]->view(state), std::vector{0, 4, 2}));
                            CHECK(lists[1]->shape(state)[0] == 0);
                            CHECK(std::ranges::equal(lists[2]->view(state), std::vector{3, 1}));
                        }
                    }

                    AND_WHEN("We revert") {
                        graph.revert(state, graph.descendants(state, {ptr}));

                        THEN("The changes are undone") {
                            CHECK(std::ranges::equal(lists[0]->view(state),
                                                     std::vector{0, 1, 2, 3, 4}));
                            CHECK(lists[0]->diff(state).size() == 0);
                            CHECK(lists[1]->shape(state)[0] == 0);
                            CHECK(lists[2]->shape(state)[0] == 0);
                        }
                    }
                }

                SECTION("set_state") {
                    AND_WHEN("Set the full state of the first and third lists") {
                        ptr->set_state(state, 0, {{0, 3, 1}});
                        ptr->set_state(state, 2, {{4, 2}});

                        THEN("The state is correct") {
                            CHECK(std::ranges::equal(lists[0]->view(state), std::vector{0, 3, 1}));
                            CHECK(std::ranges::equal(lists[2]->view(state), std::vector{4, 2}));
                            graph.propagate(state, graph.descendants(state, {ptr}));
                        }
                    }
                }
            }

            AND_WHEN("We initialize the state with given partitions") {
                auto state = graph.empty_state();
                ptr->initialize_state(state, {{4}, {2, 0}, {1, 3}});
                graph.initialize_state(state);

                THEN("We can read out the expected state from the disjoint list nodes") {
                    CHECK(std::ranges::equal(lists[0]->view(state), std::vector{4}));
                    CHECK(std::ranges::equal(lists[1]->view(state), std::vector{2, 0}));
                    CHECK(std::ranges::equal(lists[2]->view(state), std::vector{1, 3}));
                }
            }

            THEN("We get an error when trying to initialize invalid partitions") {
                auto state = graph.empty_state();

                CHECK_THROWS(ptr->initialize_state(state, {{4}, {2}, {1, 3}}));
                CHECK_THROWS(ptr->initialize_state(state, {{0, 4}, {0, 2}, {1, 3}}));
                CHECK_THROWS(ptr->initialize_state(state, {{0, 4, 5}, {0, 2}, {1, 3}}));
                CHECK_THROWS(ptr->initialize_state(state, {{0}, {2}, {1, 3}}));
            }
        }
    }
}

TEST_CASE("ListNode") {
    auto graph = Graph();

    GIVEN("A list node representing permutations of range(5)") {
        const int num_elements = 5;
        auto ptr = graph.emplace_node<ListNode>(num_elements);

        THEN("The state is not deterministic") {
            CHECK(!ptr->deterministic_state());
        }

        THEN("We already know a lot about the size etc") {
            CHECK(ptr->size() == 5);
            CHECK_THAT(ptr->shape(), RangeEquals({5}));
            CHECK_THAT(ptr->strides(), RangeEquals({sizeof(double)}));

            CHECK(ptr->sizeinfo() == 5);
        }

        THEN("The ListNode broadcasts that it contains exactly 5 integers") {
            CHECK(ptr->min() == 0);
            CHECK(ptr->max() == num_elements - 1);
            CHECK(ptr->integral());
        }

        WHEN("We default-initialize the state") {
            auto state = graph.initialize_state();

            THEN("The node's state defaults to range(5)") {
                CHECK_THAT(ptr->view(state), RangeEquals({0, 1, 2, 3, 4}));
                CHECK(ptr->size(state) == 5);
                CHECK_THAT(ptr->shape(state), RangeEquals({5}));
            }

            AND_WHEN("We then mutate the node and propagate") {
                ptr->exchange(state, 1, 3);  // [0 3 2 1 4]
                ptr->exchange(state, 1, 2);  // [0 2 3 1 4]

                // As a technical detail, the state is eagerly updated, but in general
                // one has to propagate
                graph.propagate(state, graph.descendants(state, {ptr}));

                THEN("The node's state reflects the relevant changes") {
                    CHECK_THAT(ptr->view(state), RangeEquals({0, 2, 3, 1, 4}));

                    // todo: check diff()
                }

                AND_WHEN("We commit") {
                    graph.commit(state, graph.descendants(state, {ptr}));

                    THEN("The changes persist") {
                        CHECK_THAT(ptr->view(state), RangeEquals({0, 2, 3, 1, 4}));
                        CHECK(ptr->diff(state).size() == 0);
                    }
                }

                AND_WHEN("We revert") {
                    graph.revert(state, graph.descendants(state, {ptr}));

                    THEN("The changes are undone") {
                        CHECK_THAT(ptr->view(state), RangeEquals({0, 1, 2, 3, 4}));
                        CHECK(ptr->diff(state).size() == 0);
                    }
                }
            }
        }
    }
}

TEST_CASE("SetNode") {
    auto graph = Graph();

    GIVEN("A SetNode representing subsets of 10 elements") {
        const int num_elements = 10;

        auto ptr = graph.emplace_node<SetNode>(num_elements);

        graph.emplace_node<ArrayValidationNode>(ptr);

        THEN("The state is not deterministic") {
            CHECK(!ptr->deterministic_state());
        }

        THEN("The shape is dynamic") {
            CHECK(ptr->ndim() == 1);
            CHECK(ptr->size() == Array::DYNAMIC_SIZE);
            CHECK_THAT(ptr->shape(), RangeEquals({Array::DYNAMIC_SIZE}));
            CHECK_THAT(ptr->strides(), RangeEquals({sizeof(double)}));

            auto sizeinfo = ptr->sizeinfo();
            CHECK(sizeinfo.min == 0);
            CHECK(sizeinfo.max == 10);
            CHECK(sizeinfo.array_ptr == ptr);
            CHECK(sizeinfo.multiplier == 1);
            CHECK(sizeinfo.offset == 0);
        }

        WHEN("We create a state using the default value") {
            auto state = graph.initialize_state();

            THEN("We can read the state of the set") {
                CHECK(ptr->size(state) == 0);
                CHECK_THAT(ptr->shape(state), RangeEquals({0}));
                CHECK_THAT(ptr->view(state), RangeEquals(std::vector<double>{}));
            }

            AND_WHEN("We grow by one") {
                ptr->grow(state);
                graph.propagate(state, graph.descendants(state, {ptr}));

                THEN("The diff has been updated") {
                    // the specific value added is implementation-dependant
                    CHECK_THAT(ptr->view(state), RangeEquals({0}));

                    CHECK(ptr->size_diff(state) == 1);
                }

                AND_WHEN("We commit") {
                    graph.commit(state, graph.descendants(state, {ptr}));

                    THEN("The values are updated but the diff is cleared") {
                        CHECK_THAT(ptr->view(state), RangeEquals({0}));
                        CHECK(ptr->size_diff(state) == 0);
                        CHECK(ptr->diff(state).size() == 0);
                    }
                }

                AND_WHEN("We revert") {
                    graph.revert(state, graph.descendants(state, {ptr}));

                    THEN("We're back where we started") {
                        CHECK_THAT(ptr->view(state), RangeEquals(std::vector<double>{}));
                        CHECK(ptr->size_diff(state) == 0);
                        CHECK(ptr->diff(state).size() == 0);
                    }
                }
            }
        }

        WHEN("We create a state using a specified value") {
            auto state = graph.empty_state();
            ptr->initialize_state(state, {2, 0, 3});
            graph.initialize_state(state);

            THEN("We can read the state") {
                CHECK(ptr->size(state) == 3);
                CHECK_THAT(ptr->view(state), RangeEquals({2, 0, 3}));
            }

            AND_WHEN("We shrink by one") {
                ptr->shrink(state);
                graph.propagate(state, graph.descendants(state, {ptr}));

                THEN("The diff has been updated") {
                    // the specific value added is implementation-dependant
                    CHECK_THAT(ptr->view(state), RangeEquals({2, 0}));

                    CHECK(ptr->size_diff(state) == -1);
                }

                AND_WHEN("We commit") {
                    graph.commit(state, graph.descendants(state, {ptr}));

                    THEN("The values are updated but the diff is cleared") {
                        CHECK_THAT(ptr->view(state), RangeEquals({2, 0}));
                        CHECK(ptr->size_diff(state) == 0);
                        CHECK(ptr->diff(state).size() == 0);
                    }
                }

                AND_WHEN("We revert") {
                    graph.revert(state, graph.descendants(state, {ptr}));

                    THEN("We're back where we started") {
                        CHECK_THAT(ptr->view(state), RangeEquals({2, 0, 3}));
                        CHECK(ptr->size_diff(state) == 0);
                        CHECK(ptr->diff(state).size() == 0);
                    }
                }
            }
        }
    }
}

}  // namespace dwave::optimization

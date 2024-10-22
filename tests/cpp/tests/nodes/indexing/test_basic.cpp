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

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/nodes/collections.hpp"
#include "dwave-optimization/nodes/constants.hpp"
#include "dwave-optimization/nodes/indexing.hpp"
#include "dwave-optimization/nodes/mathematical.hpp"
#include "dwave-optimization/nodes/numbers.hpp"
#include "dwave-optimization/nodes/testing.hpp"
#include "../../utils.hpp"

namespace dwave::optimization {

TEST_CASE("BasicIndexingNode") {
    SECTION("SizeInfo") {
        SECTION("SetNode(10)[:]") {
            auto set = SetNode(10);

            auto slice = BasicIndexingNode(&set, Slice());

            auto sizeinfo = slice.sizeinfo();
            CHECK(sizeinfo.array_ptr == &set);
            CHECK(sizeinfo.multiplier == 1);
            CHECK(sizeinfo.offset == 0);
            CHECK(!sizeinfo.min.has_value());
            CHECK(!sizeinfo.max.has_value());

            // now substitute so we can see what we did to the array's size
            sizeinfo = sizeinfo.substitute();
            CHECK(sizeinfo.array_ptr == &set);
            CHECK(sizeinfo.multiplier == 1);
            CHECK(sizeinfo.offset == 0);
            CHECK(sizeinfo.min == 0);
            CHECK(sizeinfo.max == 10);
        }

        SECTION("SetNode(10)[::2]") {
            auto set = SetNode(10);

            auto slice = BasicIndexingNode(&set, Slice(std::nullopt, std::nullopt, 2));

            auto sizeinfo = slice.sizeinfo();
            CHECK(sizeinfo.array_ptr == &set);
            CHECK(sizeinfo.multiplier == fraction(1, 2));
            CHECK(sizeinfo.offset == 0);
            CHECK(!sizeinfo.min.has_value());
            CHECK(!sizeinfo.max.has_value());

            // now substitute so we can see what we did to the array's size
            sizeinfo = sizeinfo.substitute();
            CHECK(sizeinfo.array_ptr == &set);
            CHECK(sizeinfo.multiplier == fraction(1, 2));
            CHECK(sizeinfo.offset == 0);
            CHECK(sizeinfo.min == 0);
            CHECK(sizeinfo.max == 5);
        }

        SECTION("SetNode(10)[1:]") {
            auto set = SetNode(10);

            auto slice = BasicIndexingNode(&set, Slice(1, std::nullopt));

            auto sizeinfo = slice.sizeinfo();
            CHECK(sizeinfo.array_ptr == &set);
            CHECK(sizeinfo.multiplier == 1);
            CHECK(sizeinfo.offset == -1);
            CHECK(!sizeinfo.min.has_value());
            CHECK(!sizeinfo.max.has_value());

            // now substitute so we can see what we did to the array's size
            sizeinfo = sizeinfo.substitute();
            CHECK(sizeinfo.array_ptr == &set);
            CHECK(sizeinfo.multiplier == 1);
            CHECK(sizeinfo.offset == -1);
            CHECK(sizeinfo.min == 0);
            CHECK(sizeinfo.max == 9);
        }

        SECTION("SetNode(10)[11:]") {
            // dev note: this is an interesting case because clearly we have
            // enough information at construction time to determine that the
            // resulting array will have a fixed size of exactly 0. We should
            // think about having the constructor check and potentially raise
            // an error or set itself to not be dynamic.
            auto set = SetNode(10);

            auto slice = BasicIndexingNode(&set, Slice(11, std::nullopt));

            auto sizeinfo = slice.sizeinfo();
            CHECK(sizeinfo.array_ptr == &set);
            CHECK(sizeinfo.multiplier == 1);
            CHECK(sizeinfo.offset == -11);
            CHECK(!sizeinfo.min.has_value());
            CHECK(!sizeinfo.max.has_value());

            // now substitute so we can see what we did to the array's size
            // in this case, it's always 0!
            sizeinfo = sizeinfo.substitute();
            CHECK(sizeinfo == 0);
        }

        SECTION("SetNode(10)[:1]") {
            auto set = SetNode(10);

            auto slice = BasicIndexingNode(&set, Slice(1));

            auto sizeinfo = slice.sizeinfo();
            CHECK(sizeinfo.array_ptr == &set);
            CHECK(sizeinfo.multiplier == 1);
            CHECK(sizeinfo.offset == 0);
            CHECK(!sizeinfo.min.has_value());
            CHECK(sizeinfo.max == 1);

            // now substitute so we can see what we did to the array's size
            sizeinfo = sizeinfo.substitute();
            CHECK(sizeinfo.array_ptr == &set);
            CHECK(sizeinfo.multiplier == 1);
            CHECK(sizeinfo.offset == 0);
            CHECK(sizeinfo.min == 0);
            CHECK(sizeinfo.max == 1);
        }

        SECTION("SetNode(10)[:-1]") {
            auto set = SetNode(10);

            auto slice = BasicIndexingNode(&set, Slice(0, -1));

            auto sizeinfo = slice.sizeinfo();
            CHECK(sizeinfo.array_ptr == &set);
            CHECK(sizeinfo.multiplier == 1);
            CHECK(sizeinfo.offset == -1);
            CHECK(!sizeinfo.min.has_value());
            CHECK(!sizeinfo.max.has_value());

            // now substitute so we can see what we did to the array's size
            sizeinfo = sizeinfo.substitute();
            CHECK(sizeinfo.array_ptr == &set);
            CHECK(sizeinfo.multiplier == 1);
            CHECK(sizeinfo.offset == -1);
            CHECK(sizeinfo.min == 0);
            CHECK(sizeinfo.max == 9);
        }

        SECTION("SetNode(10)[-2:]") {
            auto set = SetNode(10);

            auto slice = BasicIndexingNode(&set, Slice(-2, std::nullopt));

            auto sizeinfo = slice.sizeinfo();
            CHECK(sizeinfo.array_ptr == &set);
            CHECK(sizeinfo.multiplier == 1);
            CHECK(sizeinfo.offset == 0);
            CHECK(!sizeinfo.min.has_value());
            CHECK(sizeinfo.max == 2);

            // now substitute so we can see what we did to the array's size
            sizeinfo = sizeinfo.substitute();
            CHECK(sizeinfo.array_ptr == &set);
            CHECK(sizeinfo.multiplier == 1);
            CHECK(sizeinfo.offset == 0);
            CHECK(sizeinfo.min == 0);
            CHECK(sizeinfo.max == 2);
        }

        SECTION("SetNode(10)[5:4]") {
            auto set = SetNode(10);

            auto slice = BasicIndexingNode(&set, Slice(5, 4));

            auto sizeinfo = slice.sizeinfo();
            CHECK(sizeinfo.array_ptr == &set);
            CHECK(sizeinfo.multiplier == 1);
            CHECK(sizeinfo.offset == -5);
            CHECK(!sizeinfo.min.has_value());
            CHECK(sizeinfo.max == 4);

            // now substitute so we can see what we did to the array's size
            sizeinfo = sizeinfo.substitute();
            CHECK(sizeinfo.array_ptr == &set);
            CHECK(sizeinfo.multiplier == 0);
            CHECK(sizeinfo.offset == 0);
            CHECK(sizeinfo.min == 0);
            CHECK(sizeinfo.max == 0);
        }
    }

    auto graph = Graph();

    GIVEN("A 1d ConstantNode") {
        std::vector<double> values = {30, 10, 40, 20};
        auto array_ptr = graph.emplace_node<ConstantNode>(values);

        WHEN("We do integer basic indexing") {
            auto ptr = graph.emplace_node<BasicIndexingNode>(array_ptr, 3);

            THEN("The resulting node has the shape we expect") {
                CHECK(ptr->ndim() == 0);
                CHECK(ptr->size() == 1);
                CHECK(std::ranges::equal(ptr->shape(), std::vector<ssize_t>()));
                CHECK(std::ranges::equal(ptr->strides(), std::vector<ssize_t>()));

                CHECK(ptr->contiguous());
            }

            THEN("The resulting node has the predecessors we expect") {
                CHECK(std::ranges::equal(ptr->predecessors(), std::vector<Node*>{array_ptr}));
                CHECK(std::ranges::equal(array_ptr->successors(), std::vector<Node*>{ptr}));
            }

            AND_WHEN("We create and then read the state") {
                auto state = graph.initialize_state();

                THEN("It has the values and shape we expect") {
                    CHECK(ptr->size(state) == 1);
                    CHECK(std::ranges::equal(ptr->shape(state), std::vector<ssize_t>()));
                    CHECK(std::ranges::equal(ptr->view(state), std::vector{20}));
                }
            }
        }

        WHEN("We do integer negative indexing") {
            auto ptr = graph.emplace_node<BasicIndexingNode>(array_ptr, -2);

            THEN("The resulting node has the shape we expect") {
                CHECK(ptr->ndim() == 0);
                CHECK(ptr->size() == 1);
                CHECK(std::ranges::equal(ptr->shape(), std::vector<ssize_t>()));
                CHECK(std::ranges::equal(ptr->strides(), std::vector<ssize_t>()));

                CHECK(ptr->contiguous());
            }

            AND_WHEN("We create and then read the state") {
                auto state = graph.initialize_state();

                THEN("It has the values and shape we expect") {
                    CHECK(ptr->size(state) == 1);
                    CHECK(std::ranges::equal(ptr->shape(state), std::vector<ssize_t>()));
                    CHECK(std::ranges::equal(ptr->view(state), std::vector{40}));
                }
            }
        }

        WHEN("We do indexing by an explicit Slice of length 1") {
            auto ptr = graph.emplace_node<BasicIndexingNode>(array_ptr, Slice(1, 2));

            THEN("The resulting node has the shape we expect") {
                CHECK(ptr->ndim() == 1);
                CHECK(ptr->size() == 1);
                CHECK(std::ranges::equal(ptr->shape(), std::vector{1}));
                CHECK(std::ranges::equal(ptr->strides(), std::vector{sizeof(double)}));

                CHECK(ptr->contiguous());
            }

            AND_WHEN("We create and then read the state") {
                auto state = graph.initialize_state();

                THEN("It has the values and shape we expect") {
                    CHECK(ptr->size(state) == 1);
                    CHECK(std::ranges::equal(ptr->shape(state), std::vector{1}));
                    CHECK(std::ranges::equal(ptr->view(state), std::vector{10}));
                }
            }
        }

        WHEN("We do indexing by an explicit Slice of length 2 using negative indexing") {
            auto ptr = graph.emplace_node<BasicIndexingNode>(array_ptr, Slice(1, -1));

            THEN("The resulting node has the shape we expect") {
                CHECK(ptr->ndim() == 1);
                CHECK(ptr->size() == 2);
                CHECK(std::ranges::equal(ptr->shape(), std::vector{2}));
                CHECK(std::ranges::equal(ptr->strides(), std::vector{sizeof(double)}));

                CHECK(ptr->contiguous());
            }

            AND_WHEN("We create and then read the state") {
                auto state = graph.initialize_state();

                THEN("It has the values and shape we expect") {
                    CHECK(ptr->size(state) == 2);
                    CHECK(std::ranges::equal(ptr->shape(state), std::vector{2}));
                    CHECK(std::ranges::equal(ptr->view(state), std::vector{10, 40}));
                }
            }
        }
    }

    GIVEN("A 2d ConstantNode") {
        std::vector<double> values = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        auto array_ptr =
                graph.emplace_node<ConstantNode>(values, std::initializer_list<ssize_t>{3, 3});

        WHEN("We do integer basic indexing") {
            auto ptr = graph.emplace_node<BasicIndexingNode>(array_ptr, 2, 0);

            THEN("The resulting node has the shape we expect") {
                CHECK(ptr->ndim() == 0);
                CHECK(ptr->size() == 1);
                CHECK(std::ranges::equal(ptr->shape(), std::vector<ssize_t>()));
                CHECK(std::ranges::equal(ptr->strides(), std::vector<ssize_t>()));

                CHECK(ptr->contiguous());
            }

            AND_WHEN("We create and then read the state") {
                auto state = graph.initialize_state();

                THEN("It has the values and shape we expect") {
                    CHECK(ptr->size(state) == 1);
                    CHECK(std::ranges::equal(ptr->shape(state), std::vector<int>{}));
                    CHECK(std::ranges::equal(ptr->view(state), std::vector{7}));
                }
            }
        }

        WHEN("We do indexing by an explicit Slice of length 1") {
            auto ptr = graph.emplace_node<BasicIndexingNode>(array_ptr, Slice(1, 2), Slice(0, 1));

            THEN("The resulting node has the shape we expect") {
                CHECK(ptr->ndim() == 2);
                CHECK(ptr->size() == 1);
                CHECK(std::ranges::equal(ptr->shape(), std::vector{1, 1}));
                CHECK(std::ranges::equal(ptr->strides(),
                                         std::vector{3 * sizeof(double), sizeof(double)}));

                CHECK(ptr->contiguous());
            }

            AND_WHEN("We create and then read the state") {
                auto state = graph.initialize_state();

                THEN("It has the values and shape we expect") {
                    CHECK(ptr->size(state) == 1);
                    CHECK(std::ranges::equal(ptr->shape(state), std::vector{1, 1}));
                    CHECK(std::ranges::equal(ptr->view(state), std::vector{4}));
                }
            }
        }

        WHEN("We do indexing by an explicit Slices of length 2 using negative indexing") {
            auto ptr = graph.emplace_node<BasicIndexingNode>(array_ptr, Slice(0, -1), Slice(-3, 2));

            THEN("The resulting node has the shape we expect") {
                CHECK(ptr->ndim() == 2);
                CHECK(ptr->size() == 4);
                CHECK(std::ranges::equal(ptr->shape(), std::vector{2, 2}));
                CHECK(std::ranges::equal(ptr->strides(),
                                         std::vector{3 * sizeof(double), sizeof(double)}));

                CHECK(!ptr->contiguous());
            }

            AND_WHEN("We create and then read the state") {
                auto state = graph.initialize_state();

                THEN("It has the values and shape we expect") {
                    CHECK(ptr->size(state) == 4);
                    CHECK(std::ranges::equal(ptr->shape(state), std::vector{2, 2}));
                    CHECK(std::ranges::equal(ptr->view(state), std::vector{1, 2, 4, 5}));
                }
            }
        }

        WHEN("We do indexing by a mix of slices and indices (slice, index)") {
            auto ptr = graph.emplace_node<BasicIndexingNode>(array_ptr, Slice(0, -1), 1);

            THEN("The resulting node has the shape we expect") {
                CHECK(ptr->ndim() == 1);
                CHECK(ptr->size() == 2);
                CHECK(std::ranges::equal(ptr->shape(), std::vector{2}));
                CHECK(std::ranges::equal(ptr->strides(), std::vector{3 * sizeof(double)}));

                CHECK(!ptr->contiguous());
            }

            AND_WHEN("We create and then read the state") {
                auto state = graph.initialize_state();

                THEN("It has the values and shape we expect") {
                    CHECK(ptr->size(state) == 2);
                    CHECK(std::ranges::equal(ptr->shape(state), std::vector{2}));
                    CHECK(std::ranges::equal(ptr->view(state), std::vector{2, 5}));
                }
            }
        }

        WHEN("We do indexing by a mix of slices and negative indices (slice, index)") {
            auto ptr = graph.emplace_node<BasicIndexingNode>(array_ptr, Slice(0, 3), -2);

            THEN("The resulting node has the shape we expect") {
                CHECK(ptr->ndim() == 1);
                CHECK(ptr->size() == 3);
                CHECK(std::ranges::equal(ptr->shape(), std::vector{3}));
                CHECK(std::ranges::equal(ptr->strides(), std::vector{3 * sizeof(double)}));

                CHECK(!ptr->contiguous());
            }

            AND_WHEN("We create and then read the state") {
                auto state = graph.initialize_state();

                THEN("It has the values and shape we expect") {
                    CHECK(ptr->size(state) == 3);
                    CHECK(std::ranges::equal(ptr->shape(state), std::vector{3}));
                    CHECK(std::ranges::equal(ptr->view(state), std::vector{2, 5, 8}));
                }
            }
        }

        WHEN("We do indexing by a mix of slices and indices (index, slice)") {
            auto ptr = graph.emplace_node<BasicIndexingNode>(array_ptr, 1, Slice(1, 3));

            THEN("The resulting node has the shape we expect") {
                CHECK(ptr->ndim() == 1);
                CHECK(ptr->size() == 2);
                CHECK(std::ranges::equal(ptr->shape(), std::vector{2}));
                CHECK(std::ranges::equal(ptr->strides(), std::vector{sizeof(double)}));

                CHECK(ptr->contiguous());
            }

            AND_WHEN("We create and then read the state") {
                auto state = graph.initialize_state();

                THEN("It has the values and shape we expect") {
                    CHECK(ptr->size(state) == 2);
                    CHECK(std::ranges::equal(ptr->shape(state), std::vector{2}));
                    CHECK(std::ranges::equal(ptr->view(state), std::vector{5, 6}));
                }
            }
        }
    }

    GIVEN("A 3d ConstantNode") {
        std::vector<double> values = {1, 2, 3, 4, 5, 6, 7, 8};
        auto array_ptr =
                graph.emplace_node<ConstantNode>(values, std::initializer_list<ssize_t>{2, 2, 2});

        WHEN("We do integer basic indexing") {
            auto ptr = graph.emplace_node<BasicIndexingNode>(array_ptr, 1, 0, 1);

            THEN("The resulting node has the shape we expect") {
                CHECK(ptr->ndim() == 0);
                CHECK(ptr->size() == 1);
                CHECK(std::ranges::equal(ptr->shape(), std::vector<ssize_t>()));
                CHECK(std::ranges::equal(ptr->strides(), std::vector<ssize_t>()));
            }

            THEN("We get the min/max/integral we expect") {
                CHECK(ptr->min() == 1);
                CHECK(ptr->max() == 8);
                CHECK(ptr->integral());
            }

            AND_WHEN("We create and then read the state") {
                auto state = graph.initialize_state();

                THEN("It has the values and shape we expect") {
                    CHECK(ptr->size(state) == 1);
                    CHECK(std::ranges::equal(ptr->shape(state), ptr->shape()));
                    CHECK(std::ranges::equal(ptr->view(state), std::vector{6}));
                }
            }
        }

        WHEN("We do indexing by a mix of slices and indices (slice, index, slice)") {
            auto ptr =
                    graph.emplace_node<BasicIndexingNode>(array_ptr, Slice(0, -1), 1, Slice(0, 2));

            THEN("The resulting node has the shape we expect") {
                CHECK(ptr->ndim() == 2);
                CHECK(ptr->size() == 2);
                CHECK(std::ranges::equal(ptr->shape(), std::vector{1, 2}));
                CHECK(std::ranges::equal(ptr->strides(),
                                         std::vector{4 * sizeof(double), sizeof(double)}));
            }

            AND_WHEN("We create and then read the state") {
                auto state = graph.initialize_state();

                THEN("It has the values and shape we expect") {
                    CHECK(ptr->size(state) == 2);
                    CHECK(std::ranges::equal(ptr->shape(state), ptr->shape()));
                    CHECK(std::ranges::equal(ptr->view(state), std::vector{3, 4}));
                }
            }
        }
    }

    GIVEN("x = List(n), that is x is fixed length") {
        constexpr int num_items = 5;

        auto x_ptr = graph.emplace_node<ListNode>(num_items);

        WHEN("y = x[:]") {
            auto y_ptr = graph.emplace_node<BasicIndexingNode>(x_ptr, Slice());

            THEN("y's shape is independent of state") {
                CHECK(std::ranges::equal(y_ptr->shape(), std::vector{num_items}));
                CHECK(y_ptr->size() == num_items);
                CHECK(y_ptr->ndim() == 1);
            }

            THEN("y == x") {
                auto state = graph.initialize_state();
                CHECK(std::ranges::equal(x_ptr->view(state), y_ptr->view(state)));
            }
        }
    }

    WHEN("We access a constant fixed-length 1d array by a positive index") {
        const std::vector<double> values{0, 10, 20, 30, 40};
        auto arr_ptr = graph.emplace_node<ConstantNode>(values);

        auto a_ptr = graph.emplace_node<BasicIndexingNode>(arr_ptr, 3);

        THEN("The resulting BasicIndexingNode has the shape we expect") {
            CHECK(a_ptr->size() == 1);
            CHECK(a_ptr->ndim() == 0);
            CHECK(a_ptr->shape().size() == 0);
        }

        AND_WHEN("We inspect the state") {
            auto state = graph.initialize_state();

            THEN("We get the value we expect") {
                CHECK(a_ptr->size(state) == 1);
                CHECK(a_ptr->shape(state).size() == 0);
                CHECK(std::ranges::equal(a_ptr->view(state), std::vector{30}));
            }
        }
    }

    WHEN("We access a constant fixed-length 1d array by a negative index") {
        const std::vector<double> values{0, 10, 20, 30, 40};
        auto arr_ptr = graph.emplace_node<ConstantNode>(values);

        auto a_ptr = graph.emplace_node<BasicIndexingNode>(arr_ptr, -1);

        THEN("The resulting BasicIndexingNode has the shape we expect") {
            CHECK(a_ptr->size() == 1);
            CHECK(a_ptr->ndim() == 0);
            CHECK(a_ptr->shape().size() == 0);
        }

        AND_WHEN("We inspect the state") {
            auto state = graph.initialize_state();

            THEN("We get the value we expect") {
                CHECK(a_ptr->size(state) == 1);
                CHECK(a_ptr->shape(state).size() == 0);
                CHECK(std::ranges::equal(a_ptr->view(state), std::vector{40}));
            }
        }
    }

    WHEN("We access a fixed-length 1d array by a slice specified with negative indices") {
        auto x_ptr = graph.emplace_node<ListNode>(5);

        auto y_ptr = graph.emplace_node<BasicIndexingNode>(x_ptr, Slice(-1));  // x[:-1]

        THEN("The resulting BasicIndexingNode has the shape we expect") {
            CHECK(y_ptr->size() == x_ptr->size() - 1);
            CHECK(y_ptr->ndim() == 1);
            CHECK(y_ptr->shape().size() == 1);
            CHECK(y_ptr->shape()[0] == y_ptr->size());
        }

        AND_WHEN("We inspect the state") {
            auto state = graph.initialize_state();

            THEN("We get the value we expect") {
                CHECK(y_ptr->size(state) == x_ptr->size() - 1);
                CHECK(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == y_ptr->size());

                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{0, 1, 2, 3}));
            }
        }
    }

    WHEN("We access a fixed-length 1d array by a slice") {
        auto x_ptr = graph.emplace_node<ListNode>(5);

        auto y_ptr = graph.emplace_node<BasicIndexingNode>(x_ptr, Slice(1, std::nullopt));  // x[1:]

        THEN("The resulting BasicIndexingNode has the shape we expect") {
            CHECK(y_ptr->size() == x_ptr->size() - 1);
            CHECK(y_ptr->ndim() == 1);
            CHECK(y_ptr->shape().size() == 1);
            CHECK(y_ptr->shape()[0] == y_ptr->size());
        }

        AND_WHEN("We inspect the state") {
            auto state = graph.initialize_state();

            THEN("We get the value we expect") {
                CHECK(y_ptr->size(state) == x_ptr->size() - 1);
                CHECK(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == y_ptr->size());

                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{1, 2, 3, 4}));
            }
        }
    }

    GIVEN("x = List(5); y = x[1::2]") {
        auto x_ptr = graph.emplace_node<ListNode>(5);
        auto y_ptr = graph.emplace_node<BasicIndexingNode>(x_ptr, Slice(1, std::nullopt, 2));

        THEN("y has the shape we expect") {
            CHECK(y_ptr->size() == 2);
            CHECK(y_ptr->ndim() == 1);
        }

        auto state = graph.initialize_state();

        THEN("We get the default states we expect") {
            CHECK(std::ranges::equal(x_ptr->view(state), std::vector{0, 1, 2, 3, 4}));
            CHECK(std::ranges::equal(y_ptr->view(state), std::vector{1, 3}));
        }

        WHEN("We do propagation") {
            x_ptr->exchange(state, 1, 2);

            x_ptr->propagate(state);
            y_ptr->propagate(state);

            THEN("The states are as expected") {
                CHECK(std::ranges::equal(x_ptr->view(state), std::vector{0, 2, 1, 3, 4}));
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{2, 3}));

                verify_array_diff({1, 3}, {2, 3}, y_ptr->diff(state));
            }
        }
    }

    GIVEN("x = Binary((3, 3)); y = x[::2, 1:]") {
        auto x_ptr = graph.emplace_node<BinaryNode>(std::vector<ssize_t>{3, 3});
        auto y_ptr = graph.emplace_node<BasicIndexingNode>(
                x_ptr, Slice(std::nullopt, std::nullopt, 2), Slice(1, std::nullopt));

        THEN("y has the shape and strides we expect") {
            CHECK(std::ranges::equal(y_ptr->shape(), std::vector{2, 2}));

            CHECK(std::ranges::equal(y_ptr->strides(), std::vector{48, 8}));
        }

        auto state = graph.initialize_state();

        THEN("We get the default states we expect") {
            CHECK(std::ranges::equal(x_ptr->view(state), std::vector<ssize_t>(9)));
            CHECK(std::ranges::equal(y_ptr->view(state), std::vector<ssize_t>(4)));
        }

        WHEN("We do propagation") {
            x_ptr->flip(state, 0);
            x_ptr->flip(state, 1);

            x_ptr->propagate(state);
            y_ptr->propagate(state);

            THEN("The states are as expected") {
                CHECK(std::ranges::equal(x_ptr->view(state),
                                         std::vector{1, 1, 0, 0, 0, 0, 0, 0, 0}));
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{1, 0, 0, 0}));

                verify_array_diff({0, 0, 0, 0}, {1, 0, 0, 0}, y_ptr->diff(state));
            }
        }
    }

    WHEN("We access a dynamic length 1d array by a slice to the end") {
        auto x_ptr = graph.emplace_node<SetNode>(5);
        auto y_ptr = graph.emplace_node<BasicIndexingNode>(x_ptr, Slice(1, std::nullopt));  // x[1:]

        THEN("The resulting BasicIndexingNode has the shape we expect") {
            CHECK(y_ptr->size() == Array::DYNAMIC_SIZE);
            CHECK(y_ptr->ndim() == 1);
            CHECK(y_ptr->shape().size() == 1);
            CHECK(y_ptr->shape()[0] == y_ptr->size());
        }

        AND_WHEN("We inspect the state") {
            auto state = graph.initialize_state();

            THEN("We get the value we expect") {
                CHECK(y_ptr->size(state) == 0);
                CHECK(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == y_ptr->size(state));

                CHECK(std::ranges::equal(y_ptr->view(state), std::vector<double>()));
            }

            AND_WHEN("We change the shape of the dynamic array") {
                x_ptr->grow(state);

                x_ptr->propagate(state);
                y_ptr->propagate(state);

                CHECK(y_ptr->size(state) == x_ptr->size(state) - 1);
                CHECK(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == y_ptr->size(state));
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector<double>()));

                x_ptr->grow(state);
                x_ptr->grow(state);

                x_ptr->propagate(state);
                y_ptr->propagate(state);

                CHECK(y_ptr->size(state) == x_ptr->size(state) - 1);
                CHECK(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == y_ptr->size(state));
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{1, 2}));
            }
        }
    }

    WHEN("We access a dynamic length 1d array by a slice from the beginning with negative end") {
        auto x_ptr = graph.emplace_node<SetNode>(5);
        auto y_ptr = graph.emplace_node<BasicIndexingNode>(x_ptr, Slice(0, -2));  // x[:-2]

        THEN("The resulting BasicIndexingNode has the shape we expect") {
            CHECK(y_ptr->size() == Array::DYNAMIC_SIZE);
            CHECK(y_ptr->ndim() == 1);
            CHECK(y_ptr->shape().size() == 1);
            CHECK(y_ptr->shape()[0] == y_ptr->size());
        }

        AND_WHEN("We inspect the state") {
            auto state = graph.initialize_state();

            THEN("We get the value we expect") {
                CHECK(y_ptr->size(state) == 0);
                CHECK(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == y_ptr->size(state));

                CHECK(std::ranges::equal(y_ptr->view(state), std::vector<double>()));
            }

            AND_WHEN("We grow the dynamic array below the range") {
                x_ptr->grow(state);
                x_ptr->grow(state);

                x_ptr->propagate(state);
                y_ptr->propagate(state);

                // Still should have 0 size
                CHECK(y_ptr->size(state) == 0);
                CHECK(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == y_ptr->size(state));
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector<double>()));
                CHECK(y_ptr->diff(state).size() == 0);
            }

            AND_WHEN("We grow the dynamic array up to the range") {
                x_ptr->grow(state);
                x_ptr->grow(state);
                x_ptr->grow(state);

                x_ptr->propagate(state);
                y_ptr->propagate(state);

                CHECK(y_ptr->size(state) == 1);
                REQUIRE(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == y_ptr->size(state));
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{0}));

                verify_array_diff({}, {0}, y_ptr->diff(state));

                x_ptr->commit(state);
                y_ptr->commit(state);

                x_ptr->grow(state);

                x_ptr->propagate(state);
                y_ptr->propagate(state);

                verify_array_diff({0}, {0, 1}, y_ptr->diff(state));

                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{0, 1}));
            }

            AND_WHEN("We grow the dynamic array past the range") {
                for (int i = 0; i < 4; i++) x_ptr->grow(state);

                x_ptr->propagate(state);
                y_ptr->propagate(state);

                CHECK(y_ptr->size(state) == 2);
                REQUIRE(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == y_ptr->size(state));
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{0, 1}));

                verify_array_diff({}, {0, 1}, y_ptr->diff(state));

                x_ptr->commit(state);
                y_ptr->commit(state);

                x_ptr->grow(state);

                x_ptr->propagate(state);
                y_ptr->propagate(state);

                verify_array_diff({0, 1}, {0, 1, 2}, y_ptr->diff(state));
            }

            AND_WHEN("We shrink the dynamic array below the range") {
                for (int i = 0; i < 4; i++) x_ptr->grow(state);

                x_ptr->propagate(state);
                y_ptr->propagate(state);
                x_ptr->commit(state);
                y_ptr->commit(state);

                // These should have already been tested
                REQUIRE(std::ranges::equal(y_ptr->view(state), std::vector{0, 1}));
                REQUIRE(y_ptr->diff(state).size() == 0);

                // Now shrink
                x_ptr->shrink(state);
                x_ptr->propagate(state);
                y_ptr->propagate(state);
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{0}));
                verify_array_diff({0, 1}, {0}, y_ptr->diff(state));

                x_ptr->commit(state);
                y_ptr->commit(state);

                // Shrink again
                x_ptr->shrink(state);
                x_ptr->propagate(state);
                y_ptr->propagate(state);
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector<double>()));
                verify_array_diff({0}, {}, y_ptr->diff(state));

                x_ptr->commit(state);
                y_ptr->commit(state);

                // Shrink again, should be no change
                x_ptr->shrink(state);
                x_ptr->propagate(state);
                y_ptr->propagate(state);
                x_ptr->commit(state);
                y_ptr->commit(state);
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector<double>()));
                CHECK(y_ptr->diff(state).size() == 0);
            }

            AND_WHEN(
                    "We shrink the dynamic array below the range, and then update a value above "
                    "that range") {
                for (int i = 0; i < 4; i++) x_ptr->grow(state);

                x_ptr->propagate(state);
                y_ptr->propagate(state);
                x_ptr->commit(state);
                y_ptr->commit(state);

                // These should have already been tested
                REQUIRE(std::ranges::equal(y_ptr->view(state), std::vector{0, 1}));
                REQUIRE(y_ptr->diff(state).size() == 0);

                // Now shrink
                x_ptr->grow(state);            // [0, 1, 2, 3, 4][:-2] = [0, 1, 2]
                x_ptr->exchange(state, 2, 4);  // [0, 1, 4, 3, 2][:-2] = [0, 1, 4]
                x_ptr->shrink(state);          // [0, 1, 4, 3][:-2] = [0, 1]
                x_ptr->propagate(state);
                y_ptr->propagate(state);

                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{0, 1}));
                verify_array_diff({0, 1}, {0, 1}, y_ptr->diff(state));
            }

            AND_WHEN("We shrink and grow the dynamic array") {
                for (int i = 0; i < 3; i++) {
                    x_ptr->grow(state);
                    x_ptr->grow(state);

                    x_ptr->shrink(state);
                }

                x_ptr->propagate(state);
                y_ptr->propagate(state);

                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{0}));
                verify_array_diff({}, {0}, y_ptr->diff(state));

                x_ptr->commit(state);
                y_ptr->commit(state);

                x_ptr->grow(state);
                x_ptr->exchange(state, 1, 2);
                x_ptr->shrink(state);

                x_ptr->propagate(state);
                y_ptr->propagate(state);

                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{0}));
                verify_array_diff({0}, {0}, y_ptr->diff(state));
            }

            AND_WHEN("We change the shape of the dynamic array, and then revert") {
                x_ptr->grow(state);
                x_ptr->grow(state);
                x_ptr->grow(state);

                x_ptr->revert(state);
                y_ptr->revert(state);

                CHECK(y_ptr->size(state) == 0);
                REQUIRE(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == y_ptr->size(state));
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector<double>()));
                CHECK(y_ptr->diff(state).size() == 0);
            }
        }
    }

    WHEN("We access a dynamic length 1d array by a slice with a negative start") {
        auto x_ptr = graph.emplace_node<SetNode>(5);
        auto y_ptr =
                graph.emplace_node<BasicIndexingNode>(x_ptr, Slice(-2, std::nullopt));  // x[-2:]

        THEN("The resulting BasicIndexingNode has the shape we expect") {
            CHECK(y_ptr->size() == Array::DYNAMIC_SIZE);
            CHECK(y_ptr->ndim() == 1);
            CHECK(y_ptr->shape().size() == 1);
            CHECK(y_ptr->shape()[0] == y_ptr->size());
        }

        AND_WHEN("We inspect the state") {
            auto state = graph.initialize_state();

            THEN("We get the value we expect") {
                CHECK(y_ptr->size(state) == 0);
                CHECK(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == y_ptr->size(state));

                CHECK(std::ranges::equal(y_ptr->view(state), std::vector<double>()));
            }

            AND_WHEN("We grow the dynamic array below the range") {
                x_ptr->grow(state);

                x_ptr->propagate(state);
                y_ptr->propagate(state);

                CHECK(y_ptr->size(state) == 1);
                CHECK(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == 1);
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{0}));
                verify_array_diff({}, {0}, y_ptr->diff(state));
            }

            AND_WHEN("We grow the dynamic array up to the range") {
                x_ptr->grow(state);
                x_ptr->grow(state);

                x_ptr->propagate(state);
                y_ptr->propagate(state);

                CHECK(y_ptr->size(state) == 2);
                REQUIRE(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == 2);
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{0, 1}));

                verify_array_diff({}, {0, 1}, y_ptr->diff(state));

                x_ptr->commit(state);
                y_ptr->commit(state);

                x_ptr->grow(state);

                x_ptr->propagate(state);
                y_ptr->propagate(state);

                verify_array_diff({0, 1}, {1, 2}, y_ptr->diff(state));

                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{1, 2}));
            }

            AND_WHEN("We grow the dynamic array past the range") {
                for (int i = 0; i < 4; i++) x_ptr->grow(state);

                x_ptr->propagate(state);
                y_ptr->propagate(state);

                CHECK(y_ptr->size(state) == 2);
                REQUIRE(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == y_ptr->size(state));
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{2, 3}));

                verify_array_diff({}, {2, 3}, y_ptr->diff(state));

                x_ptr->commit(state);
                y_ptr->commit(state);

                x_ptr->grow(state);

                x_ptr->propagate(state);
                y_ptr->propagate(state);

                verify_array_diff({2, 3}, {3, 4}, y_ptr->diff(state));
            }

            AND_WHEN("We shrink the dynamic array below the range") {
                for (int i = 0; i < 4; i++) x_ptr->grow(state);

                x_ptr->propagate(state);  // [0, 1, 2, 3]
                y_ptr->propagate(state);  // [2, 3]
                x_ptr->commit(state);
                y_ptr->commit(state);

                // These should have already been tested
                REQUIRE(std::ranges::equal(y_ptr->view(state), std::vector{2, 3}));
                REQUIRE(y_ptr->diff(state).size() == 0);

                // Now shrink
                x_ptr->shrink(state);
                x_ptr->propagate(state);
                y_ptr->propagate(state);
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{1, 2}));
                verify_array_diff({2, 3}, {1, 2}, y_ptr->diff(state));

                x_ptr->commit(state);
                y_ptr->commit(state);

                // Shrink twice, should now be only length one
                x_ptr->shrink(state);
                x_ptr->shrink(state);
                x_ptr->propagate(state);
                y_ptr->propagate(state);
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{0}));
                verify_array_diff({1, 2}, {0}, y_ptr->diff(state));

                x_ptr->commit(state);
                y_ptr->commit(state);

                // Shrink again, should now be length zero
                x_ptr->shrink(state);
                x_ptr->propagate(state);
                y_ptr->propagate(state);
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector<double>()));
                verify_array_diff({0}, {}, y_ptr->diff(state));
            }

            AND_WHEN("We shrink and grow the dynamic array") {
                for (int i = 0; i < 4; i++) {
                    x_ptr->grow(state);
                    x_ptr->grow(state);

                    x_ptr->shrink(state);
                }

                x_ptr->propagate(state);  // [0, 1, 2, 3]
                y_ptr->propagate(state);  // [2, 3]

                CHECK(std::ranges::equal(y_ptr->view(state), std::vector{2, 3}));
                verify_array_diff({}, {2, 3}, y_ptr->diff(state));
            }

            AND_WHEN("We change the shape of the dynamic array, and then revert") {
                x_ptr->grow(state);
                x_ptr->grow(state);
                x_ptr->grow(state);

                x_ptr->revert(state);
                y_ptr->revert(state);

                CHECK(y_ptr->size(state) == 0);
                REQUIRE(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == y_ptr->size(state));
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector<double>()));
                CHECK(y_ptr->diff(state).size() == 0);
            }
        }
    }

    WHEN("We use the DynamicArrayTestingNode to do some fuzzing with BasicIndexingNode on 1D "
         "dynamic arrays") {
        auto slice = GENERATE(Slice(0, std::nullopt), Slice(0, 1), Slice(0, 2), Slice(0, 10),
                              Slice(1, std::nullopt), Slice(5, std::nullopt), Slice(0, -1),
                              Slice(0, -5), Slice(1, -5), Slice(3, -5), Slice(-1, std::nullopt),
                              Slice(-5, std::nullopt), Slice(-5, -2));

        auto x_ptr =
                graph.emplace_node<DynamicArrayTestingNode>(std::initializer_list<ssize_t>{-1});
        auto y_ptr = graph.emplace_node<BasicIndexingNode>(x_ptr, slice);
        auto validation = graph.emplace_node<ArrayValidationNode>(y_ptr);

        AND_WHEN("We do random moves and accept") {
            auto state = graph.initialize_state();
            auto rng = RngAdaptor(std::mt19937(42));

            for (int i = 0; i < 1000; ++i) {
                x_ptr->random_moves(state, rng, 20);

                x_ptr->propagate(state);
                y_ptr->propagate(state);
                validation->propagate(state);

                x_ptr->commit(state);
                y_ptr->commit(state);
                validation->commit(state);
            }
        }
    }

    // TODO slicing on multidimensional dynamic array, when we have one to test...
}

}  // namespace dwave::optimization

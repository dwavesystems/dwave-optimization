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
#include "catch2/matchers/catch_matchers_all.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/nodes/collections.hpp"
#include "dwave-optimization/nodes/constants.hpp"
#include "dwave-optimization/nodes/indexing.hpp"
#include "dwave-optimization/nodes/numbers.hpp"
#include "dwave-optimization/nodes/testing.hpp"

using Catch::Matchers::RangeEquals;

namespace dwave::optimization {

TEST_CASE("BasicIndexingNode") {
    SECTION("SizeInfo") {
        SECTION("ListNode(10)[3:7]") {
            auto set = ListNode(10);

            auto slice = BasicIndexingNode(&set, Slice(3, 7));

            auto sizeinfo = slice.sizeinfo();
            CHECK(sizeinfo.array_ptr == nullptr);
            CHECK(sizeinfo.offset == 4);
            REQUIRE(sizeinfo.min.has_value());
            REQUIRE(sizeinfo.max.has_value());
            CHECK(sizeinfo.min.value() == 4);
            CHECK(sizeinfo.max.value() == 4);
            // substitute should return the same sizeinfo
            CHECK(sizeinfo == sizeinfo.substitute());
        }

        SECTION("SetNode(10)[:]") {
            auto set = SetNode(10);

            auto slice = BasicIndexingNode(&set, Slice());

            auto sizeinfo = slice.sizeinfo();
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

            // in this case, size should be always 0!
            CHECK(slice.sizeinfo() == 0);
        }

        SECTION("SetNode(10)[:1]") {
            auto set = SetNode(10);

            auto slice = BasicIndexingNode(&set, Slice(1));

            auto sizeinfo = slice.sizeinfo();
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
            CHECK(sizeinfo.min == 0);
            CHECK(sizeinfo.max == 2);
        }

        SECTION("SetNode(10)[5:4]") {
            auto set = SetNode(10);

            auto slice = BasicIndexingNode(&set, Slice(5, 4));

            auto sizeinfo = slice.sizeinfo();
            CHECK(sizeinfo.array_ptr == &set);
            CHECK(sizeinfo.multiplier == 0);
            CHECK(sizeinfo.offset == 0);
            CHECK(sizeinfo.min == 0);
            CHECK(sizeinfo.max == 0);
        }

        SECTION("SetNode(10)[-4:-5]") {
            auto set = SetNode(10);

            auto slice = BasicIndexingNode(&set, Slice(-4, -5));

            auto sizeinfo = slice.sizeinfo();
            CHECK(sizeinfo.array_ptr == &set);
            CHECK(sizeinfo.multiplier == 0);
            CHECK(sizeinfo.offset == 0);
            CHECK(sizeinfo.min == 0);
            CHECK(sizeinfo.max == 0);
        }

        SECTION("SetNode(10)[-4:5]") {
            auto set = SetNode(3);

            auto slice = BasicIndexingNode(&set, Slice(-4, 5));

            // Check that it "gives up" because the size is not a linear
            // function of the predecessor's size
            auto sizeinfo = slice.sizeinfo();
            CHECK(sizeinfo.array_ptr == &slice);
            CHECK(sizeinfo.multiplier == 1);
            CHECK(sizeinfo.offset == 0);
            CHECK(sizeinfo.min == 0);
            // NOTE: unfortunate that we can't limit it to 3 here given the base array is max 3,
            // but substitute has to stop here and for now we're not doing any recursion in the
            // initial sizeinfo() call.
            CHECK(sizeinfo.max == 4);
        }

        SECTION("SetNode(10)[:-1][-1:]") {
            auto set = SetNode(10);

            auto slice0 = BasicIndexingNode(&set, Slice(std::nullopt, -1));
            auto slice1 = BasicIndexingNode(&slice0, Slice(-1, std::nullopt));

            auto sizeinfo = slice1.sizeinfo();
            CHECK(sizeinfo.array_ptr == &set);
            CHECK(sizeinfo.multiplier == 1);
            CHECK(sizeinfo.offset == -1);
            CHECK(sizeinfo.min == 0);
            CHECK(sizeinfo.max == 1);
        }

        SECTION("Dynamic(n, 5, 2)[5:7, 1:, :]") {
            // the resuling array will either be (0, 4, 2), (1, 4, 2), or (2, 4, 2) array.
            auto dynamic = DynamicArrayTestingNode(std::initializer_list<ssize_t>{-1, 5, 2});

            auto slice = BasicIndexingNode(&dynamic, Slice(5, 7), Slice(1, std::nullopt), Slice());

            CHECK_THAT(slice.shape(), RangeEquals({-1, 4, 2}));  // sanity check

            auto sizeinfo = slice.sizeinfo();
            CHECK(sizeinfo.array_ptr == &dynamic);
            CHECK(sizeinfo.multiplier == fraction(4 * 2, 5 * 2));
            CHECK(sizeinfo.offset == -5 * 5 * 2 * sizeinfo.multiplier);  // output has shape 0 when
                                                                         // input has shape 5x5x2
            CHECK(!sizeinfo.min.has_value());
            CHECK(sizeinfo.max == 2 * 4 * 2);

            sizeinfo = sizeinfo.substitute();
            CHECK(sizeinfo.array_ptr == &dynamic);
            CHECK(sizeinfo.multiplier == fraction(4 * 2, 5 * 2));
            CHECK(sizeinfo.offset == -5 * 5 * 2 * sizeinfo.multiplier);
            CHECK(!sizeinfo.min.has_value());  // dynamic doesn't know either (by construction)
            CHECK(!dynamic.sizeinfo().min.has_value());
            CHECK(sizeinfo.max == 2 * 4 * 2);
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
                    CHECK_THAT(ptr->view(state), RangeEquals({20}));
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
                    CHECK_THAT(ptr->view(state), RangeEquals({40}));
                }
            }
        }

        WHEN("We do indexing by an explicit Slice of length 1") {
            auto ptr = graph.emplace_node<BasicIndexingNode>(array_ptr, Slice(1, 2));

            THEN("The resulting node has the shape we expect") {
                CHECK(ptr->ndim() == 1);
                CHECK(ptr->size() == 1);
                CHECK_THAT(ptr->shape(), RangeEquals({1}));
                CHECK_THAT(ptr->strides(), RangeEquals({sizeof(double)}));

                CHECK(ptr->contiguous());
            }

            AND_WHEN("We create and then read the state") {
                auto state = graph.initialize_state();

                THEN("It has the values and shape we expect") {
                    CHECK(ptr->size(state) == 1);
                    CHECK_THAT(ptr->shape(state), RangeEquals({1}));
                    CHECK_THAT(ptr->view(state), RangeEquals({10}));
                }
            }
        }

        WHEN("We do indexing by an explicit Slice of length 2 using negative indexing") {
            auto ptr = graph.emplace_node<BasicIndexingNode>(array_ptr, Slice(1, -1));

            THEN("The resulting node has the shape we expect") {
                CHECK(ptr->ndim() == 1);
                CHECK(ptr->size() == 2);
                CHECK_THAT(ptr->shape(), RangeEquals({2}));
                CHECK_THAT(ptr->strides(), RangeEquals({sizeof(double)}));

                CHECK(ptr->contiguous());
            }

            AND_WHEN("We create and then read the state") {
                auto state = graph.initialize_state();

                THEN("It has the values and shape we expect") {
                    CHECK(ptr->size(state) == 2);
                    CHECK_THAT(ptr->shape(state), RangeEquals({2}));
                    CHECK_THAT(ptr->view(state), RangeEquals({10, 40}));
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
                    CHECK_THAT(ptr->shape(state), RangeEquals(std::vector<ssize_t>{}));
                    CHECK_THAT(ptr->view(state), RangeEquals({7}));
                }
            }
        }

        WHEN("We do indexing by an explicit Slice of length 1") {
            auto ptr = graph.emplace_node<BasicIndexingNode>(array_ptr, Slice(1, 2), Slice(0, 1));

            THEN("The resulting node has the shape we expect") {
                CHECK(ptr->ndim() == 2);
                CHECK(ptr->size() == 1);
                CHECK_THAT(ptr->shape(), RangeEquals({1, 1}));
                CHECK_THAT(ptr->strides(), RangeEquals({3 * sizeof(double), sizeof(double)}));

                CHECK(ptr->contiguous());
            }

            AND_WHEN("We create and then read the state") {
                auto state = graph.initialize_state();

                THEN("It has the values and shape we expect") {
                    CHECK(ptr->size(state) == 1);
                    CHECK_THAT(ptr->shape(state), RangeEquals({1, 1}));
                    CHECK_THAT(ptr->view(state), RangeEquals({4}));
                }
            }
        }

        WHEN("We do indexing by an explicit Slices of length 2 using negative indexing") {
            auto ptr = graph.emplace_node<BasicIndexingNode>(array_ptr, Slice(0, -1), Slice(-3, 2));

            THEN("The resulting node has the shape we expect") {
                CHECK(ptr->ndim() == 2);
                CHECK(ptr->size() == 4);
                CHECK_THAT(ptr->shape(), RangeEquals({2, 2}));
                CHECK_THAT(ptr->strides(), RangeEquals({3 * sizeof(double), sizeof(double)}));

                CHECK(!ptr->contiguous());
            }

            AND_WHEN("We create and then read the state") {
                auto state = graph.initialize_state();

                THEN("It has the values and shape we expect") {
                    CHECK(ptr->size(state) == 4);
                    CHECK_THAT(ptr->shape(state), RangeEquals({2, 2}));
                    CHECK_THAT(ptr->view(state), RangeEquals({1, 2, 4, 5}));
                }
            }
        }

        WHEN("We do indexing by a mix of slices and indices (slice, index)") {
            auto ptr = graph.emplace_node<BasicIndexingNode>(array_ptr, Slice(0, -1), 1);

            THEN("The resulting node has the shape we expect") {
                CHECK(ptr->ndim() == 1);
                CHECK(ptr->size() == 2);
                CHECK_THAT(ptr->shape(), RangeEquals({2}));
                CHECK_THAT(ptr->strides(), RangeEquals({3 * sizeof(double)}));

                CHECK(!ptr->contiguous());
            }

            AND_WHEN("We create and then read the state") {
                auto state = graph.initialize_state();

                THEN("It has the values and shape we expect") {
                    CHECK(ptr->size(state) == 2);
                    CHECK_THAT(ptr->shape(state), RangeEquals({2}));
                    CHECK_THAT(ptr->view(state), RangeEquals({2, 5}));
                }
            }
        }

        WHEN("We do indexing by a mix of slices and negative indices (slice, index)") {
            auto ptr = graph.emplace_node<BasicIndexingNode>(array_ptr, Slice(0, 3), -2);

            THEN("The resulting node has the shape we expect") {
                CHECK(ptr->ndim() == 1);
                CHECK(ptr->size() == 3);
                CHECK_THAT(ptr->shape(), RangeEquals({3}));
                CHECK_THAT(ptr->strides(), RangeEquals({3 * sizeof(double)}));

                CHECK(!ptr->contiguous());
            }

            AND_WHEN("We create and then read the state") {
                auto state = graph.initialize_state();

                THEN("It has the values and shape we expect") {
                    CHECK(ptr->size(state) == 3);
                    CHECK_THAT(ptr->shape(state), RangeEquals({3}));
                    CHECK_THAT(ptr->view(state), RangeEquals({2, 5, 8}));
                }
            }
        }

        WHEN("We do indexing by a mix of slices and indices (index, slice)") {
            auto ptr = graph.emplace_node<BasicIndexingNode>(array_ptr, 1, Slice(1, 3));

            THEN("The resulting node has the shape we expect") {
                CHECK(ptr->ndim() == 1);
                CHECK(ptr->size() == 2);
                CHECK_THAT(ptr->shape(), RangeEquals({2}));
                CHECK_THAT(ptr->strides(), RangeEquals({sizeof(double)}));

                CHECK(ptr->contiguous());
            }

            AND_WHEN("We create and then read the state") {
                auto state = graph.initialize_state();

                THEN("It has the values and shape we expect") {
                    CHECK(ptr->size(state) == 2);
                    CHECK_THAT(ptr->shape(state), RangeEquals({2}));
                    CHECK_THAT(ptr->view(state), RangeEquals({5, 6}));
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
                    CHECK_THAT(ptr->view(state), RangeEquals({6}));
                }
            }
        }

        WHEN("We do indexing by a mix of slices and indices (slice, index, slice)") {
            auto ptr =
                    graph.emplace_node<BasicIndexingNode>(array_ptr, Slice(0, -1), 1, Slice(0, 2));

            THEN("The resulting node has the shape we expect") {
                CHECK(ptr->ndim() == 2);
                CHECK(ptr->size() == 2);
                CHECK_THAT(ptr->shape(), RangeEquals({1, 2}));
                CHECK_THAT(ptr->strides(), RangeEquals({4 * sizeof(double), sizeof(double)}));
            }

            AND_WHEN("We create and then read the state") {
                auto state = graph.initialize_state();

                THEN("It has the values and shape we expect") {
                    CHECK(ptr->size(state) == 2);
                    CHECK(std::ranges::equal(ptr->shape(state), ptr->shape()));
                    CHECK_THAT(ptr->view(state), RangeEquals({3, 4}));
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
                CHECK_THAT(y_ptr->shape(), RangeEquals({num_items}));
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
                CHECK_THAT(a_ptr->view(state), RangeEquals({30}));
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
                CHECK_THAT(a_ptr->view(state), RangeEquals({40}));
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

                CHECK_THAT(y_ptr->view(state), RangeEquals({0, 1, 2, 3}));
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

                CHECK_THAT(y_ptr->view(state), RangeEquals({1, 2, 3, 4}));
            }
        }
    }

    GIVEN("x = List(5); y = x[1::2]") {
        auto x_ptr = graph.emplace_node<ListNode>(5);
        auto y_ptr = graph.emplace_node<BasicIndexingNode>(x_ptr, Slice(1, std::nullopt, 2));

        graph.emplace_node<ArrayValidationNode>(y_ptr);

        THEN("y has the shape we expect") {
            CHECK(y_ptr->size() == 2);
            CHECK(y_ptr->ndim() == 1);
        }

        auto state = graph.initialize_state();

        THEN("We get the default states we expect") {
            CHECK_THAT(x_ptr->view(state), RangeEquals({0, 1, 2, 3, 4}));
            CHECK_THAT(y_ptr->view(state), RangeEquals({1, 3}));
        }

        WHEN("We do propagation") {
            x_ptr->exchange(state, 1, 2);

            graph.propagate(state, graph.descendants(state, {x_ptr}));

            THEN("The states are as expected") {
                CHECK_THAT(x_ptr->view(state), RangeEquals({0, 2, 1, 3, 4}));
                CHECK_THAT(y_ptr->view(state), RangeEquals({2, 3}));
            }
        }
    }

    GIVEN("x = Set(10); y = x[1::3]") {
        auto x_ptr = graph.emplace_node<SetNode>(10);
        auto y_ptr = graph.emplace_node<BasicIndexingNode>(x_ptr, Slice(1, std::nullopt, 3));

        graph.emplace_node<ArrayValidationNode>(x_ptr);
        graph.emplace_node<ArrayValidationNode>(y_ptr);

        auto state = graph.empty_state();
        x_ptr->initialize_state(state, {});
        graph.initialize_state(state);

        THEN("We get the default states we expect") {
            REQUIRE(x_ptr->size(state) == 0);
            CHECK(y_ptr->size(state) == 0);

            AND_WHEN("We grow the set") {
                x_ptr->assign(state, {0, 1, 2});
                graph.propagate(state);
                graph.commit(state);

                THEN("The state is as expected") {
                    CHECK_THAT(y_ptr->view(state), RangeEquals({1}));
                }

                AND_WHEN("We shrink back to nothing") {
                    x_ptr->assign(state, {});
                    graph.propagate(state);
                    graph.commit(state);

                    THEN("The state is as expected") { CHECK(y_ptr->size(state) == 0); }
                }
            }
        }
    }

    GIVEN("x = Set(10) and several slices and expected outputs") {
        // These tests were generated with this python script
        //
        // ---- BEGIN PYTHON SCRIPT ----
        //
        // import numpy as np
        //
        //
        // slices = [
        //     (None, None, None),
        //     (1, None, None),
        //     (-2, None, None),
        //     (None, 2, None),
        //     (None, -3, None),
        //     (None, None, 2),
        //     (None, None, 3),
        //     (None, None, -2),
        //     (None, None, -3),
        //     (1, 4, None),
        //     (-4, 5, None),
        //     (-4, 8, None),
        //     (3, -3, None),
        //     (7, -5, None),
        //     (-4, -2, None),
        //     (-4, -6, None),
        //     (1, 8, 2),
        //     (1, 8, 3),
        //     (1, 8, -1),
        //     (1, 8, -3),
        //     (8, 1, -2),
        //     (8, 1, -3),
        //     (5, 15, None),
        //     (5, 15, 2),
        //     (-5, 15, 3),
        //     (12, 15, None),
        //     (-15, 5, None),
        //     (-10, 5, 3),
        //     (-15, 5, 3),
        //     (-15, -5, None),
        //     (-15, -12, 3),
        //     (-15, -20, 3),
        // ]
        //
        //
        // def fmt(v):
        //     return "std::nullopt" if v is None else f"{v}"
        //
        //
        // for start, stop, step in slices:
        //     outputs = [np.arange(i)[start:stop:step] for i in range(1, 11)]
        //     # some of these cases are not supported yet, so we expect the constructor to throw
        //     should_throw = step is not None and (step < 0 or (step != 1 and start is not None and
        //     start < 0)) formatted_should_throw = "true" if should_throw else "false" slice_args =
        //     "{" + f"{fmt(start)}, {fmt(stop)}, {fmt(step)}" + "}" formatted_outputs = ",
        //     ".join("{" + ", ".join(str(v) for v in a) + "}" for a in outputs) print("{ " +
        //     slice_args + ", {{ " + formatted_outputs + "}}, " + formatted_should_throw + "},")
        //
        // ---- END PYTHON SCRIPT ----

        std::tuple<Slice, std::array<std::vector<int>, 10>, bool> tests[] = {
                {{std::nullopt, std::nullopt, std::nullopt},
                 {{{0},
                   {0, 1},
                   {0, 1, 2},
                   {0, 1, 2, 3},
                   {0, 1, 2, 3, 4},
                   {0, 1, 2, 3, 4, 5},
                   {0, 1, 2, 3, 4, 5, 6},
                   {0, 1, 2, 3, 4, 5, 6, 7},
                   {0, 1, 2, 3, 4, 5, 6, 7, 8},
                   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}}},
                 false},
                {{1, std::nullopt, std::nullopt},
                 {{{},
                   {1},
                   {1, 2},
                   {1, 2, 3},
                   {1, 2, 3, 4},
                   {1, 2, 3, 4, 5},
                   {1, 2, 3, 4, 5, 6},
                   {1, 2, 3, 4, 5, 6, 7},
                   {1, 2, 3, 4, 5, 6, 7, 8},
                   {1, 2, 3, 4, 5, 6, 7, 8, 9}}},
                 false},
                {{-2, std::nullopt, std::nullopt},
                 {{{0}, {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 8}, {8, 9}}},
                 false},
                {{std::nullopt, 2, std::nullopt},
                 {{{0}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}}},
                 false},
                {{std::nullopt, -3, std::nullopt},
                 {{{},
                   {},
                   {},
                   {0},
                   {0, 1},
                   {0, 1, 2},
                   {0, 1, 2, 3},
                   {0, 1, 2, 3, 4},
                   {0, 1, 2, 3, 4, 5},
                   {0, 1, 2, 3, 4, 5, 6}}},
                 false},
                {{std::nullopt, std::nullopt, 2},
                 {{{0},
                   {0},
                   {0, 2},
                   {0, 2},
                   {0, 2, 4},
                   {0, 2, 4},
                   {0, 2, 4, 6},
                   {0, 2, 4, 6},
                   {0, 2, 4, 6, 8},
                   {0, 2, 4, 6, 8}}},
                 false},
                {{std::nullopt, std::nullopt, 3},
                 {{{0},
                   {0},
                   {0},
                   {0, 3},
                   {0, 3},
                   {0, 3},
                   {0, 3, 6},
                   {0, 3, 6},
                   {0, 3, 6},
                   {0, 3, 6, 9}}},
                 false},
                {{std::nullopt, std::nullopt, -2},
                 {{{0},
                   {1},
                   {2, 0},
                   {3, 1},
                   {4, 2, 0},
                   {5, 3, 1},
                   {6, 4, 2, 0},
                   {7, 5, 3, 1},
                   {8, 6, 4, 2, 0},
                   {9, 7, 5, 3, 1}}},
                 true},
                {{std::nullopt, std::nullopt, -3},
                 {{{0},
                   {1},
                   {2},
                   {3, 0},
                   {4, 1},
                   {5, 2},
                   {6, 3, 0},
                   {7, 4, 1},
                   {8, 5, 2},
                   {9, 6, 3, 0}}},
                 true},
                {{1, 4, std::nullopt},
                 {{{},
                   {1},
                   {1, 2},
                   {1, 2, 3},
                   {1, 2, 3},
                   {1, 2, 3},
                   {1, 2, 3},
                   {1, 2, 3},
                   {1, 2, 3},
                   {1, 2, 3}}},
                 false},
                {{-4, 5, std::nullopt},
                 {{{0},
                   {0, 1},
                   {0, 1, 2},
                   {0, 1, 2, 3},
                   {1, 2, 3, 4},
                   {2, 3, 4},
                   {3, 4},
                   {4},
                   {},
                   {}}},
                 false},
                {{-4, 8, std::nullopt},
                 {{{0},
                   {0, 1},
                   {0, 1, 2},
                   {0, 1, 2, 3},
                   {1, 2, 3, 4},
                   {2, 3, 4, 5},
                   {3, 4, 5, 6},
                   {4, 5, 6, 7},
                   {5, 6, 7},
                   {6, 7}}},
                 false},
                {{3, -3, std::nullopt},
                 {{{}, {}, {}, {}, {}, {}, {3}, {3, 4}, {3, 4, 5}, {3, 4, 5, 6}}},
                 false},
                {{7, -5, std::nullopt}, {{{}, {}, {}, {}, {}, {}, {}, {}, {}, {}}}, false},
                {{-4, -2, std::nullopt},
                 {{{}, {}, {0}, {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}}},
                 false},
                {{-4, -6, std::nullopt}, {{{}, {}, {}, {}, {}, {}, {}, {}, {}, {}}}, false},
                {{1, 8, 2},
                 {{{},
                   {1},
                   {1},
                   {1, 3},
                   {1, 3},
                   {1, 3, 5},
                   {1, 3, 5},
                   {1, 3, 5, 7},
                   {1, 3, 5, 7},
                   {1, 3, 5, 7}}},
                 false},
                {{1, 8, 3},
                 {{{}, {1}, {1}, {1}, {1, 4}, {1, 4}, {1, 4}, {1, 4, 7}, {1, 4, 7}, {1, 4, 7}}},
                 false},
                {{1, 8, -1}, {{{}, {}, {}, {}, {}, {}, {}, {}, {}, {}}}, true},
                {{1, 8, -3}, {{{}, {}, {}, {}, {}, {}, {}, {}, {}, {}}}, true},
                {{8, 1, -2},
                 {{{},
                   {},
                   {2},
                   {3},
                   {4, 2},
                   {5, 3},
                   {6, 4, 2},
                   {7, 5, 3},
                   {8, 6, 4, 2},
                   {8, 6, 4, 2}}},
                 true},
                {{8, 1, -3},
                 {{{}, {}, {2}, {3}, {4}, {5, 2}, {6, 3}, {7, 4}, {8, 5, 2}, {8, 5, 2}}},
                 true},
                {{5, 15, std::nullopt},
                 {{{}, {}, {}, {}, {}, {5}, {5, 6}, {5, 6, 7}, {5, 6, 7, 8}, {5, 6, 7, 8, 9}}},
                 false},
                {{5, 15, 2}, {{{}, {}, {}, {}, {}, {5}, {5}, {5, 7}, {5, 7}, {5, 7, 9}}}, false},
                {{-5, 15, 3},
                 {{{0}, {0}, {0}, {0, 3}, {0, 3}, {1, 4}, {2, 5}, {3, 6}, {4, 7}, {5, 8}}},
                 true},
                {{12, 15, std::nullopt}, {{{}, {}, {}, {}, {}, {}, {}, {}, {}, {}}}, false},
                {{-15, 5, std::nullopt},
                 {{{0},
                   {0, 1},
                   {0, 1, 2},
                   {0, 1, 2, 3},
                   {0, 1, 2, 3, 4},
                   {0, 1, 2, 3, 4},
                   {0, 1, 2, 3, 4},
                   {0, 1, 2, 3, 4},
                   {0, 1, 2, 3, 4},
                   {0, 1, 2, 3, 4}}},
                 false},
                {{-10, 5, 3},
                 {{{0}, {0}, {0}, {0, 3}, {0, 3}, {0, 3}, {0, 3}, {0, 3}, {0, 3}, {0, 3}}},
                 true},
                {{-15, 5, 3},
                 {{{0}, {0}, {0}, {0, 3}, {0, 3}, {0, 3}, {0, 3}, {0, 3}, {0, 3}, {0, 3}}},
                 true},
                {{-15, -5, std::nullopt},
                 {{{}, {}, {}, {}, {}, {0}, {0, 1}, {0, 1, 2}, {0, 1, 2, 3}, {0, 1, 2, 3, 4}}},
                 false},
                {{-15, -12, 3}, {{{}, {}, {}, {}, {}, {}, {}, {}, {}, {}}}, true},
                {{-15, -20, 3}, {{{}, {}, {}, {}, {}, {}, {}, {}, {}, {}}}, true},
        };

        for (const auto& [slice, expected_outputs, should_throw] : tests) {
            auto graph = Graph();
            auto x_ptr = graph.emplace_node<SetNode>(10);

            if (should_throw) {
                REQUIRE_THROWS_AS(graph.emplace_node<BasicIndexingNode>(x_ptr, slice),
                                  std::invalid_argument);
                continue;
            }

            auto y_ptr = graph.emplace_node<BasicIndexingNode>(x_ptr, slice);
            graph.emplace_node<ArrayValidationNode>(y_ptr);

            auto state = graph.empty_state();
            x_ptr->initialize_state(state, {});
            graph.initialize_state(state);

            REQUIRE(x_ptr->size(state) == 0);
            CHECK(y_ptr->size(state) == 0);

            for (const std::vector<int>& expected_output : expected_outputs) {
                x_ptr->grow(state);
                graph.propagate(state);
                graph.commit(state);

                INFO("checking equivalent to np.arange(" << x_ptr->size(state) << ")["
                                                         << slice.start << ":" << slice.stop << ":"
                                                         << slice.step << "] (after growing)");
                CHECK(y_ptr->size(state) == static_cast<ssize_t>(expected_output.size()));
                CHECK_THAT(y_ptr->view(state), RangeEquals(expected_output));
            }

            REQUIRE(x_ptr->size(state) == 10);

            for (const std::vector<int>& expected_output :
                 expected_outputs | std::views::reverse | std::views::drop(1)) {
                x_ptr->shrink(state);
                graph.propagate(state);
                graph.commit(state);

                INFO("checking equivalent to np.arange(" << x_ptr->size(state) << ")["
                                                         << slice.start << ":" << slice.stop << ":"
                                                         << slice.step << "] (after shrinking)");
                CHECK(y_ptr->size(state) == static_cast<ssize_t>(expected_output.size()));
                CHECK_THAT(y_ptr->view(state), RangeEquals(expected_output));
            }

            x_ptr->shrink(state);
            graph.propagate(state);
            graph.commit(state);

            REQUIRE(x_ptr->size(state) == 0);
            CHECK(y_ptr->size(state) == 0);
        }
    }

    GIVEN("Dynamic(n, 5, 2)[5:7, 1:, :]") {
        auto dynamic_ptr = graph.emplace_node<DynamicArrayTestingNode>(
                std::initializer_list<ssize_t>{-1, 5, 2});

        // the resuling array will either be (0, 4, 2), (1, 4, 2), or (2, 4, 2) array.
        auto y_ptr = graph.emplace_node<BasicIndexingNode>(dynamic_ptr, Slice(5, 7),
                                                           Slice(1, std::nullopt), Slice());

        graph.emplace_node<ArrayValidationNode>(y_ptr);

        auto state = graph.initialize_state();

        WHEN("We grow the dynamic node to size [5, 5, 2]") {
            std::vector<double> values;
            for (auto v : std::ranges::iota_view(0, 50)) {
                values.emplace_back(v);
            }

            dynamic_ptr->grow(state, values);
            graph.propagate(state);
            graph.commit(state);

            REQUIRE(dynamic_ptr->size(state) == 50);

            THEN("The slices is still empty") { CHECK(y_ptr->size(state) == 0); }

            AND_WHEN("We grow the dynamic node to size [7, 5, 2]") {
                std::vector<double> values;
                for (auto v : std::ranges::iota_view(50, 70)) {
                    values.emplace_back(v);
                }

                dynamic_ptr->grow(state, values);
                graph.propagate(state);
                graph.commit(state);

                REQUIRE(dynamic_ptr->size(state) == 70);

                THEN("The slice is correct") {
                    CHECK(y_ptr->size(state) == 16);
                    CHECK_THAT(y_ptr->view(state), RangeEquals({52, 53, 54, 55, 56, 57, 58, 59, 62,
                                                                63, 64, 65, 66, 67, 68, 69}));
                }

                AND_WHEN("We shrink the dynamic node to size [6, 5, 2]") {
                    dynamic_ptr->shrink(state);
                    graph.propagate(state);
                    graph.commit(state);

                    REQUIRE(dynamic_ptr->size(state) == 60);

                    THEN("The slice is correct") {
                        CHECK(y_ptr->size(state) == 8);
                        CHECK_THAT(y_ptr->view(state),
                                   RangeEquals({52, 53, 54, 55, 56, 57, 58, 59}));
                    }
                }
            }
        }
    }

    GIVEN("Dynamic(n, 5, 2)[5:8:2, 1:, :]") {
        auto dynamic_ptr = graph.emplace_node<DynamicArrayTestingNode>(
                std::initializer_list<ssize_t>{-1, 5, 2});

        // the resuling array will either be (0, 4, 2), or (1, 4, 2) array.
        auto y_ptr = graph.emplace_node<BasicIndexingNode>(dynamic_ptr, Slice(5, 8, 2),
                                                           Slice(1, std::nullopt), Slice());

        graph.emplace_node<ArrayValidationNode>(y_ptr);

        auto state = graph.initialize_state();

        WHEN("We grow the dynamic node to size [5, 5, 2]") {
            std::vector<double> values;
            for (auto v : std::ranges::iota_view(0, 50)) {
                values.emplace_back(v);
            }

            dynamic_ptr->grow(state, values);
            graph.propagate(state);
            graph.commit(state);

            REQUIRE(dynamic_ptr->size(state) == 50);

            THEN("The slices is still empty") { CHECK(y_ptr->size(state) == 0); }

            AND_WHEN("We grow the dynamic node to size [8, 5, 2]") {
                std::vector<double> values;
                for (auto v : std::ranges::iota_view(50, 80)) {
                    values.emplace_back(v);
                }

                dynamic_ptr->grow(state, values);
                graph.propagate(state);
                graph.commit(state);

                REQUIRE(dynamic_ptr->size(state) == 80);

                THEN("The slice is correct") {
                    CHECK(y_ptr->size(state) == 16);
                    CHECK_THAT(y_ptr->view(state), RangeEquals({52, 53, 54, 55, 56, 57, 58, 59, 72,
                                                                73, 74, 75, 76, 77, 78, 79}));
                }

                AND_WHEN("We shrink the dynamic node to size [7, 5, 2]") {
                    dynamic_ptr->shrink(state);
                    graph.propagate(state);
                    graph.commit(state);

                    REQUIRE(dynamic_ptr->size(state) == 70);

                    THEN("The slice is correct") {
                        CHECK(y_ptr->size(state) == 8);
                        CHECK_THAT(y_ptr->view(state),
                                   RangeEquals({52, 53, 54, 55, 56, 57, 58, 59}));
                    }
                }
            }
        }
    }

    GIVEN("x = Dynamic(n, 2) and several slices on the first dim and expected outputs") {
        // These tests were generated with this python script
        //
        // ---- BEGIN PYTHON SCRIPT ----
        //
        // import numpy as np
        // slices = [
        //     (None, None, None),
        //     (1, None, None),
        //     (-2, None, None),
        //     (None, 2, None),
        //     (None, -3, None),
        //     (None, None, 2),
        //     (None, None, 3),
        //     (None, None, -2),
        //     (None, None, -3),
        //     (1, 4, None),
        //     (-4, 5, None),
        //     (-4, 8, None),
        //     (3, -3, None),
        //     (7, -5, None),
        //     (-4, -2, None),
        //     (-4, -6, None),
        //     (1, 8, 2),
        //     (1, 8, 3),
        //     (1, 8, -1),
        //     (1, 8, -3),
        //     (8, 1, -2),
        //     (8, 1, -3),
        //     (5, 15, None),
        //     (5, 15, 2),
        //     (-5, 15, 3),
        //     (12, 15, None),
        //     (-15, 5, None),
        //     (-10, 5, 3),
        //     (-15, 5, 3),
        //     (-15, -5, None),
        //     (-15, -12, 3),
        //     (-15, -20, 3),
        // ]
        //
        //
        // def fmt(v):
        //     return "std::nullopt" if v is None else f"{v}"
        //
        //
        // for start, stop, step in slices:
        //     outputs = [np.arange(i * 2).reshape(i, 2)[start:stop:step, :] for i in range(1, 11)]
        //     # some of these cases are not supported yet, so we expect the constructor to throw
        //     should_throw = step is not None and (step < 0 or (step != 1 and start is not None and
        //     start < 0)) formatted_should_throw = "true" if should_throw else "false" slice_args =
        //     "{" + f"{fmt(start)}, {fmt(stop)}, {fmt(step)}" + "}" formatted_outputs = ",
        //     ".join("{" + ", ".join(str(v) for v in a.flatten()) + "}" for a in outputs) print("{
        //     " + slice_args + ", {{ " + formatted_outputs + "}}, " + formatted_should_throw +
        //     "},")
        //
        // ---- END PYTHON SCRIPT ----

        const ssize_t dim0_max = 10;
        const ssize_t dim1 = 2;

        std::tuple<Slice, std::array<std::vector<int>, dim0_max>, bool> tests[] = {
                {{std::nullopt, std::nullopt, std::nullopt},
                 {{{0, 1},
                   {0, 1, 2, 3},
                   {0, 1, 2, 3, 4, 5},
                   {0, 1, 2, 3, 4, 5, 6, 7},
                   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13},
                   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
                   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17},
                   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}}},
                 false},
                {{1, std::nullopt, std::nullopt},
                 {{{},
                   {2, 3},
                   {2, 3, 4, 5},
                   {2, 3, 4, 5, 6, 7},
                   {2, 3, 4, 5, 6, 7, 8, 9},
                   {2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                   {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13},
                   {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
                   {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17},
                   {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}}},
                 false},
                {{-2, std::nullopt, std::nullopt},
                 {{{0, 1},
                   {0, 1, 2, 3},
                   {2, 3, 4, 5},
                   {4, 5, 6, 7},
                   {6, 7, 8, 9},
                   {8, 9, 10, 11},
                   {10, 11, 12, 13},
                   {12, 13, 14, 15},
                   {14, 15, 16, 17},
                   {16, 17, 18, 19}}},
                 false},
                {{std::nullopt, 2, std::nullopt},
                 {{{0, 1},
                   {0, 1, 2, 3},
                   {0, 1, 2, 3},
                   {0, 1, 2, 3},
                   {0, 1, 2, 3},
                   {0, 1, 2, 3},
                   {0, 1, 2, 3},
                   {0, 1, 2, 3},
                   {0, 1, 2, 3},
                   {0, 1, 2, 3}}},
                 false},
                {{std::nullopt, -3, std::nullopt},
                 {{{},
                   {},
                   {},
                   {0, 1},
                   {0, 1, 2, 3},
                   {0, 1, 2, 3, 4, 5},
                   {0, 1, 2, 3, 4, 5, 6, 7},
                   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}}},
                 false},
                {{std::nullopt, std::nullopt, 2},
                 {{{0, 1},
                   {0, 1},
                   {0, 1, 4, 5},
                   {0, 1, 4, 5},
                   {0, 1, 4, 5, 8, 9},
                   {0, 1, 4, 5, 8, 9},
                   {0, 1, 4, 5, 8, 9, 12, 13},
                   {0, 1, 4, 5, 8, 9, 12, 13},
                   {0, 1, 4, 5, 8, 9, 12, 13, 16, 17},
                   {0, 1, 4, 5, 8, 9, 12, 13, 16, 17}}},
                 false},
                {{std::nullopt, std::nullopt, 3},
                 {{{0, 1},
                   {0, 1},
                   {0, 1},
                   {0, 1, 6, 7},
                   {0, 1, 6, 7},
                   {0, 1, 6, 7},
                   {0, 1, 6, 7, 12, 13},
                   {0, 1, 6, 7, 12, 13},
                   {0, 1, 6, 7, 12, 13},
                   {0, 1, 6, 7, 12, 13, 18, 19}}},
                 false},
                {{std::nullopt, std::nullopt, -2},
                 {{{0, 1},
                   {2, 3},
                   {4, 5, 0, 1},
                   {6, 7, 2, 3},
                   {8, 9, 4, 5, 0, 1},
                   {10, 11, 6, 7, 2, 3},
                   {12, 13, 8, 9, 4, 5, 0, 1},
                   {14, 15, 10, 11, 6, 7, 2, 3},
                   {16, 17, 12, 13, 8, 9, 4, 5, 0, 1},
                   {18, 19, 14, 15, 10, 11, 6, 7, 2, 3}}},
                 true},
                {{std::nullopt, std::nullopt, -3},
                 {{{0, 1},
                   {2, 3},
                   {4, 5},
                   {6, 7, 0, 1},
                   {8, 9, 2, 3},
                   {10, 11, 4, 5},
                   {12, 13, 6, 7, 0, 1},
                   {14, 15, 8, 9, 2, 3},
                   {16, 17, 10, 11, 4, 5},
                   {18, 19, 12, 13, 6, 7, 0, 1}}},
                 true},
                {{1, 4, std::nullopt},
                 {{{},
                   {2, 3},
                   {2, 3, 4, 5},
                   {2, 3, 4, 5, 6, 7},
                   {2, 3, 4, 5, 6, 7},
                   {2, 3, 4, 5, 6, 7},
                   {2, 3, 4, 5, 6, 7},
                   {2, 3, 4, 5, 6, 7},
                   {2, 3, 4, 5, 6, 7},
                   {2, 3, 4, 5, 6, 7}}},
                 false},
                {{-4, 5, std::nullopt},
                 {{{0, 1},
                   {0, 1, 2, 3},
                   {0, 1, 2, 3, 4, 5},
                   {0, 1, 2, 3, 4, 5, 6, 7},
                   {2, 3, 4, 5, 6, 7, 8, 9},
                   {4, 5, 6, 7, 8, 9},
                   {6, 7, 8, 9},
                   {8, 9},
                   {},
                   {}}},
                 false},
                {{-4, 8, std::nullopt},
                 {{{0, 1},
                   {0, 1, 2, 3},
                   {0, 1, 2, 3, 4, 5},
                   {0, 1, 2, 3, 4, 5, 6, 7},
                   {2, 3, 4, 5, 6, 7, 8, 9},
                   {4, 5, 6, 7, 8, 9, 10, 11},
                   {6, 7, 8, 9, 10, 11, 12, 13},
                   {8, 9, 10, 11, 12, 13, 14, 15},
                   {10, 11, 12, 13, 14, 15},
                   {12, 13, 14, 15}}},
                 false},
                {{3, -3, std::nullopt},
                 {{{},
                   {},
                   {},
                   {},
                   {},
                   {},
                   {6, 7},
                   {6, 7, 8, 9},
                   {6, 7, 8, 9, 10, 11},
                   {6, 7, 8, 9, 10, 11, 12, 13}}},
                 false},
                {{7, -5, std::nullopt}, {{{}, {}, {}, {}, {}, {}, {}, {}, {}, {}}}, false},
                {{-4, -2, std::nullopt},
                 {{{},
                   {},
                   {0, 1},
                   {0, 1, 2, 3},
                   {2, 3, 4, 5},
                   {4, 5, 6, 7},
                   {6, 7, 8, 9},
                   {8, 9, 10, 11},
                   {10, 11, 12, 13},
                   {12, 13, 14, 15}}},
                 false},
                {{-4, -6, std::nullopt}, {{{}, {}, {}, {}, {}, {}, {}, {}, {}, {}}}, false},
                {{1, 8, 2},
                 {{{},
                   {2, 3},
                   {2, 3},
                   {2, 3, 6, 7},
                   {2, 3, 6, 7},
                   {2, 3, 6, 7, 10, 11},
                   {2, 3, 6, 7, 10, 11},
                   {2, 3, 6, 7, 10, 11, 14, 15},
                   {2, 3, 6, 7, 10, 11, 14, 15},
                   {2, 3, 6, 7, 10, 11, 14, 15}}},
                 false},
                {{1, 8, 3},
                 {{{},
                   {2, 3},
                   {2, 3},
                   {2, 3},
                   {2, 3, 8, 9},
                   {2, 3, 8, 9},
                   {2, 3, 8, 9},
                   {2, 3, 8, 9, 14, 15},
                   {2, 3, 8, 9, 14, 15},
                   {2, 3, 8, 9, 14, 15}}},
                 false},
                {{1, 8, -1}, {{{}, {}, {}, {}, {}, {}, {}, {}, {}, {}}}, true},
                {{1, 8, -3}, {{{}, {}, {}, {}, {}, {}, {}, {}, {}, {}}}, true},
                {{8, 1, -2},
                 {{{},
                   {},
                   {4, 5},
                   {6, 7},
                   {8, 9, 4, 5},
                   {10, 11, 6, 7},
                   {12, 13, 8, 9, 4, 5},
                   {14, 15, 10, 11, 6, 7},
                   {16, 17, 12, 13, 8, 9, 4, 5},
                   {16, 17, 12, 13, 8, 9, 4, 5}}},
                 true},
                {{8, 1, -3},
                 {{{},
                   {},
                   {4, 5},
                   {6, 7},
                   {8, 9},
                   {10, 11, 4, 5},
                   {12, 13, 6, 7},
                   {14, 15, 8, 9},
                   {16, 17, 10, 11, 4, 5},
                   {16, 17, 10, 11, 4, 5}}},
                 true},
                {{5, 15, std::nullopt},
                 {{{},
                   {},
                   {},
                   {},
                   {},
                   {10, 11},
                   {10, 11, 12, 13},
                   {10, 11, 12, 13, 14, 15},
                   {10, 11, 12, 13, 14, 15, 16, 17},
                   {10, 11, 12, 13, 14, 15, 16, 17, 18, 19}}},
                 false},
                {{5, 15, 2},
                 {{{},
                   {},
                   {},
                   {},
                   {},
                   {10, 11},
                   {10, 11},
                   {10, 11, 14, 15},
                   {10, 11, 14, 15},
                   {10, 11, 14, 15, 18, 19}}},
                 false},
                {{-5, 15, 3},
                 {{{0, 1},
                   {0, 1},
                   {0, 1},
                   {0, 1, 6, 7},
                   {0, 1, 6, 7},
                   {2, 3, 8, 9},
                   {4, 5, 10, 11},
                   {6, 7, 12, 13},
                   {8, 9, 14, 15},
                   {10, 11, 16, 17}}},
                 true},
                {{12, 15, std::nullopt}, {{{}, {}, {}, {}, {}, {}, {}, {}, {}, {}}}, false},
                {{-15, 5, std::nullopt},
                 {{{0, 1},
                   {0, 1, 2, 3},
                   {0, 1, 2, 3, 4, 5},
                   {0, 1, 2, 3, 4, 5, 6, 7},
                   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}}},
                 false},
                {{-10, 5, 3},
                 {{{0, 1},
                   {0, 1},
                   {0, 1},
                   {0, 1, 6, 7},
                   {0, 1, 6, 7},
                   {0, 1, 6, 7},
                   {0, 1, 6, 7},
                   {0, 1, 6, 7},
                   {0, 1, 6, 7},
                   {0, 1, 6, 7}}},
                 true},
                {{-15, 5, 3},
                 {{{0, 1},
                   {0, 1},
                   {0, 1},
                   {0, 1, 6, 7},
                   {0, 1, 6, 7},
                   {0, 1, 6, 7},
                   {0, 1, 6, 7},
                   {0, 1, 6, 7},
                   {0, 1, 6, 7},
                   {0, 1, 6, 7}}},
                 true},
                {{-15, -5, std::nullopt},
                 {{{},
                   {},
                   {},
                   {},
                   {},
                   {0, 1},
                   {0, 1, 2, 3},
                   {0, 1, 2, 3, 4, 5},
                   {0, 1, 2, 3, 4, 5, 6, 7},
                   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}}},
                 false},
                {{-15, -12, 3}, {{{}, {}, {}, {}, {}, {}, {}, {}, {}, {}}}, true},
                {{-15, -20, 3}, {{{}, {}, {}, {}, {}, {}, {}, {}, {}, {}}}, true},
        };

        for (const auto& [slice, expected_outputs, should_throw] : tests) {
            auto graph = Graph();
            auto x_ptr = graph.emplace_node<DynamicArrayTestingNode>(
                    std::initializer_list<ssize_t>{-1, dim1});

            if (should_throw) {
                REQUIRE_THROWS_AS(graph.emplace_node<BasicIndexingNode>(x_ptr, slice, Slice()),
                                  std::invalid_argument);
                continue;
            }

            auto y_ptr = graph.emplace_node<BasicIndexingNode>(x_ptr, slice, Slice());
            graph.emplace_node<ArrayValidationNode>(y_ptr);

            auto state = graph.empty_state();
            x_ptr->initialize_state(state, {});
            graph.initialize_state(state);

            REQUIRE(x_ptr->size(state) == 0);
            CHECK(y_ptr->size(state) == 0);

            for (const std::vector<int>& expected_output : expected_outputs) {
                std::vector<double> values;
                for (auto v :
                     std::ranges::iota_view(x_ptr->size(state), x_ptr->size(state) + dim1)) {
                    values.emplace_back(v);
                }
                x_ptr->grow(state, values);
                graph.propagate(state);
                graph.commit(state);

                INFO("checking equivalent to np.arange("
                     << x_ptr->size(state) << ").reshape(-1, " << dim1 << ")[" << slice.start << ":"
                     << slice.stop << ":" << slice.step << "] (after growing)");
                CHECK(y_ptr->size(state) == static_cast<ssize_t>(expected_output.size()));
                CHECK_THAT(y_ptr->view(state), RangeEquals(expected_output));
            }

            REQUIRE(x_ptr->size(state) == dim0_max * dim1);

            for (const std::vector<int>& expected_output :
                 expected_outputs | std::views::reverse | std::views::drop(1)) {
                x_ptr->shrink(state);
                graph.propagate(state);
                graph.commit(state);

                INFO("checking equivalent to np.arange("
                     << x_ptr->size(state) << ").reshape(-1, " << dim1 << ")[" << slice.start << ":"
                     << slice.stop << ":" << slice.step << "] (after shrinking)");
                CHECK(y_ptr->size(state) == static_cast<ssize_t>(expected_output.size()));
                CHECK_THAT(y_ptr->view(state), RangeEquals(expected_output));
            }

            x_ptr->shrink(state);
            graph.propagate(state);
            graph.commit(state);

            REQUIRE(x_ptr->size(state) == 0);
            CHECK(y_ptr->size(state) == 0);
        }
    }

    GIVEN("x = Dynamic(n, 2)[slice, 1:] for several slices on the first dim and expected outputs") {
        // These tests were generated with this python script
        //
        // ---- BEGIN PYTHON SCRIPT ----
        //
        // import numpy as np
        //
        //
        // slices = [
        //     (None, None, None),
        //     (1, None, None),
        //     (-2, None, None),
        //     (None, 2, None),
        //     (None, -3, None),
        //     (None, None, 2),
        //     (None, None, 3),
        //     (None, None, -2),
        //     (None, None, -3),
        //     (1, 4, None),
        //     (-4, 5, None),
        //     (-4, 8, None),
        //     (3, -3, None),
        //     (7, -5, None),
        //     (-4, -2, None),
        //     (-4, -6, None),
        //     (1, 8, 2),
        //     (1, 8, 3),
        //     (1, 8, -1),
        //     (1, 8, -3),
        //     (8, 1, -2),
        //     (8, 1, -3),
        //     (5, 15, None),
        //     (5, 15, 2),
        //     (-5, 15, 3),
        //     (12, 15, None),
        //     (-15, 5, None),
        //     (-10, 5, 3),
        //     (-15, 5, 3),
        //     (-15, -5, None),
        //     (-15, -12, 3),
        //     (-15, -20, 3),
        // ]
        //
        //
        // def fmt(v):
        //     return "std::nullopt" if v is None else f"{v}"
        //
        //
        // for start, stop, step in slices:
        //     outputs = [np.arange(i * 2).reshape(i, 2)[start:stop:step, 1:] for i in range(1, 11)]
        //     # some of these cases are not supported yet, so we expect the constructor to throw
        //     should_throw = step is not None and (step < 0 or (step != 1 and start is not None and
        //     start < 0)) should_throw |= (start or 0) < 0 or (stop or 0) < 0
        //     formatted_should_throw = "true" if should_throw else "false"
        //     slice_args = "{" + f"{fmt(start)}, {fmt(stop)}, {fmt(step)}" + "}"
        //     formatted_outputs = ", ".join("{" + ", ".join(str(v) for v in a.flatten()) + "}" for
        //     a in outputs) print("{ " + slice_args + ", {{ " + formatted_outputs + "}}, " +
        //     formatted_should_throw + "},")
        //
        // ---- END PYTHON SCRIPT ----

        const ssize_t dim0_max = 10;
        const ssize_t dim1 = 2;

        std::tuple<Slice, std::array<std::vector<int>, dim0_max>, bool> tests[] = {
                {{std::nullopt, std::nullopt, std::nullopt},
                 {{{1},
                   {1, 3},
                   {1, 3, 5},
                   {1, 3, 5, 7},
                   {1, 3, 5, 7, 9},
                   {1, 3, 5, 7, 9, 11},
                   {1, 3, 5, 7, 9, 11, 13},
                   {1, 3, 5, 7, 9, 11, 13, 15},
                   {1, 3, 5, 7, 9, 11, 13, 15, 17},
                   {1, 3, 5, 7, 9, 11, 13, 15, 17, 19}}},
                 false},
                {{1, std::nullopt, std::nullopt},
                 {{{},
                   {3},
                   {3, 5},
                   {3, 5, 7},
                   {3, 5, 7, 9},
                   {3, 5, 7, 9, 11},
                   {3, 5, 7, 9, 11, 13},
                   {3, 5, 7, 9, 11, 13, 15},
                   {3, 5, 7, 9, 11, 13, 15, 17},
                   {3, 5, 7, 9, 11, 13, 15, 17, 19}}},
                 false},
                {{-2, std::nullopt, std::nullopt},
                 {{{1},
                   {1, 3},
                   {3, 5},
                   {5, 7},
                   {7, 9},
                   {9, 11},
                   {11, 13},
                   {13, 15},
                   {15, 17},
                   {17, 19}}},
                 true},
                {{std::nullopt, 2, std::nullopt},
                 {{{1}, {1, 3}, {1, 3}, {1, 3}, {1, 3}, {1, 3}, {1, 3}, {1, 3}, {1, 3}, {1, 3}}},
                 false},
                {{std::nullopt, -3, std::nullopt},
                 {{{},
                   {},
                   {},
                   {1},
                   {1, 3},
                   {1, 3, 5},
                   {1, 3, 5, 7},
                   {1, 3, 5, 7, 9},
                   {1, 3, 5, 7, 9, 11},
                   {1, 3, 5, 7, 9, 11, 13}}},
                 true},
                {{std::nullopt, std::nullopt, 2},
                 {{{1},
                   {1},
                   {1, 5},
                   {1, 5},
                   {1, 5, 9},
                   {1, 5, 9},
                   {1, 5, 9, 13},
                   {1, 5, 9, 13},
                   {1, 5, 9, 13, 17},
                   {1, 5, 9, 13, 17}}},
                 false},
                {{std::nullopt, std::nullopt, 3},
                 {{{1},
                   {1},
                   {1},
                   {1, 7},
                   {1, 7},
                   {1, 7},
                   {1, 7, 13},
                   {1, 7, 13},
                   {1, 7, 13},
                   {1, 7, 13, 19}}},
                 false},
                {{std::nullopt, std::nullopt, -2},
                 {{{1},
                   {3},
                   {5, 1},
                   {7, 3},
                   {9, 5, 1},
                   {11, 7, 3},
                   {13, 9, 5, 1},
                   {15, 11, 7, 3},
                   {17, 13, 9, 5, 1},
                   {19, 15, 11, 7, 3}}},
                 true},
                {{std::nullopt, std::nullopt, -3},
                 {{{1},
                   {3},
                   {5},
                   {7, 1},
                   {9, 3},
                   {11, 5},
                   {13, 7, 1},
                   {15, 9, 3},
                   {17, 11, 5},
                   {19, 13, 7, 1}}},
                 true},
                {{1, 4, std::nullopt},
                 {{{},
                   {3},
                   {3, 5},
                   {3, 5, 7},
                   {3, 5, 7},
                   {3, 5, 7},
                   {3, 5, 7},
                   {3, 5, 7},
                   {3, 5, 7},
                   {3, 5, 7}}},
                 false},
                {{-4, 5, std::nullopt},
                 {{{1},
                   {1, 3},
                   {1, 3, 5},
                   {1, 3, 5, 7},
                   {3, 5, 7, 9},
                   {5, 7, 9},
                   {7, 9},
                   {9},
                   {},
                   {}}},
                 true},
                {{-4, 8, std::nullopt},
                 {{{1},
                   {1, 3},
                   {1, 3, 5},
                   {1, 3, 5, 7},
                   {3, 5, 7, 9},
                   {5, 7, 9, 11},
                   {7, 9, 11, 13},
                   {9, 11, 13, 15},
                   {11, 13, 15},
                   {13, 15}}},
                 true},
                {{3, -3, std::nullopt},
                 {{{}, {}, {}, {}, {}, {}, {7}, {7, 9}, {7, 9, 11}, {7, 9, 11, 13}}},
                 true},
                {{7, -5, std::nullopt}, {{{}, {}, {}, {}, {}, {}, {}, {}, {}, {}}}, true},
                {{-4, -2, std::nullopt},
                 {{{}, {}, {1}, {1, 3}, {3, 5}, {5, 7}, {7, 9}, {9, 11}, {11, 13}, {13, 15}}},
                 true},
                {{-4, -6, std::nullopt}, {{{}, {}, {}, {}, {}, {}, {}, {}, {}, {}}}, true},
                {{1, 8, 2},
                 {{{},
                   {3},
                   {3},
                   {3, 7},
                   {3, 7},
                   {3, 7, 11},
                   {3, 7, 11},
                   {3, 7, 11, 15},
                   {3, 7, 11, 15},
                   {3, 7, 11, 15}}},
                 false},
                {{1, 8, 3},
                 {{{}, {3}, {3}, {3}, {3, 9}, {3, 9}, {3, 9}, {3, 9, 15}, {3, 9, 15}, {3, 9, 15}}},
                 false},
                {{1, 8, -1}, {{{}, {}, {}, {}, {}, {}, {}, {}, {}, {}}}, true},
                {{1, 8, -3}, {{{}, {}, {}, {}, {}, {}, {}, {}, {}, {}}}, true},
                {{8, 1, -2},
                 {{{},
                   {},
                   {5},
                   {7},
                   {9, 5},
                   {11, 7},
                   {13, 9, 5},
                   {15, 11, 7},
                   {17, 13, 9, 5},
                   {17, 13, 9, 5}}},
                 true},
                {{8, 1, -3},
                 {{{}, {}, {5}, {7}, {9}, {11, 5}, {13, 7}, {15, 9}, {17, 11, 5}, {17, 11, 5}}},
                 true},
                {{5, 15, std::nullopt},
                 {{{},
                   {},
                   {},
                   {},
                   {},
                   {11},
                   {11, 13},
                   {11, 13, 15},
                   {11, 13, 15, 17},
                   {11, 13, 15, 17, 19}}},
                 false},
                {{5, 15, 2},
                 {{{}, {}, {}, {}, {}, {11}, {11}, {11, 15}, {11, 15}, {11, 15, 19}}},
                 false},
                {{-5, 15, 3},
                 {{{1}, {1}, {1}, {1, 7}, {1, 7}, {3, 9}, {5, 11}, {7, 13}, {9, 15}, {11, 17}}},
                 true},
                {{12, 15, std::nullopt}, {{{}, {}, {}, {}, {}, {}, {}, {}, {}, {}}}, false},
                {{-15, 5, std::nullopt},
                 {{{1},
                   {1, 3},
                   {1, 3, 5},
                   {1, 3, 5, 7},
                   {1, 3, 5, 7, 9},
                   {1, 3, 5, 7, 9},
                   {1, 3, 5, 7, 9},
                   {1, 3, 5, 7, 9},
                   {1, 3, 5, 7, 9},
                   {1, 3, 5, 7, 9}}},
                 true},
                {{-10, 5, 3},
                 {{{1}, {1}, {1}, {1, 7}, {1, 7}, {1, 7}, {1, 7}, {1, 7}, {1, 7}, {1, 7}}},
                 true},
                {{-15, 5, 3},
                 {{{1}, {1}, {1}, {1, 7}, {1, 7}, {1, 7}, {1, 7}, {1, 7}, {1, 7}, {1, 7}}},
                 true},
                {{-15, -5, std::nullopt},
                 {{{}, {}, {}, {}, {}, {1}, {1, 3}, {1, 3, 5}, {1, 3, 5, 7}, {1, 3, 5, 7, 9}}},
                 true},
                {{-15, -12, 3}, {{{}, {}, {}, {}, {}, {}, {}, {}, {}, {}}}, true},
                {{-15, -20, 3}, {{{}, {}, {}, {}, {}, {}, {}, {}, {}, {}}}, true},
        };

        for (const auto& [slice, expected_outputs, should_throw] : tests) {
            auto graph = Graph();
            auto x_ptr = graph.emplace_node<DynamicArrayTestingNode>(
                    std::initializer_list<ssize_t>{-1, dim1});

            if (should_throw) {
                REQUIRE_THROWS_AS(
                        graph.emplace_node<BasicIndexingNode>(x_ptr, slice, Slice(1, std::nullopt)),
                        std::invalid_argument);
                continue;
            }

            auto y_ptr =
                    graph.emplace_node<BasicIndexingNode>(x_ptr, slice, Slice(1, std::nullopt));
            graph.emplace_node<ArrayValidationNode>(y_ptr);

            auto state = graph.empty_state();
            x_ptr->initialize_state(state, {});
            graph.initialize_state(state);

            REQUIRE(x_ptr->size(state) == 0);
            CHECK(y_ptr->size(state) == 0);

            for (const std::vector<int>& expected_output : expected_outputs) {
                std::vector<double> values;
                for (auto v :
                     std::ranges::iota_view(x_ptr->size(state), x_ptr->size(state) + dim1)) {
                    values.emplace_back(v);
                }
                x_ptr->grow(state, values);
                graph.propagate(state);
                graph.commit(state);

                INFO("checking equivalent to np.arange("
                     << x_ptr->size(state) << ").reshape(-1, " << dim1 << ")[" << slice.start << ":"
                     << slice.stop << ":" << slice.step << ", 1:] (after growing)");
                CHECK(y_ptr->size(state) == static_cast<ssize_t>(expected_output.size()));
                CHECK_THAT(y_ptr->view(state), RangeEquals(expected_output));
            }

            REQUIRE(x_ptr->size(state) == dim0_max * dim1);

            for (const std::vector<int>& expected_output :
                 expected_outputs | std::views::reverse | std::views::drop(1)) {
                x_ptr->shrink(state);
                graph.propagate(state);
                graph.commit(state);

                INFO("checking equivalent to np.arange("
                     << x_ptr->size(state) << ").reshape(-1, " << dim1 << ")[" << slice.start << ":"
                     << slice.stop << ":" << slice.step << ", 1:] (after shrinking)");
                CHECK(y_ptr->size(state) == static_cast<ssize_t>(expected_output.size()));
                CHECK_THAT(y_ptr->view(state), RangeEquals(expected_output));
            }

            x_ptr->shrink(state);
            graph.propagate(state);
            graph.commit(state);

            REQUIRE(x_ptr->size(state) == 0);
            CHECK(y_ptr->size(state) == 0);
        }
    }

    GIVEN("x = Binary((3, 3)); y = x[::2, 1:]") {
        auto x_ptr = graph.emplace_node<BinaryNode>(std::vector<ssize_t>{3, 3});
        auto y_ptr = graph.emplace_node<BasicIndexingNode>(
                x_ptr, Slice(std::nullopt, std::nullopt, 2), Slice(1, std::nullopt));

        graph.emplace_node<ArrayValidationNode>(y_ptr);

        THEN("y has the shape and strides we expect") {
            CHECK(y_ptr->size() == 4);
            CHECK_THAT(y_ptr->shape(), RangeEquals({2, 2}));
            CHECK_THAT(y_ptr->strides(), RangeEquals({48, 8}));
        }

        auto state = graph.initialize_state();

        THEN("We get the default states we expect") {
            CHECK(std::ranges::equal(x_ptr->view(state), std::vector<ssize_t>(9)));
            CHECK(std::ranges::equal(y_ptr->view(state), std::vector<ssize_t>(4)));
        }

        WHEN("We do propagation") {
            x_ptr->flip(state, 0);
            x_ptr->flip(state, 1);

            graph.propagate(state, graph.descendants(state, {x_ptr}));

            THEN("The states are as expected") {
                CHECK_THAT(x_ptr->view(state), RangeEquals({1, 1, 0, 0, 0, 0, 0, 0, 0}));
                CHECK_THAT(y_ptr->view(state), RangeEquals({1, 0, 0, 0}));
            }
        }
    }

    GIVEN("x = Binary((10, 3)); y = x[5:7, 1]") {
        auto x_ptr = graph.emplace_node<BinaryNode>(std::vector<ssize_t>{10, 3});
        auto y_ptr = graph.emplace_node<BasicIndexingNode>(x_ptr, Slice(5, 7), 1);

        graph.emplace_node<ArrayValidationNode>(y_ptr);

        THEN("y has the shape and strides we expect") {
            CHECK(y_ptr->size() == 2);
            CHECK_THAT(y_ptr->shape(), RangeEquals({2}));
            CHECK_THAT(y_ptr->strides(), RangeEquals({24}));
        }

        auto state = graph.initialize_state();

        THEN("We get the default states we expect") {
            CHECK(std::ranges::equal(x_ptr->view(state), std::vector<ssize_t>(30)));
            CHECK(std::ranges::equal(y_ptr->view(state), std::vector<ssize_t>(2)));
        }

        WHEN("We do propagation") {
            x_ptr->flip(state, 19);

            graph.propagate(state, graph.descendants(state, {x_ptr}));

            THEN("The states are as expected") {
                CHECK_THAT(y_ptr->view(state), RangeEquals({0, 1}));
            }
        }
    }

    GIVEN("x = Binary((3, 10, 5)); y = x[1, 5:8, ::2]") {
        auto x_ptr = graph.emplace_node<BinaryNode>(std::vector<ssize_t>{3, 10, 5});
        auto y_ptr = graph.emplace_node<BasicIndexingNode>(x_ptr, 1, Slice(5, 8),
                                                           Slice(std::nullopt, std::nullopt, 2));

        graph.emplace_node<ArrayValidationNode>(y_ptr);

        THEN("y has the shape and strides we expect") {
            CHECK(y_ptr->size() == 9);
            CHECK_THAT(y_ptr->shape(), RangeEquals({3, 3}));
            CHECK_THAT(y_ptr->strides(), RangeEquals({40, 16}));
        }

        auto state = graph.initialize_state();

        THEN("We get the default states we expect") {
            CHECK(std::ranges::equal(x_ptr->view(state), std::vector<ssize_t>(150)));
            CHECK(std::ranges::equal(y_ptr->view(state), std::vector<ssize_t>(9)));
        }

        WHEN("We do propagation") {
            x_ptr->flip(state, 87);

            graph.propagate(state, graph.descendants(state, {x_ptr}));

            THEN("The states are as expected") {
                CHECK_THAT(y_ptr->view(state), RangeEquals({
                                                       0, 0, 0,  //
                                                       0, 0, 0,  //
                                                       0, 1, 0,  //
                                               }));
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

                graph.propagate(state, graph.descendants(state, {x_ptr}));

                CHECK(y_ptr->size(state) == x_ptr->size(state) - 1);
                CHECK(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == y_ptr->size(state));
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector<double>()));

                x_ptr->grow(state);
                x_ptr->grow(state);

                graph.propagate(state, graph.descendants(state, {x_ptr}));

                CHECK(y_ptr->size(state) == x_ptr->size(state) - 1);
                CHECK(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == y_ptr->size(state));
                CHECK_THAT(y_ptr->view(state), RangeEquals({1, 2}));
            }
        }
    }

    WHEN("We access a dynamic length 1d array by a slice from the beginning with negative end") {
        auto x_ptr = graph.emplace_node<SetNode>(5);
        auto y_ptr = graph.emplace_node<BasicIndexingNode>(x_ptr, Slice(0, -2));  // x[:-2]

        graph.emplace_node<ArrayValidationNode>(y_ptr);

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

                graph.propagate(state, graph.descendants(state, {x_ptr}));

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

                graph.propagate(state, graph.descendants(state, {x_ptr}));

                CHECK(y_ptr->size(state) == 1);
                REQUIRE(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == y_ptr->size(state));
                CHECK_THAT(y_ptr->view(state), RangeEquals({0}));

                graph.commit(state, graph.descendants(state, {x_ptr}));

                x_ptr->grow(state);

                graph.propagate(state, graph.descendants(state, {x_ptr}));

                CHECK_THAT(y_ptr->view(state), RangeEquals({0, 1}));
            }

            AND_WHEN("We grow the dynamic array past the range") {
                for (int i = 0; i < 4; i++) x_ptr->grow(state);

                graph.propagate(state, graph.descendants(state, {x_ptr}));

                CHECK(y_ptr->size(state) == 2);
                REQUIRE(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == y_ptr->size(state));
                CHECK_THAT(y_ptr->view(state), RangeEquals({0, 1}));

                graph.commit(state, graph.descendants(state, {x_ptr}));

                x_ptr->grow(state);

                graph.propagate(state, graph.descendants(state, {x_ptr}));
            }

            AND_WHEN("We shrink the dynamic array below the range") {
                for (int i = 0; i < 4; i++) x_ptr->grow(state);

                graph.propagate(state, graph.descendants(state, {x_ptr}));
                graph.commit(state, graph.descendants(state, {x_ptr}));

                // These should have already been tested
                REQUIRE(std::ranges::equal(y_ptr->view(state), std::vector{0, 1}));
                REQUIRE(y_ptr->diff(state).size() == 0);

                // Now shrink
                x_ptr->shrink(state);
                graph.propagate(state, graph.descendants(state, {x_ptr}));
                CHECK_THAT(y_ptr->view(state), RangeEquals({0}));

                graph.commit(state, graph.descendants(state, {x_ptr}));

                // Shrink again
                x_ptr->shrink(state);
                graph.propagate(state, graph.descendants(state, {x_ptr}));
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector<double>()));

                graph.commit(state, graph.descendants(state, {x_ptr}));

                // Shrink again, should be no change
                x_ptr->shrink(state);
                graph.propagate(state, graph.descendants(state, {x_ptr}));
                graph.commit(state, graph.descendants(state, {x_ptr}));
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector<double>()));
                CHECK(y_ptr->diff(state).size() == 0);
            }

            AND_WHEN(
                    "We shrink the dynamic array below the range, and then update a value above "
                    "that range") {
                for (int i = 0; i < 4; i++) x_ptr->grow(state);

                graph.propagate(state, graph.descendants(state, {x_ptr}));
                graph.commit(state, graph.descendants(state, {x_ptr}));

                // These should have already been tested
                REQUIRE(std::ranges::equal(y_ptr->view(state), std::vector{0, 1}));
                REQUIRE(y_ptr->diff(state).size() == 0);

                // Now shrink
                x_ptr->grow(state);            // [0, 1, 2, 3, 4][:-2] = [0, 1, 2]
                x_ptr->exchange(state, 2, 4);  // [0, 1, 4, 3, 2][:-2] = [0, 1, 4]
                x_ptr->shrink(state);          // [0, 1, 4, 3][:-2] = [0, 1]
                graph.propagate(state, graph.descendants(state, {x_ptr}));

                CHECK_THAT(y_ptr->view(state), RangeEquals({0, 1}));
            }

            AND_WHEN("We shrink and grow the dynamic array") {
                for (int i = 0; i < 3; i++) {
                    x_ptr->grow(state);
                    x_ptr->grow(state);

                    x_ptr->shrink(state);
                }

                graph.propagate(state, graph.descendants(state, {x_ptr}));

                CHECK_THAT(y_ptr->view(state), RangeEquals({0}));

                graph.commit(state, graph.descendants(state, {x_ptr}));

                x_ptr->grow(state);
                x_ptr->exchange(state, 1, 2);
                x_ptr->shrink(state);

                graph.propagate(state, graph.descendants(state, {x_ptr}));

                CHECK_THAT(y_ptr->view(state), RangeEquals({0}));
            }

            AND_WHEN("We change the shape of the dynamic array, and then revert") {
                x_ptr->grow(state);
                x_ptr->grow(state);
                x_ptr->grow(state);

                graph.revert(state, graph.descendants(state, {x_ptr}));

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

        graph.emplace_node<ArrayValidationNode>(y_ptr);

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

                graph.propagate(state, graph.descendants(state, {x_ptr}));

                CHECK(y_ptr->size(state) == 1);
                CHECK(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == 1);
                CHECK_THAT(y_ptr->view(state), RangeEquals({0}));
            }

            AND_WHEN("We grow the dynamic array up to the range") {
                x_ptr->grow(state);
                x_ptr->grow(state);

                graph.propagate(state, graph.descendants(state, {x_ptr}));

                CHECK(y_ptr->size(state) == 2);
                REQUIRE(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == 2);
                CHECK_THAT(y_ptr->view(state), RangeEquals({0, 1}));

                graph.commit(state, graph.descendants(state, {x_ptr}));

                x_ptr->grow(state);

                graph.propagate(state, graph.descendants(state, {x_ptr}));

                CHECK_THAT(y_ptr->view(state), RangeEquals({1, 2}));
            }

            AND_WHEN("We grow the dynamic array past the range") {
                for (int i = 0; i < 4; i++) x_ptr->grow(state);

                graph.propagate(state, graph.descendants(state, {x_ptr}));

                CHECK(y_ptr->size(state) == 2);
                REQUIRE(y_ptr->shape(state).size() == 1);
                CHECK(y_ptr->shape(state)[0] == y_ptr->size(state));
                CHECK_THAT(y_ptr->view(state), RangeEquals({2, 3}));

                graph.commit(state, graph.descendants(state, {x_ptr}));

                x_ptr->grow(state);

                graph.propagate(state, graph.descendants(state, {x_ptr}));
            }

            AND_WHEN("We shrink the dynamic array below the range") {
                for (int i = 0; i < 4; i++) x_ptr->grow(state);

                graph.propagate(state, graph.descendants(state, {x_ptr}));
                graph.commit(state, graph.descendants(state, {x_ptr}));

                // These should have already been tested
                REQUIRE(std::ranges::equal(y_ptr->view(state), std::vector{2, 3}));
                REQUIRE(y_ptr->diff(state).size() == 0);

                // Now shrink
                x_ptr->shrink(state);
                graph.propagate(state, graph.descendants(state, {x_ptr}));
                CHECK_THAT(y_ptr->view(state), RangeEquals({1, 2}));

                graph.commit(state, graph.descendants(state, {x_ptr}));

                // Shrink twice, should now be only length one
                x_ptr->shrink(state);
                x_ptr->shrink(state);
                graph.propagate(state, graph.descendants(state, {x_ptr}));
                CHECK_THAT(y_ptr->view(state), RangeEquals({0}));

                graph.commit(state, graph.descendants(state, {x_ptr}));

                // Shrink again, should now be length zero
                x_ptr->shrink(state);
                graph.propagate(state, graph.descendants(state, {x_ptr}));
                CHECK(std::ranges::equal(y_ptr->view(state), std::vector<double>()));
            }

            AND_WHEN("We shrink and grow the dynamic array") {
                for (int i = 0; i < 4; i++) {
                    x_ptr->grow(state);
                    x_ptr->grow(state);

                    x_ptr->shrink(state);
                }

                graph.propagate(state, graph.descendants(state, {x_ptr}));

                CHECK_THAT(y_ptr->view(state), RangeEquals({2, 3}));
            }

            AND_WHEN("We change the shape of the dynamic array, and then revert") {
                x_ptr->grow(state);
                x_ptr->grow(state);
                x_ptr->grow(state);

                graph.revert(state, graph.descendants(state, {x_ptr}));

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

        auto x_ptr = graph.emplace_node<DynamicArrayTestingNode>(std::initializer_list<ssize_t>{-1},
                                                                 -10, 10, false);
        auto y_ptr = graph.emplace_node<BasicIndexingNode>(x_ptr, slice);
        graph.emplace_node<ArrayValidationNode>(y_ptr);

        AND_WHEN("We do random moves and accept") {
            auto state = graph.initialize_state();
            auto rng = std::default_random_engine(42);

            for (int i = 0; i < 100; ++i) {
                // do up to 20 random moves
                const int num_moves = std::uniform_int_distribution<int>(0, 20)(rng);
                std::uniform_int_distribution<int> move_type(0, 2);
                for (int m = 0; m < num_moves; ++m) {
                    x_ptr->random_move(state, rng);
                }

                graph.propagate(state, graph.descendants(state, {x_ptr}));
                graph.commit(state, graph.descendants(state, {x_ptr}));
            }
        }
    }

    // todo: slicing on multidimensional dynamic array, when we have one to test...
}

}  // namespace dwave::optimization

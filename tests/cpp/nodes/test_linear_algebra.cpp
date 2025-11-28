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
#include <vector>

#include "dwave-optimization/nodes/binaryop.hpp"
#include "dwave-optimization/nodes/collections.hpp"
#include "dwave-optimization/nodes/constants.hpp"
#include "dwave-optimization/nodes/indexing.hpp"
#include "dwave-optimization/nodes/linear_algebra.hpp"
#include "dwave-optimization/nodes/manipulation.hpp"
#include "dwave-optimization/nodes/numbers.hpp"
#include "dwave-optimization/nodes/testing.hpp"

using Catch::Matchers::RangeEquals;

namespace dwave::optimization {

TEST_CASE("MatrixMultiplyNode") {
    auto graph = Graph();

    GIVEN("A dynamic testing array node and MatrixMultiply on it") {
        auto arr = DynamicArrayTestingNode(std::initializer_list<ssize_t>{-1}, -10.0, -5.0, false);
        auto constant = ConstantNode(15.0);
        auto add = AddNode(&arr, &constant);
        REQUIRE(add.min() == 5.0);
        REQUIRE(add.max() == 10.0);
        auto matmul = MatrixMultiplyNode(&arr, &add);
        THEN("MatrixMultiplyNode reports correct min and max") {
            CHECK(matmul.min() == ValuesInfo().min);
            CHECK(matmul.max() == 0.0);
        }
    }

    GIVEN("A dynamic testing array node with minimum size and matmul on it") {
        auto arr = DynamicArrayTestingNode(std::initializer_list<ssize_t>{-1}, -10.0, -5.0, false,
                                           3, std::nullopt);
        auto constant = ConstantNode(15.0);
        auto add = AddNode(&arr, &constant);
        REQUIRE(add.min() == 5.0);
        REQUIRE(add.max() == 10.0);
        auto matmul = MatrixMultiplyNode(&arr, &add);
        THEN("MatrixMultiplyNode reports correct min and max") {
            CHECK(matmul.min() == ValuesInfo().min);
            CHECK(matmul.max() == -5.0 * 5.0 * 3);
        }
    }

    GIVEN("A dynamic testing array node with minimum and maximum size and matmul on it") {
        auto arr = DynamicArrayTestingNode(std::initializer_list<ssize_t>{-1}, -10.0, -5.0, false,
                                           3, 7);
        auto constant = ConstantNode(15.0);
        auto add = AddNode(&arr, &constant);
        REQUIRE(add.min() == 5.0);
        REQUIRE(add.max() == 10.0);
        auto matmul = MatrixMultiplyNode(&arr, &add);
        THEN("MatrixMultiplyNode reports correct min and max") {
            CHECK(matmul.min() == -10.0 * 10.0 * 7);
            CHECK(matmul.max() == -5.0 * 5.0 * 3);
        }
    }

    SECTION("Higher order broadcasting") {
        auto a = IntegerNode({5, 4, 3, 2});

        auto b = IntegerNode({2, 7});
        CHECK_THROWS_AS(MatrixMultiplyNode(&a, &b), std::invalid_argument);

        auto c = IntegerNode({4, 2, 1});
        CHECK_THROWS_AS(MatrixMultiplyNode(&a, &c), std::invalid_argument);

        auto d = IntegerNode({5, 1, 2, 1});
        CHECK_THROWS_AS(MatrixMultiplyNode(&a, &d), std::invalid_argument);
    }

    GIVEN("Two constant 1d nodes and a MatrixMultiplyNode") {
        auto c1_ptr = graph.emplace_node<ConstantNode>(std::vector{1.0, 2.0, 3.0});
        auto c2_ptr = graph.emplace_node<ConstantNode>(std::vector{4.0, 5.0, 6.0});

        auto matmul_ptr = graph.emplace_node<MatrixMultiplyNode>(c1_ptr, c2_ptr);

        graph.emplace_node<ArrayValidationNode>(matmul_ptr);

        CHECK(matmul_ptr->size() == 1);
        CHECK(matmul_ptr->ndim() == 0);

        CHECK(matmul_ptr->min() == 1.0 * 4.0 * 3);
        CHECK(matmul_ptr->max() == 3.0 * 6.0 * 3);

        WHEN("We initialize a state") {
            auto state = graph.initialize_state();

            THEN("The initial MatrixMultiplyNode state is correct") {
                CHECK_THAT(matmul_ptr->view(state), RangeEquals({32}));
            }
        }
    }

    GIVEN("Two constant 2d nodes and a MatrixMultiplyNode") {
        auto c1_ptr = graph.emplace_node<ConstantNode>(std::vector{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                                                       std::vector<ssize_t>{2, 3});
        auto c2_ptr = graph.emplace_node<ConstantNode>(std::vector{7.0, 8.0, 9.0, 10.0, 11.0, 12.0},
                                                       std::vector<ssize_t>{3, 2});

        auto matmul_ptr = graph.emplace_node<MatrixMultiplyNode>(c1_ptr, c2_ptr);

        graph.emplace_node<ArrayValidationNode>(matmul_ptr);

        CHECK(matmul_ptr->size() == 4);
        CHECK(matmul_ptr->ndim() == 2);
        CHECK_THAT(matmul_ptr->shape(), RangeEquals({2, 2}));

        CHECK(matmul_ptr->min() == 1.0 * 7.0 * 3);
        CHECK(matmul_ptr->max() == 6.0 * 12.0 * 3);

        WHEN("We initialize a state") {
            auto state = graph.initialize_state();

            THEN("The initial MatrixMultiplyNode state is correct") {
                CHECK_THAT(matmul_ptr->view(state), RangeEquals({58, 64, 139, 154}));
            }
        }
    }

    GIVEN("Two constant 2d nodes and a MatrixMultiplyNode") {
        auto c1_ptr = graph.emplace_node<ConstantNode>(std::vector{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                                                       std::vector<ssize_t>{2, 3});
        auto c2_ptr = graph.emplace_node<ConstantNode>(std::vector{7.0, 8.0, 9.0, 10.0, 11.0, 12.0},
                                                       std::vector<ssize_t>{3, 2});

        auto matmul_ptr = graph.emplace_node<MatrixMultiplyNode>(c2_ptr, c1_ptr);

        graph.emplace_node<ArrayValidationNode>(matmul_ptr);

        CHECK(matmul_ptr->size() == 9);
        CHECK(matmul_ptr->ndim() == 2);
        CHECK_THAT(matmul_ptr->shape(), RangeEquals({3, 3}));

        WHEN("We initialize a state") {
            auto state = graph.initialize_state();

            THEN("The initial MatrixMultiplyNode state is correct") {
                CHECK_THAT(matmul_ptr->view(state),
                           RangeEquals({39, 54, 69, 49, 68, 87, 59, 82, 105}));
            }
        }
    }

    GIVEN("One constant 1d node and one constant 2d node") {
        auto c1_ptr = graph.emplace_node<ConstantNode>(std::vector{1.0, 2.0, 3.0},
                                                       std::vector<ssize_t>{3});
        auto c2_ptr = graph.emplace_node<ConstantNode>(
                std::vector{4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0},
                std::vector<ssize_t>{3, 3});

        AND_GIVEN("The 1d node @ the 2d node") {
            auto matmul_ptr = graph.emplace_node<MatrixMultiplyNode>(c1_ptr, c2_ptr);

            graph.emplace_node<ArrayValidationNode>(matmul_ptr);

            CHECK(matmul_ptr->size() == 3);
            CHECK(matmul_ptr->ndim() == 1);

            WHEN("We initialize a state") {
                auto state = graph.initialize_state();

                THEN("The initial MatrixMultiplyNode state is correct") {
                    CHECK_THAT(matmul_ptr->view(state), RangeEquals({48, 54, 60}));
                }
            }
        }

        AND_GIVEN("The 2d node @ the 1d node") {
            auto matmul_ptr = graph.emplace_node<MatrixMultiplyNode>(c2_ptr, c1_ptr);

            graph.emplace_node<ArrayValidationNode>(matmul_ptr);

            CHECK(matmul_ptr->size() == 3);
            CHECK(matmul_ptr->ndim() == 1);

            WHEN("We initialize a state") {
                auto state = graph.initialize_state();

                THEN("The initial MatrixMultiplyNode state is correct") {
                    CHECK_THAT(matmul_ptr->view(state), RangeEquals({32, 50, 68}));
                }
            }
        }
    }

    GIVEN("One 1d and one 4d constant nodes and a MatrixMultiplyNode") {
        std::vector<ssize_t> c1_shape{7};
        const ssize_t c1_size = 7;
        std::vector<double> c1_data(c1_size);
        std::iota(c1_data.begin(), c1_data.end(), 2.0);
        auto c1_ptr = graph.emplace_node<ConstantNode>(c1_data, c1_shape);
        REQUIRE(c1_ptr->min() == 2.0);
        REQUIRE(c1_ptr->max() == 8.0);

        std::vector<ssize_t> c2_shape{5, 3, 7, 2};
        ssize_t c2_size = 5 * 3 * 7 * 2;
        std::vector<double> c2_data(c2_size);
        std::iota(c2_data.begin(), c2_data.end(), -10.0);
        auto c2_ptr = graph.emplace_node<ConstantNode>(c2_data, c2_shape);
        REQUIRE(c2_ptr->min() == -10.0);
        REQUIRE(c2_ptr->max() == 199.0);

        auto matmul_ptr = graph.emplace_node<MatrixMultiplyNode>(c1_ptr, c2_ptr);

        graph.emplace_node<ArrayValidationNode>(matmul_ptr);

        CHECK(matmul_ptr->size() == 5 * 3 * 2);
        CHECK(matmul_ptr->ndim() == 3);
        CHECK_THAT(matmul_ptr->shape(), RangeEquals({5, 3, 2}));

        CHECK(matmul_ptr->min() == 8.0 * -10.0 * 7);
        CHECK(matmul_ptr->max() == 8.0 * 199.0 * 7);

        WHEN("We initialize a state") {
            auto state = graph.initialize_state();

            THEN("The initial MatrixMultiplyNode state is correct") {
                CHECK_THAT(
                        matmul_ptr->view(state),
                        RangeEquals({-84,  -49,  406,  441,  896,  931,  1386, 1421, 1876, 1911,
                                     2366, 2401, 2856, 2891, 3346, 3381, 3836, 3871, 4326, 4361,
                                     4816, 4851, 5306, 5341, 5796, 5831, 6286, 6321, 6776, 6811}));
            }
        }
    }

    GIVEN("One 4d and one 1d constant nodes and a MatrixMultiplyNode") {
        std::vector<ssize_t> c1_shape{5, 3, 2, 7};
        ssize_t c1_size = 5 * 3 * 2 * 7;
        std::vector<double> c1_data(c1_size);
        std::iota(c1_data.begin(), c1_data.end(), -10.0);
        auto c1_ptr = graph.emplace_node<ConstantNode>(c1_data, c1_shape);
        REQUIRE(c1_ptr->min() == -10.0);
        REQUIRE(c1_ptr->max() == 199.0);

        std::vector<ssize_t> c2_shape{7};
        const ssize_t c2_size = 7;
        std::vector<double> c2_data(c2_size);
        std::iota(c2_data.begin(), c2_data.end(), 2.0);
        auto c2_ptr = graph.emplace_node<ConstantNode>(c2_data, c2_shape);
        REQUIRE(c2_ptr->min() == 2.0);
        REQUIRE(c2_ptr->max() == 8.0);

        auto matmul_ptr = graph.emplace_node<MatrixMultiplyNode>(c1_ptr, c2_ptr);

        graph.emplace_node<ArrayValidationNode>(matmul_ptr);

        CHECK(matmul_ptr->size() == 5 * 3 * 2);
        CHECK(matmul_ptr->ndim() == 3);
        CHECK_THAT(matmul_ptr->shape(), RangeEquals({5, 3, 2}));

        CHECK(matmul_ptr->min() == 8.0 * -10.0 * 7);
        CHECK(matmul_ptr->max() == 8.0 * 199.0 * 7);

        WHEN("We initialize a state") {
            auto state = graph.initialize_state();

            THEN("The initial MatrixMultiplyNode state is correct") {
                CHECK_THAT(
                        matmul_ptr->view(state),
                        RangeEquals({-217, 28,   273,  518,  763,  1008, 1253, 1498, 1743, 1988,
                                     2233, 2478, 2723, 2968, 3213, 3458, 3703, 3948, 4193, 4438,
                                     4683, 4928, 5173, 5418, 5663, 5908, 6153, 6398, 6643, 6888}));
            }
        }
    }

    GIVEN("Two 4d constant nodes and a MatrixMultiplyNode") {
        std::vector<ssize_t> c1_shape{5, 3, 1, 7};
        const ssize_t c1_size = 5 * 3 * 1 * 7;
        std::vector<double> c1_data(c1_size);
        std::iota(c1_data.begin(), c1_data.end(), -20.0);
        auto c1_ptr = graph.emplace_node<ConstantNode>(c1_data, c1_shape);
        REQUIRE(c1_ptr->min() == -20.0);
        REQUIRE(c1_ptr->max() == 84);

        std::vector<ssize_t> c2_shape{5, 3, 7, 2};
        ssize_t c2_size = 5 * 3 * 7 * 2;
        std::vector<double> c2_data(c2_size);
        std::iota(c2_data.begin(), c2_data.end(), -10.0);
        auto c2_ptr = graph.emplace_node<ConstantNode>(c2_data, c2_shape);
        REQUIRE(c2_ptr->min() == -10.0);
        REQUIRE(c2_ptr->max() == 199.0);

        auto matmul_ptr = graph.emplace_node<MatrixMultiplyNode>(c1_ptr, c2_ptr);

        graph.emplace_node<ArrayValidationNode>(matmul_ptr);

        CHECK(matmul_ptr->size() == 5 * 3 * 1 * 2);
        CHECK(matmul_ptr->ndim() == 4);
        CHECK_THAT(matmul_ptr->shape(), RangeEquals({5, 3, 1, 2}));

        CHECK(matmul_ptr->min() == -20.0 * 199.0 * 7);
        CHECK(matmul_ptr->max() == 84.0 * 199.0 * 7);

        WHEN("We initialize a state") {
            auto state = graph.initialize_state();

            THEN("The initial MatrixMultiplyNode state is correct") {
                CHECK_THAT(matmul_ptr->view(state),
                           RangeEquals({532,   413,   -644,  -714,  -448,   -469,  1120,  1148,
                                        4060,  4137,  8372,  8498,  14056,  14231, 21112, 21336,
                                        29540, 29813, 39340, 39662, 50512,  50883, 63056, 63476,
                                        76972, 77441, 92260, 92778, 108920, 109487}));
            }
        }
    }

    GIVEN("A set node and MatrixMultiplyNode representing self dot product") {
        auto set_ptr = graph.emplace_node<SetNode>(10);
        auto matmul_ptr = graph.emplace_node<MatrixMultiplyNode>(set_ptr, set_ptr);
        graph.emplace_node<ArrayValidationNode>(matmul_ptr);

        CHECK(matmul_ptr->ndim() == 0);

        WHEN("We initialize a state") {
            auto state = graph.initialize_state();
            REQUIRE(set_ptr->size(state) == 0);

            THEN("The initial MatrixMultiplyNode state is correct") {
                CHECK(matmul_ptr->size(state) == 1);
                CHECK(matmul_ptr->shape(state).size() == 0);
                CHECK_THAT(matmul_ptr->view(state), RangeEquals({0.0}));
            }

            AND_WHEN("We grow the set and propagate") {
                set_ptr->assign(state, {5, 7, 1});
                graph.propagate(state);

                THEN("The state is correct") {
                    CHECK_THAT(matmul_ptr->view(state), RangeEquals({5 * 5 + 7 * 7 + 1 * 1}));
                }

                AND_WHEN("We revert") {
                    graph.revert(state);

                    THEN("The state is correct") {
                        CHECK(matmul_ptr->size(state) == 1);
                        CHECK(matmul_ptr->shape(state).size() == 0);
                        CHECK_THAT(matmul_ptr->view(state), RangeEquals({0.0}));
                    }
                }

                AND_WHEN("We commit") {
                    graph.commit(state);

                    THEN("The state is correct") {
                        CHECK(matmul_ptr->size(state) == 1);
                        CHECK(matmul_ptr->shape(state).size() == 0);
                        CHECK_THAT(matmul_ptr->view(state), RangeEquals({5 * 5 + 7 * 7 + 1 * 1}));
                    }
                }
            }
        }
    }

    GIVEN("A 2d dynamic testing node and a 1d constant") {
        auto arr_ptr = graph.emplace_node<DynamicArrayTestingNode>(
                std::initializer_list<ssize_t>{-1, 3}, -3.0, 10.0, false);
        auto c_ptr = graph.emplace_node<ConstantNode>(std::vector{1, 2, 3});

        CHECK_THROWS_AS(MatrixMultiplyNode(c_ptr, arr_ptr), std::invalid_argument);

        auto matmul_ptr = graph.emplace_node<MatrixMultiplyNode>(arr_ptr, c_ptr);
        graph.emplace_node<ArrayValidationNode>(matmul_ptr);

        CHECK(matmul_ptr->dynamic());
        CHECK(matmul_ptr->ndim() == 1);

        CHECK(matmul_ptr->min() == 3.0 * -3.0 * 3);
        CHECK(matmul_ptr->max() == 3.0 * 10.0 * 3);

        WHEN("We initialize a state") {
            auto state = graph.initialize_state();
            REQUIRE(arr_ptr->size(state) == 0);

            THEN("The initial MatrixMultiplyNode state is correct") {
                CHECK(matmul_ptr->size(state) == 0);
                CHECK_THAT(matmul_ptr->shape(state), RangeEquals({0}));
                CHECK(matmul_ptr->view(state).size() == 0);
            }

            AND_WHEN("We grow the set and propagate") {
                arr_ptr->grow(state, {5.0, 7.0, 9.0, 6.0, 8.0, 10.0});
                graph.propagate(state);

                THEN("The state is correct") {
                    CHECK(matmul_ptr->size(state) == 2);
                    CHECK_THAT(matmul_ptr->shape(state), RangeEquals({2}));
                    CHECK_THAT(matmul_ptr->view(state), RangeEquals({46, 52}));
                }

                AND_WHEN("We revert") {
                    graph.revert(state);

                    THEN("The state is correct") {
                        CHECK(matmul_ptr->size(state) == 0);
                        CHECK_THAT(matmul_ptr->shape(state), RangeEquals({0}));
                        CHECK(matmul_ptr->view(state).size() == 0);
                    }
                }

                AND_WHEN("We commit") {
                    graph.commit(state);

                    THEN("The state is correct") {
                        CHECK(matmul_ptr->size(state) == 2);
                        CHECK_THAT(matmul_ptr->shape(state), RangeEquals({2}));
                        CHECK_THAT(matmul_ptr->view(state), RangeEquals({46, 52}));
                    }
                }
            }
        }
    }

    GIVEN("A 2d dynamic testing node and a 1d column slice") {
        auto arr_ptr =
                graph.emplace_node<DynamicArrayTestingNode>(std::initializer_list<ssize_t>{-1, 3});
        auto vec_ptr = graph.emplace_node<BasicIndexingNode>(arr_ptr, Slice(), 0);

        CHECK_THROWS_AS(MatrixMultiplyNode(arr_ptr, vec_ptr), std::invalid_argument);

        auto matmul_ptr = graph.emplace_node<MatrixMultiplyNode>(vec_ptr, arr_ptr);
        graph.emplace_node<ArrayValidationNode>(matmul_ptr);

        CHECK(not matmul_ptr->dynamic());
        CHECK(matmul_ptr->ndim() == 1);
        CHECK(matmul_ptr->size() == 3);
        CHECK_THAT(matmul_ptr->shape(), RangeEquals({3}));

        WHEN("We initialize a state") {
            auto state = graph.initialize_state();
            REQUIRE(arr_ptr->size(state) == 0);

            THEN("The initial MatrixMultiplyNode state is correct") {
                CHECK_THAT(matmul_ptr->view(state), RangeEquals({0, 0, 0}));
            }

            AND_WHEN("We grow the set and propagate") {
                arr_ptr->grow(state, {5.0, 7.0, 9.0, 6.0, 8.0, 10.0});
                graph.propagate(state);

                THEN("The state is correct") {
                    CHECK_THAT(matmul_ptr->view(state), RangeEquals({61, 83, 105}));
                }

                AND_WHEN("We revert") {
                    graph.revert(state);

                    THEN("The state is correct") {
                        CHECK_THAT(matmul_ptr->view(state), RangeEquals({0, 0, 0}));
                    }
                }

                AND_WHEN("We commit") {
                    graph.commit(state);

                    THEN("The state is correct") {
                        CHECK_THAT(matmul_ptr->view(state), RangeEquals({61, 83, 105}));
                    }
                }
            }
        }
    }

    GIVEN("A 4d dynamic testing node and a matmulable reshape") {
        auto arr_ptr = graph.emplace_node<DynamicArrayTestingNode>(
                std::initializer_list<ssize_t>{-1, 3, 2, 7});
        auto reshape_ptr =
                graph.emplace_node<ReshapeNode>(arr_ptr, std::vector<ssize_t>{-1, 3, 7, 2});

        auto matmul_ptr = graph.emplace_node<MatrixMultiplyNode>(arr_ptr, reshape_ptr);
        graph.emplace_node<ArrayValidationNode>(matmul_ptr);

        CHECK(matmul_ptr->dynamic());
        CHECK(matmul_ptr->ndim() == 4);
        CHECK_THAT(matmul_ptr->shape(), RangeEquals({-1, 3, 2, 2}));

        WHEN("We initialize a state") {
            auto state = graph.initialize_state();
            REQUIRE(arr_ptr->size(state) == 0);

            THEN("The initial MatrixMultiplyNode state is correct") {
                CHECK(matmul_ptr->view(state).size() == 0);
            }

            AND_WHEN("We grow the set and propagate") {
                std::vector<double> arr_data(2 * 3 * 2 * 7);
                std::iota(arr_data.begin(), arr_data.end(), 0.0);
                arr_ptr->grow(state, arr_data);
                graph.propagate(state);

                std::vector<double> expected = {182,   203,   476,   546,   2436,  2555,
                                                3416,  3584,  7434,  7651,  9100,  9366,
                                                15176, 15491, 17528, 17892, 25662, 26075,
                                                28700, 29162, 38892, 39403, 42616, 43176};

                THEN("The state is correct") {
                    CHECK_THAT(matmul_ptr->view(state), RangeEquals(expected));
                }

                AND_WHEN("We revert") {
                    graph.revert(state);

                    THEN("The state is correct") { CHECK(matmul_ptr->view(state).size() == 0); }
                }

                AND_WHEN("We commit") {
                    graph.commit(state);

                    THEN("The state is correct") {
                        CHECK_THAT(matmul_ptr->view(state), RangeEquals(expected));
                    }
                }
            }
        }
    }
}

}  // namespace dwave::optimization

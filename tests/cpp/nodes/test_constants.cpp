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
#include "dwave-optimization/nodes/constants.hpp"

using Catch::Matchers::RangeEquals;

namespace dwave::optimization {

TEST_CASE("ConstantNode") {
    auto graph = Graph();

    GIVEN("A default-constructed constant node") {
        auto ptr = graph.emplace_node<ConstantNode>();

        THEN("It defaults to an empty 1d array") {
            CHECK(ptr->ndim() == 1);
            CHECK(ptr->size() == 0);
            CHECK_THAT(ptr->data(), RangeEquals(std::vector<double>()));
            CHECK_THAT(ptr->shape(), RangeEquals({0}));
            CHECK_THAT(ptr->strides(), RangeEquals({sizeof(double)}));
        }

        THEN("min/max/integral are all well-defined") {
            CHECK(ptr->integral());
            CHECK(ptr->max() == 0);
            CHECK(ptr->min() == 0);
        }
    }

    GIVEN("A constant node copied from a vector") {
        std::vector<double> values = {30, 10, 40, 20};
        auto ptr = graph.emplace_node<ConstantNode>(values);

        THEN("It copies the values into a 1d array") {
            CHECK(ptr->ndim() == 1);
            CHECK(ptr->size() == 4);
            CHECK(std::ranges::equal(ptr->data(), values));
            CHECK_THAT(ptr->shape(), RangeEquals({4}));
            CHECK_THAT(ptr->strides(), RangeEquals({sizeof(double)}));
        }

        THEN("min/max/integral all match the vector") {
            CHECK(ptr->integral());
            CHECK(ptr->max() == 40);
            CHECK(ptr->min() == 10);
        }

        WHEN("We mutate the original vector") {
            values[2] = -105;

            THEN("The ConstantNode is not affected because it's a copy") {
                CHECK_THAT(ptr->data(), RangeEquals({30, 10, 40, 20}));
            }
        }
    }

    GIVEN("A constant node copied from a vector with a reshape") {
        std::vector<double> values = {30, 10, 40, 20, 50};  // deliberately too large
        auto ptr = graph.emplace_node<ConstantNode>(values, std::initializer_list<ssize_t>{2, 2});

        THEN("It acts as a 2d array") {
            CHECK(ptr->ndim() == 2);
            CHECK(ptr->size() == 4);  // shrunk to fit
            CHECK_THAT(ptr->data(), RangeEquals({30, 10, 40, 20}));
            CHECK_THAT(ptr->shape(), RangeEquals({2, 2}));
            CHECK_THAT(ptr->strides(), RangeEquals({2 * sizeof(double), sizeof(double)}));
        }

        WHEN("We mutate the original vector") {
            values[2] = -105;

            THEN("The ConstantNode is not affected because it's a copy") {
                CHECK_THAT(ptr->data(), RangeEquals({30, 10, 40, 20}));
            }
        }
    }

    GIVEN("A constant node representing a single value") {
        auto ptr = graph.emplace_node<ConstantNode>(6.5);

        THEN("It acts as a single value") {
            CHECK(ptr->ndim() == 0);
            CHECK(ptr->size() == 1);
            CHECK_THAT(ptr->data(), RangeEquals({6.5}));
            CHECK_THAT(ptr->shape(), RangeEquals(std::vector<int>{}));
            CHECK_THAT(ptr->strides(), RangeEquals(std::vector<int>{}));
        }

        THEN("min/max/integral all match the value") {
            CHECK(!ptr->integral());
            CHECK(ptr->max() == 6.5);
            CHECK(ptr->min() == 6.5);
        }
    }

    GIVEN("A constant node that is a view of data held by a vector") {
        std::vector<double> values = {30, 10, 40, 20, 50};  // deliberately too large
        auto ptr = graph.emplace_node<ConstantNode>(values.data(),
                                                    std::initializer_list<ssize_t>{2, 2});

        THEN("It acts as a 2d array") {
            CHECK(ptr->ndim() == 2);
            CHECK(ptr->size() == 4);  // shrunk to fit
            CHECK_THAT(ptr->data(), RangeEquals({30, 10, 40, 20}));
            CHECK_THAT(ptr->shape(), RangeEquals({2, 2}));
            CHECK_THAT(ptr->strides(), RangeEquals({2 * sizeof(double), sizeof(double)}));
        }

        WHEN("We mutate the original vector") {
            values[2] = -105;

            THEN("The ConstantNode is mutated accordingly") {
                CHECK_THAT(ptr->data(), RangeEquals({30, 10, -105, 20}));
            }
        }
    }
}

}  // namespace dwave::optimization

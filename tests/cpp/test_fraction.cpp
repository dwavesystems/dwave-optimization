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

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_all.hpp"
#include "dwave-optimization/fraction.hpp"

using Catch::Matchers::RangeEquals;

namespace dwave::optimization {

TEST_CASE("Test fraction") {
    // everything can be done with constexpr so we can static require everything

    STATIC_REQUIRE(fraction(3) == fraction(3, 1));
    STATIC_REQUIRE(fraction(3) == fraction(6, 2));
    STATIC_REQUIRE(fraction(3) == fraction(-12, -4));

    STATIC_REQUIRE(fraction(0) == fraction(0, 5));
    STATIC_REQUIRE(fraction(1, 5) * 0 == fraction(0));
    STATIC_REQUIRE(fraction(1, 2) * 0 == fraction(0, 2) / 2);

    STATIC_REQUIRE(fraction(3) == 3);
    STATIC_REQUIRE(3 == fraction(3));
    STATIC_REQUIRE(fraction(3, -1) == -3);
    STATIC_REQUIRE(fraction(-3, 1) == -3);
    STATIC_REQUIRE(fraction(-3, -1) == 3);

    STATIC_REQUIRE(fraction(3) != 4);
    STATIC_REQUIRE(4 != fraction(3));
    STATIC_REQUIRE(fraction(3, -1) != 3);
    STATIC_REQUIRE(fraction(-3, 1) != 3);
    STATIC_REQUIRE(fraction(100, 50) == 2);
    STATIC_REQUIRE(fraction(100, 50).numerator() == 2);
    STATIC_REQUIRE(fraction(100, 50).denominator() == 1);
    STATIC_REQUIRE(fraction(50, 100).numerator() == 1);
    STATIC_REQUIRE(fraction(50, 100).denominator() == 2);
    STATIC_REQUIRE(fraction(50, -100).numerator() == -1);
    STATIC_REQUIRE(fraction(50, -100).denominator() == 2);

    STATIC_REQUIRE(static_cast<double>(fraction(1, 2)) == .5);

    STATIC_REQUIRE(fraction() == 0);

    STATIC_REQUIRE(fraction(0, 5).ceil() == 0);
    STATIC_REQUIRE(fraction(1, 5).ceil() == 1);
    STATIC_REQUIRE(fraction(11, 5).ceil() == 3);
    STATIC_REQUIRE(fraction(-9, 5).ceil() == -1);
    STATIC_REQUIRE(fraction(-10, 5).ceil() == -2);
    STATIC_REQUIRE(fraction(-11, 5).ceil() == -2);

    GIVEN("An uninitialized fraction") {
        fraction a;

        WHEN("We assign a value to it") {
            a = 5;

            THEN("It gets the value we expect") { CHECK(a == 5); }
        }
    }
}

}  // namespace dwave::optimization

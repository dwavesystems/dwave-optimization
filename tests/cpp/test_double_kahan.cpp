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
#include "double_kahan.hpp"

using Catch::Matchers::RangeEquals;

namespace dwave::optimization {

TEST_CASE("double_kahan") {
    GIVEN("An uninitialized struct for compensated sum") {
        double_kahan a;

        CHECK(a == 0);
        CHECK(a.value() == 0);
        CHECK(a.compensator() == 0);

        WHEN("We assign a value to it") {
            a = 10.5;

            THEN("We can read it properly") {
                CHECK(a == 10.5);
                CHECK(a.value() == 10.5);
                CHECK(a.compensator() == 0);
            }
        }
    }

    WHEN("We run a loop with predefined additions and subtractions") {
        double d = 0;
        double_kahan d_k = 0;
        for (int i = 0; i < 1000000; i++) {
            d += 0.0001;
            d -= 0.00005;
            d_k += 0.0001;
            d_k -= 0.00005;
        }

        THEN("The compensated accumulator is more accurate than the non-compensated one") {
            CHECK(std::abs(d_k.value() - 5000.0) < std::abs(d - 5000.0));
        }

        AND_WHEN("When we copy-assign another double_kahan") {
            double_kahan a;
            a = d_k;

            THEN("The compensator is preserved") {
                CHECK(a == d_k);
                CHECK(a.compensator() == d_k.compensator());
            }
        }

        AND_WHEN("When we copy-construct another double_kahan") {
            double_kahan a(d_k);

            THEN("The compensator is preserved") {
                CHECK(a == d_k);
                CHECK(a.compensator() == d_k.compensator());
            }
        }
    }

    STATIC_REQUIRE(double_kahan(5) == double_kahan(5));
    STATIC_REQUIRE(double_kahan(5) == 5.0);
    STATIC_REQUIRE(double_kahan(5) == 5);
    STATIC_REQUIRE(double_kahan(5) != double_kahan(6));
    STATIC_REQUIRE(double_kahan(5) != 6.0);
    STATIC_REQUIRE(double_kahan(5) != 6);

    STATIC_REQUIRE(5.0 == double_kahan(5));
    STATIC_REQUIRE(5 == double_kahan(5));
    STATIC_REQUIRE(6.0 != double_kahan(5));
    STATIC_REQUIRE(6 != double_kahan(5));
}

}  // namespace dwave::optimization

// Copyright 2026 D-Wave
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

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "dwave-optimization/interval.hpp"

namespace dwave::optimization {

TEMPLATE_TEST_CASE(
    "interval",
    "",
    float,
    double,
    bool,
    std::int8_t,
    std::int16_t,
    std::int32_t,
    std::int64_t
) {
    GIVEN("a default-constructed interval") {
        auto in = interval<TestType>();

        CHECK(not static_cast<bool>(in));
        CHECK(in.size() == 0);
    }

    GIVEN("an interval containing only the value 0") {
        auto in = interval<TestType>(0);

        CHECK(static_cast<bool>(in));
        CHECK(in.size() == 0);
        CHECK(in.contains(0));
        CHECK(not in.contains(1));
        CHECK(not in.contains(std::numeric_limits<double>::epsilon()));

        auto [inf, sup] = in;
        STATIC_REQUIRE(std::same_as<decltype(inf), TestType>);
        STATIC_REQUIRE(std::same_as<decltype(sup), TestType>);
        CHECK(inf == 0);
        CHECK(sup == 0);
    }
}

TEST_CASE("interval") {
    SECTION("union operations") {
        auto in = interval<int>();
        in |= .5;
        CHECK(in == interval<int>(0, 1));
        in |= -5;
        CHECK(in == interval<int>(-5, 1));
    }
    // CHECK((interval<int>() | static_cast<float>(.5)) == interval<float>(0, 1));
}

}  // namespace dwave::optimization

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

#include "dwave-optimization/interval.hpp"

namespace dwave::optimization {

TEMPLATE_LIST_TEST_CASE("interval", "", DTypes) {
    SECTION("::operator bool()") {
        STATIC_REQUIRE(not interval<TestType>());
        STATIC_REQUIRE(interval<TestType>(0, 1));
        STATIC_REQUIRE(interval<TestType>(0, 0));
        STATIC_REQUIRE(interval<TestType>(1, 1));

        // We only allow explicit conversion to bool
        STATIC_REQUIRE(not std::convertible_to<interval<TestType>, bool>);
    }

    SECTION("::operator==") {
        STATIC_REQUIRE(interval<TestType>(0, 1) == interval<TestType>(0, 1));

        STATIC_REQUIRE(interval<TestType>(0, 0) != interval<TestType>(0, 1));
        STATIC_REQUIRE(interval<TestType>(0, 1) != interval<TestType>());

        STATIC_REQUIRE(interval<int>(10, -10) == interval<int>(1, 0));  // null always equals null
    }

    SECTION("::operator<= (i.e., subset)") {
        STATIC_REQUIRE(interval<TestType>(0, 0) <= interval<TestType>(0, 1));
        STATIC_REQUIRE(interval<TestType>(0, 1) <= interval<TestType>(0, 1));  // equality allowed
    }

    SECTION("::operator-") {
        if constexpr (std::same_as<TestType, bool>) {
            STATIC_REQUIRE(interval<TestType>(0, 1) == -interval<TestType>(0, 1));
            STATIC_REQUIRE(interval<TestType>(0, 0) == -interval<TestType>(1, 1));
            STATIC_REQUIRE(interval<TestType>(1, 1) == -interval<TestType>(0, 0));
        } else {
            STATIC_REQUIRE(interval<TestType>(1, 10) == -interval<TestType>(-10, -1));
        }

        if constexpr (std::integral<TestType>) {
            STATIC_REQUIRE(-interval<TestType>::all() == interval<TestType>::all());
        }
    }

    SECTION("::operator&=/::operator& (i.e., intersection)") {
        STATIC_REQUIRE(not static_cast<bool>(interval<TestType>(0, 1) & interval<TestType>()));
        STATIC_REQUIRE(not static_cast<bool>(interval<TestType>() & interval<TestType>(0, 1)));

        if constexpr (not std::same_as<TestType, bool>) {
            STATIC_REQUIRE(
                (interval<TestType>(0, 5) & interval<TestType>(2, 3)) == interval<TestType>(2, 3)
            );
            STATIC_REQUIRE(
                (interval<TestType>(0, 5) & interval<TestType>(3, 10)) == interval<TestType>(3, 5)
            );
        }
    }

    SECTION("::operator|=/::operator| (i.e., union)") {
        STATIC_REQUIRE(
            (interval<TestType>(0, 1) | interval<TestType>()) == interval<TestType>(0, 1)
        );
        STATIC_REQUIRE(
            (interval<TestType>() | interval<TestType>(0, 1)) == interval<TestType>(0, 1)
        );

        if constexpr (not std::same_as<TestType, bool>) {
            STATIC_REQUIRE(
                (interval<TestType>(0, 5) | interval<TestType>(2, 3)) == interval<TestType>(0, 5)
            );
            STATIC_REQUIRE(
                (interval<TestType>(0, 5) | interval<TestType>(3, 10)) == interval<TestType>(0, 10)
            );
        }
    }

    SECTION("::contains") {
        STATIC_REQUIRE(not interval<TestType>().contains(0));
        STATIC_REQUIRE(interval<TestType>(0, 1).contains(0));
        STATIC_REQUIRE(interval<TestType>(1, 1).contains(1));
    }

    SECTION("printing") {
        SECTION("integral") {
            std::stringstream ss;
            ss << interval<TestType>(0, 1);
            CHECK(ss.str() == "[0, 1]");
        }

        SECTION("floating") {
            if constexpr (std::floating_point<TestType>) {
                std::stringstream ss;
                ss << interval<TestType>(.5, 1.5);
                CHECK(ss.str() == "[0.5, 1.5]");
            }
        }
    }

    SECTION("structured binding") {
        SECTION("const reference") {
            auto in = interval<TestType>(0, 1);
            const auto& [inf, sup] = in;
            STATIC_REQUIRE(std::same_as<decltype(inf), const TestType>);
            STATIC_REQUIRE(std::same_as<decltype(sup), const TestType>);
            CHECK(inf == 0);
            CHECK(sup == 1);
        }

        SECTION("rvalue") {
            auto in = interval<TestType>(0, 1);
            auto [inf, sup] = in;
            STATIC_REQUIRE(std::same_as<decltype(inf), TestType>);
            STATIC_REQUIRE(std::same_as<decltype(sup), TestType>);
            CHECK(inf == 0);
            CHECK(sup == 1);
        }

        SECTION("const rvalue") {
            const auto in = interval<TestType>(0, 1);

            auto&& [inf, sup] = in;
            STATIC_REQUIRE(std::same_as<decltype(inf), const TestType>);
            STATIC_REQUIRE(std::same_as<decltype(sup), const TestType>);
            CHECK(inf == 0);
            CHECK(sup == 1);
        }
    }
}

TEST_CASE("interval") {
    STATIC_REQUIRE(interval<double>(interval<float>(0, 5)) == interval<double>(0, 5));
}

}  // namespace dwave::optimization

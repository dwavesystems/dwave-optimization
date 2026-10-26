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

#include <cmath>
#include <concepts>
#include <cstdint>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "kahan.hpp"

namespace dwave::optimization {

namespace {

// Test whether `k += u`, `k -= u` and `k == u` compile.
template <typename K, typename U>
concept addable = requires(K k, U u) { k += u; };
template <typename K, typename U>
concept subtractable = requires(K k, U u) { k -= u; };
template <typename K, typename U>
concept comparable = requires(K k, U u) { k == u; };

}  // namespace

TEMPLATE_TEST_CASE("kahan", "", float, double) {
    GIVEN("An uninitialized struct for compensated sum") {
        kahan<TestType> a;

        CHECK(a == TestType{0});
        CHECK(a.value() == 0);
        CHECK(a.compensator() == 0);

        WHEN("We assign a value to it") {
            a = TestType{10.5};

            THEN("We can read it properly") {
                CHECK(a == TestType{10.5});
                CHECK(a.value() == 10.5);
                CHECK(a.compensator() == 0);
                CHECK(static_cast<TestType>(a) == 10.5);
            }
        }
    }

    WHEN("We run a loop with predefined additions and subtractions") {
        TestType d = 0;
        kahan<TestType> d_k;
        for (int i = 0; i < 1000000; i++) {
            d += TestType{0.0001};
            d -= TestType{0.00005};
            d_k += TestType{0.0001};
            d_k -= TestType{0.00005};
        }

        THEN("The compensated accumulator is more accurate than the non-compensated one") {
            CHECK(std::abs(d_k.value() - 50) < std::abs(d - 50));
        }

        AND_WHEN("When we copy-assign another kahan") {
            kahan<TestType> a;
            a = d_k;

            THEN("The compensator is preserved") {
                CHECK(a == d_k);
                CHECK(a.compensator() == d_k.compensator());
            }
        }

        AND_WHEN("When we copy-construct another kahan") {
            kahan<TestType> a{d_k};

            THEN("The compensator is preserved") {
                CHECK(a == d_k);
                CHECK(a.compensator() == d_k.compensator());
            }
        }
    }

    STATIC_REQUIRE(kahan<TestType>{TestType{5}} == kahan<TestType>{TestType{5}});
    STATIC_REQUIRE(kahan<TestType>{TestType{5}} == TestType{5});
    STATIC_REQUIRE(kahan<TestType>{TestType{5}} != kahan<TestType>{TestType{6}});
    STATIC_REQUIRE(kahan<TestType>{TestType{5}} != TestType{6});

    STATIC_REQUIRE(TestType{5} == kahan<TestType>{TestType{5}});
    STATIC_REQUIRE(TestType{6} != kahan<TestType>{TestType{5}});
}

TEST_CASE("kahan only accepts values that can be safely cast to its type") {
    // Expected values follow NumPy, e.g. np.can_cast(np.int32, np.float32) is False

    SECTION("kahan<double> accepts float and integers") {
        STATIC_REQUIRE(kahan<double>{5} == kahan<double>{5});
        STATIC_REQUIRE(kahan<double>{5} == 5.0);
        STATIC_REQUIRE(kahan<double>{5} == 5.0f);
        STATIC_REQUIRE(kahan<double>{5} == 5);
        STATIC_REQUIRE(kahan<double>{5} != 6);
        STATIC_REQUIRE(5 == kahan<double>{5});
        STATIC_REQUIRE(6 != kahan<double>{5});

        STATIC_REQUIRE(addable<kahan<double>, float>);
        STATIC_REQUIRE(addable<kahan<double>, std::int64_t>);
        STATIC_REQUIRE(subtractable<kahan<double>, std::int64_t>);
    }

    SECTION("kahan<float> accepts small integers") {
        STATIC_REQUIRE(kahan<float>{std::int16_t{5}} == std::int16_t{5});
        STATIC_REQUIRE(addable<kahan<float>, std::int16_t>);
        STATIC_REQUIRE(subtractable<kahan<float>, std::int8_t>);
    }

    SECTION("kahan<float> rejects double and larger integers") {
        STATIC_REQUIRE(not std::constructible_from<kahan<float>, double>);
        STATIC_REQUIRE(not addable<kahan<float>, double>);
        STATIC_REQUIRE(not subtractable<kahan<float>, double>);
        STATIC_REQUIRE(not comparable<kahan<float>, double>);

        STATIC_REQUIRE(not std::constructible_from<kahan<float>, std::int32_t>);
        STATIC_REQUIRE(not addable<kahan<float>, std::int32_t>);
        STATIC_REQUIRE(not comparable<kahan<float>, std::int32_t>);
    }
}

}  // namespace dwave::optimization

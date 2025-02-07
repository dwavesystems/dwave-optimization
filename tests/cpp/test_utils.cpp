// Copyright 2024 D-Wave Systems Inc.
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
#include "dwave-optimization/array.hpp"
#include "dwave-optimization/utils.hpp"

using Catch::Matchers::RangeEquals;

namespace dwave::optimization {

TEST_CASE("CompensatedSum") {
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

    GIVEN("An uninitialized fraction") {
        fraction a;

        WHEN("We assign a value to it") {
            a = 5;

            THEN("It gets the value we expect") { CHECK(a == 5); }
        }
    }
}

TEST_CASE("Test deduplicate_diff") {
    GIVEN("An empty vector of updates") {
        std::vector<Update> updates;

        WHEN("We call deduplicate_diff") {
            deduplicate_diff(updates);
            THEN("deduplicate_diff() doesn't do anything") { CHECK(updates.size() == 0); }
        }
    }

    GIVEN("A list of updates with no duplicates or noop Updates") {
        std::vector<Update> updates = {Update(3, 0, 1), Update(5, 0, -1), Update(2, 1, 0)};

        WHEN("We call deduplicate_diff") {
            deduplicate_diff(updates);
            THEN("deduplicate_diff() sorts them") {
                CHECK(std::ranges::equal(
                        updates,
                        std::vector<Update>{Update(2, 1, 0), Update(3, 0, 1), Update(5, 0, -1)}));
            }
        }
    }

    GIVEN("A list of updates with no duplicates and noop Updates") {
        std::vector<Update> updates = {Update(3, 0, 1), Update(5, -1, -1), Update(2, 1, 0)};

        WHEN("We call deduplicate_diff") {
            deduplicate_diff(updates);
            THEN("deduplicate_diff() sorts them and removes noops") {
                CHECK_THAT(updates, RangeEquals({Update(2, 1, 0), Update(3, 0, 1)}));
            }
        }
    }

    GIVEN("A list of update of one noop Updates") {
        std::vector<Update> updates = {Update(3, 1, 1)};

        WHEN("We call deduplicate_diff") {
            deduplicate_diff(updates);
            THEN("deduplicate_diff() removes the noop") { CHECK(updates.size() == 0); }
        }
    }

    GIVEN("A list of update of many noop Updates") {
        std::vector<Update> updates = {Update(3, 1, 1), Update(3, 1, 1), Update(4, -1, -1),
                                       Update(0, 57, 57)};

        WHEN("We call deduplicate_diff") {
            deduplicate_diff(updates);
            THEN("deduplicate_diff() removes all the noops") { CHECK(updates.size() == 0); }
        }
    }

    GIVEN("A list of updates with no duplicates and one noop at the beginning") {
        std::vector<Update> updates = {Update(3, 1, 5), Update(4, 1, 5), Update(6, -1, -2),
                                       Update(0, 57, 57)};

        WHEN("We call deduplicate_diff") {
            deduplicate_diff(updates);
            THEN("deduplicate_diff() removes the noop") {
                CHECK(std::ranges::equal(
                        updates,
                        std::vector<Update>{Update(3, 1, 5), Update(4, 1, 5), Update(6, -1, -2)}));
            }
        }
    }

    GIVEN("A list of updates with no duplicates and one noop at the end") {
        std::vector<Update> updates = {Update(3, 1, 5), Update(4, 1, 5), Update(6, -1, -2),
                                       Update(8, 57, 57)};

        WHEN("We call deduplicate_diff") {
            deduplicate_diff(updates);
            THEN("deduplicate_diff() removes the noop") {
                CHECK(std::ranges::equal(
                        updates,
                        std::vector<Update>{Update(3, 1, 5), Update(4, 1, 5), Update(6, -1, -2)}));
            }
        }
    }

    GIVEN("A list of updates with duplicates") {
        std::vector<Update> updates = {Update(3, 1, 5),   Update(3, 5, -3),  Update(6, -1, -2),
                                       Update(6, -2, 57), Update(3, -3, -4), Update(2, 0, 1)};

        WHEN("We call deduplicate_diff") {
            deduplicate_diff(updates);
            THEN("deduplicate_diff() removes the noop") {
                CHECK(std::ranges::equal(
                        updates,
                        std::vector<Update>{Update(2, 0, 1), Update(3, 1, -4), Update(6, -1, 57)}));
            }
        }
    }

    GIVEN("A list of updates that grows and shrinks") {
        std::vector<Update> updates = {Update(3, 1, 5), Update::placement(5, 7),
                                       Update::placement(6, 7), Update(5, 7, 4),
                                       Update::removal(6, 7)};

        WHEN("We call deduplicate_diff") {
            deduplicate_diff(updates);
            THEN("deduplicate_diff() trims the diff properly") {
                CHECK(std::ranges::equal(
                        updates, std::vector<Update>{Update(3, 1, 5), Update::placement(5, 4)}));
            }
        }
    }
}

}  // namespace dwave::optimization

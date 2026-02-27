// Copyright 2026 D-Wave Systems Inc.
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

#include <memory>
#include <numeric>
#include <sstream>

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_all.hpp"
#include "dwave-optimization/array.hpp"
#include "dwave-optimization/cp/core/interval_array.hpp"
#include "dwave-optimization/cp/state/copier.hpp"
#include "dwave-optimization/state.hpp"
#include "utils.hpp"

using Catch::Matchers::RangeEquals;

namespace dwave::optimization::cp {

TEST_CASE("Interval double") {
    GIVEN("A state manager and 10 intervals with default boundary") {
        std::unique_ptr<Copier> sm = std::make_unique<Copier>();
        IntervalArray<double> interval = IntervalArray<double>(sm.get(), 10);
        REQUIRE(interval.num_domains() == 10);
        REQUIRE(sm->store.size() == 20);
        for (size_t i = 0; i < interval.num_domains(); ++i) {
            REQUIRE(interval.min(i) == -std::numeric_limits<double>::max() / 2);
            REQUIRE(interval.max(i) == std::numeric_limits<double>::max() / 2);
            REQUIRE(interval.size(i) == std::numeric_limits<double>::max());
        }

        AND_GIVEN("A test domain listener") {
            std::unique_ptr<TestListener> tl = std::make_unique<TestListener>();
            TestListener* tl_ptr = tl.get();

            WHEN("We remove below 0 from all indices") {
                for (int i = 0; i < 10; ++i) {
                    auto status = interval.remove_below(0., i, tl_ptr);
                    REQUIRE(status == CPStatus::OK);
                }

                THEN("The minimum is correctly changed and the maximum is left unchanged") {
                    for (int i = 0; i < 10; ++i) {
                        REQUIRE(interval.min(i) == 0.);
                        REQUIRE(interval.max(i) == std::numeric_limits<double>::max() / 2);
                    }
                }

                AND_WHEN("We remove above 3 from the fourth index") {
                    auto status = interval.remove_above(3, 4, tl_ptr);
                    REQUIRE(status == CPStatus::OK);
                    REQUIRE(interval.max(4) == 3);

                    THEN("The other indices are unchanged") {
                        for (int i = 0; i < 10; ++i) {
                            if (i == 4) continue;
                            REQUIRE(interval.min(i) == 0);
                            REQUIRE(interval.max(i) == std::numeric_limits<double>::max() / 2);
                        }
                    }

                    AND_WHEN("We remove above -3 from the first index") {
                        auto status = interval.remove_above(-3, 1, tl_ptr);
                        REQUIRE(status == CPStatus::Inconsistency);
                    }
                }
            }
        }
    }

    GIVEN("An array of intervals initialized by two vectors") {
        std::unique_ptr<Copier> sm = std::make_unique<Copier>();
        std::vector<double> lb(10);
        std::vector<double> ub(10);

        for (int i = 0; i < 10; ++i) {
            lb[i] = -i;
            ub[i] = i;
        }

        IntervalArray<double> interval2 = IntervalArray<double>(sm.get(), lb, ub);
        REQUIRE(sm->store.size() == 20);
        THEN("The fomains are correctly initialized") {
            REQUIRE(interval2.num_domains() == lb.size());
            for (int i = 0; i < 10; ++i) {
                REQUIRE(interval2.min(i) == -i);
                REQUIRE(interval2.max(i) == i);
                REQUIRE(interval2.size(i) == 2 * i);
            }
        }
    }

    GIVEN("An array of 15 intervals between 0.1 and 0.5") {
        std::unique_ptr<Copier> sm = std::make_unique<Copier>();
        IntervalArray<double> interval3 = IntervalArray<double>(sm.get(), 15, 0.1, 0.5);
        REQUIRE(sm->store.size() == 30);
        THEN("The domains are correctly set") {
            REQUIRE(interval3.num_domains() == 15);
            for (int i = 0; i < 15; ++i) {
                REQUIRE(interval3.min(i) == 0.1);
                REQUIRE(interval3.max(i) == 0.5);
                REQUIRE(interval3.size(i) == 0.4);
            }
        }
    }
}

}  // namespace dwave::optimization::cp

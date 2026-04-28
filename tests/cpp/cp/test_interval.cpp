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

TEST_CASE("Interval double (fixed size)") {
    GIVEN("A state manager and 10 intervals with default boundary") {
        std::unique_ptr<Copier> sm = std::make_unique<Copier>();
        RealIntervalArray interval = RealIntervalArray(sm.get(), 10);
        REQUIRE(interval.num_domains() == 10);
        REQUIRE(interval.max_size() == 10);
        REQUIRE(interval.min_size() == 10);

        // store size should be 2 x 10 + 2 for min and max size
        REQUIRE(sm->store.size() == 22);
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

        RealIntervalArray interval2 = RealIntervalArray(sm.get(), lb, ub);
        REQUIRE(sm->store.size() == 22);
        THEN("The fomains are correctly initialized") {
            REQUIRE(interval2.num_domains() == lb.size());
            REQUIRE(interval2.min_size() == 10);
            REQUIRE(interval2.max_size() == 10);
            for (int i = 0; i < 10; ++i) {
                REQUIRE(interval2.min(i) == -i);
                REQUIRE(interval2.max(i) == i);
                REQUIRE(interval2.size(i) == 2 * i);
            }
        }
    }

    GIVEN("An array of 15 intervals between 0.1 and 0.5") {
        std::unique_ptr<Copier> sm = std::make_unique<Copier>();
        RealIntervalArray interval3 = RealIntervalArray(sm.get(), 15, 0.1, 0.5);
        REQUIRE(sm->store.size() == 32);
        THEN("The domains are correctly set") {
            REQUIRE(interval3.num_domains() == 15);
            REQUIRE(interval3.min_size() == 15);
            REQUIRE(interval3.max_size() == 15);
            for (int i = 0; i < 15; ++i) {
                REQUIRE(interval3.min(i) == 0.1);
                REQUIRE(interval3.max(i) == 0.5);
                REQUIRE(interval3.size(i) == 0.4);
            }
        }
    }
}

TEST_CASE("Interval double (dynamic size)") {
    GIVEN("A state manager and a dynamic array with minimum size 6 and max size 10") {
        std::unique_ptr<Copier> sm = std::make_unique<Copier>();
        RealIntervalArray interval = RealIntervalArray(sm.get(), 6, 10);
        REQUIRE(interval.num_domains() == 10);
        REQUIRE(interval.min_size() == 6);
        REQUIRE(interval.max_size() == 10);
        // store size should be 2 x 10 + 2 for min and max
        REQUIRE(sm->store.size() == 22);
        for (size_t i = 0; i < interval.num_domains(); ++i) {
            REQUIRE(interval.min(i) == -std::numeric_limits<double>::max() / 2);
            REQUIRE(interval.max(i) == std::numeric_limits<double>::max() / 2);
            REQUIRE(interval.size(i) == std::numeric_limits<double>::max());
        }

        // Check the active and optional entries
        for (size_t i = 0; i < 6; ++i) {
            REQUIRE(interval.is_active(i));
        }
        for (size_t i = 7; i < 10; ++i) {
            REQUIRE_FALSE(interval.is_active(i));
            REQUIRE(interval.maybe_active(i));
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

                    AND_WHEN("We remove above -3 from the 7th index") {
                        auto status = interval.remove_above(-3, 7, tl_ptr);
                        REQUIRE(status == CPStatus::OK);
                        REQUIRE(interval.max_size() == 7);
                        for (int i = 7; i < 10; ++i) {
                            REQUIRE_FALSE(interval.is_active(i));
                            REQUIRE_FALSE(interval.maybe_active(i));
                        }
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

        RealIntervalArray interval2 = RealIntervalArray(sm.get(), 6, lb, ub);
        REQUIRE(sm->store.size() == 22);
        THEN("The domains are correctly initialized") {
            REQUIRE(interval2.num_domains() == lb.size());
            REQUIRE(interval2.min_size() == 6);
            REQUIRE(interval2.max_size() == static_cast<ssize_t>(lb.size()));
            for (int i = 0; i < 10; ++i) {
                REQUIRE(interval2.min(i) == -i);
                REQUIRE(interval2.max(i) == i);
                REQUIRE(interval2.size(i) == 2 * i);
                if (i < 6) {
                    REQUIRE(interval2.is_active(i));
                } else {
                    REQUIRE_FALSE(interval2.is_active(i));
                    REQUIRE(interval2.maybe_active(i));
                }
            }
        }
    }

    GIVEN("An array of 15 intervals between 0.1 and 0.5") {
        std::unique_ptr<Copier> sm = std::make_unique<Copier>();
        RealIntervalArray interval3 = RealIntervalArray(sm.get(), 6, 15, 0.1, 0.5);
        REQUIRE(sm->store.size() == 32);
        THEN("The domains are correctly set") {
            REQUIRE(interval3.num_domains() == 15);
            REQUIRE(interval3.min_size() == 6);
            REQUIRE(interval3.max_size() == 15);
            for (int i = 0; i < 15; ++i) {
                REQUIRE(interval3.min(i) == 0.1);
                REQUIRE(interval3.max(i) == 0.5);
                REQUIRE(interval3.size(i) == 0.4);
                if (i < 6) {
                    REQUIRE(interval3.is_active(i));
                } else {
                    REQUIRE_FALSE(interval3.is_active(i));
                    REQUIRE(interval3.maybe_active(i));
                }
            }
        }
    }
}

}  // namespace dwave::optimization::cp

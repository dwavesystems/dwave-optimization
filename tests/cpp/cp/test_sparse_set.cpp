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
#include "dwave-optimization/cp/core/sparse_set_array.hpp"
#include "dwave-optimization/cp/state/copier.hpp"
#include "dwave-optimization/state.hpp"
#include "utils.hpp"

using Catch::Matchers::RangeEquals;

namespace dwave::optimization::cp {

TEST_CASE("Sparse set (fixed size)") {
    GIVEN("A state manager and 5 sparse sets between 0 and 4") {
        std::unique_ptr<Copier> sm = std::make_unique<Copier>();
        SparseSetArray sp_set = SparseSetArray(sm.get(), 5, 5, -2, 8);
        REQUIRE(sp_set.num_domains() == 5);
        REQUIRE(sp_set.max_size() == 5);
        REQUIRE(sp_set.min_size() == 5);

        // store size should be 2 (array size) + 5 x 3 (size, min, max) = 17 for min and max size
        REQUIRE(sm->store.size() == 17);
        for (size_t i = 0; i < sp_set.num_domains(); ++i) {
            REQUIRE(sp_set.min(i) == -2);
            REQUIRE(sp_set.max(i) == 8);
            REQUIRE(sp_set.size(i) == 11);
        }

        AND_GIVEN("A test domain listener") {
            std::unique_ptr<TestListener> tl = std::make_unique<TestListener>();
            TestListener* tl_ptr = tl.get();

            WHEN("We remove below 0.5 from all indices") {
                for (int i = 0; i < 5; ++i) {
                    auto status = sp_set.remove_below(0.5, i, tl_ptr);
                    REQUIRE(status == CPStatus::OK);
                }

                THEN("The minimum is correctly changed and the maximum is left unchanged") {
                    for (int i = 0; i < 5; ++i) {
                        REQUIRE(sp_set.min(i) == 1);
                        REQUIRE(sp_set.max(i) == 8);
                    }
                }

                AND_WHEN("We remove above -1.3 from the fourth index") {
                    auto status = sp_set.remove_above(1.3, 4, tl_ptr);
                    REQUIRE(status == CPStatus::OK);
                    REQUIRE(sp_set.max(4) == 1);
                    REQUIRE(sp_set.is_bound(4));

                    THEN("The other indices are unchanged") {
                        for (int i = 0; i < 5; ++i) {
                            if (i == 4) continue;
                            REQUIRE(sp_set.min(i) == 1);
                            REQUIRE(sp_set.max(i) == 8);
                            REQUIRE(sp_set.size(i) == 8);
                        }
                    }

                    AND_WHEN("We remove above -3 from the first index") {
                        auto status = sp_set.remove_above(-3, 1, tl_ptr);
                        REQUIRE(status == CPStatus::Inconsistency);
                    }
                }
            }
        }
    }
}

TEST_CASE("Sparse Set (dynamic size)") {
    GIVEN("A state manager and a dynamic array with minimum size 1 and max size 5") {
        std::unique_ptr<Copier> sm = std::make_unique<Copier>();
        SparseSetArray sp_set = SparseSetArray(sm.get(), 1, 5, -2, 8);
        REQUIRE(sp_set.num_domains() == 5);
        REQUIRE(sp_set.min_size() == 1);
        REQUIRE(sp_set.max_size() == 5);
        // store size should be 3 x 5 + 2 for min and max
        REQUIRE(sm->store.size() == 17);
        for (size_t i = 0; i < sp_set.num_domains(); ++i) {
            REQUIRE(sp_set.min(i) == -2);
            REQUIRE(sp_set.max(i) == 8);
            REQUIRE(sp_set.size(i) == 11);
        }

        // Check the active and optional entries
        for (size_t i = 0; i < 1; ++i) {
            REQUIRE(sp_set.is_active(i));
        }
        for (size_t i = 1; i < 5; ++i) {
            REQUIRE_FALSE(sp_set.is_active(i));
            REQUIRE(sp_set.maybe_active(i));
        }

        AND_GIVEN("A test domain listener") {
            std::unique_ptr<TestListener> tl = std::make_unique<TestListener>();
            TestListener* tl_ptr = tl.get();

            WHEN("We remove below -0.1 from all indices") {
                for (int i = 0; i < 5; ++i) {
                    auto status = sp_set.remove_below(-0.1, i, tl_ptr);
                    REQUIRE(status == CPStatus::OK);
                }

                THEN("The minimum is correctly changed and the maximum is left unchanged") {
                    for (int i = 0; i < 5; ++i) {
                        REQUIRE(sp_set.min(i) == 0.);
                        REQUIRE(sp_set.max(i) == 8);
                    }
                }

                AND_WHEN("We remove above 3 from the fourth index") {
                    auto status = sp_set.remove_above(3, 4, tl_ptr);
                    REQUIRE(status == CPStatus::OK);
                    REQUIRE(sp_set.max(4) == 3);

                    THEN("The other indices are unchanged") {
                        for (int i = 0; i < 5; ++i) {
                            if (i == 4) continue;
                            REQUIRE(sp_set.min(i) == 0);
                            REQUIRE(sp_set.max(i) == 8);
                        }
                    }

                    AND_WHEN("We remove above -3 from the zeroth index") {
                        auto status = sp_set.remove_above(-3, 0, tl_ptr);
                        REQUIRE(status == CPStatus::Inconsistency);
                    }

                    AND_WHEN("We remove above -3 from the 2nd index") {
                        auto status = sp_set.remove_above(-3, 2, tl_ptr);
                        REQUIRE(status == CPStatus::OK);
                        REQUIRE(sp_set.max_size() == 2);
                        for (int i = 2; i < 5; ++i) {
                            REQUIRE_FALSE(sp_set.is_active(i));
                            REQUIRE_FALSE(sp_set.maybe_active(i));
                        }
                    }
                }
            }
        }
    }
}

}  // namespace dwave::optimization::cp

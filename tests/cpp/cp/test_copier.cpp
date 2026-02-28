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

#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_all.hpp"
#include "dwave-optimization/cp/state/copier.hpp"
#include "dwave-optimization/cp/state/copy.hpp"
#include "dwave-optimization/cp/state/state.hpp"
#include "dwave-optimization/cp/state/state_manager.hpp"
#include "utils.hpp"

using Catch::Matchers::RangeEquals;

namespace dwave::optimization::cp {

TEST_CASE("State") {
    StateInt si = StateInt(8);
    REQUIRE(si.get_value() == 8);

    si.increment();
    REQUIRE(si.get_value() == 9);

    si.decrement();
    REQUIRE(si.get_value() == 8);

    StateBool sb = StateBool(true);
    REQUIRE(sb.get_value());
    sb.set_value(false);
    REQUIRE_FALSE(sb.get_value());
}

TEST_CASE("Copy") {
    CopyInt ci = CopyInt(0);
    REQUIRE(ci.get_value() == 0);

    auto cci = ci.save();

    ci.increment();
    REQUIRE(ci.get_value() == 1);

    cci->restore();
    REQUIRE(ci.get_value() == 0);
}

TEST_CASE("Copier") {
    Copier sm = Copier();
    REQUIRE(sm.get_level() == -1);

    StateInt* si = sm.make_state_int(0);
    REQUIRE(si->get_value() == 0);

    StateBool* sb = sm.make_state_bool(false);
    REQUIRE(!sb->get_value());

    sm.save_state();
    REQUIRE(sm.get_level() == 0);

    si->increment();
    REQUIRE(si->get_value() == 1);

    sm.save_state();

    sm.restore_state_until(-1);
    REQUIRE(sm.get_level() == -1);
    REQUIRE(si->get_value() == 0);
    REQUIRE(!sb->get_value());

    // try with new state method
    auto f = [&]() {
        si->decrement();
        sb->set_value(true);
        REQUIRE(si->get_value() == -1);
        REQUIRE(sb->get_value());
    };

    sm.with_new_state(f);
    REQUIRE(si->get_value() == 0);
    REQUIRE(!sb->get_value());
}

}  // namespace dwave::optimization::cp

// Copyright 2023 D-Wave Inc.
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

// Utility functions for testing

#pragma once

#include <span>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "dwave-optimization/array.hpp"

namespace dwave::optimization {

inline void verify_array_diff(std::vector<double> state, const std::vector<double>& final_state,
                              const std::span<const Update>& diff) {
    for (const auto& update : diff) {
        const ssize_t size = state.size();
        if (update.placed()) {
            // Can only place at the end of an array
            REQUIRE(update.index == size);
            state.push_back(update.value);
        } else if (update.removed()) {
            // Can only remove at the end of an array
            REQUIRE(update.index == size - 1);
            state.pop_back();
        } else {
            REQUIRE(update.index >= 0);
            REQUIRE(update.index < size);
            CHECK(update.old == state[update.index]);
            state[update.index] = update.value;
        }
    }

    CHECK(std::ranges::equal(state, final_state));
}

}  // namespace dwave::optimization

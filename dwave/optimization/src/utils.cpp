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

#include "dwave-optimization/utils.hpp"

#include <cmath>

#include "dwave-optimization/array.hpp"

namespace dwave::optimization {

void deduplicate_diff(std::vector<Update>& diff) {
    if (diff.empty()) return;

    std::ranges::stable_sort(diff);

    // Find the index of first non-noop Update. If there are none, leave it as -1
    // to represent no final updates.
    ssize_t new_index = -1;
    for (size_t i = 0; i < diff.size(); ++i) {
        if (!diff[i].identity()) {
            new_index = i;
            break;
        }
    }
    ssize_t start = 1;
    if (new_index > 0) {
        assert(new_index < static_cast<ssize_t>(diff.size()));
        // Move the first non-noop update into the first spot
        diff[0] = diff[new_index];
        start = new_index + 1;
        new_index = 0;
    }

    if (new_index >= 0) {
        for (size_t i = start; i < diff.size(); ++i) {
            if (diff[i].index == diff[new_index].index) {
                diff[new_index].value = diff[i].value;
            } else if (diff[new_index].null()) {
                // We have finished processing the update at that index, but both the
                // old and new value are NaN which means it was added and then deleted,
                // and should be discarded.
                // At this point we are done because all updates at following indices
                // should also have been added and deleted.
                new_index--;
                break;
            } else if (!diff[i].identity()) {
                // Move to the next update only if the final state of the update is not a noop
                new_index++;
                diff[new_index] = diff[i];
            }
        }

        // In case the very last value is a place and removal, discard it
        if (diff[new_index].null()) {
            new_index--;
        }
    }

    assert(new_index >= -1);
    assert(new_index + 1 <= static_cast<ssize_t>(diff.size()));

    // Shrink the final diff array if necessary. Since this is always reszing smaller,
    // the value passed to resize doesn't matter (just to avoid implementing a
    // construct_at method for Update)
    diff.resize(new_index + 1, Update::placement(-666, 666));
}

bool is_integer(const double& value) {
    static double dummy = 0;
    return std::modf(value, &dummy) == 0.0;
}

};  // namespace dwave::optimization

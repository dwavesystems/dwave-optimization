// Copyright 2024 D-Wave
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

#include "dwave-optimization/nodes/constants.hpp"

#include <cmath>
#include <ranges>

namespace dwave::optimization {

bool ConstantNode::integral() const {
    auto values = this->data();  // all of the values in the array

    // If we're empty we return true because this gives maximum compatibility
    // with other nodes. There are several other nodes that require integral,
    // and none (as of Sept 2024) that require *not* integral.
    if (values.empty()) return true;

    double dummy = 0;
    auto is_integer = [&dummy](const double& val) { return std::modf(val, &dummy) == 0.0; };

    // We could cache this information, but there is nothing stopping the user
    // from modifying the underlying data, so for safety let's just do the
    // O(n) thing each time for now.
    return std::ranges::all_of(values, is_integer);
}

double ConstantNode::max() const {
    auto values = this->data();  // all of the values in the array

    // If we're empty we return 0.
    // We don't want undefined behavior because there are use cases e.g.
    // indexing by an empty array.
    // So 0 seems like a reasonable default.
    if (values.empty()) return 0.0;

    // We could cache this information, but there is nothing stopping the user
    // from modifying the underlying data, so for safety let's just do the
    // O(n) thing each time for now.
    return std::ranges::max(values);
}

double ConstantNode::min() const {
    auto values = this->data();  // all of the values in the array

    // If we're empty we return 0.
    // We don't want undefined behavior because there are use cases e.g.
    // indexing by an empty array.
    // So 0 seems like a reasonable default.
    if (values.empty()) return 0.0;

    // We could cache this information, but there is nothing stopping the user
    // from modifying the underlying data, so for safety let's just do the
    // O(n) thing each time for now.
    return std::ranges::min(values);
}

}  // namespace dwave::optimization

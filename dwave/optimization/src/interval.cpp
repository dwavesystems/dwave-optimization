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

#include "dwave-optimization/interval.hpp"
#include "dwave-optimization/common.hpp"  // for ssize_t
#include "dwave-optimization/typing.hpp"

#include <iostream>

namespace dwave::optimization {

std::ostream& operator<<(std::ostream& os, const interval<std::int64_t>& rhs) {
    return os << "[" << rhs.infimum() << ", " << rhs.supremum() << "]";
}
std::ostream& operator<<(std::ostream& os, const interval<double>& rhs) {
    return os << "[" << rhs.infimum() << ", " << rhs.supremum() << "]";
}

}  // namespace dwave::optimization

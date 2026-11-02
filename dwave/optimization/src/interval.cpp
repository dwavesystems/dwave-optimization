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

#include <iostream>

namespace dwave::optimization {

template <DType T>
std::ostream& operator<<(std::ostream& os, const interval<T>& in) {
    if (not static_cast<bool>(in)) return os << "[]";

    os << "[";

    // Not all compilers print all dtypes. So coerce them into a smaller set of
    // possible types
    if constexpr (std::integral<T>) {
        os << static_cast<long>(in.infimum) << ", " << static_cast<long>(in.supremum);
    } else if constexpr (std::floating_point<T>) {
        os << static_cast<double>(in.infimum) << ", " << static_cast<double>(in.supremum);
    } else {
        assert(false and "unexpected dtype");
    }

    os << "]";
    return os;
}

template std::ostream& operator<<(std::ostream&, const interval<float>&);
template std::ostream& operator<<(std::ostream&, const interval<double>&);
template std::ostream& operator<<(std::ostream&, const interval<bool>&);
template std::ostream& operator<<(std::ostream&, const interval<signed char>&);
template std::ostream& operator<<(std::ostream&, const interval<short>&);
template std::ostream& operator<<(std::ostream&, const interval<int>&);
template std::ostream& operator<<(std::ostream&, const interval<long>&);
template std::ostream& operator<<(std::ostream&, const interval<long long>&);

}  // namespace dwave::optimization

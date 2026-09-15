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

#include <utility>

#include <catch2/catch_test_macros.hpp>

#include "dwave-optimization/typing.hpp"

namespace dwave::optimization {

void forwarding_reference_func(DTypeLike auto&&) {}

TEST_CASE("DTypeLike") {
    int a = 1;
    forwarding_reference_func(a);
    forwarding_reference_func(static_cast<const int>(a));
    forwarding_reference_func(static_cast<const int&>(a));
    forwarding_reference_func(std::move(a));
}

TEST_CASE("can_cast") {
    // import itertools

    // import numpy as np

    // dtypes = {
    //     np.float32: "float",
    //     np.float64: "double",
    //     np.bool: "bool",
    //     np.int8: "std::int8_t",
    //     np.int16: "std::int16_t",
    //     np.int32: "std::int32_t",
    //     np.int64: "std::int64_t",
    // }

    // for dt0, dt1 in itertools.product(dtypes, repeat=2):
    //     From = dtypes[dt0]
    //     To = dtypes[dt1]
    //     if np.can_cast(dt0, dt1):
    //         print(f"STATIC_REQUIRE(can_cast<{From}, {To}>);")
    //     else:
    //         print(f"STATIC_REQUIRE(not can_cast<{From}, {To}>);")

    STATIC_REQUIRE(can_cast<float, float>);
    STATIC_REQUIRE(can_cast<float, double>);
    STATIC_REQUIRE(not can_cast<float, bool>);
    STATIC_REQUIRE(not can_cast<float, std::int8_t>);
    STATIC_REQUIRE(not can_cast<float, std::int16_t>);
    STATIC_REQUIRE(not can_cast<float, std::int32_t>);
    STATIC_REQUIRE(not can_cast<float, std::int64_t>);
    STATIC_REQUIRE(not can_cast<double, float>);
    STATIC_REQUIRE(can_cast<double, double>);
    STATIC_REQUIRE(not can_cast<double, bool>);
    STATIC_REQUIRE(not can_cast<double, std::int8_t>);
    STATIC_REQUIRE(not can_cast<double, std::int16_t>);
    STATIC_REQUIRE(not can_cast<double, std::int32_t>);
    STATIC_REQUIRE(not can_cast<double, std::int64_t>);
    STATIC_REQUIRE(can_cast<bool, float>);
    STATIC_REQUIRE(can_cast<bool, double>);
    STATIC_REQUIRE(can_cast<bool, bool>);
    STATIC_REQUIRE(can_cast<bool, std::int8_t>);
    STATIC_REQUIRE(can_cast<bool, std::int16_t>);
    STATIC_REQUIRE(can_cast<bool, std::int32_t>);
    STATIC_REQUIRE(can_cast<bool, std::int64_t>);
    STATIC_REQUIRE(can_cast<std::int8_t, float>);
    STATIC_REQUIRE(can_cast<std::int8_t, double>);
    STATIC_REQUIRE(not can_cast<std::int8_t, bool>);
    STATIC_REQUIRE(can_cast<std::int8_t, std::int8_t>);
    STATIC_REQUIRE(can_cast<std::int8_t, std::int16_t>);
    STATIC_REQUIRE(can_cast<std::int8_t, std::int32_t>);
    STATIC_REQUIRE(can_cast<std::int8_t, std::int64_t>);
    STATIC_REQUIRE(can_cast<std::int16_t, float>);
    STATIC_REQUIRE(can_cast<std::int16_t, double>);
    STATIC_REQUIRE(not can_cast<std::int16_t, bool>);
    STATIC_REQUIRE(not can_cast<std::int16_t, std::int8_t>);
    STATIC_REQUIRE(can_cast<std::int16_t, std::int16_t>);
    STATIC_REQUIRE(can_cast<std::int16_t, std::int32_t>);
    STATIC_REQUIRE(can_cast<std::int16_t, std::int64_t>);
    STATIC_REQUIRE(not can_cast<std::int32_t, float>);
    STATIC_REQUIRE(can_cast<std::int32_t, double>);
    STATIC_REQUIRE(not can_cast<std::int32_t, bool>);
    STATIC_REQUIRE(not can_cast<std::int32_t, std::int8_t>);
    STATIC_REQUIRE(not can_cast<std::int32_t, std::int16_t>);
    STATIC_REQUIRE(can_cast<std::int32_t, std::int32_t>);
    STATIC_REQUIRE(can_cast<std::int32_t, std::int64_t>);
    STATIC_REQUIRE(not can_cast<std::int64_t, float>);
    STATIC_REQUIRE(can_cast<std::int64_t, double>);
    STATIC_REQUIRE(not can_cast<std::int64_t, bool>);
    STATIC_REQUIRE(not can_cast<std::int64_t, std::int8_t>);
    STATIC_REQUIRE(not can_cast<std::int64_t, std::int16_t>);
    STATIC_REQUIRE(not can_cast<std::int64_t, std::int32_t>);
    STATIC_REQUIRE(can_cast<std::int64_t, std::int64_t>);
}

TEST_CASE("promote_types") {
    // import itertools

    // import numpy as np

    // dtypes = {
    //     np.float32: "float",
    //     np.float64: "double",
    //     np.bool: "bool",
    //     np.int8: "std::int8_t",
    //     np.int16: "std::int16_t",
    //     np.int32: "std::int32_t",
    //     np.int64: "std::int64_t",
    // }

    // for dt0, dt1 in itertools.product(dtypes, repeat=2):
    //     lhs = dtypes[dt0]
    //     rhs = dtypes[dt1]
    //     result = dtypes[np.promote_types(dt0, dt1).type]
    //     print(f"STATIC_REQUIRE(std::same_as<promote_types<{lhs}, {rhs}>::type, {result}>);")

    STATIC_REQUIRE(std::same_as<promote_types<float, float>::type, float>);
    STATIC_REQUIRE(std::same_as<promote_types<float, double>::type, double>);
    STATIC_REQUIRE(std::same_as<promote_types<float, bool>::type, float>);
    STATIC_REQUIRE(std::same_as<promote_types<float, std::int8_t>::type, float>);
    STATIC_REQUIRE(std::same_as<promote_types<float, std::int16_t>::type, float>);
    STATIC_REQUIRE(std::same_as<promote_types<float, std::int32_t>::type, double>);
    STATIC_REQUIRE(std::same_as<promote_types<float, std::int64_t>::type, double>);
    STATIC_REQUIRE(std::same_as<promote_types<double, float>::type, double>);
    STATIC_REQUIRE(std::same_as<promote_types<double, double>::type, double>);
    STATIC_REQUIRE(std::same_as<promote_types<double, bool>::type, double>);
    STATIC_REQUIRE(std::same_as<promote_types<double, std::int8_t>::type, double>);
    STATIC_REQUIRE(std::same_as<promote_types<double, std::int16_t>::type, double>);
    STATIC_REQUIRE(std::same_as<promote_types<double, std::int32_t>::type, double>);
    STATIC_REQUIRE(std::same_as<promote_types<double, std::int64_t>::type, double>);
    STATIC_REQUIRE(std::same_as<promote_types<bool, float>::type, float>);
    STATIC_REQUIRE(std::same_as<promote_types<bool, double>::type, double>);
    STATIC_REQUIRE(std::same_as<promote_types<bool, bool>::type, bool>);
    STATIC_REQUIRE(std::same_as<promote_types<bool, std::int8_t>::type, std::int8_t>);
    STATIC_REQUIRE(std::same_as<promote_types<bool, std::int16_t>::type, std::int16_t>);
    STATIC_REQUIRE(std::same_as<promote_types<bool, std::int32_t>::type, std::int32_t>);
    STATIC_REQUIRE(std::same_as<promote_types<bool, std::int64_t>::type, std::int64_t>);
    STATIC_REQUIRE(std::same_as<promote_types<std::int8_t, float>::type, float>);
    STATIC_REQUIRE(std::same_as<promote_types<std::int8_t, double>::type, double>);
    STATIC_REQUIRE(std::same_as<promote_types<std::int8_t, bool>::type, std::int8_t>);
    STATIC_REQUIRE(std::same_as<promote_types<std::int8_t, std::int8_t>::type, std::int8_t>);
    STATIC_REQUIRE(std::same_as<promote_types<std::int8_t, std::int16_t>::type, std::int16_t>);
    STATIC_REQUIRE(std::same_as<promote_types<std::int8_t, std::int32_t>::type, std::int32_t>);
    STATIC_REQUIRE(std::same_as<promote_types<std::int8_t, std::int64_t>::type, std::int64_t>);
    STATIC_REQUIRE(std::same_as<promote_types<std::int16_t, float>::type, float>);
    STATIC_REQUIRE(std::same_as<promote_types<std::int16_t, double>::type, double>);
    STATIC_REQUIRE(std::same_as<promote_types<std::int16_t, bool>::type, std::int16_t>);
    STATIC_REQUIRE(std::same_as<promote_types<std::int16_t, std::int8_t>::type, std::int16_t>);
    STATIC_REQUIRE(std::same_as<promote_types<std::int16_t, std::int16_t>::type, std::int16_t>);
    STATIC_REQUIRE(std::same_as<promote_types<std::int16_t, std::int32_t>::type, std::int32_t>);
    STATIC_REQUIRE(std::same_as<promote_types<std::int16_t, std::int64_t>::type, std::int64_t>);
    STATIC_REQUIRE(std::same_as<promote_types<std::int32_t, float>::type, double>);
    STATIC_REQUIRE(std::same_as<promote_types<std::int32_t, double>::type, double>);
    STATIC_REQUIRE(std::same_as<promote_types<std::int32_t, bool>::type, std::int32_t>);
    STATIC_REQUIRE(std::same_as<promote_types<std::int32_t, std::int8_t>::type, std::int32_t>);
    STATIC_REQUIRE(std::same_as<promote_types<std::int32_t, std::int16_t>::type, std::int32_t>);
    STATIC_REQUIRE(std::same_as<promote_types<std::int32_t, std::int32_t>::type, std::int32_t>);
    STATIC_REQUIRE(std::same_as<promote_types<std::int32_t, std::int64_t>::type, std::int64_t>);
    STATIC_REQUIRE(std::same_as<promote_types<std::int64_t, float>::type, double>);
    STATIC_REQUIRE(std::same_as<promote_types<std::int64_t, double>::type, double>);
    STATIC_REQUIRE(std::same_as<promote_types<std::int64_t, bool>::type, std::int64_t>);
    STATIC_REQUIRE(std::same_as<promote_types<std::int64_t, std::int8_t>::type, std::int64_t>);
    STATIC_REQUIRE(std::same_as<promote_types<std::int64_t, std::int16_t>::type, std::int64_t>);
    STATIC_REQUIRE(std::same_as<promote_types<std::int64_t, std::int32_t>::type, std::int64_t>);
    STATIC_REQUIRE(std::same_as<promote_types<std::int64_t, std::int64_t>::type, std::int64_t>);
}

}  // namespace dwave::optimization

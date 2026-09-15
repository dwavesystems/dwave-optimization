// Copyright 2025 D-Wave
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

#include <cmath>
#include <concepts>
#include <limits>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "dwave-optimization/functional.hpp"
#include "dwave-optimization/interval.hpp"
#include "dwave-optimization/typing.hpp"

namespace dwave::optimization::functional {

TEMPLATE_LIST_TEST_CASE("absolute", "", DTypes) {
    constexpr absolute op{};

    SECTION("absolute(scalar)") {
        CHECK(op(TestType(0)) == 0);
        CHECK(op(TestType(1)) == 1);

        if constexpr (std::same_as<bool, TestType>) {
            CHECK(op(true) == 1);  // abs(bool) is identity
        } else if constexpr (std::integral<TestType>) {
            CHECK(op(TestType(-1)) == 1);
            CHECK(op(TestType(-10)) == 10);
            CHECK(op(TestType(3)) == 3);
            // We define abs(lowest) == max (see functional.hpp)
            CHECK(
                op(std::numeric_limits<TestType>::lowest()) == std::numeric_limits<TestType>::max()
            );
        } else {  // floating
            CHECK(op(TestType(-1.5)) == 1.5);
            CHECK(op(TestType(1.5)) == 1.5);
        }
    }

    SECTION("absolute(interval)") {
        CHECK(not op(interval<TestType>()));  // op(empty) -> empty

        CHECK(op(interval<TestType>(0, 0)) == interval<TestType>(0, 0));
        CHECK(op(interval<TestType>(0, 1)) == interval<TestType>(0, 1));

        if constexpr (not std::same_as<TestType, bool>) {
            CHECK(op(interval<TestType>(0, 5)) == interval<TestType>(0, 5));
            CHECK(op(interval<TestType>(-7, -4)) == interval<TestType>(4, 7));
            CHECK(op(interval<TestType>(-3, 1)) == interval<TestType>(0, 3));
            CHECK(op(interval<TestType>(-1, 3)) == interval<TestType>(0, 3));
            CHECK(op(interval<TestType>(-5, 5)) == interval<TestType>(0, 5));
        }
        if constexpr (std::floating_point<TestType>) {
            CHECK(op(interval<TestType>(0.5, 5.2)) == interval<TestType>(0.5, 5.2));
            CHECK(op(interval<TestType>(-5.2, -0.5)) == interval<TestType>(0.5, 5.2));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("cos", "", DTypes) {
    constexpr cos op{};

    SECTION("cos(scalar)") {
        CHECK(op(TestType(0)) == 1);  // cos(0) == 1 exactly
        if constexpr (not std::same_as<bool, TestType>) {
            CHECK(op(TestType(1)) == std::cos(TestType(1)));
            CHECK(op(TestType(3)) == std::cos(TestType(3)));
        }
    }

    SECTION("cos(interval)") {
        CHECK(not op(interval<TestType>()));  // op(empty) -> empty
        CHECK(op(interval<TestType>(0, 0)) == interval<double>(-1, +1));
    }
}

TEMPLATE_LIST_TEST_CASE("exp", "", DTypes) {
    constexpr exp op{};

    SECTION("exp(scalar)") {
        CHECK(op(TestType(0)) == 1);  // exp(0) == 1 exactly
        if constexpr (not std::same_as<bool, TestType>) {
            CHECK(op(TestType(1)) == std::exp(TestType(1)));
            CHECK(op(TestType(-2)) == std::exp(TestType(-2)));
        }
    }

    SECTION("exp(interval)") {
        CHECK(not op(interval<TestType>()));  // op(empty) -> empty
        CHECK(op(interval<TestType>(0, 1)) == interval(op(TestType(0)), op(TestType(1))));
        if constexpr (not std::same_as<bool, TestType>) {
            CHECK(op(interval<TestType>(-2, 3)) == interval(op(TestType(-2)), op(TestType(3))));
        }
    }

    SECTION("exp domain is unrestricted") {
        CHECK(exp::domain<TestType> == interval<TestType>::all());
    }
}

TEMPLATE_LIST_TEST_CASE("expit", "", DTypes) {
    constexpr expit op{};

    SECTION("expit(scalar)") {
        CHECK(op(TestType(0)) == 0.5);  // 1 / (1 + 1)
        if constexpr (std::floating_point<TestType>) {
            // no NaN at the extremes
            CHECK(op(TestType(-1000)) == 0);
            CHECK(op(TestType(1000)) == 1);
        }
    }

    SECTION("expit(interval)") {
        CHECK(not op(interval<TestType>()));  // op(empty) -> empty
        CHECK(op(interval<TestType>(0, 1)) == interval(op(TestType(0)), op(TestType(1))));
        if constexpr (not std::same_as<bool, TestType>) {
            CHECK(op(interval<TestType>(-2, 3)) == interval(op(TestType(-2)), op(TestType(3))));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("log", "", DTypes) {
    constexpr log op{};

    SECTION("log(scalar)") {
        CHECK(op(TestType(1)) == 0);  // log(1) == 0 exactly
        if constexpr (std::same_as<bool, TestType>) {
        } else if constexpr (std::integral<TestType>) {
            CHECK(op(TestType(2)) == std::log(TestType(2)));
            CHECK(op(TestType(10)) == std::log(TestType(10)));
        } else {  // floating
            CHECK(op(TestType(2.5)) == std::log(TestType(2.5)));
            CHECK(op(TestType(0.5)) == std::log(TestType(0.5)));
        }
    }

    SECTION("log(interval)") {
        CHECK(not op(interval<TestType>()));  // op(empty) -> empty
        if constexpr (std::same_as<bool, TestType>) {
            CHECK(op(interval<TestType>(1, 1)) == interval(op(TestType(1)), op(TestType(1))));
        } else {
            CHECK(op(interval<TestType>(1, 4)) == interval(op(TestType(1)), op(TestType(4))));
            CHECK(op(interval<TestType>(2, 10)) == interval(op(TestType(2)), op(TestType(10))));
        }
    }

    SECTION("log domain is non-negative") {
        CHECK(log::domain<TestType> == interval<TestType>::nonnegative());
    }
}

TEMPLATE_LIST_TEST_CASE("logical", "", DTypes) {
    constexpr logical op{};

    SECTION("logical(<scalar>)") {
        CHECK(op(TestType(0)) == 0);

        if constexpr (std::same_as<bool, TestType>) {
            CHECK(op(true) == 1);
        } else if constexpr (std::integral<TestType>) {
            CHECK(op(TestType(-1)) == 1);
            CHECK(op(TestType(1)) == 1);
            CHECK(op(TestType(3)) == 1);
        } else {  // floating
            CHECK(op(TestType(-.000001)) == 1);
            CHECK(op(TestType(.000001)) == 1);
        }
    }

    SECTION("logical(<interval>)") {
        CHECK(not op(interval<TestType>()));  // op(null) -> null

        CHECK(op(interval<TestType>(0, 0)) == interval(false, false));
        CHECK(op(interval<TestType>(1, 1)) == interval(true, true));
        CHECK(op(interval<TestType>(0, 1)) == interval(false, true));

        if constexpr (std::same_as<bool, TestType>) {
            // already covered
        } else if constexpr (std::integral<TestType>) {
            CHECK(op(interval<TestType>(0, 5)) == interval(false, true));
            CHECK(op(interval<TestType>(1, 5)) == interval(true, true));

            CHECK(op(interval<TestType>(-3, 5)) == interval(false, true));

            CHECK(op(interval<TestType>(-3, 0)) == interval(false, true));
            CHECK(op(interval<TestType>(-3, -1)) == interval(true, true));
        } else {  // floating
            CHECK(op(interval<TestType>(0, .00001)) == interval(false, true));
            CHECK(op(interval<TestType>(.000001, 5.5)) == interval(true, true));

            CHECK(op(interval<TestType>(-3.4, 13.2)) == interval(false, true));

            CHECK(op(interval<TestType>(-.00000001, 0)) == interval(false, true));
            CHECK(op(interval<TestType>(-3.3, -.01)) == interval(true, true));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("logical_not", "", DTypes) {
    constexpr logical_not op{};

    SECTION("logical_not(<scalar>)") {
        CHECK(op(TestType(0)) == 1);

        if constexpr (std::same_as<bool, TestType>) {
            CHECK(op(true) == 0);
        } else if constexpr (std::integral<TestType>) {
            CHECK(op(TestType(-1)) == 0);
            CHECK(op(TestType(1)) == 0);
            CHECK(op(TestType(3)) == 0);
        } else {  // floating
            CHECK(op(TestType(-.000001)) == 0);
            CHECK(op(TestType(.000001)) == 0);
        }
    }

    SECTION("logical_not(<interval>)") {
        CHECK(not op(interval<TestType>()));  // op(null) -> null

        CHECK(op(interval<TestType>(0, 0)) == interval(true, true));
        CHECK(op(interval<TestType>(1, 1)) == interval(false, false));
        CHECK(op(interval<TestType>(0, 1)) == interval(false, true));

        if constexpr (std::same_as<bool, TestType>) {
            // already covered
        } else if constexpr (std::integral<TestType>) {
            CHECK(op(interval<TestType>(0, 5)) == interval(false, true));
            CHECK(op(interval<TestType>(1, 5)) == interval(false, false));

            CHECK(op(interval<TestType>(-3, 5)) == interval(false, true));

            CHECK(op(interval<TestType>(-3, 0)) == interval(false, true));
            CHECK(op(interval<TestType>(-3, -1)) == interval(false, false));
        } else {  // floating
            CHECK(op(interval<TestType>(0, .00001)) == interval(false, true));
            CHECK(op(interval<TestType>(.000001, 5.5)) == interval(false, false));

            CHECK(op(interval<TestType>(-3.4, 13.2)) == interval(false, true));

            CHECK(op(interval<TestType>(-.00000001, 0)) == interval(false, true));
            CHECK(op(interval<TestType>(-3.3, -.01)) == interval(false, false));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("modulus", "", DTypes) {
    constexpr modulus<TestType> op;

    CHECK(op(1, 0) == 0);
    CHECK(op(0, 1) == 0);

    if constexpr (not std::same_as<TestType, bool>) {
        CHECK(op(-1, 0) == 0);
        CHECK(op(0, -1) == 0);

        CHECK(op(-1, -10) == -1);
        CHECK(op(-1, 10) == 9);
        CHECK(op(1, -10) == -9);
    }
}

TEMPLATE_LIST_TEST_CASE("negative", "", DTypes) {
    constexpr negative op{};
    if constexpr (not std::same_as<TestType, bool>) {
        SECTION("negative(scalar)") {
            CHECK(op(TestType(0)) == 0);

            if constexpr (std::integral<TestType>) {
                CHECK(op(TestType(3)) == -3);
                CHECK(op(TestType(-3)) == 3);
            } else {  // floating
                CHECK(op(TestType(1.5)) == -1.5);
                CHECK(op(TestType(-1.5)) == 1.5);
            }
        }

        SECTION("negative(interval)") {
            CHECK(not op(interval<TestType>()));  // op(empty) -> empty

            CHECK(op(interval<TestType>(0, 1)) == interval(op(TestType(1)), op(TestType(0))));
            CHECK(op(interval<TestType>(-2, 3)) == interval(op(TestType(3)), op(TestType(-2))));
            CHECK(op(interval<TestType>(-5, -1)) == interval(op(TestType(-1)), op(TestType(-5))));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("rint", "", DTypes) {
    constexpr rint op{};

    SECTION("rint(scalar)") {
        CHECK(op(TestType(0)) == 0);
        if constexpr (std::same_as<bool, TestType>) {
            CHECK(op(true) == 1);
        } else if constexpr (std::integral<TestType>) {
            CHECK(op(TestType(3)) == 3);
            CHECK(op(TestType(-4)) == -4);
        } else {  // floating: rounds half to even
            CHECK(op(TestType(2.5)) == 2);
            CHECK(op(TestType(3.5)) == 4);
            CHECK(op(TestType(-2.5)) == -2);
            CHECK(op(TestType(2.4)) == std::rint(TestType(2.4)));
        }
    }

    SECTION("rint(interval)") {
        CHECK(not op(interval<TestType>()));  // op(empty) -> empty
        CHECK(op(interval<TestType>(0, 1)) == interval(op(TestType(0)), op(TestType(1))));
        if constexpr (not std::same_as<bool, TestType>) {
            CHECK(op(interval<TestType>(-3, 4)) == interval(op(TestType(-3)), op(TestType(4))));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("sin", "", DTypes) {
    constexpr sin op{};

    SECTION("sin(scalar)") {
        CHECK(op(TestType(0)) == 0);  // sin(0) == 0 exactly
        if constexpr (not std::same_as<bool, TestType>) {
            CHECK(op(TestType(1)) == std::sin(TestType(1)));
            CHECK(op(TestType(2)) == std::sin(TestType(2)));
        }
    }

    SECTION("sin(interval)") {
        CHECK(not op(interval<TestType>()));  // op(empty) -> empty
        CHECK(op(interval<TestType>(0, 0)) == interval<double>(-1, +1));
    }
}

TEMPLATE_LIST_TEST_CASE("square", "", DTypes) {
    constexpr square op{};

    SECTION("square(scalar)") {
        CHECK(op(TestType(0)) == 0);
        CHECK(op(TestType(1)) == 1);
        if constexpr (std::same_as<bool, TestType>) {
            // square(bool) is identity
        } else if constexpr (std::integral<TestType>) {
            CHECK(op(TestType(3)) == 9);
            CHECK(op(TestType(-3)) == 9);
            CHECK(op(TestType(4)) == 16);
        } else {  // floating
            CHECK(op(TestType(2.5)) == 6.25);
            CHECK(op(TestType(-1.5)) == 2.25);
        }
    }

    SECTION("square(interval)") {
        CHECK(not op(interval<TestType>()));  // op(empty) -> empty

        if constexpr (std::same_as<bool, TestType>) {
            CHECK(op(interval<bool>(0, 1)) == interval<bool>(0, 1));
        } else {
            CHECK(op(interval<TestType>(2, 3)) == interval(op(TestType(2)), op(TestType(3))));
            CHECK(op(interval<TestType>(-3, -2)) == interval(op(TestType(-2)), op(TestType(-3))));
            CHECK(op(interval<TestType>(-3, 2)) == interval(TestType(0), op(TestType(-3))));
            CHECK(op(interval<TestType>(-2, 3)) == interval(TestType(0), op(TestType(3))));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("square_root", "", DTypes) {
    constexpr square_root op{};

    SECTION("square_root(scalar)") {
        CHECK(op(TestType(0)) == 0);
        CHECK(op(TestType(1)) == 1);
        if constexpr (not std::same_as<bool, TestType>) {
            CHECK(op(TestType(4)) == 2);
            CHECK(op(TestType(9)) == 3);
            if constexpr (std::floating_point<TestType>) {
                CHECK(op(TestType(2.0)) == std::sqrt(TestType(2.0)));
            }
        }
    }

    SECTION("square_root(interval)") {
        CHECK(not op(interval<TestType>()));  // op(empty) -> empty
        CHECK(op(interval<TestType>(0, 1)) == interval(op(TestType(0)), op(TestType(1))));
        if constexpr (not std::same_as<bool, TestType>) {
            CHECK(op(interval<TestType>(0, 4)) == interval(op(TestType(0)), op(TestType(4))));
            CHECK(op(interval<TestType>(1, 9)) == interval(op(TestType(1)), op(TestType(9))));
        }
    }

    SECTION("square_root domain is non-negative") {
        CHECK(square_root::domain<TestType> == interval<TestType>::nonnegative());
    }
}

TEMPLATE_LIST_TEST_CASE("tanh", "", DTypes) {
    constexpr tanh op{};

    SECTION("tanh(scalar)") {
        CHECK(op(TestType(0)) == 0);  // tanh(0) == 0 exactly
        if constexpr (not std::same_as<bool, TestType>) {
            CHECK(op(TestType(1)) == std::tanh(TestType(1)));
            CHECK(op(TestType(-2)) == std::tanh(TestType(-2)));
        }
    }

    SECTION("tanh(interval)") {
        CHECK(not op(interval<TestType>()));  // op(empty) -> empty
        CHECK(op(interval<TestType>(0, 1)) == interval(op(TestType(0)), op(TestType(1))));
        if constexpr (not std::same_as<bool, TestType>) {
            CHECK(op(interval<TestType>(-2, 3)) == interval(op(TestType(-2)), op(TestType(3))));
        }
    }
}

}  // namespace dwave::optimization::functional

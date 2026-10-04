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
    // Our various complilers are not in agreement about whether std::abs() is
    // constexpr or not, so we use CHECK() rather than STATIC_REQUIRE.

    constexpr absolute op{};

    using limits = std::numeric_limits<TestType>;

    SECTION("op(scalar)") {
        STATIC_REQUIRE(std::same_as<decltype(op(TestType())), TestType>);

        CHECK(op(TestType{0}) == TestType{0});
        CHECK(op(TestType{1}) == TestType{1});

        if constexpr (std::same_as<bool, TestType>) {
            // already covered
        } else if constexpr (std::signed_integral<TestType>) {
            CHECK(op(TestType{-1}) == TestType{1});
            CHECK(op(TestType{-10}) == TestType{10});
            CHECK(op(TestType{3}) == TestType{3});

            CHECK(op(limits::lowest()) == limits::max());  // we define this to be true
            CHECK(op(limits::max()) == limits::max());
        } else {  // floating
            CHECK(op(TestType{-1.5}) == TestType{1.5});
            CHECK(op(TestType{1.5}) == TestType{1.5});
        }

        if constexpr (limits::has_infinity) {
            CHECK(op(-limits::infinity()) == limits::infinity());
            CHECK(op(+limits::infinity()) == limits::infinity());
        }
    }

    SECTION("op(interval)") {
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
    // dev note: std::cos() isn't constexpr until C++26 so we need to use CHECK().

    constexpr cos op{};

    using limits = std::numeric_limits<TestType>;

    SECTION("op(scalar)") {
        CHECK(op(TestType{0}) == TestType{1});  // cos(0) == 1 exactly

        // Following NumPy, if can be cast to float it will be, otherwise it'll be a double
        if constexpr (can_cast<TestType, float>) {
            STATIC_REQUIRE(std::same_as<decltype(op(TestType())), float>);

            CHECK(op(TestType{1}) == std::cosf(1));

            if constexpr (not std::same_as<TestType, bool>) {
                CHECK(op(TestType{3}) == std::cosf(3));
            }

        } else {
            STATIC_REQUIRE(std::same_as<decltype(op(TestType())), double>);

            CHECK(op(TestType{1}) == std::cos(1.0));
            CHECK(op(TestType{3}) == std::cos(3.0));
        }

        if constexpr (limits::has_infinity) {
            STATIC_REQUIRE(op(-limits::infinity()) == 0);
            STATIC_REQUIRE(op(+limits::infinity()) == 0);
        }
    }

    SECTION("op(interval)") {
        if constexpr (can_cast<TestType, float>) {
            STATIC_REQUIRE(std::same_as<decltype(op(interval<TestType>())), interval<float>>);

            CHECK(op(interval<TestType>(0, 0)) == interval<float>(-1, +1));
            CHECK(op(interval<TestType>::all()) == interval<float>(-1, +1));
        } else {
            STATIC_REQUIRE(std::same_as<decltype(op(interval<TestType>())), interval<double>>);

            CHECK(op(interval<TestType>(0, 0)) == interval<double>(-1, +1));
            CHECK(op(interval<TestType>::all()) == interval<double>(-1, +1));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("exp", "", DTypes) {
    // dev note: std::exp() isn't constexpr until C++26 so we need to use CHECK().

    constexpr exp op{};

    using limits = std::numeric_limits<TestType>;

    SECTION("op(scalar)") {
        // Following NumPy, if can be cast to float it will be, otherwise it'll be a double
        if constexpr (can_cast<TestType, float>) {
            STATIC_REQUIRE(std::same_as<decltype(op(TestType())), float>);

            CHECK(op(TestType{1}) == std::expf(1));

            if constexpr (not std::same_as<TestType, bool>) {
                CHECK(op(TestType{3}) == std::expf(3));
            }

        } else {
            STATIC_REQUIRE(std::same_as<decltype(op(TestType())), double>);

            CHECK(op(TestType{1}) == std::exp(1.0));
            CHECK(op(TestType{3}) == std::exp(3.0));
        }

        CHECK(op(TestType{0}) == 1);  // exp(0) == 1 exactly

        if constexpr (limits::has_infinity) {
            CHECK(op(-limits::infinity()) == 0);
            CHECK(op(+limits::infinity()) == limits::infinity());
        }
    }

    SECTION("op(interval)") {
        CHECK(op(interval<TestType>(0, 1)) == interval(op(TestType{0}), op(TestType{1})));
        if constexpr (not std::same_as<bool, TestType>) {
            CHECK(op(interval<TestType>(-2, 3)) == interval(op(TestType{-2}), op(TestType{3})));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("expit", "", DTypes) {
    // dev note: std::exp() isn't constexpr until C++26 so we need to use CHECK().

    constexpr expit op{};

    using limits = std::numeric_limits<TestType>;

    SECTION("op(scalar)") {
        // Following SciPy, if can be cast to float it will be, otherwise it'll be a double
        if constexpr (can_cast<TestType, float>) {
            STATIC_REQUIRE(std::same_as<decltype(op(TestType())), float>);
        } else {
            STATIC_REQUIRE(std::same_as<decltype(op(TestType())), double>);
        }

        if constexpr (std::floating_point<TestType>) {
            // no NaN at the extremes
            CHECK(op(TestType{-1000}) == 0);
            CHECK(op(TestType{1000}) == 1);

            CHECK(op(limits::lowest()) == 0);
            CHECK(op(limits::max()) == 1);

            CHECK(op(-limits::infinity()) == 0);
            CHECK(op(+limits::infinity()) == 1);
        }

        CHECK(op(TestType{0}) == 0.5);  // 1 / (1 + 1)
    }

    SECTION("op(interval)") {
        CHECK(op(interval<TestType>(0, 1)) == interval(op(TestType{0}), op(TestType{1})));
        if constexpr (not std::same_as<bool, TestType>) {
            CHECK(op(interval<TestType>(-2, 3)) == interval(op(TestType{-2}), op(TestType{3})));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("log", "", DTypes) {
    constexpr log op{};

    using limits = std::numeric_limits<TestType>;

    SECTION("op(scalar)") {
        // Following NumPy, if can be cast to float it will be, otherwise it'll be a double
        if constexpr (can_cast<TestType, float>) {
            STATIC_REQUIRE(std::same_as<decltype(op(TestType())), float>);

            CHECK(op(TestType{1}) == std::logf(1));

            if constexpr (not std::same_as<TestType, bool>) {
                CHECK(op(TestType{3}) == std::logf(3));
            }

        } else {
            STATIC_REQUIRE(std::same_as<decltype(op(TestType())), double>);

            CHECK(op(TestType{1}) == std::log(1.0));
            CHECK(op(TestType{3}) == std::log(3.0));
        }

        CHECK(op(TestType{0}) == -std::numeric_limits<float>::infinity());

        if constexpr (limits::has_infinity) {
            CHECK(op(limits::infinity()) == limits::infinity());
        }
    }

    SECTION("op(interval)") {
        if constexpr (std::floating_point<TestType>) {
            CHECK(op(interval<TestType>::nonnegative()) == interval<TestType>::all());
        }
    }
}

TEMPLATE_LIST_TEST_CASE("logical", "", DTypes) {
    constexpr logical op{};

    using limits = std::numeric_limits<TestType>;

    SECTION("logical(<scalar>)") {
        STATIC_REQUIRE(op(TestType{0}) == 0);

        if constexpr (std::same_as<bool, TestType>) {
            STATIC_REQUIRE(op(true) == 1);
        } else if constexpr (std::signed_integral<TestType>) {
            STATIC_REQUIRE(op(TestType{-1}) == 1);
            STATIC_REQUIRE(op(TestType{1}) == 1);
            STATIC_REQUIRE(op(TestType{3}) == 1);
        } else if constexpr (std::floating_point<TestType>) {
            STATIC_REQUIRE(op(TestType{-.000001}) == 1);
            STATIC_REQUIRE(op(TestType{.000001}) == 1);

            STATIC_REQUIRE(op(-limits::infinity()));
            STATIC_REQUIRE(op(+limits::infinity()));
        } else {
            static_assert(false, "unexpected type");
        }
    }

    SECTION("logical(<interval>)") {
        STATIC_REQUIRE(op(interval<TestType>(0, 0)) == interval(false, false));
        STATIC_REQUIRE(op(interval<TestType>(0, 1)) == interval(false, true));
        STATIC_REQUIRE(op(interval<TestType>(1, 1)) == interval(true, true));

        if constexpr (std::same_as<bool, TestType>) {
            // already covered
        } else if constexpr (std::signed_integral<TestType>) {
            STATIC_REQUIRE(op(interval<TestType>(0, 5)) == interval(false, true));
            STATIC_REQUIRE(op(interval<TestType>(1, 5)) == interval(true, true));

            STATIC_REQUIRE(op(interval<TestType>(-3, 5)) == interval(false, true));

            STATIC_REQUIRE(op(interval<TestType>(-3, 0)) == interval(false, true));
            STATIC_REQUIRE(op(interval<TestType>(-3, -1)) == interval(true, true));
        } else if constexpr (std::floating_point<TestType>) {
            STATIC_REQUIRE(op(interval<TestType>(0, .00001)) == interval(false, true));
            STATIC_REQUIRE(op(interval<TestType>(.000001, 5.5)) == interval(true, true));

            STATIC_REQUIRE(op(interval<TestType>(-3.4, 13.2)) == interval(false, true));

            STATIC_REQUIRE(op(interval<TestType>(-.00000001, 0)) == interval(false, true));
            STATIC_REQUIRE(op(interval<TestType>(-3.3, -.01)) == interval(true, true));
        } else {
            static_assert(false, "unexpected type");
        }
    }
}

TEMPLATE_LIST_TEST_CASE("logical_not", "", DTypes) {
    constexpr logical_not op{};

    using limits = std::numeric_limits<TestType>;

    SECTION("logical_not(<scalar>)") {
        STATIC_REQUIRE(op(TestType{0}) == 1);

        if constexpr (std::same_as<bool, TestType>) {
            STATIC_REQUIRE(op(true) == 0);
        } else if constexpr (std::signed_integral<TestType>) {
            STATIC_REQUIRE(op(TestType{-1}) == 0);
            STATIC_REQUIRE(op(TestType{1}) == 0);
            STATIC_REQUIRE(op(TestType{3}) == 0);
        } else if constexpr (std::floating_point<TestType>) {
            STATIC_REQUIRE(op(TestType{-.000001}) == 0);
            STATIC_REQUIRE(op(TestType{.000001}) == 0);

            STATIC_REQUIRE(not op(-limits::infinity()));
            STATIC_REQUIRE(not op(+limits::infinity()));
        } else {
            static_assert(false, "unexpected type");
        }
    }

    SECTION("logical_not(<interval>)") {
        STATIC_REQUIRE(op(interval<TestType>(0, 0)) == interval(true, true));
        STATIC_REQUIRE(op(interval<TestType>(0, 1)) == interval(false, true));
        STATIC_REQUIRE(op(interval<TestType>(1, 1)) == interval(false, false));

        if constexpr (std::same_as<bool, TestType>) {
            // already covered
        } else if constexpr (std::signed_integral<TestType>) {
            STATIC_REQUIRE(op(interval<TestType>(0, 5)) == interval(false, true));
            STATIC_REQUIRE(op(interval<TestType>(1, 5)) == interval(false, false));

            STATIC_REQUIRE(op(interval<TestType>(-3, 5)) == interval(false, true));

            STATIC_REQUIRE(op(interval<TestType>(-3, 0)) == interval(false, true));
            STATIC_REQUIRE(op(interval<TestType>(-3, -1)) == interval(false, false));
        } else if constexpr (std::floating_point<TestType>) {
            STATIC_REQUIRE(op(interval<TestType>(0, .00001)) == interval(false, true));
            STATIC_REQUIRE(op(interval<TestType>(.000001, 5.5)) == interval(false, false));

            STATIC_REQUIRE(op(interval<TestType>(-3.4, 13.2)) == interval(false, true));

            STATIC_REQUIRE(op(interval<TestType>(-.00000001, 0)) == interval(false, true));
            STATIC_REQUIRE(op(interval<TestType>(-3.3, -.01)) == interval(false, false));
        } else {
            static_assert(false, "unexpected type");
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

    using limits = std::numeric_limits<TestType>;

    if constexpr (not std::same_as<TestType, bool>) {
        SECTION("op(scalar)") {
            STATIC_REQUIRE(op(TestType{0}) == 0);

            if constexpr (std::integral<TestType>) {
                STATIC_REQUIRE(op(TestType{3}) == -3);
                STATIC_REQUIRE(op(TestType{-3}) == 3);
            } else {  // floating
                STATIC_REQUIRE(op(TestType{1.5}) == -1.5);
                STATIC_REQUIRE(op(TestType{-1.5}) == 1.5);

                STATIC_REQUIRE(op(-limits::infinity()) == +limits::infinity());
                STATIC_REQUIRE(op(+limits::infinity()) == -limits::infinity());
            }
        }

        SECTION("op(interval)") {
            STATIC_REQUIRE(
                op(interval<TestType>(0, 1)) == interval(op(TestType{1}), op(TestType{0}))
            );
            STATIC_REQUIRE(
                op(interval<TestType>(-2, 3)) == interval(op(TestType{3}), op(TestType{-2}))
            );
            STATIC_REQUIRE(
                op(interval<TestType>(-5, -1)) == interval(op(TestType{-1}), op(TestType{-5}))
            );
        }
    }
}

TEMPLATE_LIST_TEST_CASE("rint", "", DTypes) {
    constexpr rint op{};

    using limits = std::numeric_limits<TestType>;

    SECTION("rint(scalar)") {
        CHECK(op(TestType{0}) == 0);
        if constexpr (std::same_as<bool, TestType>) {
            CHECK(op(true) == 1);
        } else if constexpr (std::signed_integral<TestType>) {
            CHECK(op(TestType{3}) == 3);
            CHECK(op(TestType{-4}) == -4);
        } else {  // floating: rounds half to even
            CHECK(op(TestType{2.5}) == 2);
            CHECK(op(TestType{3.5}) == 4);
            CHECK(op(TestType{-2.5}) == -2);
            CHECK(op(TestType{2.4}) == std::rint(TestType{2.4}));

            CHECK(op(-limits::infinity()) == -limits::infinity());
            CHECK(op(+limits::infinity()) == +limits::infinity());
        }
    }

    SECTION("rint(interval)") {
        CHECK(op(interval<TestType>(0, 1)) == interval(op(TestType{0}), op(TestType{1})));
        if constexpr (not std::same_as<bool, TestType>) {
            CHECK(op(interval<TestType>(-3, 4)) == interval(op(TestType{-3}), op(TestType{4})));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("sin", "", DTypes) {
    // dev note: std::sin() isn't constexpr until C++26 so we need to use CHECK().

    constexpr sin op{};

    using limits = std::numeric_limits<TestType>;

    SECTION("op(scalar)") {
        // Following NumPy, if can be cast to float it will be, otherwise it'll be a double
        if constexpr (can_cast<TestType, float>) {
            STATIC_REQUIRE(std::same_as<decltype(op(TestType())), float>);

            CHECK(op(TestType{1}) == std::sinf(1));

            if constexpr (not std::same_as<TestType, bool>) {
                CHECK(op(TestType{3}) == std::sinf(3));
            }

        } else {
            STATIC_REQUIRE(std::same_as<decltype(op(TestType())), double>);

            CHECK(op(TestType{1}) == std::sin(1.0));
            CHECK(op(TestType{3}) == std::sin(3.0));
        }

        CHECK(op(TestType{0}) == TestType{0});  // sin(0) == 0 exactly

        if constexpr (std::floating_point<TestType>) {
            STATIC_REQUIRE(op(-limits::infinity()) == 0);
            STATIC_REQUIRE(op(+limits::infinity()) == 0);
        }
    }

    SECTION("op(interval)") {
        if constexpr (can_cast<TestType, float>) {
            STATIC_REQUIRE(std::same_as<decltype(op(interval<TestType>())), interval<float>>);

            CHECK(op(interval<TestType>(0, 0)) == interval<float>(-1, +1));
            CHECK(op(interval<TestType>::all()) == interval<float>(-1, +1));
        } else {
            STATIC_REQUIRE(std::same_as<decltype(op(interval<TestType>())), interval<double>>);

            CHECK(op(interval<TestType>(0, 0)) == interval<double>(-1, +1));
            CHECK(op(interval<TestType>::all()) == interval<double>(-1, +1));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("sqrt", "", DTypes) {
    // dev note: std::sqrt() isn't constexpr until C++26 so we need to use CHECK().

    constexpr sqrt op{};

    using limits = std::numeric_limits<TestType>;

    SECTION("op(scalar)") {
        CHECK(op(TestType{0}) == 0);
        CHECK(op(TestType{1}) == 1);
        if constexpr (not std::same_as<bool, TestType>) {
            CHECK(op(TestType{4}) == 2);
            CHECK(op(TestType{9}) == 3);
            if constexpr (std::floating_point<TestType>) {
                CHECK(op(TestType{2.0}) == std::sqrt(TestType{2.0}));

                CHECK(op(limits::infinity()) == limits::infinity());
            }
        }
    }

    SECTION("op(interval)") {
        CHECK(op(interval<TestType>(0, 1)) == interval(op(TestType{0}), op(TestType{1})));
        if constexpr (not std::same_as<bool, TestType>) {
            CHECK(op(interval<TestType>(0, 4)) == interval(op(TestType{0}), op(TestType{4})));
            CHECK(op(interval<TestType>(1, 9)) == interval(op(TestType{1}), op(TestType{9})));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("square", "", DTypes) {
    constexpr square op{};

    using limits = std::numeric_limits<TestType>;

    SECTION("square(scalar)") {
        STATIC_REQUIRE(op(TestType{0}) == 0);
        STATIC_REQUIRE(op(TestType{1}) == 1);

        if constexpr (std::same_as<bool, TestType>) {
            // square(bool) is identity
        } else if constexpr (std::signed_integral<TestType>) {
            STATIC_REQUIRE(op(TestType{3}) == 9);
            STATIC_REQUIRE(op(TestType{-3}) == 9);
            STATIC_REQUIRE(op(TestType{4}) == 16);

            // saturating
            STATIC_REQUIRE(op(limits::max()) == limits::max());
            STATIC_REQUIRE(op(limits::min()) == limits::max());

        } else if constexpr (std::floating_point<TestType>) {
            STATIC_REQUIRE(op(TestType{2.5}) == 6.25);
            STATIC_REQUIRE(op(TestType{-1.5}) == 2.25);

            STATIC_REQUIRE(op(-limits::infinity()) == +limits::infinity());
            STATIC_REQUIRE(op(+limits::infinity()) == +limits::infinity());
        } else {
            static_assert(false, "unexpected dtype");
        }
    }

    SECTION("op(interval)") {
        if constexpr (std::same_as<bool, TestType>) {
            CHECK(op(interval<bool>(0, 1)) == interval<bool>(0, 1));
        } else {
            CHECK(op(interval<TestType>(2, 3)) == interval(op(TestType{2}), op(TestType{3})));
            CHECK(op(interval<TestType>(-3, -2)) == interval(op(TestType{-2}), op(TestType{-3})));
            CHECK(op(interval<TestType>(-3, 2)) == interval(TestType{0}, op(TestType{-3})));
            CHECK(op(interval<TestType>(-2, 3)) == interval(TestType{0}, op(TestType{3})));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("tanh", "", DTypes) {
    // dev note: std::tanh() isn't constexpr until C++26 so we need to use CHECK().

    constexpr tanh op{};

    using limits = std::numeric_limits<TestType>;

    SECTION("tanh(scalar)") {
        CHECK(op(TestType{0}) == 0);  // tanh(0) == 0 exactly
        if constexpr (not std::same_as<bool, TestType>) {
            if constexpr (can_cast<TestType, float>) {
                CHECK(op(TestType{1}) == std::tanhf(TestType{1}));
                CHECK(op(TestType{-2}) == std::tanhf(TestType{-2}));
            } else {
                CHECK(op(TestType{1}) == std::tanh(TestType{1}));
                CHECK(op(TestType{-2}) == std::tanh(TestType{-2}));
            }
        }

        if constexpr (std::floating_point<TestType>) {
            CHECK(op(-limits::infinity()) == -1);
            CHECK(op(+limits::infinity()) == +1);
        }
    }

    SECTION("tanh(interval)") {
        CHECK(op(interval<TestType>(0, 1)) == interval(op(TestType{0}), op(TestType{1})));
        if constexpr (not std::same_as<bool, TestType>) {
            CHECK(op(interval<TestType>(-2, 3)) == interval(op(TestType{-2}), op(TestType{3})));
        }
    }
}

}  // namespace dwave::optimization::functional

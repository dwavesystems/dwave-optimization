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

namespace mixins {
namespace {

// Test doubles exercising each monotonicity combination of the mixin's corner
// selection. Deliberately trivial so the expected intervals are obvious.
struct test_add : mixins::BinaryOpMixin<test_add> {  // f(x, y) = x + y
    static constexpr std::int32_t operator()(std::int32_t lhs, std::int32_t rhs) {
        return lhs + rhs;
    }
    using BinaryOpMixin::operator();
    static constexpr std::array<Monotonicity, 2> monotonicity{
        Monotonicity::Increasing,
        Monotonicity::Increasing
    };
};

struct test_sub : mixins::BinaryOpMixin<test_sub> {  // f(x, y) = x - y
    static constexpr std::int32_t operator()(std::int32_t lhs, std::int32_t rhs) {
        return lhs - rhs;
    }
    using BinaryOpMixin::operator();
    static constexpr std::array<Monotonicity, 2> monotonicity{
        Monotonicity::Increasing,
        Monotonicity::Decreasing
    };
};

struct test_rsub : mixins::BinaryOpMixin<test_rsub> {  // f(x, y) = y - x
    static constexpr std::int32_t operator()(std::int32_t lhs, std::int32_t rhs) {
        return rhs - lhs;
    }
    using BinaryOpMixin::operator();
    static constexpr std::array<Monotonicity, 2> monotonicity{
        Monotonicity::Decreasing,
        Monotonicity::Increasing
    };
};

struct test_nadd : mixins::BinaryOpMixin<test_nadd> {  // f(x, y) = -x - y
    static constexpr std::int32_t operator()(std::int32_t lhs, std::int32_t rhs) {
        return -lhs - rhs;
    }
    using BinaryOpMixin::operator();
    static constexpr std::array<Monotonicity, 2> monotonicity{
        Monotonicity::Decreasing,
        Monotonicity::Decreasing
    };
};

}  // namespace

TEST_CASE("BinaryOpMixin corner selection") {
    constexpr interval<std::int32_t> lhs(1, 2);
    constexpr interval<std::int32_t> rhs(10, 20);

    // Each combination picks a different pair of corners, so a transposed endpoint
    // in any of the four ternaries changes at least one of these
    STATIC_REQUIRE(test_add{}(lhs, rhs) == interval<std::int32_t>(11, 22));
    STATIC_REQUIRE(test_sub{}(lhs, rhs) == interval<std::int32_t>(-19, -8));
    STATIC_REQUIRE(test_rsub{}(lhs, rhs) == interval<std::int32_t>(8, 19));
    STATIC_REQUIRE(test_nadd{}(lhs, rhs) == interval<std::int32_t>(-22, -11));
}

}  // namespace mixins

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

TEMPLATE_LIST_TEST_CASE("add", "", DTypes) {
    constexpr add op{};

    using limits = std::numeric_limits<TestType>;

    SECTION("add(scalar, scalar)") {
        // dtype is preseved
        STATIC_REQUIRE(std::same_as<decltype(op(TestType{0}, TestType{0})), TestType>);

        STATIC_REQUIRE(op(TestType{1}, TestType{0}) == TestType{1});
        STATIC_REQUIRE(op(TestType{0}, TestType{1}) == TestType{1});

        if constexpr (std::signed_integral<TestType>) {
            STATIC_REQUIRE(op(limits::max(), TestType{1}) == limits::max());
            STATIC_REQUIRE(op(limits::max(), TestType{0}) == limits::max());
            STATIC_REQUIRE(op(limits::max(), TestType{-1}) == limits::max() - 1);
            STATIC_REQUIRE(op(limits::max(), limits::max()) == limits::max());
            STATIC_REQUIRE(op(limits::lowest(), TestType{-1}) == limits::lowest());
            STATIC_REQUIRE(op(limits::lowest(), limits::lowest()) == limits::lowest());
        }

        if constexpr (std::floating_point<TestType>) {
            STATIC_REQUIRE(op(-limits::infinity(), -limits::infinity()) == -limits::infinity());
            STATIC_REQUIRE(op(-limits::infinity(), +limits::infinity()) == +limits::infinity());
            STATIC_REQUIRE(op(+limits::infinity(), -limits::infinity()) == +limits::infinity());
            STATIC_REQUIRE(op(+limits::infinity(), +limits::infinity()) == +limits::infinity());
        }
    }

    SECTION("op(interval, interval)") {
        STATIC_REQUIRE(
            op(interval<TestType>(0, 5), interval<TestType>(1, 2)) == interval<TestType>(1, 7)
        );
        STATIC_REQUIRE(
            op(interval<TestType>::all(), interval<TestType>::all()) == interval<TestType>::all()
        );
        STATIC_REQUIRE(
            op(interval<TestType>::nonnegative(), interval<TestType>::all()) ==
            interval<TestType>::all()
        );
        STATIC_REQUIRE(
            op(interval<TestType>::nonnegative(), interval<TestType>::nonnegative()) ==
            interval<TestType>::nonnegative()
        );
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

TEMPLATE_LIST_TEST_CASE("divide", "", DTypes) {
    constexpr divide op{};

    using limits = std::numeric_limits<TestType>;

    SECTION("op(scalar, scalar)") {
        STATIC_REQUIRE(op(TestType{0}, TestType{0}) == 0);  // safe division
        STATIC_REQUIRE(op(TestType{0}, TestType{1}) == 0);
        STATIC_REQUIRE(op(TestType{1}, TestType{0}) == 0);  // safe division
        STATIC_REQUIRE(op(TestType{1}, TestType{1}) == 1);

        if constexpr (std::same_as<TestType, bool>) {
            // bool is promoted to double
            STATIC_REQUIRE(std::same_as<double, decltype(op(TestType(), TestType()))>);
        } else if constexpr (std::signed_integral<TestType>) {
            // all integers are promoted to double
            STATIC_REQUIRE(std::same_as<double, decltype(op(TestType(), TestType()))>);

            // these are mostly sanity checks, the rest is covered in the floating branch
            STATIC_REQUIRE(op(TestType{1}, TestType{1}) == 1);
            STATIC_REQUIRE(op(TestType{1}, TestType{2}) == 0.5);
            STATIC_REQUIRE(op(TestType{6}, TestType{3}) == 2);

        } else if constexpr (std::floating_point<TestType>) {
            // floating dtype is preserved
            STATIC_REQUIRE(std::same_as<TestType, decltype(op(TestType(), TestType()))>);

            constexpr TestType inf = limits::infinity();

            STATIC_REQUIRE(op(TestType{1}, TestType{1}) == 1);
            STATIC_REQUIRE(op(TestType{1}, TestType{2}) == 0.5);
            STATIC_REQUIRE(op(TestType{6}, TestType{3}) == 2);

            STATIC_REQUIRE(op(TestType{-1}, TestType{2}) == TestType{-0.5});
            STATIC_REQUIRE(op(TestType{1}, TestType{-2}) == TestType{-0.5});
            STATIC_REQUIRE(op(TestType{-1}, TestType{-2}) == TestType{0.5});
            STATIC_REQUIRE(op(TestType{-6}, TestType{-3}) == TestType{2});
            // x / 0 := 0, including for -0.0 since -0.0 == 0.0
            STATIC_REQUIRE(op(TestType{0}, TestType{0}) == TestType{0});
            STATIC_REQUIRE(op(TestType{5}, TestType{0}) == TestType{0});
            STATIC_REQUIRE(op(TestType{-5}, TestType{0}) == TestType{0});
            STATIC_REQUIRE(op(TestType{5}, TestType{-0.0}) == TestType{0});
            STATIC_REQUIRE(op(TestType{-5}, TestType{-0.0}) == TestType{0});
            STATIC_REQUIRE(op(limits::max(), TestType{0}) == TestType{0});
            STATIC_REQUIRE(op(inf, TestType{0}) == TestType{0});
            STATIC_REQUIRE(op(-inf, TestType{0}) == TestType{0});

            // safe division keeps the usual sign rule"
            STATIC_REQUIRE(not std::signbit(op(TestType{5}, TestType{0.0})));
            STATIC_REQUIRE(std::signbit(op(TestType{5}, TestType{-0.0})));
            STATIC_REQUIRE(std::signbit(op(TestType{-5}, TestType{0.0})));
            STATIC_REQUIRE(not std::signbit(op(TestType{-5}, TestType{-0.0})));

            // both operands zero
            STATIC_REQUIRE(not std::signbit(op(TestType{0.0}, TestType{0.0})));
            STATIC_REQUIRE(std::signbit(op(TestType{-0.0}, TestType{0.0})));
            STATIC_REQUIRE(std::signbit(op(TestType{0.0}, TestType{-0.0})));
            STATIC_REQUIRE(not std::signbit(op(TestType{-0.0}, TestType{-0.0})));

            // an infinite numerator over zero
            STATIC_REQUIRE(not std::signbit(op(inf, TestType{0.0})));
            STATIC_REQUIRE(std::signbit(op(inf, TestType{-0.0})));
            STATIC_REQUIRE(std::signbit(op(-inf, TestType{0.0})));
            STATIC_REQUIRE(not std::signbit(op(-inf, TestType{-0.0})));

            // signed zero results from ordinary division
            STATIC_REQUIRE(not std::signbit(op(TestType{0.0}, TestType{1})));
            STATIC_REQUIRE(std::signbit(op(TestType{-0.0}, TestType{1})));
            STATIC_REQUIRE(std::signbit(op(TestType{0.0}, TestType{-1})));
            STATIC_REQUIRE(not std::signbit(op(TestType{-0.0}, TestType{-1})));

            STATIC_REQUIRE(not std::signbit(op(TestType{1}, inf)));
            STATIC_REQUIRE(std::signbit(op(TestType{1}, -inf)));
            STATIC_REQUIRE(std::signbit(op(TestType{-1}, inf)));
            STATIC_REQUIRE(not std::signbit(op(TestType{-1}, -inf)));

            // infinities
            STATIC_REQUIRE(op(inf, TestType{1}) == inf);
            STATIC_REQUIRE(op(inf, TestType{-1}) == -inf);
            STATIC_REQUIRE(op(-inf, TestType{1}) == -inf);
            STATIC_REQUIRE(op(-inf, TestType{-1}) == inf);

            STATIC_REQUIRE(op(TestType{1}, inf) == TestType{0});
            STATIC_REQUIRE(op(TestType{1}, -inf) == TestType{0});
            STATIC_REQUIRE(op(limits::max(), inf) == TestType{0});

            // unlike NumPy, inf / inf is an infinity rather than nan, signed as usual
            STATIC_REQUIRE(op(inf, inf) == inf);
            STATIC_REQUIRE(op(inf, -inf) == -inf);
            STATIC_REQUIRE(op(-inf, inf) == -inf);
            STATIC_REQUIRE(op(-inf, -inf) == inf);

            // underflows to zero
            STATIC_REQUIRE(op(limits::denorm_min(), limits::max()) == TestType{0});

            // overflows to infinity -- the overflow flag makes these non-constexpr on
            // GCC even though clang accepts them
            CHECK(op(limits::max(), TestType{0.5}) == inf);
            CHECK(op(limits::max(), limits::denorm_min()) == inf);

        } else {
            static_assert(false, "unexpected dtype");
        }
    }

    SECTION("op(interval, interval)") {
        if constexpr (std::floating_point<TestType>) {
            constexpr interval<TestType> zero(0, 0);
            constexpr interval<TestType> one(1, 1);
            constexpr interval<TestType> both(0, 1);

            constexpr TestType inf = limits::infinity();

            STATIC_REQUIRE(std::same_as<decltype(op(zero, zero)), interval<TestType>>);

            // rhs is exactly {0}, so every quotient is the safe-division zero
            STATIC_REQUIRE(op(zero, zero) == zero);
            STATIC_REQUIRE(op(one, zero) == zero);
            STATIC_REQUIRE(op(both, zero) == zero);
            STATIC_REQUIRE(op(interval<TestType>::all(), zero) == zero);

            // lhs is exactly {0}, so every quotient is zero whatever rhs holds
            STATIC_REQUIRE(op(zero, one) == zero);
            STATIC_REQUIRE(op(zero, both) == zero);
            STATIC_REQUIRE(op(zero, interval<TestType>(-1, 1)) == zero);
            STATIC_REQUIRE(op(zero, interval<TestType>::all()) == zero);

            // rhs excludes 0, so the extremes are at the corners
            STATIC_REQUIRE(op(one, one) == one);
            STATIC_REQUIRE(op(both, one) == both);
            STATIC_REQUIRE(
                op(interval<TestType>(1, 2), interval<TestType>(2, 4)) ==
                interval<TestType>(0.25, 1)
            );
            STATIC_REQUIRE(
                op(interval<TestType>(-2, -1), interval<TestType>(2, 4)) ==
                interval<TestType>(-1, -0.25)
            );
            STATIC_REQUIRE(
                op(interval<TestType>(1, 2), interval<TestType>(-4, -2)) ==
                interval<TestType>(-1, -0.25)
            );
            STATIC_REQUIRE(
                op(interval<TestType>(-2, -1), interval<TestType>(-4, -2)) ==
                interval<TestType>(0.25, 1)
            );

            // an lhs straddling zero needs all four corners, not two
            STATIC_REQUIRE(
                op(interval<TestType>(-1, 1), interval<TestType>(2, 4)) ==
                interval<TestType>(-0.5, 0.5)
            );
            STATIC_REQUIRE(
                op(interval<TestType>(-1, 1), interval<TestType>(-4, -2)) ==
                interval<TestType>(-0.5, 0.5)
            );

            // an infinite rhs endpoint, still excluding 0
            STATIC_REQUIRE(op(one, interval<TestType>(1, inf)) == interval<TestType>(0, 1));
            STATIC_REQUIRE(op(one, interval<TestType>(-inf, -1)) == interval<TestType>(-1, 0));

            // rhs straddles 0 from the non-negative side only
            STATIC_REQUIRE(op(one, both) == interval<TestType>::nonnegative());
            STATIC_REQUIRE(op(both, both) == interval<TestType>::nonnegative());
            STATIC_REQUIRE(op(interval<TestType>(1, 2), both) == interval<TestType>::nonnegative());

            // ... with a non-positive lhs the unbounded side flips
            STATIC_REQUIRE(op(interval<TestType>(-1, -1), both) == interval<TestType>(-inf, 0));
            STATIC_REQUIRE(op(interval<TestType>(-2, -1), both) == interval<TestType>(-inf, 0));

            // ... and an lhs straddling 0 reaches both infinities
            STATIC_REQUIRE(op(interval<TestType>(-1, 1), both) == interval<TestType>::all());

            // rhs straddles 0 with both signs, so both infinities are reachable
            STATIC_REQUIRE(op(one, interval<TestType>(-1, 1)) == interval<TestType>::all());
            STATIC_REQUIRE(
                op(interval<TestType>(-1, -1), interval<TestType>(-1, 1)) ==
                interval<TestType>::all()
            );
            STATIC_REQUIRE(
                op(interval<TestType>::all(), interval<TestType>::all()) ==
                interval<TestType>::all()
            );
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

TEMPLATE_LIST_TEST_CASE("less_equal", "", DTypes) {
    constexpr less_equal op{};

    using limits = std::numeric_limits<TestType>;

    SECTION("op(scalar, scalar)") {
        STATIC_REQUIRE(std::same_as<decltype(op(TestType{0}, TestType{0})), bool>);
        STATIC_REQUIRE(op(TestType{0}, TestType{0}));
        STATIC_REQUIRE(op(TestType{0}, TestType{1}));
        STATIC_REQUIRE(op(TestType{1}, TestType{1}));
        STATIC_REQUIRE(not op(TestType{1}, TestType{0}));

        if constexpr (std::floating_point<TestType>) {
            constexpr TestType inf = limits::infinity();

            STATIC_REQUIRE(op(-inf, -inf));
            STATIC_REQUIRE(op(-inf, +inf));
            STATIC_REQUIRE(not op(+inf, -inf));
            STATIC_REQUIRE(op(+inf, +inf));
        }
    }

    SECTION("op(interval, interval)") {
        constexpr interval<TestType> zero(0, 0);
        constexpr interval<TestType> one(1, 1);
        constexpr interval<TestType> both(0, 1);
        constexpr interval<bool> yes(true, true);
        constexpr interval<bool> no(false, false);
        constexpr interval<bool> maybe(false, true);

        STATIC_REQUIRE(std::same_as<decltype(op(zero, zero)), interval<bool>>);

        STATIC_REQUIRE(op(zero, zero) == yes);
        STATIC_REQUIRE(op(zero, one) == yes);
        STATIC_REQUIRE(op(zero, both) == yes);

        STATIC_REQUIRE(op(one, zero) == no);
        STATIC_REQUIRE(op(one, one) == yes);
        STATIC_REQUIRE(op(one, both) == maybe);

        STATIC_REQUIRE(op(both, zero) == maybe);
        STATIC_REQUIRE(op(both, one) == yes);  // 0 <= 1 and 1 <= 1
        STATIC_REQUIRE(op(both, both) == maybe);

        if constexpr (not std::same_as<TestType, bool>) {
            STATIC_REQUIRE(op(interval<TestType>(0, 1), interval<TestType>(2, 3)) == yes);
            STATIC_REQUIRE(op(interval<TestType>(2, 3), interval<TestType>(0, 1)) == no);
            STATIC_REQUIRE(op(interval<TestType>(-3, -2), interval<TestType>(1, 2)) == yes);
            STATIC_REQUIRE(op(interval<TestType>(1, 2), interval<TestType>(-3, -2)) == no);

            // domains touching at one endpoint: unlike equal, this is decided
            STATIC_REQUIRE(op(interval<TestType>(0, 2), interval<TestType>(2, 4)) == yes);
            STATIC_REQUIRE(op(interval<TestType>(2, 4), interval<TestType>(0, 2)) == maybe);

            STATIC_REQUIRE(op(interval<TestType>(0, 3), interval<TestType>(2, 4)) == maybe);
            STATIC_REQUIRE(op(interval<TestType>(1, 5), interval<TestType>(2, 3)) == maybe);
        }

        if constexpr (std::floating_point<TestType>) {
            constexpr TestType inf = std::numeric_limits<TestType>::infinity();
            constexpr interval<TestType> ninf(-inf, -inf);
            constexpr interval<TestType> pinf(inf, inf);

            // -0.0 == 0.0, so both orderings hold
            STATIC_REQUIRE(op(interval<TestType>(-0.0, -0.0), interval<TestType>(0.0, 0.0)) == yes);
            STATIC_REQUIRE(op(interval<TestType>(0.0, 0.0), interval<TestType>(-0.0, -0.0)) == yes);

            STATIC_REQUIRE(op(ninf, ninf) == yes);  // -inf <= -inf
            STATIC_REQUIRE(op(pinf, pinf) == yes);  //  inf <=  inf
            STATIC_REQUIRE(op(ninf, pinf) == yes);
            STATIC_REQUIRE(op(pinf, ninf) == no);
            STATIC_REQUIRE(op(ninf, interval<TestType>::all()) == yes);
            STATIC_REQUIRE(op(interval<TestType>::all(), pinf) == yes);
            STATIC_REQUIRE(op(pinf, interval<TestType>::all()) == maybe);
            STATIC_REQUIRE(op(interval<TestType>::all(), interval<TestType>::all()) == maybe);
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

TEMPLATE_LIST_TEST_CASE("logical_and", "", DTypes) {
    constexpr logical_and op{};

    SECTION("op(scalar, scalar)") {
        STATIC_REQUIRE(std::same_as<decltype(op(TestType{0}, TestType{0})), bool>);

        STATIC_REQUIRE(not op(TestType{0}, TestType{0}));
        STATIC_REQUIRE(not op(TestType{0}, TestType{1}));
        STATIC_REQUIRE(not op(TestType{1}, TestType{0}));
        STATIC_REQUIRE(op(TestType{1}, TestType{1}));

        if constexpr (not std::same_as<TestType, bool>) {
            STATIC_REQUIRE(not op(TestType{0}, TestType{1}));
            STATIC_REQUIRE(not op(TestType{-1}, TestType{0}));
            STATIC_REQUIRE(op(TestType{-1}, TestType{14}));
        }
        if constexpr (std::floating_point<TestType>) {
            STATIC_REQUIRE(not op(TestType{-0.0}, TestType{0.0}));
            STATIC_REQUIRE(op(TestType{.0000001}, TestType{.0000001}));

            constexpr TestType inf = std::numeric_limits<TestType>::infinity();

            STATIC_REQUIRE(op(-inf, -inf));
            STATIC_REQUIRE(op(-inf, +inf));
            STATIC_REQUIRE(op(+inf, -inf));
            STATIC_REQUIRE(op(+inf, +inf));

            STATIC_REQUIRE(not op(-inf, TestType{0}));
            STATIC_REQUIRE(not op(+inf, TestType{0}));
            STATIC_REQUIRE(not op(TestType{0}, -inf));
            STATIC_REQUIRE(not op(TestType{0}, +inf));
        }
    }

    SECTION("op(interval, interval)") {
        STATIC_REQUIRE(
            std::same_as<decltype(op(interval<TestType>(), interval<TestType>())), interval<bool>>
        );

        constexpr interval<TestType> falsy(0, 0);
        constexpr interval<TestType> truthy(1, 1);
        constexpr interval<TestType> ambiguous(0, 1);

        STATIC_REQUIRE(op(falsy, falsy) == falsy);
        STATIC_REQUIRE(op(falsy, truthy) == falsy);
        STATIC_REQUIRE(op(falsy, ambiguous) == falsy);

        STATIC_REQUIRE(op(truthy, falsy) == falsy);
        STATIC_REQUIRE(op(truthy, truthy) == truthy);
        STATIC_REQUIRE(op(truthy, ambiguous) == ambiguous);

        STATIC_REQUIRE(op(ambiguous, falsy) == falsy);
        STATIC_REQUIRE(op(ambiguous, truthy) == ambiguous);
        STATIC_REQUIRE(op(ambiguous, ambiguous) == ambiguous);

        if constexpr (not std::same_as<TestType, bool>) {
            constexpr interval<TestType> positive(1, 2);
            constexpr interval<TestType> negative(-3, -2);
            constexpr interval<TestType> wide(-5, 10);

            STATIC_REQUIRE(op(positive, falsy) == falsy);
            STATIC_REQUIRE(op(positive, truthy) == truthy);
            STATIC_REQUIRE(op(positive, ambiguous) == ambiguous);
            STATIC_REQUIRE(op(positive, negative) == truthy);
            STATIC_REQUIRE(op(positive, wide) == ambiguous);

            STATIC_REQUIRE(op(negative, falsy) == falsy);
            STATIC_REQUIRE(op(negative, truthy) == truthy);
            STATIC_REQUIRE(op(negative, ambiguous) == ambiguous);
            STATIC_REQUIRE(op(negative, negative) == truthy);
            STATIC_REQUIRE(op(negative, wide) == ambiguous);

            STATIC_REQUIRE(op(wide, falsy) == falsy);
            STATIC_REQUIRE(op(wide, truthy) == ambiguous);
            STATIC_REQUIRE(op(wide, ambiguous) == ambiguous);
            STATIC_REQUIRE(op(wide, negative) == ambiguous);
            STATIC_REQUIRE(op(wide, wide) == ambiguous);
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

TEMPLATE_LIST_TEST_CASE("logical_or", "", DTypes) {
    constexpr logical_or op{};

    SECTION("op(scalar, scalar)") {
        STATIC_REQUIRE(std::same_as<decltype(op(TestType{0}, TestType{0})), bool>);

        STATIC_REQUIRE(not op(TestType{0}, TestType{0}));
        STATIC_REQUIRE(op(TestType{0}, TestType{1}));
        STATIC_REQUIRE(op(TestType{1}, TestType{0}));
        STATIC_REQUIRE(op(TestType{1}, TestType{1}));

        if constexpr (not std::same_as<TestType, bool>) {
            STATIC_REQUIRE(op(TestType{0}, TestType{1}));
            STATIC_REQUIRE(op(TestType{-1}, TestType{0}));
            STATIC_REQUIRE(op(TestType{-1}, TestType{14}));
        }
        if constexpr (std::floating_point<TestType>) {
            STATIC_REQUIRE(not op(TestType{-0.0}, TestType{0.0}));
            STATIC_REQUIRE(op(TestType{.0000001}, TestType{.0000001}));

            constexpr TestType inf = std::numeric_limits<TestType>::infinity();

            STATIC_REQUIRE(op(-inf, -inf));
            STATIC_REQUIRE(op(-inf, +inf));
            STATIC_REQUIRE(op(+inf, -inf));
            STATIC_REQUIRE(op(+inf, +inf));

            STATIC_REQUIRE(op(-inf, TestType{0}));
            STATIC_REQUIRE(op(+inf, TestType{0}));
            STATIC_REQUIRE(op(TestType{0}, -inf));
            STATIC_REQUIRE(op(TestType{0}, +inf));
        }
    }

    SECTION("op(interval, interval)") {
        STATIC_REQUIRE(
            std::same_as<decltype(op(interval<TestType>(), interval<TestType>())), interval<bool>>
        );

        constexpr interval<TestType> falsy(0, 0);
        constexpr interval<TestType> truthy(1, 1);
        constexpr interval<TestType> ambiguous(0, 1);

        STATIC_REQUIRE(op(falsy, falsy) == falsy);
        STATIC_REQUIRE(op(falsy, truthy) == truthy);
        STATIC_REQUIRE(op(falsy, ambiguous) == ambiguous);

        STATIC_REQUIRE(op(truthy, falsy) == truthy);
        STATIC_REQUIRE(op(truthy, truthy) == truthy);
        STATIC_REQUIRE(op(truthy, ambiguous) == truthy);

        STATIC_REQUIRE(op(ambiguous, falsy) == ambiguous);
        STATIC_REQUIRE(op(ambiguous, truthy) == truthy);
        STATIC_REQUIRE(op(ambiguous, ambiguous) == ambiguous);

        if constexpr (not std::same_as<TestType, bool>) {
            constexpr interval<TestType> positive(1, 2);
            constexpr interval<TestType> negative(-3, -2);
            constexpr interval<TestType> wide(-5, 10);

            STATIC_REQUIRE(op(positive, falsy) == truthy);
            STATIC_REQUIRE(op(positive, truthy) == truthy);
            STATIC_REQUIRE(op(positive, ambiguous) == truthy);
            STATIC_REQUIRE(op(positive, negative) == truthy);
            STATIC_REQUIRE(op(positive, wide) == truthy);

            STATIC_REQUIRE(op(negative, falsy) == truthy);
            STATIC_REQUIRE(op(negative, truthy) == truthy);
            STATIC_REQUIRE(op(negative, ambiguous) == truthy);
            STATIC_REQUIRE(op(negative, negative) == truthy);
            STATIC_REQUIRE(op(negative, wide) == truthy);

            STATIC_REQUIRE(op(wide, falsy) == ambiguous);
            STATIC_REQUIRE(op(wide, truthy) == truthy);
            STATIC_REQUIRE(op(wide, ambiguous) == ambiguous);
            STATIC_REQUIRE(op(wide, negative) == truthy);
            STATIC_REQUIRE(op(wide, wide) == ambiguous);
        }
    }
}

TEMPLATE_LIST_TEST_CASE("logical_xor", "", DTypes) {
    constexpr logical_xor op{};

    SECTION("op(scalar, scalar)") {
        STATIC_REQUIRE(std::same_as<decltype(op(TestType{0}, TestType{0})), bool>);

        STATIC_REQUIRE(not op(TestType{0}, TestType{0}));
        STATIC_REQUIRE(op(TestType{0}, TestType{1}));
        STATIC_REQUIRE(op(TestType{1}, TestType{0}));
        STATIC_REQUIRE(not op(TestType{1}, TestType{1}));

        if constexpr (not std::same_as<TestType, bool>) {
            STATIC_REQUIRE(op(TestType{0}, TestType{1}));
            STATIC_REQUIRE(op(TestType{-1}, TestType{0}));
            STATIC_REQUIRE(not op(TestType{-1}, TestType{14}));
        }
        if constexpr (std::floating_point<TestType>) {
            STATIC_REQUIRE(not op(TestType{-0.0}, TestType{0.0}));
            STATIC_REQUIRE(not op(TestType{.0000001}, TestType{.0000001}));

            constexpr TestType inf = std::numeric_limits<TestType>::infinity();

            STATIC_REQUIRE(not op(-inf, -inf));
            STATIC_REQUIRE(not op(-inf, +inf));
            STATIC_REQUIRE(not op(+inf, -inf));
            STATIC_REQUIRE(not op(+inf, +inf));

            STATIC_REQUIRE(op(-inf, TestType{0}));
            STATIC_REQUIRE(op(+inf, TestType{0}));
            STATIC_REQUIRE(op(TestType{0}, -inf));
            STATIC_REQUIRE(op(TestType{0}, +inf));
        }
    }

    SECTION("op(interval, interval)") {
        STATIC_REQUIRE(
            std::same_as<decltype(op(interval<TestType>(), interval<TestType>())), interval<bool>>
        );

        constexpr interval<TestType> falsy(0, 0);
        constexpr interval<TestType> truthy(1, 1);
        constexpr interval<TestType> ambiguous(0, 1);

        STATIC_REQUIRE(op(falsy, falsy) == falsy);
        STATIC_REQUIRE(op(falsy, truthy) == truthy);
        STATIC_REQUIRE(op(falsy, ambiguous) == ambiguous);

        STATIC_REQUIRE(op(truthy, falsy) == truthy);
        STATIC_REQUIRE(op(truthy, truthy) == falsy);
        STATIC_REQUIRE(op(truthy, ambiguous) == ambiguous);

        STATIC_REQUIRE(op(ambiguous, falsy) == ambiguous);
        STATIC_REQUIRE(op(ambiguous, truthy) == ambiguous);
        STATIC_REQUIRE(op(ambiguous, ambiguous) == ambiguous);

        if constexpr (not std::same_as<TestType, bool>) {
            constexpr interval<TestType> positive(1, 2);
            constexpr interval<TestType> negative(-3, -2);
            constexpr interval<TestType> wide(-5, 10);

            STATIC_REQUIRE(op(positive, falsy) == truthy);
            STATIC_REQUIRE(op(positive, truthy) == falsy);
            STATIC_REQUIRE(op(positive, ambiguous) == ambiguous);
            STATIC_REQUIRE(op(positive, negative) == falsy);
            STATIC_REQUIRE(op(positive, wide) == ambiguous);

            STATIC_REQUIRE(op(negative, falsy) == truthy);
            STATIC_REQUIRE(op(negative, truthy) == falsy);
            STATIC_REQUIRE(op(negative, ambiguous) == ambiguous);
            STATIC_REQUIRE(op(negative, negative) == falsy);
            STATIC_REQUIRE(op(negative, wide) == ambiguous);

            STATIC_REQUIRE(op(wide, falsy) == ambiguous);
            STATIC_REQUIRE(op(wide, truthy) == ambiguous);
            STATIC_REQUIRE(op(wide, ambiguous) == ambiguous);
            STATIC_REQUIRE(op(wide, negative) == ambiguous);
            STATIC_REQUIRE(op(wide, wide) == ambiguous);
        }
    }
}

TEMPLATE_LIST_TEST_CASE("maximum", "", DTypes) {
    constexpr maximum op{};

    SECTION("op(scalar, scalar)") {
        STATIC_REQUIRE(std::same_as<decltype(op(TestType{0}, TestType{0})), TestType>);
        STATIC_REQUIRE(op(TestType{0}, TestType{0}) == TestType{0});
        STATIC_REQUIRE(op(TestType{0}, TestType{1}) == TestType{1});
        STATIC_REQUIRE(op(TestType{1}, TestType{0}) == TestType{1});
        STATIC_REQUIRE(op(TestType{1}, TestType{1}) == TestType{1});

        STATIC_REQUIRE(
            op(std::numeric_limits<TestType>::lowest(), std::numeric_limits<TestType>::max()) ==
            std::numeric_limits<TestType>::max()
        );
    }

    SECTION("op(interval, interval)") {
        constexpr interval<TestType> zero(0, 0);
        constexpr interval<TestType> one(1, 1);
        constexpr interval<TestType> both(0, 1);

        STATIC_REQUIRE(std::same_as<decltype(op(zero, zero)), interval<TestType>>);

        STATIC_REQUIRE(op(zero, zero) == zero);
        STATIC_REQUIRE(op(zero, one) == one);
        STATIC_REQUIRE(op(zero, both) == both);

        STATIC_REQUIRE(op(one, zero) == one);
        STATIC_REQUIRE(op(one, one) == one);
        STATIC_REQUIRE(op(one, both) == one);

        STATIC_REQUIRE(op(both, zero) == both);
        STATIC_REQUIRE(op(both, one) == one);
        STATIC_REQUIRE(op(both, both) == both);

        if constexpr (not std::same_as<TestType, bool>) {
            using inter = interval<TestType>;
            STATIC_REQUIRE(op(inter(1, 5), inter(2, 3)) == inter(2, 5));
            STATIC_REQUIRE(op(inter(0, 10), inter(5, 5)) == inter(5, 10));
            STATIC_REQUIRE(op(inter(3, 7), inter(5, 5)) == inter(5, 7));
            STATIC_REQUIRE(op(inter(0, 1), inter(2, 3)) == inter(2, 3));
            STATIC_REQUIRE(op(inter(2, 3), inter(0, 1)) == inter(2, 3));
            STATIC_REQUIRE(op(inter(-3, -2), inter(1, 2)) == inter(1, 2));
        }

        if constexpr (std::floating_point<TestType>) {
            constexpr TestType inf = std::numeric_limits<TestType>::infinity();

            STATIC_REQUIRE(
                op(interval<TestType>(inf, inf), interval<TestType>(inf, inf)) ==
                interval<TestType>(inf, inf)
            );
            STATIC_REQUIRE(
                op(interval<TestType>(-inf, -inf), interval<TestType>::all()) ==
                interval<TestType>(-inf, inf)
            );
            STATIC_REQUIRE(
                op(interval<TestType>(inf, inf), interval<TestType>::all()) ==
                interval<TestType>(inf, inf)
            );
            STATIC_REQUIRE(
                op(interval<TestType>::all(), interval<TestType>::all()) ==
                interval<TestType>::all()
            );

            // -0.0 == 0.0 so which representation wins the tie is unobservable
            STATIC_REQUIRE(
                op(interval<TestType>(-0.0, -0.0), interval<TestType>(0.0, 0.0)) ==
                interval<TestType>(0.0, 0.0)
            );
        }
    }
}

TEMPLATE_LIST_TEST_CASE("minimum", "", DTypes) {
    constexpr minimum op{};

    SECTION("op(scalar, scalar)") {
        STATIC_REQUIRE(std::same_as<decltype(op(TestType{0}, TestType{0})), TestType>);
        STATIC_REQUIRE(op(TestType{0}, TestType{0}) == TestType{0});
        STATIC_REQUIRE(op(TestType{0}, TestType{1}) == TestType{0});
        STATIC_REQUIRE(op(TestType{1}, TestType{0}) == TestType{0});
        STATIC_REQUIRE(op(TestType{1}, TestType{1}) == TestType{1});

        STATIC_REQUIRE(
            op(std::numeric_limits<TestType>::lowest(), std::numeric_limits<TestType>::max()) ==
            std::numeric_limits<TestType>::lowest()
        );
    }

    SECTION("op(interval, interval)") {
        constexpr interval<TestType> zero(0, 0);
        constexpr interval<TestType> one(1, 1);
        constexpr interval<TestType> both(0, 1);

        STATIC_REQUIRE(std::same_as<decltype(op(zero, zero)), interval<TestType>>);

        STATIC_REQUIRE(op(zero, zero) == zero);
        STATIC_REQUIRE(op(zero, one) == zero);
        STATIC_REQUIRE(op(zero, both) == zero);

        STATIC_REQUIRE(op(one, zero) == zero);
        STATIC_REQUIRE(op(one, one) == one);
        STATIC_REQUIRE(op(one, both) == both);

        STATIC_REQUIRE(op(both, zero) == zero);
        STATIC_REQUIRE(op(both, one) == both);
        STATIC_REQUIRE(op(both, both) == both);

        if constexpr (not std::same_as<TestType, bool>) {
            using inter = interval<TestType>;
            STATIC_REQUIRE(op(inter(1, 5), inter(2, 3)) == inter(1, 3));
            STATIC_REQUIRE(op(inter(0, 10), inter(5, 5)) == inter(0, 5));
            STATIC_REQUIRE(op(inter(3, 7), inter(5, 5)) == inter(3, 5));
            STATIC_REQUIRE(op(inter(0, 1), inter(2, 3)) == inter(0, 1));
            STATIC_REQUIRE(op(inter(2, 3), inter(0, 1)) == inter(0, 1));
            STATIC_REQUIRE(op(inter(-3, -2), inter(1, 2)) == inter(-3, -2));
        }

        if constexpr (std::floating_point<TestType>) {
            constexpr TestType inf = std::numeric_limits<TestType>::infinity();

            STATIC_REQUIRE(
                op(interval<TestType>(inf, inf), interval<TestType>(inf, inf)) ==
                interval<TestType>(inf, inf)
            );
            STATIC_REQUIRE(
                op(interval<TestType>(-inf, -inf), interval<TestType>::all()) ==
                interval<TestType>(-inf, -inf)
            );
            STATIC_REQUIRE(
                op(interval<TestType>(inf, inf), interval<TestType>::all()) ==
                interval<TestType>::all()
            );
            STATIC_REQUIRE(
                op(interval<TestType>::all(), interval<TestType>::all()) ==
                interval<TestType>::all()
            );

            // -0.0 == 0.0 so which representation wins the tie is unobservable
            STATIC_REQUIRE(
                op(interval<TestType>(-0.0, -0.0), interval<TestType>(0.0, 0.0)) ==
                interval<TestType>(0.0, 0.0)
            );
        }
    }
}

TEMPLATE_LIST_TEST_CASE("multiply", "", DTypes) {
    constexpr multiply op{};
    using limits = std::numeric_limits<TestType>;

    SECTION("op(scalar, scalar)") {
        STATIC_REQUIRE(std::same_as<decltype(op(TestType{0}, TestType{0})), TestType>);

        STATIC_REQUIRE(op(TestType{0}, TestType{0}) == TestType{0});
        STATIC_REQUIRE(op(TestType{0}, TestType{1}) == TestType{0});
        STATIC_REQUIRE(op(TestType{1}, TestType{0}) == TestType{0});
        STATIC_REQUIRE(op(TestType{1}, TestType{1}) == TestType{1});

        STATIC_REQUIRE(op(limits::max(), TestType{1}) == limits::max());
        STATIC_REQUIRE(op(limits::max(), TestType{0}) == TestType{0});
        STATIC_REQUIRE(op(limits::lowest(), TestType{1}) == limits::lowest());
        STATIC_REQUIRE(op(limits::lowest(), TestType{0}) == TestType{0});

        if constexpr (not std::same_as<TestType, bool>) {
            STATIC_REQUIRE(op(TestType{2}, TestType{3}) == TestType{6});
            STATIC_REQUIRE(op(TestType{-2}, TestType{3}) == TestType{-6});
            STATIC_REQUIRE(op(TestType{2}, TestType{-3}) == TestType{-6});
            STATIC_REQUIRE(op(TestType{-2}, TestType{-3}) == TestType{6});
            STATIC_REQUIRE(op(TestType{1}, TestType{-1}) == TestType{-1});
            STATIC_REQUIRE(op(TestType{0}, TestType{-1}) == TestType{0});
        }
    }

    if constexpr (std::signed_integral<TestType>) {
        SECTION("saturating") {
            // positive overflow
            STATIC_REQUIRE(op(limits::max(), TestType{2}) == limits::max());
            STATIC_REQUIRE(op(TestType{2}, limits::max()) == limits::max());
            STATIC_REQUIRE(op(limits::max(), limits::max()) == limits::max());
            STATIC_REQUIRE(op(limits::lowest(), limits::lowest()) == limits::max());
            STATIC_REQUIRE(op(limits::lowest(), TestType{-1}) == limits::max());
            STATIC_REQUIRE(op(limits::lowest(), TestType{-2}) == limits::max());

            // negative overflow
            STATIC_REQUIRE(op(limits::lowest(), TestType{2}) == limits::lowest());
            STATIC_REQUIRE(op(TestType{2}, limits::lowest()) == limits::lowest());
            STATIC_REQUIRE(op(limits::max(), TestType{-2}) == limits::lowest());
            STATIC_REQUIRE(op(TestType{-2}, limits::max()) == limits::lowest());
            STATIC_REQUIRE(op(limits::lowest(), limits::max()) == limits::lowest());
            STATIC_REQUIRE(op(limits::max(), limits::lowest()) == limits::lowest());

            // just shy of saturating
            STATIC_REQUIRE(op(limits::max(), TestType{-1}) == TestType(limits::lowest() + 1));
            STATIC_REQUIRE(op(TestType{-1}, limits::max()) == TestType(limits::lowest() + 1));
        }
    }

    if constexpr (std::floating_point<TestType>) {
        constexpr TestType inf = limits::infinity();

        SECTION("infinities") {
            STATIC_REQUIRE(op(inf, TestType{2}) == inf);
            STATIC_REQUIRE(op(inf, TestType{-2}) == -inf);
            STATIC_REQUIRE(op(-inf, TestType{2}) == -inf);
            STATIC_REQUIRE(op(-inf, TestType{-2}) == inf);
            STATIC_REQUIRE(op(inf, inf) == inf);
            STATIC_REQUIRE(op(inf, -inf) == -inf);
            STATIC_REQUIRE(op(-inf, -inf) == inf);

            // underflows to zero, which raises no flag that blocks constant evaluation
            STATIC_REQUIRE(op(limits::denorm_min(), TestType{0.5}) == TestType{0});

            // overflows to infinity -- the overflow flag makes these non-constexpr on
            // GCC even though clang accepts them
            CHECK(op(limits::max(), TestType{2}) == inf);
            CHECK(op(limits::max(), limits::max()) == inf);
            CHECK(op(limits::max(), limits::lowest()) == -inf);
        }

        SECTION("signed zeros") {
            STATIC_REQUIRE(not std::signbit(op(TestType{0.0}, TestType{0.0})));
            STATIC_REQUIRE(std::signbit(op(TestType{-0.0}, TestType{0.0})));
            STATIC_REQUIRE(std::signbit(op(TestType{0.0}, TestType{-0.0})));
            STATIC_REQUIRE(not std::signbit(op(TestType{-0.0}, TestType{-0.0})));

            STATIC_REQUIRE(std::signbit(op(TestType{-1}, TestType{0.0})));
            STATIC_REQUIRE(std::signbit(op(TestType{1}, TestType{-0.0})));
            STATIC_REQUIRE(not std::signbit(op(TestType{-1}, TestType{-0.0})));
        }

        SECTION("zero times infinity") {
            STATIC_REQUIRE(op(TestType{0}, inf) == TestType{0});
            STATIC_REQUIRE(op(inf, TestType{0}) == TestType{0});
            STATIC_REQUIRE(not std::signbit(op(TestType{0.0}, inf)));
            STATIC_REQUIRE(std::signbit(op(TestType{-0.0}, inf)));
            STATIC_REQUIRE(std::signbit(op(TestType{0.0}, -inf)));
            STATIC_REQUIRE(not std::signbit(op(TestType{-0.0}, -inf)));
        }
    }

    SECTION("op(interval, interval)") {
        constexpr interval<TestType> zero(0, 0);
        constexpr interval<TestType> one(1, 1);
        constexpr interval<TestType> both(0, 1);

        STATIC_REQUIRE(std::same_as<decltype(op(zero, zero)), interval<TestType>>);

        STATIC_REQUIRE(op(zero, zero) == zero);
        STATIC_REQUIRE(op(zero, one) == zero);
        STATIC_REQUIRE(op(zero, both) == zero);
        STATIC_REQUIRE(op(one, zero) == zero);
        STATIC_REQUIRE(op(one, one) == one);
        STATIC_REQUIRE(op(one, both) == both);
        STATIC_REQUIRE(op(both, zero) == zero);
        STATIC_REQUIRE(op(both, one) == both);
        STATIC_REQUIRE(op(both, both) == both);

        if constexpr (not std::same_as<TestType, bool>) {
            // both operands positive
            STATIC_REQUIRE(
                op(interval<TestType>(2, 3), interval<TestType>(4, 5)) == interval<TestType>(8, 15)
            );

            // both operands negative -- two-corner selection gives [4, 1], i.e. empty
            STATIC_REQUIRE(
                op(interval<TestType>(-2, -1), interval<TestType>(-2, -1)) ==
                interval<TestType>(1, 4)
            );
            STATIC_REQUIRE(
                op(interval<TestType>(-3, -2), interval<TestType>(-5, -4)) ==
                interval<TestType>(8, 15)
            );

            // mixed signs -- two-corner selection gives [-2, -2], non-empty but wrong
            STATIC_REQUIRE(
                op(interval<TestType>(-2, -1), interval<TestType>(1, 2)) ==
                interval<TestType>(-4, -1)
            );
            STATIC_REQUIRE(
                op(interval<TestType>(1, 2), interval<TestType>(-2, -1)) ==
                interval<TestType>(-4, -1)
            );

            // an operand straddling zero
            STATIC_REQUIRE(
                op(interval<TestType>(-1, 1), interval<TestType>(-1, 1)) ==
                interval<TestType>(-1, 1)
            );
            STATIC_REQUIRE(
                op(interval<TestType>(-1, 2), interval<TestType>(3, 4)) == interval<TestType>(-4, 8)
            );
            STATIC_REQUIRE(
                op(interval<TestType>(-1, 2), interval<TestType>(-4, -3)) ==
                interval<TestType>(-8, 4)
            );
            STATIC_REQUIRE(
                op(interval<TestType>(-1, 2), interval<TestType>(-3, 4)) ==
                interval<TestType>(-6, 8)
            );
        }

        if constexpr (std::signed_integral<TestType>) {
            STATIC_REQUIRE(
                op(
                    interval<TestType>(limits::lowest(), limits::lowest()), interval<TestType>(2, 2)
                ) == interval<TestType>(limits::lowest(), limits::lowest())
            );
            STATIC_REQUIRE(
                op(interval<TestType>::all(), interval<TestType>::all()) ==
                interval<TestType>::all()
            );
        }
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

TEMPLATE_LIST_TEST_CASE("remainder", "", DTypes) {
    // dev note: As of Sept 2026, no version of Apple Clang supports `constexpr std::fmod()`,
    // so we use CHECK for the tests that run through the floating branch.

    constexpr remainder op;

    SECTION("op(scalar, scalar)") {
        CHECK(op(TestType{0}, TestType{0}) == 0);
        CHECK(op(TestType{1}, TestType{0}) == 0);
        CHECK(op(TestType{0}, TestType{1}) == 0);
        CHECK(op(TestType{1}, TestType{1}) == 0);

        if constexpr (not std::same_as<TestType, bool>) {
            CHECK(op(TestType{0}, TestType{-1}) == 0);
            CHECK(op(TestType{-1}, TestType{-10}) == -1);
            CHECK(op(TestType{-1}, TestType{10}) == 9);
            CHECK(op(TestType{1}, TestType{-10}) == -9);
        }

        if constexpr (std::floating_point<TestType>) {
            CHECK(op(TestType{-5.5}, TestType{-4}) == -1.5);
            CHECK(op(TestType{-5.5}, TestType{4}) == 2.5);
            CHECK(op(TestType{5.5}, TestType{-4}) == -2.5);
            CHECK(op(TestType{5.5}, TestType{4}) == 1.5);

            CHECK(std::signbit(op(TestType{-0.0}, TestType{-0.0})));
            CHECK(std::signbit(op(TestType{0.0}, TestType{-0.0})));
            CHECK(not std::signbit(op(TestType{-0.0}, TestType{0.0})));
            CHECK(not std::signbit(op(TestType{0.0}, TestType{0.0})));

            CHECK(not std::signbit(op(TestType{-5.0}, TestType{5.0})));
            CHECK(std::signbit(op(TestType{5.0}, TestType{-5.0})));
            CHECK(not std::signbit(op(TestType{-0.0}, TestType{5.0})));

            constexpr TestType inf = std::numeric_limits<TestType>::infinity();

            CHECK(op(TestType{-5.5}, -inf) == -5.5);
            CHECK(op(TestType{-5.5}, +inf) == +inf);
            CHECK(op(TestType{+5.5}, -inf) == -inf);
            CHECK(op(TestType{+5.5}, +inf) == TestType{5.5});

            CHECK(op(+inf, TestType{0}) == 0);
            CHECK(op(-inf, TestType{0}) == 0);

            CHECK(op(-inf, TestType{-5}) == 0);
            CHECK(op(+inf, TestType{-5}) == 0);
            CHECK(op(-inf, TestType{+5}) == 0);
            CHECK(op(+inf, TestType{+5}) == 0);

            CHECK(op(-inf, -inf) == 0);
            CHECK(op(+inf, -inf) == 0);
            CHECK(op(-inf, +inf) == 0);
            CHECK(op(+inf, +inf) == 0);

            CHECK(std::signbit(op(-inf, TestType{-0.0})));
            CHECK(std::signbit(op(-inf, TestType{-5.0})));

            CHECK(std::signbit(op(+inf, TestType{-0.0})));
            CHECK(std::signbit(op(+inf, TestType{-5.0})));

            CHECK(not std::signbit(op(-inf, TestType{+0.0})));
            CHECK(not std::signbit(op(-inf, TestType{+5.0})));

            CHECK(not std::signbit(op(+inf, TestType{+0.0})));
            CHECK(not std::signbit(op(+inf, TestType{+5.0})));
        }
    }

    SECTION("op(interval, interval)") {
        constexpr interval<TestType> zero(0, 0);
        constexpr interval<TestType> one(1, 1);
        constexpr interval<TestType> both(0, 1);
        constexpr auto all = interval<TestType>::all();

        STATIC_REQUIRE(std::same_as<decltype(op(zero, zero)), interval<TestType>>);

        CHECK(op(zero, zero) == zero);

        if constexpr (std::integral<TestType>) {
            STATIC_REQUIRE(op(zero, one) == zero);
            STATIC_REQUIRE(op(zero, both) == zero);

            STATIC_REQUIRE(op(one, one) == zero);
            STATIC_REQUIRE(op(one, both) == zero);

            STATIC_REQUIRE(op(both, one) == zero);
            STATIC_REQUIRE(op(both, both) == zero);
        }
        if constexpr (std::floating_point<TestType>) {
            CHECK(op(zero, one) == both);
            CHECK(op(zero, both) == both);
            CHECK(op(zero, all) == all);

            CHECK(op(one, one) == both);
            CHECK(op(one, both) == both);
            CHECK(op(one, all) == all);

            CHECK(op(both, one) == both);
            CHECK(op(both, both) == both);
            CHECK(op(both, all) == all);

            CHECK(op(all, zero) == zero);
            CHECK(op(all, one) == both);
            CHECK(op(all, both) == both);
            CHECK(op(all, one) == both);
            CHECK(op(all, all) == all);
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

TEMPLATE_LIST_TEST_CASE("subtract", "", DTypes) {
    constexpr subtract op{};

    using limits = std::numeric_limits<TestType>;

    if constexpr (std::same_as<TestType, bool>) {
        // Follow NumPy and disallow boolean subtraction
        STATIC_REQUIRE(not std::invocable<subtract, TestType, TestType>);
        STATIC_REQUIRE(not std::invocable<subtract, interval<TestType>, interval<TestType>>);
    } else {
        SECTION("op(scalar, scalar)") {
            // dtype is preseved
            STATIC_REQUIRE(std::same_as<decltype(op(TestType{0}, TestType{0})), TestType>);

            STATIC_REQUIRE(op(TestType{0}, TestType{0}) == TestType{0});
            STATIC_REQUIRE(op(TestType{1}, TestType{0}) == TestType{1});
            STATIC_REQUIRE(op(TestType{0}, TestType{1}) == TestType{-1});
            STATIC_REQUIRE(op(TestType{1}, TestType{1}) == TestType{0});
            STATIC_REQUIRE(op(TestType{5}, TestType{3}) == TestType{2});
            STATIC_REQUIRE(op(TestType{3}, TestType{5}) == TestType{-2});

            STATIC_REQUIRE(op(limits::max(), limits::max()) == TestType{0});
            STATIC_REQUIRE(op(limits::lowest(), limits::lowest()) == TestType{0});

            // Check saturation for integers
            if constexpr (std::signed_integral<TestType>) {
                STATIC_REQUIRE(op(limits::max(), TestType{-1}) == limits::max());
                STATIC_REQUIRE(op(limits::lowest(), TestType{1}) == limits::lowest());
                STATIC_REQUIRE(op(limits::max(), limits::lowest()) == limits::max());
                STATIC_REQUIRE(op(limits::lowest(), limits::max()) == limits::lowest());
                STATIC_REQUIRE(op(TestType{0}, limits::lowest()) == limits::max());

                STATIC_REQUIRE(op(limits::min(), TestType{1}) == limits::min());
                STATIC_REQUIRE(op(limits::lowest(), limits::max()) == limits::lowest());
                STATIC_REQUIRE(op(limits::lowest(), TestType{1}) == limits::lowest());
                STATIC_REQUIRE(op(limits::lowest(), limits::max()) == limits::lowest());
                STATIC_REQUIRE(op(limits::max(), TestType{1}) == TestType(limits::max() - 1));
                STATIC_REQUIRE(
                    op(limits::lowest(), TestType{-1}) == TestType(limits::lowest() + 1)
                );
            }

            if constexpr (std::floating_point<TestType>) {
                constexpr TestType inf = limits::infinity();

                STATIC_REQUIRE(op(inf, TestType{1}) == inf);
                STATIC_REQUIRE(op(TestType{1}, inf) == -inf);
                STATIC_REQUIRE(op(-inf, TestType{1}) == -inf);
                STATIC_REQUIRE(op(TestType{1}, -inf) == inf);

                STATIC_REQUIRE(op(inf, -inf) == inf);
                STATIC_REQUIRE(op(-inf, inf) == -inf);

                // unlike NumPy, inf - inf is inf rather than nan
                STATIC_REQUIRE(op(inf, inf) == inf);
                STATIC_REQUIRE(op(-inf, -inf) == inf);

                // overflows to infinity rather than saturating
                // Use CHECK because GCC doesn't like compile-time overflows. Clang doesn't care.
                CHECK(op(limits::max(), limits::lowest()) == inf);
                CHECK(op(limits::lowest(), limits::max()) == -inf);

                STATIC_REQUIRE(not std::signbit(op(TestType{0.0}, TestType{0.0})));
                STATIC_REQUIRE(not std::signbit(op(TestType{0.0}, TestType{-0.0})));
                STATIC_REQUIRE(std::signbit(op(TestType{-0.0}, TestType{0.0})));
                STATIC_REQUIRE(not std::signbit(op(TestType{-0.0}, TestType{-0.0})));
            }
        }

        SECTION("op(interval, interval)") {
            constexpr interval<TestType> zero(0, 0);
            constexpr interval<TestType> one(1, 1);
            constexpr interval<TestType> both(0, 1);

            STATIC_REQUIRE(std::same_as<decltype(op(zero, zero)), interval<TestType>>);

            STATIC_REQUIRE(op(zero, zero) == zero);
            STATIC_REQUIRE(op(zero, one) == interval<TestType>(-1, -1));
            STATIC_REQUIRE(op(zero, both) == interval<TestType>(-1, 0));

            STATIC_REQUIRE(op(one, zero) == one);
            STATIC_REQUIRE(op(one, one) == zero);
            STATIC_REQUIRE(op(one, both) == both);

            STATIC_REQUIRE(op(both, zero) == both);
            STATIC_REQUIRE(op(both, one) == interval<TestType>(-1, 0));
            STATIC_REQUIRE(op(both, both) == interval<TestType>(-1, 1));

            STATIC_REQUIRE(
                op(interval<TestType>(1, 2), interval<TestType>(10, 20)) ==
                interval<TestType>(-19, -8)
            );
            STATIC_REQUIRE(
                op(interval<TestType>::all(), interval<TestType>::all()) ==
                interval<TestType>::all()
            );

            if constexpr (std::integral<TestType>) {
                STATIC_REQUIRE(
                    op(interval<TestType>(limits::lowest(), limits::lowest()), one) ==
                    interval<TestType>(limits::lowest(), limits::lowest())
                );
                STATIC_REQUIRE(
                    op(zero, interval<TestType>(limits::lowest(), limits::lowest())) ==
                    interval<TestType>(limits::max(), limits::max())
                );
            }

            if constexpr (std::floating_point<TestType>) {
                constexpr TestType inf = limits::infinity();
                STATIC_REQUIRE(
                    op(interval<TestType>(inf, inf), interval<TestType>(inf, inf)) ==
                    interval<TestType>(inf, inf)
                );
                STATIC_REQUIRE(
                    op(interval<TestType>(-inf, -inf), interval<TestType>(-inf, -inf)) ==
                    interval<TestType>(inf, inf)
                );
            }
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

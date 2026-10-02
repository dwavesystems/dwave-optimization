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

    SECTION("empty domains propagate") {
        constexpr interval<std::int32_t> empty;
        STATIC_REQUIRE(not static_cast<bool>(test_sub{}(empty, rhs)));
        STATIC_REQUIRE(not static_cast<bool>(test_sub{}(lhs, empty)));
        STATIC_REQUIRE(not static_cast<bool>(test_sub{}(empty, empty)));
    }
}

}  // namespace mixins

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

TEMPLATE_LIST_TEST_CASE("add", "", DTypes) {
    using T = TestType;
    constexpr add op{};

    SECTION("add(scalar, scalar)") {
        // dtype is preseved
        STATIC_REQUIRE(std::same_as<decltype(op(T(0), T(0))), T>);

        STATIC_REQUIRE(op(T(1), T(0)) == T(1));
        STATIC_REQUIRE(op(T(0), T(1)) == T(1));

        // Check saturation for integers
        if constexpr (std::signed_integral<T>) {
            using limits = std::numeric_limits<T>;

            STATIC_REQUIRE(op(limits::max(), T(1)) == limits::max());
            STATIC_REQUIRE(op(limits::max(), limits::max()) == limits::max());
            STATIC_REQUIRE(op(limits::lowest(), T(-1)) == limits::lowest());
            STATIC_REQUIRE(op(limits::lowest(), limits::lowest()) == limits::lowest());
        }
    }

    SECTION("op(interval, interval)") {
        STATIC_REQUIRE(
            op(interval<TestType>(0, 5), interval<TestType>(1, 2)) == interval<TestType>(1, 7)
        );
    }
}

TEST_CASE("add") {
    constexpr add op{};

    using int8_limits = std::numeric_limits<std::int8_t>;
    using int16_limits = std::numeric_limits<std::int16_t>;
    using int32_limits = std::numeric_limits<std::int32_t>;
    using int64_limits = std::numeric_limits<std::int64_t>;
    using float_limits = std::numeric_limits<float>;
    using double_limits = std::numeric_limits<double>;

    SECTION("op(scalar, scalar)") {
        STATIC_REQUIRE(op(std::int8_t(-128), std::int8_t(127)) == -1);
        STATIC_REQUIRE(op(std::int8_t(64), std::int8_t(63)) == 127);
        STATIC_REQUIRE(op(std::int64_t(-1), std::int64_t(1)) == 0);

        STATIC_REQUIRE(op(std::int8_t(127), std::int8_t(127)) == int8_limits::max());
        STATIC_REQUIRE(op(std::int8_t(100), std::int8_t(100)) == int8_limits::max());
        STATIC_REQUIRE(op(std::int16_t(32767), std::int16_t(2)) == int16_limits::max());
        STATIC_REQUIRE(op(std::int32_t(105), std::int32_t(2147483647)) == int32_limits::max());
        STATIC_REQUIRE(op(int8_limits::max(), std::int8_t(1)) == int8_limits::max());
        STATIC_REQUIRE(op(int16_limits::max(), std::int16_t(1)) == int16_limits::max());
        STATIC_REQUIRE(op(int32_limits::max(), std::int32_t(1)) == int32_limits::max());
        STATIC_REQUIRE(op(int64_limits::max(), std::int64_t(1)) == int64_limits::max());

        STATIC_REQUIRE(op(std::int8_t(-128), std::int8_t(-1)) == int8_limits::lowest());
        STATIC_REQUIRE(op(std::int8_t(-128), std::int8_t(-128)) == int8_limits::lowest());
        STATIC_REQUIRE(op(int16_limits::lowest(), std::int16_t(-1)) == int16_limits::lowest());
        STATIC_REQUIRE(op(int32_limits::lowest(), std::int32_t(-1)) == int32_limits::lowest());
        STATIC_REQUIRE(op(int64_limits::lowest(), std::int64_t(-1)) == int64_limits::lowest());

        STATIC_REQUIRE(
            op(-double_limits::infinity(), double_limits::infinity()) == double_limits::infinity()
        );
    }

    SECTION("op(interval, interval)") {
        STATIC_REQUIRE(
            op(interval<float>(-float_limits::infinity(), 0),
               interval<float>(float_limits::infinity(), float_limits::infinity())) ==
            interval<float>(float_limits::infinity(), float_limits::infinity())
        );
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

TEMPLATE_LIST_TEST_CASE("divide", "", DTypes) {
    if constexpr (not std::floating_point<TestType>) {
        // Follow NumPy, which has no integral or boolean loops for true_divide --
        // integer division there happens by promoting to float64 first
        STATIC_REQUIRE(not std::invocable<divide, TestType, TestType>);
        STATIC_REQUIRE(not std::invocable<divide, interval<TestType>, interval<TestType>>);
    } else {
        constexpr divide op{};
        using limits = std::numeric_limits<TestType>;
        constexpr TestType inf = limits::infinity();

        SECTION("op(scalar, scalar)") {
            STATIC_REQUIRE(std::same_as<decltype(op(TestType(0), TestType(0))), TestType>);

            STATIC_REQUIRE(op(TestType(0), TestType(1)) == TestType(0));
            STATIC_REQUIRE(op(TestType(1), TestType(1)) == TestType(1));
            STATIC_REQUIRE(op(TestType(1), TestType(2)) == TestType(0.5));
            STATIC_REQUIRE(op(TestType(6), TestType(3)) == TestType(2));

            STATIC_REQUIRE(op(TestType(-1), TestType(2)) == TestType(-0.5));
            STATIC_REQUIRE(op(TestType(1), TestType(-2)) == TestType(-0.5));
            STATIC_REQUIRE(op(TestType(-1), TestType(-2)) == TestType(0.5));
            STATIC_REQUIRE(op(TestType(-6), TestType(-3)) == TestType(2));
            // x / 0 := 0, including for -0.0 since -0.0 == 0.0
            STATIC_REQUIRE(op(TestType(0), TestType(0)) == TestType(0));
            STATIC_REQUIRE(op(TestType(5), TestType(0)) == TestType(0));
            STATIC_REQUIRE(op(TestType(-5), TestType(0)) == TestType(0));
            STATIC_REQUIRE(op(TestType(5), TestType(-0.0)) == TestType(0));
            STATIC_REQUIRE(op(TestType(-5), TestType(-0.0)) == TestType(0));
            STATIC_REQUIRE(op(limits::max(), TestType(0)) == TestType(0));
            STATIC_REQUIRE(op(inf, TestType(0)) == TestType(0));
            STATIC_REQUIRE(op(-inf, TestType(0)) == TestType(0));

            // safe division keeps the usual sign rule"
            STATIC_REQUIRE(not std::signbit(op(TestType(5), TestType(0.0))));
            STATIC_REQUIRE(std::signbit(op(TestType(5), TestType(-0.0))));
            STATIC_REQUIRE(std::signbit(op(TestType(-5), TestType(0.0))));
            STATIC_REQUIRE(not std::signbit(op(TestType(-5), TestType(-0.0))));

            // both operands zero
            STATIC_REQUIRE(not std::signbit(op(TestType(0.0), TestType(0.0))));
            STATIC_REQUIRE(std::signbit(op(TestType(-0.0), TestType(0.0))));
            STATIC_REQUIRE(std::signbit(op(TestType(0.0), TestType(-0.0))));
            STATIC_REQUIRE(not std::signbit(op(TestType(-0.0), TestType(-0.0))));

            // an infinite numerator over zero
            STATIC_REQUIRE(not std::signbit(op(inf, TestType(0.0))));
            STATIC_REQUIRE(std::signbit(op(inf, TestType(-0.0))));
            STATIC_REQUIRE(std::signbit(op(-inf, TestType(0.0))));
            STATIC_REQUIRE(not std::signbit(op(-inf, TestType(-0.0))));

            // signed zero results from ordinary division
            STATIC_REQUIRE(not std::signbit(op(TestType(0.0), TestType(1))));
            STATIC_REQUIRE(std::signbit(op(TestType(-0.0), TestType(1))));
            STATIC_REQUIRE(std::signbit(op(TestType(0.0), TestType(-1))));
            STATIC_REQUIRE(not std::signbit(op(TestType(-0.0), TestType(-1))));

            STATIC_REQUIRE(not std::signbit(op(TestType(1), inf)));
            STATIC_REQUIRE(std::signbit(op(TestType(1), -inf)));
            STATIC_REQUIRE(std::signbit(op(TestType(-1), inf)));
            STATIC_REQUIRE(not std::signbit(op(TestType(-1), -inf)));

            // infinities
            STATIC_REQUIRE(op(inf, TestType(1)) == inf);
            STATIC_REQUIRE(op(inf, TestType(-1)) == -inf);
            STATIC_REQUIRE(op(-inf, TestType(1)) == -inf);
            STATIC_REQUIRE(op(-inf, TestType(-1)) == inf);

            STATIC_REQUIRE(op(TestType(1), inf) == TestType(0));
            STATIC_REQUIRE(op(TestType(1), -inf) == TestType(0));
            STATIC_REQUIRE(op(limits::max(), inf) == TestType(0));

            // unlike NumPy, inf / inf is an infinity rather than nan, signed as usual
            STATIC_REQUIRE(op(inf, inf) == inf);
            STATIC_REQUIRE(op(inf, -inf) == -inf);
            STATIC_REQUIRE(op(-inf, inf) == -inf);
            STATIC_REQUIRE(op(-inf, -inf) == inf);

            // underflows to zero, which raises no flag that blocks constant evaluation
            STATIC_REQUIRE(op(limits::denorm_min(), limits::max()) == TestType(0));

            // overflows to infinity -- the overflow flag makes these non-constexpr on
            // GCC even though clang accepts them
            CHECK(op(limits::max(), TestType(0.5)) == inf);
            CHECK(op(limits::max(), limits::denorm_min()) == inf);
        }

        SECTION("op(interval, interval)") {
            constexpr interval<TestType> zero(0, 0);
            constexpr interval<TestType> one(1, 1);
            constexpr interval<TestType> both(0, 1);
            constexpr interval<TestType> empty;

            STATIC_REQUIRE(std::same_as<decltype(op(zero, zero)), interval<TestType>>);

            STATIC_REQUIRE(op(zero, empty) == empty);
            STATIC_REQUIRE(op(one, empty) == empty);
            STATIC_REQUIRE(op(both, empty) == empty);
            STATIC_REQUIRE(op(empty, zero) == empty);
            STATIC_REQUIRE(op(empty, one) == empty);
            STATIC_REQUIRE(op(empty, both) == empty);
            STATIC_REQUIRE(op(empty, empty) == empty);

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

TEMPLATE_LIST_TEST_CASE("equal", "", DTypes) {
    constexpr equal op{};

    SECTION("op(scalar, scalar)") {
        STATIC_REQUIRE(std::same_as<decltype(op(TestType(0), TestType(0))), bool>);
        STATIC_REQUIRE(op(TestType(0), TestType(0)));
        STATIC_REQUIRE(op(TestType(1), TestType(1)));
        STATIC_REQUIRE(not op(TestType(0), TestType(1)));
        STATIC_REQUIRE(not op(TestType(1), TestType(0)));
    }

    SECTION("op(interval, interval)") {
        constexpr interval<TestType> zero(0, 0);
        constexpr interval<TestType> one(1, 1);
        constexpr interval<TestType> both(0, 1);
        constexpr interval<TestType> empty;
        constexpr interval<bool> yes(true, true);
        constexpr interval<bool> no(false, false);
        constexpr interval<bool> maybe(false, true);

        STATIC_REQUIRE(std::same_as<decltype(op(zero, zero)), interval<bool>>);

        STATIC_REQUIRE(op(zero, zero) == yes);
        STATIC_REQUIRE(op(zero, one) == no);
        STATIC_REQUIRE(op(zero, both) == maybe);
        STATIC_REQUIRE(op(zero, empty) == empty);

        STATIC_REQUIRE(op(one, zero) == no);
        STATIC_REQUIRE(op(one, one) == yes);
        STATIC_REQUIRE(op(one, both) == maybe);
        STATIC_REQUIRE(op(one, empty) == empty);

        STATIC_REQUIRE(op(both, zero) == maybe);
        STATIC_REQUIRE(op(both, one) == maybe);
        STATIC_REQUIRE(op(both, both) == maybe);
        STATIC_REQUIRE(op(both, empty) == empty);

        STATIC_REQUIRE(op(empty, zero) == empty);
        STATIC_REQUIRE(op(empty, one) == empty);
        STATIC_REQUIRE(op(empty, both) == empty);
        STATIC_REQUIRE(op(empty, empty) == empty);

        if constexpr (not std::same_as<TestType, bool>) {
            STATIC_REQUIRE(op(interval<TestType>(1, 1), interval<TestType>(2, 2)) == no);
            STATIC_REQUIRE(op(interval<TestType>(0, 1), interval<TestType>(2, 3)) == no);
            STATIC_REQUIRE(op(interval<TestType>(-3, -2), interval<TestType>(1, 2)) == no);

            // touching at a single endpoint is still an overlap
            STATIC_REQUIRE(op(interval<TestType>(0, 2), interval<TestType>(2, 4)) == maybe);
            // containment
            STATIC_REQUIRE(op(interval<TestType>(1, 5), interval<TestType>(2, 3)) == maybe);
        }

        if constexpr (std::floating_point<TestType>) {
            constexpr TestType inf = std::numeric_limits<TestType>::infinity();

            // -0.0 == 0.0, so these are the same single value
            STATIC_REQUIRE(op(interval<TestType>(-0.0, -0.0), interval<TestType>(0.0, 0.0)) == yes);

            STATIC_REQUIRE(op(interval<TestType>(inf, inf), interval<TestType>(inf, inf)) == yes);
            // nothing finite equals inf
            STATIC_REQUIRE(
                op(interval<TestType>(inf, inf),
                   interval<TestType>(0, std::numeric_limits<TestType>::max())) == no
            );
            STATIC_REQUIRE(op(interval<TestType>::all(), one) == maybe);
            STATIC_REQUIRE(op(interval<TestType>::all(), interval<TestType>::all()) == maybe);
        }
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

TEMPLATE_LIST_TEST_CASE("less_equal", "", DTypes) {
    constexpr less_equal op{};

    SECTION("op(scalar, scalar)") {
        STATIC_REQUIRE(std::same_as<decltype(op(TestType(0), TestType(0))), bool>);
        STATIC_REQUIRE(op(TestType(0), TestType(0)));
        STATIC_REQUIRE(op(TestType(0), TestType(1)));
        STATIC_REQUIRE(op(TestType(1), TestType(1)));
        STATIC_REQUIRE(not op(TestType(1), TestType(0)));
    }

    SECTION("op(interval, interval)") {
        constexpr interval<TestType> zero(0, 0);
        constexpr interval<TestType> one(1, 1);
        constexpr interval<TestType> both(0, 1);
        constexpr interval<TestType> empty;
        constexpr interval<bool> yes(true, true);
        constexpr interval<bool> no(false, false);
        constexpr interval<bool> maybe(false, true);

        STATIC_REQUIRE(std::same_as<decltype(op(zero, zero)), interval<bool>>);

        STATIC_REQUIRE(op(zero, zero) == yes);
        STATIC_REQUIRE(op(zero, one) == yes);
        STATIC_REQUIRE(op(zero, both) == yes);
        STATIC_REQUIRE(op(zero, empty) == empty);

        STATIC_REQUIRE(op(one, zero) == no);
        STATIC_REQUIRE(op(one, one) == yes);
        STATIC_REQUIRE(op(one, both) == maybe);
        STATIC_REQUIRE(op(one, empty) == empty);

        STATIC_REQUIRE(op(both, zero) == maybe);
        STATIC_REQUIRE(op(both, one) == yes);  // 0 <= 1 and 1 <= 1
        STATIC_REQUIRE(op(both, both) == maybe);
        STATIC_REQUIRE(op(both, empty) == empty);

        STATIC_REQUIRE(op(empty, zero) == empty);
        STATIC_REQUIRE(op(empty, one) == empty);
        STATIC_REQUIRE(op(empty, both) == empty);
        STATIC_REQUIRE(op(empty, empty) == empty);

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

TEMPLATE_LIST_TEST_CASE("logical_and", "", DTypes) {
    constexpr logical_and op{};

    SECTION("op(scalar, scalar)") {
        STATIC_REQUIRE(std::same_as<decltype(op(TestType(0), TestType(0))), bool>);

        STATIC_REQUIRE(not op(TestType(0), TestType(0)));
        STATIC_REQUIRE(not op(TestType(0), TestType(1)));
        STATIC_REQUIRE(not op(TestType(1), TestType(0)));
        STATIC_REQUIRE(op(TestType(1), TestType(1)));

        if constexpr (not std::same_as<TestType, bool>) {
            STATIC_REQUIRE(not op(TestType(0), TestType(1)));
            STATIC_REQUIRE(not op(TestType(-1), TestType(0)));
            STATIC_REQUIRE(op(TestType(-1), TestType(14)));
        }
        if constexpr (std::floating_point<TestType>) {
            STATIC_REQUIRE(not op(TestType(-0.0), TestType(0.0)));
            STATIC_REQUIRE(op(TestType(.0000001), TestType(.0000001)));
        }
    }

    SECTION("op(interval, interval)") {
        STATIC_REQUIRE(
            std::same_as<decltype(op(interval<TestType>(), interval<TestType>())), interval<bool>>
        );

        constexpr interval<TestType> falsy(0, 0);
        constexpr interval<TestType> truthy(1, 1);
        constexpr interval<TestType> ambiguous(0, 1);
        constexpr interval<TestType> empty;

        STATIC_REQUIRE(op(falsy, falsy) == falsy);
        STATIC_REQUIRE(op(falsy, truthy) == falsy);
        STATIC_REQUIRE(op(falsy, ambiguous) == falsy);
        STATIC_REQUIRE(op(falsy, empty) == empty);

        STATIC_REQUIRE(op(truthy, falsy) == falsy);
        STATIC_REQUIRE(op(truthy, truthy) == truthy);
        STATIC_REQUIRE(op(truthy, ambiguous) == ambiguous);
        STATIC_REQUIRE(op(truthy, empty) == empty);

        STATIC_REQUIRE(op(ambiguous, falsy) == falsy);
        STATIC_REQUIRE(op(ambiguous, truthy) == ambiguous);
        STATIC_REQUIRE(op(ambiguous, ambiguous) == ambiguous);
        STATIC_REQUIRE(op(ambiguous, empty) == empty);

        STATIC_REQUIRE(op(empty, falsy) == empty);
        STATIC_REQUIRE(op(empty, truthy) == empty);
        STATIC_REQUIRE(op(empty, ambiguous) == empty);
        STATIC_REQUIRE(op(empty, empty) == empty);

        if constexpr (not std::same_as<TestType, bool>) {
            constexpr interval<TestType> positive(1, 2);
            constexpr interval<TestType> negative(-3, -2);
            constexpr interval<TestType> wide(-5, 10);

            STATIC_REQUIRE(op(positive, falsy) == falsy);
            STATIC_REQUIRE(op(positive, truthy) == truthy);
            STATIC_REQUIRE(op(positive, ambiguous) == ambiguous);
            STATIC_REQUIRE(op(positive, empty) == empty);
            STATIC_REQUIRE(op(positive, negative) == truthy);
            STATIC_REQUIRE(op(positive, wide) == ambiguous);

            STATIC_REQUIRE(op(negative, falsy) == falsy);
            STATIC_REQUIRE(op(negative, truthy) == truthy);
            STATIC_REQUIRE(op(negative, ambiguous) == ambiguous);
            STATIC_REQUIRE(op(negative, empty) == empty);
            STATIC_REQUIRE(op(negative, negative) == truthy);
            STATIC_REQUIRE(op(negative, wide) == ambiguous);

            STATIC_REQUIRE(op(wide, falsy) == falsy);
            STATIC_REQUIRE(op(wide, truthy) == ambiguous);
            STATIC_REQUIRE(op(wide, ambiguous) == ambiguous);
            STATIC_REQUIRE(op(wide, empty) == empty);
            STATIC_REQUIRE(op(wide, negative) == ambiguous);
            STATIC_REQUIRE(op(wide, wide) == ambiguous);
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

TEMPLATE_LIST_TEST_CASE("logical_or", "", DTypes) {
    constexpr logical_or op{};

    SECTION("op(scalar, scalar)") {
        STATIC_REQUIRE(std::same_as<decltype(op(TestType(0), TestType(0))), bool>);

        STATIC_REQUIRE(not op(TestType(0), TestType(0)));
        STATIC_REQUIRE(op(TestType(0), TestType(1)));
        STATIC_REQUIRE(op(TestType(1), TestType(0)));
        STATIC_REQUIRE(op(TestType(1), TestType(1)));

        if constexpr (not std::same_as<TestType, bool>) {
            STATIC_REQUIRE(op(TestType(0), TestType(1)));
            STATIC_REQUIRE(op(TestType(-1), TestType(0)));
            STATIC_REQUIRE(op(TestType(-1), TestType(14)));
        }
        if constexpr (std::floating_point<TestType>) {
            STATIC_REQUIRE(not op(TestType(-0.0), TestType(0.0)));
            STATIC_REQUIRE(op(TestType(.0000001), TestType(.0000001)));
        }
    }

    SECTION("op(interval, interval)") {
        STATIC_REQUIRE(
            std::same_as<decltype(op(interval<TestType>(), interval<TestType>())), interval<bool>>
        );

        constexpr interval<TestType> falsy(0, 0);
        constexpr interval<TestType> truthy(1, 1);
        constexpr interval<TestType> ambiguous(0, 1);
        constexpr interval<TestType> empty;

        STATIC_REQUIRE(op(falsy, falsy) == falsy);
        STATIC_REQUIRE(op(falsy, truthy) == truthy);
        STATIC_REQUIRE(op(falsy, ambiguous) == ambiguous);
        STATIC_REQUIRE(op(falsy, empty) == empty);

        STATIC_REQUIRE(op(truthy, falsy) == truthy);
        STATIC_REQUIRE(op(truthy, truthy) == truthy);
        STATIC_REQUIRE(op(truthy, ambiguous) == truthy);
        STATIC_REQUIRE(op(truthy, empty) == empty);

        STATIC_REQUIRE(op(ambiguous, falsy) == ambiguous);
        STATIC_REQUIRE(op(ambiguous, truthy) == truthy);
        STATIC_REQUIRE(op(ambiguous, ambiguous) == ambiguous);
        STATIC_REQUIRE(op(ambiguous, empty) == empty);

        STATIC_REQUIRE(op(empty, falsy) == empty);
        STATIC_REQUIRE(op(empty, truthy) == empty);
        STATIC_REQUIRE(op(empty, ambiguous) == empty);
        STATIC_REQUIRE(op(empty, empty) == empty);

        if constexpr (not std::same_as<TestType, bool>) {
            constexpr interval<TestType> positive(1, 2);
            constexpr interval<TestType> negative(-3, -2);
            constexpr interval<TestType> wide(-5, 10);

            STATIC_REQUIRE(op(positive, falsy) == truthy);
            STATIC_REQUIRE(op(positive, truthy) == truthy);
            STATIC_REQUIRE(op(positive, ambiguous) == truthy);
            STATIC_REQUIRE(op(positive, empty) == empty);
            STATIC_REQUIRE(op(positive, negative) == truthy);
            STATIC_REQUIRE(op(positive, wide) == truthy);

            STATIC_REQUIRE(op(negative, falsy) == truthy);
            STATIC_REQUIRE(op(negative, truthy) == truthy);
            STATIC_REQUIRE(op(negative, ambiguous) == truthy);
            STATIC_REQUIRE(op(negative, empty) == empty);
            STATIC_REQUIRE(op(negative, negative) == truthy);
            STATIC_REQUIRE(op(negative, wide) == truthy);

            STATIC_REQUIRE(op(wide, falsy) == ambiguous);
            STATIC_REQUIRE(op(wide, truthy) == truthy);
            STATIC_REQUIRE(op(wide, ambiguous) == ambiguous);
            STATIC_REQUIRE(op(wide, empty) == empty);
            STATIC_REQUIRE(op(wide, negative) == truthy);
            STATIC_REQUIRE(op(wide, wide) == ambiguous);
        }
    }
}

TEMPLATE_LIST_TEST_CASE("logical_xor", "", DTypes) {
    constexpr logical_xor op{};

    SECTION("op(scalar, scalar)") {
        STATIC_REQUIRE(std::same_as<decltype(op(TestType(0), TestType(0))), bool>);

        STATIC_REQUIRE(not op(TestType(0), TestType(0)));
        STATIC_REQUIRE(op(TestType(0), TestType(1)));
        STATIC_REQUIRE(op(TestType(1), TestType(0)));
        STATIC_REQUIRE(not op(TestType(1), TestType(1)));

        if constexpr (not std::same_as<TestType, bool>) {
            STATIC_REQUIRE(op(TestType(0), TestType(1)));
            STATIC_REQUIRE(op(TestType(-1), TestType(0)));
            STATIC_REQUIRE(not op(TestType(-1), TestType(14)));
        }
        if constexpr (std::floating_point<TestType>) {
            STATIC_REQUIRE(not op(TestType(-0.0), TestType(0.0)));
            STATIC_REQUIRE(not op(TestType(.0000001), TestType(.0000001)));
        }
    }

    SECTION("op(interval, interval)") {
        STATIC_REQUIRE(
            std::same_as<decltype(op(interval<TestType>(), interval<TestType>())), interval<bool>>
        );

        constexpr interval<TestType> falsy(0, 0);
        constexpr interval<TestType> truthy(1, 1);
        constexpr interval<TestType> ambiguous(0, 1);
        constexpr interval<TestType> empty;

        STATIC_REQUIRE(op(falsy, falsy) == falsy);
        STATIC_REQUIRE(op(falsy, truthy) == truthy);
        STATIC_REQUIRE(op(falsy, ambiguous) == ambiguous);
        STATIC_REQUIRE(op(falsy, empty) == empty);

        STATIC_REQUIRE(op(truthy, falsy) == truthy);
        STATIC_REQUIRE(op(truthy, truthy) == falsy);
        STATIC_REQUIRE(op(truthy, ambiguous) == ambiguous);
        STATIC_REQUIRE(op(truthy, empty) == empty);

        STATIC_REQUIRE(op(ambiguous, falsy) == ambiguous);
        STATIC_REQUIRE(op(ambiguous, truthy) == ambiguous);
        STATIC_REQUIRE(op(ambiguous, ambiguous) == ambiguous);
        STATIC_REQUIRE(op(ambiguous, empty) == empty);

        STATIC_REQUIRE(op(empty, falsy) == empty);
        STATIC_REQUIRE(op(empty, truthy) == empty);
        STATIC_REQUIRE(op(empty, ambiguous) == empty);
        STATIC_REQUIRE(op(empty, empty) == empty);

        if constexpr (not std::same_as<TestType, bool>) {
            constexpr interval<TestType> positive(1, 2);
            constexpr interval<TestType> negative(-3, -2);
            constexpr interval<TestType> wide(-5, 10);

            STATIC_REQUIRE(op(positive, falsy) == truthy);
            STATIC_REQUIRE(op(positive, truthy) == falsy);
            STATIC_REQUIRE(op(positive, ambiguous) == ambiguous);
            STATIC_REQUIRE(op(positive, empty) == empty);
            STATIC_REQUIRE(op(positive, negative) == falsy);
            STATIC_REQUIRE(op(positive, wide) == ambiguous);

            STATIC_REQUIRE(op(negative, falsy) == truthy);
            STATIC_REQUIRE(op(negative, truthy) == falsy);
            STATIC_REQUIRE(op(negative, ambiguous) == ambiguous);
            STATIC_REQUIRE(op(negative, empty) == empty);
            STATIC_REQUIRE(op(negative, negative) == falsy);
            STATIC_REQUIRE(op(negative, wide) == ambiguous);

            STATIC_REQUIRE(op(wide, falsy) == ambiguous);
            STATIC_REQUIRE(op(wide, truthy) == ambiguous);
            STATIC_REQUIRE(op(wide, ambiguous) == ambiguous);
            STATIC_REQUIRE(op(wide, empty) == empty);
            STATIC_REQUIRE(op(wide, negative) == ambiguous);
            STATIC_REQUIRE(op(wide, wide) == ambiguous);
        }
    }
}

TEMPLATE_LIST_TEST_CASE("maximum", "", DTypes) {
    constexpr maximum op{};

    SECTION("op(scalar, scalar)") {
        STATIC_REQUIRE(std::same_as<decltype(op(TestType(0), TestType(0))), TestType>);
        STATIC_REQUIRE(op(TestType(0), TestType(0)) == TestType(0));
        STATIC_REQUIRE(op(TestType(0), TestType(1)) == TestType(1));
        STATIC_REQUIRE(op(TestType(1), TestType(0)) == TestType(1));
        STATIC_REQUIRE(op(TestType(1), TestType(1)) == TestType(1));

        STATIC_REQUIRE(
            op(std::numeric_limits<TestType>::lowest(), std::numeric_limits<TestType>::max()) ==
            std::numeric_limits<TestType>::max()
        );
    }

    SECTION("op(interval, interval)") {
        constexpr interval<TestType> zero(0, 0);
        constexpr interval<TestType> one(1, 1);
        constexpr interval<TestType> both(0, 1);
        constexpr interval<TestType> empty;

        STATIC_REQUIRE(std::same_as<decltype(op(zero, zero)), interval<TestType>>);

        STATIC_REQUIRE(op(zero, zero) == zero);
        STATIC_REQUIRE(op(zero, one) == one);
        STATIC_REQUIRE(op(zero, both) == both);
        STATIC_REQUIRE(op(zero, empty) == empty);

        STATIC_REQUIRE(op(one, zero) == one);
        STATIC_REQUIRE(op(one, one) == one);
        STATIC_REQUIRE(op(one, both) == one);
        STATIC_REQUIRE(op(one, empty) == empty);

        STATIC_REQUIRE(op(both, zero) == both);
        STATIC_REQUIRE(op(both, one) == one);
        STATIC_REQUIRE(op(both, both) == both);
        STATIC_REQUIRE(op(both, empty) == empty);

        STATIC_REQUIRE(op(empty, zero) == empty);
        STATIC_REQUIRE(op(empty, one) == empty);
        STATIC_REQUIRE(op(empty, both) == empty);
        STATIC_REQUIRE(op(empty, empty) == empty);

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
            using inter = interval<TestType>;

            STATIC_REQUIRE(op(inter(inf, inf), inter(inf, inf)) == inter(inf, inf));
            STATIC_REQUIRE(op(inter(-inf, -inf), inter::all()) == inter(-inf, inf));
            STATIC_REQUIRE(op(inter(inf, inf), inter::all()) == inter(inf, inf));
            STATIC_REQUIRE(op(inter::all(), inter::all()) == inter::all());

            // -0.0 == 0.0 so which representation wins the tie is unobservable
            STATIC_REQUIRE(op(inter(-0.0, -0.0), inter(0.0, 0.0)) == inter(0.0, 0.0));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("minimum", "", DTypes) {
    constexpr minimum op{};

    SECTION("op(scalar, scalar)") {
        STATIC_REQUIRE(std::same_as<decltype(op(TestType(0), TestType(0))), TestType>);
        STATIC_REQUIRE(op(TestType(0), TestType(0)) == TestType(0));
        STATIC_REQUIRE(op(TestType(0), TestType(1)) == TestType(0));
        STATIC_REQUIRE(op(TestType(1), TestType(0)) == TestType(0));
        STATIC_REQUIRE(op(TestType(1), TestType(1)) == TestType(1));

        STATIC_REQUIRE(
            op(std::numeric_limits<TestType>::lowest(), std::numeric_limits<TestType>::max()) ==
            std::numeric_limits<TestType>::lowest()
        );
    }

    SECTION("op(interval, interval)") {
        constexpr interval<TestType> zero(0, 0);
        constexpr interval<TestType> one(1, 1);
        constexpr interval<TestType> both(0, 1);
        constexpr interval<TestType> empty;

        STATIC_REQUIRE(std::same_as<decltype(op(zero, zero)), interval<TestType>>);

        STATIC_REQUIRE(op(zero, zero) == zero);
        STATIC_REQUIRE(op(zero, one) == zero);
        STATIC_REQUIRE(op(zero, both) == zero);
        STATIC_REQUIRE(op(zero, empty) == empty);

        STATIC_REQUIRE(op(one, zero) == zero);
        STATIC_REQUIRE(op(one, one) == one);
        STATIC_REQUIRE(op(one, both) == both);
        STATIC_REQUIRE(op(one, empty) == empty);

        STATIC_REQUIRE(op(both, zero) == zero);
        STATIC_REQUIRE(op(both, one) == both);
        STATIC_REQUIRE(op(both, both) == both);
        STATIC_REQUIRE(op(both, empty) == empty);

        STATIC_REQUIRE(op(empty, zero) == empty);
        STATIC_REQUIRE(op(empty, one) == empty);
        STATIC_REQUIRE(op(empty, both) == empty);
        STATIC_REQUIRE(op(empty, empty) == empty);

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
            using inter = interval<TestType>;

            STATIC_REQUIRE(op(inter(inf, inf), inter(inf, inf)) == inter(inf, inf));
            STATIC_REQUIRE(op(inter(-inf, -inf), inter::all()) == inter(-inf, -inf));
            STATIC_REQUIRE(op(inter(inf, inf), inter::all()) == inter::all());
            STATIC_REQUIRE(op(inter::all(), inter::all()) == inter::all());

            // -0.0 == 0.0 so which representation wins the tie is unobservable
            STATIC_REQUIRE(op(inter(-0.0, -0.0), inter(0.0, 0.0)) == inter(0.0, 0.0));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("multiply", "", DTypes) {
    constexpr multiply op{};
    using limits = std::numeric_limits<TestType>;

    SECTION("op(scalar, scalar)") {
        STATIC_REQUIRE(std::same_as<decltype(op(TestType(0), TestType(0))), TestType>);

        STATIC_REQUIRE(op(TestType(0), TestType(0)) == TestType(0));
        STATIC_REQUIRE(op(TestType(0), TestType(1)) == TestType(0));
        STATIC_REQUIRE(op(TestType(1), TestType(0)) == TestType(0));
        STATIC_REQUIRE(op(TestType(1), TestType(1)) == TestType(1));

        STATIC_REQUIRE(op(limits::max(), TestType(1)) == limits::max());
        STATIC_REQUIRE(op(limits::max(), TestType(0)) == TestType(0));
        STATIC_REQUIRE(op(limits::lowest(), TestType(1)) == limits::lowest());
        STATIC_REQUIRE(op(limits::lowest(), TestType(0)) == TestType(0));

        if constexpr (not std::same_as<TestType, bool>) {
            STATIC_REQUIRE(op(TestType(2), TestType(3)) == TestType(6));
            STATIC_REQUIRE(op(TestType(-2), TestType(3)) == TestType(-6));
            STATIC_REQUIRE(op(TestType(2), TestType(-3)) == TestType(-6));
            STATIC_REQUIRE(op(TestType(-2), TestType(-3)) == TestType(6));
            STATIC_REQUIRE(op(TestType(1), TestType(-1)) == TestType(-1));
            STATIC_REQUIRE(op(TestType(0), TestType(-1)) == TestType(0));
        }
    }

    if constexpr (std::signed_integral<TestType>) {
        SECTION("saturating") {
            // positive overflow
            STATIC_REQUIRE(op(limits::max(), TestType(2)) == limits::max());
            STATIC_REQUIRE(op(TestType(2), limits::max()) == limits::max());
            STATIC_REQUIRE(op(limits::max(), limits::max()) == limits::max());
            STATIC_REQUIRE(op(limits::lowest(), limits::lowest()) == limits::max());
            STATIC_REQUIRE(op(limits::lowest(), TestType(-1)) == limits::max());
            STATIC_REQUIRE(op(limits::lowest(), TestType(-2)) == limits::max());

            // negative overflow
            STATIC_REQUIRE(op(limits::lowest(), TestType(2)) == limits::lowest());
            STATIC_REQUIRE(op(TestType(2), limits::lowest()) == limits::lowest());
            STATIC_REQUIRE(op(limits::max(), TestType(-2)) == limits::lowest());
            STATIC_REQUIRE(op(TestType(-2), limits::max()) == limits::lowest());
            STATIC_REQUIRE(op(limits::lowest(), limits::max()) == limits::lowest());
            STATIC_REQUIRE(op(limits::max(), limits::lowest()) == limits::lowest());

            // just shy of saturating
            STATIC_REQUIRE(op(limits::max(), TestType(-1)) == TestType(limits::lowest() + 1));
            STATIC_REQUIRE(op(TestType(-1), limits::max()) == TestType(limits::lowest() + 1));
        }
    }

    if constexpr (std::floating_point<TestType>) {
        constexpr TestType inf = limits::infinity();

        SECTION("infinities") {
            STATIC_REQUIRE(op(inf, TestType(2)) == inf);
            STATIC_REQUIRE(op(inf, TestType(-2)) == -inf);
            STATIC_REQUIRE(op(-inf, TestType(2)) == -inf);
            STATIC_REQUIRE(op(-inf, TestType(-2)) == inf);
            STATIC_REQUIRE(op(inf, inf) == inf);
            STATIC_REQUIRE(op(inf, -inf) == -inf);
            STATIC_REQUIRE(op(-inf, -inf) == inf);

            // underflows to zero, which raises no flag that blocks constant evaluation
            STATIC_REQUIRE(op(limits::denorm_min(), TestType(0.5)) == TestType(0));

            // overflows to infinity -- the overflow flag makes these non-constexpr on
            // GCC even though clang accepts them
            CHECK(op(limits::max(), TestType(2)) == inf);
            CHECK(op(limits::max(), limits::max()) == inf);
            CHECK(op(limits::max(), limits::lowest()) == -inf);
        }

        SECTION("signed zeros") {
            STATIC_REQUIRE(not std::signbit(op(TestType(0.0), TestType(0.0))));
            STATIC_REQUIRE(std::signbit(op(TestType(-0.0), TestType(0.0))));
            STATIC_REQUIRE(std::signbit(op(TestType(0.0), TestType(-0.0))));
            STATIC_REQUIRE(not std::signbit(op(TestType(-0.0), TestType(-0.0))));

            STATIC_REQUIRE(std::signbit(op(TestType(-1), TestType(0.0))));
            STATIC_REQUIRE(std::signbit(op(TestType(1), TestType(-0.0))));
            STATIC_REQUIRE(not std::signbit(op(TestType(-1), TestType(-0.0))));
        }

        SECTION("zero times infinity") {
            STATIC_REQUIRE(op(TestType(0), inf) == TestType(0));
            STATIC_REQUIRE(op(inf, TestType(0)) == TestType(0));
            STATIC_REQUIRE(not std::signbit(op(TestType(0.0), inf)));
            STATIC_REQUIRE(std::signbit(op(TestType(-0.0), inf)));
            STATIC_REQUIRE(std::signbit(op(TestType(0.0), -inf)));
            STATIC_REQUIRE(not std::signbit(op(TestType(-0.0), -inf)));
        }
    }

    SECTION("op(interval, interval)") {
        constexpr interval<TestType> zero(0, 0);
        constexpr interval<TestType> one(1, 1);
        constexpr interval<TestType> both(0, 1);
        constexpr interval<TestType> empty;

        STATIC_REQUIRE(std::same_as<decltype(op(zero, zero)), interval<TestType>>);

        STATIC_REQUIRE(op(zero, empty) == empty);
        STATIC_REQUIRE(op(one, empty) == empty);
        STATIC_REQUIRE(op(both, empty) == empty);
        STATIC_REQUIRE(op(empty, zero) == empty);
        STATIC_REQUIRE(op(empty, one) == empty);
        STATIC_REQUIRE(op(empty, both) == empty);
        STATIC_REQUIRE(op(empty, empty) == empty);

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

TEMPLATE_LIST_TEST_CASE("remainder", "", DTypes) {
    // dev note: As of Sept 2026, no version of Apple Clang supports `constexpr std::fmod()`,
    // so we use CHECK for the tests that run through the floating branch.

    constexpr remainder op;

    SECTION("op(scalar, scalar)") {
        CHECK(op(TestType(0), TestType(0)) == 0);
        CHECK(op(TestType(1), TestType(0)) == 0);
        CHECK(op(TestType(0), TestType(1)) == 0);
        CHECK(op(TestType(1), TestType(1)) == 0);

        if constexpr (not std::same_as<TestType, bool>) {
            CHECK(op(TestType(0), TestType(-1)) == 0);
            CHECK(op(TestType(-1), TestType(-10)) == -1);
            CHECK(op(TestType(-1), TestType(10)) == 9);
            CHECK(op(TestType(1), TestType(-10)) == -9);
        }

        if constexpr (std::floating_point<TestType>) {
            CHECK(op(TestType(-5.5), TestType(-4)) == -1.5);
            CHECK(op(TestType(-5.5), TestType(4)) == 2.5);
            CHECK(op(TestType(5.5), TestType(-4)) == -2.5);
            CHECK(op(TestType(5.5), TestType(4)) == 1.5);

            CHECK(std::signbit(op(TestType(-0.0), TestType(-0.0))));
            CHECK(std::signbit(op(TestType(0.0), TestType(-0.0))));
            CHECK(not std::signbit(op(TestType(-0.0), TestType(0.0))));
            CHECK(not std::signbit(op(TestType(0.0), TestType(0.0))));

            CHECK(not std::signbit(op(TestType(-5.0), TestType(5.0))));
            CHECK(std::signbit(op(TestType(5.0), TestType(-5.0))));
            CHECK(not std::signbit(op(TestType(-0.0), TestType(5.0))));

            constexpr TestType inf = std::numeric_limits<TestType>::infinity();

            CHECK(op(TestType(-5.5), -inf) == -5.5);
            CHECK(op(TestType(-5.5), +inf) == +inf);
            CHECK(op(TestType(+5.5), -inf) == -inf);
            CHECK(op(TestType(+5.5), +inf) == TestType(5.5));

            CHECK(op(+inf, TestType(0)) == 0);
            CHECK(op(-inf, TestType(0)) == 0);

            CHECK(op(-inf, TestType(-5)) == 0);
            CHECK(op(+inf, TestType(-5)) == 0);
            CHECK(op(-inf, TestType(+5)) == 0);
            CHECK(op(+inf, TestType(+5)) == 0);

            CHECK(op(-inf, -inf) == 0);
            CHECK(op(+inf, -inf) == 0);
            CHECK(op(-inf, +inf) == 0);
            CHECK(op(+inf, +inf) == 0);

            CHECK(std::signbit(op(-inf, TestType(-0.0))));
            CHECK(std::signbit(op(-inf, TestType(-5.0))));

            CHECK(std::signbit(op(+inf, TestType(-0.0))));
            CHECK(std::signbit(op(+inf, TestType(-5.0))));

            CHECK(not std::signbit(op(-inf, TestType(+0.0))));
            CHECK(not std::signbit(op(-inf, TestType(+5.0))));

            CHECK(not std::signbit(op(+inf, TestType(+0.0))));
            CHECK(not std::signbit(op(+inf, TestType(+5.0))));
        }
    }

    SECTION("op(interval, interval)") {
        constexpr interval<TestType> zero(0, 0);
        constexpr interval<TestType> one(1, 1);
        constexpr interval<TestType> both(0, 1);
        constexpr interval<TestType> empty;
        constexpr auto all = interval<TestType>::all();

        STATIC_REQUIRE(std::same_as<decltype(op(zero, zero)), interval<TestType>>);

        CHECK(op(zero, zero) == zero);
        CHECK(op(zero, empty) == empty);

        CHECK(op(empty, zero) == empty);
        CHECK(op(empty, one) == empty);
        CHECK(op(empty, both) == empty);
        CHECK(op(empty, empty) == empty);

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

TEMPLATE_LIST_TEST_CASE("subtract", "", DTypes) {
    constexpr subtract op{};

    using T = TestType;
    using limits = std::numeric_limits<T>;

    if constexpr (std::same_as<T, bool>) {
        // Follow NumPy and disallow boolean subtraction
        STATIC_REQUIRE(not std::invocable<subtract, T, T>);
        STATIC_REQUIRE(not std::invocable<subtract, interval<T>, interval<T>>);
    } else {
        SECTION("op(scalar, scalar)") {
            // dtype is preseved
            STATIC_REQUIRE(std::same_as<decltype(op(T(0), T(0))), T>);

            STATIC_REQUIRE(op(T(0), T(0)) == T(0));
            STATIC_REQUIRE(op(T(1), T(0)) == T(1));
            STATIC_REQUIRE(op(T(0), T(1)) == T(-1));
            STATIC_REQUIRE(op(T(1), T(1)) == T(0));
            STATIC_REQUIRE(op(T(5), T(3)) == T(2));
            STATIC_REQUIRE(op(T(3), T(5)) == T(-2));

            STATIC_REQUIRE(op(limits::max(), limits::max()) == T(0));
            STATIC_REQUIRE(op(limits::lowest(), limits::lowest()) == T(0));

            // Check saturation for integers
            if constexpr (std::signed_integral<T>) {
                STATIC_REQUIRE(op(limits::max(), T(-1)) == limits::max());
                STATIC_REQUIRE(op(limits::lowest(), T(1)) == limits::lowest());
                STATIC_REQUIRE(op(limits::max(), limits::lowest()) == limits::max());
                STATIC_REQUIRE(op(limits::lowest(), limits::max()) == limits::lowest());
                STATIC_REQUIRE(op(T(0), limits::lowest()) == limits::max());

                STATIC_REQUIRE(op(limits::min(), T(1)) == limits::min());
                STATIC_REQUIRE(op(limits::lowest(), limits::max()) == limits::lowest());
                STATIC_REQUIRE(op(limits::lowest(), T(1)) == limits::lowest());
                STATIC_REQUIRE(op(limits::lowest(), limits::max()) == limits::lowest());
                STATIC_REQUIRE(op(limits::max(), T(1)) == T(limits::max() - 1));
                STATIC_REQUIRE(op(limits::lowest(), T(-1)) == T(limits::lowest() + 1));
            }

            if constexpr (std::floating_point<T>) {
                constexpr T inf = limits::infinity();

                STATIC_REQUIRE(op(inf, T(1)) == inf);
                STATIC_REQUIRE(op(T(1), inf) == -inf);
                STATIC_REQUIRE(op(-inf, T(1)) == -inf);
                STATIC_REQUIRE(op(T(1), -inf) == inf);

                STATIC_REQUIRE(op(inf, -inf) == inf);
                STATIC_REQUIRE(op(-inf, inf) == -inf);

                // unlike NumPy, inf - inf is inf rather than nan
                STATIC_REQUIRE(op(inf, inf) == inf);
                STATIC_REQUIRE(op(-inf, -inf) == inf);

                // overflows to infinity rather than saturating
                // Use CHECK because GCC doesn't like compile-time overflows. Clang doesn't care.
                CHECK(op(limits::max(), limits::lowest()) == inf);
                CHECK(op(limits::lowest(), limits::max()) == -inf);

                STATIC_REQUIRE(not std::signbit(op(T(0.0), T(0.0))));
                STATIC_REQUIRE(not std::signbit(op(T(0.0), T(-0.0))));
                STATIC_REQUIRE(std::signbit(op(T(-0.0), T(0.0))));
                STATIC_REQUIRE(not std::signbit(op(T(-0.0), T(-0.0))));
            }
        }

        SECTION("op(interval, interval)") {
            constexpr interval<T> zero(0, 0);
            constexpr interval<T> one(1, 1);
            constexpr interval<T> both(0, 1);
            constexpr interval<T> empty;

            STATIC_REQUIRE(std::same_as<decltype(op(zero, zero)), interval<T>>);

            STATIC_REQUIRE(op(zero, zero) == zero);
            STATIC_REQUIRE(op(zero, one) == interval<T>(-1, -1));
            STATIC_REQUIRE(op(zero, both) == interval<T>(-1, 0));
            STATIC_REQUIRE(op(zero, empty) == empty);

            STATIC_REQUIRE(op(one, zero) == one);
            STATIC_REQUIRE(op(one, one) == zero);
            STATIC_REQUIRE(op(one, both) == both);
            STATIC_REQUIRE(op(one, empty) == empty);

            STATIC_REQUIRE(op(both, zero) == both);
            STATIC_REQUIRE(op(both, one) == interval<T>(-1, 0));
            STATIC_REQUIRE(op(both, both) == interval<T>(-1, 1));
            STATIC_REQUIRE(op(both, empty) == empty);

            STATIC_REQUIRE(op(empty, zero) == empty);
            STATIC_REQUIRE(op(empty, one) == empty);
            STATIC_REQUIRE(op(empty, both) == empty);
            STATIC_REQUIRE(op(empty, empty) == empty);

            STATIC_REQUIRE(op(interval<T>(1, 2), interval<T>(10, 20)) == interval<T>(-19, -8));
            STATIC_REQUIRE(op(interval<T>::all(), interval<T>::all()) == interval<T>::all());

            if constexpr (std::integral<T>) {
                STATIC_REQUIRE(
                    op(interval<T>(limits::lowest(), limits::lowest()), one) ==
                    interval<T>(limits::lowest(), limits::lowest())
                );
                STATIC_REQUIRE(
                    op(zero, interval<T>(limits::lowest(), limits::lowest())) ==
                    interval<T>(limits::max(), limits::max())
                );
            }

            if constexpr (std::floating_point<T>) {
                constexpr T inf = limits::infinity();
                STATIC_REQUIRE(
                    op(interval<T>(inf, inf), interval<T>(inf, inf)) == interval<T>(inf, inf)
                );
                STATIC_REQUIRE(
                    op(interval<T>(-inf, -inf), interval<T>(-inf, -inf)) == interval<T>(inf, inf)
                );
            }
        }
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

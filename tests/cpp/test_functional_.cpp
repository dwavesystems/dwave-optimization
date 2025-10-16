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

#include "catch2/catch_template_test_macros.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "functional_.hpp"

namespace dwave::optimization::functional_ {

TEST_CASE("Add") {
    static_assert(Add<double>::associative);
    static_assert(Add<double>::commutative);
    static_assert(Add<double>::invertible);

    SECTION(".operator(...)") {
        const Add<double> op;

        CHECK(op(2, 3) == 5);
        CHECK(op(Add<double>::reduction_type(2), 3) == 5);
    }

    SECTION(".reduce(...)") {
        const Add<double> op;

        CHECK(op.reduce(std::vector{0, 1, 2}) == 0 + 1 + 2);
        CHECK(op.reduce(std::vector{0, 1, 2}, 5) == 0 + 1 + 2 + 5);
        CHECK(op.reduce(std::vector<int>{}, -85) == -85);

        auto reduction = op.reduce(std::vector{1, 3, 2}, 5);
        CHECK(reduction == 1 + 3 + 2 + 5);
        CHECK(op.inverse(reduction, 3) == 1 + 2 + 5);
    }

    SECTION(".result_bounds(...)") {
        const Add<double> op;

        CHECK(op.result_bounds({0, 5, true}, {1, 2, true}) == ValuesInfo(1, 7, true));
        CHECK(op.result_bounds({0, 5, true}, {1, 2, false}) == ValuesInfo(1, 7, false));

        CHECK(op.result_bounds({0, 5, true}, 5) == ValuesInfo(0, 25, true));
        CHECK(op.result_bounds({-1, 5, true}, 5) == ValuesInfo(-5, 25, true));

        // non-negative
        {
            auto info = op.result_bounds({0, 5, true}, limit_type());
            CHECK(info.min == 0);
            CHECK(info.max >= std::numeric_limits<double>::max());
            CHECK(info.integral);
        }

        // strictly positive
        {
            auto info = op.result_bounds({.00000000001, 5, false}, limit_type());
            CHECK(info.min >= std::numeric_limits<double>::max());
            CHECK(info.max >= std::numeric_limits<double>::max());
            CHECK(!info.integral);
        }

        // non-positive
        {
            auto info = op.result_bounds({-1, 0, true}, limit_type());
            CHECK(info.min <= std::numeric_limits<double>::min());
            CHECK(info.max == 0);
            CHECK(info.integral);
        }
    }
}

TEST_CASE("LogicalAnd") {
    static_assert(LogicalAnd::associative);
    static_assert(LogicalAnd::commutative);
    static_assert(LogicalAnd::invertible);

    SECTION(".inverse(...)") {
        const LogicalAnd op;

        SECTION("with numeric types") {
            CHECK(!op.inverse(0, 0).has_value());  // ambiguous
            CHECK(!op.inverse(1, 0).has_value());  // undefined
            CHECK(op.inverse(0, 1).value() == 0);
            CHECK(op.inverse(1, 1).value() == 1);
        }

        SECTION("with reduction_type") {
            auto b = GENERATE(0, 1);
            auto c = GENERATE(0, 1);
            auto d = LogicalAnd::reduction_type(b);
            CHECK(op.inverse(op(d, c), c) == d);  // always defined
        }
    }

    SECTION(".operator(...)") {
        const LogicalAnd op;

        CHECK(op(true, true));
        CHECK(!op(false, true));
        CHECK(!op(true, false));
        CHECK(!op(false, false));
        CHECK(op(LogicalAnd::reduction_type(true), true));
        CHECK(!op(LogicalAnd::reduction_type(false), true));
        CHECK(!op(LogicalAnd::reduction_type(true), false));
        CHECK(!op(LogicalAnd::reduction_type(false), false));

        CHECK(op.reduce(std::array{true, true, true}));
        CHECK(!op.reduce(std::array{false, true, true}));
        CHECK(op.reduce(std::array{true, true, true}, true));
        CHECK(!op.reduce(std::array{true, true, true}, false));
        CHECK(op.reduce(std::array<int, 0>{}, true));
        CHECK(!op.reduce(std::array<int, 0>{}, false));
    }

    SECTION(".reduce(...)") {
        const LogicalAnd op;

        auto reduction = op.reduce(std::vector{1, 0, 2}, 5);
        CHECK(!reduction);
        CHECK(op.inverse(reduction, 0));
    }

    SECTION(".result_bounds(...)") {
        const LogicalAnd op;

        CHECK(op.result_bounds({0, 0, false}, {0, 1, false}) == ValuesInfo(0, 0, true));
        CHECK(op.result_bounds({-100, 100, true}, 1) == ValuesInfo(0, 1, true));
        CHECK(op.result_bounds({0, 10.5, false}, limit_type()) == ValuesInfo(0, 1, true));
    }
}

TEST_CASE("LogicalOr") {
    static_assert(LogicalOr::associative);
    static_assert(LogicalOr::commutative);
    static_assert(LogicalOr::invertible);

    SECTION(".inverse(...)") {
        const LogicalOr op;

        SECTION("with numeric types") {
            CHECK(op.inverse(0, 0).value() == 0);
            CHECK(!op.inverse(0, 1).has_value());  // undefined
            CHECK(op.inverse(1, 0).value() == 1);
            CHECK(!op.inverse(1, 1).has_value());  // ambiguous
        }

        SECTION("with reduction_type") {
            auto b = GENERATE(0, 1);
            auto c = GENERATE(0, 1);
            auto d = LogicalOr::reduction_type(b);
            CHECK(op.inverse(op(d, c), c) == d);  // always defined
        }
    }

    SECTION(".operator(...)") {
        const LogicalOr op;

        CHECK(op(true, true));
        CHECK(op(false, true));
        CHECK(op(true, false));
        CHECK(!op(false, false));
        CHECK(op(LogicalOr::reduction_type(true), true));
        CHECK(op(LogicalOr::reduction_type(false), true));
        CHECK(op(LogicalOr::reduction_type(true), false));
        CHECK(!op(LogicalOr::reduction_type(false), false));

        CHECK(op.reduce(std::array{true, false, false}));
        CHECK(!op.reduce(std::array{false, false, false}));
        CHECK(op.reduce(std::array{false, false, false}, true));
        CHECK(!op.reduce(std::array{false, false, false}, false));
        CHECK(op.reduce(std::array<int, 0>{}, true));
        CHECK(!op.reduce(std::array<int, 0>{}, false));
    }

    SECTION(".reduce(...)") {
        const LogicalOr op;

        auto reduction = op.reduce(std::vector{0, 0, 1}, 0);
        CHECK(reduction);
        CHECK(op.inverse(reduction, 1).value() == 0);
    }

    SECTION(".result_bounds(...)") {
        const LogicalOr op;

        CHECK(op.result_bounds({0, 1, false}, {1, 1, false}) == ValuesInfo(1, 1, true));
        CHECK(op.result_bounds({-100, 100, true}, 1) == ValuesInfo(0, 1, true));
        CHECK(op.result_bounds({0, 10.5, false}, limit_type()) == ValuesInfo(0, 1, true));
    }
}

TEST_CASE("Maximum") {
    static_assert(Maximum<double>::associative);
    static_assert(Maximum<double>::commutative);
    static_assert(Maximum<double>::invertible);

    SECTION(".inverse(...)") {
        const Maximum<double> op;

        CHECK(op.inverse(10, 1) == 10);
        CHECK(!op.inverse(1, 1).has_value());
    }

    SECTION(".operator(...)") {
        const Maximum<double> op;

        CHECK(op(10, 1) == 10);
        CHECK(op(-100, 100) == 100);

        CHECK(op(Maximum<double>::reduction_type(10), 1) == 10);
        CHECK(op(Maximum<double>::reduction_type(-100), 100) == 100);
    }

    SECTION(".reduce(...)") {
        const Maximum<double> op;

        auto reduction = op.reduce(std::vector{10, 20, 30}, 25);
        CHECK(reduction == 30);
        CHECK(!op.inverse(reduction, 30).has_value());
    }

    SECTION(".result_bounds(...)") {
        const Maximum<double> op;

        CHECK(op.result_bounds({0, 5, true}, {1, 2, true}) == ValuesInfo(1, 5, true));
        CHECK(op.result_bounds({0, 5, true}, {1, 2, false}) == ValuesInfo(1, 5, false));

        CHECK(op.result_bounds({0, 5, true}, 5) == ValuesInfo(0, 5, true));
        CHECK(op.result_bounds({-1, 5, true}, 5) == ValuesInfo(-1, 5, true));
    }
}

TEST_CASE("Minimum") {
    static_assert(Minimum<double>::associative);
    static_assert(Minimum<double>::commutative);
    static_assert(Minimum<double>::invertible);

    SECTION(".inverse(...)") {
        const Minimum<double> op;

        CHECK(op.inverse(10, 11) == 10);
        CHECK(!op.inverse(1, 1).has_value());
    }

    SECTION(".operator(...)") {
        const Minimum<double> op;

        CHECK(op(10, 1) == 1);
        CHECK(op(-100, 100) == -100);

        CHECK(op(Minimum<double>::reduction_type(10), 1) == 1);
        CHECK(op(Minimum<double>::reduction_type(-100), 100) == -100);
    }

    SECTION(".reduce(...)") {
        const Minimum<double> op;

        auto reduction = op.reduce(std::vector{10, 20, 30}, 15);
        CHECK(reduction == 10);
        CHECK(!op.inverse(reduction, 10).has_value());
    }

    SECTION(".result_bounds(...)") {
        const Minimum<double> op;

        CHECK(op.result_bounds({0, 5, true}, {1, 2, true}) == ValuesInfo(0, 2, true));
        CHECK(op.result_bounds({0, 5, true}, {1, 2, false}) == ValuesInfo(0, 2, false));

        CHECK(op.result_bounds({0, 5, true}, 5) == ValuesInfo(0, 5, true));
        CHECK(op.result_bounds({-1, 5, true}, 5) == ValuesInfo(-1, 5, true));
    }
}

TEST_CASE("Multiply") {
    static_assert(Add<double>::associative);
    static_assert(Add<double>::commutative);
    static_assert(Add<double>::invertible);

    SECTION(".inverse(...)") {
        const Multiply<double> op;

        SECTION("with numeric types") {
            CHECK(op.inverse(op(0, 1), 1) == 0);
            CHECK(!op.inverse(op(1, 0), 0).has_value());
        }

        SECTION("with reduction_type") {
            auto b = GENERATE(-1, 0, 1);
            auto c = GENERATE(-1, 0, 1);
            auto d = Multiply<double>::reduction_type(b);

            CHECK(op.inverse(op(d, c), c) == d);
        }
    }

    SECTION(".operator(...)") {
        const Multiply<double> op;

        CHECK(op(2, 3) == 6);
        CHECK(op(Multiply<double>::reduction_type(2), 3) == 6);
    }

    SECTION(".reduce(...)") {
        const Multiply<double> op;

        CHECK(op.reduce(std::vector{3, 1, 2}) == 3 * 1 * 2);
        CHECK(op.reduce(std::vector{3, 1, 2}, 5) == 3 * 1 * 2 * 5);
        CHECK(op.reduce(std::vector<int>{}, -85) == -85);

        auto reduction = op.reduce(std::vector{0, 3, 2}, 5);
        CHECK(reduction == 0);
        CHECK(op.inverse(reduction, 0) == 3 * 2 * 5);
    }

    SECTION(".result_bounds(...)") {
        const Multiply<double> op;

        CHECK(op.result_bounds({0, 5, true}, {1, 2, true}) == ValuesInfo(0, 10, true));
        CHECK(op.result_bounds({0, 5, true}, {1, 2, false}) == ValuesInfo(0, 10, false));
        CHECK(op.result_bounds({1, 5, true}, {-1, 0, false}) == ValuesInfo(-5, 0, false));

        CHECK(op.result_bounds({0, 5, true}, 2) == ValuesInfo(0, 25, true));
        CHECK(op.result_bounds({-1, 5, true}, 2) == ValuesInfo(-5, 25, true));
    }
}

}  // namespace dwave::optimization::functional_

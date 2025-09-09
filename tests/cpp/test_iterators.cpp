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

#include <array>
#include <cstdint>
#include <numeric>
#include <vector>

#include "catch2/catch_template_test_macros.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "catch2/matchers/catch_matchers_all.hpp"
#include "dwave-optimization/iterators.hpp"

using Catch::Matchers::RangeEquals;

namespace dwave::optimization {

// note: we don't test bool here because it's a bit of an edge case
TEMPLATE_TEST_CASE("BufferIterator - templated", "",  //
                   float, double, std::int8_t, std::int16_t, std::int32_t, std::int64_t) {
    // Check that we can interpret buffers of each type as a double
    GIVEN("A buffer of the given type") {
        std::vector<TestType> buffer(10);
        std::iota(buffer.begin(), buffer.end(), 0);

        THEN("We can construct an iterator reading it as if it was doubles without specifying the "
             "buffer type at compile-time") {
            auto it = BufferIterator<double>(buffer.data());

            // the output of the iterator is double as requested
            static_assert(std::same_as<decltype(*it), double>);

            CHECK_THAT(std::ranges::subrange(it, it + buffer.size()), RangeEquals(buffer));
        }

        THEN("We can construct an iterator reading it as if it was doubles specifying the buffer "
             "type at compile-time") {
            auto it = BufferIterator<double, TestType>(buffer.data());

            // the output of the iterator is double as requested
            if constexpr (std::same_as<TestType, double>) {
                static_assert(std::same_as<decltype(*it), const double&>);
            } else {
                static_assert(std::same_as<decltype(*it), double>);
            }

            CHECK_THAT(std::ranges::subrange(it, it + buffer.size()), RangeEquals(buffer));
        }

        THEN("We can construct a non-const iterator reading as the same type") {
            auto it = BufferIterator<TestType, TestType, false>(buffer.data());

            // the output of the iterator is a non-const reference
            static_assert(std::same_as<decltype(*it), TestType&>);

            CHECK_THAT(std::ranges::subrange(it, it + buffer.size()), RangeEquals(buffer));

            // we can even mutate the underlying buffer!
            *it = 5;
            CHECK(buffer[0] == 5);
        }
    }

    GIVEN("A buffer of doubles") {
        std::vector<double> buffer(10);
        std::iota(buffer.begin(), buffer.end(), 0);

        THEN("We can construct an iterator reading it as the given type") {
            auto it = BufferIterator<TestType>(buffer.data());

            // the output of the iterator is TestType as requested
            static_assert(std::same_as<decltype(*it), TestType>);

            CHECK_THAT(std::ranges::subrange(it, it + buffer.size()), RangeEquals(buffer));
        }

        THEN("We can construct a strided 1D iterator equivalent to buffer[::2]") {
            const ssize_t size = 5;
            const ssize_t stride = 2 * sizeof(double);
            auto it = BufferIterator<TestType>(buffer.data(), 1, &size, &stride);

            CHECK_THAT(std::ranges::subrange(it, it + 5), RangeEquals({0, 2, 4, 6, 8}));
        }
    }
}

TEST_CASE("BufferIterator - bool") {
    GIVEN("An array of bools") {
        std::array<bool, 5> buffer{false, true, false, false, true};

        auto it = BufferIterator<double, bool>(buffer.data());

        CHECK_THAT(std::ranges::subrange(it, it + 5), RangeEquals({0, 1, 0, 0, 1}));
    }

    GIVEN("An array of doubles") {
        std::array<double, 10> buffer{0, 1, 2, 3, 0, 1, 2, 3, 0, 1};

        auto it = BufferIterator<bool, double>(buffer.data());

        CHECK_THAT(std::ranges::subrange(it, it + 10),
                   RangeEquals({false, true, true, true, false, true, true, true, false, true}));
    }
}

TEST_CASE("BufferIterator") {
    using ArrayIterator = BufferIterator<double, double, false>;
    using ConstArrayIterator = BufferIterator<double, double, true>;

    static_assert(std::is_nothrow_constructible<ArrayIterator>::value);
    static_assert(std::is_nothrow_constructible<ConstArrayIterator>::value);
    static_assert(std::is_nothrow_copy_constructible<ArrayIterator>::value);
    static_assert(std::is_nothrow_move_constructible<ArrayIterator>::value);
    static_assert(std::is_nothrow_copy_assignable<ArrayIterator>::value);
    static_assert(std::is_nothrow_move_assignable<ArrayIterator>::value);
    static_assert(std::is_nothrow_copy_constructible<ConstArrayIterator>::value);
    static_assert(std::is_nothrow_move_constructible<ConstArrayIterator>::value);
    static_assert(std::is_nothrow_copy_assignable<ConstArrayIterator>::value);
    static_assert(std::is_nothrow_move_assignable<ConstArrayIterator>::value);

    // The iterators are random access but not contiguous
    static_assert(std::random_access_iterator<ArrayIterator>);
    static_assert(std::random_access_iterator<ConstArrayIterator>);
    static_assert(!std::contiguous_iterator<ArrayIterator>);
    static_assert(!std::contiguous_iterator<ConstArrayIterator>);

    // Both iterators are input iterators
    static_assert(std::input_iterator<ArrayIterator>);
    static_assert(std::input_iterator<ConstArrayIterator>);

    // But only ArrayIterator is an output iterator.
    static_assert(std::output_iterator<ArrayIterator, double>);
    static_assert(!std::output_iterator<ConstArrayIterator, double>);

    GIVEN("A buffer encoding 2d array [[0, 1, 2], [3, 4, 5]]") {
        auto values = std::array<double, 6>{0, 1, 2, 3, 4, 5};
        auto shape = std::array<ssize_t, 2>{2, 3};
        // auto strides = std::array<ssize_t, 2>{24, 8};

        auto [strides, start] = GENERATE(std::pair(std::array<ssize_t, 2>{24, 8}, 0),
                                         std::pair(std::array<ssize_t, 2>{-24, 8}, 3));

        auto n = GENERATE(0, 3, 4, 5);

        AND_GIVEN("a = buffer.begin(), b = a + n") {
            const ConstArrayIterator a = ConstArrayIterator(values.data() + start, shape, strides);
            const ConstArrayIterator b = a + n;

            // semantic requirements
            THEN("a += n is equal to b") {
                ConstArrayIterator c = a;
                CHECK((c += n) == b);
            }
            THEN("(a + n) is equal to (a += n)") {
                ConstArrayIterator c = a + n;
                ConstArrayIterator d = a;
                CHECK(c == (d += n));
            }
            THEN("(a + n) is equal to (n + a)") { CHECK(a + n == n + a); }
            THEN("If (a + (n - 1)) is valid, then --b is equal to (a + (n - 1))") {
                ConstArrayIterator c = b;
                CHECK(--c == a + (n - 1));
            }
            THEN("(b += -n) and (b -= n) are both equal to a") {
                ConstArrayIterator c = b;
                auto d = b;
                CHECK((c += -n) == a);
                CHECK((d -= n) == a);
            }
            THEN("(b - n) is equal to (b -= n)") {
                ConstArrayIterator c = b;
                CHECK(b - n == (c -= n));
            }
            THEN("If b is dereferenceable, then a[n] is valid and is equal to *b") {
                CHECK(a[n] == *b);
            }
            THEN("bool(a <= b) is true") { CHECK(a <= b); }
        }
    }

    GIVEN("A buffer encoding 3d array [[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]") {
        auto values = std::array<double, 12>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        auto shape = std::array<ssize_t, 3>{2, 2, 3};
        auto strides = std::array<ssize_t, 3>{48, 24, 8};

        auto it = ConstArrayIterator(values.data(), shape, strides);
        auto end = it + values.size();

        THEN("Various iterator arithmetic operations are all self-consistent") {
            auto it2 = it;
            CHECK(++(++(++(++it2))) == it + 4);
            auto end2 = end;
            CHECK(--(--(--(--end2))) == end - 4);
            CHECK(((it + 1) + 3) == it + 4);
            CHECK(((it - 1005) + 1009) == it + 4);
            CHECK(((it + 1) + 13) == it + 14);
        }
    }

    GIVEN("A std::vector<double>{0, 1, 2, 3, 4, 5, 6, 7, 8}") {
        auto values = std::vector<double>{0, 1, 2, 3, 4, 5, 6, 7, 8};

        WHEN("We use a strided ArrayIterator to mutate every other value") {
            std::vector<ssize_t> shape{5};
            std::vector<ssize_t> strides{2 * sizeof(double)};
            auto it = ArrayIterator(values.data(), shape, strides);

            for (auto end = it + 5; it != end; ++it) {
                *it = -5;
            }

            THEN("The vector is edited") {
                CHECK_THAT(values, RangeEquals({-5, 1, -5, 3, -5, 5, -5, 7, -5}));
            }
        }

        WHEN("We make an ConstArrayIterator from values.data()") {
            auto it = ConstArrayIterator(values.data());

            THEN("It behaves like a contiguous iterator") {
                CHECK(*it == 0);
                ++it;
                CHECK(*it == 1);
            }

            THEN("We can do iterator arithmetic") {
                CHECK(*(it + 0) == 0);
                CHECK(*(it + 1) == 1);
                CHECK(*(it + 2) == 2);
                CHECK(*(it + 3) == 3);
                CHECK(*(it + 4) == 4);

                CHECK(ConstArrayIterator(values.data() + 4) - it == 4);
                CHECK(it - ConstArrayIterator(values.data() + 4) == -4);
            }
        }

        WHEN("We specify a shape and strides defining a subset of the values") {
            std::vector<ssize_t> shape{5};
            std::vector<ssize_t> strides{2 * sizeof(double)};
            auto it = ConstArrayIterator(values.data(), 1, shape.data(), strides.data());

            THEN("It behaves like a strided iterator") {
                CHECK(*it == 0);
                ++it;
                CHECK(*it == 2);
            }

            THEN("We can do iterator arithmetic") {
                CHECK(*(it + 0) == 0);
                CHECK(*(it + 1) == 2);
                CHECK(*(it + 2) == 4);
                CHECK(*(it + 3) == 6);
                CHECK(*(it + 4) == 8);
            }

            THEN("We can decrement an advanced iterator") {
                it += 4;
                CHECK(*(it--) == 8);
                CHECK(*(it--) == 6);
                CHECK(*(it--) == 4);
                CHECK(*(it--) == 2);
                CHECK(*(it--) == 0);
            }
        }

        WHEN("We specify a shape and strides for a 1D dynamic array") {
            std::vector<ssize_t> shape{-1, 2};
            std::vector<ssize_t> strides{4 * sizeof(double), sizeof(double)};

            auto it = ConstArrayIterator(values.data(), 2, shape.data(), strides.data());

            THEN("We can calculate the distance between two pointers in the strided array") {
                const auto end = it + 2;
                CHECK(end - it == 2);
            }

            THEN("We decrement an advanced iterator") {
                it += 4;
                CHECK(*(--it) == 5);
                CHECK(*(--it) == 4);
                CHECK(*(--it) == 1);
                CHECK(*(--it) == 0);
            }
        }

        THEN("We can construct another vector using reverse iterators") {
            auto copy = std::vector<double>();
            copy.assign(std::reverse_iterator(ConstArrayIterator(values.data()) + 6),
                        std::reverse_iterator(ConstArrayIterator(values.data())));
            CHECK_THAT(copy, RangeEquals({5, 4, 3, 2, 1, 0}));
        }
    }
}

}  // namespace dwave::optimization

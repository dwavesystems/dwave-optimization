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
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "catch2/benchmark/catch_benchmark_all.hpp"
#include "catch2/catch_template_test_macros.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "catch2/generators/catch_generators_random.hpp"
#include "catch2/matchers/catch_matchers_all.hpp"
#include "dwave-optimization/iterators.hpp"

using Catch::Generators::RandomIntegerGenerator;
using Catch::Matchers::RangeEquals;

namespace dwave::optimization {

TEMPLATE_PRODUCT_TEST_CASE("BufferIterator", "", BufferIterator,
                           ((const float, const void),                 //
                            (const double, const void),                //
                            (const bool, const void),                  //
                            (const std::int8_t, const void),           //
                            (const std::int16_t, const void),          //
                            (const std::int32_t, const void),          //
                            (const std::int64_t, const void),          //
                            (float, float),                            //
                            (double, double),                          //
                            (bool, bool),                              //
                            (std::int8_t, std::int8_t),                //
                            (std::int16_t, std::int16_t),              //
                            (std::int32_t, std::int32_t),              //
                            (std::int64_t, std::int64_t),              //
                            (const float, float),                      //
                            (const double, double),                    //
                            (const bool, bool),                        //
                            (const std::int8_t, std::int8_t),          //
                            (const std::int16_t, std::int16_t),        //
                            (const std::int32_t, std::int32_t),        //
                            (const std::int64_t, std::int64_t),        //
                            (const float, const float),                //
                            (const double, const double),              //
                            (const bool, const bool),                  //
                            (const std::int8_t, const std::int8_t),    //
                            (const std::int16_t, const std::int16_t),  //
                            (const std::int32_t, const std::int32_t),  //
                            (const std::int64_t, const std::int64_t))) {
    using To = TestType::value_type;

    static_assert(std::is_nothrow_constructible<TestType>::value);
    static_assert(std::is_nothrow_copy_constructible<TestType>::value);
    static_assert(std::is_nothrow_move_constructible<TestType>::value);
    static_assert(std::is_nothrow_copy_assignable<TestType>::value);
    static_assert(std::is_nothrow_move_assignable<TestType>::value);

    static_assert(std::random_access_iterator<TestType>);
    static_assert(std::random_access_iterator<TestType>);
    static_assert(not std::contiguous_iterator<TestType>);
    static_assert(not std::contiguous_iterator<TestType>);

    // BufferIterator is always an input iterator
    static_assert(std::input_iterator<TestType>);

    // But it's only sometimes an output iterator
    if constexpr (not std::is_const<To>::value) {
        static_assert(std::output_iterator<TestType, To>);
    } else {
        static_assert(not std::output_iterator<TestType, To>);
    }

    SECTION("BufferIterator<...>()") {
        auto it = TestType();

        CHECK(it == std::default_sentinel);
        CHECK(std::default_sentinel == it);
        CHECK_FALSE(it != std::default_sentinel);
        CHECK_FALSE(std::default_sentinel != it);

        CHECK(it - std::default_sentinel == 0);
        CHECK(std::default_sentinel - it == 0);

        CHECK(it == TestType());
        CHECK(it <= TestType());
        CHECK(it >= TestType());
        CHECK_FALSE(it != TestType());
        CHECK_FALSE(it < TestType());
        CHECK_FALSE(it > TestType());

        CHECK(it - TestType() == 0);
        CHECK(TestType() - it == 0);

        CHECK(it + 1 == it);  // does nothing

        CHECK_THAT(it.location(), RangeEquals(std::vector<ssize_t>{}));
    }
}

TEMPLATE_TEST_CASE("BufferIterator", "",
                   float,         //
                   double,        //
                   bool,          //
                   std::int8_t,   //
                   std::int16_t,  //
                   std::int32_t,  //
                   std::int64_t) {
    // Check that we can interpret buffers of each type as a double
    GIVEN("A buffer of the given type") {
        std::array<TestType, 10> buffer;
        std::iota(buffer.begin(), buffer.end(), 0);
        std::array<const ssize_t, 1> shape{10};
        std::array<const ssize_t, 1> strides{sizeof(TestType)};

        THEN("We can construct an iterator reading it as if it was doubles without specifying the "
             "buffer type at compile-time") {
            auto it = BufferIterator<const double, const void>(buffer.data(), shape, strides);

            // the output of the iterator is double as requested
            static_assert(std::same_as<decltype(*it), double>);

            CHECK_THAT(std::ranges::subrange(it, it + buffer.size()), RangeEquals(buffer));
        }

        THEN("We can construct an iterator reading it as if it was doubles specifying the buffer "
             "type at compile-time") {
            auto it = BufferIterator<const double, TestType>(buffer.data(), shape, strides);

            // the output of the iterator is double as requested
            if constexpr (std::same_as<TestType, double>) {
                static_assert(std::same_as<decltype(*it), const double&>);
            } else {
                static_assert(std::same_as<decltype(*it), double>);
            }

            CHECK_THAT(std::ranges::subrange(it, it + buffer.size()), RangeEquals(buffer));
        }

        THEN("We can construct a non-const iterator reading as the same type") {
            auto it = BufferIterator<TestType, TestType>(buffer.data(), shape, strides);

            // the output of the iterator is a non-const reference
            static_assert(std::same_as<decltype(*it), TestType&>);

            CHECK_THAT(std::ranges::subrange(it, it + buffer.size()), RangeEquals(buffer));

            // we can even mutate the underlying buffer!
            *it = 5;
            CHECK(buffer[0] == static_cast<TestType>(5));
        }
    }

    GIVEN("A buffer of doubles") {
        std::vector<double> buffer(10);
        std::iota(buffer.begin(), buffer.end(), 0);
        std::array<const ssize_t, 1> shape{10};
        std::array<const ssize_t, 1> strides{sizeof(double)};

        THEN("We can construct an iterator reading it as the given type") {
            auto it = BufferIterator<const TestType, const void>(buffer.data(), shape, strides);

            // the output of the iterator is TestType as requested
            static_assert(std::same_as<decltype(*it), TestType>);

            for (const double& val : buffer) {
                CHECK(*it == static_cast<TestType>(val));
                ++it;
            }
        }

        THEN("We can construct a strided 1D iterator equivalent to buffer[::2]") {
            const ssize_t size = 5;
            const ssize_t stride = 2 * sizeof(double);
            auto it = BufferIterator<const TestType, const void>(buffer.data(), 1, &size, &stride);

            for (const auto& val : {0, 2, 4, 6, 8}) {
                CHECK(*it == static_cast<TestType>(val));
                ++it;
            }
        }
    }
}

TEST_CASE("BufferIterator") {
    GIVEN("A buffer encoding 2d array [[0, 1, 2], [3, 4, 5]]") {
        auto values = std::array<double, 6>{0, 1, 2, 3, 4, 5};
        auto shape = std::array<ssize_t, 2>{2, 3};
        auto [strides, start] = GENERATE(std::pair(std::array<ssize_t, 2>{24, 8}, 0),
                                         std::pair(std::array<ssize_t, 2>{-24, 8}, 3));

        auto n = GENERATE(0, 3, 4, 5);

        AND_GIVEN("a = buffer.begin(), b = a + n") {
            const BufferIterator<const double, const double> a =
                    BufferIterator<const double, const double>(values.data() + start, shape,
                                                               strides);
            const BufferIterator<const double, const double> b = a + n;

            // semantic requirements
            THEN("a += n is equal to b") {
                BufferIterator<const double, const double> c = a;
                CHECK((c += n) == b);
            }
            THEN("(a + n) is equal to (a += n)") {
                BufferIterator<const double, const double> c = a + n;
                BufferIterator<const double, const double> d = a;
                CHECK(c == (d += n));
            }
            THEN("(a + n) is equal to (n + a)") { CHECK(a + n == n + a); }
            THEN("If (a + (n - 1)) is valid, then --b is equal to (a + (n - 1))") {
                BufferIterator<const double, const double> c = b;
                CHECK(--c == a + (n - 1));
            }
            THEN("(b += -n) and (b -= n) are both equal to a") {
                BufferIterator<const double, const double> c = b;
                auto d = b;
                CHECK((c += -n) == a);
                CHECK((d -= n) == a);
            }
            THEN("(b - n) is equal to (b -= n)") {
                BufferIterator<const double, const double> c = b;
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

        auto it = BufferIterator<const double, const double>(values.data(), shape, strides);
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

        THEN("The shaped iterator can be used with a sentinel") {
            CHECK(it != std::default_sentinel);
            CHECK(it + values.size() == std::default_sentinel);
            CHECK(std::default_sentinel - it == values.size());
            CHECK(it - std::default_sentinel == -(std::default_sentinel - it));
            CHECK_THAT(std::ranges::subrange(it, std::default_sentinel), RangeEquals(values));
        }

        THEN("We can do inplace multi-increments with various ranges") {
            it += {1, 0, 2};
            CHECK_THAT(it.location(), RangeEquals({1, 0, 2}));
            CHECK(*it == 8);

            it += std::array<signed char, 3>{-1, 1, -1};  // backwards is OK
            CHECK_THAT(it.location(), RangeEquals({0, 1, 1}));
            CHECK(*it == 4);

            it += std::vector<int>{2, -1, -1};  // end is OK
            CHECK_THAT(it.location(), RangeEquals({2, 0, 0}));
            CHECK(it == end);
        }

        THEN("We can do multi-increments with various  ranges") {
            // alas, cannot do initializer_list because C++ doesn't allow it
            CHECK(*(it + std::array{1, 0, 2}) == 8);
            CHECK(*(it + std::vector{0, 1, 1}) == 4);
            CHECK((it + std::array{2, 0, 0}) == end);
        }
    }

    GIVEN("A std::vector<double>{0, 1, 2, 3, 4, 5, 6, 7, 8}") {
        auto values = std::vector<double>{0, 1, 2, 3, 4, 5, 6, 7, 8};

        WHEN("We use a strided BufferIterator<double, double> to mutate every other value") {
            std::vector<ssize_t> shape{5};
            std::vector<ssize_t> strides{2 * sizeof(double)};
            auto it = BufferIterator<double, double>(values.data(), shape, strides);

            for (auto end = it + 5; it != end; ++it) {
                *it = -5;
            }

            THEN("The vector is edited") {
                CHECK_THAT(values, RangeEquals({-5, 1, -5, 3, -5, 5, -5, 7, -5}));
            }
        }

        WHEN("We specify a shape and strides defining a subset of the values") {
            std::vector<ssize_t> shape{5};
            std::vector<ssize_t> strides{2 * sizeof(double)};
            auto it = BufferIterator<const double, const double>(values.data(), 1, shape.data(),
                                                                 strides.data());

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

            THEN("We can construct another vector using reverse iterators") {
                auto end = it + 5;
                auto copy = std::vector<double>();
                copy.insert(copy.begin(), std::reverse_iterator(end), std::reverse_iterator(it));
                CHECK_THAT(copy, RangeEquals({8, 6, 4, 2, 0}));
            }
        }

        WHEN("We specify a shape and strides for a 1D dynamic array") {
            std::vector<ssize_t> shape{-1, 2};
            std::vector<ssize_t> strides{4 * sizeof(double), sizeof(double)};

            auto it = BufferIterator<const double, const double>(values.data(), 2, shape.data(),
                                                                 strides.data());

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
    }

    GIVEN("An empty buffer with shape (0,)") {
        std::array<double, 0> buffer;
        std::array<ssize_t, 3> shape{0};
        std::array<ssize_t, 3> strides{0};
        auto it = BufferIterator<double>(buffer.data(), shape, strides);
        CHECK(it == std::default_sentinel);
    }

    GIVEN("An empty buffer with shape (3, 0, 2)") {
        std::array<double, 0> buffer;
        std::array<ssize_t, 3> shape{3, 0, 2};
        std::array<ssize_t, 3> strides{0, 0, 0};
        auto it = BufferIterator<double>(buffer.data(), shape, strides);
        CHECK(it == std::default_sentinel);
    }

    GIVEN("A buffer of int32 each separated by 3 pad bytes") {
        std::vector<std::uint8_t> bytes(7 * 10);
        std::uint8_t* ptr = bytes.data();
        for (ssize_t i = 0; i < 10; ++i, ptr += 7) {
            *(reinterpret_cast<std::int32_t*>(ptr)) = i;
        }

        WHEN("We iterate through the buffer as a 1d array") {
            std::array<ssize_t, 1> shape{10};
            std::array<ssize_t, 1> strides{7};

            auto it = BufferIterator<const double, const std::int32_t>(
                    reinterpret_cast<std::int32_t*>(bytes.data()), shape, strides);

            CHECK_THAT(std::ranges::subrange(it, std::default_sentinel),
                       RangeEquals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
        }

        WHEN("We iterate through the buffer as a 1d strided array") {
            std::array<ssize_t, 1> shape{5};
            std::array<ssize_t, 1> strides{2 * 7};

            auto it = BufferIterator<const double, const std::int32_t>(
                    reinterpret_cast<std::int32_t*>(bytes.data()), shape, strides);

            CHECK_THAT(std::ranges::subrange(it, std::default_sentinel),
                       RangeEquals({0, 2, 4, 6, 8}));
        }
    }

    GIVEN("An unsorted buffer of doubles") {
        std::vector<double> buffer{2, 6, 7, 0, 8, 9, 5, 1, 4, 3};

        THEN("We can sort it as a 1d array using BufferIterator and std::sort()") {
            std::array<ssize_t, 1> shape{10};
            std::array<ssize_t, 1> strides{sizeof(double)};

            auto begin = BufferIterator<double, double>(buffer.data(), shape, strides);
            auto end = begin + 10;

            std::sort(begin, end);

            CHECK_THAT(buffer, RangeEquals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
        }

        THEN("We can reverse sort it as a 1d array using BufferIterator and std::sort()") {
            std::array<ssize_t, 1> shape{10};
            std::array<ssize_t, 1> strides{-static_cast<ssize_t>(sizeof(double))};

            auto begin = BufferIterator<double, double>(buffer.data() + 9, shape, strides);
            auto end = begin + 10;

            std::sort(begin, end);

            CHECK_THAT(buffer, RangeEquals({9, 8, 7, 6, 5, 4, 3, 2, 1, 0}));
        }

        THEN("We can interpret it as a 2d array and sort") {
            std::array<ssize_t, 2> shape{2, 5};
            std::array<ssize_t, 2> strides{-5 * static_cast<ssize_t>(sizeof(double)),
                                           sizeof(double)};

            auto begin = BufferIterator<double, double>(buffer.data() + 5, shape, strides);
            auto end = begin + 10;

            std::sort(begin, end);

            CHECK_THAT(buffer, RangeEquals({5, 6, 7, 8, 9, 0, 1, 2, 3, 4}));
        }
    }
}

}  // namespace dwave::optimization

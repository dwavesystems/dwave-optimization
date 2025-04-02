// Copyright 2023 D-Wave Systems Inc.
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

// #include <sstream>
#include <numeric>

#include "catch2/benchmark/catch_benchmark_all.hpp"
#include "catch2/catch_template_test_macros.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_all.hpp"

#include "dwave-optimization/array.hpp"
// #include "dwave-optimization/state.hpp"

#include "dwave-optimization/iterators.hpp"

using Catch::Matchers::RangeEquals;

namespace dwave::optimization {

TEST_CASE("BufferIterator vs ArrayIterator") {
    // for benchmarking we don't really care about the non-const case
    using ArrayIterator = Array::const_iterator;

    // get a large buffer of doubles
    std::vector<double> buffer(1000);
    std::iota(buffer.begin(), buffer.end(), 0);

    BENCHMARK("BufferIterator<double, void>, contiguous") {
        // for the moment let's cheat and not do the random access thing
        auto it = exp::BufferIterator<double>(buffer.data());
        double val = 0;
        for (size_t i = 0, stop = buffer.size(); i < stop; ++i, ++it) {
            val += *it;
        }
        return val;
    };

    BENCHMARK("BufferIterator<int, void>, contiguous") {
        // for the moment let's cheat and not do the random access thing
        auto it = exp::BufferIterator<int>(buffer.data());
        int val = 0;
        for (size_t i = 0, stop = buffer.size(); i < stop; ++i, ++it) {
            val += *it;
        }
        return val;
    };

    BENCHMARK("BufferIterator<double, double> contiguous") {
        // for the moment let's cheat and not do the random access thing
        auto it = exp::BufferIterator<double, double>(buffer.data());
        double val = 0;
        for (size_t i = 0, stop = buffer.size(); i < stop; ++i, ++it) {
            val += *it;
        }
        return val;
    };

    BENCHMARK("BufferIterator<int, double>, contiguous") {
        // for the moment let's cheat and not do the random access thing
        auto it = exp::BufferIterator<int, double>(buffer.data());
        int val = 0;
        for (size_t i = 0, stop = buffer.size(); i < stop; ++i, ++it) {
            val += *it;
        }
        return val;
    };

    BENCHMARK("ArrayIterator contiguous") {
        // for the moment let's cheat and not do the random access thing
        auto it = ArrayIterator(buffer.data());
        double val = 0;
        for (size_t i = 0, stop = buffer.size(); i < stop; ++i, ++it) {
            val += *it;
        }
        return val;
    };

    BENCHMARK("for-loop contiguous") {
        double val = 0;
        for (const double& v : buffer) {
            val += v;
        }
        return val;
    };

    BENCHMARK("for-loop (int) contiguous") {
        int val = 0;
        for (const int v : buffer) {
            val += v;
        }
        return val;
    };

    const ssize_t shape = buffer.size() / 2;
    const ssize_t stride = 2 * sizeof(double);

    BENCHMARK("BufferIterator<double, void>, strided") {
        // for the moment let's cheat and not do the random access thing
        auto it = exp::BufferIterator<double>(buffer.data(), 1, &shape, &stride);
        double val = 0;
        for (ssize_t i = 0; i < shape; ++i, ++it) {
            val += *it;
        }
        return val;
    };

    // BENCHMARK("BufferIterator<double, double>, strided") {
    //     // for the moment let's cheat and not do the random access thing
    //     auto it = exp::BufferIterator<double, double>(buffer.data(), 1, &shape, &stride);
    //     double val = 0;
    //     for (ssize_t i = 0; i < shape; ++i, ++it) {
    //         val += *it;
    //     }
    //     return val;
    // };

    BENCHMARK("ArrayIterator strided") {
        // for the moment let's cheat and not do the random access thing
        auto it = ArrayIterator(buffer.data(), 1, &shape, &stride);
        double val = 0;
        for (ssize_t i = 0; i < shape; ++i, ++it) {
            val += *it;
        }
        return val;
    };

    BENCHMARK("for-loop strided") {
        auto it = buffer.begin();
        double val = 0;
        for (size_t i = 0, stop = buffer.size() / 2; i < stop; ++i, ++it, ++it) {
            val += *it;
        }
        return val;
    };
}

TEMPLATE_TEST_CASE("BufferIterator", "",  //
                   float, double, std::int8_t, std::int16_t, std::int32_t, std::int64_t) {
    // Check that we can interpret buffers of each type as a double
    GIVEN("A buffer of the given type") {
        std::vector<TestType> buffer(10);
        std::iota(buffer.begin(), buffer.end(), 0);

        THEN("We can construct an iterator reading it as if it was doubles without specifying the "
             "buffer type at compile-time") {
            auto it = exp::BufferIterator<double>(buffer.data());

            // the output of the iterator is double as requested
            static_assert(std::same_as<decltype(*it), double>);

            CHECK_THAT(std::ranges::subrange(it, it + buffer.size()), RangeEquals(buffer));
        }

        THEN("We can construct an iterator reading it as if it was doubles specifying the buffer "
             "type at compile-time") {
            auto it = exp::BufferIterator<double, TestType>(buffer.data());

            // the output of the iterator is double as requested
            static_assert(std::same_as<decltype(*it), double>);

            CHECK_THAT(std::ranges::subrange(it, it + buffer.size()), RangeEquals(buffer));
        }

        THEN("We can construct a non-const iterator reading as the same type") {
            auto it = exp::BufferIterator<TestType, TestType, false>(buffer.data());

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
            auto it = exp::BufferIterator<TestType>(buffer.data());

            // the output of the iterator is TestType as requested
            static_assert(std::same_as<decltype(*it), TestType>);

            CHECK_THAT(std::ranges::subrange(it, it + buffer.size()), RangeEquals(buffer));
        }

        THEN("We can construct a strided 1D iterator equivalent to buffer[::2]") {
            const ssize_t size = 5;
            const ssize_t stride = 2 * sizeof(double);
            auto it = exp::BufferIterator<TestType>(buffer.data(), 1, &size, &stride);

            CHECK_THAT(std::ranges::subrange(it, it + 5), RangeEquals({0, 2, 4, 6, 8}));
        }
    }
}

}  // namespace dwave::optimization

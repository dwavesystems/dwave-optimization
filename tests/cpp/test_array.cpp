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

#include <sstream>

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_all.hpp"
#include "dwave-optimization/array.hpp"
#include "dwave-optimization/state.hpp"

using Catch::Matchers::RangeEquals;

namespace dwave::optimization {

TEST_CASE("Array::View") {
    static_assert(std::semiregular<Array::View>);

    static_assert(std::is_nothrow_copy_constructible<Array::View>::value);
    static_assert(std::is_nothrow_move_constructible<Array::View>::value);
    static_assert(std::is_nothrow_copy_assignable<Array::View>::value);
    static_assert(std::is_nothrow_move_assignable<Array::View>::value);

    static_assert(!std::ranges::contiguous_range<Array::View>);
    static_assert(std::ranges::random_access_range<Array::View>);
    static_assert(std::ranges::range<Array::View>);
    static_assert(std::ranges::sized_range<Array::View>);
    static_assert(std::ranges::input_range<Array::View>);

    // What we want is std::ranges::constant_range<...> but that's a c++23 feature,
    // so we make due. This test actually might fail if say we didn't retern a reference
    // but it's good enough for now
    static_assert(std::is_const_v<std::remove_reference_t<decltype(Array::View()[0])>>);

    // Because non-empty Views are so tied to Arrays, we do most of the
    // testing in the Array tests. But we can test the empty ones here
    SECTION("Empty View") {
        auto view = Array::View();

        CHECK(view.size() == 0);
        CHECK(view.begin() == view.end());
        CHECK(view.empty());
    }
}

TEST_CASE("Test SizeInfo") {
    CHECK(SizeInfo(0) == SizeInfo(0));
    CHECK(SizeInfo(0) == 0);
    CHECK(SizeInfo(5) == SizeInfo(5));
    CHECK(SizeInfo(5) == 5);
}

TEST_CASE("Slice") {
    SECTION("exceptions") {
        CHECK_THROWS_AS(Slice(0, 1, 0), std::invalid_argument);
        auto badslice = Slice();
        badslice.step = 0;
        CHECK_THROWS_AS(badslice.fit_at(10), std::invalid_argument);
        CHECK_THROWS_AS(Slice(0, 1, 1).fit_at(-10), std::invalid_argument);
    }

    SECTION("Slice(...)") {
        STATIC_REQUIRE(Slice(0) == Slice(0));
        STATIC_REQUIRE(Slice(5) == Slice(0, 5, 1));
        STATIC_REQUIRE(Slice(2, 5) == Slice(2, 5, 1));
    }

    SECTION("Slice::empty()") {
        STATIC_REQUIRE(Slice().empty());
        STATIC_REQUIRE(Slice(std::nullopt, std::nullopt, std::nullopt).empty());
        STATIC_REQUIRE(!Slice(1, std::nullopt, std::nullopt).empty());
        STATIC_REQUIRE(!Slice(std::nullopt, 1, std::nullopt).empty());
        STATIC_REQUIRE(Slice(std::nullopt, std::nullopt, 1).empty());
    }

    SECTION("Slice::fit()") {
        STATIC_REQUIRE(Slice(0, std::nullopt).fit(10) == Slice().fit(10));
        STATIC_REQUIRE(Slice(std::nullopt, std::nullopt, -1).fit(10) == Slice(9, -1, -1));
        STATIC_REQUIRE(Slice(-1).fit(10) == Slice(0, 9, 1));
        STATIC_REQUIRE(Slice(-3, -1).fit(10) == Slice(7, 9, 1));
        STATIC_REQUIRE(Slice(-11, -1).fit(10) == Slice(0, 9, 1));
        STATIC_REQUIRE(Slice(-12, -11).fit(10) == Slice(0, 0, 1));
        STATIC_REQUIRE(Slice(105, 107).fit(10) == Slice(10, 10, 1));
        STATIC_REQUIRE(Slice(5, 104).fit(10) == Slice(5, 10, 1));
        STATIC_REQUIRE(Slice(5, 3).fit(10) == Slice(5, 3, 1));  // start can come after stop
    }

    SECTION("Slice::size()") {
        STATIC_REQUIRE(Slice(3, 5).size() == 2);
        STATIC_REQUIRE(Slice(5, 3).size() == 0);
        STATIC_REQUIRE(Slice(3, 3, 2).size() == 0);
        STATIC_REQUIRE(Slice(3, 4, 2).size() == 1);
        STATIC_REQUIRE(Slice(3, 5, 2).size() == 1);
        STATIC_REQUIRE(Slice(3, 6, 2).size() == 2);
        STATIC_REQUIRE(Slice(3, 7, 2).size() == 2);
        STATIC_REQUIRE(Slice(3, 8, 2).size() == 3);
        STATIC_REQUIRE(Slice(3, 9, 2).size() == 3);
        STATIC_REQUIRE(Slice(3, 7, 100).size() == 1);

        STATIC_REQUIRE(Slice(3, 5, -1).size() == 0);
        STATIC_REQUIRE(Slice(5, 3, -1).size() == 2);

        constexpr ssize_t MAX = std::numeric_limits<ssize_t>::max();
        constexpr ssize_t MIN = std::numeric_limits<ssize_t>::min();

        CHECK(Slice().size() == MAX);
        CHECK(Slice(MIN, MAX).size() == static_cast<std::size_t>(MAX) - MIN);
        CHECK(Slice(MAX, MIN, -1).size() == static_cast<std::size_t>(MAX) - MIN);
    }

    SECTION("copy assignment") {
        Slice slice;
        Slice slice2(0, 1);
        slice = slice2;
        CHECK(slice == slice2);
    }

    SECTION("copy construction") {
        Slice slice(0, 10, -1);
        Slice slice2(std::move(slice));
        CHECK(slice2 == Slice(0, 10, -1));
        CHECK(slice == slice2);
    }

    SECTION("move assignment") {
        Slice slice;
        slice = Slice(0, 1);
        CHECK(slice == Slice(0, 1));
    }

    SECTION("move construction") {
        Slice slice(0, 10, -1);
        Slice slice2(std::move(slice));
        CHECK(slice2 == Slice(0, 10, -1));
    }

    SECTION("printing") {
        std::stringstream ss;
        ss << Slice(5);
        CHECK(ss.str() == "Slice(0, 5, 1)");
    }
}

TEST_CASE("Scalar") {
    auto state = State();

    // Represent a single double using a 0-dimensional
    class FloatZeroDimArray : public ArrayOutputMixin<Array> {
     public:
        explicit FloatZeroDimArray(double value) : ArrayOutputMixin({}), update_(0, value, value) {}

        double const* buff(const State&) const override { return &update_.value; }

        std::span<const Update> diff(const State&) const override { return std::span(&update_, 1); }

        bool integral() const override { return false; }
        double max() const override { return std::numeric_limits<double>::max(); }
        double min() const override { return std::numeric_limits<double>::lowest(); }

     private:
        // We use a trick where we just store the value in an Update
        Update update_;
    };

    WHEN("We instantiate a class representing a scalar inheriting from ArrayOutputMixin") {
        auto a = FloatZeroDimArray(5.5);

        THEN("It has all of the buffer protocol values we expect") {
            CHECK(a.format() == "d");
            CHECK(a.itemsize() == sizeof(double));

            CHECK(a.ndim() == 0);
            CHECK(a.shape().empty());
            CHECK(a.strides().empty());

            CHECK(a.size() == 1);

            CHECK(std::ranges::equal(a.view(state), std::vector{5.5}));
            CHECK(a.view(state).front() == 5.5);
            CHECK(a.view(state).back() == 5.5);

            // for contiguous buff() points to the value
            CHECK(*(a.buff(state)) == 5.5);
        }
    }

    class FloatScalar : public ScalarOutputMixin<Array> {
     public:
        explicit FloatScalar(double value) : ScalarOutputMixin<Array>(), update_(0, value, value) {}

        double const* buff(const State&) const override { return &update_.value; }

        std::span<const Update> diff(const State&) const override { return std::span(&update_, 1); }

        bool integral() const override { return false; }
        double max() const override { return std::numeric_limits<double>::max(); }
        double min() const override { return std::numeric_limits<double>::lowest(); }

     private:
        // We use a trick where we just store the value in an Update
        Update update_;
    };

    WHEN("We instantiate a class representing a scalar inheriting from "
         "ScalarOutputMixin<ArrayBase, double>") {
        auto a = FloatScalar(7.5);

        THEN("It has all of the buffer protocol values we expect") {
            CHECK(a.format() == "d");
            CHECK(a.itemsize() == sizeof(double));

            CHECK(a.size() == 1);
            CHECK(a.ndim() == 0);
            CHECK(a.shape().empty());
            CHECK(a.strides().empty());

            CHECK(a.size() == 1);

            CHECK(std::ranges::equal(a.view(state), std::vector{7.5}));

            // for contiguous buff() points to the value
            CHECK(*(a.buff(state)) == 7.5);
        }

        AND_WHEN("We create a second one") {
            auto b = FloatScalar(-123);

            THEN("Both share the same span data") {
                CHECK(a.shape().data() == b.shape().data());
                CHECK(a.shape(state).data() == b.shape(state).data());
                CHECK(a.size(state) == b.size(state));
                CHECK(a.strides().data() == b.strides().data());
            }
        }
    }
}

TEST_CASE("Dynamically Sized 1d Array") {
    auto state = State();

    class Vector : public ArrayOutputMixin<Array> {
     public:
        Vector() : ArrayOutputMixin<Array>({-1}) {}  // dynamically sized

        double const* buff(const State&) const override { return state_.data(); }

        using ArrayOutputMixin::size;  // for stateless overload
        ssize_t size(const State&) const noexcept override { return state_.size(); }

        std::span<const ssize_t> shape(const State&) const override { return shape_; }
        using ArrayOutputMixin::shape;  // for the stateless overload

        std::span<const Update> diff(const State&) const override { return {}; }

        bool integral() const override { return false; }
        double max() const override { return std::numeric_limits<double>::max(); }
        double min() const override { return std::numeric_limits<double>::lowest(); }

        // Normally this would be stored in the State, but for testing we just keep it here
        std::vector<double> state_ = {0, 1, 2, 3};
        std::vector<ssize_t> shape_ = {4};
    };

    WHEN("We create a vector-as-array") {
        auto v = Vector();

        THEN("It has the size/shape/etc we expect") {
            CHECK(v.format() == "d");
            CHECK(v.itemsize() == sizeof(double));

            CHECK(v.ndim() == 1);
            CHECK(std::ranges::equal(v.shape(), std::vector{-1}));
            CHECK(std::ranges::equal(v.shape(state), std::vector{v.size(state)}));
            CHECK(std::ranges::equal(v.strides(), std::vector{sizeof(double)}));

            CHECK(v.size() == -1);
            CHECK(v.size(state) == 4);
            CHECK(v.len(state) == 4 * sizeof(double));

            CHECK(std::ranges::equal(v.view(state), v.state_));

            // for contiguous view is the same as span(buff(), size())
            CHECK(std::ranges::equal(std::span(v.buff(state), v.size(state)), v.view(state)));
        }

        AND_WHEN("We adjust the state") {
            v.state_.emplace_back(5);
            v.shape_[0] += 1;

            THEN("It has the size/shape/etc we expect") {
                CHECK(v.format() == "d");
                CHECK(v.itemsize() == sizeof(double));

                CHECK(v.ndim() == 1);
                CHECK(std::ranges::equal(v.shape(), std::vector{-1}));
                CHECK(std::ranges::equal(v.shape(state), std::vector{5}));
                CHECK(std::ranges::equal(v.strides(), std::vector{sizeof(double)}));

                CHECK(v.size() == -1);
                CHECK(v.size(state) == 5);
                CHECK(v.len(state) == 5 * sizeof(double));

                CHECK(std::ranges::equal(v.view(state), v.state_));

                CHECK(v.view(state).front() == v.state_.front());
                CHECK(v.view(state).back() == v.state_.back());

                // for contiguous view is the same as span(buff(), size())
                CHECK(std::ranges::equal(std::span(v.buff(state), v.size(state)), v.view(state)));
            }
        }
    }
}

TEST_CASE("Dynamically Sized 2d Array") {
    auto state = State();

    class Array2d : public ArrayOutputMixin<Array> {
     public:
        Array2d() : ArrayOutputMixin({-1, 3}) {}

        double const* buff(const State&) const override { return state_.data(); }

        using ArrayOutputMixin::size;  // for stateless overload
        ssize_t size(const State&) const noexcept override { return shape_[0] * shape_[1]; }

        std::span<const ssize_t> shape(const State&) const override { return shape_; }
        using ArrayOutputMixin::shape;  // for the stateless overload

        std::span<const Update> diff(const State&) const override { return {}; }

        bool integral() const override { return false; }
        double max() const override { return std::numeric_limits<double>::max(); }
        double min() const override { return std::numeric_limits<double>::lowest(); }

        // Normally this would be stored in the State, but for testing we just keep it here
        std::vector<double> state_ = {};
        std::vector<ssize_t> shape_ = {0, 3};
    };

    WHEN("We create an empty 2d array") {
        auto arr = Array2d();

        THEN("It has the size/shape/etc we expect") {
            CHECK(arr.format() == "d");
            CHECK(arr.itemsize() == sizeof(double));

            CHECK(arr.ndim() == 2);
            CHECK(std::ranges::equal(arr.shape(), std::vector{-1, 3}));
            CHECK(std::ranges::equal(arr.shape(state), std::vector{0, 3}));
            CHECK(std::ranges::equal(arr.strides(),
                                     std::vector{3 * sizeof(double), sizeof(double)}));

            CHECK(arr.size() == -1);
            CHECK(arr.size(state) == 0);
            CHECK(arr.len(state) == 0);

            CHECK(std::ranges::equal(arr.view(state), arr.state_));

            // for contiguous view is the same as span(buff(), size())
            CHECK(std::ranges::equal(std::span(arr.buff(state), arr.size(state)), arr.view(state)));

            CHECK(arr.min() < 0);
            CHECK(arr.max() > 0);
        }

        AND_WHEN("We adjust the state") {
            arr.state_.emplace_back(5);
            arr.state_.emplace_back(7);
            arr.state_.emplace_back(14);
            arr.shape_[0] = 1;

            THEN("It has the size/shape/etc we expect") {
                CHECK(arr.format() == "d");
                CHECK(arr.itemsize() == sizeof(double));

                CHECK(arr.ndim() == 2);
                CHECK(std::ranges::equal(arr.shape(), std::vector{-1, 3}));
                CHECK(std::ranges::equal(arr.shape(state), std::vector{1, 3}));
                CHECK(std::ranges::equal(arr.strides(),
                                         std::vector{3 * sizeof(double), sizeof(double)}));

                CHECK(arr.size() == -1);
                CHECK(arr.size(state) == 3);
                CHECK(arr.len(state) == 3 * sizeof(double));

                CHECK(std::ranges::equal(arr.view(state), arr.state_));

                // for contiguous view is the same as span(buff(), size())
                CHECK(std::ranges::equal(std::span(arr.buff(state), arr.size(state)),
                                         arr.view(state)));
            }
        }
    }
}

TEST_CASE("Update") {
    static_assert(std::totally_ordered<Update>);

    SECTION("Update::removal()") {
        auto update = Update::removal(105, 56.5);
        CHECK(update.index == 105);
        CHECK(update.old == 56.5);
        CHECK(update.removed());
    }
    SECTION("Update::placement()") {
        auto update = Update::placement(105, 56.5);
        CHECK(update.index == 105);
        CHECK(update.value == 56.5);
        CHECK(update.placed());
    }
    GIVEN("A vector of updated, sometimes over the same index") {
        std::vector<Update> updates{{3, 1.0, 2.0},
                                    {0, 0.0, 1.0},
                                    {3, 0.0, 1.0},
                                    {105, 5.0, 6.0},
                                    Update::placement(2, 5.0),
                                    Update::removal(3, 6.0)};

        WHEN("We do a stable sort") {
            std::stable_sort(updates.begin(), updates.end());

            THEN("We get the order we expect") {
                CHECK(updates[0].index == 0);
                CHECK(updates[0].old == 0.0);
                CHECK(updates[0].value == 1.0);

                CHECK(updates[1].index == 2);
                CHECK(updates[1].placed());
                CHECK(updates[1].value == 5.0);

                // The threes have kept their relative order
                CHECK(updates[2].index == 3);
                CHECK(updates[2].old == 1.0);
                CHECK(updates[2].value == 2.0);
                CHECK(updates[3].index == 3);
                CHECK(updates[3].old == 0.0);
                CHECK(updates[3].value == 1.0);
                CHECK(updates[4].index == 3);
                CHECK(updates[4].old == 6.0);
                CHECK(updates[4].removed());

                CHECK(updates[5].index == 105);
                CHECK(updates[5].old == 5.0);
                CHECK(updates[5].value == 6.0);
            }
        }
    }
}

TEST_CASE("Test resulting_shape()") {
    SECTION("(256,256,3) x (3,) -> (256,256,3)") {
        CHECK(std::ranges::equal(broadcast_shape({256, 256, 3}, {3}), std::vector{256, 256, 3}));
    }
    SECTION("(8,1,6,1) x (7,1,5) -> (8,7,6,5)") {
        CHECK(std::ranges::equal(broadcast_shape({8, 1, 6, 1}, {7, 1, 5}),
                                 std::vector{8, 7, 6, 5}));
    }
    SECTION("(7,1,5) x (8,1,6,1) -> (8,7,6,5)") {
        CHECK(std::ranges::equal(broadcast_shape({7, 1, 5}, {8, 1, 6, 1}),
                                 std::vector{8, 7, 6, 5}));
    }
    SECTION("(5,4) x (1,) -> (5, 4)") {
        CHECK(std::ranges::equal(broadcast_shape({5, 4}, {1}), std::vector{5, 4}));
    }
    SECTION("(5,4) x (4,) -> (5, 4)") {
        CHECK(std::ranges::equal(broadcast_shape({5, 4}, {4}), std::vector{5, 4}));
    }
    SECTION("(15,3,5) x (15,1,5) -> (15,3,5)") {
        CHECK(std::ranges::equal(broadcast_shape({15, 3, 5}, {15, 1, 5}), std::vector{15, 3, 5}));
    }
    SECTION("(15,3,5) x (3,5) -> (15,3,5)") {
        CHECK(std::ranges::equal(broadcast_shape({15, 3, 5}, {3, 5}), std::vector{15, 3, 5}));
    }
    SECTION("(15,3,5) x (3,1) -> (15,3,5)") {
        CHECK(std::ranges::equal(broadcast_shape({15, 3, 5}, {3, 1}), std::vector{15, 3, 5}));
    }
    SECTION("(3,) x (4,)") {
        CHECK_THROWS_WITH(broadcast_shape({3}, {4}),
                          "operands could not be broadcast together with shapes (3,) (4,)");
    }
    SECTION("(2,1) x (8,4,3)") {
        CHECK_THROWS_WITH(broadcast_shape({2, 1}, {8, 4, 3}),
                          "operands could not be broadcast together with shapes (2, 1) (8, 4, 3)");
    }
}

TEST_CASE("Test deduplicate_diff") {
    GIVEN("An empty vector of updates") {
        std::vector<Update> updates;

        WHEN("We call deduplicate_diff") {
            deduplicate_diff(updates);
            THEN("deduplicate_diff() doesn't do anything") { CHECK(updates.size() == 0); }
        }

        THEN("Iteration over it with a view does nothing") {
            for ([[maybe_unused]] const Update& v : deduplicate_diff_view(updates)) {
                CHECK(false);  // shouldn't get here
            }
            for ([[maybe_unused]] const Update& v : deduplicate_diff_view(std::span(updates))) {
                CHECK(false);  // shouldn't get here
            }
        }
    }

    GIVEN("A list of updates with no duplicates or noop Updates") {
        std::vector<Update> updates = {Update(3, 0, 1), Update(5, 0, -1), Update(2, 1, 0)};

        WHEN("We call deduplicate_diff") {
            deduplicate_diff(updates);
            CHECK_THAT(updates, RangeEquals({Update(2, 1, 0), Update(3, 0, 1), Update(5, 0, -1)}));
        }

        THEN("We can use deduplicate_diff_view in a for-loop") {
            std::vector<Update> deduped;
            for (const auto& u : deduplicate_diff_view(updates)) {
                deduped.emplace_back(u);
            }
            CHECK_THAT(deduped, RangeEquals({Update(2, 1, 0), Update(3, 0, 1), Update(5, 0, -1)}));
        }
    }

    GIVEN("A list of updates with no duplicates and noop Updates") {
        std::vector<Update> updates = {Update(3, 0, 1), Update(5, -1, -1), Update(2, 1, 0)};

        WHEN("We call deduplicate_diff") {
            deduplicate_diff(updates);
            THEN("deduplicate_diff() sorts them and removes noops") {
                CHECK_THAT(updates, RangeEquals({Update(2, 1, 0), Update(3, 0, 1)}));
            }
        }
    }

    GIVEN("A list of update of one noop Updates") {
        std::vector<Update> updates = {Update(3, 1, 1)};

        WHEN("We call deduplicate_diff") {
            deduplicate_diff(updates);
            THEN("deduplicate_diff() removes the noop") { CHECK(updates.size() == 0); }
        }
    }

    GIVEN("A list of update of many noop Updates") {
        std::vector<Update> updates = {Update(3, 1, 1), Update(3, 1, 1), Update(4, -1, -1),
                                       Update(0, 57, 57)};

        WHEN("We call deduplicate_diff") {
            deduplicate_diff(updates);
            THEN("deduplicate_diff() removes all the noops") { CHECK(updates.size() == 0); }
        }
    }

    GIVEN("A list of updates with no duplicates and one noop at the beginning") {
        std::vector<Update> updates = {Update(3, 1, 5), Update(4, 1, 5), Update(6, -1, -2),
                                       Update(0, 57, 57)};

        WHEN("We call deduplicate_diff") {
            deduplicate_diff(updates);
            THEN("deduplicate_diff() removes the noop") {
                CHECK(std::ranges::equal(
                        updates,
                        std::vector<Update>{Update(3, 1, 5), Update(4, 1, 5), Update(6, -1, -2)}));
            }
        }
    }

    GIVEN("A list of updates with no duplicates and one noop at the end") {
        std::vector<Update> updates = {Update(3, 1, 5), Update(4, 1, 5), Update(6, -1, -2),
                                       Update(8, 57, 57)};

        WHEN("We call deduplicate_diff") {
            deduplicate_diff(updates);
            THEN("deduplicate_diff() removes the noop") {
                CHECK(std::ranges::equal(
                        updates,
                        std::vector<Update>{Update(3, 1, 5), Update(4, 1, 5), Update(6, -1, -2)}));
            }
        }
    }

    GIVEN("A list of updates with duplicates") {
        std::vector<Update> updates = {Update(3, 1, 5),   Update(3, 5, -3),  Update(6, -1, -2),
                                       Update(6, -2, 57), Update(3, -3, -4), Update(2, 0, 1)};

        WHEN("We call deduplicate_diff") {
            deduplicate_diff(updates);
            THEN("deduplicate_diff() removes the noop") {
                CHECK(std::ranges::equal(
                        updates,
                        std::vector<Update>{Update(2, 0, 1), Update(3, 1, -4), Update(6, -1, 57)}));
            }
        }
    }

    GIVEN("A list of updates that grows and shrinks") {
        std::vector<Update> updates = {Update(3, 1, 5), Update::placement(5, 7),
                                       Update::placement(6, 7), Update(5, 7, 4),
                                       Update::removal(6, 7)};

        WHEN("We call deduplicate_diff") {
            deduplicate_diff(updates);
            THEN("deduplicate_diff() trims the diff properly") {
                CHECK(std::ranges::equal(
                        updates, std::vector<Update>{Update(3, 1, 5), Update::placement(5, 4)}));
            }
        }
    }
}

// Adapted from NumPy
// https://github.com/numpy/numpy/blob/d02611a/numpy/lib/tests/test_index_tricks.py#L29
TEST_CASE("Test ravel_multi_index()") {
    CHECK(ravel_multi_index({1, 0}, {2, 2}) == 2);
    CHECK(ravel_multi_index({2, 66}, {17, 94}) == 254);
    CHECK(ravel_multi_index({2, 66}, {-1, 94}) == 254);  // dynamic
    CHECK(ravel_multi_index({2, 1, 4}, {4, 3, 6}) == (2 * 3 + 1) * 6 + 4);
    CHECK(ravel_multi_index({3, 1, 4, 1}, {6, 7, 8, 9}) == 1621);
    CHECK(ravel_multi_index({3, 1, 4, 1}, {-1, 7, 8, 9}) == 1621);  // dynamic
}
TEST_CASE("Test unravel_index()") {
    CHECK_THAT(unravel_index(2, {2, 2}), RangeEquals({1, 0}));
    CHECK_THAT(unravel_index(254, {17, 94}), RangeEquals({2, 66}));
    CHECK_THAT(unravel_index(4, {-1, 2}), RangeEquals({2, 0}));  // dynamic
    CHECK_THAT(unravel_index((2 * 3 + 1) * 6 + 4, {4, 3, 6}), RangeEquals({2, 1, 4}));
    CHECK_THAT(unravel_index(1621, {6, 7, 8, 9}), RangeEquals({3, 1, 4, 1}));
}

}  // namespace dwave::optimization

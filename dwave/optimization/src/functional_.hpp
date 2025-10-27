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

/// These are intended to replace the current public functional.hpp but need
/// more thought. So for now we keep them private.
/// These classes are obviously heavily inspired by NumPy's universal functions
/// (ufunc). With the key difference that we wish to do operations like removing
/// a value from the output of a reduction.
/// Each class exposes a `result_type` and a `reduction_type`. The requirement
/// for the `reduction_type` is that it must be abe to be statically cast to
/// `result_type`. The `reduction_type` may include other information about
/// the reduction to allow for easier inversion.

#pragma once

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/functional.hpp"
#include "dwave-optimization/typing.hpp"

namespace dwave::optimization::functional_ {

/// For a function that would otherwise accept a counter `n`, the `limit_type`
/// is an empty class used to indicate that we wish to know the value of that
/// function as `n` approaches infinity.
struct limit_type {};

/// A mixin class using the curiously recurring template pattern (CRTP) to
/// provide some methods to all of our classes.
template <typename BinaryFunction>
struct BinaryFunctionMixin {
    // Dev note: once this approach is a bit more mature, we should implement
    // some concepts to enforce the structure of the BinaryFunction class.

    /// Return the result of applying a binary function to the input `range`.
    ///
    /// Roughly equivalent to
    ///     op = BinaryFunction();
    ///     typename BinaryFunction::reduction_type lhs = *initial;
    ///     for (const auto& rhs : range) {
    ///         lhs = op(std::move(lhs), rhs);
    ///     }
    ///     return lhs;
    ///
    /// If `initial` is not provided then the `range` must not be empty.
    template <std::ranges::range Range, DType U>
    static auto reduce(Range&& range, std::optional<U> initial) {
        // We require that initial be provided or the range not be empty
        assert(initial.has_value() or std::ranges::begin(range) != std::ranges::end(range));

        using reduction_type = BinaryFunction::reduction_type;  // our return type

        // Get the binary operation we'll be doing.
        const auto op = BinaryFunction();

        // Ok, let's get calculating the reduction
        auto it = std::ranges::begin(range);

        // If there isn't an initial value, draw it from the range
        reduction_type lhs = initial.has_value() ? *initial : *(it++);

        // Complete the reduction
        for (const auto end = std::ranges::end(range); it != end; ++it) {
            lhs = op(std::move(lhs), *it);
        }

        // Return, establishing the return type to be reduction_type
        return lhs;
    }
    template <std::ranges::range Range>
    static auto reduce(Range&& range) {
        return reduce(std::forward<Range>(range), std::optional<double>());
    }
    template <std::ranges::range Range, DType T>
    static auto reduce(Range&& range, T initial) {
        return reduce(std::forward<Range>(range), std::optional<T>(initial));
    }
};

/// Add two elements.
template <DType T>
struct Add : BinaryFunctionMixin<Add<double>> {
    /// The type we ultimately wish to interpret the result of the binary
    /// operation as.
    using result_type = T;

    /// The type returned by a `reduce()` operation. Must be statically castable
    /// and equality comparable to `result_type`, but may include other
    /// information useful for reductions.
    using reduction_type = result_type;

    /// @brief Return the sum of `lhs` and `rhs`.
    ///
    /// @details
    /// The `lhs` may be `result_type` or `reduction_type`. Some binary
    /// operations might support additional types.
    ///
    /// The `rhs` may be `result_type`. Some binary operations might support
    /// additional types.
    result_type operator()(const DType auto& lhs, const DType auto& rhs) const noexcept {
        return lhs + rhs;
    }

    /// @brief Invert an addition.
    ///
    /// @details
    /// This method is the inverse of the `operator()` method. For example if
    ///     a == op(b, c)
    /// Then
    ///     b == op.inverse(a, c)
    ///
    /// The `lhs` may be `result_type` or `reduction_type`. Some binary
    /// operations might support additional types.
    ///
    /// The `rhs` may be `result_type`. Some binary operations might support
    /// additional types.
    ///
    /// The inverse operation is allowed to fail, in which case the return value
    /// will not contain any elements. This is useful for handling annihilators
    /// for instance.
    static std::optional<result_type> inverse(const DType auto& lhs,
                                              const DType auto& rhs) noexcept {
        // dev note: we could check whether the operation results in `inf` or
        // `nan` and in that case return std::nullopt. Needs more thought.
        return lhs - rhs;
    }

    /// For `ValuesInfo result_bounds(ValuesInfo lhs, ValuesInfo rhs)` return
    /// the bounds/integrality that would result by applying the operation to
    /// elements with the given `lhs` and `rhs` bounds.
    ///
    /// For `ValuesInfo result_bounds(ValuesInfo bounds, ssize_t n)`
    /// Equivalent to the result of
    ///     auto lhs = bounds;
    ///     for (ssize_t i = 1; i < n; ++i) {
    ///         lhs = result_bounds(std::move(lhs), bounds);
    ///     }
    ///     return lhs;
    /// `n` must be positive.
    ///
    /// For `ValuesInfo result_bounds(ValuesInfo bounds, limit_type)` take the
    /// limit as `n` approaches infinity.
    static ValuesInfo result_bounds(ValuesInfo lhs, ValuesInfo rhs) noexcept {
        return ValuesInfo(lhs.min + rhs.min, lhs.max + rhs.max, lhs.integral and rhs.integral);
    }
    static ValuesInfo result_bounds(ValuesInfo bounds, ssize_t n) noexcept {
        // developer note: we have numeric issues here because these might result
        // in `inf`. However, this is a special case of a larger problem around
        // numeric issues so for now I am going to ignore it.
        assert(n > 0 and "n must be positive");
        return ValuesInfo(bounds.min * n, bounds.max * n, bounds.integral);
    }
    static ValuesInfo result_bounds(ValuesInfo bounds, limit_type) noexcept {
        // dev note: we currently don't want infs for the bounds, so we use
        // ::max() and ::lowest(). We might change that behavior in ValuesInfo
        // in the future in which case this code will need to change.
        auto take_limit = [](auto bound) -> decltype(bound) {
            if (bound < 0) return std::numeric_limits<decltype(bound)>::lowest();
            if (bound > 0) return std::numeric_limits<decltype(bound)>::max();
            return 0;
        };
        return ValuesInfo(take_limit(bounds.min), take_limit(bounds.max), bounds.integral);
    }

    static constexpr bool associative = true;
    static constexpr bool commutative = true;
    static constexpr bool invertible = true;
};

/// The logical and of two elements.
struct LogicalAnd : BinaryFunctionMixin<LogicalAnd> {
    /// @copydoc Add::result_type
    using result_type = bool;

    /// @copydoc Add::reduction_type
    class reduction_type {
     public:
        reduction_type() = delete;
        reduction_type(result_type value) noexcept : num_falsy_(value == 0) {}
        bool operator==(const reduction_type& rhs) const { return num_falsy_ == rhs.num_falsy_; }
        explicit operator result_type() const noexcept { return num_falsy_ == 0; }

     private:
        friend LogicalAnd;
        ssize_t num_falsy_;
    };

    /// @brief Return the logical and of `lhs` and `rhs`.
    /// @copydetails Add::operator()
    result_type operator()(const DType auto& lhs, const DType auto& rhs) const noexcept {
        return lhs and rhs;
    }
    reduction_type operator()(reduction_type lhs, const DType auto& rhs) const noexcept {
        if (rhs == 0) lhs.num_falsy_ += 1;
        return lhs;
    }

    /// @brief Revert a logical and.
    /// @copydetails Add::inverse()
    static std::optional<result_type> inverse(const DType auto& lhs,
                                              const DType auto& rhs) noexcept {
        if (rhs) return static_cast<result_type>(lhs);
        return {};  // ambiguous or undefined
    }
    static std::optional<reduction_type> inverse(reduction_type lhs,
                                                 const DType auto& rhs) noexcept {
        if (rhs == 0) lhs.num_falsy_ -= 1;
        return lhs;
    }

    /// @copydoc Add::result_bounds
    static ValuesInfo result_bounds(ValuesInfo lhs, ValuesInfo rhs) {
        if (lhs.min == 0 and lhs.max == 0) return ValuesInfo(0, 0, true);
        if (rhs.min == 0 and rhs.max == 0) return ValuesInfo(0, 0, true);
        return ValuesInfo(0, 1, true);
    }
    static ValuesInfo result_bounds(ValuesInfo bounds, ssize_t) {
        if (bounds.min == 0 and bounds.max == 0) return ValuesInfo(0, 0, true);
        return ValuesInfo(0, 1, true);
    }
    static ValuesInfo result_bounds(ValuesInfo bounds, limit_type) {
        return result_bounds(bounds, 1);
    }

    static constexpr bool associative = true;
    static constexpr bool commutative = true;
    static constexpr bool invertible = true;  // sort of anyway
};

/// The logical or of two elements.
struct LogicalOr : BinaryFunctionMixin<LogicalOr> {
    /// @copydoc Add::result_type
    using result_type = bool;

    /// @copydoc Add::reduction_type
    class reduction_type {
     public:
        reduction_type() = delete;
        reduction_type(result_type value) noexcept : num_truthy_(value != 0) {}
        bool operator==(const reduction_type& rhs) const { return num_truthy_ == rhs.num_truthy_; }
        explicit operator result_type() const noexcept { return num_truthy_ > 0; }

     private:
        friend LogicalOr;
        ssize_t num_truthy_;
    };

    /// @brief Return the logical or of `lhs` and `rhs`.
    /// @copydetails Add::operator()
    result_type operator()(const DType auto& lhs, const DType auto& rhs) const noexcept {
        return lhs or rhs;
    }
    reduction_type operator()(reduction_type lhs, const DType auto& rhs) const noexcept {
        lhs.num_truthy_ += (rhs != 0);
        return lhs;
    }

    /// @brief Revert a logical or.
    /// @copydetails Add::inverse()
    static std::optional<result_type> inverse(const DType auto& lhs, const DType auto& rhs) {
        if (not rhs) return static_cast<result_type>(lhs);
        return {};  // ambiguous or undefined
    }
    static std::optional<reduction_type> inverse(reduction_type lhs,
                                                 const DType auto& rhs) noexcept {
        lhs.num_truthy_ -= (rhs != 0);
        return lhs;
    }

    /// @copydoc Add::result_bounds
    static ValuesInfo result_bounds(ValuesInfo lhs, ValuesInfo rhs) {
        if (lhs.min == 1 and lhs.max == 1) return ValuesInfo(1, 1, true);
        if (rhs.min == 1 and rhs.max == 1) return ValuesInfo(1, 1, true);
        return ValuesInfo(0, 1, true);
    }
    static ValuesInfo result_bounds(ValuesInfo bounds, ssize_t) {
        if (bounds.min == 1 and bounds.max == 1) return ValuesInfo(1, 1, true);
        return ValuesInfo(0, 1, true);
    }
    static ValuesInfo result_bounds(ValuesInfo bounds, limit_type) {
        return result_bounds(bounds, 1);
    }

    static constexpr bool associative = true;
    static constexpr bool commutative = true;
    static constexpr bool invertible = true;  // sort of anyway
};

/// The max of two elements.
template <DType T>
struct Maximum : BinaryFunctionMixin<Maximum<T>> {
    /// @copydoc Add::result_type
    using result_type = T;

    /// @copydoc Add::reduction_type
    using reduction_type = result_type;

    /// @brief Return the max of `lhs` and `rhs`.
    /// @copydetails Add::operator()
    result_type operator()(const DType auto& lhs, const DType auto& rhs) const noexcept {
        return std::max<result_type>(lhs, rhs);
    }

    /// @brief Revert a maximum.
    /// @copydetails Add::inverse()
    static std::optional<result_type> inverse(const DType auto& lhs,
                                              const DType auto& rhs) noexcept {
        if (lhs > rhs) return lhs;  // We're removing a value smaller than our current max
        return {};                  // Otherwise its ambiguous or undefined
    }

    /// @copydoc Add::result_bounds
    static ValuesInfo result_bounds(ValuesInfo lhs, ValuesInfo rhs) {
        if (lhs.min >= rhs.max) return lhs;  // it'll always be lhs
        if (lhs.max <= rhs.min) return rhs;  // it'll always be rhs

        return ValuesInfo(std::max(lhs.min, rhs.min),  //
                          std::max(lhs.max, rhs.max),  //
                          lhs.integral and rhs.integral);
    }
    static ValuesInfo result_bounds(ValuesInfo bounds, ssize_t) noexcept { return bounds; }
    static ValuesInfo result_bounds(ValuesInfo bounds, limit_type) noexcept { return bounds; }

    static constexpr bool associative = true;
    static constexpr bool commutative = true;
    static constexpr bool invertible = true;  // sort of anyway
};

/// The min of two elements.
template <DType T>
struct Minimum : BinaryFunctionMixin<Minimum<T>> {
    /// @copydoc Add::result_type
    using result_type = T;

    /// @copydoc Add::reduction_type
    using reduction_type = result_type;

    /// @brief Return the min of `lhs` and `rhs`.
    /// @copydetails Add::operator()
    result_type operator()(const DType auto& lhs, const DType auto& rhs) const noexcept {
        return std::min<result_type>(lhs, rhs);
    }

    /// @brief Revert a minimum.
    /// @copydetails Add::inverse()
    static std::optional<result_type> inverse(const DType auto& lhs,
                                              const DType auto& rhs) noexcept {
        if (lhs < rhs) return lhs;  // We're removing a value greater than our current min
        return {};                  // Otherwise its ambiguous or undefined
    }

    /// @copydoc Add::result_bounds
    static ValuesInfo result_bounds(ValuesInfo lhs, ValuesInfo rhs) {
        if (lhs.min >= rhs.max) return rhs;  // it'll always be rhs
        if (lhs.max <= rhs.min) return lhs;  // it'll always be lhs

        return ValuesInfo(std::min(lhs.min, rhs.min),  //
                          std::min(lhs.max, rhs.max),  //
                          lhs.integral and rhs.integral);
    }
    static ValuesInfo result_bounds(ValuesInfo bounds, ssize_t) { return bounds; }
    static ValuesInfo result_bounds(ValuesInfo bounds, limit_type) { return bounds; }

    static constexpr bool associative = true;
    static constexpr bool commutative = true;
    static constexpr bool invertible = true;  // sort of anyway
};

/// The product of two elements.
template <DType T>
struct Multiply : BinaryFunctionMixin<Multiply<T>> {
    /// @copydoc Add::result_type
    using result_type = T;

    /// @copydoc Add::reduction_type
    struct reduction_type {
     public:
        reduction_type() = delete;
        reduction_type(result_type value) noexcept
                : nonzero_(value ? value : 1), num_zero_(value == 0) {}
        bool operator==(const reduction_type& rhs) const {
            return nonzero_ == rhs.nonzero_ and num_zero_ == rhs.num_zero_;
        }
        bool operator==(result_type rhs) const { return static_cast<result_type>(*this) == rhs; }
        explicit operator result_type() const noexcept { return num_zero_ ? 0 : nonzero_; }

     private:
        friend Multiply;

        result_type nonzero_;
        ssize_t num_zero_;
    };

    /// @brief Return the product of `lhs` and `rhs`.
    /// @copydetails Add::operator()
    result_type operator()(const DType auto& lhs, const DType auto& rhs) const noexcept {
        return lhs * rhs;
    }
    reduction_type operator()(reduction_type lhs, const DType auto& rhs) const noexcept {
        if (rhs == 0) {
            lhs.num_zero_ += 1;
        } else {
            lhs.nonzero_ *= rhs;
        }
        return lhs;
    }

    /// @brief Revert a multiply.
    /// @copydetails Add::inverse()
    static std::optional<result_type> inverse(const DType auto& lhs,
                                              const DType auto& rhs) noexcept {
        if (not rhs) return {};  // cannot divide by zero
        return lhs / rhs;
    }
    static std::optional<reduction_type> inverse(reduction_type lhs,
                                                 const DType auto& rhs) noexcept {
        if (rhs == 0) {
            lhs.num_zero_ -= 1;
        } else {
            lhs.nonzero_ /= rhs;
        }
        return lhs;
    }

    /// @copydoc Add::result_bounds
    static ValuesInfo result_bounds(ValuesInfo lhs, ValuesInfo rhs) {
        const auto [min, max] = std::minmax({
                lhs.min * rhs.min,
                lhs.min * rhs.max,
                lhs.max * rhs.min,
                lhs.max * rhs.max,
        });

        return ValuesInfo(min, max, lhs.integral and rhs.integral);
    }
    static ValuesInfo result_bounds(ValuesInfo bounds, ssize_t n) {
        assert(n > 0 and "n must be positive");

        double low = bounds.min;
        double high = bounds.max;

        // A bunch of cases we need to handle
        std::vector<double> candidates{std::pow(low, n), std::pow(high, n)};
        if (n > 1) {
            candidates.emplace_back(low * std::pow(high, n - 1));
            candidates.emplace_back(high * std::pow(low, n - 1));
        }

        return ValuesInfo(std::ranges::min(candidates), std::ranges::max(candidates),
                          bounds.integral);
    }
    static ValuesInfo result_bounds(ValuesInfo bounds, limit_type) {
        // dev note: we currently don't want infs for the bounds, so we use
        // ::max() and ::lowest(). We might change that behavior in ValuesInfo
        // in the future in which case this code will need to change.
        auto take_limit = [](auto bound) -> decltype(bound) {
            if (bound < 0) return std::numeric_limits<decltype(bound)>::lowest();
            if (bound > 0) return std::numeric_limits<decltype(bound)>::max();
            return 0;
        };
        return ValuesInfo(take_limit(bounds.min), take_limit(bounds.max), bounds.integral);
    }

    static constexpr bool associative = true;
    static constexpr bool commutative = true;
    static constexpr bool invertible = true;
};

// If/when we replace ::functional with the above classes, we won't need this
// mapping. But for now we need a way to map the public std/::functional names
// to our private funcs.
template <class BinaryOp>
struct std_to_ufunc {};
template <>
struct std_to_ufunc<functional::max<double>> {
    using type = Maximum<double>;
};
template <>
struct std_to_ufunc<functional::min<double>> {
    using type = Minimum<double>;
};
template <>
struct std_to_ufunc<std::logical_and<double>> {
    using type = LogicalAnd;
};
template <>
struct std_to_ufunc<std::logical_or<double>> {
    using type = LogicalOr;
};
template <>
struct std_to_ufunc<std::multiplies<double>> {
    using type = Multiply<double>;
};
template <>
struct std_to_ufunc<std::plus<double>> {
    using type = Add<double>;
};

}  // namespace dwave::optimization::functional_

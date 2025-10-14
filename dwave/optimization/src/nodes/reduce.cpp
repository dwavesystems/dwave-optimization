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

#include "dwave-optimization/nodes/reduce.hpp"

#include <cmath>
#include <concepts>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <tuple>
#include <type_traits>

#include "_state.hpp"

namespace dwave::optimization {

// developer note: These might someday evolve into a dwave-optimization version
// of NumPy's ufunc concept. But for now let's keep them behind the compilation
// barrier so we can get a bit of experience.
// A few notes:
// - For the same reason that our BufferIterators assume that the node using them
//   knows what type it is, we likewise assume that we know at compile-time what
//   result type we want. For now it's always double

struct limit_type {};

template <DType result_type>
struct add {
    class reduction_type {
     public:
        reduction_type() = delete;
        reduction_type(result_type value) noexcept : value_(value) {}
        bool operator==(const reduction_type& rhs) const { return this->value_ == rhs.value_; }
        result_type value() const noexcept { return value_; }

     private:
        friend add;
        result_type value_;
    };

    constexpr result_type operator()(const DType auto& lhs, const DType auto& rhs) {
        return lhs + rhs;
    }
    reduction_type operator()(reduction_type lhs, const DType auto& rhs) {
        lhs.value_ += rhs;
        return lhs;
    }

    std::optional<result_type> inverse(const DType auto& lhs, const DType auto& rhs) {
        // dev note: we could check whether we have any infs and return nullopt in that
        // case. Needs though/testing.
        return lhs - rhs;
    }
    std::optional<reduction_type> inverse(reduction_type lhs, const DType auto& rhs) {
        lhs.value_ -= rhs;
        return lhs;
    }

    // If `range` is empty then `initial` must be provided.
    template <std::ranges::range Range, DType T>
    reduction_type reduce(const Range&& range, std::optional<T> initial) {
        if (!initial.has_value()) {
            auto begin = std::ranges::begin(range);
            const auto end = std::ranges::end(range);
            assert(begin != end && "initial must be provided for an empty range");
            std::optional<T> init = *begin;
            return reduce(std::ranges::subrange(++begin, end), std::move(init));
        }

        // Unfortunately as of C++20 there is not a std::ranges::accumulate so in order
        // to support sentinels etc we have to do this "by hand".
        auto lhs = reduction_type(initial.value());
        for (const auto& rhs : range) {
            lhs = (*this)(std::move(lhs), rhs);
        }
        return lhs;
    }

    /// Given a lhs and rhs where we know the min/max/integrality. Return the
    /// resulting min/max/integrality. `n` defines the number of times it is
    /// applied (e.g., in a accumulation or reduction).
    ValuesInfo result_bounds(ValuesInfo lhs, ValuesInfo rhs) {
        return ValuesInfo(lhs.min + rhs.min, lhs.max + rhs.max, lhs.integral && rhs.integral);
    }

    /// Equivalent to the result of
    ///     auto lhs = bounds;
    ///     for (ssize_t i = 1; i < n; ++i) {
    ///         lhs = results_bounds(std::move(lhs), bounds);
    ///     }
    ///     return lhs;
    /// `n` must be positive.
    ValuesInfo result_bounds(ValuesInfo bounds, ssize_t n) const {
        // developer note: This might return `inf` in some cases. For now I think
        // that's OK, because the alterative (replacing inf with numeric_limits::max())
        // seems ad hoc and pretty ill-defined. And the node itself might return `inf`s.
        assert(n > 0 && "n must be positive");
        return ValuesInfo(bounds.min * n, bounds.max * n, bounds.integral);
    }
    ValuesInfo result_bounds(ValuesInfo bounds, limit_type) const {
        // dev note: we currently don't want infs for the bounds
        return ValuesInfo(std::numeric_limits<double>::lowest(),  //
                          std::numeric_limits<double>::max(),     //
                          bounds.integral);
    }

    static constexpr bool associative = true;
    static constexpr bool commutative = true;
    static constexpr bool invertible = true;
};
struct all {
    using result_type = bool;

    class reduction_type {
     public:
        reduction_type() = delete;
        reduction_type(result_type value) noexcept : num_falsy_(value == 0) {}
        bool operator==(const reduction_type& rhs) const { return num_falsy_ == rhs.num_falsy_; }
        result_type value() const noexcept { return num_falsy_ == 0; }

     private:
        friend all;
        ssize_t num_falsy_;
    };

    constexpr result_type operator()(const DType auto& lhs, const DType auto& rhs) {
        return lhs and rhs;
    }
    reduction_type operator()(reduction_type lhs, const DType auto& rhs) {
        if (rhs == 0) lhs.num_falsy_ += 1;
        return lhs;
    }


    std::optional<result_type> inverse(const DType auto& lhs, const DType auto& rhs) {
        assert(false && "not yet implemeted");
        return 0;
    }
    static std::optional<reduction_type> inverse(reduction_type lhs, const DType auto& rhs) {
        if (rhs == 0) lhs.num_falsy_ -= 1;
        return lhs;
    }

    // If `range` is empty then `initial` must be provided.
    template <std::ranges::range Range, DType T>
    reduction_type reduce(const Range&& range, std::optional<T> initial) {
        if (!initial.has_value()) {
            auto begin = std::ranges::begin(range);
            const auto end = std::ranges::end(range);
            assert(begin != end && "initial must be provided for an empty range");
            std::optional<T> init = *begin;
            return reduce(std::ranges::subrange(++begin, end), std::move(init));
        }

        // Unfortunately as of C++20 there is not a std::ranges::accumulate so in order
        // to support sentinels etc we have to do this "by hand".
        auto lhs = reduction_type(initial.value());
        for (const auto& rhs : range) {
            lhs = (*this)(std::move(lhs), rhs);
        }
        return lhs;
    }

    /// Given a lhs and rhs where we know the min/max/integrality. Return the
    /// resulting min/max/integrality. `n` defines the number of times it is
    /// applied (e.g., in a accumulation or reduction).
    ValuesInfo result_bounds(ValuesInfo lhs, ValuesInfo rhs) {
        return ValuesInfo(0, 1, true);
    }

    /// Equivalent to the result of
    ///     auto lhs = bounds;
    ///     for (ssize_t i = 1; i < n; ++i) {
    ///         lhs = results_bounds(std::move(lhs), bounds);
    ///     }
    ///     return lhs;
    /// `n` must be positive.
    ValuesInfo result_bounds(ValuesInfo bounds, ssize_t n) const {
        // Dev note: do we want to consider all falsy predecessors?
        assert(n > 0 && "n must be positive");
        return ValuesInfo(0, 1, true);
    }
    ValuesInfo result_bounds(ValuesInfo bounds, limit_type) const {
        return ValuesInfo(0, 1, true);
    }

    static constexpr bool associative = true;
    static constexpr bool commutative = true;
    static constexpr bool invertible = true;  // sort of anyway
};
struct any {
    using result_type = bool;

    class reduction_type {
     public:
        reduction_type() = delete;
        reduction_type(result_type value) noexcept : num_truthy_(value != 0) {}
        bool operator==(const reduction_type& rhs) const { return num_truthy_ == rhs.num_truthy_; }
        result_type value() const noexcept { return num_truthy_ > 0; }
     private:
        friend any;
        ssize_t num_truthy_;
    };

    constexpr result_type operator()(const DType auto& lhs, const DType auto& rhs) {
        return lhs or rhs;
    }
    reduction_type operator()(reduction_type lhs, const DType auto& rhs) {
        lhs.num_truthy_ += (rhs != 0);
        return lhs;
    }


    std::optional<result_type> inverse(const DType auto& lhs, const DType auto& rhs) {
        assert(false && "not yet implemeted");
        return 0;
    }
    static std::optional<reduction_type> inverse(reduction_type lhs, const DType auto& rhs) {
        lhs.num_truthy_ -= (rhs != 0);
        return lhs;
    }

    // If `range` is empty then `initial` must be provided.
    template <std::ranges::range Range, DType T>
    reduction_type reduce(const Range&& range, std::optional<T> initial) {
        if (!initial.has_value()) {
            auto begin = std::ranges::begin(range);
            const auto end = std::ranges::end(range);
            assert(begin != end && "initial must be provided for an empty range");
            std::optional<T> init = *begin;
            return reduce(std::ranges::subrange(++begin, end), std::move(init));
        }

        // Unfortunately as of C++20 there is not a std::ranges::accumulate so in order
        // to support sentinels etc we have to do this "by hand".
        auto lhs = reduction_type(initial.value());
        for (const auto& rhs : range) {
            lhs = (*this)(std::move(lhs), rhs);
        }
        return lhs;
    }

    /// Given a lhs and rhs where we know the min/max/integrality. Return the
    /// resulting min/max/integrality. `n` defines the number of times it is
    /// applied (e.g., in a accumulation or reduction).
    ValuesInfo result_bounds(ValuesInfo lhs, ValuesInfo rhs) { return ValuesInfo(0, 1, true); }

    /// Equivalent to the result of
    ///     auto lhs = bounds;
    ///     for (ssize_t i = 1; i < n; ++i) {
    ///         lhs = results_bounds(std::move(lhs), bounds);
    ///     }
    ///     return lhs;
    /// `n` must be positive.
    ValuesInfo result_bounds(ValuesInfo bounds, ssize_t n) const {
        // Dev note: do we want to consider all falsy predecessors?
        assert(n > 0 && "n must be positive");
        return ValuesInfo(0, 1, true);
    }
    ValuesInfo result_bounds(ValuesInfo bounds, limit_type) const { return ValuesInfo(0, 1, true); }

    static constexpr bool associative = true;
    static constexpr bool commutative = true;
    static constexpr bool invertible = true;  // sort of anyway
};
template <DType result_type>
struct max {
    // TODO: consider replacing with using double and the requirement becomes explicitly castable
    class reduction_type {
     public:
        reduction_type() = delete;
        reduction_type(result_type value) noexcept : value_(value) {}
        bool operator==(const reduction_type& rhs) const { return value_ == rhs.value_; }
        result_type value() const noexcept { return value_; }

     private:
        friend max;
        result_type value_;
    };

    constexpr result_type operator()(const DType auto& lhs, const DType auto& rhs) {
        assert(false);
    }
    reduction_type operator()(reduction_type lhs, const DType auto& rhs) { 
        lhs.value_ = std::max(lhs.value_, rhs);
        return lhs;
    }

    std::optional<result_type> inverse(const DType auto& lhs, const DType auto& rhs) {
        assert(false);
    }
    static std::optional<reduction_type> inverse(reduction_type lhs, const DType auto& rhs) {
        if (lhs.value_ > rhs) return lhs;  // We're removing a value smaller than our current max
        return {};  // Otherwise we failed! We cannot invert
    }

    // If `range` is empty then `initial` must be provided.
    template <std::ranges::range Range, DType T>
    reduction_type reduce(const Range&& range, std::optional<T> initial) {
        if (!initial.has_value()) {
            auto begin = std::ranges::begin(range);
            const auto end = std::ranges::end(range);
            assert(begin != end && "initial must be provided for an empty range");
            std::optional<T> init = *begin;
            return reduce(std::ranges::subrange(++begin, end), std::move(init));
        }

        // Unfortunately as of C++20 there is not a std::ranges::accumulate so in order
        // to support sentinels etc we have to do this "by hand".
        auto lhs = reduction_type(initial.value());
        for (const auto& rhs : range) {
            lhs = (*this)(std::move(lhs), rhs);
        }
        return lhs;
    }

    /// Given a lhs and rhs where we know the min/max/integrality. Return the
    /// resulting min/max/integrality. `n` defines the number of times it is
    /// applied (e.g., in a accumulation or reduction).
    ValuesInfo result_bounds(ValuesInfo lhs, ValuesInfo rhs) {
        return ValuesInfo(std::max(lhs.min, rhs.min),  //
                          std::max(lhs.max, rhs.max),  //
                          lhs.integral and rhs.integral);
    }

    /// Equivalent to the result of
    ///     auto lhs = bounds;
    ///     for (ssize_t i = 1; i < n; ++i) {
    ///         lhs = results_bounds(std::move(lhs), bounds);
    ///     }
    ///     return lhs;
    /// `n` must be positive.
    ValuesInfo result_bounds(ValuesInfo bounds, ssize_t) const { return bounds; }
    ValuesInfo result_bounds(ValuesInfo bounds, limit_type) const { return bounds; }

    static constexpr bool associative = true;
    static constexpr bool commutative = true;
    static constexpr bool invertible = true;  // sort of anyway
};
template <DType result_type>
struct min {
    class reduction_type {
     public:
        reduction_type() = delete;
        reduction_type(result_type value) noexcept : value_(value) {}
        bool operator==(const reduction_type& rhs) const { return value_ == rhs.value_; }
        result_type value() const noexcept { return value_; }

     private:
        friend min;
        result_type value_;
    };

    constexpr result_type operator()(const DType auto& lhs, const DType auto& rhs) {
        assert(false);
    }
    reduction_type operator()(reduction_type lhs, const DType auto& rhs) { 
        lhs.value_ = std::min(lhs.value_, rhs);
        return lhs;
    }

    std::optional<result_type> inverse(const DType auto& lhs, const DType auto& rhs) {
        assert(false);
    }
    static std::optional<reduction_type> inverse(reduction_type lhs, const DType auto& rhs) {
        if (lhs.value_ < rhs) return lhs;  // We're removing a value greater than our current min
        return {};  // Otherwise we failed! We cannot invert
    }

    // If `range` is empty then `initial` must be provided.
    template <std::ranges::range Range, DType T>
    reduction_type reduce(const Range&& range, std::optional<T> initial) {
        if (!initial.has_value()) {
            auto begin = std::ranges::begin(range);
            const auto end = std::ranges::end(range);
            assert(begin != end && "initial must be provided for an empty range");
            std::optional<T> init = *begin;
            return reduce(std::ranges::subrange(++begin, end), std::move(init));
        }

        // Unfortunately as of C++20 there is not a std::ranges::accumulate so in order
        // to support sentinels etc we have to do this "by hand".
        auto lhs = reduction_type(initial.value());
        for (const auto& rhs : range) {
            lhs = (*this)(std::move(lhs), rhs);
        }
        return lhs;
    }

    /// Given a lhs and rhs where we know the min/max/integrality. Return the
    /// resulting min/max/integrality. `n` defines the number of times it is
    /// applied (e.g., in a accumulation or reduction).
    ValuesInfo result_bounds(ValuesInfo lhs, ValuesInfo rhs) {
        return ValuesInfo(std::min(lhs.min, rhs.min),  //
                          std::min(lhs.max, rhs.max),  //
                          lhs.integral and rhs.integral);
    }

    /// Equivalent to the result of
    ///     auto lhs = bounds;
    ///     for (ssize_t i = 1; i < n; ++i) {
    ///         lhs = results_bounds(std::move(lhs), bounds);
    ///     }
    ///     return lhs;
    /// `n` must be positive.
    ValuesInfo result_bounds(ValuesInfo bounds, ssize_t) const { return bounds; }
    ValuesInfo result_bounds(ValuesInfo bounds, limit_type) const { return bounds; }

    static constexpr bool associative = true;
    static constexpr bool commutative = true;
    static constexpr bool invertible = true;  // sort of anyway
};
template <DType result_type>
struct prod {
    struct reduction_type {
     public:
        reduction_type() = delete;

        reduction_type(result_type value) noexcept
                : nonzero_(value ? value : 0), num_zero_(value == 0) {}

        bool operator==(const reduction_type& rhs) const {
            return this->nonzero_ == rhs.nonzero_ && this->num_zero_ == rhs.num_zero_;
        }

        result_type value() const noexcept { return num_zero_ ? 0 : nonzero_; }

     private:
        friend prod;

        result_type nonzero_;
        ssize_t num_zero_;
    };

    constexpr result_type operator()(const DType auto& lhs, const DType auto& rhs) {
        assert(false);
    }
    reduction_type operator()(reduction_type lhs, const DType auto& rhs) {
        if (rhs == 0) {
            lhs.num_zero_ += 1;
        } else {
            lhs.nonzero_ *= rhs;
        }
        return lhs;
    }

    std::optional<result_type> inverse(const DType auto& lhs, const DType auto& rhs) {
        assert(false);
    }
    std::optional<reduction_type> inverse(reduction_type lhs, const DType auto& rhs) {
        if (rhs == 0) {
            lhs.num_zero_ -= 1;
        } else {
            lhs.nonzero_ /= rhs;
        }
        return lhs;
    }

    // If `range` is empty then `initial` must be provided.
    template <std::ranges::range Range, DType T>
    reduction_type reduce(const Range&& range, std::optional<T> initial) {
        if (!initial.has_value()) {
            auto begin = std::ranges::begin(range);
            const auto end = std::ranges::end(range);
            assert(begin != end && "initial must be provided for an empty range");
            std::optional<T> init = *begin;
            return reduce(std::ranges::subrange(++begin, end), std::move(init));
        }

        // Unfortunately as of C++20 there is not a std::ranges::accumulate so in order
        // to support sentinels etc we have to do this "by hand".
        auto lhs = reduction_type(initial.value());
        for (const auto& rhs : range) {
            lhs = (*this)(std::move(lhs), rhs);
        }
        return lhs;
    }

    /// Given a lhs and rhs where we know the min/max/integrality. Return the
    /// resulting min/max/integrality. `n` defines the number of times it is
    /// applied (e.g., in a accumulation or reduction).
    ValuesInfo result_bounds(ValuesInfo lhs, ValuesInfo rhs) {
        const auto [min, max] = std::minmax({
                lhs.min * rhs.min,
                lhs.min * rhs.max,
                lhs.max * rhs.min,
                lhs.max * rhs.max,
        });

        return ValuesInfo(min, max, lhs.integral and rhs.integral);
    }

    /// Equivalent to the result of
    ///     auto lhs = bounds;
    ///     for (ssize_t i = 1; i < n; ++i) {
    ///         lhs = results_bounds(std::move(lhs), bounds);
    ///     }
    ///     return lhs;
    /// `n` must be positive.
    ValuesInfo result_bounds(ValuesInfo bounds, ssize_t n) const {
        assert(n > 0 && "n must be positive");

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
    ValuesInfo result_bounds(ValuesInfo bounds, limit_type) const {
        assert(false && "not yet implemeted");
        return ValuesInfo(0, 0, false);
    }

    static constexpr bool associative = true;
    static constexpr bool commutative = true;
    static constexpr bool invertible = true;
};

template <class BinaryOp>
struct std_to_ufunc {};
template <>
struct std_to_ufunc<functional::max<double>> {
    using type = max<double>;
};
template <>
struct std_to_ufunc<functional::min<double>> {
    using type = min<double>;
};
template <>
struct std_to_ufunc<std::logical_and<double>> {
    using type = all;
};
template <>
struct std_to_ufunc<std::logical_or<double>> {
    using type = any;
};
template <>
struct std_to_ufunc<std::multiplies<double>> {
    using type = prod<double>;
};
template <>
struct std_to_ufunc<std::plus<double>> {
    using type = add<double>;
};

template <class BinaryOp>
class ReduceNodeData : public NodeStateData {
 public:
    using ufunc_type = std_to_ufunc<BinaryOp>::type;

    // Pull this into the namespace for convenience.
    using reduction_type = ufunc_type::reduction_type;

    enum class ReductionFlag {
        invalid,    // no longer holds the correct value, the whole reduction must be recalculated
        unchanged,  // no change since the last propagation
        updated,    // has been updated and holds the correct value
    };

    ReduceNodeData(reduction_type reduction) : ReduceNodeData(std::vector{reduction}) {}

    ReduceNodeData(std::vector<reduction_type>&& reductions): reductions_(std::move(reductions)), flags_(reductions_.size(), ReductionFlag::unchanged) {
        buffer_.reserve(reductions_.size());
        for (const auto& reduction : reductions_) buffer_.emplace_back(reduction.value());
    }

    ReduceNodeData(std::vector<reduction_type>&& reductions, std::span<const ssize_t> shape) : ReduceNodeData(std::move(reductions)) {
        shape_info_ = shape_info_type(reductions_.size(), shape);
    }

    // Add a value to the reduction at the given `index`.
    // `index` can be one-past-the-end in which case a new reduction
    // is appended.
    void add_to_reduction(ssize_t index, double value) {
        static_assert(decltype(ufunc)::associative);  // Otherwise this function doesn't make sense

        assert(not std::isnan(value));  // Not trying to add a placement
        assert(0 <= index);   // Index must be non-negative
        assert(static_cast<std::size_t>(index) <= reductions_.size());  // one-past-end is OK
        assert(reductions_.size() == reductions_.size());

        if (static_cast<std::size_t>(index) == reductions_.size()) {
            return append_reduction(value);
        }

        update_reduction_(index, ufunc(reductions_[index], value));
    }

    void append_reduction(reduction_type reduction) {
        reductions_.emplace_back(reduction);
        flags_.emplace_back(ReductionFlag::updated);
    }

    const double* buff() const { return buffer_.data(); }

    void commit() {
        // flip any flags back to unchanged
        // Should we do this as part of prepare_diff?
        flags_.clear();
        flags_.resize(reductions_.size(), ReductionFlag::unchanged);

        // Clear the diffs
        diff_.clear();
        reductions_diff_.clear();

        // Update the shape_info to match the current state
        if (shape_info_) shape_info_->previous_size = buffer_.size();
    }

    std::span<const Update> diff() const { return diff_; }

    void pop_reduction() {
        assert(reductions_.size() > 0);

        const ssize_t index = reductions_.size() - 1;

        reductions_diff_.emplace_back(index, reductions_[index]);
        reductions_.pop_back();
        flags_.pop_back();
    }

    void prepare_diff(std::function<reduction_type(ssize_t)> reduce) {
        while (buffer_.size() > reductions_.size()) {
            diff_.emplace_back(Update::removal(buffer_.size() - 1, buffer_.back()));
            buffer_.pop_back();
        }

        for (ssize_t index = 0, size = buffer_.size(); index < size; ++index) {
            double value;

            switch (flags_[index]) {
                case ReductionFlag::updated:
                    value = reductions_[index].value();
                    if (buffer_[index] != value) {
                        diff_.emplace_back(index, buffer_[index], value);
                        buffer_[index] = value;
                    }
                    break;
                case ReductionFlag::invalid:
                    reductions_[index] = reduce(index);
                    value = reductions_[index].value();
                    if (buffer_[index] != value) {
                        diff_.emplace_back(index, buffer_[index], value);
                        buffer_[index] = value;
                    }
                    break;
                case ReductionFlag::unchanged:
                    // Nothing to do!
                    break;
            }
        }

        while (buffer_.size() < reductions_.size()) {
            const ssize_t index = buffer_.size();
            switch (flags_[index]) {
                case ReductionFlag::updated:
                    buffer_.emplace_back(reductions_[index].value());
                    diff_.emplace_back(Update::placement(index, buffer_[index]));
                    break;
                case ReductionFlag::unchanged:
                    assert(false);
                case ReductionFlag::invalid:
                    assert(false);
            }
        }

        if (shape_info_) {
            shape_info_->shape[0] = buffer_.size() / shape_info_->size_divisor;
        }
    }

    void remove_from_reduction(ssize_t index, double value) {
        // Otherwise this function implementation doesn't make sense
        static_assert(decltype(ufunc)::associative);
        static_assert(decltype(ufunc)::invertible);

        assert(not std::isnan(value));  // Not trying to remove a removal
        assert(0 <= index && static_cast<std::size_t>(index) < buffer_.size());

        // Then try to remove the value
        auto inverse = ufunc.inverse(reductions_[index], value);
        if (not inverse.has_value()) {
            // We broke the reduction, so mark it as invalid and return
            flags_[index] = ReductionFlag::invalid;
            return;
        }

        update_reduction_(index, std::move(inverse.value()));
    }

    void revert() {
        // flip any flags back to unchanged
        // Should we do this as part of prepare_diff?
        flags_.clear();
        flags_.resize(reductions_.size(), ReductionFlag::unchanged);

        // These should be unique, but might as well do it in reverse just in case
        for (const auto& [index, old_reduction] : reductions_diff_ | std::views::reverse) {
            assert(0 <= index && static_cast<std::size_t>(index) <= reductions_.size());

            if (static_cast<std::size_t>(index) == reductions_.size()) {
                reductions_.emplace_back(old_reduction);
            } else {
                reductions_[index] = old_reduction;
            }
        }
        reductions_diff_.clear();

        // likewise these should be unique
        for (const auto& [index, old_value, _] : diff_ | std::views::reverse) {
            assert(0 <= index && static_cast<std::size_t>(index) <= buffer_.size());
            if (static_cast<std::size_t>(index) == buffer_.size()) {
                buffer_.emplace_back(old_value);
            } else {
                buffer_[index] = old_value;
            }
            
        }
        diff_.clear();

        if (shape_info_) {
            if (static_cast<std::size_t>(shape_info_->previous_size) < buffer_.size()) {
                assert(false && "not yet implemeted");
            }

            shape_info_->previous_size = buffer_.size();
            shape_info_->shape[0] = buffer_.size() / shape_info_->size_divisor;
        }
    }

    std::span<const ssize_t> shape() const {
        assert(shape_info_);
        return shape_info_->shape;
    }

    ssize_t size() const {
        return buffer_.size();
    }

    ssize_t size_diff() const {
        if (!shape_info_) return 0;
        return static_cast<ssize_t>(buffer_.size()) - shape_info_->previous_size;
    }

    void update_reduction(ssize_t index, double from, double to) {
        // Otherwise this function implementation doesn't make sense
        static_assert(decltype(ufunc)::associative);
        static_assert(decltype(ufunc)::invertible);

        // Should always be true if we've implemented things correctly
        assert(flags_.size() == reductions_.size());

        // Some input checking
        assert(not std::isnan(from));  // Not trying to remove a removal
        assert(not std::isnan(to));  // Not trying to add a placement
        assert(0 <= index && static_cast<std::size_t>(index) < reductions_.size());

        // If we've already marked this location as invalid, then nothing to do
        if (flags_[index] == ReductionFlag::invalid) return;

        // Likewise if from == to then no point doing any recalculations
        if (from == to) return;

        // Get the reduction we're potentially updating, as a copy
        auto reduction = reductions_[index];

        // Add the value to the reduction first
        reduction = ufunc(std::move(reduction), to);

        // Then try to remove the value
        auto inverse = ufunc.inverse(std::move(reduction), from);
        if (not inverse.has_value()) {
            flags_[index] = ReductionFlag::invalid;
            return;
        }
        reduction = std::move(inverse.value());

        return update_reduction_(index, std::move(reduction));
    }

 private:
    void update_reduction_(ssize_t index, reduction_type reduction) {
        assert(flags_.size() == reductions_.size());
        assert(0 <= index && static_cast<std::size_t>(index) < reductions_.size());

        // No change so don't save anything
        if (reductions_[index] == reduction) return;

        switch (flags_[index]) {
            case ReductionFlag::invalid:
                assert(false && "not yet implemeted");
            case ReductionFlag::unchanged:
                // We haven't previously made any changes to this location, so we
                // save the old value, mark ourselves as updated, and then
                // 
                reductions_diff_.emplace_back(index, reductions_[index]);
                reductions_[index] = std::move(reduction);
                flags_[index] = ReductionFlag::updated;
                break;
            case ReductionFlag::updated:
                // We've already updated this index once, so need to save the old value
                reductions_[index] = std::move(reduction);
                break;
        };
    }

    ufunc_type ufunc;

    // A vector of reductions
    std::vector<reduction_type> reductions_;
    std::vector<ReductionFlag> flags_;
    std::vector<double> buffer_;

    std::vector<std::tuple<ssize_t, reduction_type>> reductions_diff_ = {};
    std::vector<Update> diff_ = {};

    // Dynamic nodes need us to hold some information about their shape etc
    struct shape_info_type {
        shape_info_type() = delete;

        shape_info_type(ssize_t size, std::span<const ssize_t> shape)
                : shape(shape.begin(), shape.end()),
                  size_divisor(std::reduce(shape.begin() + 1, shape.end(), 1,
                                           std::multiplies<ssize_t>())),
                  previous_size(size) {
            this->shape[0] = size / size_divisor;
        }

        std::vector<ssize_t> shape;

        // relationship between the size of the buffer and the size of the first dimension
        ssize_t size_divisor;

        ssize_t previous_size;
    };
    std::optional<shape_info_type> shape_info_;
};

// Drop the given axes from the range. Assumes axes is sorted and unique
std::vector<ssize_t> drop_axes(std::ranges::sized_range auto&& range,
                               std::span<const ssize_t> axes) {
    assert(std::ranges::is_sorted(axes));
    assert(([&]() {
        std::vector<ssize_t> ax(axes.begin(), axes.end());
        const auto [ret, end] = std::ranges::unique(ax);
        return ret == end;
    })());

    std::vector<ssize_t> out;

    auto axes_it = axes.begin();
    auto range_it = std::ranges::begin(range);
    for (ssize_t dim = 0, stop = std::ranges::size(range); dim < stop; ++dim, ++range_it) {
        if (dim == *axes_it) {
            // skipped axis, this relies on axes being sorted and unique
            ++axes_it;
        } else {
            // kept axis
            out.emplace_back(*range_it);
        }
    }

    return out;
}

bool is_unique(std::ranges::range auto&& range) {
    assert(std::ranges::is_sorted(range));
    auto trail = std::ranges::begin(range);
    const auto end = std::ranges::end(range);
    if (trail == end) return true;
    for (auto lead = std::next(trail); lead != end; ++trail, ++lead) {
        if (*trail == *lead) return false;
    }
    return true;
}

std::vector<ssize_t> keep_axes(std::ranges::random_access_range auto&& range,
                             std::span<const ssize_t> axes) {
    std::vector<ssize_t> out;
    const auto begin = std::ranges::cbegin(range);
    for (const ssize_t& dim : axes) {
        out.emplace_back(*(begin + dim));
    }
    return out;
}

// Given a set of axes we wish to perform the reduction, normalize them into a sorted, unique, and
// nonnegative vector. Raise errors if they are invalid.
std::vector<ssize_t> normalize_axes(const ArrayNode* array_ptr, std::span<const ssize_t> axes) {
    const ssize_t ndim = array_ptr->ndim();

    std::vector<ssize_t> normalized(axes.begin(), axes.end());

    // First, check bounds and make nonnegative
    for (ssize_t& dim : normalized) {
        // NumPy raises `AxisError: axis -5 is out of bounds for array of dimension 2`
        if (dim < -ndim || ndim <= dim) {
            throw std::invalid_argument("axis " + std::to_string(dim) +
                                        " is out of bounds for array of dimension " +
                                        std::to_string(ndim));
        }

        if (dim < 0) dim += ndim;  // handle negative indices
    }

    // Next, sort
    std::ranges::sort(normalized);

    // Finally, throw an error if they are not unique
    // NumPy raises `ValueError: duplicate value in 'axis'`, we use a slightly
    // different message because in C++ we name the parameter "axes"
    if (const auto [ret, end] = std::ranges::unique(normalized); ret != end) {
        throw std::invalid_argument("duplicate axis value");
    }

    return normalized;
}

// Return the product of all elements in the range
auto product(std::ranges::range auto&& range) {
    return std::reduce(std::ranges::begin(range), std::ranges::end(range), 1,
                        std::multiplies<void>());
}

// The resulting shape when reducing the given array over the given axes
std::vector<ssize_t> reduce_shape(const ArrayNode* array_ptr, std::span<const ssize_t> axes) {
    // If axes is empty then we're reducing everything and therefore our resulting
    // shape is (). This does not check whether we have an initial value to handle the dynamic
    // case; that's handled by the constructor.
    if (!axes.size()) return {};

    std::vector<ssize_t> normalized = normalize_axes(array_ptr, axes);
    return drop_axes(array_ptr->shape(), normalized);
}

// The min/max/integrality when reducing the given array over the given `axes`.
template <class BinaryOp>
ValuesInfo values_info(const Array* array_ptr, std::span<const ssize_t> axes,
                       std::optional<double> initial) {
    // This function assumes axes is sorted and unique
    assert(std::ranges::is_sorted(axes));
    assert(is_unique(axes));

    // Our ufunc will determine our bounds
    typename std_to_ufunc<BinaryOp>::type ufunc;
    static_assert(decltype(ufunc)::associative);  // otherwise this doesn't make sense
    static_assert(decltype(ufunc)::commutative);  // otherwise this doesn't make sense

    // Get the bounds for our predecessor
    const auto array_bounds = ValuesInfo(array_ptr);

    // The easy case is that the size of each reduction is fixed.
    if (not array_ptr->dynamic() or (axes.size() > 0 && axes[0] != 0)) {
        const ssize_t reduction_size =
                axes.size() ? product(keep_axes(array_ptr->shape(), axes)) : array_ptr->size();
        assert(reduction_size >= 0);

        if (reduction_size and initial) {
            // cast initial to the correct output type
            auto init = typename decltype(ufunc)::reduction_type(*initial).value();

            return ufunc.result_bounds(
                    ufunc.result_bounds(array_bounds, reduction_size),  // from reducing the array
                    ValuesInfo(init, init, is_integer(init))            // from initial
            );
        } else if (reduction_size) {
            return ufunc.result_bounds(array_bounds, reduction_size);
        } else if (initial) {
            // cast initial to the correct output type
            auto init = typename decltype(ufunc)::reduction_type(*initial).value();

            return ValuesInfo(init, init, is_integer(init));
        } else {
            throw std::invalid_argument(
                    "cannot perform a reduction operation with no "
                    "identity on an array that might be empty");
        }
    }

    //
    // Unfortunately we're in the more complex case where the reduction size varies as a function
    // of the size of our predecessor array.
    //

    auto array_sizeinfo = array_ptr->sizeinfo();

    ssize_t min_size = array_sizeinfo.min.value_or(0);     // if we don't know we have to assume 0
    std::optional<ssize_t> max_size = array_sizeinfo.max;  // possibly unknown!

    // For each axis that we're *not* reducing over, we divide the size because it's not part of
    // the reduction space
    if (axes.size()) {
        for (const auto& dim_size : drop_axes(array_ptr->shape(), axes)) {
            assert(dim_size >= 0);

            min_size /= dim_size;
            if (max_size) *max_size /= dim_size;
        }
    }  // else we're reducing over all axes

    // If we don't have an initial value and min_size == 0 then we have a problem
    if (min_size == 0 and not initial) {
        // NumPy error message: ValueError: zero-size array to reduction operation maximum which has no identity
        throw std::invalid_argument(
                "cannot perform a reduction operation with no identity on an array that might be "
                "empty");
    }

    //
    // Ok, now that we (maybe) know the min/max size of our reduction space. we can start fiddling
    // with bounds.
    //

    if (max_size.has_value() && *max_size == 0) {
        // TODO Is it even possible to get here? Needs testing
        assert(false && "not yet implemeted");
    }

    // Get the bounds at each of the smallest and largest sizes we can be
    ValuesInfo min_bounds = ufunc.result_bounds(array_bounds, std::max<ssize_t>(min_size, 1));
    ValuesInfo max_bounds = max_size ? ufunc.result_bounds(array_bounds, *max_size)
                                     : ufunc.result_bounds(array_bounds, limit_type());

    // If there is not initial value, then our bounds are just the union of min_/max_bounds
    if (!initial.has_value()) return min_bounds | max_bounds;

    // Otherwise we need to account for the initial value in the bounds
    auto init = typename decltype(ufunc)::reduction_type(*initial).value();  // cast to bool etc
    auto initial_bounds = ValuesInfo(init, init, is_integer(init));
    min_bounds = ufunc.result_bounds(std::move(min_bounds), initial_bounds);
    max_bounds = ufunc.result_bounds(std::move(max_bounds), initial_bounds);

    auto bounds = min_bounds | max_bounds;  // the union of the domains

    // One more case (empty array)
    if (min_size == 0) bounds |= initial_bounds;

    return bounds;
}

template <class BinaryOp>
ReduceNode<BinaryOp>::ReduceNode(ArrayNode* array_ptr) : ReduceNode(array_ptr, {}) {}

template <class BinaryOp>
ReduceNode<BinaryOp>::ReduceNode(ArrayNode* array_ptr, std::span<const ssize_t> axes,
                                   std::optional<double> initial)
        : ArrayOutputMixin(reduce_shape(array_ptr, axes)),
          initial(initial),
          array_ptr_(array_ptr),
          axes_(normalize_axes(array_ptr, axes)),
          values_info_(values_info<BinaryOp>(array_ptr_, axes_, initial)) {
    // surely there is more I need to do here

    // Handle sizeinfo?

    // Handle values_info

    add_predecessor(array_ptr);
}

template <class BinaryOp>
ReduceNode<BinaryOp>::ReduceNode(ArrayNode* array_ptr, std::initializer_list<ssize_t> axes, std::optional<double> initial)
        : ReduceNode(array_ptr, std::span(axes), initial) {}

template <class BinaryOp>
double const* ReduceNode<BinaryOp>::buff(const State& state) const {
    return data_ptr<ReduceNodeData<BinaryOp>>(state)->buff();
}

template <class BinaryOp>
void ReduceNode<BinaryOp>::commit(State& state) const {
    return data_ptr<ReduceNodeData<BinaryOp>>(state)->commit();
}

template <class BinaryOp>
std::span<const Update> ReduceNode<BinaryOp>::diff(const State& state) const {
    return data_ptr<ReduceNodeData<BinaryOp>>(state)->diff();
}

template <class BinaryOp>
void ReduceNode<BinaryOp>::initialize_state(State& state) const {
    std::vector<typename ReduceNodeData<BinaryOp>::reduction_type> reductions;

    if (this->dynamic()) {
        const ssize_t subspace_size = product(keep_axes(array_ptr_->shape(state), axes_));
        const ssize_t size = array_ptr_->size(state) / subspace_size;

        for (ssize_t index = 0; index < size; ++index) {
            reductions.emplace_back(reduce_(state, index));
        }
        emplace_data_ptr<ReduceNodeData<BinaryOp>>(state, std::move(reductions), this->shape());
    } else {
        for (ssize_t index = 0; index < this->size(); ++index) {
            reductions.emplace_back(reduce_(state, index));
        }
        emplace_data_ptr<ReduceNodeData<BinaryOp>>(state, std::move(reductions));
    }
}

template <class BinaryOp>
bool ReduceNode<BinaryOp>::integral() const {
    return values_info_.integral;
}

template <class BinaryOp>
double ReduceNode<BinaryOp>::max() const {
    return values_info_.max;
}

template <class BinaryOp>
double ReduceNode<BinaryOp>::min() const {
    return values_info_.min;
}

template <class BinaryOp>
void ReduceNode<BinaryOp>::propagate(State& state) const {
    auto* const state_ptr = data_ptr<ReduceNodeData<BinaryOp>>(state);

    // We are reducing over all axes, so this is nice and simple
    if (axes_.empty() || axes_.size() == static_cast<std::size_t>(this->ndim())) {
        for (const Update& update : array_ptr_->diff(state)) {
            if (update.placed()) {
                state_ptr->add_to_reduction(0, update.value);
            } else if (update.removed()) {
                state_ptr->remove_from_reduction(0, update.old);
            } else {
                state_ptr->update_reduction(0, update.old, update.value);
            }
        }

        state_ptr->prepare_diff([&](const ssize_t idx) { return this->reduce_(state, idx); });
        return;
    }

    // Alas, we're in the more complex case where we have several axis to reduce over

    // It's OK for these to both be dynamic
    auto array_shape = array_ptr_->shape();
    auto reduce_shape = this->shape();

    if (this->dynamic() && array_ptr_->size_diff(state) > 0 && initial) {
        // Make sure we're including the initial values
        const ssize_t size_diff = array_ptr_->size_diff(state);

        const auto subspace_shape = keep_axes(array_ptr_->shape(state), axes_);
        const ssize_t subspace_size = std::reduce(subspace_shape.begin(), subspace_shape.end(), 1,
                                                  std::multiplies<ssize_t>());
        for (ssize_t i = 0, stop = size_diff / subspace_size; i < stop; ++i) {
            state_ptr->append_reduction(initial.value());
        }
    }

    for (const Update& update : array_ptr_->diff(state)) {

        // Convert the flat index in our predecessor to a multi-index
        auto multi_index = unravel_index(update.index, array_shape);

        // Convert the multi-index from the array's shape to ours
        multi_index = drop_axes(std::move(multi_index), axes_);

        // Then convert it back to a flat index
        ssize_t reduction_index = ravel_multi_index(multi_index, reduce_shape);

        if (update.placed()) {
            state_ptr->add_to_reduction(reduction_index, update.value);
        } else if (update.removed()) {
            state_ptr->remove_from_reduction(reduction_index, update.old);
        } else {
            state_ptr->update_reduction(reduction_index, update.old, update.value);
        }
    }

    if (this->dynamic() && array_ptr_->size_diff(state) < 0) {
        const ssize_t size_diff = array_ptr_->size_diff(state);

        const auto subspace_shape = keep_axes(array_ptr_->shape(state), axes_);
        const ssize_t subspace_size = std::reduce(subspace_shape.begin(), subspace_shape.end(), 1,
                                                  std::multiplies<ssize_t>());

        for (ssize_t i = size_diff / subspace_size; i < 0; ++i) {
            state_ptr->pop_reduction();
        }
    }

    state_ptr->prepare_diff([&](ssize_t index) { return this->reduce_(state, index); });
}

template <class BinaryOp>
auto ReduceNode<BinaryOp>::reduce_(const State& state, const ssize_t index) const {
    typename std_to_ufunc<BinaryOp>::type ufunc;

    // Everything is being reduced
    if (axes_.empty() || axes_.size() == static_cast<std::size_t>(array_ptr_->ndim())) {
        assert(index == 0);
        return ufunc.reduce(array_ptr_->view(state), initial);
    }

    const auto subspace_shape = keep_axes(array_ptr_->shape(state), axes_);  // state-dependent!
    const auto subspace_strides = keep_axes(array_ptr_->strides(), axes_);

    // Get the multi-index pointing to the location that the reduction will
    // be placed in *our* graph.
    std::vector<ssize_t> multi_index = unravel_index(index, this->shape());

    // Add back in the axes we reduced over and set them to 0 so that
    // multi_index now points to the beginning of each reduction in the
    // array we're reducing over
    for (const ssize_t& dim : axes_) {
        multi_index.insert(multi_index.begin() + dim, 0);
    }

    // We can then create an iterator that iterates of the reduction group
    auto begin = array_ptr_->begin(state);
    if (begin.shaped()) {
        begin += multi_index;
    } else {
        begin += ravel_multi_index(multi_index, array_ptr_->shape());
    }
    auto it = BufferIterator<double, double, true>(&*begin, subspace_shape, subspace_strides);

    return ufunc.reduce(std::ranges::subrange(it, std::default_sentinel), initial);
}

template <class BinaryOp>
void ReduceNode<BinaryOp>::revert(State& state) const {
    return data_ptr<ReduceNodeData<BinaryOp>>(state)->revert();
}

template <class BinaryOp>
std::span<const ssize_t> ReduceNode<BinaryOp>::shape(const State& state) const {
    if (ssize_t size = this->size(); size >= 0) return this->shape();  // if we're not dynamic
    return data_ptr<ReduceNodeData<BinaryOp>>(state)->shape();
}

template <class BinaryOp>
ssize_t ReduceNode<BinaryOp>::size(const State& state) const {
    if (ssize_t size = this->size(); size >= 0) return size;  // if we're not dynamic
    return data_ptr<ReduceNodeData<BinaryOp>>(state)->size();
}

template <class BinaryOp>
ssize_t ReduceNode<BinaryOp>::size_diff(const State& state) const {
    if (ssize_t size = this->size(); size >= 0) return 0;  // if we're not dynamic
    return data_ptr<ReduceNodeData<BinaryOp>>(state)->size_diff();
}

template class ReduceNode<functional::max<double>>;
template class ReduceNode<functional::min<double>>;
template class ReduceNode<std::logical_and<double>>;
template class ReduceNode<std::logical_or<double>>;
template class ReduceNode<std::multiplies<double>>;
template class ReduceNode<std::plus<double>>;

}  // namespace dwave::optimization

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
        reduction_type() = default;

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
    }

    static constexpr bool associative = true;
    static constexpr bool commutative = true;
    static constexpr bool invertible = true;
};

template <class BinaryOp>
struct std_to_ufunc {};
template <>
struct std_to_ufunc<std::multiplies<double>> {
    using type = prod<double>;
};
template <>
struct std_to_ufunc<std::plus<double>> {
    using type = add<double>;
};

template <class BinaryOp>
class ReduceNode2Data : public NodeStateData {
 public:
    using ufunc_type = std_to_ufunc<BinaryOp>::type;

    // Pull this into the namespace for convenience.
    using reduction_type = ufunc_type::reduction_type;

    enum class ReductionFlag {
        invalid,    // no longer holds the correct value, the whole reduction must be recalculated
        unchanged,  // no change since the last propagation
        updated,    // has been updated and holds the correct value
    };

    ReduceNode2Data(reduction_type reduction) : ReduceNode2Data(std::vector{reduction}) {}

    ReduceNode2Data(std::vector<reduction_type>&& reductions): reductions_(std::move(reductions)), flags_(reductions_.size(), ReductionFlag::unchanged) {
        buffer_.reserve(reductions_.size());
        for (const auto& reduction : reductions_) buffer_.emplace_back(reduction.value());
    }

    ReduceNode2Data(std::vector<reduction_type>&& reductions, std::span<const ssize_t> shape) : ReduceNode2Data(std::move(reductions)) {
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

    void prepare_diff() {
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
                    assert(false);
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
            assert(false);  // not yet tested
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
            assert(false);  // not yet tested
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
ValuesInfo values_info(ArrayNode* array_ptr, std::span<const ssize_t> axes,
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
        const ssize_t reduction_size = product(keep_axes(array_ptr->shape(), axes));
        auto bounds = ufunc.result_bounds(array_bounds, reduction_size);
        if (not initial) return bounds;
        auto initial_bounds = ValuesInfo(*initial, *initial, is_integer(*initial));
        return ufunc.result_bounds(bounds, initial_bounds);
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
    auto initial_bounds = ValuesInfo(*initial, *initial, is_integer(*initial));
    min_bounds = ufunc.result_bounds(std::move(min_bounds), initial_bounds);
    max_bounds = ufunc.result_bounds(std::move(max_bounds), initial_bounds);

    auto bounds = min_bounds | max_bounds;  // the union of the domains

    // One more case (empty array)
    if (min_size == 0) bounds |= initial_bounds;

    return bounds;
}

template <class BinaryOp>
ReduceNode2<BinaryOp>::ReduceNode2(ArrayNode* array_ptr) : ReduceNode2(array_ptr, {}) {}

template <class BinaryOp>
ReduceNode2<BinaryOp>::ReduceNode2(ArrayNode* array_ptr, std::span<const ssize_t> axes,
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
ReduceNode2<BinaryOp>::ReduceNode2(ArrayNode* array_ptr, std::initializer_list<ssize_t> axes, std::optional<double> initial)
        : ReduceNode2(array_ptr, std::span(axes), initial) {}

template <class BinaryOp>
double const* ReduceNode2<BinaryOp>::buff(const State& state) const {
    return data_ptr<ReduceNode2Data<BinaryOp>>(state)->buff();
}

template <class BinaryOp>
void ReduceNode2<BinaryOp>::commit(State& state) const {
    return data_ptr<ReduceNode2Data<BinaryOp>>(state)->commit();
}

template <class BinaryOp>
std::span<const Update> ReduceNode2<BinaryOp>::diff(const State& state) const {
    return data_ptr<ReduceNode2Data<BinaryOp>>(state)->diff();
}

template <class BinaryOp>
void ReduceNode2<BinaryOp>::initialize_state(State& state) const {
    using ufunc_type = typename ReduceNode2Data<BinaryOp>::ufunc_type;
    ufunc_type ufunc;
    assert(ufunc.associative && ufunc.commutative);  // we rely on this

    // We are reducing over all axes, so this is nice and simple
    if (axes_.empty() || axes_.size() == static_cast<std::size_t>(array_ptr_->ndim())) {
        auto reduction = ufunc.reduce(array_ptr_->view(state), initial);
        emplace_data_ptr<ReduceNode2Data<BinaryOp>>(state, reduction);
        return;
    }

    // Alas, we're in the more complex case where we have several axis to reduce over
    const auto subspace_shape = keep_axes(array_ptr_->shape(state), axes_);  // state-dependent!
    const auto subspace_strides = keep_axes(array_ptr_->strides(), axes_);
    const ssize_t subspace_size = std::reduce(subspace_shape.begin(), subspace_shape.end(), 1,
                                              std::multiplies<ssize_t>());

    const auto array_begin = array_ptr_->begin(state);

    std::vector<typename ufunc_type::reduction_type> reductions;
    for (ssize_t index = 0, size = array_ptr_->size(state) / subspace_size; index < size; ++index) {
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
        auto begin = array_begin;  // make a copy that we can mutate
        if (begin.shaped()) {
            begin.advance_to(multi_index);
        } else {
            begin += ravel_multi_index(multi_index, array_ptr_->shape());
        }
        auto it = BufferIterator<double, double, true>(&*begin, subspace_shape, subspace_strides);

        // Calculate the reduction over said group
        auto reduction = ufunc.reduce(std::ranges::subrange(it, std::default_sentinel), initial);

        // Finally add it to our state
        reductions.emplace_back(std::move(reduction));
    }

    if (this->dynamic()) {
        emplace_data_ptr<ReduceNode2Data<BinaryOp>>(state, std::move(reductions), this->shape());
    } else {
        emplace_data_ptr<ReduceNode2Data<BinaryOp>>(state, std::move(reductions));
    }
}

template <class BinaryOp>
bool ReduceNode2<BinaryOp>::integral() const {
    return values_info_.integral;
}

template <class BinaryOp>
double ReduceNode2<BinaryOp>::max() const {
    return values_info_.max;
}

template <class BinaryOp>
double ReduceNode2<BinaryOp>::min() const {
    return values_info_.min;
}

template <class BinaryOp>
void ReduceNode2<BinaryOp>::propagate(State& state) const {
    using ufunc_type = typename ReduceNode2Data<BinaryOp>::ufunc_type;
    ufunc_type ufunc;
    assert(ufunc.associative && ufunc.commutative);  // we rely on this

    auto* const state_ptr = data_ptr<ReduceNode2Data<BinaryOp>>(state);

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

        state_ptr->prepare_diff();
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

    state_ptr->prepare_diff();
}

template <class BinaryOp>
void ReduceNode2<BinaryOp>::revert(State& state) const {
    return data_ptr<ReduceNode2Data<BinaryOp>>(state)->revert();
}

template <class BinaryOp>
std::span<const ssize_t> ReduceNode2<BinaryOp>::shape(const State& state) const {
    if (ssize_t size = this->size(); size >= 0) return this->shape();  // if we're not dynamic
    return data_ptr<ReduceNode2Data<BinaryOp>>(state)->shape();
}

template <class BinaryOp>
ssize_t ReduceNode2<BinaryOp>::size(const State& state) const {
    if (ssize_t size = this->size(); size >= 0) return size;  // if we're not dynamic
    return data_ptr<ReduceNode2Data<BinaryOp>>(state)->size();
}

template <class BinaryOp>
ssize_t ReduceNode2<BinaryOp>::size_diff(const State& state) const {
    if (ssize_t size = this->size(); size >= 0) return 0;  // if we're not dynamic
    return data_ptr<ReduceNode2Data<BinaryOp>>(state)->size_diff();
}

template class ReduceNode2<std::multiplies<double>>;
template class ReduceNode2<std::plus<double>>;






// A class that encodes the product of an array. Allows for incremental updates
// without needing to recalculate the whole thing.
struct RunningProduct {
    RunningProduct() : nonzero(1), num_zero(0) {}
    explicit RunningProduct(const double init) : nonzero(init ? init : 1), num_zero(init == 0) {}

    // Get as a double. I.e., 0^num_zero * nonzero
    explicit operator double() const noexcept { return (num_zero > 0) ? 0 : nonzero; }

    // Multiply by a number, tracking the number of times it's been multiplied by 0
    RunningProduct& operator*=(const double rhs) noexcept {
        if (rhs) {
            nonzero *= rhs;
        } else {
            num_zero += 1;
        }
        return *this;
    }

    // Divide by a number, tracking the number of times it's been divided by 0
    RunningProduct& operator/=(const double rhs) {
        if (rhs) {
            nonzero /= rhs;
        } else {
            num_zero -= 1;

            // the implementation is well-defined for num_zero < 0, but conceptually
            // it's not really well defined.
            assert(num_zero >= 0);
        }
        return *this;
    }

    // Multiply and divide
    void multiply_divide(const double multiplier, const double divisor) {
        // This function exists so that in the future we can be a bit more careful
        // about numeric errors. For now we just delegate to other methods for
        // simplicity

        if (multiplier == divisor) return;  // nothing to do when they cancel out

        *this *= multiplier;
        *this /= divisor;
    }

    // The output value is 0^num_zero * nonzero
    double nonzero;  // todo: consider float128 or other way of managing floating point errors
    ssize_t num_zero;
};

// For a product reduction over n elements each taking values in [low, high]
// and an initial value of init, what is the min it can take?
double product_max(ssize_t n, double init, double low, double high) {
    if (n <= 0) return init;
    if (n == 1) return std::max<double>(init * low, init * high);
    return std::max<double>({
            init * std::pow(low, n),
            init * std::pow(high, n),
            init * low * std::pow(high, n - 1),
            init * high * std::pow(low, n - 1),
    });
}
double product_min(ssize_t n, double init, double low, double high) {
    if (n <= 0) return init;
    if (n == 1) return std::min<double>(init * low, init * high);
    return std::min<double>({
            init * std::pow(low, n),
            init * std::pow(high, n),
            init * low * std::pow(high, n - 1),
            init * high * std::pow(low, n - 1),
    });
}
std::pair<double, double> product_minmax(ssize_t n, double init, double low, double high) {
    return {product_min(n, init, low, high), product_max(n, init, low, high)};
}

// The state of the PartialReduceNode - not implemented by default
template <class BinaryOp>
class PartialReduceNodeData {};

// The state of a PartialProdNode
template <>
class PartialReduceNodeData<std::multiplies<double>> : private ArrayStateData,
                                                       public NodeStateData {
 public:
    explicit PartialReduceNodeData(std::vector<RunningProduct>&& values) noexcept
            : ArrayStateData(running_to_double(values)),
              NodeStateData(),
              products_(std::move(values)) {}

    using ArrayStateData::buff;

    void commit() {
        ArrayStateData::commit();
        products_diff_.clear();
    }

    using ArrayStateData::diff;

    void revert() {
        ArrayStateData::revert();

        for (const auto& [index, product] : products_diff_ | std::views::reverse) {
            products_[index] = product;
        }
        products_diff_.clear();
    }

    // Incorporate a single update at the given index. I.e. one entry in
    // the relevant axis has updated its value from old -> value.
    void update(ssize_t index, double old, double value) {
        products_diff_.emplace_back(index, products_[index]);               // save the old value
        products_[index].multiply_divide(value, old);                       // update the accumlator
        ArrayStateData::set(index, static_cast<double>(products_[index]));  // set our output
    }

    using ArrayStateData::size_diff;

 private:
    // convert a vector of running doubles to a vector of doubles via an explicit static_cast
    static std::vector<double> running_to_double(const std::vector<RunningProduct>& values) {
        std::vector<double> out;
        out.reserve(values.size());
        for (const auto& val : values) {
            out.emplace_back(static_cast<double>(val));
        }
        return out;
    }

    std::vector<RunningProduct> products_;
    std::vector<std::tuple<ssize_t, RunningProduct>> products_diff_;
};

// The state of a PartialSumNode
template <>
class PartialReduceNodeData<std::plus<double>> : public ArrayNodeStateData {
 public:
    explicit PartialReduceNodeData(std::vector<double>&& values) noexcept
            : ArrayNodeStateData(std::move(values)) {}

    // Incorporate a single update at the given index. I.e. one entry in
    // the relevant axis has updated its value from old -> value.
    void update(ssize_t index, double old, double value) {
        ArrayStateData::set(index, ArrayStateData::get(index) - old + value);
    }
};

std::vector<ssize_t> partial_reduce_shape(const std::span<const ssize_t> input_shape,
                                          const ssize_t axis) {
    std::vector<ssize_t> shape(input_shape.begin(), input_shape.end());
    shape.erase(shape.begin() + axis);
    return shape;
}

std::vector<ssize_t> as_contiguous_strides(const std::span<const ssize_t> shape) {
    ssize_t ndim = static_cast<ssize_t>(shape.size());

    assert(ndim > 0);
    std::vector<ssize_t> strides(ndim);
    // otherwise strides are a function of the shape
    strides[ndim - 1] = sizeof(double);
    for (auto i = ndim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

std::span<const ssize_t> nonempty(std::span<const ssize_t> span) {
    if (span.empty()) {
        throw std::invalid_argument("Input span should not be empty");
    }
    return span;
}

template <class BinaryOp>
bool partial_reduce_calculate_integral(const Array* array_ptr, const std::optional<double>& init) {
    using result_type = typename std::invoke_result<BinaryOp, double&, double&>::type;

    if constexpr (std::is_integral<result_type>::value) {
        return true;
    }
    if constexpr (std::same_as<BinaryOp, std::multiplies<double>>) {
        return array_ptr->integral() && is_integer(init.value_or(1));
    }
    if constexpr (std::same_as<BinaryOp, std::plus<double>>) {
        return array_ptr->integral() && is_integer(init.value_or(0));
    }

    assert(false && "not implemented yet");
    unreachable();
}

template <class BinaryOp>
ValuesInfo partial_reduce_calculate_values_info(const Array* array_ptr, ssize_t axis,
                                                const std::optional<double>& init) {
    // If the output of the operation is boolean, then the min/max is simple
    using result_type = typename std::invoke_result<BinaryOp, double&, double&>::type;
    if constexpr (std::same_as<result_type, bool>) {
        return {false, true, true};
    }

    // Otherwise the min and max depend on the predecessors

    bool integral = partial_reduce_calculate_integral<BinaryOp>(array_ptr, init);

    // Get the size of the axis we're reducing on
    const ssize_t size = array_ptr->shape()[axis];

    if constexpr (std::same_as<BinaryOp, std::plus<double>>) {
        return {init.value_or(0) + size * array_ptr->min(),
                init.value_or(0) + size * array_ptr->max(), integral};
    }
    if constexpr (std::same_as<BinaryOp, std::multiplies<double>>) {
        const auto& [low, high] =
                product_minmax(size, init.value_or(1), array_ptr->min(), array_ptr->max());
        return {low, high, integral};
    }

    assert(false && "not implemeted yet");
    unreachable();
}

/// TODO: support multiple axes
template <class BinaryOp>
PartialReduceNode<BinaryOp>::PartialReduceNode(ArrayNode* node_ptr, std::span<const ssize_t> axes,
                                               double init)
        : ArrayOutputMixin(partial_reduce_shape(node_ptr->shape(), nonempty(axes)[0])),
          init(init),
          array_ptr_(node_ptr),
          axes_(make_axes(axes)),
          parent_strides_c_(as_contiguous_strides(array_ptr_->shape())),
          values_info_(partial_reduce_calculate_values_info<BinaryOp>(array_ptr_, axes_[0], init)) {
    if (array_ptr_->dynamic()) {
        throw std::invalid_argument("cannot do a partial reduction on a dynamic array");
    } else if (array_ptr_->size() < 1) {
        throw std::invalid_argument("cannot do a partial reduction on an empty array");
    }

    // Validate axes
    /// TODO: support negative axes
    if (axes.size() != 1) {
        throw std::invalid_argument("Partial reduction support only one axis");
    }

    assert(this->ndim() == array_ptr_->ndim() - 1);
    ssize_t axis = axes_[0];
    if (axis < 0 || axis >= array_ptr_->ndim()) {
        throw std::invalid_argument("Axes should be integers between 0 and n_dim - 1");
    }

    add_predecessor(node_ptr);
}

template <class BinaryOp>
PartialReduceNode<BinaryOp>::PartialReduceNode(ArrayNode* node_ptr,
                                               std::initializer_list<ssize_t> axes, double init)
        : PartialReduceNode(node_ptr, std::span(axes), init) {}

template <class BinaryOp>
PartialReduceNode<BinaryOp>::PartialReduceNode(ArrayNode* node_ptr, const ssize_t axis, double init)
        : PartialReduceNode(node_ptr, {axis}, init) {}

template <>
PartialReduceNode<std::multiplies<double>>::PartialReduceNode(ArrayNode* array_ptr,
                                                              const ssize_t axis)
        : PartialReduceNode(array_ptr, axis, 1) {}

template <>
PartialReduceNode<std::multiplies<double>>::PartialReduceNode(ArrayNode* array_ptr,
                                                              std::span<const ssize_t> axes)
        : PartialReduceNode(array_ptr, nonempty(axes)[0], 1) {
    if (axes.size() != 1) {
        throw std::invalid_argument("Partial product supports only one axis");
    }
}

template <>
PartialReduceNode<std::plus<double>>::PartialReduceNode(ArrayNode* array_ptr, const ssize_t axis)
        : PartialReduceNode(array_ptr, axis, 0) {}

template <>
PartialReduceNode<std::plus<double>>::PartialReduceNode(ArrayNode* array_ptr,
                                                        std::span<const ssize_t> axes)
        : PartialReduceNode(array_ptr, nonempty(axes)[0], 0) {
    if (axes.size() != 1) {
        throw std::invalid_argument("Partial sum supports only one axis");
    }
}

template <class BinaryOp>
PartialReduceNode<BinaryOp>::PartialReduceNode(ArrayNode* array_ptr,
                                               std::initializer_list<ssize_t> axes)
        : PartialReduceNode(array_ptr, std::span(axes)) {}

template <class BinaryOp>
PartialReduceNode<BinaryOp>::PartialReduceNode(ArrayNode* array_ptr, const ssize_t axis)
        : PartialReduceNode(array_ptr, {axis}) {}

template <class BinaryOp>
std::span<const ssize_t> PartialReduceNode<BinaryOp>::axes() const {
    // todo: support multiple axes
    return std::span(axes_.get(), 1);
}

template <class BinaryOp>
double const* PartialReduceNode<BinaryOp>::buff(const State& state) const {
    return data_ptr<PartialReduceNodeData<BinaryOp>>(state)->buff();
}

template <class BinaryOp>
void PartialReduceNode<BinaryOp>::commit(State& state) const {
    data_ptr<PartialReduceNodeData<BinaryOp>>(state)->commit();
}

template <class BinaryOp>
std::span<const Update> PartialReduceNode<BinaryOp>::diff(const State& state) const {
    return data_ptr<PartialReduceNodeData<BinaryOp>>(state)->diff();
}

template <class BinaryOp>
void PartialReduceNode<BinaryOp>::initialize_state(State& state) const {
    // the type we use for the reduction depends on the node type
    using accumulator_type = std::conditional<std::same_as<BinaryOp, std::multiplies<double>>,
                                              RunningProduct, double>::type;

    std::vector<accumulator_type> values(size(state));

    const ssize_t axis = axes_[0];

    for (ssize_t index = 0, stop = size(state); index < stop; ++index) {
        assert(index < static_cast<ssize_t>(values.size()));

        /// We wish to fill `index` of the partial reduction, given the state of the parent.
        /// We proceed as follows.
        /// 1. We unravel the index. This will give us e.g. the indices i, k for
        ///     R_ik = sum_j P_ijk
        /// 2. Then we know we have to iterate through the parent starting from j=0 to its full
        /// dimension. To do so, we first find the starting index in the parent.
        /// 3. We then create an iterator through the parent array along the axis.
        /// 4. We use the above iterators to perform the reduction.

        // 1. Unravel the index we are trying to fill
        std::vector<ssize_t> indices = unravel_index(index, this->shape());
        assert(static_cast<ssize_t>(indices.size()) == this->ndim());

        // 2. Find the respective starting index in the parent
        ssize_t start_idx = 0;
        for (ssize_t ax = 0, stop = this->ndim(); ax < stop; ++ax) {
            if (ax >= axis) {
                start_idx += indices[ax] * array_ptr_->strides()[ax + 1] / array_ptr_->itemsize();
            } else {
                start_idx += indices[ax] * array_ptr_->strides()[ax] / array_ptr_->itemsize();
            }
        }
        assert(start_idx >= 0 && index < this->size(state));

        // 3. We create an iterator that starts from index just found and iterates through the axis
        // we are reducing over
        const_iterator begin =
                const_iterator(array_ptr_->buff(state) + start_idx, 1, &array_ptr_->shape()[axis],
                               &array_ptr_->strides()[axis]);

        const_iterator end = begin + array_ptr_->shape(state)[axis];

        // Get the initial value
        double init;
        if (this->init.has_value()) {
            init = this->init.value();
        } else {
            // if there is no init, we use the first value in the array as the init
            // we should only be here if the array is not dynamic and if it has at
            // least one entry
            assert(!array_ptr_->dynamic());
            assert(array_ptr_->size(state) >= 1);
            init = *begin;
            ++begin;
        }

        if constexpr (std::same_as<BinaryOp, std::multiplies<double>>) {
            // we use accumulate rather than reduce so we have an easier lambda
            // description
            values[index] = std::accumulate(begin, end, accumulator_type(init),
                                            [](const RunningProduct& lhs, const double& rhs) {
                                                RunningProduct out = lhs;
                                                out *= rhs;
                                                return out;
                                            });
        } else {
            values[index] = std::reduce(begin, end, init, BinaryOp());
        }
    }

    emplace_data_ptr<PartialReduceNodeData<BinaryOp>>(state, std::move(values));
}

template <class BinaryOp>
ssize_t PartialReduceNode<BinaryOp>::map_parent_index(const State& state,
                                                      ssize_t parent_flat_index) const {
    ssize_t axis = this->axes_[0];
    assert(axis >= 0 && axis < array_ptr_->size(state));
    assert(parent_flat_index >= 0 && parent_flat_index < array_ptr_->size(state));

    ssize_t index = 0;  // the linear index corresponding to the parent index being updated
    ssize_t current_parent_axis = 0;  // current parent axis visited
    ssize_t current_axis = 0;         // current axis

    for (auto stride : parent_strides_c_) {
        // calculate parent index on the current axis (not the linear one)
        ssize_t this_axis_index = parent_flat_index / (stride / array_ptr_->itemsize());

        // update the index now
        parent_flat_index = parent_flat_index % (stride / array_ptr_->itemsize());

        if (current_parent_axis == axis) {
            current_parent_axis++;
            continue;
        }

        // now do the calculation of the linear index of reduction
        index += this_axis_index * (this->strides()[current_axis] / this->itemsize());

        // increase the axis visited
        current_axis++;
        current_parent_axis++;
    }

    return index;
}

template <class BinaryOp>
bool PartialReduceNode<BinaryOp>::integral() const {
    return values_info_.integral;
}

template <class BinaryOp>
double PartialReduceNode<BinaryOp>::min() const {
    return values_info_.min;
}

template <class BinaryOp>
double PartialReduceNode<BinaryOp>::max() const {
    return values_info_.max;
}

template <class BinaryOp>
void PartialReduceNode<BinaryOp>::propagate(State& state) const {
    auto ptr = data_ptr<PartialReduceNodeData<BinaryOp>>(state);

    for (const auto& [p_index, old, value] : array_ptr_->diff(state)) {
        const ssize_t index = map_parent_index(state, p_index);
        ptr->update(index, old, value);
    }

    if (ptr->diff().size()) Node::propagate(state);
}

template <class BinaryOp>
void PartialReduceNode<BinaryOp>::revert(State& state) const {
    data_ptr<PartialReduceNodeData<BinaryOp>>(state)->revert();
}

template <class BinaryOp>
std::span<const ssize_t> PartialReduceNode<BinaryOp>::shape(const State& state) const {
    return this->shape();
}

template <class BinaryOp>
ssize_t PartialReduceNode<BinaryOp>::size(const State& state) const {
    return this->size();
}

template <class BinaryOp>
ssize_t PartialReduceNode<BinaryOp>::size_diff(const State& state) const {
    return data_ptr<PartialReduceNodeData<BinaryOp>>(state)->size_diff();
}

// Uncommented are the tested specializations
template class PartialReduceNode<std::multiplies<double>>;
template class PartialReduceNode<std::plus<double>>;

// Create a storage class for op-specific values.
template <class BinaryOp>
struct ExtraData {
    void commit() {}
    void revert() {}
};

template <>
struct ExtraData<std::logical_and<double>> {
    explicit ExtraData(ssize_t num_zero) : num_zero(num_zero) {}

    virtual ~ExtraData() = default;

    void commit() { old_num_zero = num_zero; }
    void revert() { num_zero = old_num_zero; }

    ssize_t num_zero;
    ssize_t old_num_zero = num_zero;
};

template <>
struct ExtraData<std::logical_or<double>> {
    explicit ExtraData(ssize_t num_nonzero) : num_nonzero(num_nonzero) {}

    virtual ~ExtraData() = default;

    void commit() { old_num_nonzero = num_nonzero; }
    void revert() { num_nonzero = old_num_nonzero; }

    ssize_t num_nonzero;  // num nonzero, following SciPy's naming
    ssize_t old_num_nonzero = num_nonzero;
};

template <>
struct ExtraData<std::multiplies<double>> {
    explicit ExtraData(RunningProduct product) : product(product), old_product(product) {}

    virtual ~ExtraData() = default;

    void commit() { old_product = product; }
    void revert() { product = old_product; }

    // RunningProduct tracks the multiplications by 0 separately, which means we
    // don't need to recalculate everything when there's a 0 in there.
    RunningProduct product;
    RunningProduct old_product;
};

template <class BinaryOp>
struct ReduceNodeData : NodeStateData {
    template <class... Ts>
    explicit ReduceNodeData(double value, Ts... extra_args)
            : values(0, value, value), extra(extra_args...) {}

    double const* buff() const { return &values.value; }

    void commit() {
        values.old = values.value;
        extra.commit();
    }
    void revert() {
        values.value = values.old;
        extra.revert();
    }

    Update values;

    // op-specific storage.
    ExtraData<BinaryOp> extra;
};

template <class BinaryOp>
bool reduce_calculate_integral(const Array* array_ptr, const std::optional<double>& init) {
    using result_type = typename std::invoke_result<BinaryOp, double&, double&>::type;

    if constexpr (std::is_integral<result_type>::value) {
        return true;
    }

    if constexpr (std::is_same<BinaryOp, functional::max<double>>::value ||
                  std::is_same<BinaryOp, functional::min<double>>::value ||
                  std::is_same<BinaryOp, std::multiplies<double>>::value ||
                  std::is_same<BinaryOp, std::plus<double>>::value) {
        // the actual value of init doesn't matter, all of the above have no default
        // or an integer default init
        return array_ptr->integral() && is_integer(init.value_or(0));
    }

    assert(false && "not implemeted yet");
    unreachable();
}

template <class BinaryOp>
ValuesInfo reduce_calculate_values_info(const Array* array_ptr, const std::optional<double>& init) {
    // If the output of the operation is boolean, the min/max are just [false, true]
    using result_type = typename std::invoke_result<BinaryOp, double&, double&>::type;
    if constexpr (std::same_as<result_type, bool>) {
        return ValuesInfo::logical_output();
    }

    // Otherwise the min and max depend on the predecessors

    auto op = BinaryOp();

    bool integral = reduce_calculate_integral<BinaryOp>(array_ptr, init);

    auto low = array_ptr->min();
    auto high = array_ptr->max();

    // These can results in inf. If we fix that in initialization/propagation we
    // should also fix it here.
    if constexpr (std::same_as<BinaryOp, functional::max<double>> ||
                  std::same_as<BinaryOp, functional::min<double>>) {
        if (init.has_value()) {
            low = op(low, init.value());
            high = op(high, init.value());
        }
        return {low, high, integral};
    }
    if constexpr (std::same_as<BinaryOp, std::multiplies<double>>) {
        // the dynamic case. For now let's just fall back to Array's default
        // implementation because this gets even more complicated
        if (array_ptr->dynamic()) {
            return {std::numeric_limits<double>::lowest(), std::numeric_limits<double>::max(),
                    integral};
        }

        auto const& [this_low, this_high] =
                product_minmax(array_ptr->size(), init.value_or(1), low, high);
        return {this_low, this_high, integral};
    }
    if constexpr (std::same_as<BinaryOp, std::plus<double>>) {
        const double init_ = init.value_or(0);

        // if the array has a finite fixed size, then just multuply the largest value
        // by that size
        if (const ssize_t size = array_ptr->size(); size >= 0) {
            return {init_ + size * low, init_ + size * high, integral};
        }

        // our predecessor array is dynamic. So there are a few more cases
        // we need to check

        // 100 is a magic number. It's how far back in the predecessor
        // chain to check to get good bounds on the size for the given array.
        // This will exit early if it converges.
        const SizeInfo sizeinfo = array_ptr->sizeinfo().substitute(100);

        if (high > 0) {
            // if high is positive, then we're interested in the maxmum size
            // of the array

            // if the array is arbitrarily large, then just fall back to the
            // default max.
            if (!sizeinfo.max.has_value()) {
                high = std::numeric_limits<double>::max();
            } else {
                high = init_ + sizeinfo.max.value() * high;
            }
        } else {
            // if high is negative, then we're interested in the minimum size
            // of the array
            high = init_ + sizeinfo.min.value_or(0) * high;
        }

        if (low < 0) {
            // if low is negative, then we're interested in the maximum size
            // of the array

            // if the array is arbitrarily large, then just fall back to the
            // default min.
            if (!sizeinfo.max.has_value()) {
                low = std::numeric_limits<double>::lowest();
            } else {
                low = init_ + sizeinfo.max.value() * low;
            }
        } else {
            // if low is positive, then we're interested in the minimum size
            // of the array
            low = init_ + sizeinfo.min.value_or(0) * low;
        }

        return {low, high, integral};
    }

    assert(false && "not implemeted yet");
    unreachable();
}

template <class BinaryOp>
ReduceNode<BinaryOp>::ReduceNode(ArrayNode* node_ptr, double init)
        : init(init),
          array_ptr_(node_ptr),
          values_info_(reduce_calculate_values_info<BinaryOp>(array_ptr_, init)) {
    add_predecessor(node_ptr);
}

template <>
ReduceNode<std::logical_and<double>>::ReduceNode(ArrayNode* array_ptr) : ReduceNode(array_ptr, 1) {}

template <>
ReduceNode<std::logical_or<double>>::ReduceNode(ArrayNode* array_ptr) : ReduceNode(array_ptr, 0) {}

template <>
ReduceNode<std::multiplies<double>>::ReduceNode(ArrayNode* array_ptr) : ReduceNode(array_ptr, 1) {}

template <>
ReduceNode<std::plus<double>>::ReduceNode(ArrayNode* array_ptr) : ReduceNode(array_ptr, 0) {}

template <class BinaryOp>
ReduceNode<BinaryOp>::ReduceNode(ArrayNode* array_ptr)
        : init(),
          array_ptr_(array_ptr),
          values_info_(reduce_calculate_values_info<BinaryOp>(array_ptr_, init)) {
    if (array_ptr_->dynamic()) {
        throw std::invalid_argument(
                "cannot do a reduction on a dynamic array with an operation that has no identity "
                "without supplying an initial value");
    } else if (array_ptr_->size() < 1) {
        throw std::invalid_argument(
                "cannot do a reduction on an empty array with an operation that has no identity "
                "without supplying an initial value");
    }

    add_predecessor(array_ptr);
}

template <class BinaryOp>
void ReduceNode<BinaryOp>::commit(State& state) const {
    data_ptr<ReduceNodeData<BinaryOp>>(state)->commit();
}

template <class BinaryOp>
double const* ReduceNode<BinaryOp>::buff(const State& state) const {
    return data_ptr<ReduceNodeData<BinaryOp>>(state)->buff();
}

template <class BinaryOp>
std::span<const Update> ReduceNode<BinaryOp>::diff(const State& state) const {
    const Update& update = data_ptr<ReduceNodeData<BinaryOp>>(state)->values;
    return std::span<const Update>(&update, static_cast<int>(update.old != update.value));
}

template <class BinaryOp>
void ReduceNode<BinaryOp>::initialize_state(State& state) const {
    emplace_data_ptr<ReduceNodeData<BinaryOp>>(state, reduce(state));
}

template <>
void ReduceNode<std::logical_and<double>>::initialize_state(State& state) const {
    ssize_t num_zero = init.value_or(1) ? 0 : 1;
    for (const double value : array_ptr_->view(state)) {
        num_zero += !value;
    }

    emplace_data_ptr<ReduceNodeData<std::logical_and<double>>>(state, !num_zero, num_zero);
}

template <>
void ReduceNode<std::logical_or<double>>::initialize_state(State& state) const {
    ssize_t num_nonzero = init.value_or(1) ? 1 : 0;
    for (const double value : array_ptr_->view(state)) {
        num_nonzero += static_cast<bool>(value);
    }

    emplace_data_ptr<ReduceNodeData<std::logical_or<double>>>(state, num_nonzero > 0, num_nonzero);
}

template <>
void ReduceNode<std::multiplies<double>>::initialize_state(State& state) const {
    // there is an edge case here for init being 0, in that case the `nonzero`
    // component will always be 0, which is dumb but everything still works and
    // it's enough of an edge case that I don't think it makes sense to do
    // anything-performance wise.

    RunningProduct product(init.value_or(1));

    for (const double value : array_ptr_->view(state)) {
        product *= value;
    }

    // and then create the state
    emplace_data_ptr<ReduceNodeData<std::multiplies<double>>>(state, static_cast<double>(product),
                                                              product);
}

template <class BinaryOp>
bool ReduceNode<BinaryOp>::integral() const {
    return values_info_.integral;
}

template <class BinaryOp>
double ReduceNode<BinaryOp>::min() const {
    return values_info_.min;
}

template <class BinaryOp>
double ReduceNode<BinaryOp>::max() const {
    return values_info_.max;
}

template <>
void ReduceNode<std::logical_and<double>>::propagate(State& state) const {
    auto ptr = data_ptr<ReduceNodeData<std::logical_and<double>>>(state);

    ssize_t& num_zero = ptr->extra.num_zero;

    // count the change in the num_zero
    for (const Update& update : array_ptr_->diff(state)) {
        const auto& [_, old, value] = update;

        if (update.placed()) {
            // added a value to the array
            num_zero += !value;
        } else if (update.removed()) {
            // removed a value from the array
            num_zero -= !old;
        } else if (!old && value) {
            // changed a 0 to a truthy value
            num_zero -= 1;
        } else if (old && !value) {
            // changed a truthy value to a 0
            num_zero += 1;
        } else {
            // otherwise we don't care because the net number of 0s hasn't changed
            assert(static_cast<bool>(old) == static_cast<bool>(value));
        }
    }

    assert(num_zero >= 0);  // should never go negative

    ptr->values.value = num_zero < 1;
    if (ptr->values.value != ptr->values.old) Node::propagate(state);
}

template <>
void ReduceNode<std::logical_or<double>>::propagate(State& state) const {
    auto ptr = data_ptr<ReduceNodeData<std::logical_or<double>>>(state);

    ssize_t& num_nonzero = ptr->extra.num_nonzero;

    // count the change in the num_nonzero
    for (const Update& update : array_ptr_->diff(state)) {
        const auto& [_, old, value] = update;

        if (update.placed()) {
            // added a value to the array
            num_nonzero += static_cast<bool>(value);
        } else if (update.removed()) {
            // removed a value from the array
            num_nonzero -= static_cast<bool>(old);
        } else if (!old && value) {
            // changed a 0 to a truthy value
            num_nonzero += 1;
        } else if (old && !value) {
            // changed a truthy value to a 0
            num_nonzero -= 1;
        } else {
            // otherwise we don't care because the net number of 0s hasn't changed
            assert(static_cast<bool>(old) == static_cast<bool>(value));
        }
    }

    assert(num_nonzero >= 0);  // should never go negative

    ptr->values.value = num_nonzero > 0;
    if (ptr->values.value != ptr->values.old) Node::propagate(state);
}

template <>
void ReduceNode<functional::max<double>>::propagate(State& state) const {
    auto ptr = data_ptr<ReduceNodeData<functional::max<double>>>(state);

    auto& value = ptr->values.value;

    bool reinitialize = false;
    for (const Update& update : array_ptr_->diff(state)) {
        if (update.removed()) {
            if (update.old >= value) {
                // uh oh, we may have just removed the known max. So we need to
                // recalculate everything from scratch
                reinitialize = true;
                break;
            }
        } else if (update.placed() || update.value > update.old) {
            // we added a new value or increased an existing once
            value = std::max(update.value, value);
        } else if (update.old == value && update.old != update.value) {
            // we potentially made the current max smaller
            assert(update.value < update.old);
            reinitialize = true;
            break;
        }
    }

    if (reinitialize) {
        value = reduce(state);
    }

    Node::propagate(state);
}

template <>
void ReduceNode<functional::min<double>>::propagate(State& state) const {
    auto ptr = data_ptr<ReduceNodeData<functional::min<double>>>(state);

    auto& value = ptr->values.value;

    bool reinitialize = false;
    for (const Update& update : array_ptr_->diff(state)) {
        if (update.removed()) {
            if (update.old <= value) {
                // uh oh, we may have just removed the known min. So we need to
                // recalculate everything from scratch
                reinitialize = true;
                break;
            }
        } else if (update.placed() || update.value < update.old) {
            // we added a new value, or decreased an existing one
            value = std::min(update.value, value);
        } else if (update.old == value && update.old != update.value) {
            // we potentially made the current min larger
            assert(update.value > update.old);
            reinitialize = true;
            break;
        }
    }

    if (reinitialize) {
        value = reduce(state);
    }

    Node::propagate(state);
}

template <>
void ReduceNode<std::multiplies<double>>::propagate(State& state) const {
    auto ptr = data_ptr<ReduceNodeData<std::multiplies<double>>>(state);

    RunningProduct& product = ptr->extra.product;

    for (const Update& update : array_ptr_->diff(state)) {
        if (update.placed()) {
            product *= update.value;
        } else if (update.removed()) {
            product /= update.old;
        } else {
            product.multiply_divide(update.value, update.old);
        }
    }

    ptr->values.value = static_cast<double>(product);
    if (ptr->values.value != ptr->values.old) Node::propagate(state);
}

template <>
void ReduceNode<std::plus<double>>::propagate(State& state) const {
    auto ptr = data_ptr<ReduceNodeData<std::plus<double>>>(state);

    // todo: consider kahan summation

    auto& value = ptr->values.value;

    for (const Update& update : array_ptr_->diff(state)) {
        if (update.placed()) {
            value += update.value;
        } else if (update.removed()) {
            value -= update.old;
        } else {
            value += update.value - update.old;
        }
    }

    Node::propagate(state);
}

template <class BinaryOp>
double ReduceNode<BinaryOp>::reduce(const State& state) const {
    auto start = array_ptr_->begin(state);
    const auto end = array_ptr_->end(state);

    double init;

    if (this->init.has_value()) {
        init = this->init.value();
    } else {
        // if there is no init, we use the first value in the array as the init
        // we should only be here if the array is not dynamic and if it has at
        // least one entry
        assert(!array_ptr_->dynamic());
        assert(array_ptr_->size(state) >= 1);
        assert(start != end);

        init = *start;
        ++start;
    }

    return std::reduce(start, end, init, BinaryOp());
}

template <class BinaryOp>
void ReduceNode<BinaryOp>::revert(State& state) const {
    data_ptr<ReduceNodeData<BinaryOp>>(state)->revert();
}

// Uncommented are the tested specializations
template class ReduceNode<functional::max<double>>;
template class ReduceNode<functional::min<double>>;
template class ReduceNode<std::logical_and<double>>;
template class ReduceNode<std::logical_or<double>>;
template class ReduceNode<std::multiplies<double>>;
template class ReduceNode<std::plus<double>>;

}  // namespace dwave::optimization

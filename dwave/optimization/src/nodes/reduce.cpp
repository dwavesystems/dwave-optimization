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
#include <iterator>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "../functional_.hpp"

namespace dwave::optimization {

template <class BinaryOp>
class ReduceNodeData : public NodeStateData {
 public:
    // Some type information about our function
    using ufunc_type = functional_::std_to_ufunc<BinaryOp>::type;
    using reduction_type = ufunc_type::reduction_type;
    using result_type = ufunc_type::result_type;

    enum class ReductionFlag {
        invalid,    // no longer holds the correct value, the whole reduction must be recalculated
        unchanged,  // no change since the last propagation
        updated,    // has been updated and holds the correct value
    };

    // Given a vector of reductions, construct the state of the array.
    ReduceNodeData(std::vector<reduction_type>&& reductions)
            : reductions_(std::move(reductions)),
              flags_(reductions_.size(), ReductionFlag::unchanged) {
        buffer_.reserve(reductions_.size());
        for (const auto& reduction : reductions_)
            buffer_.emplace_back(static_cast<result_type>(reduction));
    }

    ReduceNodeData(std::vector<reduction_type>&& reductions, std::span<const ssize_t> shape)
            : ReduceNodeData(std::move(reductions)) {
        shape_info_.emplace(shape_info_type(reductions_.size(), shape));
    }

    // Add a value to the reduction at the given `index`.
    // `index` can be one-past-the-end in which case a new reduction
    // is appended.
    void add_to_reduction(ssize_t index, double value) {
        // Make sure this method is defined for our ufunc
        static_assert(decltype(ufunc)::associative);

        // Check that we're not trying to add a removal
        assert(not std::isnan(value));

        // And that we're within bounds. One past the end is OK
        assert(0 <= index and static_cast<std::size_t>(index) <= reductions_.size());

        if (static_cast<std::size_t>(index) == reductions_.size()) {
            // If we're one-past-end then it's an append (which even rhymes!)
            append_reduction(value);
        } else {
            // Otherwise an update
            update_reduction_(index, ufunc(reductions_[index], value));
        }
    }

    // Append a new reduction value, thereby extending the length of the array
    void append_reduction(reduction_type reduction) {
        reductions_.emplace_back(reduction);
        flags_.emplace_back(ReductionFlag::updated);
        // in this case there is no old value to save to reductions_diff_
    }

    // Pointer to the beginning of the buffer. `update_reduction()` must be
    // called to incorporate changes!
    const double* buff() const { return buffer_.data(); }

    // Commit the changes to the buffer
    void commit() {
        // Clear the diffs
        diff_.clear();
        reductions_diff_.clear();

        // Update the shape_info to match the current state
        if (shape_info_) shape_info_->commit();

        // A few final consistency checks
        assert(flags_.size() == reductions_.size());
        assert(buffer_.size() == reductions_.size());
        assert(!shape_info_ or shape_info_->previous_size == static_cast<ssize_t>(flags_.size()));
        assert(std::ranges::equal(buffer_, reductions_,
                                  [](const double& lhs, const reduction_type& rhs) {
                                      return lhs == static_cast<result_type>(rhs);
                                  }));
        assert(std::ranges::all_of(flags_, [](const ReductionFlag& flag) {
            return flag == ReductionFlag::unchanged;
        }));
    }

    // The current buffer diff. `update_reduction()` must be called to
    // incorporate changes!
    std::span<const Update> diff() const { return diff_; }

    // Remove a reduction
    void pop_reduction() {
        assert(reductions_.size() > 0);
        assert(reductions_.size() == flags_.size());

        // The index we're popping
        const ssize_t index = reductions_.size() - 1;

        // If we haven't already changed the value at this index, we go ahead
        // and save the old value for later reverting
        if (flags_[index] == ReductionFlag::unchanged) {
            reductions_diff_.emplace_back(index, reductions_[index]);
        }

        reductions_.pop_back();
        flags_.pop_back();
    }

    // Synchronize the buffer with the reductions array in preparation for
    // propagation
    void prepare_diff(std::function<reduction_type(ssize_t)> reduce) {
        while (buffer_.size() > reductions_.size()) {
            diff_.emplace_back(Update::removal(buffer_.size() - 1, buffer_.back()));
            buffer_.pop_back();
        }
        while (buffer_.size() < reductions_.size()) {
            // we put Update::nothing here so that subequent Updates will be
            // interpreted (correctly) as placements
            buffer_.emplace_back(Update::nothing);
        }

        for (ssize_t index = 0, size = buffer_.size(); index < size; ++index) {
            double value;

            switch (flags_[index]) {
                case ReductionFlag::invalid:
                    reductions_[index] = reduce(index);
                    [[fallthrough]];
                case ReductionFlag::updated:
                    value = static_cast<result_type>(reductions_[index]);
                    if (buffer_[index] != value) {
                        diff_.emplace_back(index, buffer_[index], value);
                        buffer_[index] = value;
                    }
                    flags_[index] = ReductionFlag::unchanged;
                    break;
                case ReductionFlag::unchanged:
                    // Nothing to do!
                    break;
            }
        }

        if (shape_info_) shape_info_->update(buffer_.size());
    }

    // Remove a value from a reduction
    void remove_from_reduction(ssize_t index, double value) {
        // Otherwise this function implementation doesn't make sense
        static_assert(decltype(ufunc)::associative);
        static_assert(decltype(ufunc)::invertible);

        assert(not std::isnan(value));  // Not trying to remove a placement
        assert(0 <= index and static_cast<std::size_t>(index) < buffer_.size());

        // Then try to remove the value
        auto inverse = ufunc.inverse(reductions_[index], value);
        if (not inverse.has_value()) {
            // We broke the reduction, so mark it as invalid and return
            flags_[index] = ReductionFlag::invalid;
            return;
        }

        update_reduction_(index, std::move(inverse.value()));
    }

    // Revert any changes that have been made to the buffer
    void revert() {
        // First update everything to the correct size
        if (shape_info_) {
            // Note the 0s are arbitrary, they'll be overwritten later
            reductions_.resize(shape_info_->previous_size, reduction_type(0));
            flags_.resize(shape_info_->previous_size, ReductionFlag::unchanged);
            buffer_.resize(shape_info_->previous_size, 0);

            shape_info_->revert();
        }

        // Next revert the values in the buffer and reductions
        for (const auto& [index, old_reduction] : reductions_diff_ | std::views::reverse) {
            assert(0 <= index and static_cast<std::size_t>(index) < reductions_.size());
            reductions_[index] = old_reduction;
            buffer_[index] = static_cast<result_type>(reductions_[index]);
        }

        // Clear the diff
        reductions_diff_.clear();
        diff_.clear();

        // A few final consistency checks
        assert(flags_.size() == reductions_.size());
        assert(buffer_.size() == reductions_.size());
        assert(!shape_info_ or shape_info_->previous_size == static_cast<ssize_t>(flags_.size()));
        assert(std::ranges::equal(buffer_, reductions_,
                                  [](const double& lhs, const reduction_type& rhs) {
                                      return lhs == static_cast<result_type>(rhs);
                                  }));
        assert(std::ranges::all_of(flags_, [](const ReductionFlag& flag) {
            return flag == ReductionFlag::unchanged;
        }));
    }

    // The current shape of the array.
    std::span<const ssize_t> shape() const {
        assert(shape_info_);
        return shape_info_->shape;
    }

    // The current size of the array
    ssize_t size() const { return buffer_.size(); }

    // The change in size of the array
    ssize_t size_diff() const {
        if (!shape_info_) return 0;
        return static_cast<ssize_t>(buffer_.size()) - shape_info_->previous_size;
    }

    // Update a reduction by removing `from` and then adding `to`.
    void update_reduction(ssize_t index, double from, double to) {
        // Otherwise this function implementation doesn't make sense
        static_assert(decltype(ufunc)::associative);
        static_assert(decltype(ufunc)::invertible);

        // Should always be true if we've implemented things correctly
        assert(flags_.size() == reductions_.size());

        // Some input checking
        assert(not std::isnan(from));  // Not trying to remove a removal
        assert(not std::isnan(to));    // Not trying to add a placement
        assert(0 <= index and static_cast<std::size_t>(index) < reductions_.size());

        // If we've already marked this location as invalid, then nothing to do
        if (flags_[index] == ReductionFlag::invalid) return;

        // Likewise if from == to then no point doing any recalculations
        if (from == to) return;

        // Get the reduction we're potentially updating, as a copy
        auto reduction = reductions_[index];

        // Add the new `to` value to the reduction.
        // We do this first because it results in fewer indices marked invalid
        // in some common cases.
        // E.g., for `MaxNode`, if `reduction == 10`, `from == 10`, and `to == 11`
        // then if we invert first then the inversion will fail, but if we apply
        // first it wont! This can theoretically happen in the other direction as
        // well but we don't have any operations right now that work that way.
        reduction = ufunc(std::move(reduction), to);

        // Then try to remove the `from` value we previously included in the
        // reduction
        auto inverse = ufunc.inverse(std::move(reduction), from);
        if (not inverse.has_value()) {
            // if the location has not been previously updated, then save the old
            // value and mark it as invalid
            if (flags_[index] == ReductionFlag::unchanged) {
                reductions_diff_.emplace_back(index, reductions_[index]);
            }
            flags_[index] = ReductionFlag::invalid;            
            return;
        }
        reduction = std::move(inverse.value());

        return update_reduction_(index, std::move(reduction));
    }

 private:
    void update_reduction_(ssize_t index, reduction_type reduction) {
        assert(flags_.size() == reductions_.size());
        assert(0 <= index and static_cast<std::size_t>(index) < reductions_.size());

        // No change so don't save anything
        if (reductions_[index] == reduction) return;

        switch (flags_[index]) {
            case ReductionFlag::invalid:
                // it's already invalid, so nothing to do
                break;
            case ReductionFlag::unchanged:
                // We haven't previously made any changes to this location, so we
                // save the old value, update with the new value, and then mark
                // ourselves as updated
                reductions_diff_.emplace_back(index, reductions_[index]);
                reductions_[index] = std::move(reduction);
                flags_[index] = ReductionFlag::updated;
                break;
            case ReductionFlag::updated:
                // We've already updated this index once, so no need to save the old value
                reductions_[index] = std::move(reduction);
                break;
        };
    }

    ufunc_type ufunc;

    // A vector of reductions. `reduction_type` might be a double, or some other
    // type that tracks the information needed for us to invert reductions.
    std::vector<reduction_type> reductions_;

    // For each reduction, we also track a flag indicating whether the reduction
    // has changed and/or whether it needs to be recalculated.
    std::vector<ReductionFlag> flags_;

    // The buffer we expose to the other nodes
    std::vector<double> buffer_;

    // We track the original states of the reduction to support `revert()`
    std::vector<std::tuple<ssize_t, reduction_type>> reductions_diff_ = {};

    // The diff we expose to the user
    std::vector<Update> diff_ = {};

    // Dynamic nodes need us to hold some information about their shape etc
    // so we use this struct to (optionally) track that info.
    struct shape_info_type {
        shape_info_type() = delete;

        shape_info_type(ssize_t size, std::span<const ssize_t> shape)
                : shape(shape.begin(), shape.end()),
                  size_divisor(std::reduce(shape.begin() + 1, shape.end(), 1,
                                           std::multiplies<ssize_t>())),
                  previous_size(size) {
            this->shape[0] = size / size_divisor;
        }

        void commit() {
            assert(shape.size());
            previous_size = shape[0] * size_divisor;
        }

        void revert() {
            assert(shape.size());
            assert(previous_size % size_divisor == 0);
            shape[0] = previous_size / size_divisor;
        }

        void update(ssize_t size) {
            assert(shape.size());
            assert(size % size_divisor == 0);
            shape[0] = size / size_divisor;
        }

        // The current shape of the array
        std::vector<ssize_t> shape;

        // The relationship between the size of the buffer and the size of the
        // first dimension.
        const ssize_t size_divisor;

        // The previous size of the array.
        ssize_t previous_size;
    };
    std::optional<shape_info_type> shape_info_;
};

// Drop the given axes from the range. Assumes axes is sorted and unique
std::vector<ssize_t> drop_axes(std::ranges::sized_range auto&& range,
                               std::span<const ssize_t> axes) {
    assert(std::ranges::is_sorted(axes) && "axes must be sorted");
    assert(std::ranges::adjacent_find(axes) == axes.end() && "axes must be unique");

    std::vector<ssize_t> out;

    auto axes_it = axes.begin();
    const auto axes_end = axes.end();
    auto range_it = std::ranges::begin(range);
    for (ssize_t dim = 0, stop = std::ranges::size(range); dim < stop; ++dim, ++range_it) {
        if (axes_it != axes_end && dim == *axes_it) {
            // skipped axis, this relies on axes being sorted and unique
            ++axes_it;
        } else {
            // kept axis
            out.emplace_back(*range_it);
        }
    }

    return out;
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
        if (dim < -ndim or ndim <= dim) {
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
    assert(std::ranges::adjacent_find(axes) == axes.end());  // it's unique

    // Our ufunc will determine our bounds
    typename functional_::std_to_ufunc<BinaryOp>::type ufunc;
    static_assert(decltype(ufunc)::associative);  // otherwise this doesn't make sense
    static_assert(decltype(ufunc)::commutative);  // otherwise this doesn't make sense

    // Get the bounds for our predecessor
    const auto array_bounds = ValuesInfo(array_ptr);

    // The easy case is that the size of each reduction is fixed.
    if (not array_ptr->dynamic() or (axes.size() > 0 and axes[0] != 0)) {
        const ssize_t reduction_size =
                axes.size() ? product(keep_axes(array_ptr->shape(), axes)) : array_ptr->size();
        assert(reduction_size >= 0);

        if (reduction_size and initial) {
            // cast initial to the correct output type
            auto init = typename decltype(ufunc)::result_type(*initial);

            return ufunc.result_bounds(
                    ufunc.result_bounds(array_bounds, reduction_size),  // from reducing the array
                    ValuesInfo(init, init, is_integer(init))            // from initial
            );
        } else if (reduction_size) {
            return ufunc.result_bounds(array_bounds, reduction_size);
        } else if (initial) {
            // cast initial to the correct output type
            auto init = typename decltype(ufunc)::result_type(*initial);

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
        // NumPy error message: ValueError: zero-size array to reduction operation maximum which has
        // no identity
        throw std::invalid_argument(
                "cannot perform a reduction operation with no identity on an array that might be "
                "empty");
    }

    //
    // Ok, now that we (maybe) know the min/max size of our reduction space. we can start fiddling
    // with bounds.
    //

    if (max_size.has_value() and *max_size == 0) {
        // If our array is always empty (can happen that arrays think they are dynamic when
        // they are empty by construction), then our initial value is our bounds.
        assert(initial.has_value());  // previous check should have caught this
        auto init = typename decltype(ufunc)::result_type(*initial);
        return ValuesInfo(init, init, is_integer(init));
    }

    // Get the bounds at each of the smallest and largest sizes we can be
    ValuesInfo min_bounds = ufunc.result_bounds(array_bounds, std::max<ssize_t>(min_size, 1));
    ValuesInfo max_bounds = max_size ? ufunc.result_bounds(array_bounds, *max_size)
                                     : ufunc.result_bounds(array_bounds, functional_::limit_type());

    // If there is not initial value, then our bounds are just the union of min_/max_bounds
    if (!initial.has_value()) return min_bounds | max_bounds;

    // Otherwise we need to account for the initial value in the bounds
    auto init = typename decltype(ufunc)::result_type(*initial);  // cast to bool etc
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
    add_predecessor(array_ptr);
}

template <class BinaryOp>
ReduceNode<BinaryOp>::ReduceNode(ArrayNode* array_ptr, std::initializer_list<ssize_t> axes,
                                 std::optional<double> initial)
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
ssize_t ReduceNode<BinaryOp>::convert_predecessor_index_(ssize_t index) const {
    assert(index >= 0 && "index must be non-negative");  // NumPy raises here so we assert
    assert(std::ranges::is_sorted(axes_) && "axes must be sorted");
    assert(std::ranges::adjacent_find(axes_) == axes_.end() && "axes must be unique");

    // Predecessor's shape
    const std::span<const ssize_t> array_shape = array_ptr_->shape();
    if (array_shape.empty()) {
        assert(index == 0);  // otherwise it's out-of-bounds
        return index;
    }

    // ReduceNode and predecessors' shape iterator, initialized to the last
    // element in their respective ranges.
    const std::span<const ssize_t> reduce_shape = this->shape();
    auto reduce_shape_it = std::ranges::end(reduce_shape) - 1;
    auto array_shape_it = std::ranges::end(array_shape) - 1;

    // `axes_` defines the axes *excluded* from the reduction.
    auto axes_it = axes_.end() - 1;
    const auto axes_stop = axes_.begin() - 1;

    ssize_t reduce_node_flat_index = 0;
    ssize_t multiplier = 1;

    // We traverse the dimensions (and shape) of the predecessor in reverse
    // order up to and *not* including the 0th dimension.
    for (ssize_t dim = std::ranges::size(array_shape) - 1; dim > 0; --dim, --array_shape_it) {
        assert(array_shape_it != array_shape.begin() - 1 &&
               "Bad predecessor array shape iterator in ReduceNode");
        // Contribution of `index` in the given dimension `dim`
        const ssize_t multidimensional_index = index % *array_shape_it;
        index /= *array_shape_it;

        assert(0 <= dim && "all dimensions except the first must be non-negative");
        // NumPy supports "clip" and "wrap" which we could add support for
        // but for now let's just assert.
        assert(0 <= multidimensional_index && (!dim || multidimensional_index < *array_shape_it));

        // Handle included / excluded axes in reduction
        if (axes_it != axes_stop && dim == *axes_it) {
            assert(axes_it != axes_stop && "Bad axes_ iterator in ReduceNode");
            // skipped axis, this relies on axes being sorted and unique
            --axes_it;
        } else {
            assert(reduce_shape_it != reduce_shape.begin() - 1 &&
                   "Bad reduce node shape iterator in ReduceNode");
            // kept axis, determine the contribution of
            // `multidimensional_index` to the ReduceNode's flat index
            reduce_node_flat_index += multidimensional_index * multiplier;
            // this contribution is defined by the ReduceNode's shape
            multiplier *= *reduce_shape_it;
            --reduce_shape_it;
        }
    }

    // handle the contribution of the 0th dimension if it is included in the
    // reduction
    if (axes_it == axes_stop) {
        // Check if the index is out of bounds for non-dynamic shapes and assert
        assert(array_shape[0] < 0 || index < array_shape[0]);
        reduce_node_flat_index += index * multiplier;
    }

    return reduce_node_flat_index;
}

template <class BinaryOp>
void ReduceNode<BinaryOp>::propagate(State& state) const {
    auto* const state_ptr = data_ptr<ReduceNodeData<BinaryOp>>(state);

    // We are reducing over all axes, so this is nice and simple
    if (axes_.empty() or axes_.size() == static_cast<std::size_t>(array_ptr_->ndim())) {
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

    if (this->dynamic() and array_ptr_->size_diff(state) > 0 and initial) {
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
        ssize_t reduction_index = convert_predecessor_index_(update.index);
        assert(ravel_multi_index(drop_axes(unravel_index(update.index, array_ptr_->shape()), axes_),
                                 this->shape()) == reduction_index &&
               "Incorrect predecessor index conversion in ReduceNode");

        if (update.placed()) {
            state_ptr->add_to_reduction(reduction_index, update.value);
        } else if (update.removed()) {
            state_ptr->remove_from_reduction(reduction_index, update.old);
        } else {
            state_ptr->update_reduction(reduction_index, update.old, update.value);
        }
    }

    if (this->dynamic() and array_ptr_->size_diff(state) < 0) {
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
    typename functional_::std_to_ufunc<BinaryOp>::type ufunc;

    // Everything is being reduced
    if (axes_.empty() or axes_.size() == static_cast<std::size_t>(array_ptr_->ndim())) {
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

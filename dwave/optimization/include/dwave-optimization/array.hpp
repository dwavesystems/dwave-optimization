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

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "dwave-optimization/common.hpp"
#include "dwave-optimization/fraction.hpp"
#include "dwave-optimization/iterators.hpp"
#include "dwave-optimization/state.hpp"
#include "dwave-optimization/typing.hpp"

namespace dwave::optimization {

// Note: we use ssize_t throughout here because Py_ssize_t is defined to take
// that value when it's available. See https://peps.python.org/pep-0353/

class Array;

/// Information about where a dynamic array gets its size.
///
/// SizeInfo allows arrays to specify their own size as a (bounded) linear function of another
/// dynamic array's size. This allows subsequent nodes in the graph to make guarantees about their
/// predecessors' sizes, so that they can make guarantees about correctness at runtime. Since almost
/// all array nodes have a size that is a simple linear transformation of their predecessors' size
/// (which itself may be linear function of previous nodes' sizes), SizeInfo is able to capture most
/// of their behavior.
///
/// SizeInfo has five members: a pointer to an Array, a multiplier, an offset, a minimum, and a
/// maximum. It represents the following linear expression:
///
///     clamp(ceil(multiplier * array_ptr->size() + offset), min, max)
///
/// Note that multiplier and offset are represented as fractions (see
/// `dwave::optimization::fraction`).
struct SizeInfo {
    SizeInfo() : SizeInfo(0) {}
    explicit SizeInfo(const std::integral auto size)
            : array_ptr(nullptr), multiplier(0), offset(size), min(size), max(size) {
        assert(size >= 0);
    }
    explicit SizeInfo(const Array* array_ptr) : SizeInfo(array_ptr, std::nullopt, std::nullopt) {}
    SizeInfo(const Array* array_ptr, std::optional<ssize_t> min, std::optional<ssize_t> max);

    friend bool operator==(const SizeInfo& lhs, const std::integral auto rhs) {
        return lhs.multiplier == 0 && lhs.offset == rhs;
    }
    friend bool operator==(const SizeInfo& lhs, const fraction rhs) {
        return lhs.multiplier == 0 && lhs.offset == rhs;
    }
    bool operator==(const SizeInfo& other) const;

    // SizeInfos are printable
    friend std::ostream& operator<<(std::ostream& os, const SizeInfo& sizeinfo);

    SizeInfo substitute(ssize_t max_depth = 1) const;

    const Array* array_ptr;

    fraction multiplier;
    fraction offset;

    std::optional<ssize_t> min;
    std::optional<ssize_t> max;
};

/// Struct for the common use case of saving statistics about an ArrayNode's output values
struct ValuesInfo {
    ValuesInfo() = delete;
    ValuesInfo(double min, double max, bool integral) : min(min), max(max), integral(integral) {}
    ValuesInfo(std::pair<double, double> minmax, bool integral)
            : min(minmax.first), max(minmax.second), integral(integral) {}
    /// Copy the min/max/integral from the array
    ValuesInfo(const Array* array_ptr);

    /// These constructors take the min of the mins, etc for all the arrays
    ValuesInfo(std::initializer_list<const Array*> array_ptrs);

    // Unfortunately it seems we still need this span constructor for GCC11 which doesn't
    // like a vector being passed to the viewable_range constructor
    ValuesInfo(std::span<const Array* const> array_ptrs);

    template <std::ranges::viewable_range R>
    ValuesInfo(R&& array_ptrs);

    static ValuesInfo logical_output() { return {false, true, true}; };

    double min;
    double max;
    bool integral;
};

// A slice represents a set of indices specified by range(start, stop, step).
struct Slice {
    constexpr Slice() noexcept : Slice(std::nullopt, std::nullopt, std::nullopt) {}
    explicit constexpr Slice(std::optional<ssize_t> stop) noexcept
            : Slice(std::nullopt, stop, std::nullopt) {}
    constexpr Slice(std::optional<ssize_t> start, std::optional<ssize_t> stop,
                    std::optional<ssize_t> step = std::nullopt) {
        constexpr ssize_t MAX = std::numeric_limits<ssize_t>::max();
        constexpr ssize_t MIN = std::numeric_limits<ssize_t>::lowest();

        this->step = step.value_or(1);

        if (this->step == 0) throw std::invalid_argument("slice step cannot be zero");

        this->start = start.value_or(this->step < 0 ? MAX : 0);
        this->stop = stop.value_or(this->step < 0 ? MIN : MAX);
    }

    // Two slices are equal if they have exactly the same values
    constexpr bool operator==(const Slice& other) const noexcept {
        return (start == other.start && stop == other.stop && step == other.step);
    }

    // Slices are printable
    friend std::ostream& operator<<(std::ostream& os, const Slice& slice);

    // Test whether the slice is equal to an empty one
    constexpr bool empty() const noexcept { return *this == Slice(); }

    // Fit a slice while checking for validity.
    constexpr Slice fit_at(const std::integral auto size) const {
        // error messages are chosen to match Python's. Though they use "length" for some reason
        if (size < 0) throw std::invalid_argument("size should not be negative");
        if (!step) throw std::invalid_argument("slice step cannot be zero");
        return fit(size);
    }

    // Assuming a sequence of length size, calculate the start, stop, and step.
    // Out of bound indices are clipped.
    constexpr Slice fit(const std::integral auto size) const noexcept {
        ssize_t start = this->start;
        ssize_t stop = this->stop;
        ssize_t step = this->step;

        if (start < 0) {
            // handle negative index once
            start += size;

            // still out of range, so clip
            if (start < 0) {
                start = step < 0 ? -1 : 0;
            }
        } else if (start >= size) {
            // clip
            start = step < 0 ? size - 1 : size;
        }

        if (stop < 0) {
            // handle negative index once
            stop += size;

            // still out of range, so clip
            if (stop < 0) {
                stop = step < 0 ? -1 : 0;
            }
        } else if (stop >= size) {
            // clip
            stop = step < 0 ? size - 1 : size;
        }

        return Slice(start, stop, step);
    }

    // Return the length of the slice.
    // Unlike most places in dwave-optimization we use unsigned integer here
    // because for unfitted slices these values can exceed the range of ssize_t
    constexpr std::size_t size() const noexcept {
        if (step < 0) {
            if (stop < start) {
                if (start > 0) {
                    // we might exceed the range of ssize_t
                    return (static_cast<std::size_t>(start) - stop - 1) / (-step) + 1;
                }
                // else both are negative so in range for just ssize_t
                return (start - stop - 1) / (-step) + 1;
            }
        } else {
            if (start < stop) {
                if (stop > 0) {
                    // we might exceed the range of ssize_t
                    return (static_cast<std::size_t>(stop) - start - 1) / step + 1;
                }
                // else both are negative so in range for just ssize_t
                return (stop - start - 1) / step + 1;
            }
        }
        return 0;
    }

    ssize_t start;
    ssize_t stop;
    ssize_t step;
};

struct Update {
    constexpr Update(ssize_t index, double old, double value)
            : index(index), old(old), value(value) {}

    // Factory function to create an Update representing a new value added to
    // the array as part of a resize. In this case the old value is encoded as
    // a NaN.
    static constexpr Update placement(ssize_t index, double value) {
        return Update(index, nothing, value);
    }

    // Factory function to create an Update representing a value removed from
    // the array as part of a resize. In this case the new value is encoded as
    // a NaN.
    static constexpr Update removal(ssize_t index, double old) {
        return Update(index, old, nothing);
    }

    // We want to be able to stably sort updates based on the index.
    friend constexpr auto operator<=>(const Update& lhs, const Update& rhs) {
        return lhs.index <=> rhs.index;
    }
    friend constexpr bool operator==(const Update& lhs, const Update& rhs) noexcept {
        return (lhs.index == rhs.index &&
                (lhs.old == rhs.old || (std::isnan(lhs.old) && std::isnan(rhs.old))) &&
                (lhs.value == rhs.value || (std::isnan(lhs.value) && std::isnan(rhs.value))));
    }

    friend std::ostream& operator<<(std::ostream& os, const Update& update);

    // Whether the given index was placed when the state was grown
    bool placed() const {
        // We'd like to constexpr this, but std::isnan is not constexpr in C++20
        return std::isnan(old);
    }

    // Whether the given index was removed when the state was resized
    bool removed() const {
        // We'd like to constexpr this, but std::isnan is not constexpr in C++20
        return std::isnan(value);
    }

    // Returns true if the Update's goes from nothing to nothing (index can be anything)
    bool null() const { return std::isnan(old) && std::isnan(value); }

    double old_or(double val) const { return std::isnan(old) ? val : old; }

    double value_or(double val) const { return std::isnan(value) ? val : value; }

    // Return true if the update does nothing - that is old and value are the same.
    bool identity() const { return null() || old == value; }

    // Use NaN to represent the "nothing" value used in placements/removals
    static constexpr double nothing = std::numeric_limits<double>::signaling_NaN();

    ssize_t index;  // The index of the updated value in the flattened array.
    double old;     // The old value
    double value;   // The new/current value.
};

/// An array.
///
/// This interface is designed to work with
/// [Python's buffer protocol](https://docs.python.org/3/c-api/buffer.html).
///
/// However, unlike Python's buffer protocol, Array supports state-dependent
/// size; arrays are allowed to change their size based on the state of
/// decision variables.
/// Arrays are permitted to extend and contract only along axis 0.
/// Such operations are equivalent to growing or shrinking a buffer, with no
/// insertions being needed.
/// Arrays signal a state-dependent size by returning negative values for
/// Array::size() and in the first element of Array::shape(). For convenience,
/// the Array::dynamic() method is provided.
class Array {
 public:
    /// A std::random_access_iterator over the values in the array.
    using iterator = BufferIterator<double, double, false>;

    /// A std::random_access_iterator over the values in the array.
    using const_iterator = BufferIterator<double, double, true>;

    template <class T>
    using cache_type = std::unordered_map<const Array*, T>;

    template <class T>
    using optional_cache_type = std::optional<std::reference_wrapper<cache_type<T>>>;

    using View = std::ranges::subrange<const_iterator>;

    /// Constant used to signal that the size is based on the state.
    static constexpr ssize_t DYNAMIC_SIZE = -1;

    // Buffer protocol methods ************************************************

    /// A pointer to the start of the logical structure described by the buffer
    /// fields. This can be any location within the underlying physical memory
    /// block of the exporter. For example, with negative strides the value may
    /// point to the end of the memory block.
    /// For contiguous arrays, the value points to the beginning of the memory
    /// block.
    virtual double const* buff(const State& state) const = 0;

    /// For contiguous arrays, this is the length of the underlying memory block.
    /// For non-contiguous arrays, it is the length that the logical structure
    /// would have if it were copied to a contiguous representation.
    /// If the array is dynamic, returns Array::DYNAMIC_SIZE.
    ssize_t len() const { return (size() >= 0) ? size() * itemsize() : DYNAMIC_SIZE; }

    /// For contiguous arrays, this is the length of the underlying memory block.
    /// For non-contiguous arrays, it is the length that the logical structure
    /// would have if it were copied to a contiguous representation.
    /// Always returns a positive number.
    ssize_t len(const State& state) const { return size(state) * itemsize(); }

    /// Exactly `sizeof(double)`.
    constexpr ssize_t itemsize() const { return sizeof(double); }

    /// "d" for double; see https://docs.python.org/3/library/struct.html
    const std::string& format() const;

    /// The number of dimensions the memory represents as an n-dimensional array.
    /// If 0, the buffer points to a single item, which represents a scalar.
    virtual ssize_t ndim() const = 0;

    /// An array of Array::ndim() length indicating the shape of the buffer contents
    /// as an n-dimensional array.
    /// Note that this is in terms of the actual type rather than in terms of
    /// number of bytes.
    virtual std::span<const ssize_t> shape(const State& state) const = 0;

    /// A span of Array::ndim() length indicating the shape of the buffer contents
    /// as an n-dimensional array.
    /// Note that this is in terms of the actual type rather than in terms of
    /// number of bytes.
    /// If the shape is state-dependent, the first value in shape is
    /// Array::DYNAMIC_SIZE.
    virtual std::span<const ssize_t> shape() const = 0;

    /// A span of length Array::ndim() giving the number of bytes to step to get to a
    /// new element in each dimension.
    virtual std::span<const ssize_t> strides() const = 0;

    // Interface methods ******************************************************

    /// Return an iterator to the beginning of the array.
    const_iterator begin(const State& state) const {
        if (contiguous()) return const_iterator(buff(state));
        return const_iterator(buff(state), ndim(), shape().data(), strides().data());
    }

    /// Return an iterator to the end of the array.
    const_iterator end(const State& state) const { return this->begin(state) + this->size(state); }

    /// Return a container-like view over the array.
    const View view(const State& state) const { return View(begin(state), end(state)); }

    /// The number of doubles in the flattened array.
    virtual ssize_t size() const = 0;

    /// The number of doubles in the flattened array.
    /// If the size is dependent on the state, returns Array::DYNAMIC_SIZE.
    virtual ssize_t size(const State& state) const = 0;

    /// Information about how the size of a node is calculated. See SizeInfo.
    virtual SizeInfo sizeinfo() const { return dynamic() ? SizeInfo(this) : SizeInfo(size()); }

    /// The minimum value that elements in the array may take.
    virtual double min() const = 0;

    /// The maximum value that elements in the array may take.
    virtual double max() const = 0;

    /// Whether the values in the array can be interpreted as integers.
    virtual bool integral() const { return false; }

    /// Whether the values in the array can be interpreted as booleans.
    bool logical() const { return integral() && min() >= 0 && max() <= 1; }

    /// Whether the data is stored contiguously.
    virtual bool contiguous() const = 0;

    /// Whether the size of the array is state-dependent or not.
    bool dynamic() const { return size() < 0; }

    // Update signaling *******************************************************

    /// A list of the array indices that have been updated and their previous values.
    virtual std::span<const Update> diff(const State& state) const = 0;

    /// The change in the array's size.
    virtual ssize_t size_diff(const State& state) const {
        assert(size() >= 0 &&
               "size_diff(const State&) must be overloaded if the size is state-dependent");
        return 0;
    }

 protected:
    // Some utility methods that might be useful to subclasses

    // Determine whether a given shape/strides define a contiguous array or not.
    static bool is_contiguous(const ssize_t ndim, const ssize_t* shape, const ssize_t* strides) {
        assert(ndim >= 0);
        if (!ndim) return true;  // scalars are contiguous

        ssize_t sd = sizeof(double);
        for (ssize_t i = ndim - 1; i >= 0; --i) {
            const ssize_t dim = shape[i];

            // This method is fine with state-dependent shape/size under the
            // assumption that we only ever allow it on the 0-axis.
            assert(dim >= 0 || i == 0);

            // If dim == 0 then we're contiguous because we're empty
            if (!dim) return true;

            if (dim != 1 && strides[i] != sd) return false;
            sd *= dim;
        }

        return true;
    }

    // Determine the size by the shape. For a node with a fixed size, it is simply
    // the product of the shape.
    // Expects the shape to be stored in a C-style array of length ndim.
    static ssize_t shape_to_size(const ssize_t ndim, const ssize_t* shape) noexcept {
        if (ndim <= 0) return 1;
        if (shape[0] < 0) return DYNAMIC_SIZE;
        return std::reduce(shape, shape + ndim, 1, std::multiplies<ssize_t>());
    }

    static ssize_t shape_to_size(const std::span<const ssize_t> shape) noexcept {
        return shape_to_size(shape.size(), shape.data());
    }

    // Determine the strides from the shape.
    // Assumes itemsize = sizeof(double).
    // Expects the shape to be stored in a C-style array of length ndim.
    // Returns the strides as a C-style array of length ndim managed by a unique_ptr.
    static std::unique_ptr<ssize_t[]> shape_to_strides(const ssize_t ndim,
                                                       const ssize_t* shape) noexcept {
        if (ndim <= 0) return nullptr;
        auto strides = std::make_unique<ssize_t[]>(ndim);
        // otherwise strides are a function of the shape
        strides[ndim - 1] = sizeof(double);
        for (auto i = ndim - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    }
};

// A convenience class for creating contiguous Arrays.
// Meant to be used for nodes that have an array output.
template <class Base>
class ArrayOutputMixin : public Base {
 public:
    // 1D array with n elements. -1 will create a 1D dynamic array.
    explicit ArrayOutputMixin(ssize_t n) : ArrayOutputMixin({n}) {}

    explicit ArrayOutputMixin(std::initializer_list<ssize_t> shape)
            : ArrayOutputMixin(std::span(shape)) {}

    template <std::ranges::sized_range Range>
    explicit ArrayOutputMixin(Range&& shape)
            : ndim_(shape.size()), shape_(make_shape(std::forward<Range>(shape))) {}

    ssize_t ndim() const noexcept final { return ndim_; }

    ssize_t size() const noexcept final { return size_; }
    ssize_t size(const State& state) const override {
        assert(size() >= 0 &&
               "size(const State&) must be overloaded if the size is state-dependent");
        return size();
    }

    std::span<const ssize_t> shape() const final { return std::span(shape_.get(), ndim_); }
    std::span<const ssize_t> shape(const State& state) const override {
        assert(size() >= 0 &&
               "shape(const State&) must be overloaded if the size is state-dependent");
        return shape();
    }

    std::span<const ssize_t> strides() const final { return std::span(strides_.get(), ndim_); }

    constexpr bool contiguous() const noexcept final { return true; }

 private:
    template <std::ranges::sized_range Range>
    static std::unique_ptr<ssize_t[]> make_shape(Range&& shape) noexcept {
        if (shape.size() == 0) return nullptr;
        auto ptr = std::make_unique<ssize_t[]>(shape.size());
        std::copy(shape.begin(), shape.end(), ptr.get());
        return ptr;
    }

    ssize_t ndim_ = 0;
    std::unique_ptr<ssize_t[]> shape_ = nullptr;
    std::unique_ptr<ssize_t[]> strides_ = Base::shape_to_strides(ndim_, shape_.get());

    ssize_t size_ = Base::shape_to_size(ndim_, shape_.get());
};

// A convenience class for creating Arrays. Meant to be used for nodes
// that have a single numeric output value.
template <class Base, bool ProvideState = false>
class ScalarOutputMixin : public Base {};

template <class Base>
class ScalarOutputMixin<Base, false> : public Base {
 public:
    constexpr ssize_t size() const noexcept final { return 1; }
    constexpr ssize_t size(const State&) const noexcept final { return 1; }

    constexpr ssize_t ndim() const noexcept final { return 0; };

    constexpr std::span<const ssize_t> shape() const noexcept final { return {}; }
    constexpr std::span<const ssize_t> shape(const State&) const noexcept final { return {}; }

    constexpr std::span<const ssize_t> strides() const noexcept final { return {}; };

    constexpr bool contiguous() const noexcept final { return true; }

 protected:
    // Even though ScalarOutputMixinStateData is not used when ProvideState is false,
    // it still might be useful to inheriting classes who want to customize it.
    // So we make it avialable here.
    struct ScalarOutputMixinStateData : public NodeStateData {
        explicit ScalarOutputMixinStateData(double value) : update(0, value, value) {}

        const double* buff() const { return &update.value; }
        void commit() { update.old = update.value; }
        std::span<const Update> diff() const {
            return std::span(&update, update.old != update.value);
        }
        void revert() { update.value = update.old; }
        void set(double value) { update.value = value; }

        Update update;
    };
};

template <class Base>
class ScalarOutputMixin<Base, true> : public ScalarOutputMixin<Base, false> {
 public:
    // Inherits all of the methods from ScalarOutputMixin<Base, false>
    // But then also provides a few more

    /// @copydoc Array::buff()
    double const* buff(const State& state) const final {
        return this->template data_ptr<ScalarOutputMixinStateData>(state)->buff();
    }

    /// @copydoc Node::commit()
    void commit(State& state) const final {
        this->template data_ptr<ScalarOutputMixinStateData>(state)->commit();
    }

    /// @copydoc Array::diff()
    std::span<const Update> diff(const State& state) const final {
        return this->template data_ptr<ScalarOutputMixinStateData>(state)->diff();
    }

    /// @copydoc Node::revert()
    void revert(State& state) const final {
        this->template data_ptr<ScalarOutputMixinStateData>(state)->revert();
    }

 protected:
    using ScalarOutputMixinStateData = ScalarOutputMixin<Base, false>::ScalarOutputMixinStateData;

    /// Emplace a state with the given scalar value.
    void emplace_state(State& state, double value) const {
        this->template emplace_data_ptr<ScalarOutputMixinStateData>(state, value);
    }

    /// Update the state value with the given value
    void set_state(State& state, double value) const {
        this->template data_ptr<ScalarOutputMixinStateData>(state)->set(value);
    }
};

// Views are printable
std::ostream& operator<<(std::ostream& os, const Array::View& view);

// Test whether two arrays are sure to have the same shape.
bool array_shape_equal(const Array* lhs_ptr, const Array* rhs_ptr);
bool array_shape_equal(const Array& lhs, const Array& rhs);

// Test whether multiple arrays all have the same shape.
bool array_shape_equal(const std::span<const Array* const> array_ptrs);

/// Get the shape induced by broadcasting two arrays together.
/// See https://numpy.org/doc/stable/user/basics.broadcasting.html.
/// Raises an exception if the two arrays cannot be broadcast together
std::vector<ssize_t> broadcast_shape(const std::span<const ssize_t> lhs,
                                     const std::span<const ssize_t> rhs);
std::vector<ssize_t> broadcast_shape(std::initializer_list<ssize_t> lhs,
                                     std::initializer_list<ssize_t> rhs);

void deduplicate_diff(std::vector<Update>& diff);

template <std::ranges::range V>
requires(std::same_as<std::ranges::range_value_t<V>, Update>) class deduplicate_diff_view
        : public std::ranges::view_interface<deduplicate_diff_view<V>> {
 public:
    explicit deduplicate_diff_view(const V& diff) : diff_(diff.begin(), diff.end()) {
        deduplicate_diff(diff_);
    }
    explicit deduplicate_diff_view(const V&& diff) : diff_(diff.begin(), diff.end()) {
        deduplicate_diff(diff_);
    }

    auto begin() const { return diff_.begin(); }
    auto end() const { return diff_.end(); }

 private:
    std::vector<Update> diff_;
};
// todo: In C++23 once we have std::ranges::range_adaptor_closure, we should
// make this work with a range adaptor.

// Return whether the given double encodes an integer.
bool is_integer(const double& value);

/// Convert a multi index to a flat index
/// The behavior of out-of-bounds indices is undefined. Bounds are enforced via asserts.
ssize_t ravel_multi_index(std::initializer_list<ssize_t> multi_index,
                          std::initializer_list<ssize_t> shape);
ssize_t ravel_multi_index(std::span<const ssize_t> multi_index, std::span<const ssize_t> shape);

/// Convert a flat index to multi-index
std::vector<ssize_t> unravel_index(ssize_t index, std::initializer_list<ssize_t> shape);
std::vector<ssize_t> unravel_index(ssize_t index, std::span<const ssize_t> shape);

// Represent a shape (or strides) as a string in NumPy-style format.
std::string shape_to_string(const std::span<const ssize_t> shape);

template <std::ranges::viewable_range R>
ValuesInfo::ValuesInfo(R&& array_ptrs)
        : min(std::ranges::min(array_ptrs |
                               std::views::transform([](const Array* ptr) { return ptr->min(); }))),
          max(std::ranges::max(array_ptrs |
                               std::views::transform([](const Array* ptr) { return ptr->max(); }))),
          integral(std::ranges::all_of(array_ptrs,
                                       [](const Array* ptr) { return ptr->integral(); })) {}

}  // namespace dwave::optimization

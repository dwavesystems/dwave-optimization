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
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "dwave-optimization/state.hpp"
#include "dwave-optimization/utils.hpp"

namespace dwave::optimization {

// Note: we use ssize_t throughout here because Py_ssize_t is defined to take
// that value when it's available. See https://peps.python.org/pep-0353/

class Array;

// Information about where a dynamic array gets its size.
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
    bool operator==(const SizeInfo& other) const;

    // SizeInfos are printable
    friend std::ostream& operator<<(std::ostream& os, const SizeInfo& sizeinfo);

    SizeInfo substitute(ssize_t max_depth = 1) const;

    const Array* array_ptr;

    fraction multiplier;
    ssize_t offset;

    std::optional<ssize_t> min;
    std::optional<ssize_t> max;
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
 private:
    // The implementation of an iterator over the Array.
    //
    // Arrays can be contiguous or strided, so an iterator for them must be
    // too.
    //
    // For a contiguous array, the array iterator only needs to hold a pointer
    // to the memory it is iterating over.
    //
    // For a strided array, the iterator also holds information about
    // the strides and shape of the parent array, as well as its location within
    // that array.
    //
    // For a masked array, the iterator also holds information about whether
    // the current value should be masked, and what value to return if so.
    //
    // `IsConst` toggles whether the iterator is an output iterator or not.
    template <bool IsConst>
    class ArrayIteratorImpl_ {
     public:
        using difference_type = std::ptrdiff_t;
        using value_type = std::conditional<IsConst, const double, double>::type;
        using pointer = value_type*;
        using reference = value_type&;

        // Default constructor. Contiguous iterator pointing to nullptr.
        ArrayIteratorImpl_() = default;

        // Copy constructor.
        ArrayIteratorImpl_(const ArrayIteratorImpl_& other) noexcept
                : ptr_(other.ptr_),
                  mask_((other.mask_) ? std::make_unique<MaskInfo>(*other.mask_) : nullptr),
                  shape_((other.shape_) ? std::make_unique<ShapeInfo>(*other.shape_) : nullptr) {}

        // Move constructor.
        ArrayIteratorImpl_(ArrayIteratorImpl_&& other) = default;

        // Destructor.
        ~ArrayIteratorImpl_() = default;

        // Construct a contiguous iterator pointing to `ptr`.
        explicit ArrayIteratorImpl_(value_type* ptr) noexcept
                : ptr_(ptr), mask_(nullptr), shape_(nullptr) {}

        // Construct a masked iterator with a fill value.
        // Will return the value pointed at by `fill_ptr` when `*mask_ptr`
        // evaluates to true.
        // The iterator does not own the mask or the fill value. Therefore
        // the mask/fill must outlive the iterator.
        ArrayIteratorImpl_(value_type* data_ptr, const value_type* mask_ptr,
                           value_type* fill_ptr) noexcept
                : ptr_(data_ptr),
                  mask_(std::make_unique<MaskInfo>(mask_ptr, fill_ptr)),
                  shape_(nullptr) {}

        // Construct a strided iterator.
        // The iterator does not own the shape/strides, it only holds a pointer.
        // Therefore shape/strides must outlive the iterator.
        ArrayIteratorImpl_(value_type* ptr, ssize_t ndim, const ssize_t* shape,
                           const ssize_t* strides) noexcept
                : ptr_(ptr),
                  mask_(nullptr),
                  shape_(std::make_unique<ShapeInfo>(ndim, shape, strides)) {}
        ArrayIteratorImpl_(value_type* ptr, std::span<const ssize_t> shape,
                           std::span<const ssize_t> strides)
                : ArrayIteratorImpl_(ptr, shape.size(), shape.data(), strides.data()) {
            assert(shape.size() == strides.size());
        }

        // Copy and move assignment operator.
        ArrayIteratorImpl_& operator=(ArrayIteratorImpl_ other) noexcept {
            using std::swap;  // ADL, if it matters
            std::swap(ptr_, other.ptr_);
            std::swap(mask_, other.mask_);
            std::swap(shape_, other.shape_);
            return *this;
        }

        value_type& operator*() const noexcept {
            if (mask_ && *(mask_->mask_ptr)) {
                return *(mask_->fill_ptr);
            }
            return *ptr_;
        }
        value_type* operator->() const noexcept {
            if (mask_ && *(mask_->mask_ptr)) {
                return mask_->fill_ptr;
            }
            return ptr_;
        }
        value_type& operator[](ssize_t idx) const { return *(*this + idx); }

        ArrayIteratorImpl_& operator++() {
            if (shape_ && mask_) {
                assert(false && "not implemented yet");  // or maybe ever
                unreachable();
            } else if (shape_) {
                ptr_ += shape_->increment1() / itemsize;
            } else if (mask_) {
                // advance both the mask ptr and the data ptr
                ++(mask_->mask_ptr);
                ++ptr_;
            } else {
                ++ptr_;
            }
            return *this;
        }
        ArrayIteratorImpl_ operator++(int) {
            ArrayIteratorImpl_ tmp = *this;
            ++(*this);
            return tmp;
        }

        ArrayIteratorImpl_& operator--() {
            if (shape_ && mask_) {
                assert(false && "not implemented yet");  // or maybe ever
                unreachable();
            } else if (shape_) {
                ptr_ += shape_->decrement1() / itemsize;
            } else if (mask_) {
                // decrement both the mask ptr and the data ptr
                --(mask_->mask_ptr);
                --ptr_;
            } else {
                --ptr_;
            }
            return *this;
        }
        ArrayIteratorImpl_ operator--(int) {
            ArrayIteratorImpl_ tmp = *this;
            --(*this);
            return tmp;
        }

        ArrayIteratorImpl_& operator+=(difference_type rhs) {
            if (shape_ && mask_) {
                assert(false && "not implemented yet");  // or maybe ever
                unreachable();
            } else if (shape_) {
                ptr_ += shape_->increment(rhs) / itemsize;
            } else if (mask_) {
                ptr_ += rhs;
                (mask_->mask_ptr) += rhs;
            } else {
                ptr_ += rhs;
            }
            return *this;
        }
        ArrayIteratorImpl_& operator-=(difference_type rhs) { return this->operator+=(-rhs); }

        friend ArrayIteratorImpl_ operator+(ArrayIteratorImpl_ lhs, difference_type rhs) {
            return lhs += rhs;
        }
        friend ArrayIteratorImpl_ operator+(difference_type lhs, ArrayIteratorImpl_ rhs) {
            return rhs += lhs;
        }
        friend ArrayIteratorImpl_ operator-(ArrayIteratorImpl_ lhs, difference_type rhs) {
            return lhs -= rhs;
        }

        friend difference_type operator-(const ArrayIteratorImpl_& a, const ArrayIteratorImpl_& b) {
            if (!a.shape_ && !b.shape_) {
                // if they are both contiguous then it's just the distance
                // between the pointers
                return a.ptr_ - b.ptr_;
            } else if (!a.shape_ || !b.shape_) {
                assert(false && "a must be reachable from b");
                unreachable();
            }

            // Check that lhs and rhs are consistent with each other. They still
            // might have different starting locations, but with these they are
            // at least comparable.
            assert(std::ranges::equal(std::span(a.shape_->shape, a.shape_->ndim),
                                      std::span(b.shape_->shape, b.shape_->ndim)));
            assert(std::ranges::equal(std::span(a.shape_->strides, a.shape_->ndim),
                                      std::span(b.shape_->strides, b.shape_->ndim)));

            // If they point to the same pointer then they are 0 apart
            if (a.ptr_ == b.ptr_) return 0;

            // shape information about our iterators for easy access
            const ShapeInfo& shapeinfo = *a.shape_;

            if (!shapeinfo.ndim) return 0;  // 0dim arrays have no distance.

            // Calculating the distance between two strided iterators is
            // relatively complex. So we do it in a few steps.

            // We'll be mutating them, so we start by making copies of each.
            // We're trying to calculate the number of iterations from source
            // to target.
            ArrayIteratorImpl_ target = a;
            ArrayIteratorImpl_ source = b;

            // Zero out the location in each axis except the 0th and track those
            // changes in the offset
            difference_type offset = 0;
            ssize_t incs_per_step = 1;  // for each step in the axis how many increment operations
            {
                ssize_t* target_loc = a.shape_->loc.get();
                ssize_t* source_loc = b.shape_->loc.get();

                for (ssize_t axis = shapeinfo.ndim - 1; axis > 0; --axis) {
                    const ssize_t stride = shapeinfo.strides[axis] / itemsize;

                    // resetting the source removes steps
                    offset -= source_loc[axis - 1] * incs_per_step;
                    source.ptr_ -= source_loc[axis - 1] * stride;
                    source_loc[axis - 1] = 0;

                    // resetting the target adds steps
                    offset += target_loc[axis - 1] * incs_per_step;
                    target.ptr_ -= target_loc[axis - 1] * stride;
                    target_loc[axis - 1] = 0;

                    incs_per_step *= shapeinfo.shape[axis];
                }
            }

            // ok, now we're along the same "column" so we calculate how far
            // target is from source
            const ssize_t stride = shapeinfo.strides[0] / itemsize;
            offset += (target.ptr_ - source.ptr_) / stride * incs_per_step;

            return offset;
        }

        // Equal if they both point to the same underlying array location
        friend bool operator==(const ArrayIteratorImpl_& lhs, const ArrayIteratorImpl_& rhs) {
            return lhs.ptr_ == rhs.ptr_;
        }
        friend bool operator!=(const ArrayIteratorImpl_& lhs, const ArrayIteratorImpl_& rhs) {
            return lhs.ptr_ != rhs.ptr_;
        }

        // Other comparison operators are in terms of the distance. If all of the strides are
        // non-negative or all of the strides are non-positive then we could potentially check
        // faster in some cases. But for now this is easier.
        friend auto operator<=>(const ArrayIteratorImpl_& lhs, const ArrayIteratorImpl_& rhs) {
            return 0 <=> rhs - lhs;
        }

        /// Return a pointer to the underlying memory location.
        value_type* get() const noexcept { return ptr_; }

     private:
        static constexpr difference_type itemsize = sizeof(value_type);

        // All iterators hold a pointer to the underlying memory.
        value_type* ptr_ = nullptr;

        // Masked iterators store information about the mask in the MaskInfo
        // structure.
        struct MaskInfo {
            MaskInfo() = delete;

            MaskInfo(const value_type* mask_ptr, value_type* fill_ptr) noexcept
                    : mask_ptr(mask_ptr), fill_ptr(fill_ptr) {}

            // dereferencing `mask_ptr` tells the iterator whether to substitute
            // the fill value or not. Therefore this pointer is incremented along
            // with ArrayIteratorImpl_::ptr_.
            const value_type* mask_ptr;

            // A pointer to the fill value that is substituted when `!*mask_ptr`.
            // This pointer is not incremented.
            value_type* fill_ptr;
        };

        // Pointer to the MaskInfo. If nullptr then the iterator is not a masked
        // iterator.
        std::unique_ptr<MaskInfo> mask_ = nullptr;

        // Strided iterators store information about the shape and strides of
        // the parent array, as well as their location within that array in
        // the ShapeInfo structure.
        struct ShapeInfo {
            ShapeInfo() = delete;

            ShapeInfo(ssize_t ndim, const ssize_t* shape, const ssize_t* strides) noexcept
                    : ndim(ndim),
                      shape(shape),
                      strides(strides),
                      loc(std::make_unique<ssize_t[]>(std::max<ssize_t>(ndim - 1, 0))) {
                for (ssize_t i = 0; i < ndim - 1; ++i) {
                    loc[i] = 0;
                }
            }

            // Copy constructor
            ShapeInfo(const ShapeInfo& other) noexcept
                    : ndim(other.ndim),
                      shape(other.shape),
                      strides(other.strides),
                      loc(std::make_unique<ssize_t[]>(std::max<ssize_t>(ndim - 1, 0))) {
                if (ndim > 0) std::copy(other.loc.get(), other.loc.get() + ndim - 1, loc.get());
            }

            // Move constructor.
            ShapeInfo(ShapeInfo&& other) = default;

            // Copy and move assignment operator.
            ShapeInfo& operator=(ShapeInfo other) noexcept {
                using std::swap;  // ADL, if it matters
                std::swap(ndim, other.ndim);
                std::swap(shape, other.shape);
                std::swap(strides, other.strides);
                std::swap(loc, other.loc);
                return *this;
            }

            // Return the pointer offset (in bytes) relative to the current
            // position in order to decrement the iterator once.
            difference_type decrement1() {
                difference_type offset = 0;

                ssize_t axis = ndim - 1;
                for (; axis >= 1; --axis) {
                    offset -= strides[axis];

                    if (--loc[axis - 1] >= 0) break;

                    assert(loc[axis - 1] == -1);

                    offset += strides[axis] * shape[axis];
                    loc[axis - 1] = shape[axis] - 1;
                }
                if (axis == 0) {
                    offset -= strides[axis];
                }

                return offset;
            }

            // Return the pointer offset (in bytes) relative to the current
            // position in order to increment the iterator once.
            difference_type increment1() {
                difference_type offset = 0;

                ssize_t dim = ndim - 1;
                for (; dim >= 1; --dim) {
                    offset += strides[dim];

                    // we don't need to advance any other dimensions
                    if (++loc[dim - 1] < shape[dim]) break;

                    offset -= strides[dim] * shape[dim];  // zero out this dim
                    loc[dim - 1] = 0;
                }
                if (dim == 0) {
                    // For dim 0, we don't care if we're past the end of the shape.
                    // This allows us to work for dynamic arrays as well.
                    offset += strides[dim];
                }

                return offset;
            }

            // Return the pointer offset (in bytes) relative to the current
            // position in order to increment the iterator n times.
            // n can be negative.
            difference_type increment(ssize_t n) {
                // handle a few simple cases with faster implementations
                if (n == 0) return 0;
                if (n == +1) return increment1();
                if (n == -1) return decrement1();

                difference_type offset = 0;  // the number of bytes we need to move

                // working from right-to-left, figure out how many steps in each
                // axis. We handle axis 0 as a special case
                {
                    // We'll be using std::div() over ssize_t, so we'll store our
                    // current location in the struct returned by it.
                    // Unfortunately std::div() is not templated, but overloaded,
                    // so we use decltype instead.
                    decltype(std::div(ssize_t(), ssize_t())) qr{.quot = n, .rem = 0};

                    ssize_t axis = this->ndim - 1;
                    for (; axis >= 1; --axis) {
                        // if we're partway through the axis, we shift to
                        // the beginning by adding the number of steps to the total
                        // that we want to go, and updating the offset accordingly
                        if (loc[axis - 1]) {
                            assert(loc[axis - 1] > 0);  // should never be negative
                            qr.quot += loc[axis - 1];
                            offset -= loc[axis - 1] * strides[axis];
                            // loc[axis - 1] = 0;  // overwritten later, so skip resetting the loc
                        }

                        // now, the number of steps might be more than our axis
                        // can support, so we do a div
                        qr = std::div(qr.quot, shape[axis]);

                        // adjust so that the remainder is positive
                        if (qr.rem < 0) {
                            qr.quot -= 1;
                            qr.rem += shape[axis];
                        }

                        // finally adjust our location and offset
                        loc[axis - 1] = qr.rem;
                        offset += qr.rem * strides[axis];
                        // qr.rem = 0;  // overwritten later, so skip resetting the .rem

                        // if there's nothing left to do then exit early
                        if (qr.quot == 0) break;
                    }
                    if (axis == 0) {
                        offset += qr.quot * strides[0];
                    }
                }

                return offset;
            }

            // Information about the shape/strides of the parent array. Note
            // the pointers are non-owning!
            ssize_t ndim;
            const ssize_t* shape;
            const ssize_t* strides;

            // The current location of the iterator within the parent array.
            // Note that we don't care about out location in the 0th axis of the
            // array so `loc.size() == max(ndim - 1, 0)`.
            // E.g. for an array with shape (2, 3, 4), a loc of [2, 3] corresponds
            // to the location array[?, 2, 3].
            std::unique_ptr<ssize_t[]> loc;
        };

        // Pointer to the ShapeInfo. If nullptr then the iterator is not a
        // strided iterator.
        std::unique_ptr<ShapeInfo> shape_ = nullptr;
    };

 public:
    /// A std::random_access_iterator over the values in the array.
    using iterator = ArrayIteratorImpl_<false>;

    /// A std::random_access_iterator over the values in the array.
    using const_iterator = ArrayIteratorImpl_<true>;

    template<class T>
    using cache_type = std::unordered_map<const Array*, T>;

    template<class T>
    using optional_cache_type = std::optional<std::reference_wrapper<cache_type<T>>>;

    /// Container-like access to the Array's values as a flat array.
    ///
    /// Satisfies the requirements for std::ranges::random_access_range and
    /// std::ranges::sized_range.
    class View {
        // This models most, but not all, of the Container named requirements.
        // Some of the methods, like operator==(), are not modelled because it's
        // not obvious what the value to the user would be. If we ever need them
        // they are easy to add.
     public:
        /// A std::random_access_iterator over the values.
        using iterator = ArrayIteratorImpl_<false>;
        /// A std::random_access_iterator over the values.
        using const_iterator = ArrayIteratorImpl_<true>;

        /// Create an empty view.
        View() = default;

        /// Create a view from an Array and a State.
        View(const Array* array_ptr, const State* state_ptr)
                : array_ptr_(array_ptr), state_ptr_(state_ptr) {
            assert(array_ptr && "array_ptr must not be nullptr");
            assert(state_ptr && "state_ptr must not be nullptr");
        }

        /// Return a reference to the element at location `n`.
        const double& operator[](ssize_t n) const;

        /// Return a reference to the element at location `n`.
        ///
        /// This function checks whether `n` is within bounds and throws a
        /// std::out_of_range exception if it is not.
        const double& at(ssize_t n) const;

        /// Return a reference to the last element of the view.
        const double& back() const;

        /// Return an iterator to the beginning of the view.
        const_iterator begin() const;

        /// Test whether the view is empty.
        bool empty() const;

        /// Return an iterator to the end of the view.
        const_iterator end() const;

        /// Return a reference to the first element of the view.
        const double& front() const;

        /// Return the number of elements in the view.
        ssize_t size() const;

     private:
        // non-owning pointers to Array and State.
        const Array* array_ptr_;
        const State* state_ptr_;
    };

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
    const View view(const State& state) const { return View(this, &state); }

    /// The number of doubles in the flattened array.
    virtual ssize_t size() const = 0;

    /// The number of doubles in the flattened array.
    /// If the size is dependent on the state, returns Array::DYNAMIC_SIZE.
    virtual ssize_t size(const State& state) const = 0;

    /// Information about how the size of a node is calculated. See SizeInfo.
    virtual SizeInfo sizeinfo() const { return dynamic() ? SizeInfo(this) : SizeInfo(size()); }

    /// The minimum value that elements in the array may take.
    double min() const { return minmax().first; }

    /// The maximum value that elements in the array may take.
    double max() const { return minmax().second; }

    /// The smallest and largest values that elements in the array may take.
    virtual std::pair<double, double> minmax(
            optional_cache_type<std::pair<double, double>> cache = std::nullopt) const;

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

    // Return a cached value if available, and otherwise return the value returned
    // by ``func()``.
    // If the ``cache.has_value()`` returns ``false``, then the cache is ignored
    // entirely.
    template <class T, class Func>
    requires (std::same_as<std::invoke_result_t<Func>, T>)
    T memoize(optional_cache_type<T> cache, Func&& func) const {
        // If there is no cache, then just evaluate the function
        if (!cache.has_value()) return func();

        cache_type<T>& cache_ = cache->get();

        // Otherwise, check if we've already cached a value and return it if so
        if (auto it = cache_.find(this); it != cache_.end()) {
            return it->second;
        }

        // Finally, if we have a cache but we haven't already cached anything, call
        // the function and cache the output.
        auto [it, _] = cache_.emplace(this, func());
        return it->second;
    }
    template <class T>
    T memoize(optional_cache_type<T> cache, T value) const {
        if (!cache.has_value()) return value;
        auto [it, _] = cache->get().emplace(this, value);
        return it->second;
    }

    // Determine the size by the shape. For a node with a fixed size, it is simply
    // the product of the shape.
    // Expects the shape to be stored in a C-style array of length ndim.
    static ssize_t shape_to_size(const ssize_t ndim, const ssize_t* shape) noexcept {
        if (ndim <= 0) return 1;
        if (shape[0] < 0) return DYNAMIC_SIZE;
        return std::reduce(shape, shape + ndim, 1, std::multiplies<ssize_t>());
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
template <class Base>
struct ScalarOutputMixin : public Base {
    constexpr ssize_t size() const noexcept final { return 1; }
    constexpr ssize_t size(const State&) const noexcept final { return 1; }

    constexpr ssize_t ndim() const noexcept final { return 0; };

    constexpr std::span<const ssize_t> shape() const noexcept final { return {}; }
    constexpr std::span<const ssize_t> shape(const State&) const noexcept final { return {}; }

    constexpr std::span<const ssize_t> strides() const noexcept final { return {}; };

    constexpr bool contiguous() const noexcept final { return true; }
};

// Views are printable
std::ostream& operator<<(std::ostream& os, const Array::View& view);

// Test whether two arrays are sure to have the same shape.
bool array_shape_equal(const Array* lhs_ptr, const Array* rhs_ptr);
bool array_shape_equal(const Array& lhs, const Array& rhs);

/// Get the shape induced by broadcasting two arrays together.
/// See https://numpy.org/doc/stable/user/basics.broadcasting.html.
/// Raises an exception if the two arrays cannot be broadcast together
std::vector<ssize_t> broadcast_shape(const std::span<const ssize_t> lhs,
                                     const std::span<const ssize_t> rhs);
std::vector<ssize_t> broadcast_shape(std::initializer_list<ssize_t> lhs,
                                     std::initializer_list<ssize_t> rhs);

/// Convert a flat index to multi-index
std::vector<ssize_t> unravel_index(const std::span<const ssize_t> strides, ssize_t index);

/// Convert multi index to flat index
ssize_t ravel_multi_index(const std::span<const ssize_t> strides,
                          const std::span<const ssize_t> indices);

ssize_t ravel_multi_index(const std::span<const ssize_t> strides,
                          std::initializer_list<ssize_t> indices);

// Represent a shape (or strides) as a string in NumPy-style format.
std::string shape_to_string(const std::span<const ssize_t> shape);

}  // namespace dwave::optimization

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

/// The `DType<T>` concept is satisfied if and only if `T` is one of the
/// supported data types. This list is a subset of the `dtype`s supported by
/// NumPy as is designed for consistency/compatibility with NumPy.
/// See https://numpy.org/doc/stable/reference/arrays.dtypes.html for more
/// information.
template <typename T>
concept DType = std::same_as<T, float> ||  // np.float32
        std::same_as<T, double> ||         // np.float64
        std::same_as<T, std::int8_t> ||    // np.int8
        std::same_as<T, std::int16_t> ||   // np.int16
        std::same_as<T, std::int32_t> ||   // np.int32
        std::same_as<T, std::int64_t>;     // np.int64

/// The `OptionalDType<T>` concept is satisfied if and only if `T` is `void`
/// or satisfies `DType<T>`.
/// `void` is used to signal that the dtype is unknown at compile-time.
template <typename T>
concept OptionalDType = DType<T> || std::same_as<T, void>;


// Dev note: We'd like to work with `DType` only, but in order to work with the
// buffer protocol (https://docs.python.org/3/c-api/buffer.html) we need to
// use format characters. And, annoyingly, we need one more format character
// than dtype to cover all of the compilers/OS that we care about.

/// Format characters are used to signal how bytes in a fixed-size block of
/// memory should be interpreted. This list is a subset of the format characters
/// supported by Python's `struct` library.
/// See https://docs.python.org/3/library/struct.html for more information.
enum class FormatCharacter : char {
    f = 'f',  // float
    d = 'd',  // double
    i = 'i',  // int
    b = 'b',  // signed char
    h = 'h',  // signed short
    l = 'l',  // signed long
    Q = 'Q',  // signed long long
};

/// An iterator over bytes with a fixed size that can be interpreted as an
/// array.
///
/// Arrays can be contiguous or strided. For a contiguous array, the iterator
/// holds a pointer. For a strided array, the iterator holds information about
/// the strides and shape of the parent array, as well as its location within
/// that array.
///
/// `To` determines the `value_type` of the iterator.
///
/// `From` describes the type of the underlying buffer.
/// If `From` satisfies `DType<From>`, then the iterator does not need to hold
/// any information.
/// If `From` is `void`, then the iterator does the interpretation of the buffer
/// at runtime.
///
/// `IsConst` toggles whether the iterator is an output iterator or not.
/// `IsConst` must be `false` unless `To` is the same type as `From`.
template <DType To, OptionalDType From = void, bool IsConst = true>
requires(IsConst || std::same_as<To, From>) class BufferIterator {
 public:
    using difference_type = std::ptrdiff_t;
    using pointer = std::conditional<IsConst, const To*, To*>::type;
    using reference = std::conditional<IsConst, const To&, To&>::type;
    using value_type = To;

    /// Default constructor.
    BufferIterator() = default;

    /// Copy Constructor.
    BufferIterator(const BufferIterator& other) noexcept
            : ptr_(other.ptr_),
              format_(other.format_),
              shape_(other.shape_ ? std::make_unique<ShapeInfo>(*other.shape_) : nullptr) {}

    /// Move Constructor.
    BufferIterator(BufferIterator&& other) = default;

    /// Construct a contiguous iterator pointing to `ptr` when `From` is `void`.
    template <DType T>
    explicit BufferIterator(const T* ptr) noexcept requires(!DType<From>)
            : ptr_(ptr), format_(format_of(ptr)), shape_() {}

    /// Construct a contiguous iterator pointing to ptr when the type of ptr is
    /// known at compile-time.
    explicit BufferIterator(const From* ptr) noexcept requires(DType<From>)
            : ptr_(ptr), format_(), shape_() {}
    explicit BufferIterator(From* ptr) noexcept requires(DType<From> && !IsConst)
            : ptr_(ptr), format_(), shape_() {}

    /// Construct a non-contiguous iterator from a shape/strides defined as
    /// observing pointers when `From` is `void`.
    template <DType T>
    BufferIterator(const T* ptr,
                   ssize_t ndim,            // number of dimensions in the array
                   const ssize_t* shape,    // shape of the array
                   const ssize_t* strides)  // strides for each dimension of the array
            noexcept requires(!DType<From>)
            : ptr_(ptr),
              format_(format_of(ptr)),
              shape_(std::make_unique<ShapeInfo>(ndim, shape, strides)) {}

    /// Construct a non-contiguous iterator from a shape/strides defined as
    /// ranges when `From` is `void`.
    template <DType T>
    BufferIterator(const T* ptr,
                   const std::span<const ssize_t> shape,    // shape of the array
                   const std::span<const ssize_t> strides)  // strides of the array
            requires(!DType<From>)
            : BufferIterator(ptr, shape.size(), shape.data(), strides.data()) {
        assert(shape.size() == strides.size());
    }

    /// Construct a non-contiguous iterator from a shape/strides defined as
    /// observing pointers when `From` satisfies `DType<From>`.
    BufferIterator(pointer ptr,
                   ssize_t ndim,            // number of dimensions in the array
                   const ssize_t* shape,    // shape of the array
                   const ssize_t* strides)  // strides for each dimension of the array
            noexcept requires(DType<From>)
            : ptr_(ptr),
              format_(),
              shape_(std::make_unique<ShapeInfo>(ndim, shape, strides)) {}

    /// Construct a non-contiguous iterator from a shape/strides defined as
    /// ranges when `From` satisfies `DType<From>`.
    BufferIterator(pointer ptr,
                   const std::span<const ssize_t> shape,    // shape of the array
                   const std::span<const ssize_t> strides)  // strides of the array
            requires(DType<From>)
            : BufferIterator(ptr, shape.size(), shape.data(), strides.data()) {
        assert(shape.size() == strides.size());
    }

    /// Copy and move assignment operators
    BufferIterator& operator=(BufferIterator other) noexcept {
        std::swap(this->ptr_, other.ptr_);
        std::swap(this->format_, other.format_);
        std::swap(this->shape_, other.shape_);
        return *this;
    }

    /// Destructor
    ~BufferIterator() = default;

    /// Dereference the iterator when the iterator is an output iterator.
    value_type& operator*() const noexcept requires(std::same_as<To, From> && !IsConst) {
        return *static_cast<From*>(ptr_);
    }

    /// Dereference the iterator when the iterator is not an output iterator.
    value_type operator*() const noexcept requires(IsConst) {
        if constexpr (DType<From>) {
            // In this case we know at compile-time what type we're converting from
            return *static_cast<const From*>(ptr_);
        } else {
            // In this case we need to do some switch behavior at runtime
            switch (format_) {
                case FormatCharacter::f:
                    return *static_cast<const float*>(ptr_);
                case FormatCharacter::d:
                    return *static_cast<const double*>(ptr_);
                case FormatCharacter::i:
                    return *static_cast<const int*>(ptr_);
                case FormatCharacter::b:
                    return *static_cast<const signed char*>(ptr_);
                case FormatCharacter::h:
                    return *static_cast<const signed short*>(ptr_);
                case FormatCharacter::l:
                    return *static_cast<const signed long*>(ptr_);
                case FormatCharacter::Q:
                    return *static_cast<const signed long long*>(ptr_);
            }
            unreachable();
        }
    }

    /// Access the value of the iterator at the given offset when the iterator
    /// is an output iterator.
    value_type& operator[](difference_type index) const noexcept
            requires(std::same_as<To, From> && !IsConst) {
        return *(*this + index);
    }

    /// Access the value of the iterator at the given offset when the iterator
    /// is not an output iterator.
    value_type operator[](difference_type index) const noexcept requires(IsConst) {
        return *(*this + index);
    }

    /// Preincrement operator.
    BufferIterator& operator++() {
        return *this += 1;
    }

    /// Postincrement operator.
    BufferIterator operator++(int) {
        BufferIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    /// Predecrement operator.
    BufferIterator& operator--() {
        return *this -= 1;
    }

    /// Postdecrement operator.
    BufferIterator operator--(int) {
        BufferIterator tmp = *this;
        --(*this);
        return tmp;
    }

    /// Increment the iterator by `rhs` steps.
    BufferIterator& operator+=(difference_type rhs) {
        // We want to increment the pointer by a number of bytes, so we cast
        // void* to char*. But we also need to respect the const-correctness.
        using ptr_type = std::conditional<IsConst, const char*, char*>::type;

        if (shape_) {
            ptr_ = static_cast<ptr_type>(ptr_) + shape_->increment(rhs);
        } else {
            ptr_ = static_cast<ptr_type>(ptr_) + rhs * itemsize();
        }
        return *this;
    }

    /// Decrement the iterator by `rhs` steps.
    BufferIterator& operator-=(difference_type rhs) {
        return *this += -rhs;
    }

    /// Return `true` is `lhs` and `rhs` are pointing to he same underlying
    /// value.
    friend bool operator==(const BufferIterator& lhs, const BufferIterator& rhs) {
        if (lhs.shape_ && rhs.shape_) {
            return *lhs.shape_ == *rhs.shape_;
        } else if (lhs.shape_ || rhs.shape_) {
            // when one is shaped and the other is not (or they have different
            // shapes/strides) then they are not mutually reachable and this
            // method is not defined.
            assert(false && "rhs must be reachable from lhs");
            unreachable();
        } else {
            // both are contiguous, so they are equal if their pointers match.
            return lhs.ptr_ == rhs.ptr_;
        }
    }

    /// Three-way comparison between two iterators.
    friend std::strong_ordering operator<=>(const BufferIterator& lhs, const BufferIterator& rhs) {
        if (lhs.shape_ && rhs.shape_) {
            return *lhs.shape_ <=> *rhs.shape_;
        } else if (lhs.shape_ || rhs.shape_) {
            // when one is shaped and the other is not (or they have different
            // shapes/strides) then they are not mutually reachable and this
            // method is not defined.
            assert(false && "rhs must be reachable from lhs");
            unreachable();
        } else {
            // both are contiguous, so they are equal if their pointers match.
            return lhs.ptr_ <=> rhs.ptr_;
        }
    }

    /// Return an iterator pointing to the location `rhs` steps from `lhs`.
    friend BufferIterator operator+(BufferIterator lhs, difference_type rhs) { return lhs += rhs; }

    /// Return an iterator pointing to the location `lhs` steps from `rhs`.
    friend BufferIterator operator+(difference_type lhs, BufferIterator rhs) { return rhs += lhs; }

    /// Return an iterator pointing to the location `rhs` steps before `lhs`.
    friend BufferIterator operator-(BufferIterator lhs, difference_type rhs) { return lhs -= rhs; }

    /// Return the number of times we would need to increment `rhs` in order to reach `lhs`.
    friend difference_type operator-(const BufferIterator& lhs, const BufferIterator& rhs) {
        if (lhs.shape_ && rhs.shape_) {
            return *lhs.shape_ - *rhs.shape_;
        } else if (lhs.shape_ || rhs.shape_) {
            // when one is shaped and the other is not (or they have different
            // shapes/strides) then they are not mutually reachable and this
            // method is not defined.
            assert(false && "rhs must be reachable from lhs");
            unreachable();
        } else {
            // sanity check that they have the same itemsize, otherwise this is not defined
            assert(lhs.itemsize() == rhs.itemsize() && "rhs must be reachable from lhs");

            // both are contiguous, so it's just the distance between pointers
            // but we need to be careful! Because it's type dependent
            return (static_cast<const char*>(lhs.ptr_) - static_cast<const char*>(rhs.ptr_)) /
                   lhs.itemsize();
        }
    }

    /// The number of bytes used to encode values in the buffer.
    std::ptrdiff_t itemsize() const noexcept {
        if constexpr (DType<From>) {
            // If we know the type of From at compiletime, then this is easy
            return sizeof(From);
        } else {
            // Otherwise we do a runtime check
            switch (format_) {
                case FormatCharacter::f:
                    return sizeof(float);
                case FormatCharacter::d:
                    return sizeof(double);
                case FormatCharacter::i:
                    return sizeof(int);
                case FormatCharacter::b:
                    return sizeof(signed char);
                case FormatCharacter::h:
                    return sizeof(signed short);
                case FormatCharacter::l:
                    return sizeof(signed long);
                case FormatCharacter::Q:
                    return sizeof(signed long long);
            }
        }
        unreachable();
    }

 private:
    struct ShapeInfo {
        ShapeInfo() = default;

        // Copy constructor
        ShapeInfo(const ShapeInfo& other) noexcept
                : ndim(other.ndim),
                  shape(other.shape),
                  strides(other.strides),
                  loc(std::make_unique<ssize_t[]>(ndim)) {
            std::copy(other.loc.get(), other.loc.get() + ndim, loc.get());
        }

        // Move Constructor
        ShapeInfo(ShapeInfo&& other) = default;

        ShapeInfo(ssize_t ndim, const ssize_t* shape, const ssize_t* strides) noexcept
                : ndim(ndim),
                  shape(shape),
                  strides(strides),
                  loc(std::make_unique<ssize_t[]>(ndim)) {
            std::fill(loc.get(), loc.get() + ndim, 0);
        }

        // Copy and move assignment operators
        ShapeInfo& operator=(ShapeInfo other) noexcept {
            std::swap(ndim, other.ndim);
            std::swap(shape, other.shape);
            std::swap(strides, other.strides);
            std::swap(loc, other.loc);
            return *this;
        }

        ~ShapeInfo() = default;

        // Check if another ShapeInfo is in the same location. For performance we
        // only check the location when debug symbols are off.
        friend bool operator==(const ShapeInfo& lhs, const ShapeInfo& rhs) {
            // Check that lhs and rhs are consistent with each other.
            assert(std::ranges::equal(std::span(lhs.shape, lhs.ndim),
                                      std::span(rhs.shape, rhs.ndim)) &&
                   "lhs must be reachable from rhs but lhs and rhs do not have the same shape");
            assert(std::ranges::equal(std::span(lhs.strides, lhs.ndim),
                                      std::span(rhs.strides, rhs.ndim)) &&
                   "lhs must be reachable from rhs but lhs and rhs do not have the same strides");

            return std::equal(lhs.loc.get(), lhs.loc.get() + lhs.ndim, rhs.loc.get());
        }
        friend std::strong_ordering operator<=>(const ShapeInfo& lhs, const ShapeInfo& rhs) {
            // Check that lhs and rhs are consistent with each other.
            assert(std::ranges::equal(std::span(lhs.shape, lhs.ndim),
                                      std::span(rhs.shape, rhs.ndim)) &&
                   "lhs must be reachable from rhs but lhs and rhs do not have the same shape");
            assert(std::ranges::equal(std::span(lhs.strides, lhs.ndim),
                                      std::span(rhs.strides, rhs.ndim)) &&
                   "lhs must be reachable from rhs but lhs and rhs do not have the same strides");

            for (ssize_t axis = 0; axis < lhs.ndim; ++axis) {
                if (lhs.loc[axis] != rhs.loc[axis]) return lhs.loc[axis] <=> rhs.loc[axis];
            }
            return std::strong_ordering::equal;
        }

        friend difference_type operator-(const ShapeInfo& lhs, const ShapeInfo& rhs) {
            // Check that lhs and rhs are consistent with each other.
            assert(std::ranges::equal(std::span(lhs.shape, lhs.ndim),
                                      std::span(rhs.shape, rhs.ndim)) &&
                   "lhs must be reachable from rhs but lhs and rhs do not have the same shape");
            assert(std::ranges::equal(std::span(lhs.strides, lhs.ndim),
                                      std::span(rhs.strides, rhs.ndim)) &&
                   "lhs must be reachable from rhs but lhs and rhs do not have the same strides");

            // calculate the difference in steps based on the respective locs
            difference_type difference = 0;
            difference_type step = 1;
            for (ssize_t axis = lhs.ndim - 1; axis >= 0; --axis) {
                difference += (lhs.loc[axis] - rhs.loc[axis]) * step;
                step *= lhs.shape[axis];
            }

            return difference;
        }

        // Return the pointer offset (in bytes) relative to the current
        // position in order to increment the iterator n times.
        // n can be negative.
        std::ptrdiff_t increment(const ssize_t n = 1) {
            // handle a few simple cases with faster implementations
            if (n == 0) return 0;
            // if (n == +1) return increment1();
            // if (n == -1) return decrement1();

            assert(ndim > 0);  // we shouldn't be here if we're a scalar.

            // working from right-to-left, figure out how many steps in each
            // axis. We handle axis 0 as a special case

            ssize_t offset = 0;  // the number of bytes we need to move

            // We'll be using std::div() over ssize_t, so we'll store our
            // current location in the struct returned by it.
            // Unfortunately std::div() is not templated, but overloaded,
            // so we use decltype instead.
            decltype(std::div(ssize_t(), ssize_t())) qr{.quot = n, .rem = 0};

            for (ssize_t axis = this->ndim - 1; axis >= 1; --axis) {
                // A bit of sanity checking on our location
                // We can go "beyond" the relevant memory on the 0th axis, but
                // otherwise the location must be nonnegative and strictly less
                // than the size in that dimension.
                assert(0 <= loc[axis]);
                assert(axis == 0 || loc[axis] < shape[axis]);

                // if we're partway through the axis, we shift to
                // the beginning by adding the number of steps to the total
                // that we want to go, and updating the offset accordingly
                if (loc[axis]) {
                    qr.quot += loc[axis];
                    offset -= loc[axis] * strides[axis];
                    // loc[axis] = 0;  // overwritten later, so skip resetting the loc
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
                loc[axis] = qr.rem;
                offset += qr.rem * strides[axis];
                // qr.rem = 0;  // overwritten later, so skip resetting the .rem

                // if there's nothing left to do then exit early
                if (qr.quot == 0) return offset;
            }

            offset += qr.quot * strides[0];
            loc[0] += qr.quot;

            return offset;
        }

        // Information about the shape/strides of the parent array. Note
        // the pointers are non-owning!
        ssize_t ndim;
        const ssize_t* shape;
        const ssize_t* strides;

        // The current location of the iterator within the parent array.
        std::unique_ptr<ssize_t[]> loc;
    };

    // If we're const, then hold a cost void*, else hold void*
    std::conditional<IsConst, const void*, void*>::type ptr_ = nullptr;

    // If we know the type of the buffer at compile-time, we don't need to
    // store any information about it. Otherwise we keep a FormatCharacter
    // indicating what type is it.
    struct empty {};  // no information stored
    std::conditional<DType<From>, empty, FormatCharacter>::type format_;

    std::unique_ptr<ShapeInfo> shape_ = nullptr;

    // Convenience functions for getting the format character from a compile-time type.
    // todo: move into the outer namespace.
    static constexpr FormatCharacter format_of(const float* const) { return FormatCharacter::f; }
    static constexpr FormatCharacter format_of(const double* const) { return FormatCharacter::d; }
    static constexpr FormatCharacter format_of(const int* const) { return FormatCharacter::i; }
    static constexpr FormatCharacter format_of(const signed char* const) {
        return FormatCharacter::b;
    }
    static constexpr FormatCharacter format_of(const signed short* const) {
        return FormatCharacter::h;
    }
    static constexpr FormatCharacter format_of(const signed long* const) {
        return FormatCharacter::l;
    }
    static constexpr FormatCharacter format_of(const signed long long* const) {
        return FormatCharacter::Q;
    }
};

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
 public:
    /// A std::random_access_iterator over the values in the array.
    using iterator = BufferIterator<double, double, false>;

    /// A std::random_access_iterator over the values in the array.
    using const_iterator = BufferIterator<double, double, true>;

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
        using iterator = BufferIterator<double, double, false>;
        /// A std::random_access_iterator over the values.
        using const_iterator = BufferIterator<double, double, true>;

        /// Create an empty view.
        View() = default;

        /// Create a view from an Array and a State.
        View(const Array* array_ptr, const State* state_ptr)
                : array_ptr_(array_ptr), state_ptr_(state_ptr) {
            assert(array_ptr && "array_ptr must not be nullptr");
            assert(state_ptr && "state_ptr must not be nullptr");
        }

        /// Return a reference to the element at location `n`.
        double operator[](ssize_t n) const;

        /// Return a reference to the element at location `n`.
        ///
        /// This function checks whether `n` is within bounds and throws a
        /// std::out_of_range exception if it is not.
        double at(ssize_t n) const;

        /// Return a reference to the last element of the view.
        double back() const;

        /// Return an iterator to the beginning of the view.
        const_iterator begin() const;

        /// Test whether the view is empty.
        bool empty() const;

        /// Return an iterator to the end of the view.
        const_iterator end() const;

        /// Return a reference to the first element of the view.
        double front() const;

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

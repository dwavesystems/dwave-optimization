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

#pragma once

#include <algorithm>
#include <cassert>
#include <iostream>  // todo: remove
#include <memory>
#include <span>
#include <utility>

#include "dwave-optimization/utils.hpp"

namespace dwave::optimization {

// temporary experimental namespace for testing
namespace exp {

template <typename T>
concept DType = std::same_as<T, float> ||  // np.float32
        std::same_as<T, double> ||         // np.float64
        std::same_as<T, std::int8_t> ||    // np.int8
        std::same_as<T, std::int16_t> ||   // np.int16
        std::same_as<T, std::int32_t> ||   // np.int32
        std::same_as<T, std::int64_t>;     // np.int64

template <typename T>
concept OptionalDType = DType<T> || std::same_as<T, void>;

// We need to be able to go back and forth between DTypes and Python format
// characters, see https://docs.python.org/3/library/struct.html, because
// that's how the buffer protocol is designed.
// Annoyingly, to support the Windows/MacOS/Linux we need to have one more
// format character than DType.
enum class FormatCharacter : char {
    f = 'f',  // float
    d = 'd',  // double
    i = 'i',  // int
    b = 'b',  // signed char
    h = 'h',  // signed short
    l = 'l',  // signed long
    Q = 'Q',  // signed long long
};

// The iterator can only allow editing of the underlying buffer in the
// case that To and From are both known at compile-time and the same. Otherwise
// we must be const.
template <DType To, OptionalDType From = void, bool IsConst = true>
requires(IsConst || std::same_as<To, From>) class BufferIterator {
 public:
    using difference_type = std::ptrdiff_t;
    using value_type = To;

    using pointer = std::conditional<IsConst, const value_type*, value_type*>::type;
    using reference = std::conditional<IsConst, const value_type&, value_type&>::type;

    BufferIterator() = default;

    // Copy Constructor
    BufferIterator(const BufferIterator& other) noexcept
            : ptr_(other.ptr_),
              format_(other.format_),
              shape_(other.shape_ ? std::make_unique<ShapeInfo>(*other.shape_) : nullptr) {}

    // Move Constructor
    BufferIterator(BufferIterator&& other) = default;

    // Construct a contiguous iterator pointing to `ptr`.
    template <DType T>
    explicit BufferIterator(const T* ptr) noexcept requires(std::is_void_v<From>)
            : ptr_(ptr), format_(format_of(ptr)), shape_() {}

    // Construct a contiguous iterator pointing to ptr when the type of ptr is
    // known at compile-time.
    explicit BufferIterator(const From* ptr) noexcept requires(DType<From>)
            : ptr_(ptr), format_(), shape_() {}
    explicit BufferIterator(From* ptr) noexcept requires(DType<From> && !IsConst)
            : ptr_(ptr), format_(), shape_() {}

    // Construct a non-contiguous iterator from a shape/strides defined as ranges
    template <DType T>
    BufferIterator(const T* ptr,
                   ssize_t ndim,            // number of dimensions in the array
                   const ssize_t* shape,    // shape of the array
                   const ssize_t* strides)  // strides for each dimension of the array
            noexcept requires(std::is_void_v<From>)
            : ptr_(ptr),
              format_(format_of(ptr)),
              shape_(std::make_unique<ShapeInfo>(ndim, shape, strides)) {}
    template <DType T>
    BufferIterator(const T* ptr,
                   const std::span<const ssize_t> shape,    // shape of the array
                   const std::span<const ssize_t> strides)  // strides of the array
            requires(std::is_void_v<From>)
            : BufferIterator(ptr, shape.size(), shape.data(), strides.data()) {
        assert(shape.size() == strides.size());
    }

    BufferIterator(pointer ptr,
                   ssize_t ndim,            // number of dimensions in the array
                   const ssize_t* shape,    // shape of the array
                   const ssize_t* strides)  // strides for each dimension of the array
            noexcept requires(DType<From>)
            : ptr_(ptr),
              format_(),
              shape_(std::make_unique<ShapeInfo>(ndim, shape, strides)) {}
    BufferIterator(pointer ptr,
                   const std::span<const ssize_t> shape,    // shape of the array
                   const std::span<const ssize_t> strides)  // strides of the array
            requires(DType<From>)
            : BufferIterator(ptr, shape.size(), shape.data(), strides.data()) {
        assert(shape.size() == strides.size());
    }

    // Copy and move assignment operators
    BufferIterator& operator=(BufferIterator other) noexcept {
        std::swap(*this, other);
        return *this;
    }

    // Destructor
    ~BufferIterator() = default;

    value_type& operator*() const noexcept requires(std::same_as<To, From> && !IsConst) {
        return *static_cast<From*>(ptr_);
    }
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

    value_type& operator[](difference_type index) const noexcept
            requires(std::same_as<To, From> && !IsConst) {
        return *(*this + index);
    }
    value_type operator[](difference_type index) const noexcept requires(IsConst) {
        return *(*this + index);
    }

    BufferIterator& operator++() {
        return *this += 1;
    }
    BufferIterator operator++(int) {
        BufferIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    BufferIterator& operator--() {
        return *this -= 1;
    }
    BufferIterator operator--(int) {
        BufferIterator tmp = *this;
        --(*this);
        return tmp;
    }

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
    BufferIterator& operator-=(difference_type rhs) {
        return *this += -rhs;
    }

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

    friend BufferIterator operator+(BufferIterator lhs, difference_type rhs) { return lhs += rhs; }
    friend BufferIterator operator+(difference_type lhs, BufferIterator rhs) { return rhs += lhs; }

    friend BufferIterator operator-(BufferIterator lhs, difference_type rhs) { return lhs -= rhs; }

    /// The number of times we would need to increment rhs in order to reach lhs
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

extern template class BufferIterator<float>;
extern template class BufferIterator<double>;
extern template class BufferIterator<std::int8_t>;
extern template class BufferIterator<std::int16_t>;
extern template class BufferIterator<std::int32_t>;
extern template class BufferIterator<std::int64_t>;

// There are a lot of combinations...
extern template class BufferIterator<float, float>;
extern template class BufferIterator<float, double>;
extern template class BufferIterator<float, std::int8_t>;
extern template class BufferIterator<float, std::int16_t>;
extern template class BufferIterator<float, std::int32_t>;
extern template class BufferIterator<float, std::int64_t>;

extern template class BufferIterator<double, float>;
extern template class BufferIterator<double, double>;
extern template class BufferIterator<double, std::int8_t>;
extern template class BufferIterator<double, std::int16_t>;
extern template class BufferIterator<double, std::int32_t>;
extern template class BufferIterator<double, std::int64_t>;

extern template class BufferIterator<std::int8_t, float>;
extern template class BufferIterator<std::int8_t, double>;
extern template class BufferIterator<std::int8_t, std::int8_t>;
extern template class BufferIterator<std::int8_t, std::int16_t>;
extern template class BufferIterator<std::int8_t, std::int32_t>;
extern template class BufferIterator<std::int8_t, std::int64_t>;

extern template class BufferIterator<std::int16_t, float>;
extern template class BufferIterator<std::int16_t, double>;
extern template class BufferIterator<std::int16_t, std::int8_t>;
extern template class BufferIterator<std::int16_t, std::int16_t>;
extern template class BufferIterator<std::int16_t, std::int32_t>;
extern template class BufferIterator<std::int16_t, std::int64_t>;

extern template class BufferIterator<std::int32_t, float>;
extern template class BufferIterator<std::int32_t, double>;
extern template class BufferIterator<std::int32_t, std::int8_t>;
extern template class BufferIterator<std::int32_t, std::int16_t>;
extern template class BufferIterator<std::int32_t, std::int32_t>;
extern template class BufferIterator<std::int32_t, std::int64_t>;

extern template class BufferIterator<std::int64_t, float>;
extern template class BufferIterator<std::int64_t, double>;
extern template class BufferIterator<std::int64_t, std::int8_t>;
extern template class BufferIterator<std::int64_t, std::int16_t>;
extern template class BufferIterator<std::int64_t, std::int32_t>;
extern template class BufferIterator<std::int64_t, std::int64_t>;

}  // namespace exp

}  // namespace dwave::optimization

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
#include <utility>

#include "dwave-optimization/utils.hpp"

namespace dwave::optimization {

// temporary experimental namespace for testing
namespace exp {

template <typename T>
concept DType = std::same_as<T, float> ||  //
        std::same_as<T, double> ||         //
        std::same_as<T, std::int8_t> ||    //
        std::same_as<T, std::int16_t> ||   //
        std::same_as<T, std::int32_t> ||   //
        std::same_as<T, std::int64_t>;

template <typename T>
concept OptionalDType = DType<T> || std::same_as<T, void>;

namespace {

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

// The FormatInfo class tracks information about the type of the buffer
// the iterator will iterate over.
// That information can either be specified at runtime or at compile-time.
template <DType To, OptionalDType From>
struct FormatInfo {};

// Specialization for FormatInfo when want to specify both the type of the
// underlying buffer and the type we want to read at compile-time.
template <DType To, DType From>
struct FormatInfo<To, From> {
    To dereference(const void* ptr) const noexcept requires(!std::same_as<To, From>) {
        return *static_cast<const From*>(ptr);
    }
    To& dereference(void* ptr) const requires(std::same_as<To, From>) {
        return *static_cast<From*>(ptr);
    }
    const To& dereference(const void* ptr) const noexcept requires(std::same_as<To, From>) {
        return *static_cast<const From*>(ptr);
    }

    static constexpr size_t itemsize() noexcept { return sizeof(From); }
};

// Specialization for FormatInfo when we only want to specify the type we want
// to read the buffer as at compile-time.
template <DType To>
struct FormatInfo<To, void> {
    // somewhat arbitrarily default to double
    FormatInfo() noexcept : format(FormatCharacter::d) {}

    // For these we use C-style type names rather than fixed width because that's
    // how they're defined for the buffer protocol.
    // See https://docs.python.org/3/library/struct.html
    explicit FormatInfo(const float*) noexcept : format(FormatCharacter::f) {}
    explicit FormatInfo(const double*) noexcept : format(FormatCharacter::d) {}

    explicit FormatInfo(const signed int*) noexcept : format(FormatCharacter::i) {}

    explicit FormatInfo(const signed char*) noexcept : format(FormatCharacter::b) {}
    explicit FormatInfo(const signed short*) noexcept : format(FormatCharacter::h) {}
    explicit FormatInfo(const signed long*) noexcept : format(FormatCharacter::l) {}
    explicit FormatInfo(const signed long long*) noexcept : format(FormatCharacter::Q) {}

    To dereference(const void* ptr) const {
        switch (format) {
            case FormatCharacter::f:
                return *static_cast<const float*>(ptr);
            case FormatCharacter::d:
                return *static_cast<const double*>(ptr);
            case FormatCharacter::i:
                return *static_cast<const signed int*>(ptr);
            case FormatCharacter::b:
                return *static_cast<const signed char*>(ptr);
            case FormatCharacter::h:
                return *static_cast<const signed short*>(ptr);
            case FormatCharacter::l:
                return *static_cast<const signed long*>(ptr);
            case FormatCharacter::Q:
                return *static_cast<const signed long long*>(ptr);
            default:
                assert(false && "unexpected format character");
                unreachable();
        }
    }

    size_t itemsize() const {
        switch (format) {
            case FormatCharacter::f:
                return sizeof(float);
            case FormatCharacter::d:
                return sizeof(double);
            case FormatCharacter::i:
                return sizeof(signed int);
            case FormatCharacter::b:
                return sizeof(signed char);
            case FormatCharacter::h:
                return sizeof(signed short);
            case FormatCharacter::l:
                return sizeof(signed long);
            case FormatCharacter::Q:
                return sizeof(signed long long);
            default:
                assert(false && "unexpected format character");
                unreachable();
        }
    }

    FormatCharacter format;
};

struct ShapeInfo {
    ShapeInfo() = default;

    // Copy constructor
    ShapeInfo(const ShapeInfo& other) noexcept
            : ndim(other.ndim),
              shape(other.shape),
              strides(other.strides),
              loc(std::make_unique<ssize_t[]>(ndim)) {
        if (ndim > 0) std::copy(other.loc.get(), other.loc.get() + ndim, loc.get());
    }

    // Move Constructor
    ShapeInfo(ShapeInfo&& other) = default;

    // Copy and move assignment operators
    ShapeInfo& operator=(ShapeInfo other) noexcept {
        std::swap(ndim, other.ndim);
        std::swap(shape, other.shape);
        std::swap(strides, other.strides);
        std::swap(loc, other.loc);
        return *this;
    }

    ~ShapeInfo() = default;

    // Information about the shape/strides of the parent array. Note
    // the pointers are non-owning!
    ssize_t ndim;
    const ssize_t* shape;
    const ssize_t* strides;

    // The current location of the iterator within the parent array.
    std::unique_ptr<ssize_t[]> loc;
};

}  // namespace

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
            : ptr_(ptr), format_(ptr), shape_() {}

    // Construct a contiguous iterator pointing to ptr when the type of ptr is
    // known at compile-time.
    explicit BufferIterator(const From* ptr) noexcept requires(DType<From>)
            : ptr_(ptr), format_(), shape_() {}
    explicit BufferIterator(From* ptr) noexcept requires(DType<From> && !IsConst)
            : ptr_(ptr), format_(), shape_() {}

    // Copy and move assignment operators
    BufferIterator& operator=(BufferIterator other) noexcept {
        std::swap(*this, other);
        return *this;
    }

    // Destructor
    ~BufferIterator() = default;

    value_type operator*() const requires(IsConst) { return format_.dereference(ptr_); }
    value_type& operator*() const requires(!IsConst) { return format_.dereference(ptr_); }

    BufferIterator& operator++() {
        // todo: test replacing with just `return *this += 1;` for performance

        // We want to increment the pointer by a number of bytes, so we cast
        // void* to char*. But we also need to respect the const-correctness.
        using ptr_type = std::conditional<IsConst, const char*, char*>::type;

        if (shape_) {
            assert(false && "not yet implemented");
        } else {
            ptr_ = static_cast<ptr_type>(ptr_) + format_.itemsize();
        }
        return *this;
    }
    BufferIterator operator++(int) {
        BufferIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    BufferIterator& operator+=(difference_type rhs) {
        // We want to increment the pointer by a number of bytes, so we cast
        // void* to char*. But we also need to respect the const-correctness.
        using ptr_type = std::conditional<IsConst, const char*, char*>::type;

        if (shape_) {
            assert(false && "not yet implemented");
        } else {
            ptr_ = static_cast<ptr_type>(ptr_) + rhs * format_.itemsize();
        }
        return *this;
    }

    friend BufferIterator operator+(BufferIterator lhs, difference_type rhs) { return lhs += rhs; }
    friend BufferIterator operator+(difference_type lhs, BufferIterator rhs) { return rhs += lhs; }

    friend bool operator==(const BufferIterator& lhs, const BufferIterator& rhs) {
        if (lhs.shape_ && rhs.shape_) {
            assert(false);
        } else if (lhs.shape_ || rhs.shape_) {
            // one is shaped and the other isn't, so definitionally they are not
            // equal, though probably no one should be comparing them so we go
            // ahead and assert
            assert(false && "comparing a strided iterator to a non-strided one");
            return false;
        } else {
            // both are contiguous, so they are equal if their pointers match.
            return lhs.ptr_ == rhs.ptr_;
        }
    }

 private:
    // If we're const, then hold a cost void*, else hold void*
    std::conditional<IsConst, const void*, void*>::type ptr_ = nullptr;

    FormatInfo<To, From> format_;

    std::unique_ptr<ShapeInfo> shape_ = nullptr;
};

}  // namespace exp

}  // namespace dwave::optimization

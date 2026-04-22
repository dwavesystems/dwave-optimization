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

#include <cmath>
#include <concepts>
#include <memory>
#include <span>

#include "dwave-optimization/common.hpp"
#include "dwave-optimization/typing.hpp"

namespace dwave::optimization {

template <class T>
concept shape_like = std::ranges::sized_range<T> and std::integral<std::ranges::range_value_t<T>>;

// developer note: In many places it ends up being much faster to use a for-loop over shape/strides
// than to use more abstract methods such as std::equals().
// At least when tested with gcc13.
// Probably because we're doing a very simple, contiguous operations.

/// Multi-dimensional, reinterpreting iterator for iterating over arrays.
///
/// `To` specifies the `value_type` of the iterator.
/// `From` specifies the type of the underlying buffer. `void` means that the type
/// is specified at run-time.
template <DType To, OptionalDType From = To>
    requires requires {
        // Cannot create an output iterator when From != To
        requires std::is_const<To>::value or std::is_same<To, From>::value;
    }
class BufferIterator {
 public:
    // Iterator public types
    using difference_type = std::ptrdiff_t;
    using pointer = To*;
    using reference = To&;
    using value_type = To;

    // We satisfy the iterator_concept but not the iterator_category for a
    // random access iterator. We're not even a forward iterator! Because we
    // don't always return a reference type when dereferenced.
    using iterator_concept = std::random_access_iterator_tag;

    // The type of the held pointer to the underlying buffer.
    using buffer_type = From;

    /// Default constructor.
    BufferIterator() = default;

    /// Copy constructor
    BufferIterator(const BufferIterator& other) noexcept
            : ptr_(other.ptr_),
              format_(other.format_),
              ndim_(other.ndim_),
              shape_(other.shape_),
              strides_(other.strides_),
              loc_(ndim_ > 0 ? std::make_unique_for_overwrite<ssize_t[]>(ndim_) : nullptr) {
        assert(ndim_ >= 0);
        for (ssize_t axis = 0; axis < ndim_; ++axis) {
            loc_[axis] = other.loc_[axis];
        }
    }

    /// Move constructor
    BufferIterator(BufferIterator&&) = default;

    /// Construct a `BufferIterator` from a pointer, shape, and strides.
    ///
    /// Note that the pointer to shape/strides is not owning! The lifespan of those
    /// pointers must exceed that of the iterator.
    BufferIterator(From* ptr, ssize_t ndim, const ssize_t* shape, const ssize_t* strides) noexcept
        requires(DType<From>)
            : ptr_(ptr),
              format_(format_of<From>()),
              ndim_(ndim),
              shape_(shape),
              strides_(strides),
              loc_(ndim_ > 0 ? std::make_unique<ssize_t[]>(ndim_) : nullptr) {
        assert(ndim_ >= 0);

        // there is an edge case where if the shape contains any zeros we want our
        // loc to point to the end because we're an empty iterator
        for (ssize_t axis = 0; axis < ndim_; ++axis) {
            if (shape_[axis] == 0) {
                loc_[0] = shape_[0];
                break;
            }
        }
    }
    template <DType T>
    BufferIterator(T* ptr, ssize_t ndim, const ssize_t* shape, const ssize_t* strides) noexcept
        requires(not DType<From>)
            : ptr_(ptr),
              format_(format_of<T>()),
              ndim_(ndim),
              shape_(shape),
              strides_(strides),
              loc_(ndim_ > 0 ? std::make_unique<ssize_t[]>(ndim_) : nullptr) {
        assert(ndim_ >= 0);

        // there is an edge case where if the shape contains any zeros we want our
        // loc to point to the end because we're an empty iterator
        for (ssize_t axis = 0; axis < ndim_; ++axis) {
            if (shape_[axis] == 0) {
                loc_[0] = shape_[0];
                break;
            }
        }
    }

    /// Convenience constructor when using shape/strides specified as ranges.
    template <OptionalDType T>
    BufferIterator(T* ptr, std::span<const ssize_t> shape, std::span<const ssize_t> strides)
            : BufferIterator(ptr, shape.size(), shape.data(), strides.data()) {
        assert(std::ranges::size(shape) == std::ranges::size(strides));
    }

    /// Copy assigment
    BufferIterator& operator=(const BufferIterator& other) noexcept {
        if (ndim_ == other.ndim_) {
            // we don't need to reallocate our loc_, we'll overwrite it shortly
        } else if (other.ndim_ == 0) {
            loc_ = nullptr;
        } else {
            loc_ = std::make_unique_for_overwrite<ssize_t[]>(other.ndim_);
        }
        for (ssize_t axis = 0; axis < other.ndim_; ++axis) {
            loc_[axis] = other.loc_[axis];
        }

        ptr_ = other.ptr_;
        format_ = other.format_;
        ndim_ = other.ndim_;
        shape_ = other.shape_;
        strides_ = other.strides_;

        return *this;
    }

    /// Move assignment
    BufferIterator& operator=(BufferIterator&&) = default;

    /// Destructor
    ~BufferIterator() = default;

    /// Dereference the iterator. When `To` matches `From` this can return
    /// a reference, otherwise it returns a value.
    reference operator*() const noexcept
        requires(std::same_as<const To, const From>)
    {
        return *ptr_;
    }
    value_type operator*() const noexcept
        requires(not std::same_as<const To, const From>)
    {
        return value_(ptr_);
    }

    /// Access the value of the iterator at the given offset
    reference operator[](difference_type index) const noexcept
        requires(std::same_as<const To, const From>)
    {
        From* ptr = ptr_;
        increment_impl_<false>(index, &ptr, loc_.get());
        return *ptr;
    }
    value_type operator[](difference_type index) const noexcept
        requires(not std::same_as<const To, const From>)
    {
        From* ptr = ptr_;
        increment_impl_<false>(index, &ptr, loc_.get());
        return value_(ptr);
    }

    /// Preincrement operator.
    BufferIterator& operator++() { return *this += 1; }

    /// Postincrement operator.
    BufferIterator operator++(int) {
        BufferIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    /// Predecrement operator.
    BufferIterator& operator--() { return *this -= 1; }

    /// Postdecrement operator.
    BufferIterator operator--(int) {
        BufferIterator tmp = *this;
        --(*this);
        return tmp;
    }

    /// Increment the iterator by `rhs` steps.
    BufferIterator& operator+=(difference_type rhs) {
        increment_impl_<true>(rhs, &ptr_, loc_.get());
        return *this;
    }

    /// Increment the iterator in multiple dimensions. Note that this does not
    /// roll over!
    template <shape_like Shape>
    BufferIterator& operator+=(Shape&& rhs) {
        increment_multi_(std::forward<Shape>(rhs));
        return *this;
    }
    BufferIterator& operator+=(std::initializer_list<std::ptrdiff_t> rhs) {
        return (*this) += std::span(rhs);
    }

    /// Decrement the iterator by `rhs` steps.
    BufferIterator& operator-=(difference_type rhs) { return *this += -rhs; }

    /// Equality comparison between two buffer iterators.
    friend bool operator==(const BufferIterator& lhs, const BufferIterator& rhs) {
        // Check that lhs and rhs are consistent with each other.
        assert(std::ranges::equal(std::span(lhs.shape_, lhs.ndim_),
                                  std::span(rhs.shape_, rhs.ndim_)) &&
               "lhs must be reachable from rhs but lhs and rhs do not have the same shape");
        assert(std::ranges::equal(std::span(lhs.strides_, lhs.ndim_),
                                  std::span(rhs.strides_, rhs.ndim_)) &&
               "lhs must be reachable from rhs but lhs and rhs do not have the same strides");
        for (ssize_t axis = 0; axis < lhs.ndim_; ++axis) {
            if (lhs.loc_[axis] != rhs.loc_[axis]) return false;
        }
        return true;
    }

    /// Equality comparison between the iterator and the default sentinel
    friend bool operator==(const BufferIterator& lhs, std::default_sentinel_t) {
        if (lhs.ndim_ <= 0) return true;
        assert(lhs.shape_[0] >= 0 && "only defined for non-dynamic shapes");
        return lhs.loc_[0] >= lhs.shape_[0];
    }

    /// Three-way comparison between two iterators.
    friend std::strong_ordering operator<=>(const BufferIterator& lhs, const BufferIterator& rhs) {
        // Check that lhs and rhs are consistent with each other.
        assert(std::ranges::equal(std::span(lhs.shape_, lhs.ndim_),
                                  std::span(rhs.shape_, rhs.ndim_)) &&
               "lhs must be reachable from rhs but lhs and rhs do not have the same shape");
        assert(std::ranges::equal(std::span(lhs.strides_, lhs.ndim_),
                                  std::span(rhs.strides_, rhs.ndim_)) &&
               "lhs must be reachable from rhs but lhs and rhs do not have the same strides");

        for (ssize_t axis = 0; axis < lhs.ndim_; ++axis) {
            if (lhs.loc_[axis] != rhs.loc_[axis]) return lhs.loc_[axis] <=> rhs.loc_[axis];
        }
        return std::strong_ordering::equal;
    }

    /// Return an iterator pointing to the location `rhs` steps from `lhs`.
    friend BufferIterator operator+(BufferIterator lhs, difference_type rhs) { return lhs += rhs; }

    /// Return an iterator pointing to the location `lhs` steps from `rhs`.
    friend BufferIterator operator+(difference_type lhs, BufferIterator rhs) { return rhs += lhs; }

    /// Return an iterator pointing to the location `rhs` steps from `lhs`, where `rhs` is
    /// a multi-index. Note that this does not roll over!
    friend BufferIterator operator+(BufferIterator lhs, shape_like auto&& rhs) {
        lhs += rhs;
        return lhs;
    }

    /// Return an iterator pointing to the location `rhs` steps before `lhs`.
    friend BufferIterator operator-(BufferIterator lhs, difference_type rhs) { return lhs -= rhs; }

    /// Return the number of times we would need to increment `rhs` in order to reach `lhs`.
    friend difference_type operator-(const BufferIterator& lhs, const BufferIterator& rhs) {
        // Check that lhs and rhs are consistent with each other.
        assert(std::ranges::equal(std::span(lhs.shape_, lhs.ndim_),
                                  std::span(rhs.shape_, rhs.ndim_)) &&
               "lhs must be reachable from rhs but lhs and rhs do not have the same shape");
        assert(std::ranges::equal(std::span(lhs.strides_, lhs.ndim_),
                                  std::span(rhs.strides_, rhs.ndim_)) &&
               "lhs must be reachable from rhs but lhs and rhs do not have the same strides");

        // calculate the difference in steps based on the respective locs
        difference_type difference = 0;
        difference_type step = 1;
        for (ssize_t axis = lhs.ndim_ - 1; axis >= 0; --axis) {
            difference += (lhs.loc_[axis] - rhs.loc_[axis]) * step;
            step *= lhs.shape_[axis];
        }

        return difference;
    }

    /// Return the number of steps from `rhs` to the end.
    friend difference_type operator-(std::default_sentinel_t, const BufferIterator& rhs) {
        if (rhs.ndim_ <= 0) return 0;

        difference_type difference = 0;
        difference_type step = 1;
        for (ssize_t axis = rhs.ndim_ - 1; axis > 0; --axis) {
            difference -= rhs.loc_[axis] * step;
            step *= rhs.shape_[axis];
        }
        difference += (rhs.shape_[0] - rhs.loc_[0]) * step;

        return difference;
    }
    /// Return the number of steps from end to `rhs`.
    friend difference_type operator-(const BufferIterator& lhs, std::default_sentinel_t) {
        return -(std::default_sentinel - lhs);
    }

    /// Return the location in the shaped array that the iterator currently
    /// points to.
    std::span<const ssize_t> location() const { return std::span<ssize_t>(loc_.get(), ndim_); }

 private:
    void increment_multi_(shape_like auto&& multi_increment) {
        assert(static_cast<ssize_t>(std::ranges::size(multi_increment)) == ndim_);

        std::ptrdiff_t offset = 0;  // the number of bytes we need to move

        for (ssize_t axis = 0; axis < ndim_; ++axis) {
            // Make sure the new loc is in range, otherwise this function is undefined!
            assert([&]() {
                if (axis == 0) return true;  // we allow out of bounds on axis 0
                ssize_t new_loc = loc_[axis] + multi_increment[axis];
                return 0 <= new_loc and new_loc < shape_[axis];
            }());

            offset += strides_[axis] * multi_increment[axis];
            loc_[axis] += multi_increment[axis];
        }

        using ptr_type = std::conditional<std::is_const<From>::value, const char*, char*>::type;
        ptr_ = reinterpret_cast<From*>(reinterpret_cast<ptr_type>(ptr_) + offset);
    }

    // This method has a confusing signature! But this way we avoid a lot of code duplication.
    // Args:
    //   `n` is the number of steps to increment
    //   `ptr` is a pointer to the pointer we wish to update to point to the new location.
    //     it is always modified.
    //   `loc` the current location in the array as a multi-index. It is only updated if
    //     `UpdateLoc` is `true`.
    template <bool UpdateLoc>
    void increment_impl_(
            const ssize_t n, From** const ptr,
            std::conditional<UpdateLoc, ssize_t* const, const ssize_t* const>::type loc) const {
        if (n == 0) return;      // no increments to apply, exit early
        if (ndim_ == 0) return;  // incrementing a scalar does nothing

        assert(ndim_ > 0);  // negative ndim is not defined

        // working from right-to-left, figure out how many steps in each
        // axis. We handle axis 0 as a special case

        std::ptrdiff_t offset = 0;  // the number of bytes we need to move

        // We'll be using std::div() over ssize_t, so we'll store our
        // current location in the struct returned by it.
        // Unfortunately std::div() is not templated, but overloaded,
        // so we use decltype instead.
        decltype(std::div(ssize_t(), ssize_t())) qr{.quot = n, .rem = 0};

        for (ssize_t axis = ndim_ - 1; axis >= 1; --axis) {
            // A bit of sanity checking on our location
            // We can go "beyond" the relevant memory on the 0th axis, but
            // otherwise the location must be nonnegative and strictly less
            // than the size in that dimension.
            assert(0 <= loc[axis]);
            assert(axis == 0 || loc[axis] < shape_[axis]);

            // if we're partway through the axis, we shift to
            // the beginning by adding the number of steps to the total
            // that we want to go, and updating the offset accordingly
            if (loc[axis]) {
                qr.quot += loc[axis];
                offset -= loc[axis] * strides_[axis];
                // loc[axis] = 0;  // overwritten later, so skip resetting the loc
            }

            // now, the number of steps might be more than our axis
            // can support, so we do a div
            qr = std::div(qr.quot, shape_[axis]);

            // adjust so that the remainder is positive
            if (qr.rem < 0) {
                qr.quot -= 1;
                qr.rem += shape_[axis];
            }

            // finally adjust our location and offset
            if constexpr (UpdateLoc) loc[axis] = qr.rem;
            offset += qr.rem * strides_[axis];
            // qr.rem = 0;  // overwritten later, so skip resetting the .rem

            // if there's nothing left to do then exit early
            if (qr.quot == 0) break;
        }

        offset += qr.quot * strides_[0];
        if constexpr (UpdateLoc) loc[0] += qr.quot;

        using ptr_type = std::conditional<std::is_const<From>::value, const char*, char*>::type;
        *ptr = reinterpret_cast<From*>(reinterpret_cast<ptr_type>(*ptr) + offset);
    }

    /// Return the current value at the pointer as a copy.
    value_type value_(From* ptr) const noexcept {
        if constexpr (DType<From>) {
            // In this case we know at compile-time what type we're converting from
            return *ptr;
        } else {
            // In this case we need to do some switch behavior at runtime
            switch (format_) {
                case FormatCharacter::float_:
                    return *static_cast<const float*>(ptr);
                case FormatCharacter::double_:
                    return *static_cast<const double*>(ptr);
                case FormatCharacter::bool_:
                    return *static_cast<const bool*>(ptr);
                case FormatCharacter::int_:
                    return *static_cast<const int*>(ptr);
                case FormatCharacter::signedchar_:
                    return *static_cast<const signed char*>(ptr);
                case FormatCharacter::signedshort_:
                    return *static_cast<const signed short*>(ptr);
                case FormatCharacter::signedlong_:
                    return *static_cast<const signed long*>(ptr);
                case FormatCharacter::signedlonglong_:
                    return *static_cast<const signed long long*>(ptr);
            }
            unreachable();
        }
    }

    From* ptr_ = nullptr;

    // A format character indicating what type we are. We store this regardless
    // of whether the format is known at compile-time or not
    FormatCharacter format_ = FormatCharacter::double_;

    ssize_t ndim_ = 0;
    const ssize_t* shape_ = nullptr;
    const ssize_t* strides_ = nullptr;

    std::unique_ptr<ssize_t[]> loc_ = nullptr;
};

}  // namespace dwave::optimization

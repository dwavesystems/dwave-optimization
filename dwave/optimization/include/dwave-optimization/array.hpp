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
        constexpr ssize_t MIN = std::numeric_limits<ssize_t>::min();

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

    // We use the index to compare updates. This is useful because we can
    // stablely sort a vector of updates and preserve the other of operations
    // on a single index.
    friend constexpr bool operator<(const Update& lhs, const Update& rhs) {
        return lhs.index < rhs.index;
    }

    // Added for easier unit testing, checks that all members are equal (or nan and nan)
    static bool equals(const Update& lhs, const Update& rhs) {
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

    // Returns whether this is a identity update, i.e. whether the old value equals the
    // value. In the case of both old and value being NaN, this should still return
    // false.
    bool identity() const { return old == value; }

    // Use NaN to represent the "nothing" value used in placements/removals
    static constexpr double nothing = std::numeric_limits<double>::signaling_NaN();

    ssize_t index;  // The index of the updated value in the flattened array.
    double old;     // The old value
    double value;   // The new/current value.
};

// This is a generic iterator for arrays.
// It handles contiguous, strided, and masked arrays with one type.
class ArrayIterator {
 public:
    using difference_type = std::ptrdiff_t;
    using value_type = double;
    using pointer = double*;
    using reference = double&;

    ArrayIterator() = default;

    ArrayIterator(const ArrayIterator& other) noexcept
            : ptr_(other.ptr_),
              mask_((other.mask_) ? std::make_unique<MaskInfo>(*other.mask_) : nullptr),
              shape_((other.shape_) ? std::make_unique<ShapeInfo>(*other.shape_) : nullptr) {}

    ArrayIterator(ArrayIterator&& other) = default;

    ~ArrayIterator() = default;

    // Create a contiguous iterator pointing to ptr
    explicit ArrayIterator(value_type const* ptr) noexcept
            : ptr_(ptr), mask_(nullptr), shape_(nullptr) {}

    // Create a masked iterator with a fill value. Will return fill when *mask_ptr evaluates to true
    ArrayIterator(value_type const* data_ptr, value_type const* mask_ptr,
                  value_type fill = 0) noexcept
            : ptr_(data_ptr), mask_(std::make_unique<MaskInfo>(mask_ptr, fill)), shape_(nullptr) {}

    // shape and strides must outlive the iterator!
    ArrayIterator(value_type const* ptr, ssize_t ndim, const ssize_t* shape, const ssize_t* strides)
            : ptr_(ptr),
              mask_(nullptr),
              shape_((ndim >= 1) ? std::make_unique<ShapeInfo>(ndim, shape, strides) : nullptr) {}

    // Both the copy and move operator using the copy-and-swap idiom
    ArrayIterator& operator=(ArrayIterator other) noexcept {
        using std::swap;  // ADL, if it matters
        std::swap(ptr_, other.ptr_);
        std::swap(mask_, other.mask_);
        std::swap(shape_, other.shape_);
        return *this;
    }

    const value_type& operator*() const {
        if (mask_ && *(mask_->ptr)) {
            return mask_->fill;
        }

        return *ptr_;
    }
    const value_type* operator->() const {
        if (mask_ && *(mask_->ptr)) {
            return &(mask_->fill);
        }

        return ptr_;
    }

    ArrayIterator& operator++() {
        if (shape_ && mask_) {
            assert(false && "not implemented yet");  // or maybe ever
            unreachable();
        } else if (shape_) {
            // do something else
            ptr_ += shape_->advance() / sizeof(value_type);
        } else if (mask_) {
            // advance both the mask ptr and the data ptr
            ++(mask_->ptr);
            ++ptr_;
        } else {
            ++ptr_;
        }
        return *this;
    }

    ArrayIterator operator++(int) {
        ArrayIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    ArrayIterator& operator+=(difference_type rhs) {
        if (shape_ && mask_) {
            assert(false && "not implemented yet");  // or maybe ever
            unreachable();
        } else if (shape_) {
            ptr_ += shape_->advance(rhs) / sizeof(double);
        } else if (mask_) {
            ptr_ += rhs;
            (mask_->ptr) += rhs;
        } else {
            ptr_ += rhs;
        }
        return *this;
    }

    friend ArrayIterator operator+(ArrayIterator lhs, difference_type rhs) { return lhs += rhs; }

    difference_type operator-(const ArrayIterator& rhs) const {
        // We need to be careful here, we want to know how many steps the rhs
        // iterator needs to take to reach us, not the other way around.

        // if the rhs iterator is contiguous, it's just the distance from its pointer
        // to ours.
        if (!rhs.shape_) return ptr_ - rhs.ptr_;

        // otherwise we ask the rhs how far it is to us.
        return rhs.shape_->distance((ptr_ - rhs.ptr_) * sizeof(value_type));
    }

    // Equal if they both point to the same underlying array location
    bool operator==(const ArrayIterator& rhs) const { return this->ptr_ == rhs.ptr_; }

 private:
    // pointer to the underlying memory
    value_type const* ptr_ = nullptr;

    struct MaskInfo {
        MaskInfo() = delete;

        MaskInfo(value_type const* ptr, value_type fill) noexcept : ptr(ptr), fill(fill) {}

        value_type const* ptr;  // ptr to the value indicating whether to use the mask value or not
        const value_type fill;  // the value to provide for masked entries
    };

    // if this is a masked iterator, put information about the mask here
    std::unique_ptr<MaskInfo> mask_ = nullptr;

    // if this is a strided iterator, put information about the shape/strides here
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

        ShapeInfo(const ShapeInfo& other) noexcept
                : ndim(other.ndim),
                  shape(other.shape),
                  strides(other.strides),
                  loc(std::make_unique<ssize_t[]>(ndim - 1)) {
            std::copy(other.loc.get(), other.loc.get() + ndim - 1, loc.get());
        }
        ShapeInfo(ShapeInfo&& other) = default;

        // returns the number of bytes to advance. Note that this is in bytes!
        std::ptrdiff_t advance() {
            std::ptrdiff_t offset = 0;

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

        std::ptrdiff_t advance(std::ptrdiff_t n) {
            if (n < 0) {
                assert(false && "not implemented yet");
                unreachable();
            }

            // do the dumb thing for now - we can improve the performance of this
            // later
            std::ptrdiff_t offset = 0;
            for (std::ptrdiff_t i = 0; i < n; ++i) {
                offset += advance();
            }
            return offset;
        }

        // Return how many steps from the current location are need to travel
        // num_bytes
        difference_type distance(difference_type num_bytes) const {
            // each stride is multiplied by `num_bytes`, so we just take that off now
            num_bytes /= sizeof(value_type);

            // will track the number of steps in each dimension
            std::unique_ptr<ssize_t[]> steps = std::make_unique<ssize_t[]>(ndim);
            for (ssize_t dim = 0; dim < ndim; ++dim) {
                ssize_t stride = strides[dim] / sizeof(value_type);

                steps[dim] = num_bytes / stride;
                num_bytes %= stride;
            }

            // now figure out, according to shape, how many total steps we made
            ssize_t distance = 0;
            for (ssize_t dim = ndim - 1, mul = 1; dim >= 0; --dim) {
                distance += mul * steps[dim];
                mul *= shape[dim];
            }

            return distance;
        }

        const ssize_t ndim;
        const ssize_t* shape;
        const ssize_t* strides;

        // loc is of length max(ndim - 1, 0) because we don't track our location in the
        // 0th dimension
        std::unique_ptr<ssize_t[]> loc;
    };

    // shape_ == nullptr => the array is contiguous. Otherwise the stride information
    // will be encoded in the `shape_`.
    std::unique_ptr<ShapeInfo> shape_ = nullptr;
};

static_assert(std::forward_iterator<ArrayIterator>);
// todo: random access iterator?

// Represents an Array
//
// This interface is designed to work with Python's buffer protocol
// https://docs.python.org/3/c-api/buffer.html
// We however, only implement a subset of the features. Specifically:
//  * We only support doubles
//
// We also support a notion of state-dependent size. This allows nodes to
// change their size based on the state of the decision variables.
// Nodes are only permitted to grow and shrink along axis 0. This means that
// growing or shrinking a node is equivalent to extending or shrinking the
// buffer - there are no insertions needed.
// Nodes will signal that they have a state-dependent size by returning negative
// values for size() and in the first element of shape().
class Array {
 public:
    class View {
     public:
        View(const Array* array_ptr, const State* state_ptr)
                : array_ptr_(array_ptr), state_ptr_(state_ptr) {}
        ArrayIterator begin() const { return array_ptr_->begin(*state_ptr_); }
        ArrayIterator end() const { return array_ptr_->end(*state_ptr_); }

        const double operator[](std::size_t n) const { return *(begin() + n); }

        ssize_t size() const { return array_ptr_->size(*state_ptr_); }

        // Convenience access
        const double& front() const { return *begin(); }

     private:
        const Array* array_ptr_;
        const State* state_ptr_;
    };

    static_assert(std::ranges::range<View>);
    static_assert(std::ranges::sized_range<View>);

    // constant used to signal that the size is based on the state
    static constexpr ssize_t DYNAMIC_SIZE = -1;

    // Buffer protocol methods ************************************************

    // A pointer to the start of the logical structure described by the buffer
    // fields. This can be any location within the underlying physical memory
    // block of the exporter. For example, with negative strides the value may
    // point to the end of the memory block.
    // For contiguous arrays, the value points to the beginning of the memory
    // block.
    virtual double const* buff(const State& state) const = 0;

    // product(shape) * itemsize. For contiguous arrays, this is the length of
    // the underlying memory block. For non-contiguous arrays, it is the length
    // that the logical structure would have if it were copied to a contiguous
    // representation.
    ssize_t len() const { return (size() >= 0) ? size() * itemsize() : DYNAMIC_SIZE; }
    ssize_t len(const State& state) const { return size(state) * itemsize(); }

    // Exactly sizeof(double)
    constexpr ssize_t itemsize() const { return sizeof(double); }

    // "d" for double, see https://docs.python.org/3/library/struct.html
    const std::string& format() const;

    // The number of dimensions the memory represents as an n-dimensional array.
    // If 0 the buffer points to a single item representing a scalar
    virtual ssize_t ndim() const = 0;

    // An array of ndim length indicating the shape of the buffer contents
    // as an n-dimensional array.
    // Note that this is in terms of the actual type rather than in terms of
    // num bytes.
    // If the shape is state-dependent, the first value in shape will be
    // DYNAMIC_SIZE
    virtual std::span<const ssize_t> shape(const State& state) const = 0;
    virtual std::span<const ssize_t> shape() const = 0;

    // An array of length ndim giving the number of bytes to skip to get to a
    // new element in each dimension.
    virtual std::span<const ssize_t> strides() const = 0;

    // Interface methods ******************************************************

    ArrayIterator begin(const State& state) const {
        if (contiguous()) return ArrayIterator(buff(state));
        return ArrayIterator(buff(state), ndim(), shape().data(), strides().data());
    }
    ArrayIterator end(const State& state) const { return this->begin(state) + this->size(state); }

    View view(const State& state) const { return View(this, &state); }

    // The number of doubles in the flattened array.
    // If the size is dependent on the state, should return DYNAMIC_SIZE.
    virtual ssize_t size() const = 0;
    // State-dependent size must be a positive integer.
    virtual ssize_t size(const State& state) const = 0;

    // Information about how the size of a node is calculated. By default
    // a node gets its size from itself.
    virtual SizeInfo sizeinfo() const { return dynamic() ? SizeInfo(this) : SizeInfo(size()); }

    // The maximum and minimum values that elements in the array may take.
    // Defaults to the min and max values of the underlying data type.
    // todo: consider making state-dependent overloads that report the current min/max
    virtual double min() const { return std::numeric_limits<double>::min(); }
    virtual double max() const { return std::numeric_limits<double>::max(); }

    // Whether the values in the array can be interpreted as integers. This is not
    // programmatically enforced - it is up the nodes to maintain integer values.
    // Defaults to false.
    virtual bool integral() const { return false; }
    bool logical() const { return integral() && min() >= 0 && max() <= 1; }

    // Whether the data is stored contiguously.
    virtual bool contiguous() const = 0;

    // Whether the size if the array is state-dependent or not.
    bool dynamic() const { return size() < 0; }

    // Update signaling *******************************************************

    // A list of the array indices that have been updated and their previous values.
    // todo: decide whether we want to require this to be itself sorted.
    virtual std::span<const Update> diff(const State& state) const = 0;

    // The change in size
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
            : ndim_(shape.size()), shape_(make_shape(shape)) {}

    explicit ArrayOutputMixin(std::span<const ssize_t> shape)
            : ndim_(shape.size()), shape_(make_shape(shape)) {}

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
    template <class Range>
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

// Represent a shape (or strides) as a string in NumPy-style format.
std::string shape_to_string(const std::span<const ssize_t> shape);

}  // namespace dwave::optimization

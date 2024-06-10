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

#include "dwave-optimization/array.hpp"

namespace dwave::optimization {

SizeInfo::SizeInfo(const Array* array_ptr, std::optional<ssize_t> min, std::optional<ssize_t> max)
        : array_ptr(array_ptr), multiplier(1), offset(0), min(min), max(max) {
    assert(array_ptr->dynamic());
    assert(!min.has_value() || !max.has_value() || *min <= *max);
}

bool SizeInfo::operator==(const SizeInfo& other) const {
    // if one or the other is a fixed number, then this is straightforward
    if (other.multiplier == 0) return *this == other.offset;
    if (this->multiplier == 0) return this->offset == other;

    // if neither is a fixed size, then they should both be pointing to arrays
    // and those arrays should be dynamic
    assert(this->array_ptr);
    assert(other.array_ptr);
    assert(this->array_ptr->dynamic());
    assert(other.array_ptr->dynamic());

    // if they aren't the same array, then we're not equal
    if (this->array_ptr != other.array_ptr) return false;

    return (this->multiplier == other.multiplier && this->offset == other.offset &&
            this->min.value_or(0) == other.min.value_or(0) &&
            this->max == other.max);
}

SizeInfo SizeInfo::substitute(ssize_t max_depth) const {
    if (max_depth <= 0) return *this;

    SizeInfo sizeinfo = this->array_ptr->sizeinfo();

    // Check if substitution will do nothing, and return if so
    constexpr ssize_t MAX = std::numeric_limits<ssize_t>::max();
    if (this->array_ptr == sizeinfo.array_ptr && sizeinfo.multiplier == 1 && sizeinfo.offset == 0 &&
        this->min.value_or(-MAX) >= sizeinfo.min.value_or(MAX) &&
        this->max.value_or(MAX) <= sizeinfo.max.value_or(-MAX)) {
        return *this;
    }

    sizeinfo.multiplier *= this->multiplier;
    assert((this->multiplier * sizeinfo.offset).denominator() == 1);
    sizeinfo.offset = static_cast<ssize_t>(this->multiplier * sizeinfo.offset);
    sizeinfo.offset += this->offset;

    if (this->max) {
        if (sizeinfo.max) {
            sizeinfo.max = std::min<ssize_t>(*max, *sizeinfo.max);
        } else {
            sizeinfo.max = this->max;
        }
    }

    if (this->min) {
        if (sizeinfo.min) {
            sizeinfo.min = std::max<ssize_t>(*(this->min), *sizeinfo.min);
        } else {
            sizeinfo.min = this->min;
        }
    }

    if (sizeinfo.min) {
        if (this->multiplier != 1) {
            sizeinfo.min = static_cast<ssize_t>(this->multiplier * *sizeinfo.min);
        }
        sizeinfo.min = std::max<ssize_t>(0, *sizeinfo.min + this->offset);
    }
    if (sizeinfo.max) {
        if (this->multiplier != 1) {
            sizeinfo.max = static_cast<ssize_t>(this->multiplier * *sizeinfo.max);
        }
        sizeinfo.max = std::max<ssize_t>(0, *sizeinfo.max + this->offset);
    }

    // Adjust the minimum based on the offset
    if (sizeinfo.min && sizeinfo.max && sizeinfo.min == sizeinfo.max) {
        sizeinfo.offset = *sizeinfo.min;
        sizeinfo.multiplier = 0;
    }

    // in the future we can/should do an iterative rather than recurisive
    // version
    if (max_depth > 1) return sizeinfo.substitute(max_depth - 1);
    return sizeinfo;
}

std::ostream& operator<<(std::ostream& os, const SizeInfo& sizeinfo) {
    if (sizeinfo.multiplier == 0) return os << sizeinfo.offset;

    if (sizeinfo.multiplier != 1) {
        os << sizeinfo.multiplier << " * ";
    }

    os << "<Array at " << sizeinfo.array_ptr << ">.size()";

    if (sizeinfo.offset) {
        os << " + " << sizeinfo.offset;
    }

    return os;
}

const std::string& Array::format() const {
    static std::string d = "d";
    return d;
}

std::ostream& operator<<(std::ostream& os, const Slice& slice) {
    // We could get a lot fancier with this, but for now let's do the simple thing
    return os << "Slice(" << slice.start << ", " << slice.stop << ", " << slice.step << ")";
}

std::ostream& operator<<(std::ostream& os, const Update& update) {
    return os << "Update(" << update.index << ": " << update.old << " -> " << update.value << ")";
}

std::ostream& operator<<(std::ostream& os, const Array::View& view) {
    const ssize_t size = view.size();
    os << "View{";
    auto it = view.begin();
    for (ssize_t i = 0, stop = size - 1; i < stop; ++i, ++it) {
        os << *it;
        os << ", ";
    }
    if (size) os << *it;
    os << "}";
    return os;
}

// Helper function for printing shapes in error messages
// It's not efficient but it doesn't need to be
std::string shape_to_string(const std::span<const ssize_t> shape) {
    if (shape.size() == 0) {
        return "()";
    }
    if (shape.size() == 1) {
        return "(" + std::to_string(shape[0]) + ",)";
    }

    std::string out = "(";
    for (std::size_t i = 0, n = shape.size() - 1; i < n; ++i) {
        out += std::to_string(shape[i]) + ",";
    }
    out += std::to_string(shape.back()) + ")";
    return out;
}

bool array_shape_equal(const Array* lhs_ptr, const Array* rhs_ptr) {
    auto lhs_size = lhs_ptr->sizeinfo();
    auto rhs_size = rhs_ptr->sizeinfo();

    if (lhs_size == rhs_size) return true;
    if (lhs_size.array_ptr == nullptr || rhs_size.array_ptr == nullptr) return false;

    // This first loop is redundant, but often we get diamond structures
    // of predecessors so by going back together we might get to short circuit the
    // dfs
    while (lhs_size.array_ptr != lhs_ptr && rhs_size.array_ptr != rhs_ptr) {
        lhs_ptr = lhs_size.array_ptr;
        lhs_size = lhs_size.substitute();
        rhs_ptr = rhs_size.array_ptr;
        rhs_size = rhs_size.substitute();
        if (lhs_size == rhs_size) return true;
    }
    while (lhs_size.array_ptr != lhs_ptr) {
        lhs_ptr = lhs_size.array_ptr;
        lhs_size = lhs_size.substitute();
        if (lhs_size == rhs_size) return true;
    }
    while (rhs_size.array_ptr != rhs_ptr) {
        rhs_ptr = rhs_size.array_ptr;
        rhs_size = rhs_size.substitute();
        if (lhs_size == rhs_size) return true;
    }

    return false;
}
bool array_shape_equal(const Array& lhs, const Array& rhs) {
    return array_shape_equal(&lhs, &rhs);
}

// We follow NumPy's broadcasting rules
// See https://numpy.org/doc/stable/user/basics.broadcasting.html
std::vector<ssize_t> broadcast_shape(const std::span<const ssize_t> lhs,
                                     const std::span<const ssize_t> rhs) {
    // The resulting number of dimensions is the larger of the two broadcast arrays.
    std::vector<ssize_t> shape(std::max(lhs.size(), rhs.size()));

    // Walk backwards through the shapes, checking for dimension compatibility.
    // Technically span::rbegin() etc are c++23 features but it seems to work on
    // all the compilers we care about. Whereas the c++20 ranges::rbegin() etc do not.
    auto lit = lhs.rbegin();
    const auto lend = lhs.rend();
    auto rit = rhs.rbegin();
    const auto rend = rhs.rend();
    auto sit = shape.rbegin();  // input iterator
    for (; lit != lend && rit != rend; ++lit, ++rit, ++sit) {
        // Two dimensions are compatible if:
        if (*lit == *rit) {
            // they are equal
            *sit = *lit;
        } else if (*lit == 1) {
            // one of them is 1, in which case the other determines the shape
            *sit = *rit;
        } else if (*rit == 1) {
            // one of them is 1, in which case the other determines the shape
            *sit = *lit;
        } else {
            throw std::invalid_argument("operands could not be broadcast together with shapes " +
                                        shape_to_string(lhs) + " " + shape_to_string(rhs));
        }
    }

    // Missing dimensions are assumed to have size 1
    // Either lit==lend or rit==lend or both, so only one of these for-loops
    // will be entered.
    for (; lit != lend; ++lit, ++sit) {
        *sit = *lit;
    }
    for (; rit != rend; ++rit, ++sit) {
        *sit = *rit;
    }
    assert(sit == shape.rend());

    return shape;
}
std::vector<ssize_t> broadcast_shape(std::initializer_list<ssize_t> lhs,
                                     std::initializer_list<ssize_t> rhs) {
    return broadcast_shape(std::span(lhs), std::span(rhs));
}

}  // namespace dwave::optimization

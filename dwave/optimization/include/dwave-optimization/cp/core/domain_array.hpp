// Copyright 2026 D-Wave Systems Inc.
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
#include <concepts>
#include <cstddef>

#include "dwave-optimization/common.hpp"
#include "dwave-optimization/cp/core/domain_listener.hpp"
#include "dwave-optimization/cp/core/status.hpp"

namespace dwave::optimization::cp {

/// Interface for the domains of a contiguous array of variables. It is assumed that the variables
/// are flattened and they can be optional (maybe active). This is to mimic the dynamic arrays in
/// dwave-optimization.
template <typename D>
concept ReadableDomainArray = requires(const D& d, double value, int index) {
    /// Return the total number of domains
    { d.num_domains() } -> std::convertible_to<size_t>;

    /// Return the minimum size of the array
    { d.min_size() } -> std::convertible_to<ssize_t>;

    /// Return the minimum size of the array
    { d.max_size() } -> std::convertible_to<ssize_t>;

    /// Return the minimum value of the index-th variable
    { d.min(index) } -> std::convertible_to<double>;

    /// Return the maximum value of the index-th variable
    { d.max(index) } -> std::convertible_to<double>;

    /// Return the size of the domain of the index-th variable
    { d.size(index) } -> std::convertible_to<double>;

    /// Check whether the index-th variable is fixed or not
    { d.is_bound(index) } -> std::convertible_to<bool>;

    /// Check whether the index-th variable contains the value
    { d.contains(value, index) } -> std::convertible_to<bool>;

    /// Check whether the index is active
    { d.is_active(index) } -> std::convertible_to<bool>;

    /// Check whether the index-th variable is at least optional
    { d.maybe_active(index) } -> std::convertible_to<bool>;
};

template <typename D>
concept MutableDomainArray = requires(D& d, double value, int index, DomainListener* l) {
    /// Remove a value from the domain of the index-th variable
    { d.remove(value, index, l) } -> std::same_as<CPStatus>;

    /// Remove the domain of the index-th variable above the given value
    { d.remove_above(value, index, l) } -> std::same_as<CPStatus>;

    /// Remove the domain of the index-th variable below the given value
    { d.remove_below(value, index, l) } -> std::same_as<CPStatus>;

    /// Remove all the domain of the index-th variable except the given value
    { d.remove_all_but(value, index, l) } -> std::same_as<CPStatus>;

    /// Update the minimum size of the array to new_min_size
    { d.update_min_size(index, l) } -> std::same_as<CPStatus>;

    /// Update the maximum size of the array to new_max_size
    { d.update_max_size(index, l) } -> std::same_as<CPStatus>;
};

template <typename D>
concept DomainArray = ReadableDomainArray<D> and MutableDomainArray<D>;

}  // namespace dwave::optimization::cp

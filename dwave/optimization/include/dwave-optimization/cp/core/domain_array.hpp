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
#include <cstddef>

#include "dwave-optimization/common.hpp"
#include "dwave-optimization/cp/core/domain_listener.hpp"
#include "dwave-optimization/cp/core/status.hpp"

namespace dwave::optimization::cp {

/// Interface for the domains of a contiguous array of variables. It is assumed that the variables
/// are flattened and they can be optional (maybe active). This is to mimic the dynamic arrays in
/// dwave-optimization.
class DomainArray {
 public:
    virtual ~DomainArray() = default;

    /// Return the total number of domains
    virtual size_t num_domains() const = 0;

    /// Return the minimum value of the index-th variable
    virtual double min(int index) const = 0;

    /// Return the maximum value of the index-th variable
    virtual double max(int index) const = 0;

    /// Return the size of the domain of the index-th variable
    virtual double size(int index) const = 0;

    /// Return the minimum size of the array
    virtual ssize_t min_size() const = 0;

    /// Return the minimum size of the array
    virtual ssize_t max_size() const = 0;

    /// Check whether the index-th variable is fixed or not
    virtual bool is_bound(int index) const = 0;

    /// Check whether the index-th variable contains the value
    virtual bool contains(double value, int index) const = 0;

    /// Check whether the index is active
    virtual bool is_active(int index) const = 0;

    /// Check whether the index-th variable is at least optional
    virtual bool maybe_active(int index) const = 0;

    /// Remove a value from the domain of the index-th variable
    virtual CPStatus remove(double value, int index, DomainListener* l) = 0;

    /// Remove the domain of the index-th variable above the given value
    virtual CPStatus remove_above(double value, int index, DomainListener* l) = 0;

    /// Remove the domain of the index-th variable below the given value
    virtual CPStatus remove_below(double value, int index, DomainListener* l) = 0;

    /// Remove all the domain of the index-th variable except the given value
    virtual CPStatus remove_all_but(double value, int index, DomainListener* l) = 0;

    /// Update the minimum size of the array to new_min_size
    virtual CPStatus update_min_size(int new_min_size, DomainListener* l) = 0;

    /// Update the maximum size of the array to new_max_size
    virtual CPStatus update_max_size(int new_max_size, DomainListener* l) = 0;

 protected:
    // TODO: needed?
    static constexpr double PRECISION = 1e-6;
};

}  // namespace dwave::optimization::cp

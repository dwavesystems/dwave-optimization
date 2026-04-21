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
#include <vector>

#include "dwave-optimization/cp/core/domain_array.hpp"
#include "dwave-optimization/cp/state/state.hpp"
#include "dwave-optimization/cp/state/state_manager.hpp"

namespace dwave::optimization::cp {

class SparseSetArray : public DomainArray {
 public:
    SparseSetArray(StateManager* sm, ssize_t min_size, ssize_t max_size, ssize_t lb, ssize_t ub);

    size_t num_domains() const override { return min_.size(); }

    ssize_t min_size() const override { return min_size_->get_value(); }
    ssize_t max_size() const override { return max_size_->get_value(); }
    double min(int index) const override;
    double max(int index) const override;
    double size(int index) const override;
    bool is_bound(int index) const override;
    bool contains(double value, int index) const override;

    bool is_empty(int index) const;

    bool is_active(int index) const override;
    bool maybe_active(int index) const override;

    CPStatus remove(double value, int index, DomainListener* l) override;
    CPStatus remove_above(double value, int index, DomainListener* l) override;
    CPStatus remove_below(double value, int index, DomainListener* l) override;
    CPStatus remove_all_but(double value, int index, DomainListener* l) override;
    CPStatus update_min_size(int new_min_size, DomainListener* l) override;
    CPStatus update_max_size(int new_max_size, DomainListener* l) override;

 private:
    // Maximum size of the domain for each index
    std::vector<ssize_t> n_;

    // Offsets for each variable. The sparse set always stores 0...n-1.
    std::vector<ssize_t> offsets_;

    std::vector<std::vector<ssize_t>> indices_;
    std::vector<std::vector<ssize_t>> values_;

    std::vector<StateInt*> min_;
    std::vector<StateInt*> max_;
    std::vector<StateInt*> size_;

    StateInt* min_size_;
    StateInt* max_size_;

    // Exchange the positions of two values in the sparse set
    void exchange_positions(int index, ssize_t val1, ssize_t val2);

    // Update the bounds after removing a value
    void update_bounds_val_removed(int index, ssize_t val);

    // Update the maximum after removing a value
    void update_max_val_removed(int index, ssize_t val);

    // Update the minimum after removing a value
    void update_min_val_removed(int index, ssize_t val);

    // Check whether the value is inside the domain (between min and max)
    // Useful because there may be holes in the domain for sparse sets
    bool internal_contains(int index, ssize_t val) const;
};

}  // namespace dwave::optimization::cp

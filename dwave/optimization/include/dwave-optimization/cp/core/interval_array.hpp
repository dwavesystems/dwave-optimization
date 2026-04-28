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

class IntIntervalArray {
 public:
    IntIntervalArray(StateManager* sm, ssize_t size);
    IntIntervalArray(StateManager* sm, ssize_t min_size, ssize_t max_size);
    IntIntervalArray(StateManager* sm, ssize_t min_size, ssize_t max_size, ssize_t lb, ssize_t up);
    IntIntervalArray(StateManager* sm, ssize_t size, ssize_t lb, ssize_t ub);
    IntIntervalArray(StateManager* sm, ssize_t min_size, std::vector<ssize_t> lb,
                     std::vector<ssize_t> ub);
    IntIntervalArray(StateManager* sm, std::vector<ssize_t> lb, std::vector<ssize_t> ub);

    size_t num_domains() const { return min_.size(); }
    ssize_t min_size() const { return min_size_->get_value(); }
    ssize_t max_size() const { return max_size_->get_value(); }

    double min(int index) const;
    double max(int index) const;
    double size(int index) const;
    bool is_bound(int index) const;
    bool contains(double value, int index) const;

    bool is_active(int index) const;
    bool maybe_active(int index) const;

    CPStatus remove(double value, int index, DomainListener* l);
    CPStatus remove_above(double value, int index, DomainListener* l);
    CPStatus remove_below(double value, int index, DomainListener* l);
    CPStatus remove_all_but(double value, int index, DomainListener* l);
    CPStatus update_min_size(int new_min_size, DomainListener* l);
    CPStatus update_max_size(int new_max_size, DomainListener* l);

 private:
    // Change double do an object that can be backtracked.
    // And maybe get
    std::vector<StateInt*> min_;
    std::vector<StateInt*> max_;

    StateInt* min_size_;
    StateInt* max_size_;
};

class RealIntervalArray {
 public:
    RealIntervalArray(StateManager* sm, ssize_t size);
    RealIntervalArray(StateManager* sm, ssize_t min_size, ssize_t max_size);
    RealIntervalArray(StateManager* sm, ssize_t min_size, ssize_t max_size, double lb, double up);
    RealIntervalArray(StateManager* sm, ssize_t size, double lb, double up);
    RealIntervalArray(StateManager* sm, ssize_t min_size, std::vector<double> lb,
                      std::vector<double> ub);
    RealIntervalArray(StateManager* sm, std::vector<double> lb, std::vector<double> ub);

    size_t num_domains() const { return min_.size(); }
    ssize_t min_size() const { return min_size_->get_value(); }
    ssize_t max_size() const { return max_size_->get_value(); }

    double min(int index) const;
    double max(int index) const;
    double size(int index) const;
    bool is_bound(int index) const;
    bool contains(double value, int index) const;

    bool is_active(int index) const;
    bool maybe_active(int index) const;

    CPStatus remove(double value, int index, DomainListener* l);
    CPStatus remove_above(double value, int index, DomainListener* l);
    CPStatus remove_below(double value, int index, DomainListener* l);
    CPStatus remove_all_but(double value, int index, DomainListener* l);
    CPStatus update_min_size(int new_min_size, DomainListener* l);
    CPStatus update_max_size(int new_max_size, DomainListener* l);

 private:
    // Change double do an object that can be backtracked.
    // And maybe get
    std::vector<StateReal*> min_;
    std::vector<StateReal*> max_;

    StateInt* min_size_;
    StateInt* max_size_;
};

static_assert(DomainArray<IntIntervalArray>);
static_assert(DomainArray<RealIntervalArray>);

}  // namespace dwave::optimization::cp

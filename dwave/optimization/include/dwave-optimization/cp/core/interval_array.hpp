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

template <typename T>
class IntervalArray : public DomainArray {
 public:
    IntervalArray(StateManager* sm, ssize_t size);
    IntervalArray(StateManager* sm, ssize_t min_size, ssize_t max_size);
    IntervalArray(StateManager* sm, ssize_t min_size, ssize_t max_size, double lb, double up);
    IntervalArray(StateManager* sm, ssize_t size, double lb, double up);
    IntervalArray(StateManager* sm, ssize_t min_size, std::vector<double> lb,
                  std::vector<double> ub);
    IntervalArray(StateManager* sm, std::vector<double> lb, std::vector<double> ub);

    size_t num_domains() const override { return min_.size(); }

    ssize_t min_size() const override { return min_size_->get_value(); }
    ssize_t max_size() const override { return max_size_->get_value(); }
    double min(int index) const override;
    double max(int index) const override;
    double size(int index) const override;
    bool is_bound(int index) const override;
    bool contains(double value, int index) const override;

    bool is_active(int index) const override;
    bool maybe_active(int index) const override;

    CPStatus remove(double value, int index, DomainListener* l) override;
    CPStatus remove_above(double value, int index, DomainListener* l) override;
    CPStatus remove_below(double value, int index, DomainListener* l) override;
    CPStatus remove_all_but(double value, int index, DomainListener* l) override;
    CPStatus update_min_size(int new_min_size, DomainListener* l) override;
    CPStatus update_max_size(int new_max_size, DomainListener* l) override;

 private:
    // Change double do an object that can be backtracked.
    // And maybe get
    std::vector<State<T>*> min_;
    std::vector<State<T>*> max_;

    StateInt* min_size_;
    StateInt* max_size_;

    void set_sizes(ssize_t size, ssize_t min_size, ssize_t max_size);
};

using IntIntervalArray = IntervalArray<ssize_t>;
using RealIntervalArray = IntervalArray<double>;

}  // namespace dwave::optimization::cp

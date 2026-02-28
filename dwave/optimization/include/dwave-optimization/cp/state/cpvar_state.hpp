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
#include <memory>
#include <vector>

#include "dwave-optimization/cp/core/domain_array.hpp"
#include "dwave-optimization/cp/state/state_manager.hpp"

namespace dwave::optimization::cp {

class CPVarData {
 public:
    CPVarData(StateManager* sm, ssize_t size, double lb, double ub,
              std::unique_ptr<DomainListener> listener, bool integral);

    // actions on the underlying domain
    size_t num_domains() const;

    double min(int index) const;
    double max(int index) const;
    double size(int index) const;
    bool is_bound(int index) const;
    bool contains(double value, int index) const;

    CPStatus remove(double value, int index);
    CPStatus remove_above(double value, int index);
    CPStatus remove_below(double value, int index);
    CPStatus remove_all_but(double value, int index);

 protected:
    // keeping it as a unique pointer in case we wanna have different types of domains..
    std::unique_ptr<DomainArray> domains_;
    std::unique_ptr<DomainListener> listen_;
};

using CPVarsState = std::vector<std::unique_ptr<CPVarData>>;

}  // namespace dwave::optimization::cp

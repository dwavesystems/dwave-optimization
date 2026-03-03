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
class DomainArray {
 public:
    virtual ~DomainArray() = default;

    virtual size_t num_domains() const = 0;

    virtual double min(int index) const = 0;
    virtual double max(int index) const = 0;
    virtual double size(int index) const = 0;
    virtual bool is_bound(int index) const = 0;
    virtual bool contains(double value, int index) const = 0;

    virtual CPStatus remove(double value, int index, DomainListener* l) = 0;
    virtual CPStatus remove_above(double value, int index, DomainListener* l) = 0;
    virtual CPStatus remove_below(double value, int index, DomainListener* l) = 0;
    virtual CPStatus remove_all_but(double value, int index, DomainListener* l) = 0;

 protected:
    // TODO: needed?
    static constexpr double PRECISION = 1e-6;
};

}  // namespace dwave::optimization::cp

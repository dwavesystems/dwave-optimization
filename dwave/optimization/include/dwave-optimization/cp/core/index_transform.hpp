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

#include "dwave-optimization/common.hpp"

namespace dwave::optimization::cp {

/// Interface to transform index of the changing variable to output propagator index
struct IndexTransform {

    // Default destructor
    virtual ~IndexTransform() = default;

    /// Return the affected indices of the propagator/constraint given the index of a changing
    /// variable
    /// TODO: keeping this as a vector of output indices, but could well change the signature to
    /// ssize_t affected(ssize_t i)
    virtual void affected(ssize_t i, std::vector<ssize_t>& out) = 0;
};

/// Simple case, element-wise transform
struct ElementWiseTransform : IndexTransform {
    void affected(ssize_t i, std::vector<ssize_t>& out) override { out.push_back(i); }
};

/// This assumes that y = sum(x) with y.shape() == (0,) and x has any shape.
struct ReduceTransform : IndexTransform {
    void affected(ssize_t i, std::vector<ssize_t>& out) override { out.push_back(0); }
};

}  // namespace dwave::optimization::cp

// Copyright 2025 D-Wave
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

/// Interpolation routines.

#pragma once

#include <span>
#include <vector>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/state.hpp"

namespace dwave::optimization {
class BSplineNode : public ArrayOutputMixin<ArrayNode> {
 public:
    explicit BSplineNode(ArrayNode* array_ptr,
                         const int k,
                         const std::vector<double> t,
                         const std::vector<double> c);

    double const* buff(const State& state) const override;
    void commit(State& state) const override;
    std::span<const Update> diff(const State&) const override;
    void initialize_state(State& state) const override;
    bool integral() const override;

    /// @copydoc Array::minmax()
    std::pair<double, double> minmax(
            optional_cache_type<std::pair<double, double>> cache = std::nullopt) const override;

    void propagate(State& state) const override;
    void revert(State& state) const override;

    // return the values of bspline constants: degree (k), knots (t) and coefficients (c)
    int k() const;
    const std::vector<double>& t() const;
    const std::vector<double>& c() const;

    using Array::size;
    ssize_t size(const State& state) const override;

 private:
    const Array* array_ptr_;

    const int k_;
    const std::vector<double> t_;
    const std::vector<double> c_;

    std::vector<double> bspline_basis(double variable) const;
    double compute_value(double variable) const;
};

}  // namespace dwave::optimization
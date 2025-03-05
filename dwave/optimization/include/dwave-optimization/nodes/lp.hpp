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

#pragma once

#include <span>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/state.hpp"

namespace dwave::optimization {

class LPNode;

/// A logical node that propagates whether or not its predecessor LP is feasible.
/// dev note: maybe call OptimalNode or SuccessNode? to match scipy
class FeasibleNode : public ScalarOutputMixin<ArrayNode> {
 public:
    explicit FeasibleNode(LPNode* lp_ptr);

    /// @copydoc Array::buff()
    double const* buff(const State& state) const override;

    /// @copydoc Node::commit()
    void commit(State& state) const override;

    /// @copydoc Array::diff()
    std::span<const Update> diff(const State& state) const override;

    /// @copydoc Node::initialize_state()
    void initialize_state(State& state) const override;

    /// @copydoc Array::integral()
    bool integral() const override;

    /// @copydoc Array::minmax()
    std::pair<double, double> minmax(
            optional_cache_type<std::pair<double, double>> cache = std::nullopt) const override;

    /// @copydoc Node::propagate()
    void propagate(State& state) const override;

    /// @copydoc Node::revert()
    void revert(State& state) const override;

 private:
    const LPNode* lp_ptr_;
};

/// Node that solves a given LP defined by its predecessors, and outputs the optimal solution
/// (if found).
///
/// Following https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
/// linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=(0, None),
///         callback=None, options=None, x0=None, integrality=None)
class LPNode : public ArrayOutputMixin<ArrayNode> {
 public:
    // Parameter names are chosen to match scipy.optimize.lingprog()
    // for now bounds are always -inf,inf
    LPNode(ArrayNode* c_ptr,                                            // required
           ArrayNode* b_lb_ptr, ArrayNode* A_ptr, ArrayNode* b_ub_ptr,  // can be nullptr
           ArrayNode* A_eq_ptr, ArrayNode* b_eq_ptr,  // nullptr or must match size
           ArrayNode* lb_ptr, ArrayNode* ub_ptr);     // can be nullptr, both have size 1, or size n

    /// @copydoc Array::buff()
    double const* buff(const State& state) const override;

    /// @copydoc Node::commit()
    void commit(State& state) const override;

    /// The default lower bound for variables
    static const double default_lower_bound();

    /// The default lower bound for variables
    static const double default_upper_bound();

    /// @copydoc Array::diff()
    std::span<const Update> diff(const State& state) const override;

    /// Return whether the state is feasible for the LP model.
    bool feasible(const State& state) const;

    /// Any upper bound equal to or greater (or lower bound less to or equal)
    /// to this will be treated as unbounded.
    static const double infinity();

    /// @copydoc Node::initialize_state()
    void initialize_state(State& state) const override;

    /// @copydoc Array::integral()
    bool integral() const override;

    /// @copydoc Array::minmax()
    std::pair<double, double> minmax(
            optional_cache_type<std::pair<double, double>> cache = std::nullopt) const override;

    /// Return the objective value. Undefined when the LP is not feasible.
    double objective_value(const dwave::optimization::State& state) const;

    /// @copydoc Node::propagate()
    void propagate(State& state) const override;

    /// @copydoc Node::revert()
    void revert(State& state) const override;

    using ArrayOutputMixin::shape;

    using ArrayOutputMixin::size;

 private:
    template <class LPData>
    void copy_to_node_data(const State& state, LPData& lp) const;

    // The coefficients of the linear objective function to be minimized.
    // minimize c @ x
    const ArrayNode* c_ptr_;

    // The inequality constraint matrix and vector.
    // b_lb <= A @ x <= b_ub
    const ArrayNode* b_lb_ptr_;
    const ArrayNode* A_ptr_;
    const ArrayNode* b_ub_ptr_;

    // The equality constaint matrix and vector.
    // A_eq @ x == b_eq
    const ArrayNode* A_eq_ptr_;
    const ArrayNode* b_eq_ptr_;

    // The bounds on the variables. Can be nullptr, scalar, or size n
    // lb <= x <= ub
    const ArrayNode* lb_ptr_;
    const ArrayNode* ub_ptr_;
};

class ObjectiveValueNode
        : public dwave::optimization::ScalarOutputMixin<dwave::optimization::ArrayNode> {
 public:
    explicit ObjectiveValueNode(LPNode* lp_ptr);

    double const* buff(const dwave::optimization::State& state) const override;
    void commit(dwave::optimization::State& state) const override;
    std::span<const dwave::optimization::Update> diff(
            const dwave::optimization::State& state) const override;
    void initialize_state(dwave::optimization::State& state) const override;
    void propagate(dwave::optimization::State& state) const override;
    void revert(dwave::optimization::State& state) const override;

    // todo: find a bound on the min/max
    // double max() const override;
    // double min() const override;

 private:
    const LPNode* lp_ptr_;
};

}  // namespace dwave::optimization

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
#include <vector>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/state.hpp"

namespace dwave::optimization {

class LinearProgramNodeBase;

/// A logical node that propagates whether or not its predecessor LinearProgram is feasible.
class LinearProgramFeasibleNode : public ScalarOutputMixin<ArrayNode, true> {
 public:
    explicit LinearProgramFeasibleNode(LinearProgramNodeBase* lp_ptr);

    /// @copydoc Node::initialize_state()
    void initialize_state(State& state) const override;

    /// @copydoc Array::integral()
    bool integral() const override;

    /// @copydoc Array::minmax()
    std::pair<double, double> minmax(
            optional_cache_type<std::pair<double, double>> cache = std::nullopt) const override;

    /// @copydoc Node::propagate()
    void propagate(State& state) const override;

 private:
    const LinearProgramNodeBase* lp_ptr_;
};

class LinearProgramNodeBase : public Node {
 public:
    /// The default lower bound for variables
    static const double default_lower_bound();

    /// The default lower bound for variables
    static const double default_upper_bound();

    /// Return whether the state is feasible for the LP model.
    virtual bool feasible(const State& state) const = 0;

    /// Return the arguments mapped to predecessor indices
    virtual std::unordered_map<std::string, ssize_t> get_arguments() const = 0;

    /// Any upper bound equal to or greater (or lower bound less to or equal)
    /// to this will be treated as unbounded.
    static const double infinity();

    /// Return the objective value. Undefined when the LP is not feasible.
    virtual double objective_value(const State& state) const = 0;

    /// Return the solution found by the LP solver. May return an empty span in the case
    /// that the solver did not succeed or the solution is unchanged.
    virtual std::span<const double> solution(const State& state) const = 0;

    virtual std::pair<double, double> variables_minmax() const = 0;

    virtual std::span<const ssize_t> variables_shape() const = 0;

 protected:
    /// Enforce the rules on the input node(s).
    static void check_input_arguments(const ArrayNode* c_ptr, const ArrayNode* b_lb_ptr,
                                      const ArrayNode* A_ptr, const ArrayNode* b_ub_ptr,
                                      const ArrayNode* A_eq_ptr, const ArrayNode* b_eq_ptr,
                                      const ArrayNode* lb_ptr, const ArrayNode* ub_ptr);
};

/// Node that solves a given LP defined by its predecessors, and outputs the optimal solution
/// (if found).
///
/// Following https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
/// linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=(0, None),
///         callback=None, options=None, x0=None, integrality=None)
class LinearProgramNode : public LinearProgramNodeBase {
 public:
    /// Construct a LinearProgramNode
    ///
    /// Note: parameter names are chosen to match scipy.optimize.lingprog()
    LinearProgramNode(ArrayNode* c_ptr,
                      ArrayNode* b_lb_ptr, ArrayNode* A_ptr, ArrayNode* b_ub_ptr,
                      ArrayNode* A_eq_ptr, ArrayNode* b_eq_ptr,
                      ArrayNode* lb_ptr,
                      ArrayNode* ub_ptr);

    /// @copydoc Node::commit()
    void commit(State& state) const override;

    /// The LP node's state is potentially degenerate, and therefore not deterministic.
    bool deterministic_state() const override;

    /// @copydoc LinearProgramNodeBase::feasible()
    bool feasible(const State& state) const override;

    /// @copydoc LinearProgramNodeBase::get_arguments()
    std::unordered_map<std::string, ssize_t> get_arguments() const override;

    /// @copydoc Node::initialize_state()
    void initialize_state(State& state) const override;

    /// Initialize the state with the given solution
    void initialize_state(State& state, const std::span<const double> solution) const;

    /// @copydoc LinearProgramNodeBase::objective_value()
    double objective_value(const State& state) const override;

    /// @copydoc Node::propagate()
    void propagate(State& state) const override;

    /// @copydoc Node::revert()
    void revert(State& state) const override;

    /// @copydoc LinearProgramNodeBase::solution()
    std::span<const double> solution(const State& state) const override;

    /// @copydoc LinearProgramNodeBase::variables_minmax()
    std::pair<double, double> variables_minmax() const override;

    /// @copydoc LinearProgramNodeBase::variables_shape()
    std::span<const ssize_t> variables_shape() const override;

 private:
    /// Read out each of the predecessor arrays and copy the data to `lp`, where
    /// it can be easily passed in to linprog (which expects (contiguous) vectors).
    template <class LPData>
    void readout_predecessor_data(const State& state, LPData& lp) const;

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

/// A scalar node that propagates the objective value of the solution found by the
/// LinearProgramNode. Note that the output is undefined if the solution is not feasible.
class LinearProgramObjectiveValueNode : public ScalarOutputMixin<ArrayNode, true> {
 public:
    explicit LinearProgramObjectiveValueNode(LinearProgramNodeBase* lp_ptr);

    /// @copydoc Node::initialize_state()
    void initialize_state(State& state) const override;

    /// @copydoc Node::propagate()
    void propagate(State& state) const override;

 private:
    const LinearProgramNodeBase* lp_ptr_;
};

/// An array node that propagates the solution found by the LinearProgramNode. Note that the
/// solution may not be feasible or optimial.
class LinearProgramSolutionNode : public ArrayOutputMixin<ArrayNode> {
 public:
    explicit LinearProgramSolutionNode(LinearProgramNodeBase* lp_ptr);

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

    using ArrayOutputMixin::shape;

    using ArrayOutputMixin::size;

 private:
    const LinearProgramNodeBase* lp_ptr_;
};

}  // namespace dwave::optimization

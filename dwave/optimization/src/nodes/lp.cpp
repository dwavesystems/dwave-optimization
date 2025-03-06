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

#include "dwave-optimization/nodes/lp.hpp"

#include <string>

#include "../simplex.hpp"
#include "_state.hpp"

namespace dwave::optimization {

static constexpr double FEASIBILITY_TOLERANCE = 1e-07;

struct LPData {
    std::vector<double> c;
    std::vector<double> b_lb;
    std::vector<double> A;
    std::vector<double> b_ub;
    std::vector<double> A_eq;
    std::vector<double> b_eq;
    std::vector<double> lb;
    std::vector<double> ub;
};

struct LPNodeData : NodeStateData {
    explicit LPNodeData(SolveResult&& result) : result(std::move(result)) {}

    LPData lp;
    SolveResult result;
};

void check_Ab_consistency(const ssize_t num_variables, const Array* const A_ptr,
                          const Array* const b_ptr, const std::string str) {
    if (A_ptr != nullptr && b_ptr != nullptr) {
        // they are both given, so they need to have consistent shape
        auto shape = A_ptr->shape();

        if (shape.size() != 2 || shape[1] != num_variables) {
            throw std::invalid_argument(
                    "A_" + str +
                    " must have exactly two dimensions, and the number of columns in A_" + str +
                    " must be equal to the size of c");
        }

        if (b_ptr->ndim() != 1 || b_ptr->size() != shape[0]) {
            throw std::invalid_argument("b_" + str +
                                        " must be a 1-D array and the number of rows in A_ub must "
                                        "equal the number of values in b_" +
                                        str);
        }
    } else if (A_ptr != nullptr || b_ptr != nullptr) {
        // we can't have one without the other
        throw std::invalid_argument("Must provide both A_" + str + " and b_" + str + " or neither");
    }
}

LPFeasibleNode::LPFeasibleNode(LPNodeBase* lp_ptr) : lp_ptr_(lp_ptr) { add_predecessor(lp_ptr); }

double const* LPFeasibleNode::buff(const State& state) const {
    return data_ptr<ScalarNodeStateData>(state)->buff();
}

// no state to manage so nothing to do
void LPFeasibleNode::commit(State& state) const {
    return data_ptr<ScalarNodeStateData>(state)->commit();
}

std::span<const Update> LPFeasibleNode::diff(const State& state) const {
    return data_ptr<ScalarNodeStateData>(state)->diff();
}

void LPFeasibleNode::initialize_state(State& state) const {
    int index = this->topological_index();
    assert(index >= 0 && "must be topologically sorted");
    assert(static_cast<int>(state.size()) > index && "unexpected state length");
    assert(state[index] == nullptr && "already initialized state");

    state[index] = std::make_unique<ScalarNodeStateData>(lp_ptr_->feasible(state));
}

bool LPFeasibleNode::integral() const { return true; }

std::pair<double, double> LPFeasibleNode::minmax(
        optional_cache_type<std::pair<double, double>> cache) const {
    return {0, 1};
}

void LPFeasibleNode::propagate(State& state) const {
    return data_ptr<ScalarNodeStateData>(state)->set(lp_ptr_->feasible(state));
}

// no state to manage so nothing to do
void LPFeasibleNode::revert(State& state) const {
    return data_ptr<ScalarNodeStateData>(state)->revert();
}

const double LPNodeBase::default_lower_bound() { return 0.0; }

const double LPNodeBase::default_upper_bound() { return LP_INFINITY; }

const double LPNodeBase::infinity() { return LP_INFINITY; }

LPNode::LPNode(ArrayNode* c_ptr, ArrayNode* b_lb_ptr, ArrayNode* A_ptr, ArrayNode* b_ub_ptr,
               ArrayNode* A_eq_ptr, ArrayNode* b_eq_ptr, ArrayNode* lb_ptr, ArrayNode* ub_ptr)
        : c_ptr_(c_ptr),
          b_lb_ptr_(b_lb_ptr),
          A_ptr_(A_ptr),
          b_ub_ptr_(b_ub_ptr),
          A_eq_ptr_(A_eq_ptr),
          b_eq_ptr_(b_eq_ptr),
          lb_ptr_(lb_ptr),
          ub_ptr_(ub_ptr) {
    if (c_ptr_->ndim() != 1) throw std::invalid_argument("c must be 1d array");
    if (c_ptr_->dynamic()) throw std::invalid_argument("c cannot be dynamic");

    const ssize_t num_variables = c_ptr_->size();

    if (num_variables <= 0) throw std::invalid_argument("c must not be empty");

    // check_Ab_consistency(num_variables, A_ub_ptr_, b_ub_ptr_, "ub");
    // todo: fix error message for A not A_ub
    if (A_ptr != nullptr && b_lb_ptr == nullptr && b_ub_ptr == nullptr) {
        throw std::invalid_argument("if A is provided, so must b_lb or b_ub");
    }
    if (b_lb_ptr != nullptr) check_Ab_consistency(num_variables, A_ptr, b_lb_ptr, "ub");
    if (b_ub_ptr != nullptr) check_Ab_consistency(num_variables, A_ptr, b_ub_ptr, "ub");
    check_Ab_consistency(num_variables, A_eq_ptr_, b_eq_ptr_, "eq");

    // lb/ub can be null, scalar, or 1d of length num_variables
    if (lb_ptr_ == nullptr) {
    } else if (lb_ptr_->ndim() == 0) {
    } else if (lb_ptr_->ndim() == 1 && lb_ptr_->size() == num_variables) {
    } else {
        throw std::invalid_argument(
                "lb must be a scalar or a 1-D array with a number of values equal to the size of "
                "c");
    }
    if (ub_ptr_ == nullptr) {
    } else if (ub_ptr_->ndim() == 0) {
    } else if (ub_ptr_->ndim() == 1 && ub_ptr_->size() == num_variables) {
    } else {
        throw std::invalid_argument(
                "lb must be a scalar or a 1-D array with a number of values equal to the size of "
                "c");
    }

    // Finally, add the nodes (if they were passed in) as predecessors. This does
    // mean that we can't access them by index within the predecessor list, so
    // we save them as ArrayNode* on the class rather than just as Array*.
    add_predecessor(c_ptr);
    if (b_lb_ptr) add_predecessor(b_lb_ptr);
    if (A_ptr) add_predecessor(A_ptr);
    if (b_ub_ptr) add_predecessor(b_ub_ptr);
    if (A_eq_ptr) add_predecessor(A_eq_ptr);
    if (b_eq_ptr) add_predecessor(b_eq_ptr);
    if (lb_ptr) add_predecessor(lb_ptr);
    if (ub_ptr) add_predecessor(ub_ptr);
}

void LPNode::commit(State& state) const {};

bool LPNode::feasible(const State& state) const {
    return data_ptr<LPNodeData>(state)->result.feasible();
}

template <class LPData>
void LPNode::readout_predecessor_data(const State& state, LPData& lp) const {
    assert(c_ptr_ && "c should always have been provided");
    lp.c.assign(c_ptr_->view(state).begin(), c_ptr_->view(state).end());

    if (b_lb_ptr_) {
        lp.b_lb.assign(b_lb_ptr_->view(state).begin(), b_lb_ptr_->view(state).end());
    } else if (b_ub_ptr_) {
        lp.b_lb.assign(b_ub_ptr_->size(), -std::numeric_limits<double>::infinity());
    }

    if (A_ptr_) {
        lp.A.assign(A_ptr_->view(state).begin(), A_ptr_->view(state).end());
    }

    if (b_ub_ptr_) {
        lp.b_ub.assign(b_ub_ptr_->view(state).begin(), b_ub_ptr_->view(state).end());
    } else if (b_lb_ptr_) {
        lp.b_ub.assign(b_lb_ptr_->size(), std::numeric_limits<double>::infinity());
    }

    assert(lp.b_lb.size() == lp.b_ub.size() && "b lower and upper bound sizes don't match");
    assert(lp.A.size() == lp.b_ub.size() * lp.c.size() && "A has wrong size match");

    if (A_eq_ptr_) {
        lp.A_eq.assign(A_eq_ptr_->view(state).begin(), A_eq_ptr_->view(state).end());
    }

    if (b_eq_ptr_) {
        lp.b_eq.assign(b_eq_ptr_->view(state).begin(), b_eq_ptr_->view(state).end());
    }
    assert(lp.A_eq.size() == lp.b_eq.size() * lp.c.size() && "A_eq and b_eq sizes don't match");

    if (lb_ptr_) {
        lp.lb.assign(lb_ptr_->view(state).begin(), lb_ptr_->view(state).end());
    } else {
        lp.lb.assign(c_ptr_->size(state), LPNodeBase::default_lower_bound());
    }

    if (ub_ptr_) {
        lp.ub.assign(ub_ptr_->view(state).begin(), ub_ptr_->view(state).end());
    } else {
        lp.ub.assign(c_ptr_->size(state), LPNodeBase::default_upper_bound());
    }

    assert(lp.c.size() == lp.lb.size() && "c and lower bound sizes do not match");
    assert(lp.lb.size() == lp.ub.size() && "lower and upper bound sizes do not match");
}

void LPNode::initialize_state(State& state) const {
    int index = this->topological_index();
    assert(index >= 0 && "must be topologically sorted");
    assert(static_cast<int>(state.size()) > index && "unexpected state length");
    assert(state[index] == nullptr && "already initialized state");

    LPData lp;
    readout_predecessor_data(state, lp);
    SolveResult result = linprog(lp.c, lp.b_lb, lp.A, lp.b_ub, lp.A_eq, lp.b_eq, lp.lb, lp.ub,
                                 FEASIBILITY_TOLERANCE);

    state[index] = std::make_unique<LPNodeData>(std::move(result));
}

double LPNode::objective_value(const dwave::optimization::State& state) const {
    return data_ptr<LPNodeData>(state)->result.objective();
}

void LPNode::propagate(State& state) const {
    auto data = data_ptr<LPNodeData>(state);

    readout_predecessor_data(state, data->lp);
    data->result = linprog(data->lp.c, data->lp.b_lb, data->lp.A, data->lp.b_ub, data->lp.A_eq,
                           data->lp.b_eq, data->lp.lb, data->lp.ub, FEASIBILITY_TOLERANCE);

    Node::propagate(state);
}

void LPNode::revert(State& state) const {
    // Nothing to do on revert as all changes are tracked by successor nodes
}

std::span<const double> LPNode::solution(const State& state) const {
    return data_ptr<LPNodeData>(state)->result.solution();
}

std::pair<double, double> LPNode::variables_minmax() const {
    return std::make_pair(lb_ptr_ ? lb_ptr_->min() : LPNode::default_lower_bound(),
                          ub_ptr_ ? ub_ptr_->max() : LPNode::default_upper_bound());
}

std::span<const ssize_t> LPNode::variables_shape() const { return c_ptr_->shape(); }

LPObjectiveValueNode::LPObjectiveValueNode(LPNodeBase* lp_ptr) : lp_ptr_(lp_ptr) {
    add_predecessor(lp_ptr);
}

double const* LPObjectiveValueNode::buff(const State& state) const {
    return data_ptr<ScalarNodeStateData>(state)->buff();
}

// no state to manage so nothing to do
void LPObjectiveValueNode::commit(State& state) const {
    return data_ptr<ScalarNodeStateData>(state)->commit();
}

std::span<const Update> LPObjectiveValueNode::diff(const State& state) const {
    return data_ptr<ScalarNodeStateData>(state)->diff();
}

void LPObjectiveValueNode::initialize_state(State& state) const {
    int index = this->topological_index();
    assert(index >= 0 && "must be topologically sorted");
    assert(static_cast<int>(state.size()) > index && "unexpected state length");
    assert(state[index] == nullptr && "already initialized state");

    // if not feasible, we're undefined. So let's default to 0.0. Though it
    // might actually be better to default to our own maximum value.
    double value = lp_ptr_->feasible(state) ? lp_ptr_->objective_value(state) : 0;

    state[index] = std::make_unique<ScalarNodeStateData>(value);
}

void LPObjectiveValueNode::propagate(State& state) const {
    if (lp_ptr_->feasible(state)) {
        data_ptr<ScalarNodeStateData>(state)->set(lp_ptr_->objective_value(state));
    }
    // Otherwise do nothing because we're not defined.
    // We could consider setting ourselves to max()
}

// no state to manage so nothing to do
void LPObjectiveValueNode::revert(State& state) const {
    return data_ptr<ScalarNodeStateData>(state)->revert();
}

LPSolutionNode::LPSolutionNode(LPNodeBase* lp_ptr)
        : ArrayOutputMixin(lp_ptr->variables_shape()), lp_ptr_(lp_ptr) {
    add_predecessor(lp_ptr);
}

double const* LPSolutionNode::buff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->buff();
}

void LPSolutionNode::commit(State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->commit();
}

std::span<const Update> LPSolutionNode::diff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->diff();
}

void LPSolutionNode::initialize_state(State& state) const {
    int index = this->topological_index();
    assert(index >= 0 && "must be topologically sorted");
    assert(static_cast<int>(state.size()) > index && "unexpected state length");
    assert(state[index] == nullptr && "already initialized state");

    std::vector<double> tmp;
    tmp.assign(lp_ptr_->solution(state).begin(), lp_ptr_->solution(state).end());
    state[index] = std::make_unique<ArrayNodeStateData>(tmp);
}

bool LPSolutionNode::integral() const { return false; }

std::pair<double, double> LPSolutionNode::minmax(
        optional_cache_type<std::pair<double, double>> cache) const {
    return memoize(cache, [&]() { return lp_ptr_->variables_minmax(); });
}

void LPSolutionNode::propagate(State& state) const {
    data_ptr<ArrayNodeStateData>(state)->assign(lp_ptr_->solution(state));
}

void LPSolutionNode::revert(State& state) const { data_ptr<ArrayNodeStateData>(state)->revert(); }

}  // namespace dwave::optimization

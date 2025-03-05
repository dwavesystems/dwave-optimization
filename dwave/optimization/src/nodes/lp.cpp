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

#include <sstream>
#include <string>

#include "../simplex.hpp"
#include "_state.hpp"

namespace dwave::optimization {

// Default bounds for the variables
const double DEFAULT_LOWER_BOUND = 0;
const double DEFAULT_UPPER_BOUND = std::numeric_limits<double>::infinity();  // TODO:
// const double FEASIBILITY_TOLERANCE = 1e-07;

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

struct LPNodeData : ArrayNodeStateData {
    explicit LPNodeData(std::vector<double>&& values, bool is_feasible,
                        double objective_value) noexcept
            : ArrayNodeStateData(std::move(values)),
              is_feasible(is_feasible),
              old_is_feasible(is_feasible),
              objective_value(objective_value),
              old_objective_value(objective_value) {}

    bool is_feasible;
    bool old_is_feasible;

    double objective_value;
    double old_objective_value;

    LPData lp;
};

Array* c_is_1d(Array* array_ptr) {
    if (array_ptr->ndim() != 1) throw std::invalid_argument("c must be a 1d array");
    return array_ptr;
}

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

FeasibleNode::FeasibleNode(LPNode* lp_ptr) : lp_ptr_(lp_ptr) { add_predecessor(lp_ptr); }

double const* FeasibleNode::buff(const State& state) const {
    return data_ptr<ScalarNodeStateData>(state)->buff();
}

// no state to manage so nothing to do
void FeasibleNode::commit(State& state) const {
    return data_ptr<ScalarNodeStateData>(state)->commit();
}

std::span<const Update> FeasibleNode::diff(const State& state) const {
    return data_ptr<ScalarNodeStateData>(state)->diff();
}

void FeasibleNode::initialize_state(State& state) const {
    int index = this->topological_index();
    assert(index >= 0 && "must be topologically sorted");
    assert(static_cast<int>(state.size()) > index && "unexpected state length");
    assert(state[index] == nullptr && "already initialized state");

    state[index] = std::make_unique<ScalarNodeStateData>(lp_ptr_->feasible(state));
}

bool FeasibleNode::integral() const { return true; }

std::pair<double, double> FeasibleNode::minmax(
        optional_cache_type<std::pair<double, double>> cache) const {
    return {0, 1};
}

void FeasibleNode::propagate(State& state) const {
    return data_ptr<ScalarNodeStateData>(state)->set(lp_ptr_->feasible(state));
}

// no state to manage so nothing to do
void FeasibleNode::revert(State& state) const {
    return data_ptr<ScalarNodeStateData>(state)->revert();
}

LPNode::LPNode(ArrayNode* c_ptr, ArrayNode* b_lb_ptr, ArrayNode* A_ptr, ArrayNode* b_ub_ptr,
               ArrayNode* A_eq_ptr, ArrayNode* b_eq_ptr, ArrayNode* lb_ptr, ArrayNode* ub_ptr)
        : ArrayOutputMixin(c_is_1d(c_ptr)->shape()[0]),
          c_ptr_(c_ptr),
          b_lb_ptr_(b_lb_ptr),
          A_ptr_(A_ptr),
          b_ub_ptr_(b_ub_ptr),
          A_eq_ptr_(A_eq_ptr),
          b_eq_ptr_(b_eq_ptr),
          lb_ptr_(lb_ptr),
          ub_ptr_(ub_ptr) {
    // c is a 1d matrix of length num_variables. The 1d-ness has already been checked on
    // initialization
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

double const* LPNode::buff(const State& state) const { return data_ptr<LPNodeData>(state)->buff(); }

void LPNode::commit(State& state) const { return data_ptr<LPNodeData>(state)->commit(); }

std::span<const Update> LPNode::diff(const State& state) const {
    return data_ptr<LPNodeData>(state)->diff();
}

bool LPNode::feasible(const State& state) const { return data_ptr<LPNodeData>(state)->is_feasible; }

template <class LPData>
void LPNode::copy_to_node_data(const State& state, LPData& lp) const {
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
        lp.lb.assign(c_ptr_->size(state), DEFAULT_LOWER_BOUND);
    }

    if (ub_ptr_) {
        lp.ub.assign(ub_ptr_->view(state).begin(), ub_ptr_->view(state).end());
    } else {
        lp.ub.assign(c_ptr_->size(state), DEFAULT_UPPER_BOUND);
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
    copy_to_node_data(state, lp);
    SolveResult result = linprog(lp.c, lp.b_lb, lp.A, lp.b_ub, lp.A_eq, lp.b_eq, lp.lb, lp.ub);

    state[index] = std::make_unique<LPNodeData>(std::vector(result.solution()), result.feasible(),
                                                result.objective());
}

bool LPNode::integral() const { return false; }

std::pair<double, double> LPNode::minmax(
        optional_cache_type<std::pair<double, double>> cache) const {
    return memoize(cache, [&]() {
        return std::make_pair(lb_ptr_ ? lb_ptr_->min() : DEFAULT_LOWER_BOUND,
                              ub_ptr_ ? ub_ptr_->max() : DEFAULT_UPPER_BOUND);
    });
}

double LPNode::objective_value(const dwave::optimization::State& state) const {
    return data_ptr<LPNodeData>(state)->objective_value;
}

void LPNode::propagate(State& state) const {
    auto data = data_ptr<LPNodeData>(state);

    copy_to_node_data(state, data->lp);
    SolveResult result = linprog(data->lp.c, data->lp.b_lb, data->lp.A, data->lp.b_ub,
                                 data->lp.A_eq, data->lp.b_eq, data->lp.lb, data->lp.ub);

    data->assign(result.solution());

    data->old_is_feasible = data->is_feasible;
    data->is_feasible = result.feasible();

    data->old_objective_value = data->objective_value;
    data->objective_value = result.objective();

    // update successors if there is anything to update about
    if (data->updates.size()) Node::propagate(state);
}

void LPNode::revert(State& state) const {
    auto data = data_ptr<LPNodeData>(state);
    data->is_feasible = data->old_is_feasible;
    data->objective_value = data->old_objective_value;

    return data->revert();
}

ObjectiveValueNode::ObjectiveValueNode(LPNode* lp_ptr) : lp_ptr_(lp_ptr) {
    add_predecessor(lp_ptr);
}

double const* ObjectiveValueNode::buff(const State& state) const {
    return data_ptr<ScalarNodeStateData>(state)->buff();
}

// no state to manage so nothing to do
void ObjectiveValueNode::commit(State& state) const {
    return data_ptr<ScalarNodeStateData>(state)->commit();
}

std::span<const Update> ObjectiveValueNode::diff(const State& state) const {
    return data_ptr<ScalarNodeStateData>(state)->diff();
}

void ObjectiveValueNode::initialize_state(State& state) const {
    int index = this->topological_index();
    assert(index >= 0 && "must be topologically sorted");
    assert(static_cast<int>(state.size()) > index && "unexpected state length");
    assert(state[index] == nullptr && "already initialized state");

    // if not feasible, we're undefined. So let's default to 0.0. Though it
    // might actually be better to default to our own maximum value.
    double value = lp_ptr_->feasible(state) ? lp_ptr_->objective_value(state) : 0;

    state[index] = std::make_unique<ScalarNodeStateData>(value);
}

void ObjectiveValueNode::propagate(State& state) const {
    if (lp_ptr_->feasible(state)) {
        data_ptr<ScalarNodeStateData>(state)->set(lp_ptr_->objective_value(state));
    }
    // Otherwise do nothing because we're not defined.
    // We could consider setting ourselves to max()
}

// no state to manage so nothing to do
void ObjectiveValueNode::revert(State& state) const {
    return data_ptr<ScalarNodeStateData>(state)->revert();
}

}  // namespace dwave::optimization

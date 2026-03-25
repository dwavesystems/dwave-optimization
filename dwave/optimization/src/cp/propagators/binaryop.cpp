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

#include "dwave-optimization/cp/propagators/binaryop.hpp"

#include <numeric>

#include "dwave-optimization/functional.hpp"

namespace dwave::optimization::cp {

template <class BinaryOp>
BinaryOpPropagator<BinaryOp>::BinaryOpPropagator(ssize_t index, CPVar* lhs, CPVar* rhs, CPVar* out)
        : Propagator(index), lhs_(lhs), rhs_(rhs), out_(out) {
    // TODO: support only static sizes for now
    if ((lhs_->max_size() != lhs_->min_size()) or (rhs_->max_size() != rhs_->min_size()) or
        (out_->max_size() != out_->min_size())) {
        throw std::invalid_argument("BinaryOpPropagator only supports static arrays currently");
    }

    // Default to bound-consistency
    Advisor lhs_advisor(this, 0, std::make_unique<ElementWiseTransform>());
    Advisor rhs_advisor(this, 1, std::make_unique<ElementWiseTransform>());
    Advisor out_advisor(this, 2, std::make_unique<ElementWiseTransform>());
    lhs_->propagate_on_bounds_change(std::move(lhs_advisor));
    rhs_->propagate_on_bounds_change(std::move(rhs_advisor));
    out_->propagate_on_bounds_change(std::move(out_advisor));
}

template <class BinaryOp>
void BinaryOpPropagator<BinaryOp>::initialize_state(CPState& state) const {
    CPPropagatorsState& p_state = state.get_propagators_state();
    assert(propagator_index_ >= 0);
    assert(propagator_index_ < static_cast<ssize_t>(p_state.size()));
    p_state[propagator_index_] =
            std::make_unique<PropagatorData>(state.get_state_manager(), out_->max_size());
}

template <class BinaryOp>
bool op_valid(double lhs, double rhs) {
    if constexpr (std::same_as<BinaryOp, std::divides<double>>) {
        return rhs != 0.0;
    }
    if constexpr (std::same_as<BinaryOp, std::multiplies<double>>) {
        return true;
    }
    if constexpr (std::same_as<BinaryOp, functional::safe_divides<double>>) {
        return true;
    }
    assert(false && "not implemeted");
    unreachable();
}

template <class BinaryOp>
std::pair<bool, CPStatus> make_bounds_consistent_all_combos(CPVar* lhs, CPVar* rhs, CPVar* out,
                                                            CPVarsState& v_state, ssize_t i) {
    auto op = BinaryOp();

    double lhs_min = lhs->min(v_state, i);
    double lhs_max = lhs->max(v_state, i);
    double rhs_min = rhs->min(v_state, i);
    double rhs_max = rhs->max(v_state, i);

    // These are simply safe starting values
    double out_min = out->max(v_state, i);
    double out_max = out->min(v_state, i);

    if (std::same_as<BinaryOp, std::divides<double>> and rhs_min <= 0 and rhs_max >= 0) {
        // rhs includes 0 with divide. This means there is no way to restrict the output
        // domain.
        return {false, CPStatus::OK};
    }

    if (op_valid<BinaryOp>(lhs_min, rhs_min)) {
        out_min = std::min(out_min, op(lhs_min, rhs_min));
        out_max = std::max(out_max, op(lhs_min, rhs_min));
    }
    if (op_valid<BinaryOp>(lhs_min, rhs_max)) {
        out_min = std::min(out_min, op(lhs_min, rhs_max));
        out_max = std::max(out_max, op(lhs_min, rhs_max));
    }
    if (op_valid<BinaryOp>(lhs_max, rhs_min)) {
        out_min = std::min(out_min, op(lhs_max, rhs_min));
        out_max = std::max(out_max, op(lhs_max, rhs_min));
    }
    if (op_valid<BinaryOp>(lhs_max, rhs_max)) {
        out_min = std::min(out_min, op(lhs_max, rhs_max));
        out_max = std::max(out_max, op(lhs_max, rhs_max));
    }

    if (std::same_as<BinaryOp, functional::safe_divides<double>> and rhs_min <= 0 and
        rhs_max >= 0) {
        // Includes zero, so the output domain must also include zero
        out_min = std::min(out_min, 0.0);
        out_max = std::max(out_max, 0.0);
    }

    // Prune the output
    if (out_min > out->min(v_state, i)) {
        if (CPStatus status = out->remove_below(v_state, out_min, i); not status)
            return {false, status};
        if (out_min != out->min(v_state, i)) return {true, CPStatus::OK};
    }

    if (out_max < out->max(v_state, i)) {
        if (CPStatus status = out->remove_above(v_state, out_max, i); not status)
            return {false, status};
        if (out_max != out->max(v_state, i)) return {true, CPStatus::OK};
    }

    return {false, CPStatus::OK};
}

template <>
CPStatus BinaryOpPropagator<std::multiplies<double>>::propagate(CPPropagatorsState& p_state,
                                                                CPVarsState& v_state) const {
    using op = std::multiplies<double>;
    using inverse = std::divides<double>;

    auto data = data_ptr<PropagatorData>(p_state);

    std::deque<ssize_t>& indices_to_process = data->indices_to_process();

    while (data->num_indices_to_process() > 0) {
        ssize_t i = indices_to_process.front();
        indices_to_process.pop_front();

        // Forward propagation
        if (auto [_, status] = make_bounds_consistent_all_combos<op>(lhs_, rhs_, out_, v_state, i);
            not status)
            return status;

        // Backward propagation to lhs
        auto [lhs_rounded, lhs_status] =
                make_bounds_consistent_all_combos<inverse>(out_, rhs_, lhs_, v_state, i);
        if (not lhs_status) return lhs_status;

        // Backward propagation to rhs
        auto [rhs_rounded, rhs_status] =
                make_bounds_consistent_all_combos<inverse>(out_, lhs_, rhs_, v_state, i);
        if (not rhs_status) return rhs_status;

        if (lhs_rounded or rhs_rounded) {
            // Forward again
            if (auto [_, status] =
                        make_bounds_consistent_all_combos<op>(lhs_, rhs_, out_, v_state, i);
                not status)
                return status;

            assert(not make_bounds_consistent_all_combos<inverse>(out_, rhs_, lhs_, v_state, i)
                               .first and
                   "backwards propagation to lhs still not fully consistent");
            assert(not make_bounds_consistent_all_combos<inverse>(out_, lhs_, rhs_, v_state, i)
                               .first and
                   "backwards propagation to rhs still not fully consistent");
        }

        data->set_scheduled(false, i);
    }
    return CPStatus::OK;
}

template <>
CPStatus BinaryOpPropagator<std::divides<double>>::propagate(CPPropagatorsState& p_state,
                                                             CPVarsState& v_state) const {
    using op = std::divides<double>;
    using inverse = std::multiplies<double>;

    auto data = data_ptr<PropagatorData>(p_state);

    std::deque<ssize_t>& indices_to_process = data->indices_to_process();

    while (data->num_indices_to_process() > 0) {
        ssize_t i = indices_to_process.front();
        indices_to_process.pop_front();

        // Forward propagation
        if (auto [_, status] = make_bounds_consistent_all_combos<op>(lhs_, rhs_, out_, v_state, i);
            not status)
            return status;

        // Backward propagation to lhs
        auto [lhs_rounded, lhs_status] =
                make_bounds_consistent_all_combos<inverse>(out_, rhs_, lhs_, v_state, i);
        if (not lhs_status) return lhs_status;

        // Backward propagation to rhs
        // NOTE: still using division here
        auto [rhs_rounded, rhs_status] =
                make_bounds_consistent_all_combos<op>(lhs_, out_, rhs_, v_state, i);
        if (not rhs_status) return rhs_status;

        if (lhs_rounded or rhs_rounded) {
            // Forward again
            if (auto [_, status] =
                        make_bounds_consistent_all_combos<op>(lhs_, rhs_, out_, v_state, i);
                not status)
                return status;

            assert(not make_bounds_consistent_all_combos<inverse>(out_, rhs_, lhs_, v_state, i)
                               .first and
                   "backwards propagation to lhs still not fully consistent");
            assert(not make_bounds_consistent_all_combos<op>(lhs_, out_, rhs_, v_state, i).first and
                   "backwards propagation to rhs still not fully consistent");
        }

        data->set_scheduled(false, i);
    }
    return CPStatus::OK;
}

template <>
CPStatus BinaryOpPropagator<std::plus<double>>::propagate(CPPropagatorsState& p_state,
                                                          CPVarsState& v_state) const {
    auto data = data_ptr<PropagatorData>(p_state);

    std::deque<ssize_t>& indices_to_process = data->indices_to_process();

    while (data->num_indices_to_process() > 0) {
        ssize_t i = indices_to_process.front();
        indices_to_process.pop_front();
        data->set_scheduled(false, i);

        // forward
        double min_i = lhs_->min(v_state, i) + rhs_->min(v_state, i);
        double max_i = lhs_->max(v_state, i) + rhs_->max(v_state, i);

        // prune the output
        if (CPStatus status = out_->remove_above(v_state, max_i, i); not status) return status;
        if (CPStatus status = out_->remove_below(v_state, min_i, i); not status) return status;

        // backward
        // prune lhs i
        double lhs_up = out_->max(v_state, i) - rhs_->min(v_state, i);
        double lhs_lo = out_->min(v_state, i) - rhs_->max(v_state, i);
        if (CPStatus status = lhs_->remove_above(v_state, lhs_up, i); not status) return status;
        if (CPStatus status = lhs_->remove_below(v_state, lhs_lo, i); not status) return status;

        // prube rhs i
        double rhs_up = out_->max(v_state, i) - lhs_->min(v_state, i);
        double rhs_lo = out_->min(v_state, i) - lhs_->max(v_state, i);

        if (CPStatus status = rhs_->remove_above(v_state, rhs_up, i); not status) return status;
        if (CPStatus status = rhs_->remove_below(v_state, rhs_lo, i); not status) return status;
    }
    return CPStatus::OK;
}

template <>
CPStatus BinaryOpPropagator<std::less_equal<double>>::propagate(CPPropagatorsState& p_state,
                                                                CPVarsState& v_state) const {
    auto data = data_ptr<PropagatorData>(p_state);
    assert(data->num_indices_to_process() > 0);

    std::deque<ssize_t>& indices_to_process = data->indices_to_process();

    while (data->num_indices_to_process() > 0) {
        ssize_t i = indices_to_process.front();
        indices_to_process.pop_front();
        data->set_scheduled(false, i);

        // forward
        {
            CPStatus status = CPStatus::OK;

            if (rhs_->max(v_state, i) < lhs_->min(v_state, i)) {
                status = out_->assign(v_state, 0, i);
            } else if (lhs_->max(v_state, i) <= rhs_->min(v_state, i)) {
                status = out_->assign(v_state, 1, i);
            }

            if (not status) return status;
        }

        // backward
        if (out_->min(v_state, i) == 1) {
            // then we can prune the domain to guarantee the constraint is satisfied
            if (CPStatus status = lhs_->remove_above(v_state, rhs_->max(v_state, i), i); not status)
                return status;
            if (CPStatus status = rhs_->remove_below(v_state, lhs_->min(v_state, i), i); not status)
                return status;

            this->set_active(p_state, lhs_->max(v_state, i) > rhs_->min(v_state, i), i);

        } else if (out_->max(v_state, i) == 0) {
            // if the max is 0, then I have to prune in the other directions, in order to guarantee
            // the constraint is always false

            if (CPStatus status = lhs_->remove_below(v_state, rhs_->min(v_state, i), i); not status)
                return status;
            if (CPStatus status = rhs_->remove_above(v_state, lhs_->max(v_state, i), i); not status)
                return status;

            // The constraint won't be active anymore, since after this pruning, the constraint
            // won't be able to prune further
            this->set_active(p_state, false, i);
        } else {
            // Cannot prune but the constraint will be active
            // and probably this instruction is useless
            this->set_active(p_state, true, i);
        }
    }

    return CPStatus::OK;
}

template class BinaryOpPropagator<std::plus<double>>;
template class BinaryOpPropagator<std::less_equal<double>>;
template class BinaryOpPropagator<std::multiplies<double>>;
template class BinaryOpPropagator<std::divides<double>>;

}  // namespace dwave::optimization::cp

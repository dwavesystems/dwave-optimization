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

#include "dwave-optimization/cp/propagators/reduce.hpp"

#include <algorithm>

namespace dwave::optimization::cp {
class SumPropagatorData : public PropagatorData {
 public:
    SumPropagatorData(StateManager* sm, CPVar* predecessor, const CPVarsState& state,
                      ssize_t constraint_size)
            : PropagatorData(sm, constraint_size) {
        assert(predecessor->min_size() == predecessor->max_size());
        auto n_inputs = predecessor->max_size();
        min.resize(n_inputs, 0);
        max.resize(n_inputs, 0);
        fixed_idx.resize(n_inputs);
        std::iota(fixed_idx.begin(), fixed_idx.end(), 0);

        for (ssize_t i = 0; i < n_inputs; ++i) {
            min[i] = predecessor->min(state, i);
            max[i] = predecessor->max(state, i);
        }

        n_fixed = sm->make_state_int(0);
        sum_fixed = sm->make_state_real(0);
    }

    // Indices of the predecessor that are fixed
    std::vector<ssize_t> fixed_idx;

    // How many indices of the predecessor array are fixed
    StateInt* n_fixed;

    // What is the sum of the fixed indices?
    StateReal* sum_fixed;

    // Cached min and max for the predecessor indices
    std::vector<double> min, max;
};

// ----- ReducePropagator -----

template <class BinaryOp>
ReducePropagator<BinaryOp>::ReducePropagator(ssize_t index, CPVar* in, CPVar* out)
        : Propagator(index), in_(in), out_(out) {
    // TODO: not supporting dynamic arrays for now
    if (in_->max_size() != in_->min_size()) {
        throw std::invalid_argument(
                "ReducePropagator not currently compatible with dynamic arrays");
    }

    // Default to bound consistency
    Advisor in_adv(this, 0, std::make_unique<ReduceTransform>());
    Advisor out_adv(this, 1, std::make_unique<ElementWiseTransform>());

    in_->propagate_on_bounds_change(std::move(in_adv));
    out_->propagate_on_bounds_change(std::move(out_adv));
}

template <>
void ReducePropagator<std::plus<double>>::initialize_state(CPState& state) const {
    CPPropagatorsState& p_state = state.get_propagators_state();
    CPVarsState& v_state = state.get_variables_state();
    assert(propagator_index_ >= 0);
    assert(propagator_index_ < static_cast<ssize_t>(p_state.size()));
    p_state[propagator_index_] =
            std::make_unique<SumPropagatorData>(state.get_state_manager(), in_, v_state, 1);
}

template <>
CPStatus ReducePropagator<std::plus<double>>::propagate(CPPropagatorsState& p_state,
                                                        CPVarsState& v_state) const {
    auto data = data_ptr<SumPropagatorData>(p_state);
    auto n = in_->max_size();

    CPStatus status = CPStatus::OK;

    std::deque<ssize_t>& indices_to_process = data->indices_to_process();

    while (data->num_indices_to_process() > 0) {
        ssize_t i = indices_to_process.front();
        indices_to_process.pop_front();
        data->set_scheduled(false, i);
        assert(i == 0);

        int nf = data->n_fixed->get_value();
        double sum_min = data->sum_fixed->get_value();
        double sum_max = data->sum_fixed->get_value();

        for (int j = nf; j < n; ++j) {
            int idx = data->fixed_idx[j];

            data->min[idx] = in_->min(v_state, idx);
            data->max[idx] = in_->max(v_state, idx);

            // update partial sum
            sum_min += data->min[idx];
            sum_max += data->max[idx];

            if (in_->is_bound(v_state, idx)) {
                data->sum_fixed->set_value(data->sum_fixed->get_value() + in_->min(v_state, idx));
                std::swap(data->fixed_idx[j], data->fixed_idx[nf]);
                nf++;
            }
        }

        data->n_fixed->set_value(nf);
        if ((sum_min > out_->max(v_state, 0)) or (sum_max < out_->min(v_state, 0))) {
            // TODO: I guess this may not happen in the DAG
            return CPStatus::Inconsistency;
        }

        status = out_->remove_above(v_state, sum_max, 0);
        if (status == CPStatus::Inconsistency) return status;

        status = out_->remove_below(v_state, sum_min, 0);
        if (status == CPStatus::Inconsistency) return status;

        double out_min = out_->min(v_state, 0);
        double out_max = out_->max(v_state, 0);

        for (int j = nf; j < n; ++j) {
            int idx = data->fixed_idx[j];

            status = in_->remove_above(v_state, out_max - (sum_min - data->min[idx]), idx);
            if (status == CPStatus::Inconsistency) return status;

            status = in_->remove_below(v_state, out_min - (sum_max - data->max[idx]), idx);
            if (status == CPStatus::Inconsistency) return status;
        }
    }

    return status;
}

template class ReducePropagator<std::plus<double>>;

// ----- DynamicReducePropagator -----

template <class BinaryOp>
DynamicReducePropagator<BinaryOp>::DynamicReducePropagator(ssize_t index, CPVar* in, CPVar* out)
        : Propagator(index), in_(in), out_(out) {
    // Default to bound consistency
    Advisor in_adv(this, 0, std::make_unique<ReduceTransform>());
    Advisor out_adv(this, 1, std::make_unique<ElementWiseTransform>());

    in_->propagate_on_bounds_change(std::move(in_adv));
    out_->propagate_on_bounds_change(std::move(out_adv));
}

template <class BinaryOp>
void DynamicReducePropagator<BinaryOp>::initialize_state(CPState& state) const {
    CPPropagatorsState& p_state = state.get_propagators_state();
    assert(propagator_index_ >= 0);
    assert(propagator_index_ < static_cast<ssize_t>(p_state.size()));
    p_state[propagator_index_] = std::make_unique<PropagatorData>(state.get_state_manager(), 1);
}

/// Compute the upper and lower bounds derived from the definitely present variables
std::pair<double, double> compute_present_bounds(const CPVar* in, const CPVar* out,
                                                 const CPVarsState& state, ssize_t start_index,
                                                 double lb_start, double ub_start) {
    // assert(start_index < in->min_size(state));

    double lb_present = lb_start;
    double ub_present = ub_start;
    for (ssize_t j = start_index; j < in->min_size(state); ++j) {
        assert(in->is_active(state, j));
        lb_present += in->min(state, j);
        ub_present += in->max(state, j);
    }
    assert(lb_present <= ub_present);
    return {lb_present, ub_present};
}

/// TODO: can make it more C++ :P
void compute_bounds_for_size(const CPVar* in, const CPVar* out, const CPVarsState& state,
                             std::vector<double>& lb_optional, std::vector<double>& ub_optional,
                             std::vector<double>& lb_acc, std::vector<double>& ub_acc,
                             double lb_present, double ub_present) {
    // the next two variables used to accumulate the values
    double acc_lb = lb_present;
    double acc_ub = ub_present;

    lb_optional.push_back(lb_present);
    ub_optional.push_back(ub_present);
    lb_acc.push_back(lb_present);
    ub_acc.push_back(ub_present);

    for (ssize_t j = in->min_size(state); j < in->max_size(state); ++j) {
        assert(not in->is_active(state, j));
        assert(in->maybe_active(state, j));

        lb_optional.push_back(acc_lb + std::min(0., in->min(state, j)));
        ub_optional.push_back(acc_ub + std::max(0., in->max(state, j)));

        acc_lb += in->min(state, j);
        acc_ub += in->max(state, j);

        // accumulate the lower
        lb_acc.push_back(acc_lb);
        ub_acc.push_back(acc_ub);

        assert(acc_lb <= acc_ub);
    }
    assert(ub_acc.size() == lb_acc.size());
    assert(ub_acc.size() == lb_optional.size());
    assert(ub_acc.size() == ub_optional.size());
}

template <>
CPStatus DynamicReducePropagator<std::plus<double>>::propagate(CPPropagatorsState& p_state,
                                                               CPVarsState& v_state) const {
    auto data = data_ptr<SumPropagatorData>(p_state);

    CPStatus status = CPStatus::OK;

    std::deque<ssize_t>& indices_to_process = data->indices_to_process();

    while (data->num_indices_to_process() > 0) {
        ssize_t i = indices_to_process.front();
        indices_to_process.pop_front();
        data->set_scheduled(false, i);
        assert(i == 0);

        /// ==== Forward propagation ====

        /// Compute the output interval induced by the present variables
        double lb_present, ub_present;
        std::tie(lb_present, ub_present) = compute_present_bounds(in_, out_, v_state, 0, 0, 0);

        // The following vectors store the lower and upper bounds calculations for each size of the
        // vector.
        //  They are used for
        // 1. Binding the out_ min and max (fwd)
        // 2. Binding in_ min size and max size (bck).
        // 3. Binding in_ min and max for every index (bck).
        std::vector<double> lb_optional, lb_acc;
        std::vector<double> ub_optional, ub_acc;

        // Fill the vectors above
        compute_bounds_for_size(in_, out_, v_state, lb_optional, ub_optional, lb_acc, ub_acc,
                                lb_present, ub_present);

        // Prune out_ min value
        if (CPStatus status = out_->remove_below(
                    v_state, *std::min_element(lb_optional.begin(), lb_optional.end()), i);
            not status)
            return status;

        // Prune out_ max value
        if (CPStatus status = out_->remove_above(
                    v_state, *std::max_element(ub_optional.begin(), ub_optional.end()), i);
            not status)
            return status;

        /// ==== End of forward propagation ====

        /// ==== Backward propagation ====

        // Attempt to prune the size of in_.
        // We start from min_size < max_size and we loop until we find a valid size. If no valid
        // size found, we'll set some inconsistent values, and we return an inconsistency
        ssize_t min_size_new = in_->max_size(v_state) + 1;
        ssize_t max_size_new = in_->min_size(v_state) - 1;
        ssize_t initial_min_size = in_->min_size(v_state);

        double out_min = out_->min(v_state, 0);
        double out_max = out_->max(v_state, 0);

        for (ssize_t j = 0; j < static_cast<ssize_t>(lb_optional.size()); ++j) {
            // Update the min size if the domains overlap
            if (not(ub_optional[j] < out_min or out_max < lb_optional[j])) {
                min_size_new = std::min(in_->min_size(v_state) + j, min_size_new);
            }

            // Update the max size if the domains overlap
            if (not(ub_acc[j] < out_min or out_max < lb_acc[j])) {
                max_size_new = std::max(in_->min_size(v_state) + j, max_size_new);
            }
        }

        assert(min_size_new >= in_->min_size(v_state));
        bool changed_min_size = (min_size_new > in_->min_size(v_state)) ? true : false;

        // After this we set the min and max size
        if (CPStatus status = in_->set_min_size(v_state, min_size_new); not status) return status;

        if (CPStatus status = in_->set_max_size(v_state, max_size_new); not status) return status;

        // Now prune the values for each entry
        // They might have changed, so need to recompute.
        if (changed_min_size) {
            // update the bounds from present variables
            std::tie(lb_present, ub_present) = compute_present_bounds(
                    in_, out_, v_state, initial_min_size, lb_present, ub_present);

            // also update the bounds from optional variables, used to prune the
            lb_optional.clear();
            ub_optional.clear();
            lb_acc.clear();
            ub_acc.clear();
            compute_bounds_for_size(in_, out_, v_state, lb_optional, ub_optional, lb_acc, ub_acc,
                                    lb_present, ub_present);
        }

        // Prune the present variables
        // first compute the widest possible sum that there can be gven present and optional
        // variables
        {
            double sum_max = *std::max_element(ub_optional.begin(), ub_optional.end());
            double sum_min = *std::min_element(lb_optional.begin(), lb_optional.end());
            for (ssize_t j = 0; j < in_->min_size(v_state); ++j) {
                assert(in_->is_active(v_state, j));
                if (CPStatus status = in_->remove_above(
                            v_state, out_->max(v_state, 0) - (sum_min - in_->min(v_state, j)), j);
                    not status)
                    return status;

                if (CPStatus status = in_->remove_below(
                            v_state, out_->min(v_state, 0) - (sum_max - in_->max(v_state, j)), j);
                    not status)
                    return status;
            }
        }

        // Prune the optional variables, they receive the bounds from as if they were present and
        // the following optional will contribute with either present or not, whichever yields
        // bigger gap
        {
            for (ssize_t j = 0; j < in_->max_size(v_state) - in_->min_size(v_state); ++j) {
                assert(not in_->is_active(v_state, in_->min_size(v_state) + j));
                assert(in_->maybe_active(v_state, in_->min_size(v_state) + j));

                auto smin_it = std::min_element(lb_optional.begin() + j + 1, lb_optional.end());
                auto smax_it = std::max_element(ub_optional.begin() + j + 1, ub_optional.end());

                double sum_min = (smin_it != lb_optional.end()) ? *smin_it : lb_optional.back();
                double sum_max = (smax_it != ub_optional.end()) ? *smax_it : ub_optional.back();

                if (CPStatus status = in_->remove_above(
                            v_state, out_->max(v_state, 0) - (sum_min - in_->min(v_state, j)), j);
                    not status)
                    return status;

                if (CPStatus status = in_->remove_below(
                            v_state, out_->min(v_state, 0) - (sum_max - in_->max(v_state, j)), j);
                    not status)
                    return status;
            }
        }
    }

    return CPStatus::OK;
}

template class DynamicReducePropagator<std::plus<double>>;

}  // namespace dwave::optimization::cp

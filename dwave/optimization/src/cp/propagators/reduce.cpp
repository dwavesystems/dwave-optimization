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

}  // namespace dwave::optimization::cp

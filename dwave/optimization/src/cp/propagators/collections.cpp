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

#include "dwave-optimization/cp/propagators/collections.hpp"

#include <numeric>

namespace dwave::optimization::cp {

class AllDifferentFWCPropagatorData : public PropagatorData {
 public:
    AllDifferentFWCPropagatorData(StateManager* sm, CPVar* array, ssize_t constraint_size)
            : PropagatorData(sm, constraint_size) {
        fixed.resize(array->max_size());
        std::iota(fixed.begin(), fixed.end(), 0);
        n_fixed = sm->make_state_int(0);
        assert(n_fixed->get_value() == 0);
    }

    std::vector<ssize_t> fixed;
    StateInt* n_fixed;
};

AllDifferentFWCPropagator::AllDifferentFWCPropagator(ssize_t index, CPVar* array)
        : Propagator(index), array_(array) {
    // Reduce Transform because "There is one index of this propagator" and all components of the
    // array map to this one index
    Advisor adv(this, 0, std::make_unique<ReduceTransform>());
    array_->propagate_on_assignment(std::move(adv));
}

void AllDifferentFWCPropagator::initialize_state(CPState& state) const {
    CPPropagatorsState& p_state = state.get_propagators_state();
    // CPVarsState& v_state = state.get_variables_state();
    assert(propagator_index_ >= 0);
    assert(propagator_index_ < static_cast<ssize_t>(p_state.size()));
    p_state[propagator_index_] =
            std::make_unique<AllDifferentFWCPropagatorData>(state.get_state_manager(), array_, 1);
}

CPStatus AllDifferentFWCPropagator::propagate(CPPropagatorsState& p_state,
                                              CPVarsState& v_state) const {
    auto data = data_ptr<AllDifferentFWCPropagatorData>(p_state);

    std::deque<ssize_t>& indices_to_process = data->indices_to_process();

    while (data->num_indices_to_process() > 0) {
        ssize_t i = indices_to_process.front();
        indices_to_process.pop_front();
        data->set_scheduled(false, i);
        assert(i == 0);

        ssize_t num_fixed = data->n_fixed->get_value();
        auto& fixed = data->fixed;
        ssize_t n = fixed.size();

        for (ssize_t j = num_fixed; j < n; ++j) {
            ssize_t idx = data->fixed[j];

            if (array_->is_bound(v_state, idx)) {
                fixed[j] = fixed[num_fixed];
                fixed[num_fixed] = idx;
                ++num_fixed;

                // remove the values
                for (int k = num_fixed; k < n; ++k) {
                    int idx_to_remove = fixed[k];
                    auto val_to_remove = array_->min(v_state, j);
                    if (CPStatus status = array_->remove(v_state, val_to_remove, idx_to_remove);
                        not status)
                        return status;
                }
            }
        }

        data->n_fixed->set_value(num_fixed);
    }
    return CPStatus::OK;
}
}  // namespace dwave::optimization::cp

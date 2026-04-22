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

#include "dwave-optimization/cp/propagators/indexing_propagators.hpp"

#include "dwave-optimization/nodes/indexing.hpp"

namespace dwave::optimization::cp {

BasicIndexingForwardTransform::BasicIndexingForwardTransform(const ArrayNode* array_ptr,
                                                             const BasicIndexingNode* bi_ptr)
        : array_ptr_(array_ptr), bi_ptr_(bi_ptr) {
    slices = bi_ptr->infer_indices();
    for (ssize_t axis = 0; axis < array_ptr->ndim(); ++axis) {
        if (std::holds_alternative<Slice>(slices[axis])) {
            if (std::get<Slice>(slices[axis]).step != 1) {
                throw std::invalid_argument("step != 1 not supported");
            }

            slices[axis] = std::get<Slice>(slices[axis]).fit(array_ptr->shape()[axis]);
        }
    }
}

void BasicIndexingForwardTransform::affected(ssize_t i, std::vector<ssize_t>& out) {
    std::vector<ssize_t> in_multi_index = unravel_index(i, array_ptr_->shape());
    std::vector<ssize_t> out_multi_index;
    bool belongs = true;
    // Iterate through the axes to see if any index is outside the slice
    for (ssize_t axis = 0; axis < array_ptr_->ndim(); ++axis) {
        if (std::holds_alternative<ssize_t>(slices[axis])) {
            if (in_multi_index[axis] == std::get<ssize_t>(slices[axis])) continue;
        } else {
            const auto& slice = std::get<Slice>(slices[axis]);
            if (in_multi_index[axis] >= slice.start and in_multi_index[axis] < slice.stop) {
                out_multi_index.push_back(in_multi_index[axis] - slice.start);
                continue;
            }
        }

        belongs = false;
        break;
    }

    if (belongs) {
        out.push_back(ravel_multi_index(out_multi_index, bi_ptr_->shape()));
    }
}

BasicIndexingPropagator::BasicIndexingPropagator(ssize_t index, CPVar* array, CPVar* basic_indexing)
        : Propagator(index) {
    // TODO: not supporting dynamic variables for now
    if (array->min_size() != array->max_size()) {
        throw std::invalid_argument("dynamic arrays not supported");
    }

    array_ = array;
    basic_indexing_ = basic_indexing;
}

void BasicIndexingPropagator::initialize_state(CPState& state) const {
    CPPropagatorsState& p_state = state.get_propagators_state();
    assert(propagator_index_ >= 0);
    assert(propagator_index_ < static_cast<ssize_t>(p_state.size()));
    p_state[propagator_index_] = std::make_unique<PropagatorData>(state.get_state_manager(),
                                                                  basic_indexing_->max_size());
}

CPStatus BasicIndexingPropagator::propagate(CPPropagatorsState& p_state,
                                            CPVarsState& v_state) const {
    auto data = data_ptr<PropagatorData>(p_state);

    const BasicIndexingNode* bi = dynamic_cast<const BasicIndexingNode*>(basic_indexing_->node_);
    assert(bi);

    // Not caching this for now as we may need to fit these at propagate time for
    // dynamic arrays
    std::vector<BasicIndexingNode::slice_or_int> slices = bi->infer_indices();
    for (ssize_t axis = 0; axis < array_->node_->ndim(); ++axis) {
        if (std::holds_alternative<Slice>(slices[axis])) {
            assert(std::get<Slice>(slices[axis]).step == 1);
            slices[axis] = std::get<Slice>(slices[axis]).fit(array_->node_->shape()[axis]);
        }
    }

    std::deque<ssize_t>& indices_to_process = data->indices_to_process();

    assert(indices_to_process.size() > 0);
    while (indices_to_process.size() > 0) {
        ssize_t bi_index = indices_to_process.front();
        indices_to_process.pop_front();

        // Derive the original array index based on the index of the basic indexing variable.
        // We unravel the basic indexing variable index, transform the multi-index into
        // one on the original array, and then ravel it to get the final linear index on
        // the array.
        std::vector<ssize_t> bi_multi_index =
                unravel_index(bi_index, basic_indexing_->node_->shape());
        std::vector<ssize_t> arr_multi_index;
        ssize_t bi_axis = 0;
        for (ssize_t axis = 0; axis < array_->node_->ndim(); ++axis) {
            if (std::holds_alternative<ssize_t>(slices[axis])) {
                arr_multi_index.push_back(std::get<ssize_t>(slices[axis]));
                continue;
            }
            assert(std::holds_alternative<Slice>(slices[axis]));
            const auto& slice = std::get<Slice>(slices[axis]);
            assert(slice.step == 1);
            arr_multi_index.push_back(bi_multi_index[bi_axis] + slice.start);
            bi_axis++;
        }
        ssize_t array_index = ravel_multi_index(arr_multi_index, array_->node_->shape());

        // Now we make the bounds of the array element and the basic indexing element equal

        // Make the upper bounds consistent
        if (CPStatus status = basic_indexing_->remove_above(
                    v_state, array_->max(v_state, array_index), bi_index);
            not status)
            return status;
        if (CPStatus status = array_->remove_above(v_state, basic_indexing_->max(v_state, bi_index),
                                                   array_index);
            not status)
            return status;

        // Make the lower bounds consistent
        if (CPStatus status = basic_indexing_->remove_below(
                    v_state, array_->min(v_state, array_index), bi_index);
            not status)
            return status;
        if (CPStatus status = array_->remove_below(v_state, basic_indexing_->min(v_state, bi_index),
                                                   array_index);
            not status)
            return status;

        data->set_scheduled(false, bi_index);
    }

    return CPStatus::OK;
}

}  // namespace dwave::optimization::cp

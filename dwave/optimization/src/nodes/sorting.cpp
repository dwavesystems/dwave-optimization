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

#include "dwave-optimization/nodes/sorting.hpp"

#include "_state.hpp"

namespace dwave::optimization {

/// ArgSortNode

ArgSortNode::ArgSortNode(ArrayNode* arr_ptr)
        : ArrayOutputMixin(arr_ptr->shape()), arr_ptr_(arr_ptr) {
    add_predecessor(arr_ptr);
}

double const* ArgSortNode::buff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->buff();
}

void ArgSortNode::commit(State& state) const { data_ptr<ArrayNodeStateData>(state)->commit(); }

std::span<const Update> ArgSortNode::diff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->diff();
}

void ArgSortNode::initialize_state(State& state) const {
    const Array::View arr = arr_ptr_->view(state);

    std::vector<double> values(arr_ptr_->size(state));
    std::iota(values.begin(), values.end(), 0);

    auto arg_compare = [&arr](double x, double y) { return arr[x] < arr[y]; };

    std::stable_sort(values.begin(), values.end(), arg_compare);

    emplace_data_ptr<ArrayNodeStateData>(state, std::move(values));
}

bool ArgSortNode::integral() const { return arr_ptr_->integral(); }

std::pair<double, double> ArgSortNode::minmax(
        optional_cache_type<std::pair<double, double>> cache) const {
    return memoize(cache, [&]() {
        return std::make_pair(0.0, static_cast<double>(arr_ptr_->sizeinfo().max.value_or(
                                           std::numeric_limits<ssize_t>::max())));
    });
}

void ArgSortNode::propagate(State& state) const {
    const Array::View arr = arr_ptr_->view(state);

    auto node_data = data_ptr<ArrayNodeStateData>(state);

    std::vector<double> values(arr_ptr_->size(state));
    std::iota(values.begin(), values.end(), 0);

    auto arg_compare = [&arr](double x, double y) { return arr[x] < arr[y]; };

    std::stable_sort(values.begin(), values.end(), arg_compare);

    node_data->assign(values);
}

void ArgSortNode::revert(State& state) const { data_ptr<ArrayNodeStateData>(state)->revert(); }

std::span<const ssize_t> ArgSortNode::shape(const State& state) const {
    return arr_ptr_->shape(state);
}

ssize_t ArgSortNode::size(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->size();
}

ssize_t ArgSortNode::size_diff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->size_diff();
}

SizeInfo ArgSortNode::sizeinfo() const { return arr_ptr_->sizeinfo(); }

}  // namespace dwave::optimization

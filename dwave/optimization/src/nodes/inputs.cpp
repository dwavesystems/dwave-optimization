// Copyright 2024 D-Wave Inc.
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

#include "dwave-optimization/nodes/inputs.hpp"

#include "_state.hpp"
#include "dwave-optimization/array.hpp"
#include "dwave-optimization/state.hpp"
#include "dwave-optimization/utils.hpp"

namespace dwave::optimization {

void InputNode::assign(State& state, std::span<const double> new_values) const {
    if (static_cast<ssize_t>(new_values.size()) != this->size()) {
        throw std::invalid_argument("size of new values must match");
    }

    if (!new_values.empty()) {
        if (std::ranges::min(new_values) < min()) {
            throw std::invalid_argument("new data contains a value smaller than the min");
        }
        if (std::ranges::max(new_values) > max()) {
            throw std::invalid_argument("new data contains a value larger than the max");
        }
        if (integral() && !std::ranges::all_of(new_values, is_integer)) {
            throw std::invalid_argument("new data contains a non-integral value");
        }
    }

    data_ptr<ArrayNodeStateData>(state)->assign(new_values);
}

double const* InputNode::buff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->buff();
}

void InputNode::commit(State& state) const noexcept {
    data_ptr<ArrayNodeStateData>(state)->commit();
}

std::span<const Update> InputNode::diff(const State& state) const noexcept {
    return data_ptr<ArrayNodeStateData>(state)->diff();
}

void InputNode::initialize_state(State& state, std::span<const double> data) const {
    int index = this->topological_index();
    assert(index >= 0 && "must be topologically sorted");
    assert(static_cast<int>(state.size()) > index && "unexpected state length");
    assert(state[index] == nullptr && "already initialized state");

    if (static_cast<ssize_t>(data.size()) != this->size()) {
        throw std::invalid_argument("data size does not match size of InputNode");
    }

    std::vector<double> copy(data.begin(), data.end());

    state[index] = std::make_unique<ArrayNodeStateData>(std::move(copy));
}

void InputNode::revert(State& state) const noexcept {
    data_ptr<ArrayNodeStateData>(state)->revert();
}

}  // namespace dwave::optimization

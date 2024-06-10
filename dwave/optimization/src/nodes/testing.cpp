// Copyright 2024 D-Wave Systems Inc.
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

#include "dwave-optimization/nodes/testing.hpp"

namespace dwave::optimization {

class ArrayValidationNodeData : public dwave::optimization::NodeStateData {
 public:
    ArrayValidationNodeData(Array::View data)
            : old_data(data.begin(), data.end()), current_data(data.begin(), data.end()){};

    std::vector<double> old_data;
    std::vector<double> current_data;
};

void check_shape(const std::span<const ssize_t>& dynamic_shape,
                 const std::span<const ssize_t> shape, ssize_t expected_size) {
    ssize_t size = 1;
    assert(shape.size() == dynamic_shape.size());
    for (ssize_t axis = 0; axis < static_cast<ssize_t>(shape.size()); ++axis) {
        if (axis >= 1) assert(shape[axis] == dynamic_shape[axis]);
        size *= dynamic_shape[axis];
    }
    assert(size == expected_size);
}

void ArrayValidationNode::commit(State& state) const {
    auto node_data = data_ptr<ArrayValidationNodeData>(state);
    assert(array_ptr->diff(state).size() == 0);
    assert(array_ptr->size_diff(state) == 0);
    assert(array_ptr->size(state) == static_cast<ssize_t>(node_data->current_data.size()));
    assert(std::ranges::equal(array_ptr->view(state), node_data->current_data));
    check_shape(array_ptr->shape(state), array_ptr->shape(), node_data->current_data.size());
    node_data->old_data = node_data->current_data;
}

void ArrayValidationNode::initialize_state(State& state) const {
    state[topological_index()] = std::make_unique<ArrayValidationNodeData>(array_ptr->view(state));
    assert(array_ptr->diff(state).size() == 0);
    assert(array_ptr->size_diff(state) == 0);
    assert(array_ptr->view(state).size() == static_cast<ssize_t>(array_ptr->size(state)));
}

void ArrayValidationNode::propagate(State& state) const {
    auto node_data = data_ptr<ArrayValidationNodeData>(state);
    std::string node_id = typeid(*node_ptr).name() + std::string{"["} +
                          std::to_string(node_ptr->topological_index()) + std::string{"]"};

    auto& current_data = node_data->current_data;
    bool incorrect = false;

    for (const auto& update : array_ptr->diff(state)) {
        const ssize_t size = current_data.size();
        if (update.placed()) {
            // Can only place at the end of an array
            if (update.index != size) {
                do_logging&& std::cout
                        << node_id << " | placement on index which doesn't match size: " << update
                        << ", size=" << size << "\n";
                incorrect = true;
            } else {
                current_data.push_back(update.value);
            }
        } else if (update.removed()) {
            // Can only remove at the end of an array
            if (update.index != size - 1) {
                do_logging&& std::cout << node_id
                                       << " | removal on index which doesn't match size: " << update
                                       << ", size=" << size << "\n";
                incorrect = true;
            } else {
                if (update.old != current_data[update.index]) {
                    do_logging&& std::cout << node_id
                                           << " | removal with incorrect `old` value: " << update
                                           << "\n";
                    incorrect = true;
                }
                current_data.pop_back();
            }
        } else {
            if (update.index < 0 || update.index >= size) {
                do_logging&& std::cout << node_id
                                       << " | index on update outside of array: " << update
                                       << ", size=" << size << "\n";
                incorrect = true;
            } else {
                if (update.old != current_data[update.index]) {
                    do_logging&& std::cout << node_id << " | `old` value incorrect: " << update
                                           << ", size=" << size << "\n";
                    incorrect = true;
                }
                current_data[update.index] = update.value;
            }
        }
    }

    std::vector<double> expected(array_ptr->view(state).begin(), array_ptr->view(state).end());

    if (!std::ranges::equal(current_data, expected)) {
        if (do_logging) {
            std::cout << node_id
                      << " | Applying diff produced the wrong array. Expected data based on "
                         "current view:\n[";
            for (const auto& v : expected) {
                std::cout << v << " ";
            }
            std::cout << "]\n" << node_id << " | Diffs produced array:\n[";
            for (const auto& v : current_data) {
                std::cout << v << " ";
            }
            std::cout << "]\n";
        }
        incorrect = true;
    }

    if (incorrect) {
        if (do_logging) {
            std::cout << node_id << " | Previous array values:\n[";
            for (const auto& v : node_data->old_data) std::cout << v << " ";

            std::cout << "]\n" << node_id << " | Current array values:\n[";
            for (const auto& v : expected) std::cout << v << " ";

            std::cout << "]\n" << node_id << " | Updates in diff:\n";
            for (const auto& update : array_ptr->diff(state)) std::cout << update << "\n";

            std::cout << "\n";
        }

        assert(false && "ArrayValidationNode caught incorrect diff");
    }

    assert(static_cast<ssize_t>(expected.size()) -
                   static_cast<ssize_t>(node_data->old_data.size()) ==
           array_ptr->size_diff(state));

    assert(array_ptr->size(state) == static_cast<ssize_t>(expected.size()));

    check_shape(array_ptr->shape(state), array_ptr->shape(), node_data->current_data.size());

    current_data = expected;
}

void ArrayValidationNode::revert(State& state) const {
    auto node_data = data_ptr<ArrayValidationNodeData>(state);
    assert(array_ptr->diff(state).size() == 0);
    assert(array_ptr->size_diff(state) == 0);
    assert(array_ptr->size(state) == static_cast<ssize_t>(node_data->old_data.size()));
    assert(std::ranges::equal(array_ptr->view(state), node_data->old_data));
    check_shape(array_ptr->shape(state), array_ptr->shape(), node_data->old_data.size());
    node_data->current_data = node_data->old_data;
}

class DynamicArrayTestingNodeData : public dwave::optimization::NodeStateData {
 public:
    DynamicArrayTestingNodeData() {}

    explicit DynamicArrayTestingNodeData(const std::span<const ssize_t> shape)
            : current_shape(shape.begin(), shape.end()), old_shape(shape.begin(), shape.end()) {
        assert(shape.size() > 0);
    }

    DynamicArrayTestingNodeData(const std::span<const ssize_t> shape,
                                const std::span<const double> values)
            : DynamicArrayTestingNodeData(shape) {
        current_data.insert(current_data.begin(), values.begin(), values.end());
        old_data = current_data;
    }

    void commit() {
        diff.clear();
        old_data = current_data;
        old_shape = current_shape;
    }

    // Add one row to the data
    void grow(std::span<const double> values) {
        // Caller is responsible for mutating the `current_shape` to reflect the size
        // of the added data
        for (const double& value : values) {
            diff.emplace_back(Update::placement(current_data.size(), value));
            current_data.emplace_back(value);
        }
    }

    std::span<const ssize_t> shape() const { return current_shape; }

    void revert() {
        diff.clear();
        current_data = old_data;
        current_shape = old_shape;
    }

    void set(ssize_t index, double value) {
        assert(index >= 0 && index < static_cast<ssize_t>(current_data.size()));
        diff.emplace_back(index, current_data[index], value);
        current_data[index] = value;
    }

    // Remove n elements from the data, corresponding to one row.
    void shrink(ssize_t n) {
        assert(n <= static_cast<ssize_t>(current_data.size()));
        for (ssize_t i = 0; i < n; ++i) {
            diff.emplace_back(Update::removal(current_data.size() - 1, current_data.back()));
            current_data.pop_back();
        }
        current_shape[0] -= 1;
    }

    std::vector<double> old_data;
    std::vector<double> current_data;
    std::vector<Update> diff;
    std::vector<ssize_t> current_shape;
    std::vector<ssize_t> old_shape;
};

DynamicArrayTestingNode::DynamicArrayTestingNode(std::initializer_list<ssize_t> shape)
        : Node(), ArrayOutputMixin(shape), shape_(shape) {
    if (shape.size() == 0 || *shape.begin() != -1) {
        throw std::invalid_argument(
                "DynamicArrayTestingNode is meant to be used as a dynamic array");
    }
}

void DynamicArrayTestingNode::initialize_state(State& state) const { initialize_state(state, {}); }

void DynamicArrayTestingNode::initialize_state(State& state,
                                               std::initializer_list<double> values) const {
    initialize_state(state, std::span(values));
}

void DynamicArrayTestingNode::initialize_state(State& state, std::span<const double> values) const {
    std::vector<ssize_t> shape;
    for (auto dim : this->shape()) {
        shape.emplace_back(dim);
    }

    assert(values.size() % (strides()[0] / itemsize()) == 0);
    assert(shape.size() > 0);
    shape[0] = values.size() / (strides()[0] / itemsize());

    state[topological_index()] = std::make_unique<DynamicArrayTestingNodeData>(shape, values);
}

double const* DynamicArrayTestingNode::buff(const State& state) const noexcept {
    return &(data_ptr<DynamicArrayTestingNodeData>(state)->current_data[0]);
}

std::span<const Update> DynamicArrayTestingNode::diff(const State& state) const {
    return data_ptr<DynamicArrayTestingNodeData>(state)->diff;
}

ssize_t DynamicArrayTestingNode::size(const State& state) const {
    return data_ptr<DynamicArrayTestingNodeData>(state)->current_data.size();
}

std::span<const ssize_t> DynamicArrayTestingNode::shape(const State& state) const {
    return data_ptr<DynamicArrayTestingNodeData>(state)->shape();
}

ssize_t DynamicArrayTestingNode::size_diff(const State& state) const {
    auto node_data = data_ptr<DynamicArrayTestingNodeData>(state);

    return node_data->current_data.size() - node_data->old_data.size();
}

void DynamicArrayTestingNode::commit(State& state) const {
    data_ptr<DynamicArrayTestingNodeData>(state)->commit();
}

void DynamicArrayTestingNode::revert(State& state) const {
    data_ptr<DynamicArrayTestingNodeData>(state)->revert();
}

void DynamicArrayTestingNode::update(State&, int) const {}

void DynamicArrayTestingNode::grow(State& state, std::initializer_list<double> values) const {
    grow(state, std::span(values));
}

void DynamicArrayTestingNode::grow(State& state, std::span<const double> values) const {
    assert(ndim() >= 1);
    assert(values.size() % (strides()[0] / itemsize()) == 0);

    auto node_data = data_ptr<DynamicArrayTestingNodeData>(state);
    node_data->grow(values);
    node_data->current_shape[0] += values.size() / (strides()[0] / itemsize());
}

void DynamicArrayTestingNode::random_move(State& state, RngAdaptor& rng) const {
    auto node_data = data_ptr<DynamicArrayTestingNodeData>(state);

    assert(ndim() > 0);

    const ssize_t row_size = strides()[0] / itemsize();

    // Could change this depending on desired number type
    std::uniform_real_distribution<double> val_dist(-10, 10);

    size_t change_type = std::uniform_int_distribution<size_t>(0, 2)(rng);
    if (change_type == 0) {
        // Random placement
        std::vector<double> new_row;
        for (ssize_t i = 0; i < row_size; ++i) {
            new_row.emplace_back(val_dist(rng));
        }
        grow(state, new_row);
    } else if (change_type == 1 && node_data->current_data.size() > 0) {
        // Random removal
        shrink(state);
    } else if (node_data->current_data.size() > 0) {
        // Random update at a random index
        ssize_t index =
                std::uniform_int_distribution<ssize_t>(0, node_data->current_data.size() - 1)(rng);
        set(state, index, val_dist(rng));
    }
}

void DynamicArrayTestingNode::random_moves(State& state, RngAdaptor& rng,
                                           size_t max_changes) const {
    size_t num_changes = std::uniform_int_distribution<size_t>(0, max_changes)(rng);
    for (size_t i = 0; i < num_changes; ++i) {
        random_move(state, rng);
    }
}

void DynamicArrayTestingNode::set(State& state, ssize_t index, double value) const {
    data_ptr<DynamicArrayTestingNodeData>(state)->set(index, value);
}

void DynamicArrayTestingNode::shrink(State& state) const {
    const ssize_t row_size = strides()[0] / itemsize();
    data_ptr<DynamicArrayTestingNodeData>(state)->shrink(row_size);
}

void DynamicArrayTestingNode::default_move(State& state, RngAdaptor& rng) const {
    random_move(state, rng);
}

}  // namespace dwave::optimization

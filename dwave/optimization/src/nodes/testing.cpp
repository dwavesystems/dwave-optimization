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

#include <iostream>
#include <ranges>

#include "dwave-optimization/utils.hpp"

namespace dwave::optimization {

class ArrayValidationNodeData : public dwave::optimization::NodeStateData {
 public:
    explicit ArrayValidationNodeData(Array::View data)
            : old_data(data.begin(), data.end()), current_data(data.begin(), data.end()) {}

    std::vector<double> old_data;
    std::vector<double> current_data;
};

void check_shape(const std::span<const ssize_t>& dynamic_shape,
                 const std::span<const ssize_t> shape, ssize_t expected_size) {
    assert(shape.size() == dynamic_shape.size());
    assert(([&dynamic_shape, &shape, &expected_size]() -> bool {
        ssize_t size = 1;
        for (ssize_t axis = 0; axis < static_cast<ssize_t>(shape.size()); ++axis) {
            if (axis >= 1 && shape[axis] != dynamic_shape[axis]) return false;
            size *= dynamic_shape[axis];
        }
        return size == expected_size;
    })());
}

ArrayValidationNode::ArrayValidationNode(ArrayNode* node_ptr) : array_ptr(node_ptr) {
    assert(array_ptr->ndim() == static_cast<ssize_t>(array_ptr->shape().size()));
    assert(array_ptr->dynamic() == (array_ptr->size() == -1));
    assert([&]() {
        node_ptr->sizeinfo().substitute(5);
        return true;
    }());  // smoke check

    // also smoke check the string methods. There is some node and/or compiler-specific
    // stuff here so just check there isn't an obvious bug.
    assert(node_ptr->classname().size());
    assert(node_ptr->repr().size());
    assert(node_ptr->str().size());

    add_predecessor(node_ptr);
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
    emplace_data_ptr<ArrayValidationNodeData>(state, array_ptr->view(state));
    assert(array_ptr->diff(state).size() == 0);
    assert(array_ptr->size_diff(state) == 0);
    assert(static_cast<ssize_t>(array_ptr->view(state).size()) == array_ptr->size(state));

    // check that the size/shape are consistent
    assert(array_ptr->size(state) == std::reduce(array_ptr->shape(state).begin(),
                                                 array_ptr->shape(state).end(), 1,
                                                 std::multiplies<ssize_t>()));

    // check that all values are within min/max
    if (array_ptr->size(state)) {
        assert(std::ranges::min(array_ptr->view(state)) >= array_ptr->min());
        assert(std::ranges::max(array_ptr->view(state)) <= array_ptr->max());
        assert(!array_ptr->integral() || std::ranges::all_of(array_ptr->view(state), is_integer));
    }
}

void ArrayValidationNode::propagate(State& state) const {
    auto node_data = data_ptr<ArrayValidationNodeData>(state);
    std::string node_id = typeid(*array_ptr).name() + std::string{"["} +
                          std::to_string(array_ptr->topological_index()) + std::string{"]"};

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

    // check that all values are within min/max
    if (array_ptr->size(state)) {
        assert(std::ranges::min(array_ptr->view(state)) >= array_ptr->min());
        assert(std::ranges::max(array_ptr->view(state)) <= array_ptr->max());
        assert(!array_ptr->integral() || std::ranges::all_of(array_ptr->view(state), is_integer));
    }

    // check that whatever sizeinfo the array reports is accurate
    auto sizeinfo = array_ptr->sizeinfo();
    if (sizeinfo.array_ptr != nullptr) {
        // the size is at least theoretically derived from another array, so let's check that the
        // reported multiplier/offset are correct

        [[maybe_unused]] auto predicted_size = [&state](const SizeInfo& sizeinfo) -> ssize_t {
            ssize_t size = static_cast<ssize_t>(
                    sizeinfo.multiplier * sizeinfo.array_ptr->size(state) + sizeinfo.offset);
            size = std::max(size, sizeinfo.min.value_or(0));
            if (sizeinfo.max) size = std::min(size, *sizeinfo.max);
            return size;
        };

        assert(array_ptr->size(state) == predicted_size(sizeinfo));
    } else {
        assert(array_ptr->size(state) == sizeinfo.offset);
    }
    assert(!sizeinfo.max || array_ptr->size(state) <= *sizeinfo.max);
    assert(!sizeinfo.min || array_ptr->size(state) >= *sizeinfo.min);
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
    DynamicArrayTestingNodeData() = default;

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
        : DynamicArrayTestingNode(shape, std::nullopt, std::nullopt, false) {}

DynamicArrayTestingNode::DynamicArrayTestingNode(std::initializer_list<ssize_t> shape,
                                                 std::optional<double> min,
                                                 std::optional<double> max, bool integral)
        : DynamicArrayTestingNode(shape, min, max, integral, std::nullopt, std::nullopt) {}

DynamicArrayTestingNode::DynamicArrayTestingNode(std::initializer_list<ssize_t> shape,
                                                 std::optional<double> min,
                                                 std::optional<double> max, bool integral,
                                                 std::optional<ssize_t> min_size,
                                                 std::optional<ssize_t> max_size)
        : ArrayOutputMixin(shape),
          shape_(shape),
          min_(min),
          max_(max),
          integral_(integral),
          sizeinfo_(SizeInfo(this, min_size, max_size)) {
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

    emplace_data_ptr<DynamicArrayTestingNodeData>(state, shape, values);
}

double const* DynamicArrayTestingNode::buff(const State& state) const noexcept {
    return data_ptr<DynamicArrayTestingNodeData>(state)->current_data.data();
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

std::pair<double, double> DynamicArrayTestingNode::minmax(
        optional_cache_type<std::pair<double, double>> cache) const {
    const auto [low, high] = Array::minmax();
    return {min_.value_or(low), max_.value_or(high)};
}

bool DynamicArrayTestingNode::integral() const { return integral_; }

SizeInfo DynamicArrayTestingNode::sizeinfo() const { return sizeinfo_.value_or(SizeInfo(this)); }

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

void DynamicArrayTestingNode::set(State& state, ssize_t index, double value) const {
    data_ptr<DynamicArrayTestingNodeData>(state)->set(index, value);
}

void DynamicArrayTestingNode::shrink(State& state) const {
    if (!size(state)) return;  // nothing to do
    const ssize_t row_size = strides()[0] / itemsize();
    data_ptr<DynamicArrayTestingNodeData>(state)->shrink(row_size);
}

}  // namespace dwave::optimization

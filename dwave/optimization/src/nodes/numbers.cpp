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

#include "dwave-optimization/nodes/numbers.hpp"

namespace dwave::optimization {

// Base classes to be used as interfaces.

struct NumberNodeData : NodeStateData {
    explicit NumberNodeData(std::vector<double>&& number_data) : elements(number_data) {}

    void commit() { previous.clear(); }

    void revert() {
        for (const Update& update : previous | std::views::reverse) {
            elements[update.index] = update.old;
        }
        previous.clear();
    }

    bool exchange(ssize_t i, ssize_t j) {
        assert(i >= 0 && static_cast<std::size_t>(i) < elements.size());
        assert(j >= 0 && static_cast<std::size_t>(j) < elements.size());

        if (elements[i] == elements[j]) {
            return false;
        } else {
            std::swap(elements[i], elements[j]);
            previous.emplace_back(i, elements[j], elements[i]);
            previous.emplace_back(j, elements[i], elements[j]);
            return true;
        }
    }

    double get_value(ssize_t i) const {
        assert(i >= 0 && static_cast<std::size_t>(i) < elements.size());
        return elements[i];
    }

    std::vector<double> elements;
    std::vector<Update> previous;
};

double const* NumberNode::buff(const State& state) const noexcept {
    return data_ptr<NumberNodeData>(state)->elements.data();
}

std::span<const Update> NumberNode::diff(const State& state) const noexcept {
    return data_ptr<NumberNodeData>(state)->previous;
}

void NumberNode::commit(State& state) const noexcept { data_ptr<NumberNodeData>(state)->commit(); }

void NumberNode::revert(State& state) const noexcept { data_ptr<NumberNodeData>(state)->revert(); }

double NumberNode::lower_bound() const { return lower_bound_; }
double NumberNode::min() const { return lower_bound_; }

double NumberNode::upper_bound() const { return upper_bound_; }
double NumberNode::max() const { return upper_bound_; }

void NumberNode::initialize_state(State& state, std::vector<double>&& number_data) const {
    assert(this->topological_index() >= 0 && "must be topologically sorted");
    assert(static_cast<int>(state.size()) > this->topological_index() && "unexpected state length");
    assert(state[this->topological_index()] == nullptr && "already initialized state");

    if (number_data.size() != static_cast<size_t>(this->size())) {
        throw std::invalid_argument("Size of data provided does not match node size");
    }
    if (auto it = std::find_if_not(number_data.begin(), number_data.end(),
                                   [&](double value) { return is_valid(value); });
        it != number_data.end()) {
        throw std::invalid_argument("Invalid data provided for node");
    }

    state[this->topological_index()] = new_data_ptr(std::move(number_data));
}

void NumberNode::initialize_state(State& state) const {
    assert(this->topological_index() >= 0 && "must be topologically sorted");
    assert(static_cast<int>(state.size()) > this->topological_index() && "unexpected state length");
    assert(state[this->topological_index()] == nullptr && "already initialized state");

    std::vector<double> number_data(this->size(), default_value());
    initialize_state(state, std::move(number_data));
}

void NumberNode::initialize_state(State& state, RngAdaptor& rng) const {
    assert(this->topological_index() >= 0 && "must be topologically sorted");
    assert(static_cast<int>(state.size()) > this->topological_index() && "unexpected state length");
    assert(state[this->topological_index()] == nullptr && "already initialized state");

    std::vector<double> number_data(this->size());
    std::generate(number_data.begin(), number_data.end(), [&]() { return generate_value(rng); });
    initialize_state(state, std::move(number_data));
}

// Specializations for the linear case
bool NumberNode::exchange(State& state, ssize_t i, ssize_t j) const {
    return data_ptr<NumberNodeData>(state)->exchange(i, j);
}

double NumberNode::get_value(State& state, ssize_t i) const {
    return data_ptr<NumberNodeData>(state)->get_value(i);
}

ssize_t NumberNode::linear_index(ssize_t x, ssize_t y) const {
    auto shape = this->shape();
    assert(this->ndim() == 2 && "Node must be of 2 dimensional for 2D indexing");
    assert(x >= 0 && x < shape[1] && "X index out of range");
    assert(y >= 0 && y < shape[0] && "Y index out of range");
    return x + y * shape[1];
}

// Integer Node

struct IntegerNodeData : NumberNodeData {
    explicit IntegerNodeData(std::vector<double>&& integer_data)
            : NumberNodeData(std::move(integer_data)) {}

    bool set_value(ssize_t i, double value) {
        assert(i >= 0 && static_cast<std::size_t>(i) < elements.size());
        if (elements[i] == value) {
            return false;
        } else {
            std::swap(elements[i], value);
            previous.emplace_back(i, value, elements[i]);
            return true;
        }
    }

    std::unique_ptr<NodeStateData> copy() const override {
        return std::make_unique<IntegerNodeData>(*this);
    }
};

bool IntegerNode::integral() const { return true; }

bool IntegerNode::is_valid(double value) const {
    return (value >= lower_bound()) && (value <= upper_bound()) && (std::round(value) == value);
}

double IntegerNode::generate_value(RngAdaptor& rng) const {
    std::uniform_int_distribution<std::size_t> value_dist(lower_bound_, upper_bound_);
    return value_dist(rng);
}

double IntegerNode::default_value() const { return (lower_bound() <= 0 && upper_bound() >= 0) ? 0 : lower_bound(); }

std::unique_ptr<NodeStateData> IntegerNode::new_data_ptr(std::vector<double>&& number_data) const {
    return make_unique<IntegerNodeData>(std::move(number_data));
}

bool IntegerNode::set_value(State& state, ssize_t i, int value) const {
    if (!is_valid(value)) {
        throw std::invalid_argument("Invalid integer value provided");
    }
    return data_ptr<IntegerNodeData>(state)->set_value(i, value);
}

void IntegerNode::default_move(State& state, RngAdaptor& rng) const {
    std::uniform_int_distribution<std::size_t> index_dist(0, this->size(state) - 1);
    std::uniform_int_distribution<std::size_t> value_dist(lower_bound_, upper_bound_);
    this->set_value(state, index_dist(rng), value_dist(rng));
}

// Binary Node

struct BinaryNodeData : NumberNodeData {
    explicit BinaryNodeData(std::vector<double>&& binary_data)
            : NumberNodeData(std::move(binary_data)) {}

    void flip(ssize_t i) {
        assert(i >= 0 && static_cast<std::size_t>(i) < elements.size());
        double flipped_element = elements[i] ? 0 : 1;
        std::swap(elements[i], flipped_element);
        previous.emplace_back(i, flipped_element, elements[i]);
    }

    bool set(ssize_t i) {
        assert(i >= 0 && static_cast<std::size_t>(i) < elements.size());
        if (elements[i]) {
            return false;
        } else {
            elements[i] = 1;
            previous.emplace_back(i, 0, 1);
            return true;
        }
    }

    bool unset(ssize_t i) {
        assert(i >= 0 && static_cast<std::size_t>(i) < elements.size());
        if (!elements[i]) {
            return false;
        } else {
            elements[i] = 0;
            previous.emplace_back(i, 1, 0);
            return true;
        }
    }

    std::unique_ptr<NodeStateData> copy() const override {
        return std::make_unique<BinaryNodeData>(*this);
    }
};

std::unique_ptr<NodeStateData> BinaryNode::new_data_ptr(std::vector<double>&& number_data) const {
    return make_unique<BinaryNodeData>(std::move(number_data));
}

void BinaryNode::flip(State& state, ssize_t i) const {
    return data_ptr<BinaryNodeData>(state)->flip(i);
}

bool BinaryNode::set(State& state, ssize_t i) const {
    return data_ptr<BinaryNodeData>(state)->set(i);
}

bool BinaryNode::unset(State& state, ssize_t i) const {
    return data_ptr<BinaryNodeData>(state)->unset(i);
}

void BinaryNode::default_move(State& state, RngAdaptor& rng) const {
    std::uniform_int_distribution<std::size_t> index_dist(0, this->size(state) - 1);
    this->flip(state, index_dist(rng));
}

}  // namespace dwave::optimization

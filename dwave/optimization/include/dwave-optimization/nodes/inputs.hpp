// Copyright 2024 D-Wave
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

#pragma once

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/graph.hpp"

namespace dwave::optimization {

// InputNode acts like a placeholder or store of data very similar to ConstantNode,
// with the key different being that its contents *may* change in between propagations.
// However, it is not a decision variable--instead its use cases are acting as an "input"
// for "models as functions", or for placeholders in large models where (otherwise constant)
// data changes infrequently (e.g. a scheduling problem with a preference matrix).
//
// Currently there is no "default" way to initialize the state, so its must be initialized
// explicitly with some data.
class InputNode : public ArrayOutputMixin<ArrayNode> {
 public:
    explicit InputNode(std::span<const ssize_t> shape, double min, double max, bool integral)
            : ArrayOutputMixin(shape), min_(min), max_(max), integral_(integral) {}

    explicit InputNode(std::initializer_list<ssize_t> shape, double min, double max, bool integral)
            : ArrayOutputMixin(shape), min_(min), max_(max), integral_(integral) {}

    explicit InputNode()
            : InputNode({}, std::numeric_limits<double>::lowest(),
                        std::numeric_limits<double>::infinity(), false) {}

    explicit InputNode(std::initializer_list<ssize_t> shape)
            : InputNode(shape, -std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::infinity(), false) {}

    void assign(State& state, std::span<const double> new_values) const;

    double const* buff(const State&) const override;
    void commit(State& state) const noexcept override;
    std::span<const Update> diff(const State& state) const noexcept override;

    bool integral() const override { return integral_; };

    [[noreturn]] void initialize_state(State& state) const override {
        throw std::logic_error(
                "InputNode must have state explicity initialized (with `initialize_state(state, "
                "data)`)");
    }

    void initialize_state(State& state, std::span<const double> data) const;

    std::pair<double, double> minmax(
            optional_cache_type<std::pair<double, double>> cache) const override {
        return {min_, max_};
    }

    void propagate(State& state) const noexcept override{};
    void revert(State& state) const noexcept override;

 private:
    const double min_;
    const double max_;
    const bool integral_;
};

}  // namespace dwave::optimization

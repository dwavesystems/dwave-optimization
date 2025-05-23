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
    explicit InputNode(std::span<const ssize_t> shape, std::optional<double> min,
                       std::optional<double> max, std::optional<bool> integral);

    explicit InputNode(std::initializer_list<ssize_t> shape, std::optional<double> min,
                       std::optional<double> max, std::optional<bool> integral)
            : InputNode(std::span<const ssize_t>(shape), min, max, integral) {}

    explicit InputNode() : InputNode({}, std::nullopt, std::nullopt, std::nullopt) {}

    explicit InputNode(std::initializer_list<ssize_t> shape)
            : InputNode(shape, std::nullopt, std::nullopt, std::nullopt) {}

    /// Assign new values to the input node (must be the same size)
    void assign(State& state, std::span<const double> new_values) const;

    /// @copydoc Array::buff()
    double const* buff(const State&) const override;

    /// @copydoc Node::commit()
    void commit(State& state) const noexcept override;

    /// InputNode's state is not deterministic unlike most other non-decision nodes
    bool deterministic_state() const override { return false; }

    /// @copydoc Array::diff()
    std::span<const Update> diff(const State& state) const noexcept override;

    /// @copydoc Array::integral()
    bool integral() const override { return integral_; };

    [[noreturn]] void initialize_state(State& state) const override {
        throw std::logic_error(
                "InputNode must have state explicity initialized (with `initialize_state(state, "
                "data)`)");
    }

    /// Initialize a state with the given data as the values for the input node
    void initialize_state(State& state, std::span<const double> data) const;

    /// @copydoc Array::minmax()
    std::pair<double, double> minmax(
            optional_cache_type<std::pair<double, double>> cache) const override {
        return {min_, max_};
    }

    /// @copydoc Node::propagate()
    void propagate(State& state) const noexcept override{};

    /// @copydoc Node::revert()
    void revert(State& state) const noexcept override;

 private:
    void check_values(std::span<const double> new_values) const;

    const double min_;
    const double max_;
    const bool integral_;
};

}  // namespace dwave::optimization

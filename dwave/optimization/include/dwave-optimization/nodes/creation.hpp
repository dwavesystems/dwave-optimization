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

#pragma once

#include <optional>
#include <utility>
#include <variant>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/state.hpp"

namespace dwave::optimization {

/// A node encoding evenly spaced values within a given interval.
class ARangeNode : public ArrayOutputMixin<ArrayNode> {
 public:
    using array_or_int = std::variant<const Array*, ssize_t>;

    // We can avoid having so many constructors with a lot of fiddling with
    // concepts and templates. But this is more explicit and lets us put more
    // behind the compilation barrier. So let's live with the pile of
    // constructors.

    // Create an empty range.
    ARangeNode();

    /// Values are generated within the half-open interval ``[0, stop)``.
    explicit ARangeNode(ssize_t stop);

    /// Values are generated within the half-open interval ``[0, stop)``.
    explicit ARangeNode(ArrayNode* stop);

    /// Values are generated within the half-open interval ``[start, stop)``,
    /// with spacing between values given by ``step``.
    ARangeNode(ssize_t start, ssize_t stop, ssize_t step = 1);

    /// Values are generated within the half-open interval ``[start, stop)``,
    /// with spacing between values given by ``step``.
    ARangeNode(ssize_t start, ssize_t stop, ArrayNode* step);

    /// Values are generated within the half-open interval ``[start, stop)``,
    /// with spacing between values given by ``step``.
    ARangeNode(ssize_t start, ArrayNode* stop, ssize_t step = 1);

    /// Values are generated within the half-open interval ``[start, stop)``,
    /// with spacing between values given by ``step``.
    ARangeNode(ssize_t start, ArrayNode* stop, ArrayNode* step);

    /// Values are generated within the half-open interval ``[start, stop)``,
    /// with spacing between values given by ``step``.
    ARangeNode(ArrayNode* start, ssize_t stop, ssize_t step = 1);

    /// Values are generated within the half-open interval ``[start, stop)``,
    /// with spacing between values given by ``step``.
    ARangeNode(ArrayNode* start, ssize_t stop, ArrayNode* step);

    /// Values are generated within the half-open interval ``[start, stop)``,
    /// with spacing between values given by ``step``.
    ARangeNode(ArrayNode* start, ArrayNode* stop, ssize_t step = 1);

    /// Values are generated within the half-open interval ``[start, stop)``,
    /// with spacing between values given by ``step``.
    ARangeNode(ArrayNode* start, ArrayNode* stop, ArrayNode* step);

    /// @copydoc Array::buff()
    double const* buff(const State& state) const override;

    /// @copydoc Node::commit()
    void commit(State& state) const override;

    /// @copydoc Array::diff()
    std::span<const Update> diff(const State& state) const override;

    /// @copydoc Node::initialize_state()
    void initialize_state(State& state) const override;

    /// @copydoc Array::integral()
    bool integral() const override;

    /// @copydoc Array::minmax()
    std::pair<double, double> minmax(
            optional_cache_type<std::pair<double, double>> cache = std::nullopt) const override;

    /// @copydoc Node::propagate()
    void propagate(State& state) const override;

    /// @copydoc Node::revert()
    void revert(State& state) const override;

    using ArrayOutputMixin::shape;

    /// @copydoc Array::shape()
    std::span<const ssize_t> shape(const State& state) const override;

    using ArrayOutputMixin::size;

    /// @copydoc Array::size()
    ssize_t size(const State& state) const override;

    /// @copydoc Array::sizeinfo()
    SizeInfo sizeinfo() const override;

    /// @copydoc Array::size_diff()
    ssize_t size_diff(const State& state) const override;

    /// The value or array defining the start of the range.
    array_or_int start() const { return start_; }

    /// The value or array defining the stop (not inclusive) of the range.
    array_or_int stop() const { return stop_; }

    /// The value or array defining the spacing between values.
    array_or_int step() const { return step_; }

 private:
    array_or_int start_;
    array_or_int stop_;
    array_or_int step_;
};

}  // namespace dwave::optimization

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

#include "dwave-optimization/nodes/creation.hpp"

#include <variant>

#include "_state.hpp"
#include "dwave-optimization/array.hpp"
#include "dwave-optimization/graph.hpp"
#include "dwave-optimization/nodes/manipulation.hpp"

namespace dwave::optimization {

// Matches ARangeNode::array_or_int
using array_or_int = std::variant<const Array*, ssize_t>;

struct get_diff {
    explicit get_diff(const State& state) : state(state) {}
    std::pair<double, double> operator()(ssize_t value) { return {value, value}; }
    std::pair<double, double> operator()(const Array* array_ptr) {
        double value = array_ptr->view(state).front();
        if (const auto& diff = array_ptr->diff(state); diff.size() > 0) {
            return {diff[0].old, value};
        }
        return {value, value};
    }
    const State& state;
};

struct get_minmax {
    std::pair<ssize_t, ssize_t> operator()(ssize_t value) const { return {value, value}; }
    std::pair<ssize_t, ssize_t> operator()(const Array* array_ptr) const {
        return std::make_pair(array_ptr->min(), array_ptr->max());
    }
};

struct get_value {
    explicit get_value(const State& state) : state(state) {}
    ssize_t operator()(ssize_t value) const { return value; }
    ssize_t operator()(const Array* array_ptr) const { return array_ptr->view(state).front(); }
    const State& state;
};

struct validate {
    void operator()(ssize_t value) const {}
    void operator()(const Array* array_ptr) const {
        if (array_ptr->size() != 1) throw std::invalid_argument("input must be a scalar");
        if (!array_ptr->integral()) throw std::invalid_argument("input must be integral");
    }
};

ssize_t range_shape(array_or_int start_, array_or_int stop_, array_or_int step_) {
    // First do a quick validation step, checking out inputs for correctness
    {
        auto visitor = validate();
        std::visit(visitor, start_);
        std::visit(visitor, stop_);
        std::visit(visitor, step_);
    }

    // Next get the the shape of the array we're creating.

    auto visitor = get_minmax();
    const auto [start, start_high] = std::visit(visitor, start_);
    const auto [stop, stop_high] = std::visit(visitor, stop_);
    const auto [step, step_high] = std::visit(visitor, step_);

    // step cannot be zero. For an array, we ensure that it can never take the zero value
    if (step <= 0 && step_high >= 0) {
        throw std::invalid_argument("step cannot be 0 or have 0 as a possible value");
    }

    // in some cases we're always empty, so let's check those
    if (step > 0 && stop_high <= start) return 0;
    if (step < 0 && start_high <= stop) return 0;

    // if any of them are not fixed, then we're dynamically sized.
    if (start != start_high || stop != stop_high || step != step_high) {
        return Array::DYNAMIC_SIZE;
    }

    // Ok, we're not empty and we're not dynamic
    auto div = std::div(stop - start, step);

    return div.rem ? div.quot + 1 : div.quot;
}

std::vector<double> arange(const ssize_t start, const ssize_t stop, const ssize_t step) {
    auto arange = std::vector<double>();
    if (step > 0) {
        for (ssize_t x = start; x < stop; x += step) {
            arange.emplace_back(x);
        }
    } else if (step < 0) {
        for (ssize_t x = start; x > stop; x += step) {
            arange.emplace_back(x);
        }
    } else {
        assert(false && "0 step not allowed");
        unreachable();
    }

    return arange;
}
std::vector<double> arange(const State& state, array_or_int start_, array_or_int stop_,
                           array_or_int step_) {
    auto visitor = get_value(state);
    const ssize_t start = std::visit(visitor, start_);
    const ssize_t stop = std::visit(visitor, stop_);
    const ssize_t step = std::visit(visitor, step_);

    return arange(start, stop, step);
}

std::pair<double, double> calculate_values_minmax(array_or_int start_, array_or_int stop_,
                                                  array_or_int step_) {
    auto visitor = get_minmax();
    const auto [start_low, start_high] = std::visit(visitor, start_);
    const auto [stop_low, stop_high] = std::visit(visitor, stop_);
    const auto [step_low, step_high] = std::visit(visitor, step_);

    // just sanity check because we can't specify in the structured binding
    // and it matters later that we're integral when calculating max
    static_assert(std::same_as<decltype(start_low), const ssize_t>);
    static_assert(std::same_as<decltype(start_high), const ssize_t>);

    // checked on construction.
    assert(!(step_low <= 0 && step_high >= 0));

    // The direction that we're stepping determines the min/max
    if (step_low > 0) {
        if (start_low >= stop_high) return std::pair<double, double>(0, 0);

        // Our max value will always use the largest stop, but we do need
        // to check several combinations of start/step.
        const double high = std::max({
                start_low + ((stop_high - start_low - 1) / step_low) * step_low,
                start_low + ((stop_high - start_low - 1) / step_high) * step_high,
                start_high + ((stop_high - start_high - 1) / step_low) * step_low,
                start_high + ((stop_high - start_high - 1) / step_high) * step_high,
        });

        return std::pair<double, double>{start_low, high};
    }

    if (step_high < 0) {
        if (start_high <= stop_low) return std::pair<double, double>(0, 0);

        // Our min value will always use the smallest stop, but we do need
        // to check several combinations of start/step.
        const double low = std::min({
                start_low + ((start_low - stop_low - 1) / -step_low) * step_low,
                start_low + ((start_low - stop_low - 1) / -step_high) * step_high,
                start_high + ((start_high - stop_low - 1) / -step_low) * step_low,
                start_high + ((start_high - stop_low - 1) / -step_high) * step_high,
        });

        return std::pair<double, double>{low, start_high};
    }

    assert(false && "zero step not allowed");
    unreachable();
}

const SizeInfo calculate_arange_sizeinfo(const ArrayNode* node_ptr, const array_or_int start,
                                         const array_or_int stop, const array_or_int step) {
    if (!node_ptr->dynamic()) return SizeInfo(node_ptr->size());

    auto visitor = get_minmax();
    const auto [start_low, start_high] = std::visit(visitor, start);
    const auto [stop_low, stop_high] = std::visit(visitor, stop);
    const auto [step_low, step_high] = std::visit(visitor, step);

    // checked on construction.
    assert(!(step_low <= 0 && step_high >= 0));

    // The direction we're stepping determines the largest/smallest we can be.
    //
    // Given an interval: [a, b) and a step size s, arange(a, b, s) yields the
    // values: a, a+s, a+2s, ..., a+ks where k is the largest integer such that
    // a+ks < b => k < (b-a)/s. There are two cases.
    // 1) If s does NOT divide (b-a), then k = floor((b-a)/s).
    // 2) If s divides (b-a), then k = (b-a)/s - 1.
    //
    // For such a value k, arange(a, b, s) has (k + 1) values. Therefore,
    // 1) If s does NOT divide (b-a), then k+1 = floor((b-a)/s)+1 = ceil((b-a)/s).
    // 2) If s divides (b-a), then k+1 = (b-a)/s-1+1 = (b-a)/s = ceil((b-a)/s).
    //
    // Therefore, arange(a, b, s) defines ceil((b-a)/s) many values.
    ssize_t min;
    ssize_t max;
    if (step_low > 0) {
        min = std::max<ssize_t>(std::ceil(static_cast<double>(stop_low - start_high) / step_high),
                                0);
        max = std::max<ssize_t>(std::ceil(static_cast<double>(stop_high - start_low) / step_low),
                                min);
    } else if (step_high < 0) {
        min = std::max<ssize_t>(std::ceil(static_cast<double>(stop_high - start_low) / step_high),
                                0);
        max = std::max<ssize_t>(std::ceil(static_cast<double>(stop_low - start_high) / step_low),
                                min);
    } else {
        assert(false && "unreachable");
        unreachable();
    }

    // Handles all cases EXCEPT the following: "Exactly one predecessor, it
    // defines `stop`, and it is a SizeNode."
    if (std::holds_alternative<const Array*>(start) || std::holds_alternative<ssize_t>(stop) ||
        std::holds_alternative<const Array*>(step) ||
        dynamic_cast<const SizeNode*>(std::get<const Array*>(stop)) == nullptr) {
        return SizeInfo(node_ptr, min, max);
    }

    assert(step_low == step_high);
    assert(start_low == start_high);
    // The size of this node is a linear function of the size of the node its
    // predecessor (the SizeNode defining `stop`) is listening to.
    auto sizenode_pred_ptr = dynamic_cast<const ArrayNode*>(
            static_cast<const ArrayNode*>(std::get<const Array*>(stop))->predecessors()[0]);

    // SizeInfo is computed as follows (see SizeInfo docs):
    // clamp(ceil(multiplier * array_ptr->size() + offset), min, max)
    //
    // To simplify the algebra, let x := sizenode_pred_ptr->size(),
    // start := start_low == start_high, and step := step_low == step_high.
    //
    // The size of this node is clamp[ceil{(x - start) / step}, min, max].
    SizeInfo sizeinfo(sizenode_pred_ptr, min, max);
    sizeinfo.multiplier = fraction(1, step_low);
    sizeinfo.offset = fraction(-start_low, step_low);

    return sizeinfo;
}

// For all of the constructors, we force them to use the array_or_int overload
// by casting explicitly. That's the one that does the error checking etc.
// In a lot of cases, it's almost certainly redundant, but for now let's do
// that extra checking.
ARangeNode::ARangeNode() : ARangeNode(ssize_t(0)) {}
ARangeNode::ARangeNode(ssize_t stop) : ARangeNode(ssize_t(0), stop) {}
ARangeNode::ARangeNode(ArrayNode* stop) : ARangeNode(ssize_t(0), stop) {}

ARangeNode::ARangeNode(ssize_t start, ssize_t stop, ssize_t step)
        : ArrayOutputMixin(range_shape(start, stop, step)),
          start_(start),
          stop_(stop),
          step_(step),
          values_minmax_(calculate_values_minmax(start, stop, step)),
          sizeinfo_(calculate_arange_sizeinfo(this, start, stop, step)) {}

ARangeNode::ARangeNode(ssize_t start, ssize_t stop, ArrayNode* step)
        : ArrayOutputMixin(range_shape(start, stop, step)),
          start_(start),
          stop_(stop),
          step_(step),
          values_minmax_(calculate_values_minmax(start, stop, step)),
          sizeinfo_(calculate_arange_sizeinfo(this, start, stop, step)) {
    add_predecessor(step);
}
ARangeNode::ARangeNode(ssize_t start, ArrayNode* stop, ssize_t step)
        : ArrayOutputMixin(range_shape(start, stop, step)),
          start_(start),
          stop_(stop),
          step_(step),
          values_minmax_(calculate_values_minmax(start, stop, step)),
          sizeinfo_(calculate_arange_sizeinfo(this, start, stop, step)) {
    add_predecessor(stop);
}
ARangeNode::ARangeNode(ssize_t start, ArrayNode* stop, ArrayNode* step)
        : ArrayOutputMixin(range_shape(start, stop, step)),
          start_(start),
          stop_(stop),
          step_(step),
          values_minmax_(calculate_values_minmax(start, stop, step)),
          sizeinfo_(calculate_arange_sizeinfo(this, start, stop, step)) {
    add_predecessor(stop);
    add_predecessor(step);
}
ARangeNode::ARangeNode(ArrayNode* start, ssize_t stop, ssize_t step)
        : ArrayOutputMixin(range_shape(start, stop, step)),
          start_(start),
          stop_(stop),
          step_(step),
          values_minmax_(calculate_values_minmax(start, stop, step)),
          sizeinfo_(calculate_arange_sizeinfo(this, start, stop, step)) {
    add_predecessor(start);
}
ARangeNode::ARangeNode(ArrayNode* start, ssize_t stop, ArrayNode* step)
        : ArrayOutputMixin(range_shape(start, stop, step)),
          start_(start),
          stop_(stop),
          step_(step),
          values_minmax_(calculate_values_minmax(start, stop, step)),
          sizeinfo_(calculate_arange_sizeinfo(this, start, stop, step)) {
    add_predecessor(start);
    add_predecessor(step);
}
ARangeNode::ARangeNode(ArrayNode* start, ArrayNode* stop, ssize_t step)
        : ArrayOutputMixin(range_shape(start, stop, step)),
          start_(start),
          stop_(stop),
          step_(step),
          values_minmax_(calculate_values_minmax(start, stop, step)),
          sizeinfo_(calculate_arange_sizeinfo(this, start, stop, step)) {
    add_predecessor(start);
    add_predecessor(stop);
}
ARangeNode::ARangeNode(ArrayNode* start, ArrayNode* stop, ArrayNode* step)
        : ArrayOutputMixin(range_shape(start, stop, step)),
          start_(start),
          stop_(stop),
          step_(step),
          values_minmax_(calculate_values_minmax(start, stop, step)),
          sizeinfo_(calculate_arange_sizeinfo(this, start, stop, step)) {
    add_predecessor(start);
    add_predecessor(stop);
    add_predecessor(step);
}

double const* ARangeNode::buff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->buff();
}

void ARangeNode::commit(State& state) const { data_ptr<ArrayNodeStateData>(state)->commit(); }

std::span<const Update> ARangeNode::diff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->diff();
}

bool ARangeNode::integral() const { return true; }

void ARangeNode::initialize_state(State& state) const {
    emplace_data_ptr<ArrayNodeStateData>(state, arange(state, start_, stop_, step_));
}

double ARangeNode::min() const { return values_minmax_.first; }

double ARangeNode::max() const { return values_minmax_.second; }

void ARangeNode::propagate(State& state) const {
    auto visitor = get_diff(state);
    const auto [start_old, start_new] = std::visit(visitor, start_);
    const auto [stop_old, stop_new] = std::visit(visitor, stop_);
    const auto [step_old, step_new] = std::visit(visitor, step_);

    ArrayNodeStateData* ptr = data_ptr<ArrayNodeStateData>(state);

    // If the start or the step has changed, we need to change everything
    // Alternatively if there is currently nothing in the buffer at all, we might
    // as well just assign
    if (start_old != start_new || step_old != step_new || ptr->size() == 0) {
        ptr->assign(arange(state, start_, stop_, step_));
        if (ptr->diff().size()) Node::propagate(state);
        return;
    }

    // if the stop has also not changed, then nothing to do
    if (stop_old == stop_new) return;

    // only the stop has changed
    const ssize_t step = step_old;
    assert(step == step_new);

    assert(ptr->size() >= 1);

    if (step > 0) {
        if (stop_old < stop_new) {
            // we grew
            for (const double& val : arange(ptr->back() + step, stop_new, step)) {
                ptr->emplace_back(val);
            }
        } else {
            // we shrank
            while (ptr->size() && ptr->back() >= stop_new) {
                ptr->pop_back();
            }
        }
    } else if (step < 0) {
        if (stop_old < stop_new) {
            // we shrank
            while (ptr->size() && ptr->back() <= stop_new) {
                ptr->pop_back();
            }
        } else {
            // we grew
            for (const double& val : arange(ptr->back() + step, stop_new, step)) {
                ptr->emplace_back(val);
            }
        }
    } else {
        assert(false && "zero step not allowed");
        unreachable();
    }

    if (ptr->diff().size()) Node::propagate(state);
}

void ARangeNode::revert(State& state) const { data_ptr<ArrayNodeStateData>(state)->revert(); }

std::span<const ssize_t> ARangeNode::shape(const State& state) const {
    return std::span<const ssize_t>(&(data_ptr<ArrayNodeStateData>(state)->size()), 1);
}

ssize_t ARangeNode::size(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->size();
}

ssize_t ARangeNode::size_diff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->size_diff();
}

}  // namespace dwave::optimization

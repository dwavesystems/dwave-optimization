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

#include "_state.hpp"

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
    explicit get_minmax(Array::optional_cache_type<std::pair<double, double>> cache = std::nullopt)
            : cache(cache) {}

    std::pair<ssize_t, ssize_t> operator()(ssize_t value) const { return {value, value}; }
    std::pair<ssize_t, ssize_t> operator()(const Array* array_ptr) const {
        return array_ptr->minmax(cache);
    }

    Array::optional_cache_type<std::pair<double, double>> cache;
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
          step_(step) {}
ARangeNode::ARangeNode(ssize_t start, ssize_t stop, ArrayNode* step)
        : ArrayOutputMixin(range_shape(start, stop, step)),
          start_(start),
          stop_(stop),
          step_(step) {
    add_predecessor(step);
}
ARangeNode::ARangeNode(ssize_t start, ArrayNode* stop, ssize_t step)
        : ArrayOutputMixin(range_shape(start, stop, step)),
          start_(start),
          stop_(stop),
          step_(step) {
    add_predecessor(stop);
}
ARangeNode::ARangeNode(ssize_t start, ArrayNode* stop, ArrayNode* step)
        : ArrayOutputMixin(range_shape(start, stop, step)),
          start_(start),
          stop_(stop),
          step_(step) {
    add_predecessor(stop);
    add_predecessor(step);
}
ARangeNode::ARangeNode(ArrayNode* start, ssize_t stop, ssize_t step)
        : ArrayOutputMixin(range_shape(start, stop, step)),
          start_(start),
          stop_(stop),
          step_(step) {
    add_predecessor(start);
}
ARangeNode::ARangeNode(ArrayNode* start, ssize_t stop, ArrayNode* step)
        : ArrayOutputMixin(range_shape(start, stop, step)),
          start_(start),
          stop_(stop),
          step_(step) {
    add_predecessor(start);
    add_predecessor(step);
}
ARangeNode::ARangeNode(ArrayNode* start, ArrayNode* stop, ssize_t step)
        : ArrayOutputMixin(range_shape(start, stop, step)),
          start_(start),
          stop_(stop),
          step_(step) {
    add_predecessor(start);
    add_predecessor(stop);
}
ARangeNode::ARangeNode(ArrayNode* start, ArrayNode* stop, ArrayNode* step)
        : ArrayOutputMixin(range_shape(start, stop, step)),
          start_(start),
          stop_(stop),
          step_(step) {
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

std::pair<double, double> ARangeNode::minmax(
        optional_cache_type<std::pair<double, double>> cache) const {
    return memoize(cache, [&]() {
        auto visitor = get_minmax(cache);
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
    });
}

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

    const auto& buffer = ptr->buffer;  // readonly for our purposes
    assert(buffer.size() >= 1);

    if (step > 0) {
        if (stop_old < stop_new) {
            // we grew
            for (const double& val : arange(buffer.back() + step, stop_new, step)) {
                ptr->emplace_back(val);
            }
        } else {
            // we shrank
            while (buffer.size() && buffer.back() >= stop_new) {
                ptr->pop_back();
            }
        }
    } else if (step < 0) {
        if (stop_old < stop_new) {
            // we shrank
            while (buffer.size() && buffer.back() <= stop_new) {
                ptr->pop_back();
            }
        } else {
            // we grew
            for (const double& val : arange(buffer.back() + step, stop_new, step)) {
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

// There is a special case here that we're not considering, which is where
// there is exactly one stop predecessor, and it's a SizeNode.
SizeInfo ARangeNode::sizeinfo() const {
    if (!dynamic()) return SizeInfo(size());

    auto visitor = get_minmax();
    const auto [start_low, start_high] = std::visit(visitor, start_);
    const auto [stop_low, stop_high] = std::visit(visitor, stop_);
    const auto [step_low, step_high] = std::visit(visitor, step_);

    // checked on construction.
    assert(!(step_low <= 0 && step_high >= 0));

    // The direction that we're stepping determines the largest/smallest
    // we can be
    if (step_low > 0) {
        const ssize_t min_ = std::max<ssize_t>((stop_low - start_high) / step_high, 0);
        const ssize_t max_ = std::max<ssize_t>((stop_high - start_low) / step_low, min_);
        return SizeInfo(this, min_, max_);
    }
    if (step_high < 0) {
        const ssize_t min_ = std::max<ssize_t>((stop_high - start_low) / step_high, 0);
        const ssize_t max_ = std::max<ssize_t>((stop_low - start_high) / step_low, min_);
        return SizeInfo(this, min_, max_);
    }

    assert(false && "unreachable");
    unreachable();
}

ssize_t ARangeNode::size_diff(const State& state) const {
    return data_ptr<ArrayNodeStateData>(state)->size_diff();
}

}  // namespace dwave::optimization

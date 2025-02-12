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

#pragma once

#include <optional>
#include <vector>

#include "dwave-optimization/array.hpp"
#include "dwave-optimization/graph.hpp"

namespace dwave::optimization {

class ArrayValidationNode : public Node {
 public:
    explicit ArrayValidationNode(ArrayNode* node_ptr);

    // Node overloads
    void commit(State& state) const override;
    void initialize_state(State& state) const override;
    void propagate(State& state) const override;
    void revert(State& state) const override;

 private:
    const ArrayNode* array_ptr;

#ifdef NDEBUG
    static constexpr bool do_logging = false;
#else
    static constexpr bool do_logging = true;
#endif
};

class DynamicArrayTestingNode : public ArrayOutputMixin<ArrayNode>, public DecisionNode {
 public:
    DynamicArrayTestingNode(std::initializer_list<ssize_t> shape);
    DynamicArrayTestingNode(std::initializer_list<ssize_t> shape, std::optional<double> min,
                            std::optional<double> max, bool integral);
    DynamicArrayTestingNode(std::initializer_list<ssize_t> shape, std::optional<double> min,
                            std::optional<double> max, bool integral,
                            std::optional<ssize_t> min_size, std::optional<ssize_t> max_size);

    // Overloads needed by the Array ABC **************************************

    std::span<const Update> diff(const State& state) const override;

    double const* buff(const State&) const noexcept override;

    using ArrayOutputMixin::size;  // for size()
    ssize_t size(const State& state) const override;

    using ArrayOutputMixin::shape;  // for shape()
    std::span<const ssize_t> shape(const State& state) const override;

    ssize_t size_diff(const State& state) const override;

    // lie about min/max/integral for now. We lie in such a way as to create
    // maximum compatibility. In the future we might want to set these values
    // at construction time.
    std::pair<double, double> minmax(
            optional_cache_type<std::pair<double, double>> cache = std::nullopt) const override;

    bool integral() const override;
    SizeInfo sizeinfo() const override;

    void initialize_state(State& state) const override;
    void initialize_state(State& state, std::initializer_list<double> values) const;
    void initialize_state(State& state, std::span<const double> values) const;

    // Overloads required by the Node ABC *************************************

    void commit(State&) const override;
    void revert(State&) const override;
    void update(State&, int) const override;

    // State mutation methods *************************************************

    // Grow the array by a single row of the given values.
    void grow(State& state, std::span<const double> values) const;
    void grow(State& state, std::initializer_list<double> values) const;

    // Grow the array by one row with random values.
    template <std::uniform_random_bit_generator Generator>
    void grow(State& state, Generator& rng) const {
        // generate a new row randomly
        assert(ndim() > 0);
        const ssize_t row_size = strides()[0] / itemsize();

        // it will take values randomly from the range [min, max]
        std::uniform_real_distribution<double> gen_value(min(), max());

        std::vector<double> row;
        row.reserve(row_size);
        for (ssize_t i = 0; i < row_size; ++i) row.emplace_back(gen_value(rng));

        grow(state, row);
    }

    // Set the value of the array at the given index.
    void set(State& state, ssize_t index, double value) const;

    // Set the value at a random index to a random value.
    template <std::uniform_random_bit_generator Generator>
    void set(State& state, Generator& rng) const {
        const ssize_t size = this->size(state);
        assert(size >= 0);  // should always be true

        if (!size) return;  // nothing to do

        // it will take values randomly from the range [min, max]
        std::uniform_real_distribution<double> gen_value(min(), max());

        // it will be a random index
        std::uniform_int_distribution<ssize_t> gen_index(0, size - 1);

        set(state, gen_index(rng), gen_value(rng));
    }

    // Shrink the array by one row.
    void shrink(State& state) const;

    // Do either a grow, shrink, or set randomly
    template <std::uniform_random_bit_generator Generator>
    void random_move(State& state, Generator& rng) const {
        std::uniform_int_distribution<int> gen_move(0, 2);
        switch (gen_move(rng)) {
            case 0:
                return grow(state, rng);
            case 1:
                return shrink(state);
            case 2:
                return set(state, rng);
            default:
                unreachable();
        }
    }

 private:
    const std::span<const ssize_t> shape_;

    std::optional<double> min_;
    std::optional<double> max_;
    bool integral_ = false;
    std::optional<SizeInfo> sizeinfo_;
};

}  // namespace dwave::optimization

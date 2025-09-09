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

#include "dwave-optimization/nodes/reduce.hpp"

#include <cmath>
#include <concepts>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <tuple>
#include <type_traits>

#include "_state.hpp"

namespace dwave::optimization {

// A class that encodes the product of an array. Allows for incremental updates
// without needing to recalculate the whole thing.
struct RunningProduct {
    RunningProduct() : nonzero(1), num_zero(0) {}
    explicit RunningProduct(const double init) : nonzero(init ? init : 1), num_zero(init == 0) {}

    // Get as a double. I.e., 0^num_zero * nonzero
    explicit operator double() const noexcept { return (num_zero > 0) ? 0 : nonzero; }

    // Multiply by a number, tracking the number of times it's been multiplied by 0
    RunningProduct& operator*=(const double rhs) noexcept {
        if (rhs) {
            nonzero *= rhs;
        } else {
            num_zero += 1;
        }
        return *this;
    }

    // Divide by a number, tracking the number of times it's been divided by 0
    RunningProduct& operator/=(const double rhs) {
        if (rhs) {
            nonzero /= rhs;
        } else {
            num_zero -= 1;

            // the implementation is well-defined for num_zero < 0, but conceptually
            // it's not really well defined.
            assert(num_zero >= 0);
        }
        return *this;
    }

    // Multiply and divide
    void multiply_divide(const double multiplier, const double divisor) {
        // This function exists so that in the future we can be a bit more careful
        // about numeric errors. For now we just delegate to other methods for
        // simplicity

        if (multiplier == divisor) return;  // nothing to do when they cancel out

        *this *= multiplier;
        *this /= divisor;
    }

    // The output value is 0^num_zero * nonzero
    double nonzero;  // todo: consider float128 or other way of managing floating point errors
    ssize_t num_zero;
};

// For a product reduction over n elements each taking values in [low, high]
// and an initial value of init, what is the min it can take?
double product_max(ssize_t n, double init, double low, double high) {
    if (n <= 0) return init;
    if (n == 1) return std::max<double>(init * low, init * high);
    return std::max<double>({
            init * std::pow(low, n),
            init * std::pow(high, n),
            init * low * std::pow(high, n - 1),
            init * high * std::pow(low, n - 1),
    });
}
double product_min(ssize_t n, double init, double low, double high) {
    if (n <= 0) return init;
    if (n == 1) return std::min<double>(init * low, init * high);
    return std::min<double>({
            init * std::pow(low, n),
            init * std::pow(high, n),
            init * low * std::pow(high, n - 1),
            init * high * std::pow(low, n - 1),
    });
}
std::pair<double, double> product_minmax(ssize_t n, double init, double low, double high) {
    return {product_min(n, init, low, high), product_max(n, init, low, high)};
}

// The state of the PartialReduceNode - not implemented by default
template <class BinaryOp>
class PartialReduceNodeData {};

// The state of a PartialProdNode
template <>
class PartialReduceNodeData<std::multiplies<double>> : private ArrayStateData,
                                                       public NodeStateData {
 public:
    explicit PartialReduceNodeData(std::vector<RunningProduct>&& values) noexcept
            : ArrayStateData(running_to_double(values)),
              NodeStateData(),
              products_(std::move(values)) {}

    using ArrayStateData::buff;

    void commit() {
        ArrayStateData::commit();
        products_diff_.clear();
    }

    using ArrayStateData::diff;

    void revert() {
        ArrayStateData::revert();

        for (const auto& [index, product] : products_diff_ | std::views::reverse) {
            products_[index] = product;
        }
        products_diff_.clear();
    }

    // Incorporate a single update at the given index. I.e. one entry in
    // the relevant axis has updated its value from old -> value.
    void update(ssize_t index, double old, double value) {
        products_diff_.emplace_back(index, products_[index]);               // save the old value
        products_[index].multiply_divide(value, old);                       // update the accumlator
        ArrayStateData::set(index, static_cast<double>(products_[index]));  // set our output
    }

    using ArrayStateData::size_diff;

 private:
    // convert a vector of running doubles to a vector of doubles via an explicit static_cast
    static std::vector<double> running_to_double(const std::vector<RunningProduct>& values) {
        std::vector<double> out;
        out.reserve(values.size());
        for (const auto& val : values) {
            out.emplace_back(static_cast<double>(val));
        }
        return out;
    }

    std::vector<RunningProduct> products_;
    std::vector<std::tuple<ssize_t, RunningProduct>> products_diff_;
};

// The state of a PartialSumNode
template <>
class PartialReduceNodeData<std::plus<double>> : public ArrayNodeStateData {
 public:
    explicit PartialReduceNodeData(std::vector<double>&& values) noexcept
            : ArrayNodeStateData(std::move(values)) {}

    // Incorporate a single update at the given index. I.e. one entry in
    // the relevant axis has updated its value from old -> value.
    void update(ssize_t index, double old, double value) {
        ArrayStateData::set(index, ArrayStateData::get(index) - old + value);
    }
};

std::vector<ssize_t> partial_reduce_shape(const std::span<const ssize_t> input_shape,
                                          const ssize_t axis) {
    std::vector<ssize_t> shape(input_shape.begin(), input_shape.end());
    shape.erase(shape.begin() + axis);
    return shape;
}

std::vector<ssize_t> as_contiguous_strides(const std::span<const ssize_t> shape) {
    ssize_t ndim = static_cast<ssize_t>(shape.size());

    assert(ndim > 0);
    std::vector<ssize_t> strides(ndim);
    // otherwise strides are a function of the shape
    strides[ndim - 1] = sizeof(double);
    for (auto i = ndim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

std::span<const ssize_t> nonempty(std::span<const ssize_t> span) {
    if (span.empty()) {
        throw std::invalid_argument("Input span should not be empty");
    }
    return span;
}

template <class BinaryOp>
bool partial_reduce_calculate_integral(const Array* array_ptr, const std::optional<double>& init) {
    using result_type = typename std::invoke_result<BinaryOp, double&, double&>::type;

    if constexpr (std::is_integral<result_type>::value) {
        return true;
    }
    if constexpr (std::same_as<BinaryOp, std::multiplies<double>>) {
        return array_ptr->integral() && is_integer(init.value_or(1));
    }
    if constexpr (std::same_as<BinaryOp, std::plus<double>>) {
        return array_ptr->integral() && is_integer(init.value_or(0));
    }

    assert(false && "not implemented yet");
    unreachable();
}

template <class BinaryOp>
ValuesInfo partial_reduce_calculate_values_info(const Array* array_ptr, ssize_t axis,
                                                const std::optional<double>& init) {
    // If the output of the operation is boolean, then the min/max is simple
    using result_type = typename std::invoke_result<BinaryOp, double&, double&>::type;
    if constexpr (std::same_as<result_type, bool>) {
        return {false, true, true};
    }

    // Otherwise the min and max depend on the predecessors

    bool integral = partial_reduce_calculate_integral<BinaryOp>(array_ptr, init);

    // Get the size of the axis we're reducing on
    const ssize_t size = array_ptr->shape()[axis];

    if constexpr (std::same_as<BinaryOp, std::plus<double>>) {
        return {init.value_or(0) + size * array_ptr->min(),
                init.value_or(0) + size * array_ptr->max(), integral};
    }
    if constexpr (std::same_as<BinaryOp, std::multiplies<double>>) {
        const auto& [low, high] =
                product_minmax(size, init.value_or(1), array_ptr->min(), array_ptr->max());
        return {low, high, integral};
    }

    assert(false && "not implemeted yet");
    unreachable();
}

/// TODO: support multiple axes
template <class BinaryOp>
PartialReduceNode<BinaryOp>::PartialReduceNode(ArrayNode* node_ptr, std::span<const ssize_t> axes,
                                               double init)
        : ArrayOutputMixin(partial_reduce_shape(node_ptr->shape(), nonempty(axes)[0])),
          init(init),
          array_ptr_(node_ptr),
          axes_(make_axes(axes)),
          parent_strides_c_(as_contiguous_strides(array_ptr_->shape())),
          values_info_(partial_reduce_calculate_values_info<BinaryOp>(array_ptr_, axes_[0], init)) {
    if (array_ptr_->dynamic()) {
        throw std::invalid_argument("cannot do a partial reduction on a dynamic array");
    } else if (array_ptr_->size() < 1) {
        throw std::invalid_argument("cannot do a partial reduction on an empty array");
    }

    // Validate axes
    /// TODO: support negative axes
    if (axes.size() != 1) {
        throw std::invalid_argument("Partial reduction support only one axis");
    }

    assert(this->ndim() == array_ptr_->ndim() - 1);
    ssize_t axis = axes_[0];
    if (axis < 0 || axis >= array_ptr_->ndim()) {
        throw std::invalid_argument("Axes should be integers between 0 and n_dim - 1");
    }

    add_predecessor(node_ptr);
}

template <class BinaryOp>
PartialReduceNode<BinaryOp>::PartialReduceNode(ArrayNode* node_ptr,
                                               std::initializer_list<ssize_t> axes, double init)
        : PartialReduceNode(node_ptr, std::span(axes), init) {}

template <class BinaryOp>
PartialReduceNode<BinaryOp>::PartialReduceNode(ArrayNode* node_ptr, const ssize_t axis, double init)
        : PartialReduceNode(node_ptr, {axis}, init) {}

template <>
PartialReduceNode<std::multiplies<double>>::PartialReduceNode(ArrayNode* array_ptr,
                                                              const ssize_t axis)
        : PartialReduceNode(array_ptr, axis, 1) {}

template <>
PartialReduceNode<std::multiplies<double>>::PartialReduceNode(ArrayNode* array_ptr,
                                                              std::span<const ssize_t> axes)
        : PartialReduceNode(array_ptr, nonempty(axes)[0], 1) {
    if (axes.size() != 1) {
        throw std::invalid_argument("Partial product supports only one axis");
    }
}

template <>
PartialReduceNode<std::plus<double>>::PartialReduceNode(ArrayNode* array_ptr, const ssize_t axis)
        : PartialReduceNode(array_ptr, axis, 0) {}

template <>
PartialReduceNode<std::plus<double>>::PartialReduceNode(ArrayNode* array_ptr,
                                                        std::span<const ssize_t> axes)
        : PartialReduceNode(array_ptr, nonempty(axes)[0], 0) {
    if (axes.size() != 1) {
        throw std::invalid_argument("Partial sum supports only one axis");
    }
}

template <class BinaryOp>
PartialReduceNode<BinaryOp>::PartialReduceNode(ArrayNode* array_ptr,
                                               std::initializer_list<ssize_t> axes)
        : PartialReduceNode(array_ptr, std::span(axes)) {}

template <class BinaryOp>
PartialReduceNode<BinaryOp>::PartialReduceNode(ArrayNode* array_ptr, const ssize_t axis)
        : PartialReduceNode(array_ptr, {axis}) {}

template <class BinaryOp>
std::span<const ssize_t> PartialReduceNode<BinaryOp>::axes() const {
    // todo: support multiple axes
    return std::span(axes_.get(), 1);
}

template <class BinaryOp>
double const* PartialReduceNode<BinaryOp>::buff(const State& state) const {
    return data_ptr<PartialReduceNodeData<BinaryOp>>(state)->buff();
}

template <class BinaryOp>
void PartialReduceNode<BinaryOp>::commit(State& state) const {
    data_ptr<PartialReduceNodeData<BinaryOp>>(state)->commit();
}

template <class BinaryOp>
std::span<const Update> PartialReduceNode<BinaryOp>::diff(const State& state) const {
    return data_ptr<PartialReduceNodeData<BinaryOp>>(state)->diff();
}

template <class BinaryOp>
void PartialReduceNode<BinaryOp>::initialize_state(State& state) const {
    // the type we use for the reduction depends on the node type
    using accumulator_type = std::conditional<std::same_as<BinaryOp, std::multiplies<double>>,
                                              RunningProduct, double>::type;

    std::vector<accumulator_type> values(size(state));

    const ssize_t axis = axes_[0];

    for (ssize_t index = 0, stop = size(state); index < stop; ++index) {
        assert(index < static_cast<ssize_t>(values.size()));

        /// We wish to fill `index` of the partial reduction, given the state of the parent.
        /// We proceed as follows.
        /// 1. We unravel the index. This will give us e.g. the indices i, k for
        ///     R_ik = sum_j P_ijk
        /// 2. Then we know we have to iterate through the parent starting from j=0 to its full
        /// dimension. To do so, we first find the starting index in the parent.
        /// 3. We then create an iterator through the parent array along the axis.
        /// 4. We use the above iterators to perform the reduction.

        // 1. Unravel the index we are trying to fill
        std::vector<ssize_t> indices = unravel_index(index, this->shape());
        assert(static_cast<ssize_t>(indices.size()) == this->ndim());

        // 2. Find the respective starting index in the parent
        ssize_t start_idx = 0;
        for (ssize_t ax = 0, stop = this->ndim(); ax < stop; ++ax) {
            if (ax >= axis) {
                start_idx += indices[ax] * array_ptr_->strides()[ax + 1] / array_ptr_->itemsize();
            } else {
                start_idx += indices[ax] * array_ptr_->strides()[ax] / array_ptr_->itemsize();
            }
        }
        assert(start_idx >= 0 && index < this->size(state));

        // 3. We create an iterator that starts from index just found and iterates through the axis
        // we are reducing over
        const_iterator begin =
                const_iterator(array_ptr_->buff(state) + start_idx, 1, &array_ptr_->shape()[axis],
                               &array_ptr_->strides()[axis]);

        const_iterator end = begin + array_ptr_->shape(state)[axis];

        // Get the initial value
        double init;
        if (this->init.has_value()) {
            init = this->init.value();
        } else {
            // if there is no init, we use the first value in the array as the init
            // we should only be here if the array is not dynamic and if it has at
            // least one entry
            assert(!array_ptr_->dynamic());
            assert(array_ptr_->size(state) >= 1);
            init = *begin;
            ++begin;
        }

        if constexpr (std::same_as<BinaryOp, std::multiplies<double>>) {
            // we use accumulate rather than reduce so we have an easier lambda
            // description
            values[index] = std::accumulate(begin, end, accumulator_type(init),
                                            [](const RunningProduct& lhs, const double& rhs) {
                                                RunningProduct out = lhs;
                                                out *= rhs;
                                                return out;
                                            });
        } else {
            values[index] = std::reduce(begin, end, init, BinaryOp());
        }
    }

    emplace_data_ptr<PartialReduceNodeData<BinaryOp>>(state, std::move(values));
}

template <class BinaryOp>
ssize_t PartialReduceNode<BinaryOp>::map_parent_index(const State& state,
                                                      ssize_t parent_flat_index) const {
    ssize_t axis = this->axes_[0];
    assert(axis >= 0 && axis < array_ptr_->size(state));
    assert(parent_flat_index >= 0 && parent_flat_index < array_ptr_->size(state));

    ssize_t index = 0;  // the linear index corresponding to the parent index being updated
    ssize_t current_parent_axis = 0;  // current parent axis visited
    ssize_t current_axis = 0;         // current axis

    for (auto stride : parent_strides_c_) {
        // calculate parent index on the current axis (not the linear one)
        ssize_t this_axis_index = parent_flat_index / (stride / array_ptr_->itemsize());

        // update the index now
        parent_flat_index = parent_flat_index % (stride / array_ptr_->itemsize());

        if (current_parent_axis == axis) {
            current_parent_axis++;
            continue;
        }

        // now do the calculation of the linear index of reduction
        index += this_axis_index * (this->strides()[current_axis] / this->itemsize());

        // increase the axis visited
        current_axis++;
        current_parent_axis++;
    }

    return index;
}

template <class BinaryOp>
bool PartialReduceNode<BinaryOp>::integral() const {
    return values_info_.integral;
}

template <class BinaryOp>
double PartialReduceNode<BinaryOp>::min() const {
    return values_info_.min;
}

template <class BinaryOp>
double PartialReduceNode<BinaryOp>::max() const {
    return values_info_.max;
}

template <class BinaryOp>
void PartialReduceNode<BinaryOp>::propagate(State& state) const {
    auto ptr = data_ptr<PartialReduceNodeData<BinaryOp>>(state);

    for (const auto& [p_index, old, value] : array_ptr_->diff(state)) {
        const ssize_t index = map_parent_index(state, p_index);
        ptr->update(index, old, value);
    }

    if (ptr->diff().size()) Node::propagate(state);
}

template <class BinaryOp>
void PartialReduceNode<BinaryOp>::revert(State& state) const {
    data_ptr<PartialReduceNodeData<BinaryOp>>(state)->revert();
}

template <class BinaryOp>
std::span<const ssize_t> PartialReduceNode<BinaryOp>::shape(const State& state) const {
    return this->shape();
}

template <class BinaryOp>
ssize_t PartialReduceNode<BinaryOp>::size(const State& state) const {
    return this->size();
}

template <class BinaryOp>
ssize_t PartialReduceNode<BinaryOp>::size_diff(const State& state) const {
    return data_ptr<PartialReduceNodeData<BinaryOp>>(state)->size_diff();
}

// Uncommented are the tested specializations
template class PartialReduceNode<std::multiplies<double>>;
template class PartialReduceNode<std::plus<double>>;

// Create a storage class for op-specific values.
template <class BinaryOp>
struct ExtraData {
    void commit() {}
    void revert() {}
};

template <>
struct ExtraData<std::logical_and<double>> {
    explicit ExtraData(ssize_t num_zero) : num_zero(num_zero) {}

    virtual ~ExtraData() = default;

    void commit() { old_num_zero = num_zero; }
    void revert() { num_zero = old_num_zero; }

    ssize_t num_zero;
    ssize_t old_num_zero = num_zero;
};

template <>
struct ExtraData<std::logical_or<double>> {
    explicit ExtraData(ssize_t num_nonzero) : num_nonzero(num_nonzero) {}

    virtual ~ExtraData() = default;

    void commit() { old_num_nonzero = num_nonzero; }
    void revert() { num_nonzero = old_num_nonzero; }

    ssize_t num_nonzero;  // num nonzero, following SciPy's naming
    ssize_t old_num_nonzero = num_nonzero;
};

template <>
struct ExtraData<std::multiplies<double>> {
    explicit ExtraData(RunningProduct product) : product(product), old_product(product) {}

    virtual ~ExtraData() = default;

    void commit() { old_product = product; }
    void revert() { product = old_product; }

    // RunningProduct tracks the multiplications by 0 separately, which means we
    // don't need to recalculate everything when there's a 0 in there.
    RunningProduct product;
    RunningProduct old_product;
};

template <class BinaryOp>
struct ReduceNodeData : NodeStateData {
    template <class... Ts>
    explicit ReduceNodeData(double value, Ts... extra_args)
            : values(0, value, value), extra(extra_args...) {}

    double const* buff() const { return &values.value; }

    void commit() {
        values.old = values.value;
        extra.commit();
    }
    void revert() {
        values.value = values.old;
        extra.revert();
    }

    Update values;

    // op-specific storage.
    ExtraData<BinaryOp> extra;
};

template <class BinaryOp>
bool reduce_calculate_integral(const Array* array_ptr, const std::optional<double>& init) {
    using result_type = typename std::invoke_result<BinaryOp, double&, double&>::type;

    if constexpr (std::is_integral<result_type>::value) {
        return true;
    }

    if constexpr (std::is_same<BinaryOp, functional::max<double>>::value ||
                  std::is_same<BinaryOp, functional::min<double>>::value ||
                  std::is_same<BinaryOp, std::multiplies<double>>::value ||
                  std::is_same<BinaryOp, std::plus<double>>::value) {
        // the actual value of init doesn't matter, all of the above have no default
        // or an integer default init
        return array_ptr->integral() && is_integer(init.value_or(0));
    }

    assert(false && "not implemeted yet");
    unreachable();
}

template <class BinaryOp>
ValuesInfo reduce_calculate_values_info(const Array* array_ptr, const std::optional<double>& init) {
    // If the output of the operation is boolean, the min/max are just [false, true]
    using result_type = typename std::invoke_result<BinaryOp, double&, double&>::type;
    if constexpr (std::same_as<result_type, bool>) {
        return ValuesInfo::logical_output();
    }

    // Otherwise the min and max depend on the predecessors

    auto op = BinaryOp();

    bool integral = reduce_calculate_integral<BinaryOp>(array_ptr, init);

    auto low = array_ptr->min();
    auto high = array_ptr->max();

    // These can results in inf. If we fix that in initialization/propagation we
    // should also fix it here.
    if constexpr (std::same_as<BinaryOp, functional::max<double>> ||
                  std::same_as<BinaryOp, functional::min<double>>) {
        if (init.has_value()) {
            low = op(low, init.value());
            high = op(high, init.value());
        }
        return {low, high, integral};
    }
    if constexpr (std::same_as<BinaryOp, std::multiplies<double>>) {
        // the dynamic case. For now let's just fall back to Array's default
        // implementation because this gets even more complicated
        if (array_ptr->dynamic()) {
            return {std::numeric_limits<double>::lowest(), std::numeric_limits<double>::max(),
                    integral};
        }

        auto const& [this_low, this_high] =
                product_minmax(array_ptr->size(), init.value_or(1), low, high);
        return {this_low, this_high, integral};
    }
    if constexpr (std::same_as<BinaryOp, std::plus<double>>) {
        const double init_ = init.value_or(0);

        // if the array has a finite fixed size, then just multuply the largest value
        // by that size
        if (const ssize_t size = array_ptr->size(); size >= 0) {
            return {init_ + size * low, init_ + size * high, integral};
        }

        // our predecessor array is dynamic. So there are a few more cases
        // we need to check

        // 100 is a magic number. It's how far back in the predecessor
        // chain to check to get good bounds on the size for the given array.
        // This will exit early if it converges.
        const SizeInfo sizeinfo = array_ptr->sizeinfo().substitute(100);

        if (high > 0) {
            // if high is positive, then we're interested in the maxmum size
            // of the array

            // if the array is arbitrarily large, then just fall back to the
            // default max.
            if (!sizeinfo.max.has_value()) {
                high = std::numeric_limits<double>::max();
            } else {
                high = init_ + sizeinfo.max.value() * high;
            }
        } else {
            // if high is negative, then we're interested in the minimum size
            // of the array
            high = init_ + sizeinfo.min.value_or(0) * high;
        }

        if (low < 0) {
            // if low is negative, then we're interested in the maximum size
            // of the array

            // if the array is arbitrarily large, then just fall back to the
            // default min.
            if (!sizeinfo.max.has_value()) {
                low = std::numeric_limits<double>::lowest();
            } else {
                low = init_ + sizeinfo.max.value() * low;
            }
        } else {
            // if low is positive, then we're interested in the minimum size
            // of the array
            low = init_ + sizeinfo.min.value_or(0) * low;
        }

        return {low, high, integral};
    }

    assert(false && "not implemeted yet");
    unreachable();
}

template <class BinaryOp>
ReduceNode<BinaryOp>::ReduceNode(ArrayNode* node_ptr, double init)
        : init(init),
          array_ptr_(node_ptr),
          values_info_(reduce_calculate_values_info<BinaryOp>(array_ptr_, init)) {
    add_predecessor(node_ptr);
}

template <>
ReduceNode<std::logical_and<double>>::ReduceNode(ArrayNode* array_ptr) : ReduceNode(array_ptr, 1) {}

template <>
ReduceNode<std::logical_or<double>>::ReduceNode(ArrayNode* array_ptr) : ReduceNode(array_ptr, 0) {}

template <>
ReduceNode<std::multiplies<double>>::ReduceNode(ArrayNode* array_ptr) : ReduceNode(array_ptr, 1) {}

template <>
ReduceNode<std::plus<double>>::ReduceNode(ArrayNode* array_ptr) : ReduceNode(array_ptr, 0) {}

template <class BinaryOp>
ReduceNode<BinaryOp>::ReduceNode(ArrayNode* array_ptr)
        : init(),
          array_ptr_(array_ptr),
          values_info_(reduce_calculate_values_info<BinaryOp>(array_ptr_, init)) {
    if (array_ptr_->dynamic()) {
        throw std::invalid_argument(
                "cannot do a reduction on a dynamic array with an operation that has no identity "
                "without supplying an initial value");
    } else if (array_ptr_->size() < 1) {
        throw std::invalid_argument(
                "cannot do a reduction on an empty array with an operation that has no identity "
                "without supplying an initial value");
    }

    add_predecessor(array_ptr);
}

template <class BinaryOp>
void ReduceNode<BinaryOp>::commit(State& state) const {
    data_ptr<ReduceNodeData<BinaryOp>>(state)->commit();
}

template <class BinaryOp>
double const* ReduceNode<BinaryOp>::buff(const State& state) const {
    return data_ptr<ReduceNodeData<BinaryOp>>(state)->buff();
}

template <class BinaryOp>
std::span<const Update> ReduceNode<BinaryOp>::diff(const State& state) const {
    const Update& update = data_ptr<ReduceNodeData<BinaryOp>>(state)->values;
    return std::span<const Update>(&update, static_cast<int>(update.old != update.value));
}

template <class BinaryOp>
void ReduceNode<BinaryOp>::initialize_state(State& state) const {
    emplace_data_ptr<ReduceNodeData<BinaryOp>>(state, reduce(state));
}

template <>
void ReduceNode<std::logical_and<double>>::initialize_state(State& state) const {
    ssize_t num_zero = init.value_or(1) ? 0 : 1;
    for (const double value : array_ptr_->view(state)) {
        num_zero += !value;
    }

    emplace_data_ptr<ReduceNodeData<std::logical_and<double>>>(state, !num_zero, num_zero);
}

template <>
void ReduceNode<std::logical_or<double>>::initialize_state(State& state) const {
    ssize_t num_nonzero = init.value_or(1) ? 1 : 0;
    for (const double value : array_ptr_->view(state)) {
        num_nonzero += static_cast<bool>(value);
    }

    emplace_data_ptr<ReduceNodeData<std::logical_or<double>>>(state, num_nonzero > 0, num_nonzero);
}

template <>
void ReduceNode<std::multiplies<double>>::initialize_state(State& state) const {
    // there is an edge case here for init being 0, in that case the `nonzero`
    // component will always be 0, which is dumb but everything still works and
    // it's enough of an edge case that I don't think it makes sense to do
    // anything-performance wise.

    RunningProduct product(init.value_or(1));

    for (const double value : array_ptr_->view(state)) {
        product *= value;
    }

    // and then create the state
    emplace_data_ptr<ReduceNodeData<std::multiplies<double>>>(state, static_cast<double>(product),
                                                              product);
}

template <class BinaryOp>
bool ReduceNode<BinaryOp>::integral() const {
    return values_info_.integral;
}

template <class BinaryOp>
double ReduceNode<BinaryOp>::min() const {
    return values_info_.min;
}

template <class BinaryOp>
double ReduceNode<BinaryOp>::max() const {
    return values_info_.max;
}

template <>
void ReduceNode<std::logical_and<double>>::propagate(State& state) const {
    auto ptr = data_ptr<ReduceNodeData<std::logical_and<double>>>(state);

    ssize_t& num_zero = ptr->extra.num_zero;

    // count the change in the num_zero
    for (const Update& update : array_ptr_->diff(state)) {
        const auto& [_, old, value] = update;

        if (update.placed()) {
            // added a value to the array
            num_zero += !value;
        } else if (update.removed()) {
            // removed a value from the array
            num_zero -= !old;
        } else if (!old && value) {
            // changed a 0 to a truthy value
            num_zero -= 1;
        } else if (old && !value) {
            // changed a truthy value to a 0
            num_zero += 1;
        } else {
            // otherwise we don't care because the net number of 0s hasn't changed
            assert(static_cast<bool>(old) == static_cast<bool>(value));
        }
    }

    assert(num_zero >= 0);  // should never go negative

    ptr->values.value = num_zero < 1;
    if (ptr->values.value != ptr->values.old) Node::propagate(state);
}

template <>
void ReduceNode<std::logical_or<double>>::propagate(State& state) const {
    auto ptr = data_ptr<ReduceNodeData<std::logical_or<double>>>(state);

    ssize_t& num_nonzero = ptr->extra.num_nonzero;

    // count the change in the num_nonzero
    for (const Update& update : array_ptr_->diff(state)) {
        const auto& [_, old, value] = update;

        if (update.placed()) {
            // added a value to the array
            num_nonzero += static_cast<bool>(value);
        } else if (update.removed()) {
            // removed a value from the array
            num_nonzero -= static_cast<bool>(old);
        } else if (!old && value) {
            // changed a 0 to a truthy value
            num_nonzero += 1;
        } else if (old && !value) {
            // changed a truthy value to a 0
            num_nonzero -= 1;
        } else {
            // otherwise we don't care because the net number of 0s hasn't changed
            assert(static_cast<bool>(old) == static_cast<bool>(value));
        }
    }

    assert(num_nonzero >= 0);  // should never go negative

    ptr->values.value = num_nonzero > 0;
    if (ptr->values.value != ptr->values.old) Node::propagate(state);
}

template <>
void ReduceNode<functional::max<double>>::propagate(State& state) const {
    auto ptr = data_ptr<ReduceNodeData<functional::max<double>>>(state);

    auto& value = ptr->values.value;

    bool reinitialize = false;
    for (const Update& update : array_ptr_->diff(state)) {
        if (update.removed()) {
            if (update.old >= value) {
                // uh oh, we may have just removed the known max. So we need to
                // recalculate everything from scratch
                reinitialize = true;
                break;
            }
        } else if (update.placed() || update.value > update.old) {
            // we added a new value or increased an existing once
            value = std::max(update.value, value);
        } else if (update.old == value && update.old != update.value) {
            // we potentially made the current max smaller
            assert(update.value < update.old);
            reinitialize = true;
            break;
        }
    }

    if (reinitialize) {
        value = reduce(state);
    }

    Node::propagate(state);
}

template <>
void ReduceNode<functional::min<double>>::propagate(State& state) const {
    auto ptr = data_ptr<ReduceNodeData<functional::min<double>>>(state);

    auto& value = ptr->values.value;

    bool reinitialize = false;
    for (const Update& update : array_ptr_->diff(state)) {
        if (update.removed()) {
            if (update.old <= value) {
                // uh oh, we may have just removed the known min. So we need to
                // recalculate everything from scratch
                reinitialize = true;
                break;
            }
        } else if (update.placed() || update.value < update.old) {
            // we added a new value, or decreased an existing one
            value = std::min(update.value, value);
        } else if (update.old == value && update.old != update.value) {
            // we potentially made the current min larger
            assert(update.value > update.old);
            reinitialize = true;
            break;
        }
    }

    if (reinitialize) {
        value = reduce(state);
    }

    Node::propagate(state);
}

template <>
void ReduceNode<std::multiplies<double>>::propagate(State& state) const {
    auto ptr = data_ptr<ReduceNodeData<std::multiplies<double>>>(state);

    RunningProduct& product = ptr->extra.product;

    for (const Update& update : array_ptr_->diff(state)) {
        if (update.placed()) {
            product *= update.value;
        } else if (update.removed()) {
            product /= update.old;
        } else {
            product.multiply_divide(update.value, update.old);
        }
    }

    ptr->values.value = static_cast<double>(product);
    if (ptr->values.value != ptr->values.old) Node::propagate(state);
}

template <>
void ReduceNode<std::plus<double>>::propagate(State& state) const {
    auto ptr = data_ptr<ReduceNodeData<std::plus<double>>>(state);

    // todo: consider kahan summation

    auto& value = ptr->values.value;

    for (const Update& update : array_ptr_->diff(state)) {
        if (update.placed()) {
            value += update.value;
        } else if (update.removed()) {
            value -= update.old;
        } else {
            value += update.value - update.old;
        }
    }

    Node::propagate(state);
}

template <class BinaryOp>
double ReduceNode<BinaryOp>::reduce(const State& state) const {
    auto start = array_ptr_->begin(state);
    const auto end = array_ptr_->end(state);

    double init;

    if (this->init.has_value()) {
        init = this->init.value();
    } else {
        // if there is no init, we use the first value in the array as the init
        // we should only be here if the array is not dynamic and if it has at
        // least one entry
        assert(!array_ptr_->dynamic());
        assert(array_ptr_->size(state) >= 1);
        assert(start != end);

        init = *start;
        ++start;
    }

    return std::reduce(start, end, init, BinaryOp());
}

template <class BinaryOp>
void ReduceNode<BinaryOp>::revert(State& state) const {
    data_ptr<ReduceNodeData<BinaryOp>>(state)->revert();
}

// Uncommented are the tested specializations
template class ReduceNode<functional::max<double>>;
template class ReduceNode<functional::min<double>>;
template class ReduceNode<std::logical_and<double>>;
template class ReduceNode<std::logical_or<double>>;
template class ReduceNode<std::multiplies<double>>;
template class ReduceNode<std::plus<double>>;

}  // namespace dwave::optimization

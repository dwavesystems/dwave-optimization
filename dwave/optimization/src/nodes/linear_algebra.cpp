// Copyright 2025 D-Wave Inc.
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

#include "dwave-optimization/nodes/linear_algebra.hpp"

#include <algorithm>
#include <array>
#include <ranges>

#include "../functional_.hpp"
#include "_state.hpp"
#include "dwave-optimization/array.hpp"
#include "dwave-optimization/state.hpp"

namespace dwave::optimization {

////////////////////// MatrixMultiplyNode

// Valid shapes to multiply, and the resulting shape
// (-1, 2, 5, 3) and (-1, 2, 3, 7) -> (-1, 2, 5, 7)
// (-1, 3) and (3) -> (-1)
// (-1, 3) and (3, 7) -> (-1, 7)
// (-1) and (-1) -> ()
// (-1) and (-1, 5) -> (5)

ssize_t size_from_shape(std::span<const ssize_t> shape) {
    return std::reduce(shape.begin(), shape.end(), 1, std::multiplies<ssize_t>());
}

ssize_t get_axis_size(std::span<const ssize_t> shape, ssize_t index, bool vector_as_row) {
    // If vector_as_row is true, treat vector as shape (1, size), else as shape (size, 1)
    assert(index < 0);
    if (shape.size() == 0) return 1;
    if (shape.size() == 1 and index == -2 and vector_as_row) return 1;
    if (shape.size() == 1 and index == -1 and not vector_as_row) return 1;
    if (shape.size() == 1) return shape.back();
    return shape[shape.size() + index];
}

std::vector<ssize_t> output_shape(const ArrayNode* x_ptr, const ArrayNode* y_ptr) {
    if (x_ptr->ndim() == 0 or y_ptr->ndim() == 0) {
        throw std::invalid_argument("operands cannot be scalar");
    }

    // Check that last dimension of x matches the second last dimension of y
    ssize_t x_last_axis_size = get_axis_size(x_ptr->shape(), -1, true);
    ssize_t y_penultimate_axis_size = get_axis_size(y_ptr->shape(), -2, false);
    if (x_last_axis_size != y_penultimate_axis_size) {
        throw std::invalid_argument(
                "the last dimension of `x` is not the same size as the second to last dimension of "
                "`y`");
    } else if (x_last_axis_size == -1) {
        assert(x_ptr->dynamic() && y_ptr->dynamic());
        // Both are dynamic. We need to check that the dynamic dimension is
        // always the same size.
        ssize_t x_subspace_size = -1 * size_from_shape(x_ptr->shape());
        ssize_t y_subspace_size = -1 * size_from_shape(y_ptr->shape());
        if (x_ptr->sizeinfo() / x_subspace_size != y_ptr->sizeinfo() / y_subspace_size) {
            throw std::invalid_argument(
                    "the last dimension of `x` is not the same size as the second to last "
                    "dimension of `y`");
        }
    }

    // Now check that the leading subspace shape is identical (no broadcasting for now)
    if (x_ptr->ndim() >= 2 && y_ptr->ndim() >= 2) {
        if (x_ptr->ndim() != y_ptr->ndim()) {
            throw std::invalid_argument(
                    "operands have different dimensions (use BroadcastNode if you wish to "
                    "broadcast missing dimensions)");
        }
        for (ssize_t i = 0, stop = x_ptr->ndim() - 2; i < stop; i++) {
            if (x_ptr->shape()[i] != y_ptr->shape()[i]) {
                throw std::invalid_argument(
                        "operands must have matching leading shape (up to the last two "
                        "dimensions)");
            }
        }
    }

    // Now we now the leading axes match, we can construct the output shape
    std::vector<ssize_t> shape;
    // If x is being broadcast, we need to add the axes from the start of y
    if (y_ptr->ndim() > 2 && y_ptr->ndim() > x_ptr->ndim()) {
        const ssize_t num_x_leading = std::max<ssize_t>(0, x_ptr->ndim() - 2);
        const ssize_t num_y_leading = std::max<ssize_t>(0, y_ptr->ndim() - 2);
        for (ssize_t d : y_ptr->shape() | std::views::take(num_y_leading - num_x_leading)) {
            shape.push_back(d);
        }
    }
    for (ssize_t d : x_ptr->shape() | std::views::take(x_ptr->ndim() - 1)) {
        shape.push_back(d);
    }
    if (y_ptr->ndim() >= 2) {
        shape.push_back(y_ptr->shape().back());
    }

    return shape;
}

SizeInfo get_sizeinfo(const ArrayNode* x_ptr, const ArrayNode* y_ptr) {
    if (y_ptr->dynamic() and y_ptr->ndim() <= 2) {
        // x must also be dynamic, and we must be contracting along the dynamic
        // dimension, so the output is fixed size.
        std::vector<ssize_t> shape = output_shape(x_ptr, y_ptr);
        ssize_t size = size_from_shape(shape);
        assert(size >= 1);
        return SizeInfo(size);
    }
    assert(x_ptr->shape().back() != -1);
    SizeInfo sizeinfo = x_ptr->sizeinfo() / x_ptr->shape().back();
    if (y_ptr->ndim() == 2 && y_ptr->dynamic()) {
        assert(x_ptr->dynamic() && x_ptr->ndim() == 1);
    } else if (y_ptr->ndim() >= 2) {
        assert(y_ptr->shape().back() != -1);
        sizeinfo *= y_ptr->shape().back();
    }
    return sizeinfo;
}

ValuesInfo get_values_info(const ArrayNode* x_ptr, const ArrayNode* y_ptr) {
    // Get all possible combinations of values
    std::array<double, 4> combos{x_ptr->min() * y_ptr->min(), x_ptr->min() * y_ptr->max(),
                                 x_ptr->max() * y_ptr->min(), x_ptr->max() * y_ptr->max()};

    double min_val = std::ranges::min(combos);
    double max_val = std::ranges::max(combos);

    ssize_t x_subspace_size = std::reduce(x_ptr->shape().begin(), x_ptr->shape().end() - 1, 1,
                                          std::multiplies<ssize_t>());
    SizeInfo contracted_axis_size = x_ptr->sizeinfo() / x_subspace_size;

    if (contracted_axis_size.max.has_value() and *contracted_axis_size.max == 0) {
        // Output will always be empty, so we can return early
        return ValuesInfo(0.0, 0.0, true);
    }

    // Use default constructor to get default min/max
    ValuesInfo values_info;
    values_info.integral = x_ptr->integral() && y_ptr->integral();

    if (contracted_axis_size.max.has_value()) {
        if (max_val >= 0) values_info.max = max_val * contracted_axis_size.max.value();
        if (min_val <= 0) values_info.min = min_val * contracted_axis_size.max.value();
    }

    ssize_t min_size = contracted_axis_size.min.value_or(0);
    if (max_val < 0) values_info.max = max_val * min_size;
    if (min_val > 0) values_info.min = min_val * min_size;

    return values_info;
}

std::vector<ssize_t> atleast_2d_shape(std::span<const ssize_t> shape, bool as_row) {
    if (shape.size() == 0) return {1, 1};
    if (shape.size() == 1 and as_row) return {1, shape[0]};
    if (shape.size() == 1 and not as_row) return {shape[0], 1};
    return {shape.begin(), shape.end()};
}

class MatrixMultiplyNodeData : public ArrayNodeStateData {
 public:
    explicit MatrixMultiplyNodeData(std::vector<double>&& values, std::span<const ssize_t> shape)
            : ArrayNodeStateData(std::move(values)), shape(shape.begin(), shape.end()) {}

    std::vector<double> output;
    std::vector<ssize_t> shape;
};

MatrixMultiplyNode::MatrixMultiplyNode(ArrayNode* x_ptr, ArrayNode* y_ptr)
        : ArrayOutputMixin(output_shape(x_ptr, y_ptr)),
          x_ptr_(x_ptr),
          y_ptr_(y_ptr),
          sizeinfo_(get_sizeinfo(x_ptr, y_ptr)),
          values_info_(get_values_info(x_ptr, y_ptr)) {
    add_predecessor(x_ptr);
    add_predecessor(y_ptr);
}

ssize_t get_leading_stride(std::span<const ssize_t> shape) {
    if (shape.size() < 2) return 0;  // handles broadcasting for the vector case
    return shape.back() * shape[shape.size() - 2];
}

ssize_t get_stride(std::span<const ssize_t> shape, ssize_t index, bool as_row) {
    assert(index < 0 && index >= -2);
    if (get_axis_size(shape, index, as_row) == 1) return 0;
    if (index + 1 == 0) return 1;
    return get_axis_size(shape, index + 1, as_row);
}

ssize_t get_leading_subspace_size(std::span<const ssize_t> x_shape,
                                  std::span<const ssize_t> y_shape) {
    auto shape = x_shape.size() > y_shape.size() ? x_shape : y_shape;
    const ssize_t penultimate_axis = std::max<ssize_t>(0, static_cast<ssize_t>(shape.size()) - 2);
    return std::reduce(shape.begin(), shape.begin() + penultimate_axis, 1,
                       std::multiplies<ssize_t>());
}

void MatrixMultiplyNode::matmul(State& state, std::span<double> out,
                                std::span<const ssize_t> out_shape) const {
    auto x_data = x_ptr_->view(state);
    auto y_data = y_ptr_->view(state);

    const ssize_t x_penultimate_axis_size = get_axis_size(x_ptr_->shape(state), -2, true);
    const ssize_t leading_subspace_size =
            get_leading_subspace_size(x_ptr_->shape(state), y_ptr_->shape(state));

    const ssize_t x_leading_stride = get_leading_stride(x_ptr_->shape(state));
    const ssize_t y_leading_stride = get_leading_stride(y_ptr_->shape(state));
    const ssize_t out_leading_stride = [&]() -> ssize_t {
        if (x_ptr_->ndim() >= 2 and y_ptr_->ndim() >= 2) return get_leading_stride(out_shape);
        if (x_ptr_->ndim() == 1 and y_ptr_->ndim() == 1) return 0;
        return out_shape.back();
    }();

    const ssize_t y_last_axis_size = get_axis_size(y_ptr_->shape(state), -1, false);
    const ssize_t y_penultimate_axis_size = get_axis_size(y_ptr_->shape(state), -2, false);

    // TODO: consider using the parent arrays' strides directly
    // const ssize_t x_penultimate_stride = get_axis_size(x_ptr_->shape(state), -1, true);
    const ssize_t x_penultimate_stride = get_stride(x_ptr_->shape(state), -2, true);
    const ssize_t x_last_stride = 1;

    const ssize_t y_penultimate_stride = y_last_axis_size;
    const ssize_t y_last_stride = y_ptr_->ndim() >= 2 ? 1 : 0;

    const ssize_t out_penultimate_stride = [&]() -> ssize_t {
        if (y_ptr_->ndim() == 1) return 1;
        return get_axis_size(out_shape, -1, false);
    }();

    for (ssize_t w = 0; w < leading_subspace_size; w++) {
        for (ssize_t i = 0; i < x_penultimate_axis_size; i++) {
            for (ssize_t j = 0; j < y_last_axis_size; j++) {
                auto x = x_data.begin() + w * x_leading_stride + i * x_penultimate_stride;
                auto y = y_data.begin() + w * y_leading_stride + j * y_last_stride;
                double& out_val = out[w * out_leading_stride + i * out_penultimate_stride + j];
                out_val = 0.0;
                for (ssize_t k = 0; k < y_penultimate_axis_size; k++) {
                    out_val += *x * *y;
                    x += x_last_stride;
                    y += y_penultimate_stride;
                }
            }
        }
    }
}

void MatrixMultiplyNode::initialize_state(State& state) const {
    ssize_t start_size = this->size();
    std::vector<ssize_t> shape(this->shape().begin(), this->shape().end());
    if (this->dynamic()) {
        shape[0] = x_ptr_->shape(state)[0];
        start_size = size_from_shape(shape);
    }

    std::vector<double> data(start_size);
    matmul(state, data, shape);
    emplace_data_ptr<MatrixMultiplyNodeData>(state, std::move(data), shape);
}

double const* MatrixMultiplyNode::buff(const State& state) const {
    return data_ptr<MatrixMultiplyNodeData>(state)->buff();
}

void MatrixMultiplyNode::commit(State& state) const {
    return data_ptr<MatrixMultiplyNodeData>(state)->commit();
}

std::span<const Update> MatrixMultiplyNode::diff(const State& state) const {
    return data_ptr<MatrixMultiplyNodeData>(state)->diff();
}

bool MatrixMultiplyNode::integral() const { return values_info_.integral; }

double MatrixMultiplyNode::max() const { return values_info_.max; }

double MatrixMultiplyNode::min() const { return values_info_.min; }

void MatrixMultiplyNode::update_shape(State& state) const {
    if (this->dynamic()) {
        data_ptr<MatrixMultiplyNodeData>(state)->shape[0] = x_ptr_->shape(state)[0];
    }
}

void MatrixMultiplyNode::propagate(State& state) const {
    if (x_ptr_->diff(state).size() == 0 and y_ptr_->diff(state).size() == 0) return;

    auto data = data_ptr<MatrixMultiplyNodeData>(state);

    this->update_shape(state);
    ssize_t new_size = size_from_shape(data->shape);

    data->output.resize(new_size);

    this->matmul(state, data->output, data->shape);
    data->assign(data->output);
}

void MatrixMultiplyNode::revert(State& state) const {
    auto data = data_ptr<MatrixMultiplyNodeData>(state);
    data->revert();
    this->update_shape(state);
}

std::span<const ssize_t> MatrixMultiplyNode::shape(const State& state) const {
    if (not this->dynamic()) return this->shape();
    return data_ptr<MatrixMultiplyNodeData>(state)->shape;
}

ssize_t MatrixMultiplyNode::size(const State& state) const {
    if (not this->dynamic()) return this->size();
    return data_ptr<MatrixMultiplyNodeData>(state)->size();
}

ssize_t MatrixMultiplyNode::size_diff(const State& state) const {
    if (not this->dynamic()) return 0;
    return data_ptr<MatrixMultiplyNodeData>(state)->size_diff();
}

SizeInfo MatrixMultiplyNode::sizeinfo() const { return sizeinfo_; }

}  // namespace dwave::optimization

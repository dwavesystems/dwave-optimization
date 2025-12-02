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

#if __has_include(<openblas_config.h>) and __has_include(<cblas.h>)
#include "cblas.h"
#define HAS_BLAS_
#endif

namespace dwave::optimization {

// 2d matrix multiplication with a BLAS-like interface.
//
// This function will overloaded by an implementation that uses SciPy OpenBLAS
// if it's available.
//
// A is a (m,k) matrix
// B is a (k,n) matrix
// C is a (m,n) matrix
template <DType T>
void gemm(const ssize_t m, const ssize_t n, const ssize_t k,        // size of the arrays
          const T* const A, std::span<const ssize_t, 2> A_strides,  // lhs matrix
          const T* const B, std::span<const ssize_t, 2> B_strides,  // rhs matrix
          T* const C, std::span<const ssize_t, 2> C_strides) {      // output matrix

    constexpr ssize_t num_bytes = sizeof(T);

    const std::array<ssize_t, 2> A_leap{A_strides[0] / num_bytes, A_strides[1] / num_bytes};
    const std::array<ssize_t, 2> B_leap{B_strides[0] / num_bytes, B_strides[1] / num_bytes};
    const std::array<ssize_t, 2> C_leap{C_strides[0] / num_bytes, C_strides[1] / num_bytes};

    for (ssize_t i = 0; i < m; ++i) {
        for (ssize_t j = 0; j < n; ++j) {
            const double* a = A + i * A_leap[0];
            const double* b = B + j * B_leap[1];
            double* c = C + i * C_leap[0] + j * C_leap[1];

            *c = 0;
            for (ssize_t p = 0; p < k; ++p, a += A_leap[1], b += B_leap[0]) {
                *c += *a * *b;
            }
        }
    }
}

#ifdef HAS_BLAS_

// Given a strided 2D array, dump it to a contiguous vector.
std::vector<double> make_contiguous(const double* const start,  // beginning of the array
                                    const ssize_t rows, const ssize_t cols,  // assume 2D
                                    std::span<const ssize_t, 2> strides) {
    const ssize_t row_leap = strides[0] / sizeof(double);
    const ssize_t col_leap = strides[1] / sizeof(double);

    std::vector<double> out;
    out.reserve(rows * cols);

    for (ssize_t i = 0; i < rows; ++i) {
        const double* ptr = start + i * row_leap;
        for (ssize_t j = 0; j < cols; ++j, ptr += col_leap) {
            out.emplace_back(*ptr);
        }
    }

    return out;
}

// An overload of gemm() that uses BLAS for matmul with doubles. We could also extend this to
// float in the future without a lot of fuss.
template <>
void gemm<double>(const ssize_t m, const ssize_t n, const ssize_t k,        // size of the arrays
          const double* const A, std::span<const ssize_t, 2> A_strides,  // lhs matrix
          const double* const B, std::span<const ssize_t, 2> B_strides,  // rhs matrix
          double* const C, std::span<const ssize_t, 2> C_strides) {      // output matrix
    // OpenBLAS has some requirements for A,B,C.
    // Specifically, they must be contiguous within each "row" and they must
    // have positive strides in the first dimension.
    // So, if they don't satisfy those requirements, we copy the state into
    // a contiguous array that does.

    if (A_strides[0] < 0 or A_strides[1] != sizeof(double)) {
        std::vector<double> a = make_contiguous(A, m, k, A_strides);
        std::array<ssize_t, 2> a_strides{k * static_cast<ssize_t>(sizeof(double)), sizeof(double)};
        return gemm(m, n, k,              // same size
                    a.data(), a_strides,  // new A
                    B, B_strides,         // same B
                    C, C_strides          // same C
        );
    }
    assert(A_strides[0] % sizeof(double) == 0);

    if (B_strides[0] < 0 or B_strides[1] != sizeof(double)) {
        std::vector<double> b = make_contiguous(B, k, n, B_strides);
        std::array<ssize_t, 2> b_strides{n * static_cast<ssize_t>(sizeof(double)), sizeof(double)};
        return gemm(m, n, k,              // same size
                    A, A_strides,         // same A
                    b.data(), b_strides,  // new B
                    C, C_strides          // same C
        );
    }
    assert(B_strides[0] % sizeof(double) == 0);

    // We could handle non-contiguous C, but it should be true for our MatrixMultiplyNode
    // always so for now let's leave it alone.
    assert(C_strides[0] == n * static_cast<ssize_t>(sizeof(double)));
    assert(C_strides[1] == sizeof(double));

    // Ok! Everything is in good shape, now all that's left is to call BLAS.
    // Do note that BLAS's strides are *not* counted in bytes.

    scipy_cblas_dgemm64_(CblasRowMajor,  // OPENBLAS_CONST enum CBLAS_ORDER Order,
                         CblasNoTrans,   // OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
                         CblasNoTrans,   // OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB,
                         m,              // OPENBLAS_CONST blasint M,
                         n,              // OPENBLAS_CONST blasint N,
                         k,              // OPENBLAS_CONST blasint K,
                         1,              // OPENBLAS_CONST double alpha,
                         A,              // OPENBLAS_CONST double *A,
                         A_strides[0] / sizeof(double),  // OPENBLAS_CONST blasint lda,
                         B,                              // OPENBLAS_CONST double *B,
                         B_strides[0] / sizeof(double),  // OPENBLAS_CONST blasint ldb,
                         0,                              // OPENBLAS_CONST double beta,
                         C,                              // double *C,
                         C_strides[0] / sizeof(double)   // OPENBLAS_CONST blasint ldc
    );
}

#endif

////////////////////// MatrixMultiplyNode

// Valid shapes to multiply, and the resulting shape
// (-1, 2, 5, 3) and (-1, 2, 3, 7) -> (-1, 2, 5, 7)
// (-1, 3) and (3) -> (-1)
// (-1, 3) and (3, 7) -> (-1, 7)
// (-1) and (-1) -> ()
// (-1) and (-1, 5) -> (5)

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
    const ssize_t x_last_axis_size = get_axis_size(x_ptr->shape(), -1, true);
    const ssize_t y_penultimate_axis_size = get_axis_size(y_ptr->shape(), -2, false);
    if (x_last_axis_size != y_penultimate_axis_size) {
        throw std::invalid_argument(
                "the last dimension of `x` is not the same size as the second to last dimension of "
                "`y`");
    } else if (x_last_axis_size == -1) {
        assert(x_ptr->dynamic() && y_ptr->dynamic());
        // Both are dynamic. We need to check that the dynamic dimension is
        // always the same size.
        const ssize_t x_subspace_size = Array::shape_to_size(x_ptr->shape().subspan(1));
        const ssize_t y_subspace_size = Array::shape_to_size(y_ptr->shape().subspan(1));
        assert(x_subspace_size != Array::DYNAMIC_SIZE);
        assert(y_subspace_size != Array::DYNAMIC_SIZE);
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
        ssize_t size = Array::shape_to_size(shape);
        assert(size >= 1);
        return SizeInfo(size);
    }

    // The size should be x's size, divided by the size of x's last dimension,
    // multiplied by the size of y's last dimension (if matrix or higher dim).
    assert(x_ptr->shape().back() != -1);
    SizeInfo sizeinfo = x_ptr->sizeinfo() / x_ptr->shape().back();
    if (y_ptr->ndim() >= 2) {
        assert(y_ptr->shape().back() != -1);
        sizeinfo *= y_ptr->shape().back();
    }

    return sizeinfo;
}

ValuesInfo get_values_info(const ArrayNode* x_ptr, const ArrayNode* y_ptr) {
    // Get all possible combinations of values
    const std::array<double, 4> combos{x_ptr->min() * y_ptr->min(), x_ptr->min() * y_ptr->max(),
                                       x_ptr->max() * y_ptr->min(), x_ptr->max() * y_ptr->max()};

    const double min_val = std::ranges::min(combos);
    const double max_val = std::ranges::max(combos);

    const SizeInfo contracted_axis_size = [&]() {
        // If x is 1d, then the contracted axis size is equal to x's size
        if (x_ptr->ndim() == 1) return x_ptr->sizeinfo();
        // Otherwise it's always the last axis of x (which definitionally is not dynamic)
        return SizeInfo(x_ptr->shape().back());
    }();

    if (contracted_axis_size.max.has_value() and contracted_axis_size.max.value() == 0) {
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

    const ssize_t min_size = contracted_axis_size.min.value_or(0);
    if (max_val < 0) values_info.max = max_val * min_size;
    if (min_val > 0) values_info.min = min_val * min_size;

    return values_info;
}

std::vector<ssize_t> atleast_2d_shape(std::span<const ssize_t> shape, bool vector_as_row) {
    // If vector_as_row is true, treat vector as shape (1, size), else as shape (size, 1)
    if (shape.size() == 0) return {1, 1};
    if (shape.size() == 1 and vector_as_row) return {1, shape[0]};
    if (shape.size() == 1 and not vector_as_row) return {shape[0], 1};
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

ssize_t get_leading_leap(std::span<const ssize_t> shape) {
    assert(shape.size() >= 1);
    if (shape.size() == 1) return 0;  // handles broadcasting for the vector case
    if (shape.size() == 2) return 1;
    return Array::shape_to_size(shape.subspan(shape.size() - 2));
}

ssize_t get_leading_subspace_size(std::span<const ssize_t> x_shape,
                                  std::span<const ssize_t> y_shape) {
    const auto shape = x_shape.size() > y_shape.size() ? x_shape : y_shape;
    const ssize_t penultimate_axis = std::max<ssize_t>(0, static_cast<ssize_t>(shape.size()) - 2);
    return std::reduce(shape.begin(), shape.begin() + penultimate_axis, 1,
                       std::multiplies<ssize_t>());
}

void MatrixMultiplyNode::matmul_(State& state, std::span<double> out,
                                 std::span<const ssize_t> out_shape) const {
    assert(static_cast<ssize_t>(out.size()) == Array::shape_to_size(out_shape));

    // If out is empty (possible when predecessors have 0 size) there is nothing to do
    if (out.size() == 0) return;

    // Start off by getting the information we need to handle the broadcasting. That is
    // to handle x/y with more than 2 dimensions. We do all that handling in terms of
    // "leaps", i.e. the number of pointer increments to apply in each dimension.

    // const ssize_t x_penultimate_axis_size = get_axis_size(x_ptr_->shape(state), -2, true);
    const ssize_t leading_subspace_size =
            get_leading_subspace_size(x_ptr_->shape(state), y_ptr_->shape(state));

    const ssize_t x_leading_leap = get_leading_leap(x_ptr_->shape(state));
    const ssize_t y_leading_leap = get_leading_leap(y_ptr_->shape(state));
    const ssize_t out_leading_leap = [&]() -> ssize_t {
        if (x_ptr_->ndim() >= 2 and y_ptr_->ndim() >= 2) return get_leading_leap(out_shape);
        if (x_ptr_->ndim() == 1 and y_ptr_->ndim() == 1) return 0;
        return out_shape.back();
    }();

    // Now we need information about the matrix multiplication(s). Specifically we need m/n/k for
    //   (m,k) @ (k,n) -> (m,n)

    auto get = [](std::span<const ssize_t> vals, ssize_t index) -> ssize_t {
        // handles negative indexing
        assert(index < 0);
        assert(-static_cast<ssize_t>(vals.size()) <= index);
        return vals[static_cast<ssize_t>(vals.size()) + index];
    };

    const ssize_t m = (x_ptr_->ndim() > 1) ? get(x_ptr_->shape(state), -2) : 1;
    const ssize_t n = (y_ptr_->ndim() > 1) ? get(y_ptr_->shape(state), -1) : 1;
    const ssize_t k = get(x_ptr_->shape(state), -1);

    // We also need the stride information about x/y/out

    std::array<ssize_t, 2> x_matmul_strides = [&get, &k](std::span<const ssize_t> strides) {
        if (strides.size() == 1) {
            return std::array<ssize_t, 2>{k * static_cast<ssize_t>(sizeof(double)), strides[0]};
        }
        return std::array<ssize_t, 2>{get(strides, -2), get(strides, -1)};
    }(x_ptr_->strides());

    std::array<ssize_t, 2> y_matmul_strides = [&get](std::span<const ssize_t> strides) {
        if (strides.size() == 1) {
            return std::array<ssize_t, 2>{strides[0], sizeof(double)};
        }
        return std::array<ssize_t, 2>{get(strides, -2), get(strides, -1)};
    }(y_ptr_->strides());

    std::array<ssize_t, 2> out_matmul_strides{n * static_cast<ssize_t>(sizeof(double)),
                                              sizeof(double)};

    for (ssize_t w = 0; w < leading_subspace_size; w++) {
        // In order to avoid having to iterate over all leading dimensions
        // and checking the strides of both x/y, we use ArrayIterators to
        // get us to the correct subspace, and then use the strides of the
        // predecessors to iterate through the last one or two dimensions.
        const double* const x_data = &x_ptr_->view(state).begin()[w * x_leading_leap];
        const double* const y_data = &y_ptr_->view(state).begin()[w * y_leading_leap];
        double* const out_data = out.data() + w * out_leading_leap;

        gemm(m, n, k,                   //
             x_data, x_matmul_strides,  //
             y_data, y_matmul_strides,  //
             out_data, out_matmul_strides);
    }
}

void MatrixMultiplyNode::initialize_state(State& state) const {
    ssize_t start_size = this->size();
    std::vector<ssize_t> shape(this->shape().begin(), this->shape().end());
    if (this->dynamic()) {
        shape[0] = x_ptr_->shape(state)[0];
        start_size = Array::shape_to_size(shape);
    }

    std::vector<double> data(start_size);
    matmul_(state, data, shape);
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

void MatrixMultiplyNode::update_shape_(State& state) const {
    if (this->dynamic()) {
        data_ptr<MatrixMultiplyNodeData>(state)->shape[0] = x_ptr_->shape(state)[0];
    }
}

void MatrixMultiplyNode::propagate(State& state) const {
    if (x_ptr_->diff(state).size() == 0 and y_ptr_->diff(state).size() == 0) return;

    auto data = data_ptr<MatrixMultiplyNodeData>(state);

    this->update_shape_(state);
    const ssize_t new_size = Array::shape_to_size(data->shape);

    data->output.resize(new_size);

    this->matmul_(state, data->output, data->shape);
    data->assign(data->output);
}

void MatrixMultiplyNode::revert(State& state) const {
    auto data = data_ptr<MatrixMultiplyNodeData>(state);
    data->revert();
    this->update_shape_(state);
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

#ifdef HAS_BLAS_
std::string MatrixMultiplyNode::implementation = "blas";
#else
std::string MatrixMultiplyNode::implementation = "fallback";
#endif

}  // namespace dwave::optimization

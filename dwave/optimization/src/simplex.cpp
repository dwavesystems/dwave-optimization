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

#include "simplex.hpp"

#include <algorithm>
#include <string>

namespace dwave::optimization {

class Matrix {
 public:
    Matrix(ssize_t n, ssize_t m) : n_(n), m_(m) { buffer_.resize(n_ * m_); }

    Matrix(std::vector<double>&& data, ssize_t n, ssize_t m)
            : n_(n), m_(m), buffer_(std::move(data)) {
        assert(n_ * m_ == static_cast<ssize_t>(buffer_.size()) &&
               "shape does not match provided data");
    }

    Matrix(std::span<const double> data, ssize_t n, ssize_t m)
            : Matrix(std::vector<double>(data.begin(), data.end()), n, m) {}

    double& operator()(ssize_t i, ssize_t j) {
        if (i < 0) i += n_;
        if (j < 0) j += m_;
        assert(i >= 0 && i < n_);
        assert(j >= 0 && j < m_);
        return buffer_[i * m_ + j];
    }

    const double& operator()(ssize_t i, ssize_t j) const {
        if (i < 0) i += n_;
        if (j < 0) j += m_;
        assert(i >= 0 && i < n_);
        assert(j >= 0 && j < m_);
        return buffer_[i * m_ + j];
    }

    ssize_t n() const { return n_; }

    ssize_t m() const { return m_; }

 private:
    ssize_t n_;
    ssize_t m_;

    std::vector<double> buffer_;
};

/// Find the pivot column. Will return -1 if no candidate column is found.
ssize_t _pivot_col(Matrix& T, double tolerance, bool bland) {
    double min_val = std::numeric_limits<double>::infinity();
    ssize_t col = -1;
    for (ssize_t j = 0; j < T.m() - 1; j++) {
        double val = T(-1, j);
        if (val < -tolerance) {
            // Bland's rule: return the first index with negative value
            if (bland) return j;

            if (val < min_val) {
                min_val = val;
                col = j;
            }
        }
    }
    return col;
}

/// Find the pivot row. Will return -1 if no candidate row is found.
ssize_t _pivot_row(Matrix& T, const std::vector<ssize_t>& basis, ssize_t pivcol, ssize_t phase,
                   double tolerance, bool bland) {
    ssize_t k = phase == 1 ? 2 : 1;

    double min_q = std::numeric_limits<double>::infinity();
    std::vector<ssize_t> min_rows;
    for (ssize_t i = 0; i < T.n() - k; i++) {
        if (T(i, pivcol) <= tolerance) continue;

        double q = T(i, -1) / T(i, pivcol);
        if (q < min_q) {
            min_rows.clear();
            min_rows.push_back(i);
            min_q = q;
        } else if (q == min_q) {
            min_rows.push_back(i);
        }
    }

    if (min_rows.size() == 0) return -1;

    if (!bland) return min_rows[0];

    // Bland's rule
    ssize_t min_basis = std::numeric_limits<ssize_t>::max();
    ssize_t min_row = -1;
    for (const ssize_t& row : min_rows) {
        if (basis[row] < min_basis) {
            min_basis = basis[row];
            min_row = row;
        }
    }
    return min_row;
}

void _apply_pivot(Matrix& T, std::vector<ssize_t>& basis, ssize_t pivrow, ssize_t pivcol,
                  double tolerance) {
    basis[pivrow] = pivcol;

    double pivval = T(pivrow, pivcol);

    for (ssize_t j = 0; j < T.m(); j++) {
        T(pivrow, j) /= pivval;
    }

    for (ssize_t irow = 0; irow < T.n(); irow++) {
        if (irow == pivrow) continue;

        double val = T(irow, pivcol);
        for (ssize_t j = 0; j < T.m(); j++) {
            T(irow, j) -= T(pivrow, j) * val;
        }
    }
}

SolveResult _solve_simplex(Matrix& T, ssize_t n, std::vector<ssize_t>& basis,
                           ssize_t max_iterations, double tolerance, ssize_t phase, bool bland,
                           ssize_t initial_num_iterations) {
    SolveResult status(SolveResult::SolveStatus::UNSET, initial_num_iterations);

    ssize_t m = phase == 1 ? T.m() - 2 : T.m() - 1;

    if (phase == 2) {
        for (ssize_t pivrow = 0; pivrow < static_cast<ssize_t>(basis.size()); pivrow++) {
            if (basis[pivrow] <= T.m() - 2) continue;

            for (ssize_t col = 0; col < T.m() - 1; col++) {
                if (std::abs(T(pivrow, col)) > tolerance) {
                    _apply_pivot(T, basis, pivrow, col, tolerance);
                    status.num_iterations++;
                    break;
                }
            }
        }
    }

    std::vector<double> solution;
    ssize_t basis_size = basis.size();
    if (std::min(basis_size, m) == 0) {
        assert(T.m() - 1 >= 0);
        solution.resize(T.m() - 1);
    } else {
        ssize_t max_basis =
                *std::max_element(basis.begin(), basis.begin() + std::min(m, basis_size));
        ssize_t size = std::max(T.m() - 1, max_basis + 1);
        assert(size >= 0);
        solution.resize(size);
    }

    while (status.solve_status == SolveResult::SolveStatus::UNSET) {
        ssize_t pivcol = _pivot_col(T, tolerance, bland);
        ssize_t pivrow = -1;
        if (pivcol < 0) {
            status.solve_status = SolveResult::SolveStatus::SUCCESS;
        } else {
            pivrow = _pivot_row(T, basis, pivcol, phase, tolerance, bland);
            if (pivrow < 0) {
                status.solve_status = SolveResult::SolveStatus::FAILURE_UNBOUNDED;
            }
        }

        if (status.solve_status == SolveResult::SolveStatus::UNSET) {
            if (status.num_iterations >= max_iterations) {
                status.solve_status = SolveResult::SolveStatus::HIT_ITERATION_LIMIT;
            } else {
                assert(pivrow >= 0);
                assert(pivcol >= 0);
                _apply_pivot(T, basis, pivrow, pivcol, tolerance);
                status.num_iterations += 1;
            }
        }
    }
    return status;
}

Matrix construct_T(std::span<const double> c, double c0, const Matrix& A,
                   std::span<const double> b) {
    Matrix T(A.n() + 2, A.m() + A.n() + 1);
    // Copy in A to T[:n, :m] and b.T to T[:, -1]
    for (ssize_t i = 0; i < A.n(); i++) {
        double sign = b[i] < 0 ? -1.0 : 1.0;
        T(i, -1) = sign * b[i];
        for (ssize_t j = 0; j < A.m(); j++) {
            T(i, j) = sign * A(i, j);
        }
    }

    // T[:, m:m+n] = I
    for (ssize_t i = 0; i < A.n(); i++) {
        T(i, A.m() + i) = 1;
    }

    // Row objective
    for (ssize_t i = 0; i < A.m(); i++) {
        T(A.n(), i) = c[i];
    }
    T(A.n(), -1) = c0;

    // Row pseudo objective
    for (ssize_t j = 0; j < A.m(); j++) {
        for (ssize_t i = 0; i < A.n(); i++) {
            T(A.n() + 1, j) -= T(i, j);
        }
    }
    for (ssize_t i = 0; i < A.n(); i++) {
        T(-1, -1) -= T(i, -1);
    }

    return T;
}

SolveResult _linprog_simplex(std::span<const double> c, double c0, const Matrix& A,
                             std::span<const double> b, ssize_t max_iterations, double tolerance,
                             bool bland) {
    assert(static_cast<ssize_t>(c.size()) == A.m());

    std::vector<ssize_t> basis;
    for (ssize_t i = 0; i < A.n(); i++) {
        basis.push_back(i + A.m());
    }

    Matrix T = construct_T(c, c0, A, b);

    ssize_t phase = 1;
    SolveResult status =
            _solve_simplex(T, A.n(), basis, max_iterations, tolerance, phase, bland, 0);

    if (std::abs(T(-1, -1)) < tolerance) {
        Matrix newT(A.n() + 1, A.m() + 1);

        // newT[:, :m] = T[:-1, :m]
        for (ssize_t i = 0; i < newT.n(); i++) {
            for (ssize_t j = 0; j < A.m(); j++) {
                newT(i, j) = T(i, j);
            }
        }
        // newT[:, -1] = T[:, -1]
        for (ssize_t i = 0; i < newT.n(); i++) {
            newT(i, -1) = T(i, -1);
        }

        std::swap(T, newT);
    } else {
        status.solve_status = SolveResult::SolveStatus::FAILURE_NO_FEASIBLE_START;
    }

    if (status.solve_status == SolveResult::SolveStatus::SUCCESS) {
        phase = 2;
        assert(T.n() == A.n() + 1 && T.m() == A.m() + 1);
        SolveResult status2 = _solve_simplex(T, A.n(), basis, max_iterations, tolerance, phase,
                                             bland, status.num_iterations);

        std::swap(status, status2);
    }

    SolveResult result(status);

    std::vector<double> solution(A.m(), 0.0);
    for (ssize_t i = 0; i < A.n(); i++) {
        if (basis[i] < A.m()) {
            solution[basis[i]] = T(i, -1);
        }
    }
    result.set_partial_solution(std::move(solution));

    return result;
}

struct LP {
    LP(std::vector<double>&& c, double c0, Matrix&& A, std::vector<double>&& b)
            : c(c), c0(c0), A(A), b(b){};

    std::vector<double> c;
    double c0;
    Matrix A;
    std::vector<double> b;
};

void check_LP_sizes(std::span<const double> c, std::span<const double> b_lb,
                    std::span<const double> A_data, std::span<const double> b_ub,
                    std::span<const double> A_eq_data, std::span<const double> b_eq,
                    std::span<const double> lb, std::span<const double> ub) {
    [[maybe_unused]] const ssize_t num_vars = c.size();

    assert(b_lb.size() == b_ub.size() && "b_lb length does not match b_ub length");
    assert(num_vars * b_lb.size() == A_data.size() && "A_data wrong size");
    assert(num_vars * b_eq.size() == A_eq_data.size() && "A_eq_data wrong size");
    assert(num_vars == static_cast<ssize_t>(lb.size()) && "lower bounds length does not match c");
    assert(num_vars == static_cast<ssize_t>(ub.size()) && "upper bounds length does not match c");
}

bool lb_is_unbounded(double lb) { return lb <= -LP_INFINITY; }

bool ub_is_unbounded(double ub) { return ub >= LP_INFINITY; }

/// Translate the general LP form to the simple:
///     minimize(c @ x) subject to A @ x == b, x >= 0
LP translate_LP_to_simple(std::span<const double> c, std::span<const double> b_lb,
                          std::span<const double> A_data, std::span<const double> b_ub,
                          std::span<const double> A_eq_data, std::span<const double> b_eq,
                          std::span<const double> lb, std::span<const double> ub) {
    check_LP_sizes(c, b_lb, A_data, b_ub, A_eq_data, b_eq, lb, ub);

    const ssize_t num_vars = c.size();

    const Matrix A(A_data, b_lb.size(), num_vars);
    const Matrix A_eq(A_eq_data, b_eq.size(), num_vars);

    ssize_t A_constraint_count = 0;
    for (ssize_t i = 0; i < A.n(); i++) {
        // Unbounded constraint, can ignore
        if (lb_is_unbounded(b_lb[i]) && ub_is_unbounded(b_ub[i])) continue;

        if (!lb_is_unbounded(b_lb[i])) {
            A_constraint_count++;
        }
        if (!ub_is_unbounded(b_ub[i])) {
            A_constraint_count++;
        }
    }

    std::vector<double> c_(c.begin(), c.end());
    std::vector<double> lb_(lb.begin(), lb.end());
    std::vector<double> ub_(ub.begin(), ub.end());

    ssize_t upper_bounded_var_count = 0;
    ssize_t unbounded_var_count = 0;
    for (ssize_t j = 0; j < num_vars; j++) {
        if (lb_is_unbounded(lb_[j]) && ub_is_unbounded(ub_[j])) {
            unbounded_var_count++;
        }

        if (!ub_is_unbounded(ub_[j])) {
            upper_bounded_var_count++;
        }
    }

    ssize_t slack_var_count = A_constraint_count + upper_bounded_var_count;
    Matrix A_(A_eq.n() + A_constraint_count + upper_bounded_var_count,
              num_vars + unbounded_var_count + slack_var_count);
    c_.resize(A_.m());
    std::vector<double> b(A_.n(), 0);

    ssize_t A_row = 0;

    // Copy A to final A, adding extra flipped constraint for ones with lower bounds
    for (ssize_t i = 0; i < A.n(); i++) {
        if (!lb_is_unbounded(b_lb[i])) {
            // Copy the flipped constraint
            for (ssize_t j = 0; j < A.m(); j++) {
                A_(A_row, j) = -A(i, j);
            }
            // Copy the flipped bound
            b[A_row] = -b_lb[i];
            A_row++;
        }

        if (!ub_is_unbounded(b_ub[i])) {
            // Copy the constraint
            for (ssize_t j = 0; j < A.m(); j++) {
                A_(A_row, j) = A(i, j);
            }
            // Copy the bound
            b[A_row] = b_ub[i];
            A_row++;
        }
    }

    // Copy A_eq and b_eq directly
    for (ssize_t i = 0; i < A_eq.n(); i++) {
        ssize_t A_row = i + A_constraint_count + upper_bounded_var_count;
        for (ssize_t j = 0; j < A_eq.m(); j++) {
            A_(A_row, j) = A_eq(i, j);
        }
        b[A_row] = b_eq[i];
    }

    ssize_t free_variable_index = num_vars;

    A_row = A_constraint_count;
    for (ssize_t j = 0; j < num_vars; j++) {
        if (lb_is_unbounded(lb_[j]) && ub_is_unbounded(ub_[j])) {
            // Free variable, substitute xi = xi+ - xi-
            for (ssize_t i = 0; i < A_.n(); i++) {
                A_(i, free_variable_index) = -A_(i, j);
            }
            c_[free_variable_index] = -c[j];
            free_variable_index++;

        } else if (lb_is_unbounded(lb_[j]) && !ub_is_unbounded(ub_[j])) {
            // Substitute any variables xi that are unbounded below (i.e. -inf <= xi <= C) with -xi
            lb_[j] = -ub_[j];
            ub_[j] = -lb_[j];
            c_[j] = -c_[j];

            for (ssize_t i = 0; i < A_.n(); i++) {
                A_(i, j) *= -1;
            }
        }

        if (!ub_is_unbounded(ub_[j])) {
            A_(A_row, j) = 1;
            b[A_row] = ub_[j];
            A_row++;
        }
    }

    assert(A_row == A_constraint_count + upper_bounded_var_count);
    assert(free_variable_index == num_vars + unbounded_var_count);

    // Add slack variables for inequalities
    for (ssize_t i = 0; i < slack_var_count; i++) {
        A_(i, free_variable_index + i) = 1;
    }

    // Substitute the lower bounds
    double c0 = 0.0;
    for (ssize_t j = 0; j < num_vars; j++) {
        if (!lb_is_unbounded(lb_[j])) {
            c0 += lb_[j] * c[j];

            for (ssize_t i = 0; i < A_.n(); i++) {
                b[i] -= A_(i, j) * lb_[j];
            }
        }
    }

    return LP(std::move(c_), c0, std::move(A_), std::move(b));
}

void SolveResult::_postprocess_solution_variables(std::span<const double> lb,
                                                  std::span<const double> ub) {
    assert(partial_solution_set &&
           "postprocess_solution must be called after partial solution has been set");
    assert(lb.size() == ub.size());

    ssize_t num_vars = lb.size();

    ssize_t free_variable_index = 0;
    for (ssize_t j = 0; j < num_vars; j++) {
        if (lb_is_unbounded(lb[j]) && ub_is_unbounded(ub[j])) {
            assert(num_vars + free_variable_index < static_cast<ssize_t>(solution_.size()));
            solution_[j] -= solution_[num_vars + free_variable_index];
            free_variable_index++;
        } else {
            if (!lb_is_unbounded(lb[j])) {
                solution_[j] += lb[j];
            } else if (!ub_is_unbounded(ub[j])) {
                solution_[j] = ub[j] - solution_[j];
            }
        }
    }

    solution_.resize(num_vars);
}

void SolveResult::recompute_feasibility(std::span<const double> c, std::span<const double> b_lb,
                                        std::span<const double> A_data,
                                        std::span<const double> b_ub,
                                        std::span<const double> A_eq_data,
                                        std::span<const double> b_eq, std::span<const double> lb,
                                        std::span<const double> ub, double tolerance) {
    assert(solution_.size() == c.size());

    double tol = std::sqrt(tolerance) * 10.0;

    // Check b_lb <= A @ x <= b_ub
    bool feasible = true;
    for (ssize_t constraint = 0, stop = b_lb.size(); constraint < stop; constraint++) {
        auto A_row = A_data.begin() + constraint * c.size();
        double value = std::inner_product(solution_.begin(), solution_.end(), A_row, 0.0);
        if (value < b_lb[constraint] - tol || value > b_ub[constraint] + tol) {
            feasible = false;
            break;
        }
    }

    // Check A_eq @ x == b_eq
    if (feasible) {
        for (ssize_t constraint = 0; constraint < static_cast<ssize_t>(b_eq.size()); constraint++) {
            auto A_row = A_eq_data.begin() + constraint * c.size();
            double value = std::inner_product(solution_.begin(), solution_.end(), A_row, 0.0);
            if (std::abs(b_eq[constraint] - value) > tol) {
                feasible = false;
                break;
            }
        }
    }

    // Check variable bounds
    if (feasible) {
        for (ssize_t v = 0, stop = c.size(); v < stop; v++) {
            if (solution_[v] < lb[v] - tol || solution_[v] > ub[v] + tol) {
                feasible = false;
                break;
            }
        }
    }

    feasible_ = feasible;
    objective_ = std::inner_product(c.begin(), c.end(), solution_.begin(), 0.0);

    if (feasible) {
        if (solve_status == SolveResult::SolveStatus::SUCCESS) {
            solution_status_ = SolveResult::SolutionStatus::OPTIMAL;
        } else {
            solution_status_ = SolveResult::SolutionStatus::FEASIBLE_BUT_NOT_OPTIMAL;
        }
    } else {
        solution_status_ = SolveResult::SolutionStatus::INFEASIBLE;
    }
}

void SolveResult::set_solution(std::vector<double>&& solution, std::span<const double> c,
                               std::span<const double> b_lb, std::span<const double> A_data,
                               std::span<const double> b_ub, std::span<const double> A_eq_data,
                               std::span<const double> b_eq, std::span<const double> lb,
                               std::span<const double> ub, double tolerance) {
    // check shape of solution relative to the other values.
    // we do not check the LP values amoung themselves for consistency
    if (solution.size() != c.size()) {
        throw std::invalid_argument("expected a solution of size " + std::to_string(c.size()) +
                                    ", given a solution of size " +
                                    std::to_string(solution.size()));
    }

    // In the case that we're currently holding an optimal solution, we could
    // check whether the new solution has the same energy (up to tolerance) and
    // in that case mark ourselves as optimal. For now though let's just reset
    // everything
    {
        SolveResult tmp;
        std::swap(*this, tmp);
    }

    solution_ = std::move(solution);
    recompute_feasibility(c, b_lb, A_data, b_ub, A_eq_data, b_eq, lb, ub, tolerance);
    final_solution_set = true;
}

void SolveResult::postprocess_solution(std::span<const double> c, std::span<const double> b_lb,
                                       std::span<const double> A_data, std::span<const double> b_ub,
                                       std::span<const double> A_eq_data,
                                       std::span<const double> b_eq, std::span<const double> lb,
                                       std::span<const double> ub, double tolerance) {
    _postprocess_solution_variables(lb, ub);
    recompute_feasibility(c, b_lb, A_data, b_ub, A_eq_data, b_eq, lb, ub, tolerance);
    final_solution_set = true;
}

/// Solve a linear program using the Simplex method.
SolveResult linprog(std::span<const double> c, std::span<const double> b_lb,
                    std::span<const double> A_data, std::span<const double> b_ub,
                    std::span<const double> A_eq_data, std::span<const double> b_eq,
                    std::span<const double> lb, std::span<const double> ub, double tolerance) {
    const LP model = translate_LP_to_simple(c, b_lb, A_data, b_ub, A_eq_data, b_eq, lb, ub);

    ssize_t max_iterations = 1000;
    bool bland = false;

    SolveResult result =
            _linprog_simplex(model.c, model.c0, model.A, model.b, max_iterations, tolerance, bland);

    result.postprocess_solution(c, b_lb, A_data, b_ub, A_eq_data, b_eq, lb, ub, tolerance);

    return result;
}

}  // namespace dwave::optimization

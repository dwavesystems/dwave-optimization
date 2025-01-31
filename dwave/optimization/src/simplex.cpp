// Copyright 2025 D-Wave Systems Inc.
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

namespace dwave::optimization {

template <class T>
void print_vec(std::string name, const std::vector<T>& vec) {
    std::cout << name << "(" << vec.size() << "): ";
    for (const T& v : vec) std::cout << v << " ";
    std::cout << std::endl;
}

class Matrix {
 public:
    Matrix(ssize_t n, ssize_t m) : n_(n), m_(m) { buffer_.resize(n_ * m_); }

    Matrix(std::vector<double>&& data, ssize_t n, ssize_t m)
            : n_(n), m_(m), buffer_(std::move(data)) {
        if (n_ * m_ != static_cast<ssize_t>(buffer_.size())) {
            throw std::invalid_argument("shape does not match provided data");
        }
    }

    Matrix(const std::vector<double>& data, ssize_t n, ssize_t m)
            : Matrix(std::vector<double>(data), n, m) {}

    double& at(ssize_t i, ssize_t j) {
        if (i < 0) i += n_;
        if (j < 0) j += m_;
        assert(i >= 0 && i < n_);
        assert(j >= 0 && j < m_);
        return buffer_[i * m_ + j];
    }

    const double& at(ssize_t i, ssize_t j) const {
        if (i < 0) i += n_;
        if (j < 0) j += m_;
        assert(i >= 0 && i < n_);
        assert(j >= 0 && j < m_);
        return buffer_[i * m_ + j];
    }

    void print(std::string name) const {
        std::cout << name << ": (" << n_ << ", " << m_ << ")" << std::endl;
        for (ssize_t i = 0; i < n_; i++) {
            for (ssize_t j = 0; j < m_; j++) {
                std::cout << at(i, j) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    ssize_t n() const { return n_; }

    ssize_t m() const { return m_; }

 private:
    ssize_t n_;
    ssize_t m_;

    std::vector<double> buffer_;
};

ssize_t _pivot_col(Matrix& T, double tol, bool bland) {
    double min_val = std::numeric_limits<double>::infinity();
    ssize_t col = -1;
    for (ssize_t j = 0; j < T.m() - 1; j++) {
        double val = T.at(-1, j);
        if (val < -tol) {
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

ssize_t _pivot_row(Matrix& T, const std::vector<ssize_t>& basis, ssize_t pivcol, ssize_t phase,
                   double tol, bool bland) {
    ssize_t k = phase == 1 ? 2 : 1;

    double min_q = std::numeric_limits<double>::infinity();
    std::vector<ssize_t> min_rows;
    for (ssize_t i = 0; i < T.n() - k; i++) {
        if (T.at(i, pivcol) <= tol) continue;

        double q = T.at(i, -1) / T.at(i, pivcol);
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
                  double tol) {
    basis[pivrow] = pivcol;

    double pivval = T.at(pivrow, pivcol);

    for (ssize_t j = 0; j < T.m(); j++) {
        T.at(pivrow, j) /= pivval;
    }

    for (ssize_t irow = 0; irow < T.n(); irow++) {
        if (irow == pivrow) continue;

        double val = T.at(irow, pivcol);
        for (ssize_t j = 0; j < T.m(); j++) {
            T.at(irow, j) -= T.at(pivrow, j) * val;
        }
    }
}

SolverStatus _solve_simplex(Matrix& T, ssize_t n, std::vector<ssize_t>& basis, ssize_t maxiter,
                            double tol, ssize_t phase, bool bland, ssize_t nit0) {
    SolverStatus status(0, solve_status_t::UNSET);

    ssize_t m = phase == 1 ? T.m() - 2 : T.m() - 1;

    if (phase == 2) {
        for (ssize_t pivrow = 0; pivrow < static_cast<ssize_t>(basis.size()); pivrow++) {
            if (basis[pivrow] <= T.m() - 2) continue;

            for (ssize_t col = 0; col < T.m() - 1; col++) {
                if (std::abs(T.at(pivrow, col)) > tol) {
                    _apply_pivot(T, basis, pivrow, col, tol);
                    status.nit++;
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

    while (status.status == solve_status_t::UNSET) {
        ssize_t pivcol = _pivot_col(T, tol, bland);
        ssize_t pivrow = -1;
        if (pivcol < 0) {
            status.status = solve_status_t::SUCCESS;
        } else {
            pivrow = _pivot_row(T, basis, pivcol, phase, tol, bland);
            if (pivrow < 0) {
                status.status = solve_status_t::FAILURE_UNBOUNDED;
            }
        }

        if (status.status == solve_status_t::UNSET) {
            if (status.nit >= maxiter) {
                status.status = solve_status_t::HIT_ITERATION_LIMIT;
            } else {
                assert(pivrow >= 0);
                assert(pivcol >= 0);
                _apply_pivot(T, basis, pivrow, pivcol, tol);
                status.nit += 1;
            }
        }
    }
    return status;
}

SolveResult _linprog_simplex(const std::vector<double>& c, double c0, const Matrix& A,
                             const std::vector<double>& b, ssize_t maxiter, double tol,
                             bool bland) {
    assert(static_cast<ssize_t>(c.size()) == A.m());

    std::vector<ssize_t> basis;
    for (ssize_t i = 0; i < A.n(); i++) {
        basis.push_back(i + A.m());
    }

    Matrix T(A.n() + 2, A.m() + A.n() + 1);
    // Copy in A to T[:n, :m]
    for (ssize_t i = 0; i < A.n(); i++) {
        for (ssize_t j = 0; j < A.m(); j++) {
            T.at(i, j) = A.at(i, j);
        }
    }
    // T[:, m:m+n] = I
    for (ssize_t i = 0; i < A.n(); i++) {
        T.at(i, A.m() + i) = 1;
    }
    // Copy b.T to A[:, m + n]
    for (ssize_t i = 0; i < A.n(); i++) {
        T.at(i, A.m() + A.n()) = b[i];
    }

    // Row objective
    for (ssize_t i = 0; i < A.m(); i++) {
        T.at(A.n(), i) = c[i];
    }
    T.at(A.n(), T.m() - 1) = c0;

    // Row pseudo objective
    for (ssize_t j = 0; j < A.m(); j++) {
        for (ssize_t i = 0; i < A.n(); i++) {
            T.at(A.n() + 1, j) -= T.at(i, j);
        }
    }
    for (ssize_t i = 0; i < A.n(); i++) {
        T.at(A.n() + 1, T.m() - 1) -= T.at(i, T.m() - 1);
    }

    ssize_t phase = 1;
    SolverStatus status = _solve_simplex(T, A.n(), basis, maxiter, tol, phase, bland, 0);

    if (std::abs(T.at(-1, -1)) < tol) {
        Matrix newT(T.n() - 1, A.m() + 1);

        // newT[:, :m] = T[:-1, :m]
        for (ssize_t i = 0; i < newT.n(); i++) {
            for (ssize_t j = 0; j < A.m(); j++) {
                newT.at(i, j) = T.at(i, j);
            }
        }
        // newT[:, -1] = T[:, -1]
        for (ssize_t i = 0; i < newT.n(); i++) {
            newT.at(i, -1) = T.at(i, -1);
        }

        std::swap(T, newT);
    } else {
        status.status = solve_status_t::FAILURE_NO_FEASIBLE_START;
    }

    if (status.status == solve_status_t::SUCCESS) {
        phase = 2;
        SolverStatus status2 =
                _solve_simplex(T, A.n(), basis, maxiter, tol, phase, bland, status.nit);

        std::swap(status, status2);
    }

    SolveResult result(status);

    result.solution.resize(A.n() + A.m(), 0);
    for (ssize_t i = 0; i < A.n(); i++) {
        result.solution[basis[i]] = T.at(i, -1);
    }

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

void check_LP_sizes(const std::vector<double>& c, const std::vector<double>& b_lb,
                    const std::vector<double>& A_data, const std::vector<double>& b_ub,
                    const std::vector<double>& A_eq_data, const std::vector<double>& b_eq,
                    const std::vector<double>& lb, const std::vector<double>& ub) {
    const ssize_t num_vars = c.size();

    if (b_lb.size() != b_ub.size()) {
        throw std::invalid_argument("b_lb length does not match b_ub length");
    }
    if (num_vars * b_lb.size() != A_data.size()) {
        throw std::invalid_argument("A_data wrong size");
    }
    if (num_vars * b_eq.size() != A_eq_data.size()) {
        throw std::invalid_argument("A_eq_data wrong size");
    }
    if (num_vars != static_cast<ssize_t>(lb.size())) {
        throw std::invalid_argument("lower bounds length does not match c");
    }
    if (num_vars != static_cast<ssize_t>(ub.size())) {
        throw std::invalid_argument("upper bounds length does not match c");
    }
}

LP translate_LP_to_simple(const std::vector<double>& c, const std::vector<double>& b_lb,
                          const std::vector<double>& A_data, const std::vector<double>& b_ub,
                          const std::vector<double>& A_eq_data, const std::vector<double>& b_eq,
                          const std::vector<double>& lb, const std::vector<double>& ub) {
    check_LP_sizes(c, b_lb, A_data, b_ub, A_eq_data, b_eq, lb, ub);

    const ssize_t num_vars = c.size();

    const Matrix A(A_data, b_lb.size(), num_vars);
    const Matrix A_eq(A_eq_data, b_eq.size(), num_vars);

    static double inf = std::numeric_limits<double>::infinity();

    ssize_t lower_bounded_constraint_count = 0;
    ssize_t upper_bounded_constraint_count = 0;
    for (ssize_t i = 0; i < A.n(); i++) {
        // Unbounded constraint, can ignore
        if (b_lb[i] == -inf && b_ub[i] == inf) continue;

        if (b_lb[i] != -inf) {
            lower_bounded_constraint_count++;
        }
        if (b_ub[i] != inf) {
            upper_bounded_constraint_count++;
        }
    }

    std::vector<double> c_(c);
    std::vector<double> lb_(lb);
    std::vector<double> ub_(ub);

    ssize_t upper_bounded_var_count = 0;
    ssize_t unbounded_var_count = 0;
    for (ssize_t j = 0; j < num_vars; j++) {
        if (lb_[j] == -inf && ub_[j] == inf) {
            unbounded_var_count++;
        } else if (ub_[j] != inf) {
            upper_bounded_var_count++;
        }
    }

    ssize_t slack_var_count = lower_bounded_constraint_count + upper_bounded_constraint_count +
                              upper_bounded_var_count;
    Matrix A_(A_eq.n() + upper_bounded_constraint_count + upper_bounded_var_count,
              num_vars + unbounded_var_count + slack_var_count);
    c_.resize(A_.m());
    std::vector<double> b(A_.n(), 0);

    ssize_t A_row = 0;

    // Copy A_eq and bound directly
    for (; A_row < A_eq.n(); A_row++) {
        for (ssize_t j = 0; j < A_eq.m(); j++) {
            auto& val = A_eq.at(A_row, j);
            A_.at(A_row, j) = val;
        }
        b[A_row] = b_eq[A_row];
    }

    // Copy A to final A, adding extra flipped constraint for ones with lower bounds
    for (ssize_t i = 0; i < A.n(); i++) {
        if (b_lb[i] != -inf) {
            // Copy the flipped constraint
            for (ssize_t j = 0; j < A.m(); j++) {
                A_.at(A_row, j) = -A.at(i, j);
            }
            // Copy the flipped bound
            b[A_row] = -b_lb[i];
            A_row++;
        }
        if (b_ub[i] != inf) {
            // Copy the constraint
            for (ssize_t j = 0; j < A.m(); j++) {
                A_.at(A_row, j) = A.at(i, j);
            }
            // Copy the bound
            b[A_row] = b_ub[i];
            A_row++;
        }
    }

    ssize_t free_variable_index = num_vars;

    for (ssize_t j = 0; j < num_vars; j++) {
        if (lb_[j] == -inf && ub_[j] == inf) {
            // Free variable, substitue xi = xi+ - xi-
            for (ssize_t i = 0; i < A_.n(); i++) {
                A_.at(i, free_variable_index) = -A_.at(i, j);
            }
            c_[free_variable_index] = -c[j];
            free_variable_index++;

        } else if (lb_[j] == -inf && ub_[j] != inf) {
            // Substitute any variables xi that are unbounded below (i.e. -inf <= xi <= C) with -xi
            lb_[j] = -ub_[j];
            ub_[j] = -lb_[j];
            c_[j] = -c_[j];

            for (ssize_t i = 0; i < A_.n(); i++) {
                A_.at(i, j) *= -1;
            }
        }

        if (ub_[j] != inf) {
            A_.at(A_row, j) = 1;
            b[A_row] = ub_[j];
            A_row++;
        }
    }

    assert(A_row == A_.n());
    assert(free_variable_index == num_vars + unbounded_var_count);

    // Add slack variables for inequalities
    for (ssize_t i = 0; i < slack_var_count; i++) {
        A_.at(A_eq.n() + i, free_variable_index + i) = 1;
    }

    // Substitute the lower bounds
    double c0 = 0.0;
    for (ssize_t j = 0; j < num_vars; j++) {
        if (lb_[j] != -inf) {
            c0 += lb_[j] * c[j];

            for (ssize_t i = 0; i < A_.n(); i++) {
                b[i] -= A_.at(i, j) * lb_[j];
            }
        }
    }

    return LP(std::move(c_), c0, std::move(A_), std::move(b));
}

void post_process_solution(ssize_t num_vars, std::vector<double>& solution,
                           const std::vector<double>& lb, const std::vector<double>& ub) {
    static double inf = std::numeric_limits<double>::infinity();
    for (ssize_t j = 0; j < num_vars; j++) {
        if (lb[j] != -inf) {
            solution[j] += lb[j];
        }

        if (lb[j] == -inf && ub[j] == inf) {
            solution[j] -= solution[j + num_vars];
        }
    }

    solution.resize(num_vars);
}

SolveResult linprog(const std::vector<double>& c, const std::vector<double>& b_lb,
                    const std::vector<double>& A_data, const std::vector<double>& b_ub,
                    const std::vector<double>& A_eq_data, const std::vector<double>& b_eq,
                    const std::vector<double>& lb, const std::vector<double>& ub) {
    const LP model = translate_LP_to_simple(c, b_lb, A_data, b_ub, A_eq_data, b_eq, lb, ub);

    ssize_t maxiter = 1000;
    double tol = 1e-9;
    bool bland = false;

    SolveResult result = _linprog_simplex(model.c, model.c0, model.A, model.b, maxiter, tol, bland);

    post_process_solution(c.size(), result.solution, lb, ub);

    return result;
}

}  // namespace dwave::optimization

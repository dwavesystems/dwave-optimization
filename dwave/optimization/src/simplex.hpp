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

#include <cassert>
#include <cmath>
#include <span>
#include <utility>
#include <vector>

#include "dwave-optimization/utils.hpp"

namespace dwave::optimization {

/// Any upper bound equal to or greater (or lower bound less to or equal)
/// to this will be treated as unbounded.
static constexpr double LP_INFINITY = 1e30;

class SolveResult {
 public:
    enum SolveStatus {
        UNSET,
        SUCCESS,
        HIT_ITERATION_LIMIT,
        FAILURE_NO_FEASIBLE_START,
        FAILURE_UNBOUNDED,
        FAILURE_SINGULAR_MATRIX,
    };

    enum SolutionStatus {
        SOLUTION_UNSET,
        OPTIMAL,
        INFEASIBLE,
        FEASIBLE_BUT_NOT_OPTIMAL,
    };

    SolveResult() : solve_status(SolveStatus::UNSET), num_iterations(0) {}
    SolveResult(SolveStatus solve_status, ssize_t num_iterations)
            : solve_status(solve_status), num_iterations(num_iterations) {}

    void set_partial_solution(std::vector<double>&& partial_solution) {
        solution_ = std::move(partial_solution);
        partial_solution_set = true;
    }

    // Set the solution explicitly.
    // Throws an error if the ``solution`` is not the same size as ``c``.
    // Does not check the LP parameters for consistency.
    void set_solution(std::vector<double>&& solution, std::span<const double> c,
                      std::span<const double> b_lb, std::span<const double> A_data,
                      std::span<const double> b_ub, std::span<const double> A_eq_data,
                      std::span<const double> b_eq, std::span<const double> lb,
                      std::span<const double> ub, double tolerance = 1e-7);

    void postprocess_solution(std::span<const double> c, std::span<const double> b_lb,
                              std::span<const double> A_data, std::span<const double> b_ub,
                              std::span<const double> A_eq_data, std::span<const double> b_eq,
                              std::span<const double> lb, std::span<const double> ub,
                              double tolerance);

    const std::vector<double>& solution() const {
        check_final_solution_set();
        return solution_;
    }

    SolutionStatus solution_status() const {
        check_final_solution_set();
        return solution_status_;
    }

    double objective() const {
        check_final_solution_set();
        return objective_;
    }

    bool feasible() const {
        check_final_solution_set();
        return feasible_;
    }

    SolveStatus solve_status;
    ssize_t num_iterations;

 private:
    void check_final_solution_set() const {
        assert(final_solution_set && "solution has not yet been set");
    }

    void _postprocess_solution_variables(std::span<const double> lb, std::span<const double> ub);

    void recompute_feasibility(std::span<const double> c, std::span<const double> b_lb,
                               std::span<const double> A_data, std::span<const double> b_ub,
                               std::span<const double> A_eq_data, std::span<const double> b_eq,
                               std::span<const double> lb, std::span<const double> ub,
                               double tolerance);

    SolutionStatus solution_status_ = SolutionStatus::SOLUTION_UNSET;

    bool partial_solution_set = false;
    bool final_solution_set = false;

    std::vector<double> solution_;
    double objective_ = NAN;
    bool feasible_ = false;
};

/// Solve the linear program defined by
///
/// mininize c @ x
/// subject to
///     b_lb <= A @ x <= b_ub
///     A_eq @ x == b_eq
///     lb <= x <= ub
///
/// where x is a length N vector of variables, c is length N, A is an MxN matrix,
/// b_lb and b_ub are length M vectors, A_eq is a KxN matrix, b_eq is a length K
/// vector, and lb and ub are length N vectors.
///
/// This uses a very basic and slow simplex method implementation, and may not
/// find optimality on many problems.
///
/// TODO: try again with Bland's rule?
/// TODO: expose parameters or put them in a better place?
///
/// @param c Length N vector representing the coefficients of the objective.
/// @param b_lb Length M vector, lower bounds on A @ x.
/// @param A_data The values of the A matrix (total size M * N).
/// @param b_ub Length M vector, lower bounds on A @ x.
/// @param A_eq_data The values of the A_eq matrix (total size K * N).
/// @param b_ub Length K vector, bounds on A_eq @ x.
/// @param lb Length N vector, lower bounds on the variables.
/// @param ub Length N vector, upper bounds on the variables.
/// @param tolerance The absolute tolerance allowed for bounds.
SolveResult linprog(std::span<const double> c, std::span<const double> b_lb,
                    std::span<const double> A_data, std::span<const double> b_ub,
                    std::span<const double> A_eq_data, std::span<const double> b_eq,
                    std::span<const double> lb, std::span<const double> ub,
                    double tolerance = 1e-7);

}  // namespace dwave::optimization

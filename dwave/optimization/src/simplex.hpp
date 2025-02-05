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
#include <vector>

#include "dwave-optimization/utils.hpp"

namespace dwave::optimization {

struct SolveResult {
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

    SolveStatus solve_status;
    ssize_t num_iterations;

    SolutionStatus solution_status = SolutionStatus::SOLUTION_UNSET;
    std::vector<double> solution;
    double objective = NAN;
    bool feasible = false;
};

SolveResult linprog(std::span<const double> c, std::span<const double> b_lb,
                    std::span<const double> A_data, std::span<const double> b_ub,
                    std::span<const double> A_eq_data, std::span<const double> b_eq,
                    std::span<const double> lb, std::span<const double> ub);

}  // namespace dwave::optimization

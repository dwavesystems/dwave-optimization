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
#include <vector>

#include "dwave-optimization/utils.hpp"

namespace dwave::optimization {

enum solve_status_t {
    UNSET,
    SUCCESS,
    HIT_ITERATION_LIMIT,
    FAILURE_NO_FEASIBLE_START,
    FAILURE_UNBOUNDED,
    FAILURE_SINGULAR_MATRIX,
};

struct SolverStatus {
    SolverStatus(ssize_t nit, solve_status_t status) : nit(nit), status(status){};

    ssize_t nit;
    solve_status_t status;
};

struct SolveResult {
    SolveResult(SolverStatus solver_status)
            : status(solver_status.status), nit(solver_status.nit){};

    solve_status_t status;
    ssize_t nit;
    std::vector<double> solution;
};

SolveResult linprog(const std::vector<double>& c, const std::vector<double>& b_lb,
                    const std::vector<double>& A_data, const std::vector<double>& b_ub,
                    const std::vector<double>& A_eq_data, const std::vector<double>& b_eq,
                    const std::vector<double>& lb, const std::vector<double>& ub);

}  // namespace dwave::optimization

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

#include <algorithm>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "catch2/catch_test_macros.hpp"
#include "simplex.hpp"

namespace dwave::optimization {

using Catch::Matchers::WithinRel;

TEST_CASE("LP solver (simplex)", "[simplex]") {
    static const double inf = std::numeric_limits<double>::infinity();

    GIVEN("A simple LP with only upper bounds on A @ x") {
        std::vector<double> c{-2, -3, -4};

        std::vector<double> A{3, 2, 1, 2, 5, 3};

        std::vector<double> b_lb{-inf, -inf};
        std::vector<double> b_ub{10, 15};

        std::vector<double> A_eq;
        std::vector<double> b_eq;
        std::vector<double> lb{0, 0, 0};
        std::vector<double> ub{inf, inf, inf};

        THEN("We find the correct solution") {
            SolveResult result = linprog(c, b_lb, A, b_ub, A_eq, b_eq, lb, ub);
            CHECK(result.solve_status == SolveResult::SolveStatus::SUCCESS);
            CHECK(result.solution_status() == SolveResult::SolutionStatus::OPTIMAL);
            CHECK(result.feasible());
            CHECK(result.objective() == -20);

            CHECK(std::ranges::equal(std::vector{0, 0, 5}, result.solution()));
        }
    }

    GIVEN("Simple LP with unbounded variables") {
        std::vector<double> c{-5, -10, -3, -4};

        std::vector<double> A{
                1, 1, 1, 1, 0, 5, -3, 0, 1, 2, 3, 4,
        };
        std::vector<double> b_lb{-inf, -inf, -inf};
        std::vector<double> b_ub{100, 50, 20};

        std::vector<double> A_eq{
                3,
                0,
                0,
                -1,
        };
        std::vector<double> b_eq{7};

        std::vector<double> lb{-inf, -5, -50, -3};
        std::vector<double> ub{inf, 500, 50, 7};

        THEN("We find the correct solution") {
            SolveResult result = linprog(c, b_lb, A, b_ub, A_eq, b_eq, lb, ub);
            CHECK(result.solve_status == SolveResult::SolveStatus::SUCCESS);
            CHECK(result.solution_status() == SolveResult::SolutionStatus::OPTIMAL);
            CHECK(result.feasible());
            CHECK_THAT(result.objective(), WithinRel(-117.52380952380958, 1e-9));

            CHECK_THAT(result.solution()[0], WithinRel(1.333333333333333, 1e-9));
            CHECK_THAT(result.solution()[1], WithinRel(11.523809523809526, 1e-9));
            CHECK_THAT(result.solution()[2], WithinRel(2.539682539682545, 1e-9));
            CHECK_THAT(result.solution()[3], WithinRel(-3.0, 1e-9));
        }
    }

    GIVEN("Simple LP with unbounded variables, negative x in solution") {
        std::vector<double> c{-5, -10, -3};

        std::vector<double> A{
                1,
                1,
                0,
        };
        std::vector<double> b_lb{-inf};
        std::vector<double> b_ub{10};

        std::vector<double> A_eq{
                1,
                0,
                1,
        };
        std::vector<double> b_eq{7};

        std::vector<double> lb{-inf, -inf, -503};
        std::vector<double> ub{inf, inf, 50};

        THEN("We find the correct solution") {
            SolveResult result = linprog(c, b_lb, A, b_ub, A_eq, b_eq, lb, ub);
            CHECK(result.solve_status == SolveResult::SolveStatus::SUCCESS);
            CHECK(result.solution_status() == SolveResult::SolutionStatus::OPTIMAL);
            CHECK(result.feasible());
            CHECK(result.objective() == -465);

            CHECK(std::ranges::equal(std::vector{-43, 53, 50}, result.solution()));
        }
    }

    GIVEN("Simple LP with one lower bounded constraint") {
        std::vector<double> c{1, 2, 3};

        std::vector<double> A{
                1,
                1,
                1,
        };
        std::vector<double> b_lb{-2};
        std::vector<double> b_ub{inf};

        std::vector<double> A_eq{};
        std::vector<double> b_eq{};

        std::vector<double> lb{-10, -10, -10};
        std::vector<double> ub{1, 1, 1};

        THEN("We find the correct solution") {
            SolveResult result = linprog(c, b_lb, A, b_ub, A_eq, b_eq, lb, ub);
            CHECK(result.solve_status == SolveResult::SolveStatus::SUCCESS);
            CHECK(result.solution_status() == SolveResult::SolutionStatus::OPTIMAL);
            CHECK(result.feasible());
            CHECK(result.objective() == -9);

            CHECK(std::ranges::equal(std::vector{1, 1, -4}, result.solution()));
        }
    }

    GIVEN("Simple LP with one double bounded constraint") {
        std::vector<double> c{1, 2, 3};

        std::vector<double> A{
                1,
                1,
                1,
        };
        std::vector<double> b_lb{-2};
        std::vector<double> b_ub{2};

        std::vector<double> A_eq{};
        std::vector<double> b_eq{};

        std::vector<double> lb{-10, -10, -10};
        std::vector<double> ub{1, 1, 1};

        THEN("We find the correct solution") {
            SolveResult result = linprog(c, b_lb, A, b_ub, A_eq, b_eq, lb, ub);
            CHECK(result.solve_status == SolveResult::SolveStatus::SUCCESS);
            CHECK(result.solution_status() == SolveResult::SolutionStatus::OPTIMAL);
            CHECK(result.feasible());
            CHECK(result.objective() == -9);

            CHECK(std::ranges::equal(std::vector{1, 1, -4}, result.solution()));
        }
    }

    GIVEN("LP with no bounds or constraints") {
        std::vector<double> c{1};
        std::vector<double> lb{-inf};
        std::vector<double> ub{inf};

        THEN("We return failure unbounded") {
            SolveResult result = linprog(c, {}, {}, {}, {}, {}, lb, ub);
            CHECK(result.solve_status == SolveResult::SolveStatus::FAILURE_UNBOUNDED);
            CHECK(result.solution_status() == SolveResult::SolutionStatus::FEASIBLE_BUT_NOT_OPTIMAL);
        }
    }

    GIVEN("LP with overlapping bounds") {
        std::vector<double> c{1};
        std::vector<double> lb{5};
        std::vector<double> ub{-inf};

        THEN("We return infeasible") {
            SolveResult result = linprog(c, {}, {}, {}, {}, {}, lb, ub);
            CHECK(result.solve_status == SolveResult::SolveStatus::FAILURE_NO_FEASIBLE_START);
            CHECK(result.solution_status() == SolveResult::SolutionStatus::INFEASIBLE);
        }
    }
}

}  // namespace dwave::optimization
